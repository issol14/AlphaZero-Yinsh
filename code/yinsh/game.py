# game.py - YINSH Game Manager (PyTorch)

import os
import time
import uuid
import logging
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime

from .env import YinshEnv, Color, YinshAction
from .agent import YinshAgent, create_agent
from .mapper import get_action_mapper
from . import config
from .utils import time_function


class YinshGame:
    """YINSH 게임 매니저"""

    def __init__(self, white_agent: YinshAgent, black_agent: YinshAgent):
        """
        Args:
            white_agent: 흰색 플레이어 에이전트
            black_agent: 검은색 플레이어 에이전트
        """
        self.white_agent = white_agent
        self.black_agent = black_agent
        self.env = YinshEnv()
        self.action_mapper = get_action_mapper()

        # 게임 기록
        self.memory = []
        self.game_info = {}

        # 통계
        self.moves_played = 0
        self.game_start_time = None

        print(f"🎮 YINSH Game initialized")
        print(f"├── White: {white_agent.__class__.__name__}")
        print(f"├── Black: {black_agent.__class__.__name__}")
        print(f"└── Board Size: {config.BOARD_SIZE}x{config.BOARD_SIZE}")

    def reset(self):
        """게임 초기화"""
        self.env.reset()
        self.memory = []
        self.moves_played = 0
        self.game_start_time = None
        self.game_info = {}

    def get_winner_value(self, winner: Optional[Color]) -> float:
        """승자를 값으로 변환"""
        if winner == Color.WHITE:
            return 1.0
        elif winner == Color.BLACK:
            return -1.0
        else:
            return 0.0  # 무승부

    @time_function
    def play_one_game(self, stochastic: bool = True, max_moves: int = None) -> float:
        """
        한 게임을 시작부터 끝까지 플레이

        Args:
            stochastic: 확률적 액션 선택 여부
            max_moves: 최대 이동 수

        Returns:
            게임 결과 (-1, 0, 1)
        """
        self.reset()
        self.game_start_time = time.time()

        if max_moves is None:
            max_moves = config.MAX_GAME_MOVES

        print(f"🎯 Starting new YINSH game (max moves: {max_moves})")

        # 게임 메모리 초기화
        self.memory.append([])

        # 게임 진행
        while not self.env.is_game_over() and self.moves_played < max_moves:
            # 현재 플레이어 결정
            current_player = self.env.current_player
            current_agent = (
                self.white_agent if current_player == Color.WHITE else self.black_agent
            )

            # 현재 상태 저장
            current_state = self.env.get_state_tensor()

            # 온도 설정
            temperature = config.SELFPLAY_TEMPERATURE if stochastic else 0.1

            # 액션 선택
            try:
                action, action_info = current_agent.select_action(
                    self.env, temperature=temperature, add_noise=stochastic
                )

                # MCTS 통계에서 정책 추출
                if "mcts_stats" in action_info:
                    mcts_stats = action_info["mcts_stats"]
                    policy_probs = self._extract_policy_from_mcts(mcts_stats)
                else:
                    # 직접 예측인 경우 원핫 정책 생성
                    policy_probs = self._create_onehot_policy(action)

                # 메모리에 저장
                self.save_to_memory(current_state, policy_probs)

                # 액션 실행
                self.env.step(action)
                self.moves_played += 1

                if self.moves_played % 20 == 0:
                    print(f"├── Move {self.moves_played}: {self.env.get_turn_state()}")

            except Exception as e:
                print(f"❌ Error in move {self.moves_played}: {e}")
                break

        # 게임 종료 처리
        winner = self.env.get_winner()
        game_duration = time.time() - self.game_start_time
        winner_value = self.get_winner_value(winner)

        # 게임 정보 저장
        self.game_info = {
            "turns": self.moves_played,
            "duration": game_duration,
            "winner": winner.name if winner else "draw",
            "winner_value": winner_value,
            "final_state": self.env.get_turn_state(),
            "white_agent": self.white_agent.__class__.__name__,
            "black_agent": self.black_agent.__class__.__name__,
        }

        # 메모리에 승자 정보 추가
        self._add_winner_to_memory(winner_value)

        print(
            f"✅ Game finished: {self.game_info['winner']} wins in {self.moves_played} moves ({game_duration:.1f}s)"
        )

        return winner_value

    def _extract_policy_from_mcts(self, mcts_stats: Dict) -> np.ndarray:
        """MCTS 통계에서 정책 분포 추출"""
        policy_probs = np.zeros(config.POLICY_OUTPUT_SIZE)

        # MCTS 방문 횟수를 정책으로 변환
        total_visits = sum(mcts_stats.values())
        if total_visits > 0:
            for action_index, visits in mcts_stats.items():
                if action_index < config.POLICY_OUTPUT_SIZE:
                    policy_probs[action_index] = visits / total_visits

        return policy_probs

    def _create_onehot_policy(self, action: YinshAction) -> np.ndarray:
        """액션을 원핫 정책으로 변환"""
        policy_probs = np.zeros(config.POLICY_OUTPUT_SIZE)
        action_index = self.action_mapper.action_to_index(action)

        if action_index is not None and action_index < config.POLICY_OUTPUT_SIZE:
            policy_probs[action_index] = 1.0

        return policy_probs

    def save_to_memory(self, state: torch.Tensor, policy_probs: np.ndarray):
        """현재 상태와 정책을 메모리에 저장"""
        self.memory[-1].append((state, policy_probs, None))

    def _add_winner_to_memory(self, winner_value: float):
        """메모리의 모든 포지션에 승자 정보 추가"""
        for i, (state, policy, _) in enumerate(self.memory[-1]):
            self.memory[-1][i] = (state, policy, winner_value)

    def save_game(self, name: str = "yinsh_game", full_game: bool = True) -> str:
        """
        게임을 파일로 저장

        Args:
            name: 파일 이름
            full_game: 완전한 게임 여부

        Returns:
            저장된 파일 경로
        """
        # 게임 ID 생성
        game_id = f"{name}-{uuid.uuid4().hex[:8]}"
        file_path = os.path.join(config.MEMORY_DIR, f"{game_id}.npy")

        # 게임 데이터 준비
        if full_game:
            game_data = []
            for state, policy, value in self.memory[-1]:
                game_data.append((state, policy, value))
        else:
            # 마지막 몇 개 포지션만 저장
            game_data = self.memory[-1][-10:]  # 마지막 10개 포지션

        # NumPy 배열로 저장
        np.save(file_path, game_data)

        print(f"💾 Game saved to {file_path}")
        return file_path

    def get_game_summary(self) -> Dict:
        """게임 요약 정보 반환"""
        return {
            "moves_played": self.moves_played,
            "winner": self.game_info.get("winner", "unknown"),
            "duration": self.game_info.get("duration", 0.0),
            "final_state": self.game_info.get("final_state", "unknown"),
            "memory_size": len(self.memory[-1]) if self.memory else 0,
        }

    def print_board(self):
        """현재 보드 상태 출력"""
        print("Current board state:")
        print(self.env.get_turn_state())


def create_yinsh_game(
    white_type: str = "mcts", black_type: str = "mcts", model_path: Optional[str] = None
) -> YinshGame:
    """
    YINSH 게임 생성

    Args:
        white_type: 흰색 플레이어 타입
        black_type: 검은색 플레이어 타입
        model_path: 모델 경로 (신경망 에이전트용)

    Returns:
        YINSH 게임 인스턴스
    """
    white_agent = create_agent(white_type, model_path=model_path)
    black_agent = create_agent(black_type, model_path=model_path)

    return YinshGame(white_agent, black_agent)


def play_multiple_games(
    num_games: int = 10,
    white_type: str = "mcts",
    black_type: str = "mcts",
    model_path: Optional[str] = None,
    save_games: bool = True,
) -> List[Dict]:
    """
    여러 게임 플레이

    Args:
        num_games: 게임 수
        white_type: 흰색 플레이어 타입
        black_type: 검은색 플레이어 타입
        model_path: 모델 경로
        save_games: 게임 저장 여부

    Returns:
        게임 결과 리스트
    """
    game = create_yinsh_game(white_type, black_type, model_path)
    results = []

    print(f"🎮 Playing {num_games} games...")
    print(f"├── White: {white_type}")
    print(f"├── Black: {black_type}")
    print(f"└── Model: {model_path if model_path else 'Random'}")

    for i in range(num_games):
        print(f"\n🎯 Game {i + 1}/{num_games}")

        # 게임 플레이
        winner_value = game.play_one_game(stochastic=True)

        # 결과 저장
        game_summary = game.get_game_summary()
        game_summary["game_id"] = i + 1
        game_summary["winner_value"] = winner_value
        results.append(game_summary)

        # 게임 저장
        if save_games:
            game.save_game(f"game_{i + 1}")

        # 에이전트 리셋
        game.white_agent.reset_game_stats()
        game.black_agent.reset_game_stats()

    # 최종 통계
    white_wins = sum(1 for r in results if r["winner"] == "WHITE")
    black_wins = sum(1 for r in results if r["winner"] == "BLACK")
    draws = sum(1 for r in results if r["winner"] == "draw")

    print(f"\n📊 Final Results:")
    print(f"├── White Wins: {white_wins}")
    print(f"├── Black Wins: {black_wins}")
    print(f"├── Draws: {draws}")
    print(f"└── Total Games: {num_games}")

    return results
