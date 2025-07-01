#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH AlphaZero Self-Play Script
================================

This script generates training data through self-play games.
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yinsh import YinshEnv, YinshAgent, YinshModel, Color, config


def play_game(agent1, agent2, max_turns=1000):
    """두 에이전트 간의 게임 진행"""
    env = YinshEnv()
    game_history = []

    turn_count = 0

    while not env.is_game_over() and turn_count < max_turns:
        current_player = agent1 if env.current_player == Color.WHITE else agent2

        # 현재 상태 저장
        state = env.get_state_tensor()

        # 액션 선택
        action = current_player.select_action(env)

        # 액션 실행
        env.step(action)

        # 게임 히스토리에 추가
        game_history.append(
            {"state": state, "action": action, "player": env.current_player}
        )

        turn_count += 1

    # 게임 결과
    winner = env.get_winner()
    return game_history, winner, turn_count


def generate_training_data(game_history, winner):
    """게임 히스토리에서 훈련 데이터 생성"""
    training_data = []

    for i, move in enumerate(game_history):
        state = move["state"]
        action = move["action"]
        player = move["player"]

        # 정책 벡터 생성 (간단한 원핫 인코딩)
        policy = np.zeros(config.POLICY_OUTPUT_SIZE, dtype=np.float32)

        # 액션을 인덱스로 변환 (간단한 구현)
        action_index = hash(action) % config.POLICY_OUTPUT_SIZE
        policy[action_index] = 1.0

        # 가치 계산
        if winner == Color.NONE:
            value = 0.0
        elif winner == player:
            value = 1.0
        else:
            value = -1.0

        training_data.append({"state": state, "policy": policy, "value": value})

    return training_data


def save_game_data(training_data, output_dir, game_id):
    """게임 데이터 저장"""
    os.makedirs(output_dir, exist_ok=True)

    # 데이터 분리
    states = [data["state"] for data in training_data]
    policies = [data["policy"] for data in training_data]
    values = [data["value"] for data in training_data]

    # PyTorch 텐서로 변환
    states_tensor = torch.FloatTensor(states)
    policies_tensor = torch.FloatTensor(policies)
    values_tensor = torch.FloatTensor(values)

    # 저장
    data = {
        "states": states_tensor,
        "policies": policies_tensor,
        "values": values_tensor,
    }

    file_path = os.path.join(output_dir, f"game_{game_id}.pt")
    torch.save(data, file_path)

    return file_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate self-play data for YINSH AlphaZero"
    )
    parser.add_argument("--model", type=str, default=None, help="Model path to load")
    parser.add_argument("--games", type=int, default=50, help="Number of games to play")
    parser.add_argument("--output", type=str, default="memory", help="Output directory")
    parser.add_argument(
        "--no-mcts", action="store_true", help="Disable MCTS (use direct prediction)"
    )
    parser.add_argument("--mcts-sims", type=int, default=800, help="MCTS simulations")

    args = parser.parse_args()

    print("🎮 Starting YINSH AlphaZero Self-Play...")

    # 모델 로드 또는 생성
    if args.model and os.path.exists(args.model):
        print(f"📥 Loading model: {args.model}")
        model = torch.load(args.model, map_location="cpu")
    else:
        print("🆕 Creating new model")
        model = YinshModel()

    # 에이전트 생성
    agent1 = YinshAgent(
        model, use_mcts=not args.no_mcts, mcts_simulations=args.mcts_sims
    )
    agent2 = YinshAgent(
        model, use_mcts=not args.no_mcts, mcts_simulations=args.mcts_sims
    )

    print(f"🤖 Playing {args.games} games...")

    # 게임 통계
    total_positions = 0
    white_wins = 0
    black_wins = 0
    draws = 0

    # 게임 진행
    for game_id in tqdm(range(args.games), desc="Playing games"):
        # 게임 진행
        game_history, winner, turns = play_game(agent1, agent2)

        # 통계 업데이트
        if winner == Color.WHITE:
            white_wins += 1
        elif winner == Color.BLACK:
            black_wins += 1
        else:
            draws += 1

        total_positions += len(game_history)

        # 훈련 데이터 생성
        training_data = generate_training_data(game_history, winner)

        # 데이터 저장
        save_game_data(training_data, args.output, game_id)

        # 진행 상황 출력
        if (game_id + 1) % 10 == 0:
            print(
                f"📊 Game {game_id + 1}: {len(game_history)} positions, "
                f"Winner: {winner}, Turns: {turns}"
            )

    # 최종 통계
    print("\n📈 Self-Play Statistics:")
    print(f"├── Total games: {args.games}")
    print(f"├── Total positions: {total_positions}")
    print(f"├── White wins: {white_wins} ({white_wins/args.games*100:.1f}%)")
    print(f"├── Black wins: {black_wins} ({black_wins/args.games*100:.1f}%)")
    print(f"├── Draws: {draws} ({draws/args.games*100:.1f}%)")
    print(f"└── Average positions per game: {total_positions/args.games:.1f}")

    print(f"\n💾 Training data saved to: {args.output}")
    print("✅ Self-play completed successfully!")


if __name__ == "__main__":
    main()
