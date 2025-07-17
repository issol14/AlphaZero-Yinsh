#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Human vs AI YINSH Game Interface
================================

학습된 AI 모델과 사람이 대결할 수 있는 인터페이스입니다.
"""

import os
import sys
import time
import torch
import numpy as np
from typing import Optional, Tuple

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yinsh import YinshEnv, YinshAgent, YinshModel, Color, YinshAction, config


class HumanPlayer:
    """사람 플레이어 클래스"""
    
    def __init__(self, name: str = "Human"):
        self.name = name
        self.games_played = 0
    
    def select_action(self, env: YinshEnv, **kwargs) -> Tuple[YinshAction, dict]:
        """사람으로부터 액션 입력 받기"""
        while True:
            try:
                print(f"\n🎯 {self.name}의 턴입니다!")
                self.print_game_info(env)
                self.print_board(env)
                
                # 유효한 액션 표시
                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    raise ValueError("유효한 액션이 없습니다!")
                
                print(f"\n📋 유효한 액션 ({len(valid_actions)}개):")
                for i, action in enumerate(valid_actions[:10]):  # 처음 10개만 표시
                    print(f"  {i+1}. {self.format_action(action)}")
                if len(valid_actions) > 10:
                    print(f"  ... 총 {len(valid_actions)}개 액션")
                
                # 사용자 입력 받기
                print("\n💭 액션 입력 방법:")
                print("  - 링 배치: 'place x y' (예: place 5 5)")
                print("  - 링 이동: 'move x1 y1 x2 y2' (예: move 5 5 6 6)")
                print("  - 'help': 도움말, 'board': 보드 다시 보기, 'quit': 게임 종료")
                
                user_input = input("\n🎮 액션 입력: ").strip().lower()
                
                if user_input == 'quit':
                    print("게임을 종료합니다.")
                    sys.exit(0)
                elif user_input == 'help':
                    self.print_help()
                    continue
                elif user_input == 'board':
                    continue
                
                # 액션 파싱
                action = self.parse_action(user_input)
                
                # 유효성 검사
                if action in valid_actions:
                    action_info = {
                        "method": "human",
                        "action_input": user_input,
                        "valid_actions_count": len(valid_actions)
                    }
                    return action, action_info
                else:
                    print("❌ 유효하지 않은 액션입니다!")
                    
            except KeyboardInterrupt:
                print("\n게임을 종료합니다.")
                sys.exit(0)
            except Exception as e:
                print(f"❌ 오류: {e}")
                print("올바른 형식으로 입력해주세요.")
    
    def parse_action(self, user_input: str) -> YinshAction:
        """사용자 입력을 액션으로 변환"""
        parts = user_input.split()
        
        if len(parts) == 3 and parts[0] == 'place':
            x, y = int(parts[1]), int(parts[2])
            return YinshAction("PLACE_RING", to_pos=(x, y))
        elif len(parts) == 5 and parts[0] == 'move':
            x1, y1, x2, y2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            return YinshAction("MOVE_RING", from_pos=(x1, y1), to_pos=(x2, y2))
        else:
            raise ValueError("올바른 형식으로 입력해주세요")
    
    def format_action(self, action: YinshAction) -> str:
        """액션을 사람이 읽기 쉬운 형태로 변환"""
        if action.action_type == "PLACE_RING":
            return f"링 배치 ({action.to_pos[0]}, {action.to_pos[1]})"
        elif action.action_type == "MOVE_RING":
            return f"링 이동 ({action.from_pos[0]}, {action.from_pos[1]}) → ({action.to_pos[0]}, {action.to_pos[1]})"
        else:
            return str(action)
    
    def print_game_info(self, env: YinshEnv):
        """게임 정보 출력"""
        print(f"\n📊 게임 정보:")
        print(f"├── 현재 단계: {env.phase}")
        print(f"├── 턴 수: {env.move_count}")
        print(f"├── 백색 링: {len(env.ring_positions[Color.WHITE])}개 배치, {env.rings_removed[Color.WHITE]}개 제거")
        print(f"└── 흑색 링: {len(env.ring_positions[Color.BLACK])}개 배치, {env.rings_removed[Color.BLACK]}개 제거")
    
    def print_board(self, env: YinshEnv):
        """보드 상태 출력"""
        print(f"\n🎲 보드 상태 (11x11):")
        print("   " + "".join([f"{i:2d}" for i in range(11)]))
        
        for i in range(11):
            row = f"{i:2d} "
            for j in range(11):
                pos = (i, j)
                if pos in env.ring_positions[Color.WHITE]:
                    row += "⚪"  # 백색 링
                elif pos in env.ring_positions[Color.BLACK]:
                    row += "⚫"  # 흑색 링
                elif pos in env.marker_positions[Color.WHITE]:
                    row += "🔵"  # 백색 마커
                elif pos in env.marker_positions[Color.BLACK]:
                    row += "🔴"  # 흑색 마커
                elif env.is_valid_position(pos):
                    row += "🔲"  # 유효한 빈 공간
                else:
                    row += "🔳"  # 유효하지 않은 공간
            print(row)
    
    def print_help(self):
        """도움말 출력"""
        print("\n📖 YINSH 게임 도움말:")
        print("🎯 게임 목표: 상대방보다 먼저 3개의 링을 제거하기")
        print("\n📋 게임 규칙:")
        print("1. 링 배치 단계: 각 플레이어가 5개의 링을 보드에 배치")
        print("2. 링 이동 단계: 링을 이동하고 경로의 마커를 뒤집기")
        print("3. 5개 연속 마커가 생기면 해당 링 제거")
        print("4. 먼저 3개 링을 제거하는 플레이어가 승리")
        print("\n🎮 입력 형식:")
        print("- 링 배치: place x y")
        print("- 링 이동: move x1 y1 x2 y2")
        print("- 좌표는 0~10 사이의 숫자")
    
    def reset(self):
        """게임 리셋"""
        self.games_played += 1


class HumanVsAI:
    """사람 vs AI 게임 매니저"""
    
    def __init__(self, model_path: Optional[str] = None, use_mcts: bool = True):
        print("🎮 Human vs AI YINSH Game")
        print("=" * 50)
        
        # AI 에이전트 생성
        print("🤖 AI 에이전트 로딩...")
        self.ai_agent = YinshAgent(
            model_path=model_path,
            use_mcts=use_mcts,
            device="cpu"  # 실시간 플레이를 위해 CPU 사용
        )
        
        # 사람 플레이어 생성
        self.human_player = HumanPlayer()
        
        # 환경 생성
        self.env = YinshEnv()
        
        print("✅ 게임 준비 완료!")
    
    def play_game(self):
        """한 게임 실행"""
        self.env.reset()
        move_count = 0
        max_moves = 200
        
        print(f"\n🎯 게임 시작!")
        print(f"👤 사람: WHITE (⚪)")
        print(f"🤖 AI: BLACK (⚫)")
        
        while not self.env.is_game_over() and move_count < max_moves:
            current_player = self.env.current_player
            
            if current_player == Color.WHITE:
                # 사람의 턴
                action, action_info = self.human_player.select_action(self.env)
                print(f"✅ 사람이 선택한 액션: {self.human_player.format_action(action)}")
            else:
                # AI의 턴
                print(f"\n🤖 AI가 생각 중...")
                start_time = time.time()
                action, action_info = self.ai_agent.select_action(
                    self.env, 
                    temperature=0.1,  # 결정적 플레이
                    add_noise=False
                )
                thinking_time = time.time() - start_time
                
                print(f"🎯 AI가 선택한 액션: {self.human_player.format_action(action)}")
                print(f"⏱️  AI 사고 시간: {thinking_time:.2f}초")
                
                if "mcts_stats" in action_info:
                    stats = action_info["mcts_stats"]
                    print(f"🌳 MCTS 통계: {stats.get('total_visits', 0)}회 시뮬레이션")
            
            # 액션 실행
            success = self.env.step(action)
            if not success:
                print(f"❌ 액션 실행 실패: {action}")
                break
            
            move_count += 1
            
            # 게임 상태 출력 (AI 턴 후)
            if current_player == Color.BLACK:
                print(f"\n📊 턴 {move_count} 완료")
                time.sleep(0.5)  # 잠시 대기
        
        # 게임 결과
        winner = self.env.get_winner()
        print(f"\n🎉 게임 종료!")
        print(f"총 {move_count}턴 진행")
        
        if winner == Color.WHITE:
            print("🏆 사람이 승리했습니다!")
        elif winner == Color.BLACK:
            print("🤖 AI가 승리했습니다!")
        else:
            print("🤝 무승부입니다!")
        
        return winner
    
    def play_multiple_games(self, num_games: int = 1):
        """여러 게임 실행"""
        human_wins = 0
        ai_wins = 0
        draws = 0
        
        for game_num in range(num_games):
            print(f"\n{'='*50}")
            print(f"🎮 게임 {game_num + 1}/{num_games}")
            print(f"{'='*50}")
            
            winner = self.play_game()
            
            if winner == Color.WHITE:
                human_wins += 1
            elif winner == Color.BLACK:
                ai_wins += 1
            else:
                draws += 1
            
            if game_num < num_games - 1:
                play_again = input("\n계속하시겠습니까? (y/n): ").strip().lower()
                if play_again != 'y':
                    break
        
        # 최종 결과
        print(f"\n🏆 최종 결과:")
        print(f"👤 사람: {human_wins}승")
        print(f"🤖 AI: {ai_wins}승")
        print(f"🤝 무승부: {draws}게임")


def main():
    """메인 함수"""
    print("🎮 YINSH Human vs AI")
    print("=" * 50)
    
    # 설정
    model_path = input("모델 경로 입력 (엔터시 기본값): ").strip()
    if not model_path:
        model_path = None
    
    use_mcts = input("MCTS 사용? (y/n, 기본값: y): ").strip().lower()
    use_mcts = use_mcts != 'n'
    
    if use_mcts:
        print("🌳 MCTS + 신경망 모드 (강하지만 느림)")
    else:
        print("🚀 신경망 직접 모드 (빠르지만 약함)")
    
    # 게임 생성 및 시작
    try:
        game = HumanVsAI(model_path=model_path, use_mcts=use_mcts)
        game.play_multiple_games(num_games=3)
    except KeyboardInterrupt:
        print("\n게임을 종료합니다.")
    except Exception as e:
        print(f"오류 발생: {e}")


if __name__ == "__main__":
    main() 