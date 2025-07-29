#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Play Script - 학습된 모델 빠른 테스트
========================================

학습된 모델을 간단히 테스트해볼 수 있는 스크립트입니다.
"""

import os
import sys
import argparse

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yinsh import YinshEnv, YinshAgent, YinshModel, Color, config


def test_model_vs_random():
    """학습된 모델 vs 랜덤 에이전트"""
    print("🎮 학습된 모델 vs 랜덤 에이전트 테스트")
    print("=" * 50)
    
    # 에이전트 생성
    ai_agent = YinshAgent(model_path=None, use_mcts=True)  # 학습된 모델 로드
    random_agent = YinshAgent(model_path=None, use_mcts=False)  # 랜덤 에이전트
    
    env = YinshEnv()
    
    # 여러 게임 테스트
    ai_wins = 0
    random_wins = 0
    draws = 0
    
    num_games = 10
    
    for game in range(num_games):
        env.reset()
        move_count = 0
        
        print(f"\n🎯 게임 {game + 1}/{num_games}")
        
        while not env.is_game_over() and move_count < 100:
            current_player = env.current_player
            
            if current_player == Color.WHITE:
                # AI의 턴
                action, _ = ai_agent.select_action(env, temperature=0.1)
                print(f"🤖 AI 액션: {action}")
            else:
                # 랜덤의 턴
                action, _ = random_agent.select_action(env)
                print(f"🎲 랜덤 액션: {action}")
            
            env.step(action)
            move_count += 1
        
        # 결과 확인
        winner = env.get_winner()
        if winner == Color.WHITE:
            ai_wins += 1
            print("✅ AI 승리!")
        elif winner == Color.BLACK:
            random_wins += 1
            print("🎲 랜덤 승리!")
        else:
            draws += 1
            print("🤝 무승부!")
    
    # 최종 결과
    print(f"\n📊 최종 결과:")
    print(f"🤖 AI 승리: {ai_wins}/{num_games} ({ai_wins/num_games*100:.1f}%)")
    print(f"🎲 랜덤 승리: {random_wins}/{num_games} ({random_wins/num_games*100:.1f}%)")
    print(f"🤝 무승부: {draws}/{num_games} ({draws/num_games*100:.1f}%)")


def test_model_performance():
    """모델 성능 테스트"""
    print("📊 모델 성능 테스트")
    print("=" * 50)
    
    # 두 가지 모드 비교
    mcts_agent = YinshAgent(model_path=None, use_mcts=True)
    neural_agent = YinshAgent(model_path=None, use_mcts=False)
    
    env = YinshEnv()
    env.reset()
    
    import time
    
    # MCTS 모드 성능 측정
    start_time = time.time()
    action, info = mcts_agent.select_action(env, temperature=0.1)
    mcts_time = time.time() - start_time
    
    # 신경망 직접 모드 성능 측정
    start_time = time.time()
    action, info = neural_agent.select_action(env, temperature=0.1)
    neural_time = time.time() - start_time
    
    print(f"🌳 MCTS 모드:")
    print(f"   ├── 응답 시간: {mcts_time:.3f}초")
    print(f"   ├── 시뮬레이션: {config.MCTS_SIMULATIONS}회")
    print(f"   └── 강도: 높음")
    
    print(f"🚀 신경망 직접 모드:")
    print(f"   ├── 응답 시간: {neural_time:.3f}초")
    print(f"   ├── 시뮬레이션: 0회")
    print(f"   └── 강도: 중간")
    
    print(f"\n⚡ 속도 차이: {mcts_time/neural_time:.1f}배")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="학습된 모델 빠른 테스트")
    parser.add_argument("--test", choices=["vs_random", "performance", "both"], 
                       default="both", help="테스트 유형")
    parser.add_argument("--model", type=str, default=None, help="모델 경로")
    
    args = parser.parse_args()
    
    print("🎯 YINSH AlphaZero 모델 테스트")
    print("=" * 50)
    
    if args.test in ["vs_random", "both"]:
        test_model_vs_random()
    
    if args.test in ["performance", "both"]:
        print("\n")
        test_model_performance()
    
    print("\n✅ 테스트 완료!")


if __name__ == "__main__":
    main() 