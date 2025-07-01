#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH AlphaZero - Demo Script
=============================

Demo script to showcase the trained model playing games.
Perfect for testing and demonstration.

Usage:
    python demo.py --model path/to/model.pt --games 5
"""

import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yinsh import YinshEnv, YinshModel, YinshAgent, Color


def print_board(env: YinshEnv):
    """보드 상태 출력"""
    print("\n" + "=" * 50)
    print("YINSH Board State")
    print("=" * 50)

    # 간단한 보드 출력
    board = env.board
    size = env.board_size

    print("   " + " ".join([f"{i:2d}" for i in range(size)]))
    for i in range(size):
        row = f"{i:2d} "
        for j in range(size):
            if board[i, j] == 1:  # WHITE
                row += "⚪ "
            elif board[i, j] == -1:  # BLACK
                row += "⚫ "
            else:
                row += "· "
        print(row)

    print(
        f"\nCurrent Player: {'WHITE' if env.current_player == Color.WHITE else 'BLACK'}"
    )
    print(f"Phase: {env.phase}")
    print(
        f"White Rings: {len(env.ring_positions[Color.WHITE])} placed, {env.rings_removed[Color.WHITE]} removed"
    )
    print(
        f"Black Rings: {len(env.ring_positions[Color.BLACK])} placed, {env.rings_removed[Color.BLACK]} removed"
    )
    print("=" * 50)


def play_demo_game(agent1, agent2, max_turns=100):
    """데모 게임 진행"""
    env = YinshEnv()
    game_history = []

    print(f"🎮 Starting demo game...")
    print_board(env)

    turn_count = 0

    while not env.is_game_over() and turn_count < max_turns:
        current_player = agent1 if env.current_player == Color.WHITE else agent2

        # 현재 상태 출력
        print(f"\n🔄 Turn {turn_count + 1}")
        print(f"👤 Player: {'WHITE' if env.current_player == Color.WHITE else 'BLACK'}")

        # 액션 선택
        action, action_info = current_player.select_action(env, temperature=0.5)

        print(f"🎯 Action: {action}")
        print(f"📊 Info: {action_info.get('method', 'unknown')} method")

        # 액션 실행
        success = env.step(action)

        if not success:
            print(f"❌ Invalid action: {action}")
            break

        # 게임 히스토리에 추가
        game_history.append(
            {
                "turn": turn_count + 1,
                "player": env.current_player,
                "action": action,
                "action_info": action_info,
            }
        )

        # 보드 출력
        print_board(env)

        turn_count += 1

        # 게임 종료 체크
        if env.is_game_over():
            winner = env.get_winner()
            if winner == Color.WHITE:
                print(f"🏆 WHITE wins!")
            elif winner == Color.BLACK:
                print(f"🏆 BLACK wins!")
            else:
                print(f"🤝 Draw!")
            break

    if turn_count >= max_turns:
        print(f"⏰ Game ended after {max_turns} turns (max reached)")

    return game_history, env.get_winner(), turn_count


def run_demo():
    """데모 실행"""
    parser = argparse.ArgumentParser(description="YINSH AlphaZero Demo")
    parser.add_argument("--model", type=str, default=None, help="Model path to load")
    parser.add_argument("--games", type=int, default=3, help="Number of games to play")
    parser.add_argument(
        "--max-turns", type=int, default=50, help="Maximum turns per game"
    )
    parser.add_argument("--mcts-sims", type=int, default=100, help="MCTS simulations")
    parser.add_argument(
        "--output", type=str, default="demo_output", help="Output directory"
    )

    args = parser.parse_args()

    print("🎮 YINSH AlphaZero Demo")
    print("=" * 50)

    # 출력 디렉토리 생성
    os.makedirs(args.output, exist_ok=True)

    # 모델 로드 또는 생성
    if args.model and os.path.exists(args.model):
        print(f"📥 Loading model: {args.model}")
        model = torch.load(args.model, map_location="cpu")
    else:
        print("🆕 Creating new random model")
        model = YinshModel()

    # 에이전트 생성
    print(f"🤖 Creating agents with {args.mcts_sims} MCTS simulations...")
    agent1 = YinshAgent(model_path=None, use_mcts=True)  # MCTS 에이전트
    agent2 = YinshAgent(model_path=None, use_mcts=False)  # 직접 예측 에이전트

    agent1.neural_network = model
    agent2.neural_network = model

    # 게임 통계
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    total_turns = 0

    # 게임 진행
    for game_id in range(args.games):
        print(f"\n🎮 Game {game_id + 1}/{args.games}")
        print("-" * 30)

        game_history, winner, turns = play_demo_game(agent1, agent2, args.max_turns)

        # 통계 업데이트
        if winner == Color.WHITE:  # agent1 (MCTS)
            agent1_wins += 1
            print(f"🏆 MCTS Agent wins!")
        elif winner == Color.BLACK:  # agent2 (Direct)
            agent2_wins += 1
            print(f"🏆 Direct Agent wins!")
        else:
            draws += 1
            print(f"🤝 Draw!")

        total_turns += turns

        # 게임 히스토리 저장
        game_file = os.path.join(args.output, f"game_{game_id+1}.txt")
        with open(game_file, "w") as f:
            f.write(f"YINSH Demo Game {game_id+1}\n")
            f.write(f"Winner: {winner}\n")
            f.write(f"Turns: {turns}\n")
            f.write(f"Date: {datetime.now()}\n\n")

            for move in game_history:
                f.write(f"Turn {move['turn']}: {move['player']} - {move['action']}\n")

    # 최종 통계
    print(f"\n📊 Demo Statistics:")
    print(f"   ├── Total games: {args.games}")
    print(f"   ├── MCTS Agent wins: {agent1_wins} ({agent1_wins/args.games*100:.1f}%)")
    print(
        f"   ├── Direct Agent wins: {agent2_wins} ({agent2_wins/args.games*100:.1f}%)"
    )
    print(f"   ├── Draws: {draws} ({draws/args.games*100:.1f}%)")
    print(f"   └── Average turns per game: {total_turns/args.games:.1f}")

    print(f"\n📁 Demo results saved to: {args.output}")
    print("🎉 Demo completed!")


if __name__ == "__main__":
    run_demo()
