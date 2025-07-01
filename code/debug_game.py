#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH AlphaZero - Debug Game Script
===================================

게임 진행을 단계별로 자세히 보여주는 디버그 스크립트
각 턴마다 보드 상태와 액션을 시각적으로 표시합니다.

Usage:
    python debug_game.py
"""

import os
import sys
import time
import torch
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yinsh import YinshEnv, YinshModel, YinshAgent, Color, config


def print_board_detailed(env: YinshEnv, turn_num: int):
    """상세한 보드 출력"""
    print(f"\n{'='*80}")
    print(f"🎮 TURN {turn_num}")
    print(f"{'='*80}")

    # 게임 정보
    print(
        f"👤 Current Player: {'WHITE' if env.current_player == Color.WHITE else 'BLACK'}"
    )
    print(f"📊 Phase: {env.phase}")
    print(
        f"⚪ White Rings: {len(env.ring_positions[Color.WHITE])} placed, {env.rings_removed[Color.WHITE]} removed"
    )
    print(
        f"⚫ Black Rings: {len(env.ring_positions[Color.BLACK])} placed, {env.rings_removed[Color.BLACK]} removed"
    )
    print(f"🎯 Game Over: {env.is_game_over()}")

    # 보드 출력 (11x11 전체)
    board = env.board
    size = env.board_size

    print(f"\n📋 Board State ({size}x{size}):")
    print("    " + " ".join([f"{i:2d}" for i in range(size)]))

    for i in range(size):
        row = f"{i:2d} "
        for j in range(size):
            pos = (i, j)
            if pos in env.ring_positions[Color.WHITE]:
                row += "🔶 "  # White Ring
            elif pos in env.ring_positions[Color.BLACK]:
                row += "🔷 "  # Black Ring
            elif pos in env.marker_positions[Color.WHITE]:
                row += "🔸 "  # White Marker
            elif pos in env.marker_positions[Color.BLACK]:
                row += "🔹 "  # Black Marker
            elif env.is_valid_position(pos):
                row += "🔲 "  # Valid position (empty)
            else:
                row += "🔳 "  # Invalid position
        print(row)

    # 링 위치 표시
    print(f"\n📍 Ring Positions:")
    print(f"⚪ White Rings: {list(env.ring_positions[Color.WHITE])}")
    print(f"⚫ Black Rings: {list(env.ring_positions[Color.BLACK])}")

    # 마커 위치 표시
    print(f"\n🎯 Marker Positions:")
    print(f"⚪ White Markers: {list(env.marker_positions[Color.WHITE])}")
    print(f"⚫ Black Markers: {list(env.marker_positions[Color.BLACK])}")

    print(f"{'='*80}")


def print_valid_actions(env: YinshEnv):
    """유효한 액션들 출력"""
    valid_actions = env.get_valid_actions()
    print(f"\n🎯 Valid Actions ({len(valid_actions)}):")

    if len(valid_actions) <= 10:
        for i, action in enumerate(valid_actions):
            print(f"   {i+1:2d}. {action}")
    else:
        for i, action in enumerate(valid_actions[:5]):
            print(f"   {i+1:2d}. {action}")
        print(f"   ... and {len(valid_actions)-5} more actions")

    return valid_actions


def print_action_details(action, action_info):
    """액션 상세 정보 출력"""
    print(f"\n🎯 Selected Action Details:")
    print(f"   ├── Action: {action}")
    print(f"   ├── Method: {action_info.get('method', 'unknown')}")
    print(f"   ├── Temperature: {action_info.get('temperature', 'unknown')}")

    if "mcts_stats" in action_info:
        mcts_stats = action_info["mcts_stats"]
        print(f"   ├── MCTS Simulations: {mcts_stats.get('total_visits', 'unknown')}")
        print(f"   ├── Search Time: {mcts_stats.get('search_time', 'unknown'):.3f}s")
        print(f"   └── Nodes Expanded: {mcts_stats.get('nodes_expanded', 'unknown')}")

    if "action_probability" in action_info:
        print(f"   └── Probability: {action_info['action_probability']:.4f}")


def debug_single_game():
    """단일 게임 디버그"""
    print("🎮 YINSH AlphaZero Debug Game")
    print("=" * 80)

    # 환경 생성
    env = YinshEnv()
    print("✅ Environment created")

    # 에이전트 생성
    model = YinshModel()
    agent1 = YinshAgent(model_path=None, use_mcts=True)
    agent2 = YinshAgent(model_path=None, use_mcts=False)

    agent1.neural_network = model
    agent2.neural_network = model

    print("✅ Agents created")

    # 게임 진행
    turn_count = 0
    max_turns = 20  # 디버그용으로 제한

    while not env.is_game_over() and turn_count < max_turns:
        # 현재 상태 출력
        print_board_detailed(env, turn_count)

        # 유효한 액션 출력
        valid_actions = print_valid_actions(env)

        if not valid_actions:
            print("❌ No valid actions available!")
            break

        # 현재 플레이어 결정
        current_player = agent1 if env.current_player == Color.WHITE else agent2
        player_name = (
            "WHITE (MCTS)" if env.current_player == Color.WHITE else "BLACK (Direct)"
        )

        print(f"\n🤔 {player_name} is thinking...")

        # 액션 선택
        start_time = time.time()
        action, action_info = current_player.select_action(env, temperature=0.5)
        thinking_time = time.time() - start_time

        print(f"⏱️  Thinking time: {thinking_time:.3f}s")

        # 액션 상세 정보 출력
        print_action_details(action, action_info)

        # 액션 실행
        print(f"\n🔄 Executing action...")
        success = env.step(action)

        if not success:
            print(f"❌ Invalid action: {action}")
            break

        print(f"✅ Action executed successfully")

        turn_count += 1

        # 게임 종료 체크
        if env.is_game_over():
            winner = env.get_winner()
            print(f"\n🏆 GAME OVER!")
            if winner == Color.WHITE:
                print(f"🏆 WHITE wins!")
            elif winner == Color.BLACK:
                print(f"🏆 BLACK wins!")
            else:
                print(f"🤝 Draw!")
            break

    if turn_count >= max_turns:
        print(f"\n⏰ Game ended after {max_turns} turns (max reached)")

    # 최종 상태 출력
    print_board_detailed(env, turn_count)

    print(f"\n📊 Game Summary:")
    print(f"   ├── Total turns: {turn_count}")
    print(f"   ├── Winner: {env.get_winner()}")
    print(f"   └── Game over: {env.is_game_over()}")


def debug_environment():
    """환경 디버그"""
    print("🔧 YINSH Environment Debug")
    print("=" * 80)

    env = YinshEnv()

    print("📋 Initial Environment State:")
    print(f"   ├── Board size: {env.board_size}")
    print(f"   ├── Current player: {env.current_player}")
    print(f"   ├── Phase: {env.phase}")
    print(f"   ├── Game over: {env.is_game_over()}")
    print(f"   └── Winner: {env.get_winner()}")

    print("\n🎯 Testing valid actions...")
    valid_actions = env.get_valid_actions()
    print(f"   ├── Number of valid actions: {len(valid_actions)}")
    print(f"   └── First 5 actions: {valid_actions[:5] if valid_actions else 'None'}")

    print("\n🧪 Testing state tensor...")
    state_tensor = env.get_state_tensor()
    print(f"   ├── State tensor shape: {state_tensor.shape}")
    print(f"   ├── State tensor dtype: {state_tensor.dtype}")
    print(
        f"   └── State tensor range: [{state_tensor.min():.3f}, {state_tensor.max():.3f}]"
    )

    print("\n✅ Environment debug completed")


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="YINSH AlphaZero Debug")
    parser.add_argument(
        "--mode",
        choices=["game", "env"],
        default="game",
        help="Debug mode: game or environment",
    )

    args = parser.parse_args()

    if args.mode == "game":
        debug_single_game()
    elif args.mode == "env":
        debug_environment()


if __name__ == "__main__":
    main()
