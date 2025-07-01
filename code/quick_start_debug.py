#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH AlphaZero - Quick Start Script (Debug Version)
====================================================

Quick start script with enhanced debugging and visualization.
Shows detailed game progress and training statistics.

Usage:
    python quick_start_debug.py
"""

import os
import sys
import torch
import numpy as np
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yinsh import YinshModel, YinshAgent, Color, YinshEnv
from scripts.selfplay import play_game, generate_training_data, save_game_data
from scripts.train import create_training_batch, train_model


def print_board_debug(env: YinshEnv, turn_num: int, action=None):
    """디버그용 보드 출력"""
    print(f"\n🎮 Turn {turn_num}")
    print(
        f"👤 Current Player: {'WHITE' if env.current_player == Color.WHITE else 'BLACK'}"
    )
    print(f"📊 Phase: {env.phase}")

    if action:
        print(f"🎯 Action: {action}")

    # 간단한 보드 출력 (7x7만)
    board = env.board
    size = min(env.board_size, 7)

    print("   " + " ".join([f"{i:2d}" for i in range(size)]))
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
                row += "🔲 "  # Valid position
            else:
                row += "🔳 "  # Invalid position
        print(row)

    print(
        f"⚪ White Rings: {len(env.ring_positions[Color.WHITE])} placed, {env.rings_removed[Color.WHITE]} removed"
    )
    print(
        f"⚫ Black Rings: {len(env.ring_positions[Color.BLACK])} placed, {env.rings_removed[Color.BLACK]} removed"
    )


def debug_game_play(agent1, agent2, max_turns=50):
    """디버그용 게임 플레이"""
    env = YinshEnv()
    game_history = []
    turn_count = 0

    print(f"\n🎮 Starting debug game...")
    print_board_debug(env, turn_count)

    while not env.is_game_over() and turn_count < max_turns:
        # 현재 플레이어 결정
        current_agent = agent1 if env.current_player == Color.WHITE else agent2
        player_name = "WHITE" if env.current_player == Color.WHITE else "BLACK"

        # 유효한 액션 수 확인
        valid_actions = env.get_valid_actions()
        print(f"\n🤔 {player_name} thinking... ({len(valid_actions)} valid actions)")

        # 액션 선택
        action, action_info = current_agent.select_action(env, temperature=0.5)

        # 액션 정보 출력
        print(f"🎯 Selected: {action}")
        print(f"   Method: {action_info.get('method', 'unknown')}")
        if "mcts_stats" in action_info:
            mcts_stats = action_info["mcts_stats"]
            print(f"   MCTS Simulations: {mcts_stats.get('total_visits', 'unknown')}")
            print(f"   Search Time: {mcts_stats.get('search_time', 'unknown'):.3f}s")

        # 게임 상태 저장
        game_history.append(
            {
                "state": env.get_state_tensor(),
                "action": action,
                "player": env.current_player,
                "turn": turn_count,
            }
        )

        # 액션 실행
        success = env.step(action)
        if not success:
            print(f"❌ Invalid action: {action}")
            break

        turn_count += 1

        # 보드 출력 (매 5턴마다 또는 게임 종료 시)
        if turn_count % 5 == 0 or env.is_game_over():
            print_board_debug(env, turn_count, action)

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

    print_board_debug(env, turn_count)

    return game_history, env.get_winner(), turn_count


def quick_start_debug():
    """디버그 기능이 포함된 빠른 시작"""
    print("🚀 YINSH AlphaZero Quick Start (Debug Version)")
    print("=" * 60)

    # 설정
    iterations = 3  # 적은 반복으로 시작
    games_per_iteration = 3  # 적은 게임으로 시작
    epochs = 3  # 적은 에포크로 시작
    debug_games = 1  # 디버그용 게임 수

    print(f"📋 Debug Configuration:")
    print(f"   ├── Iterations: {iterations}")
    print(f"   ├── Games per iteration: {games_per_iteration}")
    print(f"   ├── Training epochs: {epochs}")
    print(f"   └── Debug games: {debug_games}")

    # 출력 디렉토리
    output_dir = f"quick_start_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)

    print(f"📁 Output directory: {output_dir}")

    # 초기 모델 생성
    print("\n🆕 Creating initial model...")
    model = YinshModel()
    print(
        f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # 메인 루프
    for iteration in range(iterations):
        print(f"\n🔄 Iteration {iteration + 1}/{iterations}")
        print("=" * 50)

        # Self-play
        print(f"🎮 Playing {games_per_iteration} games...")
        agent1 = YinshAgent(model_path=None, use_mcts=True)
        agent2 = YinshAgent(model_path=None, use_mcts=True)

        agent1.neural_network = model
        agent2.neural_network = model

        total_positions = 0
        white_wins = 0
        black_wins = 0
        draws = 0

        for game_id in range(games_per_iteration):
            print(f"\n🎯 Game {game_id + 1}/{games_per_iteration}")
            print("-" * 30)

            if game_id < debug_games:
                # 디버그 게임 (상세 출력)
                game_history, winner, turns = debug_game_play(agent1, agent2)
            else:
                # 일반 게임 (간단 출력)
                game_history, winner, turns = play_game(agent1, agent2)
                print(
                    f"   📊 {len(game_history)} positions, Winner: {winner}, Turns: {turns}"
                )

            total_positions += len(game_history)

            # 승자 통계
            if winner == Color.WHITE:
                white_wins += 1
            elif winner == Color.BLACK:
                black_wins += 1
            else:
                draws += 1

            training_data = generate_training_data(game_history, winner)
            save_game_data(
                training_data,
                os.path.join(output_dir, "data"),
                f"iter_{iteration+1}_game_{game_id}",
            )

        print(f"\n📈 Iteration {iteration + 1} Statistics:")
        print(f"   ├── Total positions: {total_positions}")
        print(f"   ├── White wins: {white_wins}")
        print(f"   ├── Black wins: {black_wins}")
        print(f"   └── Draws: {draws}")

        # Training
        print(f"\n🎯 Training model...")
        states, policies, values = create_training_batch(
            os.path.join(output_dir, "data"), batch_size=16
        )

        if states is not None:
            print(f"   📊 Training data: {len(states)} samples")
            trained_model = train_model(
                model, states, policies, values, epochs=epochs, batch_size=16, lr=0.001
            )
            model = trained_model
            print(f"✅ Training completed")
        else:
            print(f"⚠️ No training data available")

        # 모델 저장
        model_path = os.path.join(output_dir, "models", f"model_iter_{iteration+1}.pt")
        torch.save(model, model_path)
        print(f"💾 Model saved: {model_path}")

    print(f"\n🎉 Debug training completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"🔧 To continue training, run: python main.py --continue-from {model_path}")


if __name__ == "__main__":
    quick_start_debug()
