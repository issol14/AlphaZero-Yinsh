#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH AlphaZero - Quick Start Script
====================================

Quick start script for immediate training with sensible defaults.
Perfect for getting started quickly.

Usage:
    python quick_start.py
"""

import os
import sys
import torch
from datetime import datetime, timedelta
from tqdm import tqdm
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yinsh import YinshModel, YinshAgent, Color
from scripts.selfplay import play_game, generate_training_data, save_game_data
from scripts.train import create_training_batch, train_model


def format_time(seconds):
    """초를 시:분:초 형식으로 변환"""
    return str(timedelta(seconds=int(seconds)))


def quick_start():
    """빠른 시작 - 기본 설정으로 훈련"""
    print("🚀 YINSH AlphaZero Quick Start")
    print("=" * 50)

    # 설정
    iterations = 5  # 적은 반복으로 시작
    games_per_iteration = 10  # 적은 게임으로 시작
    epochs = 5  # 적은 에포크로 시작

    print(f"📋 Quick Start Configuration:")
    print(f"   ├── Iterations: {iterations}")
    print(f"   ├── Games per iteration: {games_per_iteration}")
    print(f"   └── Training epochs: {epochs}")

    # 출력 디렉토리
    output_dir = f"quick_start_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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

    # 전체 시작 시간
    total_start_time = time.time()

    # 통계 변수
    total_games_played = 0
    total_positions_generated = 0

    # 메인 루프 (진행률 바 추가)
    for iteration in tqdm(
        range(iterations),
        desc="🔄 Iterations",
        unit="iter",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    ):
        iteration_start_time = time.time()
        print(f"\n🔄 Iteration {iteration + 1}/{iterations}")
        print(f"⏱️  Started at: {datetime.now().strftime('%H:%M:%S')}")

        # Self-play
        print(f"🎮 Playing {games_per_iteration} games...")
        agent1 = YinshAgent(model_path=None, use_mcts=True)
        agent2 = YinshAgent(model_path=None, use_mcts=True)

        agent1.neural_network = model
        agent2.neural_network = model

        iteration_positions = 0
        white_wins = 0
        black_wins = 0
        draws = 0

        # 게임 진행률 바
        for game_id in tqdm(
            range(games_per_iteration),
            desc=f"🎮 Games (Iter {iteration+1})",
            unit="game",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ):
            game_history, winner, turns = play_game(agent1, agent2)
            iteration_positions += len(game_history)
            total_positions_generated += len(game_history)

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

        total_games_played += games_per_iteration

        # Iteration 통계 출력
        print(f"📈 Iteration {iteration + 1} Statistics:")
        print(f"   ├── Positions generated: {iteration_positions:,}")
        print(
            f"   ├── White wins: {white_wins} ({white_wins/games_per_iteration*100:.1f}%)"
        )
        print(
            f"   ├── Black wins: {black_wins} ({black_wins/games_per_iteration*100:.1f}%)"
        )
        print(f"   └── Draws: {draws} ({draws/games_per_iteration*100:.1f}%)")

        # Training
        print(f"🎯 Training model...")
        states, policies, values = create_training_batch(
            os.path.join(output_dir, "data"), batch_size=16
        )

        if states is not None:
            # 학습 진행률 바 추가
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

        # Iteration 완료 시간
        iteration_time = time.time() - iteration_start_time
        print(
            f"⏱️  Iteration {iteration + 1} completed in {format_time(iteration_time)}"
        )

        # 예상 남은 시간 계산
        if iteration < iterations - 1:
            avg_iteration_time = (time.time() - total_start_time) / (iteration + 1)
            remaining_iterations = iterations - iteration - 1
            estimated_remaining = avg_iteration_time * remaining_iterations
            print(f"🕐 Estimated time remaining: {format_time(estimated_remaining)}")

    # 전체 완료 시간 계산
    total_time = time.time() - total_start_time

    print(f"\n🎉 Quick start training completed!")
    print(f"⏱️ Total time: {format_time(total_time)}")
    print(f"📊 Final Statistics:")
    print(f"   ├── Total iterations: {iterations}")
    print(f"   ├── Total games played: {total_games_played}")
    print(f"   ├── Total positions generated: {total_positions_generated:,}")
    print(
        f"   └── Average positions per game: {total_positions_generated/total_games_played:.1f}"
    )
    print(f"📁 Results saved to: {output_dir}")
    print(f"🔧 To continue training, run: python main.py --continue-from {model_path}")


if __name__ == "__main__":
    quick_start()
