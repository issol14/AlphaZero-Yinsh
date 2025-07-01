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
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yinsh import YinshModel, YinshAgent, Color
from scripts.selfplay import play_game, generate_training_data, save_game_data
from scripts.train import create_training_batch, train_model


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

    # 메인 루프
    for iteration in range(iterations):
        print(f"\n🔄 Iteration {iteration + 1}/{iterations}")

        # Self-play
        print(f"🎮 Playing {games_per_iteration} games...")
        agent1 = YinshAgent(model_path=None, use_mcts=True)
        agent2 = YinshAgent(model_path=None, use_mcts=True)

        agent1.neural_network = model
        agent2.neural_network = model

        total_positions = 0

        for game_id in range(games_per_iteration):
            game_history, winner, turns = play_game(agent1, agent2)
            total_positions += len(game_history)

            training_data = generate_training_data(game_history, winner)
            save_game_data(
                training_data,
                os.path.join(output_dir, "data"),
                f"iter_{iteration+1}_game_{game_id}",
            )

            if (game_id + 1) % 5 == 0:
                print(
                    f"   📊 Game {game_id + 1}: {len(game_history)} positions, Winner: {winner}"
                )

        print(f"📈 Generated {total_positions} training positions")

        # Training
        print(f"🎯 Training model...")
        states, policies, values = create_training_batch(
            os.path.join(output_dir, "data"), batch_size=16
        )

        if states is not None:
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

    print(f"\n🎉 Quick start training completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"🔧 To continue training, run: python main.py --continue-from {model_path}")


if __name__ == "__main__":
    quick_start()
