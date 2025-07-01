#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH AlphaZero - Main Training Script
======================================

This is the main entry point for training YINSH AlphaZero.
It orchestrates the complete training pipeline including self-play,
training, and evaluation.

Usage:
    python main.py --iterations 100 --games-per-iteration 50 --epochs 10
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yinsh import (
    YinshEnv,
    YinshModel,
    YinshAgent,
    YinshMCTS,
    Color,
    create_directory_structure,
    get_device_info,
    save_training_history,
    load_training_history,
)
from scripts.selfplay import play_game, generate_training_data, save_game_data
from scripts.train import create_training_batch, train_model


def setup_logging(log_dir: str):
    """로깅 설정"""
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    return logging.getLogger(__name__)


def create_initial_model():
    """초기 모델 생성"""
    logger = logging.getLogger(__name__)
    logger.info("🆕 Creating initial random model...")

    model = YinshModel()
    logger.info(
        f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    return model


def run_self_play_iteration(
    model, games_per_iteration: int, output_dir: str, iteration: int
):
    """한 번의 self-play 반복 실행"""
    logger = logging.getLogger(__name__)
    logger.info(
        f"🎮 Starting self-play iteration {iteration} with {games_per_iteration} games..."
    )

    # 에이전트 생성
    agent1 = YinshAgent(model_path=None, use_mcts=True)
    agent2 = YinshAgent(model_path=None, use_mcts=True)

    # 모델 설정
    agent1.neural_network = model
    agent2.neural_network = model

    total_positions = 0
    white_wins = 0
    black_wins = 0
    draws = 0

    # 게임 진행
    for game_id in range(games_per_iteration):
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
        save_game_data(training_data, output_dir, f"iter_{iteration}_game_{game_id}")

        if (game_id + 1) % 10 == 0:
            logger.info(
                f"📊 Game {game_id + 1}: {len(game_history)} positions, "
                f"Winner: {winner}, Turns: {turns}"
            )

    # 통계 로깅
    logger.info(f"📈 Self-play iteration {iteration} completed:")
    logger.info(f"   ├── Total games: {games_per_iteration}")
    logger.info(f"   ├── Total positions: {total_positions}")
    logger.info(
        f"   ├── White wins: {white_wins} ({white_wins/games_per_iteration*100:.1f}%)"
    )
    logger.info(
        f"   ├── Black wins: {black_wins} ({black_wins/games_per_iteration*100:.1f}%)"
    )
    logger.info(f"   ├── Draws: {draws} ({draws/games_per_iteration*100:.1f}%)")
    logger.info(
        f"   └── Avg positions per game: {total_positions/games_per_iteration:.1f}"
    )

    return {
        "iteration": iteration,
        "games_played": games_per_iteration,
        "total_positions": total_positions,
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": draws,
        "avg_positions_per_game": total_positions / games_per_iteration,
    }


def run_training_iteration(
    model, data_dir: str, epochs: int, batch_size: int, lr: float
):
    """한 번의 훈련 반복 실행"""
    logger = logging.getLogger(__name__)
    logger.info(f"🎯 Starting training iteration with {epochs} epochs...")

    # 훈련 데이터 로드
    states, policies, values = create_training_batch(data_dir, batch_size)

    if states is None:
        logger.warning("⚠️ No training data available!")
        return None

    # 모델 훈련
    trained_model = train_model(
        model, states, policies, values, epochs=epochs, batch_size=batch_size, lr=lr
    )

    logger.info("✅ Training iteration completed")

    return trained_model


def evaluate_model(model, eval_games: int = 20):
    """모델 평가"""
    logger = logging.getLogger(__name__)
    logger.info(f"📊 Evaluating model with {eval_games} games...")

    # 에이전트 생성
    trained_agent = YinshAgent(model_path=None, use_mcts=True)
    random_agent = YinshAgent(model_path=None, use_mcts=False)  # 랜덤 에이전트

    trained_agent.neural_network = model

    wins = 0
    losses = 0
    draws = 0

    for game_id in range(eval_games):
        # 훈련된 에이전트 vs 랜덤 에이전트
        game_history, winner, turns = play_game(trained_agent, random_agent)

        if winner == Color.WHITE:  # 훈련된 에이전트 승리
            wins += 1
        elif winner == Color.BLACK:  # 랜덤 에이전트 승리
            losses += 1
        else:
            draws += 1

    win_rate = wins / eval_games
    logger.info(f"📈 Evaluation results:")
    logger.info(f"   ├── Wins: {wins} ({win_rate*100:.1f}%)")
    logger.info(f"   ├── Losses: {losses} ({losses/eval_games*100:.1f}%)")
    logger.info(f"   └── Draws: {draws} ({draws/eval_games*100:.1f}%)")

    return {"wins": wins, "losses": losses, "draws": draws, "win_rate": win_rate}


def save_checkpoint(model, iteration: int, output_dir: str, stats: dict):
    """체크포인트 저장"""
    logger = logging.getLogger(__name__)

    # 모델 저장
    model_path = os.path.join(output_dir, f"model_iter_{iteration}.pt")
    torch.save(model, model_path)

    # 통계 저장
    stats_path = os.path.join(output_dir, f"stats_iter_{iteration}.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"💾 Checkpoint saved: {model_path}")
    logger.info(f"📊 Stats saved: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="YINSH AlphaZero Training")
    parser.add_argument(
        "--iterations", type=int, default=100, help="Number of training iterations"
    )
    parser.add_argument(
        "--games-per-iteration",
        type=int,
        default=50,
        help="Games per self-play iteration",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Training epochs per iteration"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--eval-games", type=int, default=20, help="Evaluation games")
    parser.add_argument(
        "--output-dir", type=str, default="training_output", help="Output directory"
    )
    parser.add_argument(
        "--continue-from", type=str, default=None, help="Continue from checkpoint"
    )

    args = parser.parse_args()

    # 디렉토리 구조 생성
    create_directory_structure()

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "logs"), exist_ok=True)

    # 로깅 설정
    logger = setup_logging(os.path.join(args.output_dir, "logs"))

    # 디바이스 정보 출력
    device_info = get_device_info()
    logger.info(f"🖥️ Device: {device_info['device']}")
    logger.info(f"💾 Memory: {device_info['memory_gb']:.1f} GB")

    # 모델 로드 또는 생성
    if args.continue_from and os.path.exists(args.continue_from):
        logger.info(f"📥 Loading model from: {args.continue_from}")
        model = torch.load(args.continue_from, map_location="cpu")
        start_iteration = int(args.continue_from.split("_")[-1].split(".")[0])
    else:
        model = create_initial_model()
        start_iteration = 0

    # 훈련 히스토리
    training_history = []

    # 메인 훈련 루프
    logger.info("🚀 Starting YINSH AlphaZero training...")
    logger.info(f"📋 Configuration:")
    logger.info(f"   ├── Iterations: {args.iterations}")
    logger.info(f"   ├── Games per iteration: {args.games_per_iteration}")
    logger.info(f"   ├── Training epochs: {args.epochs}")
    logger.info(f"   ├── Batch size: {args.batch_size}")
    logger.info(f"   └── Learning rate: {args.lr}")

    for iteration in range(start_iteration, args.iterations):
        logger.info(f"\n🔄 Starting iteration {iteration + 1}/{args.iterations}")

        # 1. Self-play
        data_dir = os.path.join(args.output_dir, "data", f"iter_{iteration + 1}")
        os.makedirs(data_dir, exist_ok=True)

        selfplay_stats = run_self_play_iteration(
            model, args.games_per_iteration, data_dir, iteration + 1
        )

        # 2. Training
        trained_model = run_training_iteration(
            model, data_dir, args.epochs, args.batch_size, args.lr
        )

        if trained_model is not None:
            model = trained_model

        # 3. Evaluation
        eval_stats = evaluate_model(model, args.eval_games)

        # 4. 통계 저장
        iteration_stats = {
            "iteration": iteration + 1,
            "timestamp": datetime.now().isoformat(),
            "selfplay": selfplay_stats,
            "evaluation": eval_stats,
        }

        training_history.append(iteration_stats)

        # 5. 체크포인트 저장
        save_checkpoint(
            model,
            iteration + 1,
            os.path.join(args.output_dir, "models"),
            iteration_stats,
        )

        # 6. 전체 히스토리 저장
        history_path = os.path.join(args.output_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(training_history, f, indent=2)

        logger.info(f"✅ Iteration {iteration + 1} completed")
        logger.info(f"📊 Win rate: {eval_stats['win_rate']*100:.1f}%")

    logger.info("🎉 Training completed!")
    logger.info(f"📁 Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
