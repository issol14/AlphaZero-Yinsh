#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH AlphaZero - Verbose Training Script
=========================================

상세한 훈련 과정을 보여주는 스크립트
각 단계별로 자세한 정보를 출력합니다.

Usage:
    python verbose_training.py
"""

import os
import sys
import time
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yinsh import YinshEnv, YinshModel, YinshAgent, Color, config
from scripts.selfplay import play_game, generate_training_data, save_game_data
from scripts.train import create_training_batch, train_model


def print_separator(title=""):
    """구분선 출력"""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")


def print_game_state(env, turn_num):
    """게임 상태 출력"""
    print(f"\n🎮 Turn {turn_num}")
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

    # 간단한 보드 출력
    board = env.board
    size = min(env.board_size, 7)  # 7x7까지만 출력

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


def verbose_play_game(agent1, agent2, max_turns=100, show_board=True):
    """상세한 게임 진행"""
    env = YinshEnv()
    game_history = []

    print(f"🎮 Starting verbose game...")
    if show_board:
        print_game_state(env, 0)

    turn_count = 0

    while not env.is_game_over() and turn_count < max_turns:
        current_player = agent1 if env.current_player == Color.WHITE else agent2

        # 현재 상태 저장
        state = env.get_state_tensor()

        # 액션 선택
        print(
            f"\n🤔 Player {'WHITE' if env.current_player == Color.WHITE else 'BLACK'} thinking..."
        )
        start_time = time.time()

        action, action_info = current_player.select_action(env, temperature=0.5)

        thinking_time = time.time() - start_time
        print(f"⏱️  Thinking time: {thinking_time:.2f}s")
        print(f"🎯 Action: {action}")
        print(f"📊 Method: {action_info.get('method', 'unknown')}")

        # 액션 실행
        success = env.step(action)

        if not success:
            print(f"❌ Invalid action: {action}")
            break

        # 게임 히스토리에 추가
        game_history.append(
            {
                "state": state,
                "action": action,
                "player": env.current_player,
                "thinking_time": thinking_time,
                "action_info": action_info,
            }
        )

        turn_count += 1

        if show_board:
            print_game_state(env, turn_count)

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


def verbose_self_play_iteration(model, games_per_iteration, output_dir, iteration):
    """상세한 self-play 반복"""
    print_separator(f"SELF-PLAY ITERATION {iteration}")

    # 에이전트 생성
    print("🤖 Creating agents...")
    agent1 = YinshAgent(model_path=None, use_mcts=True)
    agent2 = YinshAgent(model_path=None, use_mcts=True)

    agent1.neural_network = model
    agent2.neural_network = model

    print(f"✅ Agents created with MCTS enabled")

    total_positions = 0
    white_wins = 0
    black_wins = 0
    draws = 0
    total_thinking_time = 0

    # 게임 진행
    for game_id in range(games_per_iteration):
        print_separator(f"GAME {game_id + 1}/{games_per_iteration}")

        # 첫 번째 게임만 상세히 보여주기
        show_board = game_id == 0

        game_history, winner, turns = verbose_play_game(
            agent1, agent2, max_turns=50, show_board=show_board
        )

        # 통계 업데이트
        if winner == Color.WHITE:
            white_wins += 1
        elif winner == Color.BLACK:
            black_wins += 1
        else:
            draws += 1

        total_positions += len(game_history)

        # 평균 사고 시간 계산
        game_thinking_time = sum(move["thinking_time"] for move in game_history)
        total_thinking_time += game_thinking_time

        print(f"📊 Game {game_id + 1} Summary:")
        print(f"   ├── Positions: {len(game_history)}")
        print(f"   ├── Winner: {winner}")
        print(f"   ├── Turns: {turns}")
        print(f"   └── Avg thinking time: {game_thinking_time/len(game_history):.2f}s")

        # 훈련 데이터 생성
        print("🔄 Generating training data...")
        training_data = generate_training_data(game_history, winner)
        print(f"✅ Generated {len(training_data)} training examples")

        # 데이터 저장
        save_game_data(training_data, output_dir, f"iter_{iteration}_game_{game_id}")
        print(f"💾 Data saved")

    # 반복 통계
    print_separator(f"SELF-PLAY ITERATION {iteration} COMPLETED")
    print(f"📈 Statistics:")
    print(f"   ├── Total games: {games_per_iteration}")
    print(f"   ├── Total positions: {total_positions}")
    print(
        f"   ├── White wins: {white_wins} ({white_wins/games_per_iteration*100:.1f}%)"
    )
    print(
        f"   ├── Black wins: {black_wins} ({black_wins/games_per_iteration*100:.1f}%)"
    )
    print(f"   ├── Draws: {draws} ({draws/games_per_iteration*100:.1f}%)")
    print(f"   ├── Avg positions per game: {total_positions/games_per_iteration:.1f}")
    print(f"   └── Avg thinking time: {total_thinking_time/total_positions:.2f}s")

    return {
        "iteration": iteration,
        "games_played": games_per_iteration,
        "total_positions": total_positions,
        "white_wins": white_wins,
        "black_wins": black_wins,
        "draws": draws,
        "avg_positions_per_game": total_positions / games_per_iteration,
        "avg_thinking_time": total_thinking_time / total_positions,
    }


def verbose_training_iteration(model, data_dir, epochs, batch_size, lr):
    """상세한 훈련 반복"""
    print_separator("TRAINING ITERATION")

    # 훈련 데이터 로드
    print("📂 Loading training data...")
    states, policies, values = create_training_batch(data_dir, batch_size)

    if states is None:
        print("❌ No training data available!")
        return None

    print(f"✅ Loaded {len(states)} training positions")
    print(f"📊 Data shapes:")
    print(f"   ├── States: {states.shape}")
    print(f"   ├── Policies: {policies.shape}")
    print(f"   └── Values: {values.shape}")

    # 모델 훈련
    print(f"\n🎯 Starting training for {epochs} epochs...")
    print(f"📋 Training config:")
    print(f"   ├── Batch size: {batch_size}")
    print(f"   ├── Learning rate: {lr}")
    print(f"   └── Device: {next(model.parameters()).device}")

    trained_model = train_model(
        model, states, policies, values, epochs=epochs, batch_size=batch_size, lr=lr
    )

    print("✅ Training completed")

    return trained_model


def verbose_evaluation(model, eval_games=5):
    """상세한 모델 평가"""
    print_separator("MODEL EVALUATION")

    # 에이전트 생성
    print("🤖 Creating evaluation agents...")
    trained_agent = YinshAgent(model_path=None, use_mcts=True)
    random_agent = YinshAgent(model_path=None, use_mcts=False)

    trained_agent.neural_network = model

    wins = 0
    losses = 0
    draws = 0
    total_turns = 0

    print(f"🎮 Playing {eval_games} evaluation games...")

    for game_id in range(eval_games):
        print(f"\n🎮 Evaluation Game {game_id + 1}/{eval_games}")

        game_history, winner, turns = verbose_play_game(
            trained_agent, random_agent, max_turns=30, show_board=False
        )

        if winner == Color.WHITE:  # 훈련된 에이전트 승리
            wins += 1
            result = "WIN"
        elif winner == Color.BLACK:  # 랜덤 에이전트 승리
            losses += 1
            result = "LOSS"
        else:
            draws += 1
            result = "DRAW"

        total_turns += turns

        print(f"📊 Game {game_id + 1}: {result} ({turns} turns)")

    win_rate = wins / eval_games

    print_separator("EVALUATION RESULTS")
    print(f"📈 Performance:")
    print(f"   ├── Wins: {wins} ({win_rate*100:.1f}%)")
    print(f"   ├── Losses: {losses} ({losses/eval_games*100:.1f}%)")
    print(f"   ├── Draws: {draws} ({draws/eval_games*100:.1f}%)")
    print(f"   └── Avg turns per game: {total_turns/eval_games:.1f}")

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": win_rate,
        "avg_turns": total_turns / eval_games,
    }


def verbose_training():
    """상세한 훈련 실행"""
    print_separator("YINSH ALPHAZERO VERBOSE TRAINING")

    # 설정
    iterations = 3  # 적은 반복으로 시작
    games_per_iteration = 5  # 적은 게임으로 시작
    epochs = 3  # 적은 에포크로 시작

    print(f"📋 Configuration:")
    print(f"   ├── Iterations: {iterations}")
    print(f"   ├── Games per iteration: {games_per_iteration}")
    print(f"   └── Training epochs: {epochs}")

    # 출력 디렉토리
    output_dir = f"verbose_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)

    print(f"📁 Output directory: {output_dir}")

    # 초기 모델 생성
    print_separator("MODEL INITIALIZATION")
    print("🆕 Creating initial model...")
    model = YinshModel()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {total_params:,} parameters")

    # 메인 루프
    for iteration in range(iterations):
        print_separator(f"MAIN ITERATION {iteration + 1}/{iterations}")

        # 1. Self-play
        selfplay_stats = verbose_self_play_iteration(
            model, games_per_iteration, os.path.join(output_dir, "data"), iteration + 1
        )

        # 2. Training
        trained_model = verbose_training_iteration(
            model, os.path.join(output_dir, "data"), epochs, 16, 0.001
        )

        if trained_model is not None:
            model = trained_model

        # 3. Evaluation
        eval_stats = verbose_evaluation(model, eval_games=3)

        # 4. 모델 저장
        model_path = os.path.join(output_dir, "models", f"model_iter_{iteration+1}.pt")
        torch.save(model, model_path)
        print(f"💾 Model saved: {model_path}")

        print_separator(f"ITERATION {iteration + 1} COMPLETED")
        print(f"🎯 Win rate: {eval_stats['win_rate']*100:.1f}%")

    print_separator("TRAINING COMPLETED")
    print(f"🎉 Verbose training completed!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"🔧 To continue training, run: python main.py --continue-from {model_path}")


if __name__ == "__main__":
    verbose_training()
