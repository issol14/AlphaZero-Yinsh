#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH AlphaZero Training Monitor
================================

학습 과정을 모니터링하고 분석하는 도구
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import argparse
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yinsh import YinshEnv, YinshAgent, Color, config


class TrainingMonitor:
    """학습 모니터링 클래스"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """모델 로드"""
        try:
            from yinsh.model import YinshModelBuilder
            builder = YinshModelBuilder()
            model = builder.load_model(self.model_path)
            model.to(self.device)
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def evaluate_model(self, num_games: int = 10):
        """모델 성능 평가"""
        print(f"🧪 Evaluating model with {num_games} games...")
        
        model = self.load_model()
        if model is None:
            return None
        
        agent = YinshAgent(model_path=self.model_path, use_mcts=True, device=self.device)
        
        wins = 0
        losses = 0
        draws = 0
        game_lengths = []
        
        for game_id in range(num_games):
            env = YinshEnv()
            turn_count = 0
            
            while not env.is_game_over() and turn_count < 100:
                action, _ = agent.select_action(env)
                env.step(action)
                turn_count += 1
            
            winner = env.get_winner()
            if winner == Color.WHITE:
                wins += 1
            elif winner == Color.BLACK:
                losses += 1
            else:
                draws += 1
            
            game_lengths.append(turn_count)
            
            if (game_id + 1) % 5 == 0:
                print(f"  Game {game_id + 1}/{num_games} completed")
        
        # 통계 계산
        total_games = wins + losses + draws
        win_rate = wins / total_games if total_games > 0 else 0
        avg_game_length = np.mean(game_lengths)
        
        results = {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': win_rate,
            'avg_game_length': avg_game_length,
            'game_lengths': game_lengths
        }
        
        print(f"📊 Evaluation Results:")
        print(f"  Wins: {wins}")
        print(f"  Losses: {losses}")
        print(f"  Draws: {draws}")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Avg Game Length: {avg_game_length:.1f} turns")
        
        return results
    
    def analyze_training_data(self, data_folder: str):
        """훈련 데이터 분석"""
        print(f"📊 Analyzing training data from: {data_folder}")
        
        all_states = []
        all_policies = []
        all_values = []
        
        # 데이터 로드
        for file in os.listdir(data_folder):
            if file.endswith(".pt"):
                file_path = os.path.join(data_folder, file)
                try:
                    data = torch.load(file_path, map_location="cpu")
                    all_states.extend(data["states"])
                    all_policies.extend(data["policies"])
                    all_values.extend(data["values"])
                except Exception as e:
                    print(f"⚠️  Error loading {file}: {e}")
        
        if not all_states:
            print("❌ No training data found!")
            return None
        
        # 통계 계산
        states = np.array(all_states)
        policies = np.array(all_policies)
        values = np.array(all_values)
        
        stats = {
            'num_positions': len(states),
            'num_files': len([f for f in os.listdir(data_folder) if f.endswith('.pt')]),
            'state_shape': states.shape,
            'policy_shape': policies.shape,
            'value_range': [float(np.min(values)), float(np.max(values))],
            'value_mean': float(np.mean(values)),
            'value_std': float(np.std(values)),
            'policy_mean': float(np.mean(policies)),
            'policy_std': float(np.std(policies))
        }
        
        print(f"📈 Training Data Statistics:")
        print(f"  Total Positions: {stats['num_positions']:,}")
        print(f"  Data Files: {stats['num_files']}")
        print(f"  State Shape: {stats['state_shape']}")
        print(f"  Policy Shape: {stats['policy_shape']}")
        print(f"  Value Range: [{stats['value_range'][0]:.3f}, {stats['value_range'][1]:.3f}]")
        print(f"  Value Mean: {stats['value_mean']:.3f}")
        print(f"  Value Std: {stats['value_std']:.3f}")
        
        return stats
    
    def plot_training_curves(self, log_file: str):
        """훈련 곡선 플롯"""
        if not os.path.exists(log_file):
            print(f"❌ Log file not found: {log_file}")
            return
        
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
            
            epochs = range(1, len(logs['policy_loss']) + 1)
            
            plt.figure(figsize=(15, 5))
            
            # 손실 곡선
            plt.subplot(1, 3, 1)
            plt.plot(epochs, logs['policy_loss'], label='Policy Loss', color='blue')
            plt.plot(epochs, logs['value_loss'], label='Value Loss', color='red')
            plt.plot(epochs, logs['total_loss'], label='Total Loss', color='green')
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            # 학습률
            plt.subplot(1, 3, 2)
            plt.plot(epochs, logs['learning_rate'], color='orange')
            plt.title('Learning Rate')
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.grid(True)
            
            # 정확도 (가치 예측)
            if 'value_accuracy' in logs:
                plt.subplot(1, 3, 3)
                plt.plot(epochs, logs['value_accuracy'], color='purple')
                plt.title('Value Prediction Accuracy')
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("📊 Training curves saved as 'training_curves.png'")
            
        except Exception as e:
            print(f"❌ Error plotting training curves: {e}")
    
    def generate_report(self, data_folder: str, num_eval_games: int = 10):
        """종합 보고서 생성"""
        print("📋 Generating Training Report...")
        print("=" * 60)
        
        # 모델 정보
        model = self.load_model()
        if model:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"📊 Model Information:")
            print(f"  Parameters: {total_params:,}")
            print(f"  Device: {self.device}")
        
        # 훈련 데이터 분석
        data_stats = self.analyze_training_data(data_folder)
        
        # 모델 평가
        eval_results = self.evaluate_model(num_eval_games)
        
        # 보고서 저장
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'data_stats': data_stats,
            'evaluation_results': eval_results,
            'model_params': total_params if model else 0
        }
        
        with open('training_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 Report saved as 'training_report.json'")
        print("✅ Training analysis completed!")


def main():
    parser = argparse.ArgumentParser(description="YINSH AlphaZero Training Monitor")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--data", type=str, default="memory", help="Training data folder")
    parser.add_argument("--eval-games", type=int, default=10, help="Number of evaluation games")
    parser.add_argument("--log-file", type=str, help="Training log file for plotting")
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.model)
    
    # 데이터 분석
    monitor.analyze_training_data(args.data)
    
    # 모델 평가
    monitor.evaluate_model(args.eval_games)
    
    # 훈련 곡선 플롯 (로그 파일이 있는 경우)
    if args.log_file:
        monitor.plot_training_curves(args.log_file)
    
    # 종합 보고서
    monitor.generate_report(args.data, args.eval_games)


if __name__ == "__main__":
    main()