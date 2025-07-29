#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH AlphaZero Performance Monitor
===================================

Selfplay 성능을 모니터링하고 최적화하는 도구
"""

import os
import sys
import time
import psutil
import torch
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yinsh import YinshEnv, YinshAgent, Color, config


class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self):
        self.start_time = time.time()
        self.game_times = []
        self.mcts_times = []
        self.memory_usage = []
        self.gpu_usage = []
        
    def start_game(self):
        """게임 시작 시간 기록"""
        self.game_start = time.time()
        
    def end_game(self):
        """게임 종료 시간 기록"""
        game_time = time.time() - self.game_start
        self.game_times.append(game_time)
        
    def record_mcts_time(self, mcts_time):
        """MCTS 시간 기록"""
        self.mcts_times.append(mcts_time)
        
    def record_system_stats(self):
        """시스템 통계 기록"""
        # CPU 및 메모리 사용량
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # GPU 사용량 (CUDA 사용 가능한 경우)
        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
        
        self.memory_usage.append({
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / 1024**3,
            'gpu_memory_gb': gpu_memory,
            'timestamp': time.time()
        })
        
    def get_stats(self):
        """통계 반환"""
        total_time = time.time() - self.start_time
        
        stats = {
            'total_time': total_time,
            'games_played': len(self.game_times),
            'avg_game_time': np.mean(self.game_times) if self.game_times else 0,
            'min_game_time': np.min(self.game_times) if self.game_times else 0,
            'max_game_time': np.max(self.game_times) if self.game_times else 0,
            'avg_mcts_time': np.mean(self.mcts_times) if self.mcts_times else 0,
            'games_per_hour': len(self.game_times) / (total_time / 3600) if total_time > 0 else 0,
            'system_stats': self.memory_usage
        }
        
        return stats
    
    def print_stats(self):
        """통계 출력"""
        stats = self.get_stats()
        
        print("\n📊 성능 통계")
        print("=" * 50)
        print(f"총 실행 시간: {stats['total_time']:.1f}초")
        print(f"완료된 게임 수: {stats['games_played']}")
        print(f"평균 게임 시간: {stats['avg_game_time']:.2f}초")
        print(f"최소 게임 시간: {stats['min_game_time']:.2f}초")
        print(f"최대 게임 시간: {stats['max_game_time']:.2f}초")
        print(f"평균 MCTS 시간: {stats['avg_mcts_time']:.3f}초")
        print(f"시간당 게임 수: {stats['games_per_hour']:.1f}")
        
        if stats['system_stats']:
            latest = stats['system_stats'][-1]
            print(f"\n시스템 사용량:")
            print(f"  CPU: {latest['cpu_percent']:.1f}%")
            print(f"  메모리: {latest['memory_percent']:.1f}% ({latest['memory_used_gb']:.1f}GB)")
            if latest['gpu_memory_gb']:
                print(f"  GPU 메모리: {latest['gpu_memory_gb']:.1f}GB")


def benchmark_single_game(model_path: str, mcts_sims: int = 400):
    """단일 게임 벤치마크"""
    print(f"🧪 단일 게임 벤치마크 (MCTS 시뮬레이션: {mcts_sims})")
    
    monitor = PerformanceMonitor()
    
    # 에이전트 생성
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent1 = YinshAgent(model_path=model_path, use_mcts=True, device=device)
    agent2 = YinshAgent(model_path=model_path, use_mcts=True, device=device)
    
    # MCTS 설정
    agent1.mcts_agent.mcts.num_simulations = mcts_sims
    agent2.mcts_agent.mcts.num_simulations = mcts_sims
    
    monitor.start_game()
    
    # 게임 실행
    env = YinshEnv()
    turn_count = 0
    
    while not env.is_game_over() and turn_count < 100:
        current_player = agent1 if env.current_player == Color.WHITE else agent2
        
        # MCTS 시간 측정
        mcts_start = time.time()
        action, action_info = current_player.select_action(env)
        mcts_time = time.time() - mcts_start
        
        monitor.record_mcts_time(mcts_time)
        monitor.record_system_stats()
        
        env.step(action)
        turn_count += 1
        
        if turn_count % 10 == 0:
            print(f"  턴 {turn_count}: MCTS 시간 {mcts_time:.3f}초")
    
    monitor.end_game()
    monitor.print_stats()
    
    return monitor.get_stats()


def benchmark_parallel_games(model_path: str, num_games: int = 5, workers: int = 4):
    """병렬 게임 벤치마크"""
    print(f"🧪 병렬 게임 벤치마크 ({num_games}게임, {workers}워커)")
    
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    def run_single_game_benchmark(game_id):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        agent1 = YinshAgent(model_path=model_path, use_mcts=True, device=device)
        agent2 = YinshAgent(model_path=model_path, use_mcts=True, device=device)
        
        # 빠른 설정
        agent1.mcts_agent.mcts.num_simulations = 200
        agent2.mcts_agent.mcts.num_simulations = 200
        
        start_time = time.time()
        
        env = YinshEnv()
        turn_count = 0
        
        while not env.is_game_over() and turn_count < 50:
            current_player = agent1 if env.current_player == Color.WHITE else agent2
            action, _ = current_player.select_action(env)
            env.step(action)
            turn_count += 1
        
        game_time = time.time() - start_time
        
        return {
            'game_id': game_id,
            'game_time': game_time,
            'turns': turn_count,
            'winner': env.get_winner().name if env.get_winner() else 'DRAW'
        }
    
    # 병렬 실행
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_single_game_benchmark, i) for i in range(num_games)]
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"  게임 {result['game_id']+1} 완료: {result['game_time']:.1f}초, {result['turns']}턴")
    
    total_time = time.time() - start_time
    
    # 통계 계산
    game_times = [r['game_time'] for r in results]
    avg_game_time = np.mean(game_times)
    games_per_hour = len(results) / (total_time / 3600)
    
    print(f"\n📊 병렬 벤치마크 결과:")
    print(f"총 시간: {total_time:.1f}초")
    print(f"평균 게임 시간: {avg_game_time:.2f}초")
    print(f"시간당 게임 수: {games_per_hour:.1f}")
    print(f"워커당 게임 수: {len(results)/workers:.1f}")


def main():
    parser = argparse.ArgumentParser(description="YINSH AlphaZero 성능 모니터링")
    parser.add_argument("--model", type=str, required=True, help="모델 경로")
    parser.add_argument("--benchmark", choices=["single", "parallel"], default="single",
                       help="벤치마크 타입")
    parser.add_argument("--games", type=int, default=5, help="게임 수 (병렬 모드)")
    parser.add_argument("--workers", type=int, default=4, help="워커 수 (병렬 모드)")
    parser.add_argument("--mcts-sims", type=int, default=400, help="MCTS 시뮬레이션 수")
    
    args = parser.parse_args()
    
    if args.benchmark == "single":
        benchmark_single_game(args.model, args.mcts_sims)
    else:
        benchmark_parallel_games(args.model, args.games, args.workers)


if __name__ == "__main__":
    main()