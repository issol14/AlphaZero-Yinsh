#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH MCTS Performance Benchmark
================================

일반 MCTS vs 병렬 MCTS 성능 비교
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
import argparse
import json
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yinsh import YinshEnv, YinshAgent, Color, config


class MCTSBenchmark:
    """MCTS 성능 벤치마크 클래스"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.results = {}
        
    def benchmark_sequential_mcts(self, mcts_sims: int = 400, num_games: int = 5):
        """순차 MCTS 벤치마크"""
        print(f"🧪 순차 MCTS 벤치마크 ({mcts_sims} 시뮬레이션, {num_games}게임)")
        
        # 에이전트 생성
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        agent = YinshAgent(
            model_path=self.model_path, 
            use_mcts=True, 
            use_parallel_mcts=False,
            device=device
        )
        
        # MCTS 설정
        agent.mcts_agent.mcts.num_simulations = mcts_sims
        
        total_time = 0
        total_simulations = 0
        game_times = []
        
        for game_id in range(num_games):
            print(f"  게임 {game_id + 1}/{num_games} 시작...")
            
            env = YinshEnv()
            game_start = time.time()
            turn_count = 0
            
            while not env.is_game_over() and turn_count < 50:
                current_player = agent
                
                # MCTS 시간 측정
                mcts_start = time.time()
                action, action_info = current_player.select_action(env)
                mcts_time = time.time() - mcts_start
                
                total_time += mcts_time
                if action_info and "mcts_stats" in action_info:
                    stats = action_info["mcts_stats"]
                    total_simulations += stats.get("simulations", mcts_sims)
                
                env.step(action)
                turn_count += 1
                
                if turn_count % 10 == 0:
                    print(f"    턴 {turn_count}: MCTS 시간 {mcts_time:.3f}초")
            
            game_time = time.time() - game_start
            game_times.append(game_time)
            print(f"  게임 {game_id + 1} 완료: {game_time:.1f}초, {turn_count}턴")
        
        # 통계 계산
        avg_game_time = np.mean(game_times)
        simulations_per_second = total_simulations / total_time if total_time > 0 else 0
        
        self.results['sequential'] = {
            'total_time': total_time,
            'total_simulations': total_simulations,
            'avg_game_time': avg_game_time,
            'simulations_per_second': simulations_per_second,
            'game_times': game_times,
            'mcts_sims': mcts_sims,
            'num_games': num_games
        }
        
        print(f"✅ 순차 MCTS 완료: {total_time:.1f}초, {simulations_per_second:.1f} 시뮬/초")
    
    def benchmark_parallel_mcts(self, mcts_sims: int = 400, num_games: int = 5, num_threads: int = 4):
        """병렬 MCTS 벤치마크"""
        print(f"🧪 병렬 MCTS 벤치마크 ({mcts_sims} 시뮬레이션, {num_games}게임, {num_threads}스레드)")
        
        # 에이전트 생성
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        agent = YinshAgent(
            model_path=self.model_path, 
            use_mcts=True, 
            use_parallel_mcts=True,
            device=device
        )
        
        # 병렬 MCTS 설정
        agent.mcts_agent.mcts.num_simulations = mcts_sims
        agent.mcts_agent.mcts.num_threads = num_threads
        
        total_time = 0
        total_simulations = 0
        game_times = []
        
        for game_id in range(num_games):
            print(f"  게임 {game_id + 1}/{num_games} 시작...")
            
            env = YinshEnv()
            game_start = time.time()
            turn_count = 0
            
            while not env.is_game_over() and turn_count < 50:
                current_player = agent
                
                # MCTS 시간 측정
                mcts_start = time.time()
                action, action_info = current_player.select_action(env)
                mcts_time = time.time() - mcts_start
                
                total_time += mcts_time
                if action_info and "mcts_stats" in action_info:
                    stats = action_info["mcts_stats"]
                    total_simulations += stats.get("simulations", mcts_sims)
                
                env.step(action)
                turn_count += 1
                
                if turn_count % 10 == 0:
                    print(f"    턴 {turn_count}: MCTS 시간 {mcts_time:.3f}초")
            
            game_time = time.time() - game_start
            game_times.append(game_time)
            print(f"  게임 {game_id + 1} 완료: {game_time:.1f}초, {turn_count}턴")
        
        # 통계 계산
        avg_game_time = np.mean(game_times)
        simulations_per_second = total_simulations / total_time if total_time > 0 else 0
        
        self.results['parallel'] = {
            'total_time': total_time,
            'total_simulations': total_simulations,
            'avg_game_time': avg_game_time,
            'simulations_per_second': simulations_per_second,
            'game_times': game_times,
            'mcts_sims': mcts_sims,
            'num_games': num_games,
            'num_threads': num_threads
        }
        
        print(f"✅ 병렬 MCTS 완료: {total_time:.1f}초, {simulations_per_second:.1f} 시뮬/초")
    
    def compare_results(self):
        """결과 비교"""
        if 'sequential' not in self.results or 'parallel' not in self.results:
            print("❌ 비교할 결과가 없습니다.")
            return
        
        seq = self.results['sequential']
        par = self.results['parallel']
        
        print("\n📊 MCTS 성능 비교")
        print("=" * 60)
        print(f"{'지표':<20} {'순차 MCTS':<15} {'병렬 MCTS':<15} {'개선율':<10}")
        print("-" * 60)
        
        # 총 시간 비교
        time_improvement = (seq['total_time'] - par['total_time']) / seq['total_time'] * 100
        print(f"{'총 시간 (초)':<20} {seq['total_time']:<15.1f} {par['total_time']:<15.1f} {time_improvement:>+8.1f}%")
        
        # 평균 게임 시간 비교
        game_time_improvement = (seq['avg_game_time'] - par['avg_game_time']) / seq['avg_game_time'] * 100
        print(f"{'평균 게임 시간 (초)':<20} {seq['avg_game_time']:<15.1f} {par['avg_game_time']:<15.1f} {game_time_improvement:>+8.1f}%")
        
        # 시뮬레이션/초 비교
        sims_improvement = (par['simulations_per_second'] - seq['simulations_per_second']) / seq['simulations_per_second'] * 100
        print(f"{'시뮬레이션/초':<20} {seq['simulations_per_second']:<15.1f} {par['simulations_per_second']:<15.1f} {sims_improvement:>+8.1f}%")
        
        # 총 시뮬레이션 수
        print(f"{'총 시뮬레이션 수':<20} {seq['total_simulations']:<15} {par['total_simulations']:<15}")
        
        print("-" * 60)
        
        # 결론
        if time_improvement > 0:
            print(f"🎉 병렬 MCTS가 {time_improvement:.1f}% 빠릅니다!")
        else:
            print(f"⚠️ 병렬 MCTS가 {abs(time_improvement):.1f}% 느립니다.")
    
    def save_results(self, output_file: str):
        """결과 저장"""
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'results': self.results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        print(f"📁 결과 저장: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="YINSH MCTS 성능 벤치마크")
    parser.add_argument("--model", type=str, required=True, help="모델 경로")
    parser.add_argument("--mcts-sims", type=int, default=400, help="MCTS 시뮬레이션 수")
    parser.add_argument("--games", type=int, default=5, help="게임 수")
    parser.add_argument("--threads", type=int, default=4, help="병렬 스레드 수")
    parser.add_argument("--output", type=str, default="mcts_benchmark_results.json", help="결과 저장 파일")
    
    args = parser.parse_args()
    
    print("🧪 YINSH MCTS 성능 벤치마크 시작")
    print("=" * 60)
    
    benchmark = MCTSBenchmark(args.model)
    
    # 순차 MCTS 벤치마크
    benchmark.benchmark_sequential_mcts(args.mcts_sims, args.games)
    
    print("\n" + "=" * 60)
    
    # 병렬 MCTS 벤치마크
    benchmark.benchmark_parallel_mcts(args.mcts_sims, args.games, args.threads)
    
    # 결과 비교
    benchmark.compare_results()
    
    # 결과 저장
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()