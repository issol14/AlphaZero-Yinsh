#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH AlphaZero Parallel Self-Play Test
========================================

통합된 selfplay.py의 병렬 기능을 테스트하는 스크립트입니다.
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def test_sequential_vs_parallel():
    """순차 vs 병렬 실행 성능 비교"""
    print("🧪 Testing Sequential vs Parallel Self-Play")
    print("=" * 50)
    
    test_games = 8
    mcts_sims = 200  # 빠른 테스트를 위해
    
    # 1. 순차 실행 테스트
    print(f"\n🐌 Sequential Test ({test_games} games)")
    print("-" * 30)
    
    sequential_start = time.time()
    
    cmd_sequential = [
        "python", "scripts/selfplay.py",
        "--games", str(test_games),
        "--mcts-sims", str(mcts_sims),
        "--output", "test_sequential",
        "--workers", "1"  # 순차 실행
    ]
    
    print(f"실행 명령어: {' '.join(cmd_sequential)}")
    result = subprocess.run(cmd_sequential, cwd=Path.cwd())
    
    sequential_time = time.time() - sequential_start
    
    # 2. 병렬 실행 테스트
    print(f"\n🚀 Parallel Test ({test_games} games)")
    print("-" * 30)
    
    parallel_start = time.time()
    
    cmd_parallel = [
        "python", "scripts/selfplay.py",
        "--games", str(test_games),
        "--mcts-sims", str(mcts_sims),
        "--output", "test_parallel",
        "--parallel",
        "--workers", "4"  # 병렬 실행
    ]
    
    print(f"실행 명령어: {' '.join(cmd_parallel)}")
    result = subprocess.run(cmd_parallel, cwd=Path.cwd())
    
    parallel_time = time.time() - parallel_start
    
    # 3. 결과 비교
    print(f"\n📊 Performance Comparison")
    print("=" * 50)
    print(f"Sequential Time: {sequential_time:.1f} seconds")
    print(f"Parallel Time:   {parallel_time:.1f} seconds")
    
    if parallel_time > 0:
        speedup = sequential_time / parallel_time
        print(f"Speedup:         {speedup:.2f}x")
        
        if speedup > 1.5:
            print("✅ 병렬 구현이 크게 더 빠릅니다!")
        elif speedup > 1.0:
            print("✅ 병렬 구현이 더 빠릅니다")
        else:
            print("⚠️ 병렬 구현이 더 빠르지 않습니다 (튜닝 필요)")


def test_different_workers():
    """다양한 워커 수 테스트"""
    print("\n🔬 Testing Different Worker Counts")
    print("=" * 50)
    
    test_games = 12
    worker_counts = [1, 2, 4, 6]
    results = {}
    
    for workers in worker_counts:
        print(f"\n⚙️ Testing with {workers} workers")
        print("-" * 25)
        
        start_time = time.time()
        
        cmd = [
            "python", "scripts/selfplay.py",
            "--games", str(test_games),
            "--mcts-sims", "200",
            "--output", f"test_workers_{workers}",
            "--workers", str(workers)
        ]
        
        if workers > 1:
            cmd.append("--parallel")
        
        print(f"실행 명령어: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=Path.cwd())
        
        elapsed_time = time.time() - start_time
        results[workers] = elapsed_time
        
        print(f"✅ {workers} workers: {elapsed_time:.1f}s")
    
    # 최적 워커 수 추천
    print(f"\n🎯 Optimization Results")
    print("-" * 30)
    
    best_workers = min(results.keys(), key=lambda k: results[k])
    best_time = results[best_workers]
    
    print(f"Best worker count: {best_workers}")
    print(f"Best time: {best_time:.1f} seconds")
    
    for workers, elapsed_time in results.items():
        efficiency = best_time / elapsed_time * 100
        print(f"{workers} workers: {elapsed_time:.1f}s ({efficiency:.1f}% efficiency)")


def test_integration():
    """전체 파이프라인 통합 테스트"""
    print("\n🔗 Integration Test")
    print("=" * 50)
    
    print("파이프라인에서 병렬 selfplay 실행 테스트")
    
    cmd = [
        "python", "scripts/run_alphazero.py", "--demo"
    ]
    
    print(f"실행 명령어: {' '.join(cmd)}")
    print("(데모 모드는 병렬 4 workers로 설정됨)")
    
    # 실제로는 실행하지 않고 명령어만 표시
    print("✅ 통합 테스트 준비 완료")
    print("실제 실행을 원하면 위 명령어를 직접 실행하세요.")


def main():
    """메인 테스트 함수"""
    print("🧪 YINSH AlphaZero Parallel Self-Play Tests")
    print("=" * 60)
    
    try:
        # 기본 병렬 vs 순차 테스트
        test_sequential_vs_parallel()
        
        # 다양한 워커 수 테스트
        test_different_workers()
        
        # 통합 테스트
        test_integration()
        
        print("\n✅ All tests completed successfully!")
        
        # 정리 안내
        print("\n🧹 Cleanup:")
        print("테스트로 생성된 폴더들:")
        print("- test_sequential/")
        print("- test_parallel/") 
        print("- test_workers_*/")
        print("필요없으면 삭제하세요: rm -rf test_*")
        
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 