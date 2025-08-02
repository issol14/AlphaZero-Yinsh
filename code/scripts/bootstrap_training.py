#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AlphaZero Bootstrap Training - Cold Start Problem 해결용
======================================================

랜덤 가중치로 시작하는 AlphaZero의 초기 학습 문제를 해결하기 위한 Bootstrap 훈련
- 매우 강한 exploration으로 다양한 패턴 학습
- 낮은 평가 기준으로 빠른 모델 업데이트
- 더 많은 self-play 게임으로 데이터 품질 향상
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_bootstrap_phase(base_dir="alphazero_bootstrap", iterations=20):
    """
    Bootstrap 단계 실행
    - 매우 강한 exploration
    - 낮은 평가 기준 (40%)
    - 많은 게임과 적은 simulation
    """
    print("🚀 AlphaZero Bootstrap Training 시작!")
    print("=" * 60)
    print("📋 Bootstrap 설정:")
    print("   🎯 목표: Cold Start Problem 해결")
    print("   🔥 High Exploration Mode")
    print("   📈 낮은 평가 기준 (40%)")
    print("   🎮 많은 게임, 적은 시뮬레이션")
    print("=" * 60)
    
    cmd = [
        "uv", "run", "python", "scripts/pipeline.py",
        "--base-dir", base_dir,
        "--iterations", str(iterations),
        
        # Self-play 설정 (많은 게임, 적은 시뮬레이션)
        "--selfplay-games", "400",  # 2배 증가
        "--selfplay-mcts-sims", "200",  # 2배 감소 (빠른 학습)
        "--selfplay-workers", "8",
        
        # 훈련 설정
        "--training-epochs", "50",  # 빠른 훈련
        "--training-batch-size", "256",  # 작은 배치
        "--training-lr", "0.005",  # 높은 학습률
        
        # 평가 설정 (관대한 기준)
        "--evaluation-frequency", "2",  # 자주 평가
        "--evaluation-games", "50",  # 적은 게임 (빠른 평가)
        "--evaluation-threshold", "0.40",  # 40% 기준 (관대함)
        "--evaluation-mcts-sims", "200",  # 빠른 평가
        
        # 최적화
        "--fast-mcts",  # 빠른 MCTS 모드
        "--large-batch"  # 큰 배치 (GPU 최적화)
    ]
    
    print(f"실행 명령: {' '.join(cmd)}")
    print("\n⏱️ Bootstrap 훈련 시작...")
    
    start_time = time.time()
    result = subprocess.run(cmd, cwd=".")
    end_time = time.time()
    
    if result.returncode == 0:
        elapsed_hours = (end_time - start_time) / 3600
        print(f"\n✅ Bootstrap 훈련 완료!")
        print(f"   총 소요시간: {elapsed_hours:.2f}시간")
        print(f"   생성된 모델: {base_dir}/models/best_model.pt")
        return True
    else:
        print(f"\n❌ Bootstrap 훈련 실패 (코드: {result.returncode})")
        return False

def run_regular_training(bootstrap_dir="alphazero_bootstrap", 
                        regular_dir="alphazero_pipeline", 
                        iterations=100):
    """
    정규 훈련 단계 (Bootstrap 모델에서 시작)
    """
    print("\n🎯 정규 AlphaZero 훈련 시작!")
    print("=" * 60)
    print("📋 정규 훈련 설정:")
    print("   📂 Bootstrap 모델 사용")
    print("   📈 표준 평가 기준 (55%)")
    print("   🎮 표준 게임 수, MCTS 시뮬레이션")
    print("=" * 60)
    
    # Bootstrap 모델을 정규 파이프라인으로 복사
    bootstrap_model = Path(bootstrap_dir) / "models" / "best_model.pt"
    regular_models_dir = Path(regular_dir) / "models"
    regular_models_dir.mkdir(parents=True, exist_ok=True)
    
    if bootstrap_model.exists():
        import shutil
        shutil.copy2(bootstrap_model, regular_models_dir / "current_model.pt")
        shutil.copy2(bootstrap_model, regular_models_dir / "best_model.pt")
        print(f"✅ Bootstrap 모델 복사 완료: {bootstrap_model}")
    else:
        print(f"❌ Bootstrap 모델을 찾을 수 없습니다: {bootstrap_model}")
        return False
    
    cmd = [
        "uv", "run", "python", "scripts/pipeline.py",
        "--base-dir", regular_dir,
        "--iterations", str(iterations),
        
        # 표준 설정
        "--selfplay-games", "200",
        "--selfplay-mcts-sims", "400",
        "--training-epochs", "100",
        "--training-batch-size", "512",
        "--training-lr", "0.002",
        "--evaluation-threshold", "0.55",  # 표준 기준
        "--evaluation-games", "400"
    ]
    
    print(f"실행 명령: {' '.join(cmd)}")
    
    start_time = time.time()
    result = subprocess.run(cmd, cwd=".")
    end_time = time.time()
    
    if result.returncode == 0:
        elapsed_hours = (end_time - start_time) / 3600
        print(f"\n🎉 정규 훈련 완료!")
        print(f"   총 소요시간: {elapsed_hours:.2f}시간")
        return True
    else:
        print(f"\n❌ 정규 훈련 실패 (코드: {result.returncode})")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="AlphaZero Bootstrap Training - Cold Start Problem 해결",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--bootstrap-iterations", type=int, default=20,
                       help="Bootstrap 단계 iteration 수")
    parser.add_argument("--regular-iterations", type=int, default=100,
                       help="정규 훈련 iteration 수")
    parser.add_argument("--bootstrap-only", action="store_true",
                       help="Bootstrap 단계만 실행")
    parser.add_argument("--skip-bootstrap", action="store_true",
                       help="Bootstrap 단계 건너뛰고 정규 훈련만")
    
    args = parser.parse_args()
    
    print("🚀 AlphaZero Bootstrap Training Pipeline")
    print("=" * 60)
    print("🎯 목적: Cold Start Problem 해결")
    print("📋 단계:")
    print("   1. Bootstrap 훈련 (강한 exploration, 관대한 평가)")
    print("   2. 정규 훈련 (Bootstrap 모델에서 시작)")
    print("=" * 60)
    
    success = True
    
    # Bootstrap 단계
    if not args.skip_bootstrap:
        print("\n🔥 [단계 1] Bootstrap 훈련")
        success = run_bootstrap_phase(iterations=args.bootstrap_iterations)
        
        if not success:
            print("❌ Bootstrap 훈련 실패!")
            return 1
    
    # 정규 훈련 단계
    if not args.bootstrap_only and success:
        print("\n🎯 [단계 2] 정규 훈련")
        success = run_regular_training(iterations=args.regular_iterations)
    
    if success:
        print("\n🎉 전체 훈련 파이프라인 완료!")
        print("✅ 이제 강화된 모델로 고품질 학습이 가능합니다!")
        return 0
    else:
        print("\n❌ 훈련 파이프라인 실패!")
        return 1

if __name__ == "__main__":
    sys.exit(main())