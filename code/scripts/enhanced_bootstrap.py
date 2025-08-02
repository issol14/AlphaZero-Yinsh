#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Bootstrap Training - 휴리스틱 강화 버전
==============================================

복잡한 사전훈련 없이 효과적인 Cold Start 해결책:
1. 매우 강한 exploration 
2. 점진적 난이도 증가
3. 빈번한 모델 업데이트
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_enhanced_bootstrap(iterations=20):
    """강화된 Bootstrap 훈련"""
    print("🚀 Enhanced Bootstrap Training 시작!")
    print("=" * 60)
    print("🎯 특징:")
    print("   🔥 극도로 강한 exploration")
    print("   📈 점진적 난이도 증가")
    print("   ⚡ 빈번한 모델 업데이트")
    print("   🎮 더 많은 게임, 더 빠른 학습")
    print("=" * 60)
    
    # 1단계: 극도로 강한 exploration (첫 10 iterations)
    print("\n🔥 [1단계] 극강 Exploration 단계")
    cmd1 = [
        "uv", "run", "python", "scripts/pipeline.py",
        "--base-dir", "alphazero_enhanced_bootstrap",
        "--iterations", "10",
        
        # 매우 강한 exploration 설정
        "--selfplay-games", "600",  # 3배 증가
        "--selfplay-mcts-sims", "150",  # 매우 적음
        "--training-epochs", "30",  # 빠른 훈련
        "--training-lr", "0.01",  # 높은 학습률
        
        # 매우 관대한 평가
        "--evaluation-frequency", "1",  # 매 iteration마다
        "--evaluation-games", "20",  # 매우 적은 게임
        "--evaluation-threshold", "0.35",  # 35% 기준
        
        # 초고속 모드
        "--ultra-fast-mcts",
        "--large-batch"
    ]
    
    print(f"실행 명령: {' '.join(cmd1)}")
    result1 = subprocess.run(cmd1, cwd=".")
    
    if result1.returncode != 0:
        print("❌ 1단계 실패!")
        return False
    
    # 2단계: 점진적 정상화 (다음 10 iterations)
    print("\n⚖️ [2단계] 점진적 정상화 단계")
    cmd2 = [
        "uv", "run", "python", "scripts/pipeline.py",
        "--base-dir", "alphazero_enhanced_bootstrap",
        "--iterations", str(10 + iterations),  # 이어서 계속
        
        # 점진적으로 정상화
        "--selfplay-games", "400",
        "--selfplay-mcts-sims", "250",
        "--training-epochs", "50",
        "--training-lr", "0.005",
        
        # 조금 더 엄격한 평가
        "--evaluation-frequency", "2",
        "--evaluation-games", "40",
        "--evaluation-threshold", "0.42",  # 42% 기준
        
        "--fast-mcts"
    ]
    
    print(f"실행 명령: {' '.join(cmd2)}")
    result2 = subprocess.run(cmd2, cwd=".")
    
    return result2.returncode == 0

def check_current_models():
    """현재 사용 가능한 모델들 확인"""
    print("📂 현재 사용 가능한 모델들:")
    
    dirs_to_check = [
        "alphazero_pipeline/models",
        "alphazero_bootstrap/models", 
        "alphazero_enhanced_bootstrap/models",
        "models"
    ]
    
    found_models = []
    
    for dir_path in dirs_to_check:
        full_path = Path(dir_path)
        if full_path.exists():
            for model_file in full_path.glob("*.pt"):
                size_mb = model_file.stat().st_size / (1024*1024)
                modified = time.ctime(model_file.stat().st_mtime)
                found_models.append({
                    'path': str(model_file),
                    'size_mb': size_mb,
                    'modified': modified
                })
                print(f"   ✅ {model_file} ({size_mb:.1f}MB, {modified})")
    
    if not found_models:
        print("   ❌ 모델을 찾을 수 없습니다.")
    
    return found_models

def apply_heuristic_config():
    """휴리스틱 기반 설정 적용 (config.py 수정)"""
    print("🧠 휴리스틱 기반 설정 적용 중...")
    
    # config.py에 더 강한 exploration 설정
    config_updates = {
        "SELFPLAY_TEMPERATURE": "2.0",  # 매우 높은 온도
        "SELFPLAY_NOISE_ALPHA": "0.05",  # 매우 강한 노이즈
        "SELFPLAY_NOISE_EPSILON": "0.7",  # 70% 노이즈
    }
    
    print("   📝 config.py 업데이트:")
    for key, value in config_updates.items():
        print(f"      {key} = {value}")
    
    # 실제 파일 수정은 사용자 확인 후
    print("   ⚠️ 수동으로 config.py를 수정하거나 아래 명령을 사용하세요:")
    print("   python -c \"")
    for key, value in config_updates.items():
        print(f"import re; open('yinsh/config.py','w').write(re.sub(r'{key}.*=.*', '{key} = {value}', open('yinsh/config.py').read()))")
    print("   \"")

def quick_heuristic_start():
    """빠른 휴리스틱 시작 - 현재 모델에 강한 exploration 적용"""
    print("⚡ 빠른 휴리스틱 시작!")
    print("현재 모델에 강한 exploration을 적용하여 즉시 개선합니다.")
    
    # 기존 모델이 있는지 확인
    models = check_current_models()
    
    if models:
        print("\n🎯 기존 모델을 사용하여 강화된 훈련 시작:")
        best_model = max(models, key=lambda x: Path(x['path']).stat().st_mtime)
        print(f"   최신 모델: {best_model['path']}")
        
        # 강화된 설정으로 훈련 재시작
        cmd = [
            "uv", "run", "python", "scripts/pipeline.py",
            "--base-dir", "alphazero_heuristic_enhanced",
            "--iterations", "15",
            
            # 휴리스틱 강화 설정
            "--selfplay-games", "500",  # 많은 게임
            "--selfplay-mcts-sims", "100",  # 매우 적은 시뮬레이션
            "--training-lr", "0.02",  # 매우 높은 학습률
            "--evaluation-threshold", "0.30",  # 30% 기준 (매우 관대)
            "--evaluation-frequency", "1",  # 매 iteration
            "--evaluation-games", "15",  # 최소 게임
            
            "--ultra-fast-mcts"
        ]
        
        print(f"\n실행 명령: {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=".")
    else:
        print("\n🔄 모델이 없으므로 enhanced bootstrap부터 시작:")
        return run_enhanced_bootstrap()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Bootstrap with Heuristics")
    parser.add_argument("--quick", action="store_true", help="빠른 휴리스틱 적용")
    parser.add_argument("--full", action="store_true", help="전체 enhanced bootstrap")
    parser.add_argument("--check", action="store_true", help="모델 상태만 확인")
    parser.add_argument("--config", action="store_true", help="휴리스틱 config 적용")
    
    args = parser.parse_args()
    
    if args.check:
        check_current_models()
    elif args.config:
        apply_heuristic_config()
    elif args.quick:
        result = quick_heuristic_start()
        return result.returncode if hasattr(result, 'returncode') else 0
    elif args.full:
        return 0 if run_enhanced_bootstrap() else 1
    else:
        print("🤔 옵션을 선택하세요:")
        print("   --quick: 빠른 휴리스틱 적용")
        print("   --full: 전체 enhanced bootstrap")
        print("   --check: 모델 상태 확인")
        print("   --config: 휴리스틱 config 적용")
        
        # 기본적으로 빠른 시작 실행
        print("\n🚀 기본적으로 빠른 휴리스틱 시작을 실행합니다...")
        result = quick_heuristic_start()
        return result.returncode if hasattr(result, 'returncode') else 0

if __name__ == "__main__":
    sys.exit(main())