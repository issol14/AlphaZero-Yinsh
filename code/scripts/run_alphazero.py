#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH AlphaZero Quick Start Launcher
====================================

AlphaZero 시스템을 쉽게 시작할 수 있는 런처 스크립트입니다.

사용법:
1. 빠른 시작 (기본 설정):
   uv run python scripts/run_alphazero.py --quick

2. 데모 실행 (작은 규모):
   uv run python scripts/run_alphazero.py --demo

3. 풀 스케일 실행:
   uv run python scripts/run_alphazero.py --full

4. 커스텀 설정:
   uv run python scripts/run_alphazero.py --custom
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_demo_mode():
    """데모 모드: 빠른 학습을 위한 작은 규모 설정"""
    print("🎯 DEMO 모드 실행")
    print("작은 규모로 빠르게 AlphaZero 파이프라인을 체험합니다.")
    print("=" * 60)
    
    cmd = [
        "uv", "run", "python", "scripts/pipeline.py",
        "--iterations", "5",
        "--max-hours", "2",
        "--selfplay-games", "10",
        "--selfplay-mcts-sims", "100",
        "--training-epochs", "5",
        "--training-batch-size", "16",
        "--evaluation-games", "20",
        "--evaluation-mcts-sims", "100",
        "--cleanup-data"
    ]
    
    print(f"실행 명령어: {' '.join(cmd)}")
    print("\n🚀 시작합니다...")
    
    try:
        subprocess.run(cmd, cwd=Path.cwd())
    except KeyboardInterrupt:
        print("\\n⏹️ 사용자에 의해 중단되었습니다.")


def run_quick_mode():
    """빠른 모드: 적당한 규모의 학습"""
    print("⚡ QUICK 모드 실행")
    print("적당한 규모로 효과적인 학습을 진행합니다.")
    print("=" * 60)
    
    cmd = [
        "uv", "run", "python", "scripts/pipeline.py",
        "--iterations", "20",
        "--max-hours", "8",
        "--selfplay-games", "50",
        "--selfplay-mcts-sims", "400",
        "--training-epochs", "8",
        "--evaluation-games", "50",
        "--evaluation-mcts-sims", "200",
        "--continue-training",
        "--cleanup-data"
    ]
    
    print(f"실행 명령어: {' '.join(cmd)}")
    print("\\n🚀 시작합니다...")
    
    try:
        subprocess.run(cmd, cwd=Path.cwd())
    except KeyboardInterrupt:
        print("\\n⏹️ 사용자에 의해 중단되었습니다.")


def run_full_mode():
    """풀 모드: 완전한 AlphaZero 학습"""
    print("🔥 FULL 모드 실행")
    print("완전한 규모의 AlphaZero 학습을 진행합니다 (오랜 시간 소요).")
    print("=" * 60)
    
    cmd = [
        "uv", "run", "python", "scripts/pipeline.py",
        "--iterations", "100",
        "--max-hours", "24",
        "--selfplay-games", "100",
        "--selfplay-mcts-sims", "800",
        "--training-epochs", "10",
        "--evaluation-games", "100",
        "--continue-training",
        "--cleanup-data"
    ]
    
    print(f"실행 명령어: {' '.join(cmd)}")
    print("\\n🚀 시작합니다...")
    
    try:
        subprocess.run(cmd, cwd=Path.cwd())
    except KeyboardInterrupt:
        print("\\n⏹️ 사용자에 의해 중단되었습니다.")


def run_custom_mode():
    """커스텀 모드: 사용자 설정으로 실행"""
    print("🎛️ CUSTOM 모드")
    print("원하는 설정으로 파이프라인을 실행합니다.")
    print("=" * 60)
    
    # 사용자 입력 받기
    try:
        iterations = int(input("Iteration 수 (기본 20): ") or "20")
        max_hours = int(input("최대 실행 시간 (시간, 기본 8): ") or "8")
        selfplay_games = int(input("Self-play 게임 수 (기본 50): ") or "50")
        mcts_sims = int(input("MCTS 시뮬레이션 수 (기본 400): ") or "400")
        evaluation_games = int(input("평가 게임 수 (기본 50): ") or "50")
        
        cleanup = input("임시 데이터 자동 삭제? (y/N): ").lower().startswith('y')
        continue_training = input("기존 모델에서 이어서 훈련? (Y/n): ").lower() != 'n'
        
    except ValueError:
        print("❌ 잘못된 입력입니다. 기본값을 사용합니다.")
        iterations = 20
        max_hours = 8
        selfplay_games = 50
        mcts_sims = 400
        evaluation_games = 50
        cleanup = True
        continue_training = True
    
    cmd = [
        "uv", "run", "python", "scripts/pipeline.py",
        "--iterations", str(iterations),
        "--max-hours", str(max_hours),
        "--selfplay-games", str(selfplay_games),
        "--selfplay-mcts-sims", str(mcts_sims),
        "--evaluation-games", str(evaluation_games),
        "--evaluation-mcts-sims", str(mcts_sims // 2),
    ]
    
    if cleanup:
        cmd.append("--cleanup-data")
    if continue_training:
        cmd.append("--continue-training")
    
    print(f"\\n실행 명령어: {' '.join(cmd)}")
    print("\\n🚀 시작합니다...")
    
    try:
        subprocess.run(cmd, cwd=Path.cwd())
    except KeyboardInterrupt:
        print("\\n⏹️ 사용자에 의해 중단되었습니다.")


def show_system_status():
    """현재 시스템 상태 보기"""
    print("📊 현재 시스템 상태")
    print("=" * 60)
    
    # Best 모델 확인
    best_model_path = Path("models/best_model.pt")
    if best_model_path.exists():
        print("✅ Best 모델 존재")
        
        # 모델 정보 확인
        model_info_path = Path("models/best_model_info.json")
        if model_info_path.exists():
            import json
            with open(model_info_path, 'r') as f:
                info = json.load(f)
            print(f"   마지막 업데이트: {info.get('updated', 'N/A')}")
            print(f"   Iteration: {info.get('iteration', 'N/A')}")
            print(f"   승률: {info.get('win_rate', 0)*100:.2f}%")
    else:
        print("❌ Best 모델 없음 (처음 실행)")
    
    # 로그 확인
    logs_dir = Path("pipeline_logs")
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        print(f"\\n📝 로그 파일: {len(log_files)}개")
        if log_files:
            latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
            print(f"   최신 로그: {latest_log}")
    
    # 평가 결과 확인
    eval_dir = Path("evaluation_results")
    if eval_dir.exists():
        eval_files = list(eval_dir.glob("*.json"))
        print(f"\\n🏆 평가 결과: {len(eval_files)}개")
    
    print("=" * 60)


def run_single_evaluation():
    """단일 모델 평가 실행"""
    print("🏆 단일 모델 평가")
    print("=" * 60)
    
    # 모델 파일 확인
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ models 디렉토리가 없습니다.")
        return
    
    model_files = list(models_dir.glob("*.pt"))
    if not model_files:
        print("❌ 평가할 모델이 없습니다.")
        return
    
    print("사용 가능한 모델:")
    for i, model in enumerate(model_files):
        print(f"  {i+1}. {model.name}")
    
    try:
        choice = int(input("\\n평가할 모델 번호: ")) - 1
        if 0 <= choice < len(model_files):
            candidate_model = model_files[choice]
            
            # 기존 Best 모델과 비교
            best_model = Path("models/best_model.pt")
            
            cmd = [
                "uv", "run", "python", "scripts/evaluate.py",
                "--candidate", str(candidate_model),
                "--games", "50",
                "--mcts-sims", "200"
            ]
            
            if best_model.exists() and best_model != candidate_model:
                cmd.extend(["--best", str(best_model)])
            
            print(f"\\n실행 명령어: {' '.join(cmd)}")
            print("\\n🚀 평가 시작...")
            
            subprocess.run(cmd, cwd=Path.cwd())
        else:
            print("❌ 잘못된 선택입니다.")
    except (ValueError, KeyboardInterrupt):
        print("❌ 평가가 취소되었습니다.")


def main():
    parser = argparse.ArgumentParser(
        description="YINSH AlphaZero 시스템 런처",
        epilog="""
사용 예시:
  %(prog)s --demo        # 빠른 데모 (10게임, 5 iteration)
  %(prog)s --quick       # 적당한 학습 (50게임, 20 iteration)
  %(prog)s --full        # 완전한 학습 (100게임, 100 iteration)
  %(prog)s --custom      # 사용자 설정
  %(prog)s --status      # 현재 상태 보기
  %(prog)s --evaluate    # 단일 모델 평가
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--demo", action="store_true", 
                      help="데모 모드 (빠른 체험)")
    group.add_argument("--quick", action="store_true",
                      help="빠른 모드 (적당한 규모)")
    group.add_argument("--full", action="store_true",
                      help="풀 모드 (완전한 학습)")
    group.add_argument("--custom", action="store_true",
                      help="커스텀 모드 (사용자 설정)")
    group.add_argument("--status", action="store_true",
                      help="현재 시스템 상태 보기")
    group.add_argument("--evaluate", action="store_true",
                      help="단일 모델 평가")
    
    args = parser.parse_args()
    
    print("🎮 YINSH AlphaZero 시스템")
    print("=" * 60)
    
    if args.demo:
        run_demo_mode()
    elif args.quick:
        run_quick_mode()
    elif args.full:
        run_full_mode()
    elif args.custom:
        run_custom_mode()
    elif args.status:
        show_system_status()
    elif args.evaluate:
        run_single_evaluation()


if __name__ == "__main__":
    main() 