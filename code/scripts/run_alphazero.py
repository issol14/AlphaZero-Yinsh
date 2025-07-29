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

5. 실험 폴더 지정:
   uv run python scripts/run_alphazero.py --quick --exp-name my_experiment
"""

import os
import sys
import argparse
import subprocess
import json
from datetime import datetime
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ExperimentManager:
    """실험별 격리 관리 클래스"""
    
    def __init__(self, exp_name: str = None, exp_type: str = "custom"):
        self.exp_type = exp_type
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if exp_name:
            self.exp_name = f"exp_{self.timestamp}_{exp_name}"
        else:
            self.exp_name = f"exp_{self.timestamp}_{exp_type}"
        
        # 실험 디렉토리 구조
        self.exp_dir = Path("experiments") / self.exp_name
        self.models_dir = self.exp_dir / "models"
        self.data_dir = self.exp_dir / "data"
        self.logs_dir = self.exp_dir / "logs"
        self.plots_dir = self.exp_dir / "plots"
        self.history_dir = self.models_dir / "history"
        
        # 디렉토리 생성
        self._create_directories()
        
        print(f"🧪 실험 폴더 생성: {self.exp_dir}")
    
    def _create_directories(self):
        """실험 디렉토리 구조 생성"""
        directories = [
            self.exp_dir,
            self.models_dir,
            self.data_dir,
            self.logs_dir,
            self.plots_dir,
            self.history_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_config(self, config: dict):
        """실험 설정 저장"""
        config_file = self.exp_dir / "config.json"
        config['experiment_name'] = self.exp_name
        config['created_at'] = datetime.now().isoformat()
        config['exp_type'] = self.exp_type
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"📋 실험 설정 저장: {config_file}")
    
    def get_pipeline_args(self, base_config: dict) -> list:
        """파이프라인 실행 인자 생성"""
        args = [
            "uv", "run", "python", "scripts/pipeline.py",
            "--models-dir", str(self.models_dir),
            "--data-dir", str(self.data_dir),
            "--logs-dir", str(self.logs_dir),
        ]
        
        # 기본 설정 추가
        for key, value in base_config.items():
            if key in ['iterations', 'max_hours', 'selfplay_games', 'selfplay_mcts_sims',
                      'training_epochs', 'training_batch_size', 'training_lr',
                      'evaluation_games', 'evaluation_threshold', 'evaluation_mcts_sims',
                      'workers']:
                if key == 'iterations':
                    args.extend(["--iterations", str(value)])
                elif key == 'max_hours':
                    args.extend(["--max-hours", str(value)])
                elif key == 'selfplay_games':
                    args.extend(["--selfplay-games", str(value)])
                elif key == 'selfplay_mcts_sims':
                    args.extend(["--selfplay-mcts-sims", str(value)])
                elif key == 'training_epochs':
                    args.extend(["--training-epochs", str(value)])
                elif key == 'training_batch_size':
                    args.extend(["--training-batch-size", str(value)])
                elif key == 'training_lr':
                    args.extend(["--training-lr", str(value)])
                elif key == 'evaluation_games':
                    args.extend(["--evaluation-games", str(value)])
                elif key == 'evaluation_threshold':
                    args.extend(["--evaluation-threshold", str(value)])
                elif key == 'evaluation_mcts_sims':
                    args.extend(["--evaluation-mcts-sims", str(value)])
                elif key == 'workers':
                    args.extend(["--workers", str(value)])
        
        # 플래그 옵션들
        if base_config.get('continue_training', False):
            args.append("--continue-training")
        if base_config.get('cleanup_data', True):
            args.append("--cleanup-data")
        if base_config.get('parallel', True):
            args.append("--parallel")
        
        return args


def run_demo_mode(exp_manager: ExperimentManager):
    """데모 모드: 빠른 학습을 위한 작은 규모 설정"""
    print("🎯 DEMO 모드 실행")
    print("작은 규모로 빠르게 AlphaZero 파이프라인을 체험합니다.")
    print("=" * 60)
    
    config = {
        'iterations': 5,
        'max_hours': 2,
        'selfplay_games': 10,
        'selfplay_mcts_sims': 100,
        'training_epochs': 5,
        'training_batch_size': 32,
        'training_lr': 0.001,
        'evaluation_games': 20,
        'evaluation_mcts_sims': 100,
        'evaluation_threshold': 0.55,
        'cleanup_data': True,
        'parallel': True,
        'workers': 6
    }
    
    exp_manager.save_config(config)
    cmd = exp_manager.get_pipeline_args(config)
    
    print(f"실행 명령어: {' '.join(cmd)}")
    print(f"실험 폴더: {exp_manager.exp_dir}")
    print("\n🚀 시작합니다...")
    
    try:
        subprocess.run(cmd, cwd=Path.cwd())
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다.")


def run_quick_mode(exp_manager: ExperimentManager):
    """빠른 모드: 적당한 규모의 학습"""
    print("⚡ QUICK 모드 실행")
    print("적당한 규모로 효과적인 학습을 진행합니다.")
    print("=" * 60)
    
    config = {
        'iterations': 20,
        'max_hours': 8,
        'selfplay_games': 50,
        'selfplay_mcts_sims': 400,
        'training_epochs': 8,
        'training_batch_size': 32,
        'training_lr': 0.001,
        'evaluation_games': 50,
        'evaluation_mcts_sims': 200,
        'evaluation_threshold': 0.55,
        'continue_training': True,
        'cleanup_data': True,
        'parallel': True,
        'workers': 6
    }
    
    exp_manager.save_config(config)
    cmd = exp_manager.get_pipeline_args(config)
    
    print(f"실행 명령어: {' '.join(cmd)}")
    print(f"실험 폴더: {exp_manager.exp_dir}")
    print("\n🚀 시작합니다...")
    
    try:
        subprocess.run(cmd, cwd=Path.cwd())
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다.")


def run_full_mode(exp_manager: ExperimentManager):
    """풀 모드: 완전한 AlphaZero 학습 (AlphaZero 논문 기반)"""
    print("🔥 FULL 모드 실행 (AlphaZero 논문 기반)")
    print("완전한 규모의 AlphaZero 학습을 진행합니다 (오랜 시간 소요).")
    print("=" * 60)
    
    config = {
        'iterations': 100,
        'max_hours': 20000,
        'selfplay_games': 100,
        'selfplay_mcts_sims': 800,
        'training_epochs': 100,  # AlphaZero 논문: 100 에포크
        'training_batch_size': 512,  # AlphaZero 논문: 512
        'training_lr': 0.002,  # AlphaZero 논문: 0.002
        'evaluation_games': 400,  # AlphaZero 논문: 400게임 토너먼트
        'evaluation_mcts_sims': 800,  # AlphaZero 논문: 정확한 평가용 시뮬레이션
        'evaluation_threshold': 0.55,
        'continue_training': True,
        'cleanup_data': True,
        'parallel': True,
        'workers': 6,
        # 최적화 옵션들
        'fast_mode': True,  # 빠른 모드 활성화
        'parallel_mcts': True,  # MCTS 병렬화
        'mcts_threads': 4,  # MCTS 스레드 수
        'training_optimization': True,  # 훈련 최적화
        'training_workers': 4,  # 훈련 워커 수
        'mixed_precision': True,  # 혼합 정밀도
        'evaluation_parallel': True,  # 평가 병렬화
        'evaluation_workers': 4,  # 평가 워커 수
        'evaluation_fast': True,  # 빠른 평가
        'memory_optimization': True,  # 메모리 최적화
    }
    
    exp_manager.save_config(config)
    cmd = exp_manager.get_pipeline_args(config)
    
    print(f"실행 명령어: {' '.join(cmd)}")
    print(f"실험 폴더: {exp_manager.exp_dir}")
    print("\n🚀 시작합니다...")
    
    try:
        subprocess.run(cmd, cwd=Path.cwd())
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다.")


def run_custom_mode(exp_manager: ExperimentManager):
    """커스텀 모드: 사용자 설정"""
    print("⚙️ CUSTOM 모드 실행")
    print("사용자가 직접 설정을 입력합니다.")
    print("=" * 60)
    
    try:
        print("실험 설정을 입력하세요:")
        iterations = int(input("최대 iteration 수 (기본: 20): ") or "20")
        max_hours = int(input("최대 실행 시간(시간) (기본: 8): ") or "8")
        selfplay_games = int(input("Self-play 게임 수 (기본: 50): ") or "50")
        selfplay_mcts_sims = int(input("Self-play MCTS 시뮬레이션 수 (기본: 400): ") or "400")
        training_epochs = int(input("훈련 epoch 수 (기본: 8): ") or "8")
        evaluation_games = int(input("평가 게임 수 (기본: 50): ") or "50")
        evaluation_mcts_sims = int(input("평가 MCTS 시뮬레이션 수 (기본: 200): ") or "200")
        workers = int(input("병렬 워커 수 (기본: 6): ") or "6")
        
        config = {
            'iterations': iterations,
            'max_hours': max_hours,
            'selfplay_games': selfplay_games,
            'selfplay_mcts_sims': selfplay_mcts_sims,
            'training_epochs': training_epochs,
            'training_batch_size': 32,
            'training_lr': 0.001,
            'evaluation_games': evaluation_games,
            'evaluation_mcts_sims': evaluation_mcts_sims,
            'evaluation_threshold': 0.55,
            'continue_training': True,
            'cleanup_data': True,
            'parallel': True,
            'workers': workers
        }
        
        exp_manager.save_config(config)
        cmd = exp_manager.get_pipeline_args(config)
        
        print(f"\n실행 명령어: {' '.join(cmd)}")
        print(f"실험 폴더: {exp_manager.exp_dir}")
        print("\n🚀 시작합니다...")
        
        subprocess.run(cmd, cwd=Path.cwd())
        
    except (ValueError, KeyboardInterrupt):
        print("\n❌ 설정이 취소되었습니다.")


def show_system_status():
    """현재 시스템 상태 보기"""
    print("📊 시스템 상태")
    print("=" * 60)
    
    # 실험 폴더 목록
    experiments_dir = Path("experiments")
    if experiments_dir.exists():
        experiments = list(experiments_dir.glob("exp_*"))
        if experiments:
            print("📁 실험 폴더:")
            for exp in sorted(experiments, key=lambda x: x.stat().st_mtime, reverse=True):
                exp_time = datetime.fromtimestamp(exp.stat().st_mtime)
                config_file = exp / "config.json"
                if config_file.exists():
                    try:
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        exp_type = config.get('exp_type', 'unknown')
                        print(f"  📂 {exp.name} ({exp_type}) - {exp_time.strftime('%Y-%m-%d %H:%M')}")
                    except:
                        print(f"  📂 {exp.name} - {exp_time.strftime('%Y-%m-%d %H:%M')}")
                else:
                    print(f"  📂 {exp.name} - {exp_time.strftime('%Y-%m-%d %H:%M')}")
        else:
            print("📁 실험 폴더가 없습니다.")
    else:
        print("📁 experiments 디렉토리가 없습니다.")
    
    # 기존 모델 확인
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pt"))
        if model_files:
            print("\n🤖 기존 모델:")
            for model in model_files:
                model_time = datetime.fromtimestamp(model.stat().st_mtime)
                print(f"  🧠 {model.name} - {model_time.strftime('%Y-%m-%d %H:%M')}")
        else:
            print("\n🤖 기존 모델이 없습니다.")
    else:
        print("\n🤖 models 디렉토리가 없습니다.")


def run_single_evaluation():
    """단일 모델 평가 실행"""
    print("🏆 단일 모델 평가")
    print("=" * 60)
    
    # 실험 폴더에서 모델 찾기
    experiments_dir = Path("experiments")
    available_models = []
    
    if experiments_dir.exists():
        for exp_dir in experiments_dir.glob("exp_*"):
            models_dir = exp_dir / "models"
            if models_dir.exists():
                for model_file in models_dir.glob("*.pt"):
                    available_models.append((model_file, exp_dir.name))
    
    # 기존 models 폴더도 확인
    models_dir = Path("models")
    if models_dir.exists():
        for model_file in models_dir.glob("*.pt"):
            available_models.append((model_file, "models"))
    
    if not available_models:
        print("❌ 평가할 모델이 없습니다.")
        return
    
    print("사용 가능한 모델:")
    for i, (model_file, exp_name) in enumerate(available_models):
        model_time = datetime.fromtimestamp(model_file.stat().st_mtime)
        print(f"  {i+1}. {model_file.name} ({exp_name}) - {model_time.strftime('%Y-%m-%d %H:%M')}")
    
    try:
        choice = int(input("\n평가할 모델 번호: ")) - 1
        if 0 <= choice < len(available_models):
            candidate_model, exp_name = available_models[choice]
            
            # 평가용 실험 폴더 생성
            eval_exp = ExperimentManager(exp_name="evaluation", exp_type="eval")
            
            cmd = [
                "uv", "run", "python", "scripts/evaluate.py",
                "--candidate", str(candidate_model),
                "--games", "50",
                "--mcts-sims", "200",
                "--output", str(eval_exp.exp_dir)
            ]
            
            print(f"\n실행 명령어: {' '.join(cmd)}")
            print(f"평가 결과 폴더: {eval_exp.exp_dir}")
            print("\n🚀 평가 시작...")
            
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
  %(prog)s --demo                    # 빠른 데모 (10게임, 5 iteration)
  %(prog)s --quick                   # 적당한 학습 (50게임, 20 iteration)
  %(prog)s --full                    # 완전한 학습 (100게임, 100 iteration)
  %(prog)s --custom                  # 사용자 설정
  %(prog)s --status                  # 현재 상태 보기
  %(prog)s --evaluate                # 단일 모델 평가
  %(prog)s --quick --exp-name test  # 실험 이름 지정
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
    
    # 실험 이름 옵션
    parser.add_argument("--exp-name", type=str, default=None,
                       help="실험 이름 (기본: 타임스탬프)")
    
    args = parser.parse_args()
    
    print("🎮 YINSH AlphaZero 시스템")
    print("=" * 60)
    
    if args.status:
        show_system_status()
    elif args.evaluate:
        run_single_evaluation()
    else:
        # 실험 타입 결정
        if args.demo:
            exp_type = "demo"
        elif args.quick:
            exp_type = "quick"
        elif args.full:
            exp_type = "full"
        else:
            exp_type = "custom"
        
        # 실험 매니저 생성
        exp_manager = ExperimentManager(exp_name=args.exp_name, exp_type=exp_type)
        
        if args.demo:
            run_demo_mode(exp_manager)
        elif args.quick:
            run_quick_mode(exp_manager)
        elif args.full:
            run_full_mode(exp_manager)
        elif args.custom:
            run_custom_mode(exp_manager)


if __name__ == "__main__":
    main() 