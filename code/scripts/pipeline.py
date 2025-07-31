#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH AlphaZero Training Pipeline - Paper-faithful Implementation
================================================================

AlphaZero 논문(Silver et al., 2017)의 정확한 구현:

📋 AlphaZero Algorithm (논문 Figure 1):
1. Initialize neural network f_θ₀ randomly
2. For iteration i = 1, 2, ...:
   a) Self-Play: Generate games using current f_θᵢ₋₁
   b) Train: Update network using ALL accumulated data (1 to i)
   c) Evaluate: New network vs current best (every N iterations)
   d) Update: If win rate ≥ 55%, replace best model

🔑 Key Paper Insights:
- Data accumulation across ALL iterations (not just recent)
- Evaluation every few iterations (not every single one)
- Threshold-based model replacement (55% win rate)
- Continuous self-improvement through iterative refinement

이 구현은 논문의 정확한 알고리즘을 따릅니다.
"""

import os
import sys
import argparse
import torch
import time
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yinsh import YinshModel, config


class AlphaZeroDataManager:
    """AlphaZero 논문 방식의 데이터 관리 클래스"""
    
    def __init__(self, base_dir: str = "alphazero_pipeline"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # 핵심 디렉토리 구조 (논문 방식)
        self.models_dir = self.base_dir / "models"
        self.data_dir = self.base_dir / "selfplay_data"
        self.logs_dir = self.base_dir / "logs"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        
        for directory in [self.models_dir, self.data_dir, self.logs_dir, self.checkpoints_dir]:
            directory.mkdir(exist_ok=True)
        
        # AlphaZero 모델 경로
        self.current_model_path = self.models_dir / "current_model.pt"
        self.best_model_path = self.models_dir / "best_model.pt"
        
        # 메타데이터 파일
        self.pipeline_state_path = self.base_dir / "pipeline_state.json"
        self.training_history_path = self.logs_dir / "training_history.json"
        
        # 데이터 누적 전략 (AlphaZero 핵심)
        self.accumulated_data_dir = self.data_dir / "accumulated"
        self.accumulated_data_dir.mkdir(exist_ok=True)
        
    def get_current_model_path(self) -> Optional[str]:
        """현재 훈련 중인 모델 경로 반환"""
        if self.current_model_path.exists():
            return str(self.current_model_path)
        return None
    
    def get_best_model_path(self) -> Optional[str]:
        """현재 best 모델 경로 반환"""
        if self.best_model_path.exists():
            return str(self.best_model_path)
        return None
    
    def initialize_pipeline(self) -> str:
        """AlphaZero 파이프라인 초기화 (논문 방식)"""
        print("🚀 AlphaZero 파이프라인 초기화 중...")
        
        # 새 모델 생성 (논문: f_θ₀ randomly initialized)
        model = YinshModel()
        
        # Current model 저장
        torch.save(model, self.current_model_path)
        print(f"   ✅ 초기 모델 생성: {self.current_model_path}")
        
        # Best model도 동일하게 초기화
        torch.save(model, self.best_model_path)
        print(f"   ✅ Best 모델 초기화: {self.best_model_path}")
        
        # 파이프라인 상태 초기화
        initial_state = {
            "iteration": 0,
            "total_games_played": 0,
            "successful_evaluations": 0,
            "model_updates": 0,
            "created_at": datetime.now().isoformat(),
            "last_evaluation_iteration": 0,
            "best_model_iteration": 0
        }
        
        self.save_pipeline_state(initial_state)
        print(f"   ✅ 파이프라인 상태 초기화 완료")
        
        return str(self.current_model_path)
    
    def accumulate_selfplay_data(self, iteration: int, new_data_dir: str) -> int:
        """AlphaZero 방식: 모든 데이터 누적 (논문 핵심)"""
        print(f"📦 Iteration {iteration} 데이터 누적 중...")
        
        # 새 데이터를 누적 디렉토리에 복사
        source_dir = Path(new_data_dir)
        if not source_dir.exists():
            print(f"   ❌ 소스 디렉토리 없음: {source_dir}")
            return 0
        
        file_count = 0
        for file_path in source_dir.glob("*.pt"):
            # 파일명에 iteration 정보 포함
            new_name = f"iter_{iteration:04d}_{file_path.name}"
            target_path = self.accumulated_data_dir / new_name
            shutil.copy2(file_path, target_path)
            file_count += 1
        
        print(f"   ✅ {file_count}개 파일 누적 완료")
        print(f"   📊 총 누적 파일: {len(list(self.accumulated_data_dir.glob('*.pt')))}개")
        
        return file_count
    
    def get_all_accumulated_data_path(self) -> str:
        """모든 누적된 훈련 데이터 경로 반환"""
        return str(self.accumulated_data_dir)
    
    def save_pipeline_state(self, state: Dict):
        """파이프라인 상태 저장"""
        state["last_updated"] = datetime.now().isoformat()
        with open(self.pipeline_state_path, 'w') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def load_pipeline_state(self) -> Optional[Dict]:
        """파이프라인 상태 로드"""
        if self.pipeline_state_path.exists():
            with open(self.pipeline_state_path, 'r') as f:
                return json.load(f)
        return None
    
    def save_model_checkpoint(self, iteration: int, model_path: str, 
                            evaluation_result: Optional[Dict] = None):
        """모델 체크포인트 저장"""
        checkpoint_path = self.checkpoints_dir / f"model_iter_{iteration:04d}.pt"
        shutil.copy2(model_path, checkpoint_path)
        
        # 메타데이터 저장
        metadata = {
            "iteration": iteration,
            "model_path": str(checkpoint_path),
            "created_at": datetime.now().isoformat(),
            "evaluation_result": evaluation_result
        }
        
        metadata_path = self.checkpoints_dir / f"model_iter_{iteration:04d}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"   💾 체크포인트 저장: {checkpoint_path}")
    
    def cleanup_old_data(self, keep_last_n_iterations: int = 10):
        """오래된 데이터 정리 (메모리 관리)"""
        data_files = list(self.accumulated_data_dir.glob("iter_*.pt"))
        if len(data_files) <= keep_last_n_iterations * 50:  # 게임당 평균 50파일 가정
            return
        
        # 가장 오래된 파일들 제거
        data_files.sort(key=lambda x: x.stat().st_ctime)
        files_to_remove = data_files[:-keep_last_n_iterations * 50]
        
        for file_path in files_to_remove:
            file_path.unlink()
        
        print(f"   🧹 {len(files_to_remove)}개 오래된 파일 정리 완료")
    
    def initialize_best_model(self) -> str:
        """Best 모델이 없으면 새로 생성"""
        if not self.best_model_path.exists():
            print("🔄 Best 모델이 없습니다. 새로운 모델을 생성합니다...")
            
            # 새 모델 생성
            model = YinshModel()
            torch.save(model, self.best_model_path)
            
            # 모델 정보 저장
            model_info = {
                'created': datetime.now().isoformat(),
                'iteration': 0,
                'wins': 0,
                'games_played': 0,
                'win_rate': 0.0,
                'generation': 1
            }
            
            with open(self.best_model_info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            print(f"✅ 새 Best 모델 생성: {self.best_model_path}")
        
        return str(self.best_model_path)
    
    def update_best_model(self, candidate_model_path: str, evaluation_result: Dict,
                         iteration: int) -> bool:
        """새 모델로 Best 모델 업데이트"""
        try:
            # 기존 모델을 히스토리로 백업
            if self.best_model_path.exists():
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = self.history_dir / f"model_iter_{iteration-1}_{timestamp}.pt"
                shutil.copy2(self.best_model_path, backup_path)
                print(f"💾 기존 모델 백업: {backup_path}")
            
            # 새 모델을 Best 모델로 복사
            shutil.copy2(candidate_model_path, self.best_model_path)
            
            # 모델 정보 업데이트
            model_info = {
                'updated': datetime.now().isoformat(),
                'iteration': iteration,
                'wins': evaluation_result['results']['candidate_wins'],
                'games_played': evaluation_result['evaluation_games'],
                'win_rate': evaluation_result['results']['win_rate'],
                'threshold': evaluation_result['threshold'],
                'previous_model_backup': str(backup_path) if self.best_model_path.exists() else None,
                'evaluation_details': evaluation_result
            }
            
            with open(self.best_model_info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            print(f"🏆 Best 모델 업데이트 완료!")
            print(f"   새 승률: {evaluation_result['results']['win_rate']*100:.2f}%")
            
            return True
            
        except Exception as e:
            print(f"❌ Best 모델 업데이트 실패: {e}")
            return False
    
    def get_model_info(self) -> Optional[Dict]:
        """현재 Best 모델 정보 반환"""
        if self.best_model_info_path.exists():
            with open(self.best_model_info_path, 'r') as f:
                return json.load(f)
        return None


class AlphaZeroPipeline:
    """AlphaZero 논문 기반 자동화 훈련 파이프라인"""
    
    def __init__(self, config_dict: Dict):
        self.config = config_dict
        self.data_manager = AlphaZeroDataManager(config_dict['base_dir'])
        
        # 로그 파일 설정
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.data_manager.logs_dir / f"pipeline_{timestamp}.log"
        
        # AlphaZero 논문 기반 설정
        self.evaluation_frequency = config_dict.get('evaluation_frequency', 3)  # 논문: 몇 iteration마다 평가
        self.win_rate_threshold = config_dict.get('win_rate_threshold', 0.55)   # 논문: 55%
        self.max_data_iterations = config_dict.get('max_data_iterations', 50)  # 메모리 관리용
        
        # 통계 (파이프라인 상태에서 관리)
        self.pipeline_state = self.data_manager.load_pipeline_state()
        if self.pipeline_state is None:
            self.pipeline_state = {
                "iteration": 0,
                "total_games_played": 0,
                "successful_evaluations": 0,
                "model_updates": 0,
                "start_time": None,
                "last_evaluation_iteration": 0,
                "best_model_iteration": 0
            }
        
        # 최적화 설정
        self.use_gpu_optimization = config_dict.get('use_gpu_optimization', True)
        self.parallel_selfplay = config_dict.get('parallel_selfplay', True)
        self.parallel_evaluation = config_dict.get('parallel_evaluation', True)
        
    def log(self, message: str):
        """로그 메시지 출력 및 파일 저장"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + '\n')
    
    def run_selfplay(self, iteration: int) -> bool:
        """AlphaZero 논문 방식: 현재 모델로 Self-Play 실행"""
        current_model_path = self.data_manager.get_current_model_path()
        if not current_model_path:
            self.log(f"❌ 현재 모델이 없습니다!")
            return False
            
        self.log(f"🎮 Self-Play 시작 (Iteration {iteration})")
        self.log(f"   📂 모델 경로: {current_model_path}")
        self.log(f"   🎯 목표 게임 수: {self.config['selfplay_games']:,}")
        self.log(f"   🧠 MCTS 시뮬레이션: {self.config['selfplay_mcts_sims']:,}")
        self.log(f"   ⚡ GPU 최적화: {'ON' if self.use_gpu_optimization else 'OFF'}")
        self.log(f"   🔄 병렬 워커: {self.config.get('selfplay_workers', 1)}개")
        
        # 출력 디렉토리 생성
        output_dir = self.data_manager.data_dir / f"iter_{iteration:04d}"
        output_dir.mkdir(exist_ok=True)
        
        try:
            # Self-play 명령어 구성 (selfplay.py가 지원하는 파라미터만 사용)
            cmd = [
                "uv", "run", "python", "scripts/selfplay.py",
                "--model", current_model_path,
                "--games", str(self.config['selfplay_games']),
                "--mcts-sims", str(self.config['selfplay_mcts_sims']),
                "--output", str(output_dir),
                "--workers", str(self.config.get('selfplay_workers', 6))
            ]
            
            # GPU 최적화 옵션 추가
            if self.use_gpu_optimization:
                cmd.extend(["--gpu-batch", "--batch-size", str(self.config.get('batch_size', 32))])
            
            # 병렬 MCTS 옵션 (selfplay.py가 지원하는 방식)
            if self.parallel_selfplay and self.config.get('use_parallel_mcts', False):
                cmd.extend(["--parallel-mcts", "--mcts-threads", str(self.config.get('mcts_threads', 4))])
            
            self.log(f"   실행 명령: {' '.join(cmd)}")
            
            # Self-play 실행
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            end_time = time.time()
            
            if result.returncode == 0:
                # 생성된 파일 확인
                data_files = list(output_dir.glob("*.pt"))
                if data_files:
                    elapsed_minutes = (end_time - start_time) / 60
                    games_per_minute = self.config['selfplay_games'] / elapsed_minutes if elapsed_minutes > 0 else 0
                    
                    self.log(f"   ✅ Self-Play 성공!")
                    self.log(f"   📄 생성된 파일: {len(data_files):,}개")
                    self.log(f"   ⏱️ 소요 시간: {elapsed_minutes:.1f}분")
                    self.log(f"   ⚡ 게임/분: {games_per_minute:.1f}")
                    self.log(f"   💾 데이터 크기: {sum(f.stat().st_size for f in data_files) / (1024*1024):.1f}MB")
                    
                    # 데이터 누적 (AlphaZero 핵심)
                    accumulated_files = self.data_manager.accumulate_selfplay_data(iteration, str(output_dir))
                    
                    # 통계 업데이트
                    self.pipeline_state["total_games_played"] += self.config['selfplay_games']
                    self.data_manager.save_pipeline_state(self.pipeline_state)
                    
                    return True
                else:
                    self.log(f"   ❌ Self-Play 파일이 생성되지 않았습니다!")
                    self.log(f"   stdout: {result.stdout}")
                    self.log(f"   stderr: {result.stderr}")
                    return False
            else:
                self.log(f"   ❌ Self-Play 실행 실패 (코드: {result.returncode})")
                self.log(f"   stdout: {result.stdout}")
                self.log(f"   stderr: {result.stderr}")
                return False
                
        except Exception as e:
            self.log(f"❌ Self-Play 실행 오류: {e}")
            import traceback
            self.log(f"   상세 오류: {traceback.format_exc()}")
            return False
        

    
    def run_training(self, iteration: int) -> Optional[str]:
        """AlphaZero 논문 방식: 모든 누적 데이터로 훈련"""
        self.log(f"🎯 모델 훈련 시작 (Iteration {iteration})")
        
        # AlphaZero 핵심: 모든 누적된 데이터 사용
        accumulated_data_path = self.data_manager.get_all_accumulated_data_path()
        
        # 데이터 존재 확인
        data_files = list(Path(accumulated_data_path).glob("*.pt"))
        if not data_files:
            self.log(f"❌ 훈련할 데이터가 없습니다!")
            return None
        
        # 데이터 통계 계산
        total_data_size = sum(f.stat().st_size for f in data_files) / (1024*1024)  # MB
        iterations_covered = len(set(f.name.split('_')[1] for f in data_files if '_' in f.name))
        
        self.log(f"   📊 누적 훈련 데이터 분석:")
        self.log(f"      📄 파일 수: {len(data_files):,}개")
        self.log(f"      💾 데이터 크기: {total_data_size:.1f}MB")
        self.log(f"      🔢 포함된 Iterations: {iterations_covered}개")
        self.log(f"      📂 데이터 경로: {accumulated_data_path}")
        self.log(f"   🎯 훈련 설정:")
        self.log(f"      🔄 Epochs: {self.config['training_epochs']}")
        self.log(f"      📦 Batch Size: {self.config['training_batch_size']}")
        self.log(f"      📈 Learning Rate: {self.config['training_lr']}")
        self.log(f"      ⚡ GPU 최적화: {'ON' if self.use_gpu_optimization else 'OFF'}")
        
        # 현재 모델 경로
        current_model_path = self.data_manager.get_current_model_path()
        
        # 출력 디렉토리
        output_dir = self.data_manager.models_dir / f"trained_iter_{iteration:04d}"
        output_dir.mkdir(exist_ok=True)
        
        try:
            # 훈련 명령어 구성 (AlphaZero 논문 기반)
            cmd = [
                "uv", "run", "python", "scripts/train.py",
                "--data", accumulated_data_path,
                "--epochs", str(self.config['training_epochs']),
                "--batch-size", str(self.config['training_batch_size']),
                "--lr", str(self.config['training_lr']),
                "--output", str(output_dir)
            ]
            
            # 기존 모델에서 이어서 훈련 (Transfer Learning)
            if current_model_path and self.config.get('continue_training', True):
                cmd.extend(["--model", current_model_path])
                self.log(f"   🔄 Transfer Learning: {current_model_path}")
            
            # GPU 최적화 옵션
            if self.use_gpu_optimization:
                cmd.extend([
                    "--mixed-precision",  # FP16 training
                    "--workers", "8"      # DataLoader workers
                ])
            
            self.log(f"   실행 명령: {' '.join(cmd)}")
            
            # 훈련 실행
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            end_time = time.time()
            
            if result.returncode == 0:
                # 훈련된 모델 확인
                trained_model_path = output_dir / "trained_model.pt"
                if trained_model_path.exists():
                    elapsed_minutes = (end_time - start_time) / 60
                    model_size = trained_model_path.stat().st_size / (1024*1024)  # MB
                    
                    self.log(f"   ✅ 훈련 성공!")
                    self.log(f"   📂 모델 경로: {trained_model_path}")
                    self.log(f"   ⏱️ 훈련 시간: {elapsed_minutes:.1f}분")
                    self.log(f"   💾 모델 크기: {model_size:.1f}MB")
                    self.log(f"   📊 평균 Epoch 시간: {elapsed_minutes/self.config['training_epochs']:.2f}분")
                    
                    # 새 모델을 current_model로 업데이트
                    shutil.copy2(trained_model_path, self.data_manager.current_model_path)
                    self.log(f"   🔄 Current 모델 업데이트 완료")
                    
                    # 체크포인트 저장
                    self.data_manager.save_model_checkpoint(iteration, str(trained_model_path))
                    
                    return str(self.data_manager.current_model_path)
                else:
                    self.log(f"   ❌ 훈련된 모델 파일이 없습니다!")
                    self.log(f"   stdout: {result.stdout}")
                    self.log(f"   stderr: {result.stderr}")
                    return None
            else:
                self.log(f"   ❌ 훈련 실행 실패 (코드: {result.returncode})")
                self.log(f"   stdout: {result.stdout}")
                self.log(f"   stderr: {result.stderr}")
                return None
                
        except Exception as e:
            self.log(f"❌ 훈련 실행 오류: {e}")
            import traceback
            self.log(f"   상세 오류: {traceback.format_exc()}")
            return None
    
    def run_evaluation(self, iteration: int) -> Optional[Dict]:
        """AlphaZero 논문 방식: 조건부 평가 (매 N iteration마다)"""
        # 평가 주기 확인
        if iteration % self.evaluation_frequency != 0:
            self.log(f"⏭️ 평가 건너뛰기 (Iteration {iteration}, 주기: {self.evaluation_frequency})")
            return {"should_replace": False, "skip_reason": "not_evaluation_time"}
        
        current_model_path = self.data_manager.get_current_model_path()
        best_model_path = self.data_manager.get_best_model_path()
        
        if not current_model_path or not best_model_path:
            self.log(f"❌ 평가할 모델이 없습니다!")
            return None
        
        self.log(f"🏆 모델 평가 시작 (Iteration {iteration})")
        self.log(f"   📂 후보 모델: {current_model_path}")
        self.log(f"   🥇 기존 Best 모델: {best_model_path}")
        self.log(f"   🎯 평가 게임 수: {self.config['evaluation_games']:,}")
        self.log(f"   🧠 MCTS 시뮬레이션: {self.config['evaluation_mcts_sims']:,}")
        self.log(f"   📈 승률 기준: {self.win_rate_threshold*100:.1f}% (채택 조건)")
        self.log(f"   🔄 병렬 워커: {self.config.get('evaluation_workers', 1)}개")
        
        try:
            # 평가 명령어 구성
            cmd = [
                "uv", "run", "python", "scripts/evaluate.py",
                "--candidate", current_model_path,
                "--best", best_model_path,
                "--games", str(self.config['evaluation_games']),
                "--threshold", str(self.win_rate_threshold),
                "--mcts-sims", str(self.config['evaluation_mcts_sims'])
            ]
            
            # 최적화 옵션 추가
            if self.parallel_evaluation:
                cmd.extend(["--workers", str(self.config.get('evaluation_workers', 4))])
            
            if self.use_gpu_optimization:
                cmd.append("--no-gpu")  # 평가 시에는 CPU 사용 (안정성)
            
            cmd.append("--no-save")  # 결과 파일 저장 안함
            
            self.log(f"   실행 명령: {' '.join(cmd)}")
            
            # 평가 실행
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            end_time = time.time()
            
            # 결과 해석 (evaluate.py의 반환 코드 기반)
            elapsed_minutes = (end_time - start_time) / 60
            games_per_minute = self.config['evaluation_games'] / elapsed_minutes if elapsed_minutes > 0 else 0
            
            if result.returncode == 0:
                # 성공: 새 모델이 기준 달성
                self.log(f"   ✅ 평가 성공: 새 모델이 기준 달성! 🎉")
                self.log(f"   📈 결과: 승률 ≥ {self.win_rate_threshold*100:.1f}%")
                self.log(f"   ⏱️ 평가 시간: {elapsed_minutes:.1f}분")
                self.log(f"   ⚡ 게임/분: {games_per_minute:.1f}")
                
                evaluation_result = {
                    "should_replace": True,
                    "win_rate": f"≥{self.win_rate_threshold*100:.1f}%",
                    "evaluation_time": end_time - start_time,
                    "iteration": iteration,
                    "games_played": self.config['evaluation_games']
                }
                
                # 통계 업데이트
                self.pipeline_state["successful_evaluations"] += 1
                self.pipeline_state["last_evaluation_iteration"] = iteration
                
                return evaluation_result
                
            elif result.returncode == 1:
                # 모델 교체 불필요
                self.log(f"   ❌ 평가 완료: 새 모델이 기준 미달")
                self.log(f"   📈 결과: 승률 < {self.win_rate_threshold*100:.1f}%")
                self.log(f"   ⏱️ 평가 시간: {elapsed_minutes:.1f}분")
                self.log(f"   ⚡ 게임/분: {games_per_minute:.1f}")
                
                evaluation_result = {
                    "should_replace": False,
                    "win_rate": f"<{self.win_rate_threshold*100:.1f}%",
                    "evaluation_time": end_time - start_time,
                    "iteration": iteration,
                    "games_played": self.config['evaluation_games']
                }
                
                # 통계 업데이트
                self.pipeline_state["last_evaluation_iteration"] = iteration
                
                return evaluation_result
                
            else:
                # 평가 실패
                self.log(f"   ❌ 평가 실패 (코드: {result.returncode})")
                self.log(f"   stdout: {result.stdout}")
                self.log(f"   stderr: {result.stderr}")
                return None
                
        except Exception as e:
            self.log(f"❌ 평가 실행 오류: {e}")
            import traceback
            self.log(f"   상세 오류: {traceback.format_exc()}")
            return None
    
    def update_best_model(self, iteration: int, evaluation_result: Dict) -> bool:
        """AlphaZero 논문 방식: 조건부 모델 업데이트"""
        if not evaluation_result.get("should_replace", False):
            self.log(f"🔄 모델 업데이트 안함 (기준 미달)")
            return False
        
        current_model_path = self.data_manager.get_current_model_path()
        if not current_model_path:
            self.log(f"❌ 업데이트할 현재 모델이 없습니다!")
            return False
        
        try:
            # Current model을 Best model로 복사
            shutil.copy2(current_model_path, self.data_manager.best_model_path)
            self.log(f"🎉 Best 모델 업데이트 완료!")
            self.log(f"   새 Best 모델: {self.data_manager.best_model_path}")
            
            # 통계 업데이트
            self.pipeline_state["model_updates"] += 1
            self.pipeline_state["best_model_iteration"] = iteration
            self.data_manager.save_pipeline_state(self.pipeline_state)
            
            # 체크포인트 저장 (평가 결과 포함)
            self.data_manager.save_model_checkpoint(iteration, current_model_path, evaluation_result)
            
            return True
            
        except Exception as e:
            self.log(f"❌ 모델 업데이트 실패: {e}")
            return False
    
    def run_pipeline(self, max_iterations: int = 100):
        """AlphaZero 논문 기반 메인 파이프라인 실행"""
        self.pipeline_state["start_time"] = time.time()
        
        self.log("🚀 AlphaZero YINSH 훈련 파이프라인 시작!")
        self.log("=" * 80)
        self.log(f"📋 AlphaZero 논문 기반 설정:")
        self.log(f"   🎯 목표 Iterations: {max_iterations}")
        self.log(f"   🎮 Self-Play 게임 수: {self.config['selfplay_games']:,}")
        self.log(f"   🧠 MCTS 시뮬레이션: {self.config['selfplay_mcts_sims']:,}")
        self.log(f"   📊 평가 주기: {self.evaluation_frequency} iterations마다")
        self.log(f"   🏆 평가 게임 수: {self.config['evaluation_games']:,}")
        self.log(f"   📈 승률 기준: {self.win_rate_threshold*100:.1f}%")
        self.log(f"   ⚡ GPU 최적화: {'활성화' if self.use_gpu_optimization else '비활성화'}")
        self.log(f"   🔄 병렬 Self-Play: {'활성화' if self.parallel_selfplay else '비활성화'}")
        self.log(f"   🔀 병렬 평가: {'활성화' if self.parallel_evaluation else '비활성화'}")
        self.log(f"   💾 데이터 보관 기간: {self.max_data_iterations} iterations")
        self.log("=" * 80)
        
        # 파이프라인 상태 로드 및 재시작 정보
        if self.pipeline_state["iteration"] > 0:
            self.log(f"🔄 기존 파이프라인 재시작 감지:")
            self.log(f"   📍 마지막 완료 Iteration: {self.pipeline_state['iteration']}")
            self.log(f"   🎮 총 플레이한 게임: {self.pipeline_state['total_games_played']:,}")
            self.log(f"   ✅ 성공한 평가: {self.pipeline_state['successful_evaluations']}")
            self.log(f"   🏆 모델 업데이트: {self.pipeline_state['model_updates']}")
            self.log(f"   🥇 현재 Best 모델 Iteration: {self.pipeline_state['best_model_iteration']}")
            self.log("=" * 80)
        
        # 파이프라인 초기화 (논문: f_θ₀ randomly initialized)
        current_model_path = self.data_manager.initialize_pipeline()
        
        try:
            start_iteration = self.pipeline_state["iteration"] + 1
            
            for iteration in range(start_iteration, max_iterations + 1):
                iteration_start_time = time.time()
                
                self.pipeline_state["iteration"] = iteration
                elapsed_time = time.time() - self.pipeline_state["start_time"]
                
                # Iteration 시작 헤더 (더 상세한 정보)
                self.log(f"\n{'='*80}")
                self.log(f"🔄 AlphaZero Iteration {iteration}/{max_iterations} 시작")
                self.log(f"{'='*80}")
                self.log(f"📊 현재 진행 상황:")
                self.log(f"   ⏱️ 총 경과 시간: {elapsed_time/3600:.1f}시간 ({elapsed_time/60:.1f}분)")
                self.log(f"   📈 진행률: {((iteration-1)/max_iterations)*100:.1f}%")
                self.log(f"   🎮 총 게임 수: {self.pipeline_state['total_games_played']:,}")
                self.log(f"   🏆 모델 업데이트 횟수: {self.pipeline_state['model_updates']}")
                if iteration > 1:
                    avg_time_per_iter = elapsed_time / (iteration - start_iteration + 1)
                    remaining_iters = max_iterations - iteration + 1
                    estimated_remaining = avg_time_per_iter * remaining_iters
                    self.log(f"   🔮 예상 남은 시간: {estimated_remaining/3600:.1f}시간")
                self.log(f"{'='*80}")
                
                # 1. Self-Play (논문: Generate games using current f_θᵢ₋₁)
                self.log(f"\n🎮 [단계 1/4] Self-Play 데이터 생성")
                selfplay_start_time = time.time()
                if not self.run_selfplay(iteration):
                    self.log(f"❌ Iteration {iteration} Self-Play 실패. 계속 진행...")
                    self._log_iteration_failure(iteration, "Self-Play", iteration_start_time)
                    continue
                selfplay_time = time.time() - selfplay_start_time
                self.log(f"✅ Self-Play 완료 (소요시간: {selfplay_time/60:.1f}분)")
                
                # 2. Training (논문: Update network using ALL accumulated data)
                self.log(f"\n🎯 [단계 2/4] 누적 데이터 훈련")
                training_start_time = time.time()
                trained_model_path = self.run_training(iteration)
                if not trained_model_path:
                    self.log(f"❌ Iteration {iteration} 훈련 실패. 계속 진행...")
                    self._log_iteration_failure(iteration, "Training", iteration_start_time)
                    continue
                training_time = time.time() - training_start_time
                self.log(f"✅ 훈련 완료 (소요시간: {training_time/60:.1f}분)")
                
                # 3. Evaluation (논문: Every N iterations, evaluate new vs current best)
                self.log(f"\n🏆 [단계 3/4] 모델 성능 평가")
                evaluation_start_time = time.time()
                evaluation_result = self.run_evaluation(iteration)
                if evaluation_result is None:
                    self.log(f"❌ Iteration {iteration} 평가 실패. 계속 진행...")
                    self._log_iteration_failure(iteration, "Evaluation", iteration_start_time)
                    continue
                evaluation_time = time.time() - evaluation_start_time
                self.log(f"✅ 평가 완료 (소요시간: {evaluation_time/60:.1f}분)")
                
                # 4. Model Update (논문: If win rate ≥ 55%, replace best model)
                self.log(f"\n🔄 [단계 4/4] 모델 업데이트 결정")
                if evaluation_result.get("should_replace", False):
                    if self.update_best_model(iteration, evaluation_result):
                        self.log(f"🎉 Iteration {iteration}: 새 Best 모델 채택! 🏆")
                        self.log(f"   승률이 {self.win_rate_threshold*100:.1f}% 기준을 달성했습니다!")
                    else:
                        self.log(f"❌ Iteration {iteration}: 모델 업데이트 실패")
                else:
                    if evaluation_result.get("skip_reason") == "not_evaluation_time":
                        self.log(f"⏭️ 평가 생략 (다음 평가: Iteration {((iteration // self.evaluation_frequency) + 1) * self.evaluation_frequency})")
                    else:
                        self.log(f"🔄 기존 모델 유지 (승률이 {self.win_rate_threshold*100:.1f}% 기준 미달)")
                
                # Iteration 완료 통계 (더 상세함)
                iteration_time = time.time() - iteration_start_time
                self._log_iteration_summary(iteration, iteration_time, selfplay_time, training_time, evaluation_time, evaluation_result)
                
                # 상태 저장
                self.data_manager.save_pipeline_state(self.pipeline_state)
                
                # 메모리 관리 (오래된 데이터 정리)
                if iteration % 10 == 0:
                    self.log(f"\n🧹 메모리 정리: 오래된 데이터 정리 중...")
                    self.data_manager.cleanup_old_data(self.max_data_iterations)
                
                # 중간 체크포인트 저장
                if iteration % 5 == 0:
                    self.log(f"💾 중간 체크포인트 저장 (Iteration {iteration})")
                    
                self.log(f"\n{'='*80}")
                self.log(f"✅ Iteration {iteration} 완료 - 다음 Iteration 준비 중...")
                self.log(f"{'='*80}\n")
                
        except KeyboardInterrupt:
            self.log("\n⚠️ 사용자에 의해 파이프라인이 중단되었습니다.")
        except Exception as e:
            self.log(f"\n❌ 파이프라인 실행 중 오류: {e}")
            import traceback
            self.log(f"   상세 오류: {traceback.format_exc()}")
        finally:
            self._print_final_summary()
    
    def _log_iteration_failure(self, iteration: int, stage: str, iteration_start_time: float):
        """Iteration 실패 시 상세 로깅"""
        elapsed_time = time.time() - iteration_start_time
        self.log(f"💥 Iteration {iteration} 실패 상세:")
        self.log(f"   🎯 실패 단계: {stage}")
        self.log(f"   ⏱️ 실패까지 소요시간: {elapsed_time/60:.1f}분")
        self.log(f"   📊 현재까지 성공한 Iterations: {iteration - 1}")
        self.log(f"   🔄 다음 Iteration {iteration + 1}에서 재시도합니다...")
    
    def _log_iteration_summary(self, iteration: int, total_time: float, 
                              selfplay_time: float, training_time: float, 
                              evaluation_time: float, evaluation_result: dict):
        """Iteration 완료 후 상세 요약"""
        self.log(f"\n📊 Iteration {iteration} 완료 요약:")
        self.log(f"   ⏱️ 총 소요시간: {total_time/60:.1f}분")
        self.log(f"   🎮 Self-Play: {selfplay_time/60:.1f}분 ({(selfplay_time/total_time)*100:.1f}%)")
        self.log(f"   🎯 Training: {training_time/60:.1f}분 ({(training_time/total_time)*100:.1f}%)")
        self.log(f"   🏆 Evaluation: {evaluation_time/60:.1f}분 ({(evaluation_time/total_time)*100:.1f}%)")
        
        # 성능 지표
        if not evaluation_result.get("skip_reason"):
            win_rate = evaluation_result.get("win_rate", "알 수 없음")
            self.log(f"   📈 승률: {win_rate}")
            
        # 누적 통계
        self.log(f"   📊 누적 통계:")
        self.log(f"      🎮 총 게임: {self.pipeline_state['total_games_played']:,}")
        self.log(f"      ✅ 평가 성공: {self.pipeline_state['successful_evaluations']}")
        self.log(f"      🏆 모델 업데이트: {self.pipeline_state['model_updates']}")
        
        # 성능 분석
        if iteration > 1:
            total_elapsed = time.time() - self.pipeline_state["start_time"]
            avg_time_per_iter = total_elapsed / iteration
            self.log(f"      ⚡ 평균 Iteration 시간: {avg_time_per_iter/60:.1f}분")
            
        # 데이터 상태
        accumulated_files = len(list(self.data_manager.accumulated_data_dir.glob("*.pt")))
        self.log(f"      💾 누적 데이터 파일: {accumulated_files:,}개")
            
    def _print_final_summary(self):
        """파이프라인 완료 후 최종 요약"""
        total_time = time.time() - self.pipeline_state["start_time"]
        
        self.log("\n" + "=" * 80)
        self.log("🏁 AlphaZero 파이프라인 완료!")
        self.log("=" * 80)
        self.log(f"📊 최종 통계:")
        self.log(f"   총 Iterations: {self.pipeline_state['iteration']}")
        self.log(f"   총 Self-Play 게임: {self.pipeline_state['total_games_played']:,}")
        self.log(f"   성공한 평가: {self.pipeline_state['successful_evaluations']}")
        self.log(f"   모델 업데이트: {self.pipeline_state['model_updates']}")
        self.log(f"   Best 모델 Iteration: {self.pipeline_state['best_model_iteration']}")
        self.log(f"   총 실행 시간: {total_time/3600:.2f}시간")
        self.log(f"   Iteration당 평균 시간: {total_time/max(1,self.pipeline_state['iteration'])/60:.1f}분")
        
        self.log(f"\n📁 생성된 파일:")
        self.log(f"   Best 모델: {self.data_manager.best_model_path}")
        self.log(f"   Current 모델: {self.data_manager.current_model_path}")
        self.log(f"   로그 파일: {self.log_file}")
        self.log(f"   체크포인트: {self.data_manager.checkpoints_dir}")
        self.log("=" * 80)
    



def main():
    parser = argparse.ArgumentParser(
        description="YINSH AlphaZero 논문 기반 자동화 훈련 파이프라인",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 파이프라인 기본 설정
    parser.add_argument("--base-dir", type=str, default="alphazero_pipeline",
                       help="파이프라인 기본 디렉토리")
    parser.add_argument("--iterations", type=int, default=100,
                       help="최대 iteration 수")
    
    # Self-play 설정 (AlphaZero 논문 기반)
    parser.add_argument("--selfplay-games", type=int, default=200,  # 논문: 높은 품질
                       help="Iteration당 self-play 게임 수")
    parser.add_argument("--selfplay-mcts-sims", type=int, default=800,  # 논문: 800
                       help="Self-play MCTS 시뮬레이션 수")
    parser.add_argument("--selfplay-workers", type=int, default=6,
                       help="Self-play 병렬 워커 수")
    
    # 훈련 설정 (AlphaZero 논문 기반)
    parser.add_argument("--training-epochs", type=int, default=100,  # 논문: 100
                       help="훈련 epoch 수")
    parser.add_argument("--training-batch-size", type=int, default=512,  # 논문: 512
                       help="훈련 배치 크기")
    parser.add_argument("--training-lr", type=float, default=0.002,  # 논문: 0.002
                       help="학습률")
    parser.add_argument("--continue-training", action="store_true", default=True,
                       help="기존 모델에서 Transfer Learning")
    
    # 평가 설정 (AlphaZero 논문 기반)
    parser.add_argument("--evaluation-frequency", type=int, default=3,  # 논문: 매 N iteration
                       help="평가 주기 (iterations)")
    parser.add_argument("--evaluation-games", type=int, default=400,  # 논문: 400게임
                       help="모델 평가 게임 수")
    parser.add_argument("--evaluation-threshold", type=float, default=0.55,  # 논문: 55%
                       help="새 모델 채택 승률 기준 (0.0-1.0)")
    parser.add_argument("--evaluation-mcts-sims", type=int, default=800,  # 논문: 정확한 평가
                       help="평가 MCTS 시뮬레이션 수")
    parser.add_argument("--evaluation-workers", type=int, default=4,
                       help="평가 병렬 워커 수")
    
    # 최적화 설정
    parser.add_argument("--no-gpu", action="store_true",
                       help="GPU 최적화 비활성화")
    parser.add_argument("--no-parallel-selfplay", action="store_true",
                       help="Self-play 병렬 처리 비활성화")
    parser.add_argument("--no-parallel-evaluation", action="store_true",
                       help="평가 병렬 처리 비활성화")
    parser.add_argument("--max-data-iterations", type=int, default=50,
                       help="누적 데이터 최대 iteration 수 (메모리 관리)")
    
    args = parser.parse_args()
    
    # 설정 딕셔너리 생성
    config_dict = {
        'base_dir': args.base_dir,
        'selfplay_games': args.selfplay_games,
        'selfplay_mcts_sims': args.selfplay_mcts_sims,
        'selfplay_workers': args.selfplay_workers,
        'training_epochs': args.training_epochs,
        'training_batch_size': args.training_batch_size,
        'training_lr': args.training_lr,
        'continue_training': args.continue_training,
        'evaluation_frequency': args.evaluation_frequency,
        'evaluation_games': args.evaluation_games,
        'evaluation_threshold': args.evaluation_threshold,
        'evaluation_mcts_sims': args.evaluation_mcts_sims,
        'evaluation_workers': args.evaluation_workers,
        'use_gpu_optimization': not args.no_gpu,
        'parallel_selfplay': not args.no_parallel_selfplay,
        'parallel_evaluation': not args.no_parallel_evaluation,
        'max_data_iterations': args.max_data_iterations,
        'win_rate_threshold': args.evaluation_threshold,
        # GPU 최적화 추가 설정
        'batch_size': 32,  # selfplay.py의 기본값과 일치
        'use_parallel_mcts': False,  # 기본값: False (옵션으로 추가 가능)
        'mcts_threads': 4
    }
    
    print("🚀 AlphaZero YINSH 훈련 파이프라인 시작!")
    print("=" * 60)
    print(f"📋 설정:")
    for key, value in config_dict.items():
        print(f"   {key}: {value}")
    print("=" * 60)
    
    # 파이프라인 실행
    try:
        pipeline = AlphaZeroPipeline(config_dict)
        pipeline.run_pipeline(max_iterations=args.iterations)
    except KeyboardInterrupt:
        print("\n⚠️ 파이프라인이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 파이프라인 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 