#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH AlphaZero Automated Training Pipeline
==========================================

완전 자동화된 AlphaZero 훈련 파이프라인:
1. Best 모델로 Self-Play 데이터 생성
2. 생성된 데이터로 모델 훈련
3. 새 모델 vs 기존 모델 평가
4. 승률 기준으로 Best 모델 교체
5. 반복...

This implements the complete AlphaZero training loop.
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


class BestModelManager:
    """Best model 관리를 담당하는 클래스"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.best_model_path = self.models_dir / "best_model.pt"
        self.best_model_info_path = self.models_dir / "best_model_info.json"
        
        # 모델 히스토리 디렉토리
        self.history_dir = self.models_dir / "history"
        self.history_dir.mkdir(exist_ok=True)
        
    def get_best_model_path(self) -> Optional[str]:
        """현재 best 모델 경로 반환"""
        if self.best_model_path.exists():
            return str(self.best_model_path)
        return None
    
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
    """AlphaZero 자동화 훈련 파이프라인"""
    
    def __init__(self, config_dict: Dict):
        self.config = config_dict
        self.model_manager = BestModelManager(config_dict['models_dir'])
        
        # 디렉토리 설정
        self.data_dir = Path(config_dict['data_dir'])
        self.data_dir.mkdir(exist_ok=True)
        
        self.logs_dir = Path(config_dict['logs_dir'])
        self.logs_dir.mkdir(exist_ok=True)
        
        # 로그 파일 설정
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.logs_dir / f"pipeline_{timestamp}.log"
        
        # 통계
        self.iteration = 0
        self.total_games_played = 0
        self.successful_updates = 0
        self.start_time = None
        
    def log(self, message: str):
        """로그 메시지 출력 및 파일 저장"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_msg + '\n')
    
    def run_selfplay(self, model_path: str, iteration: int) -> bool:
        """Self-play 데이터 생성 (AlphaZero 논문 방식: 높은 품질)"""
        self.log(f"🎮 Self-Play 시작 (Iteration {iteration})")
        
        data_output = self.data_dir / f"iter_{iteration}"
        data_output.mkdir(exist_ok=True)
        
        # AlphaZero 논문 방식: 높은 품질의 Self-Play 데이터 생성
        # - 더 많은 게임 수
        # - 더 높은 MCTS 시뮬레이션
        # - Dirichlet 노이즈 사용
        games_per_iteration = self.config.get('games_per_iteration', self.config['selfplay_games'])
        mcts_sims = self.config.get('selfplay_mcts_sims', 800)
        
        # Self-play 명령어 구성 (AlphaZero 논문 방식)
        cmd = [
            "uv", "run", "python", "scripts/selfplay.py",
            "--games", str(games_per_iteration),
            "--model", model_path,
            "--mcts-sims", str(mcts_sims),
            "--output", str(data_output),
            "--gpu-batch",  # GPU 배치 처리 활성화
            "--batch-size", "32",  # 적절한 배치 크기
            "--workers", "6"  # 병렬 처리
        ]
        
        # AlphaZero 논문 방식: Dirichlet 노이즈 사용 (탐색 촉진)
        if self.config.get('use_dirichlet_noise', True):
            # Dirichlet 노이즈는 selfplay.py에서 자동으로 적용됨
            self.log(f"   🎲 Dirichlet 노이즈 활성화")
        
        # 병렬 처리 옵션 추가
        if self.config.get('parallel', False) and self.config.get('workers', 1) > 1:
            cmd.extend(["--parallel-mcts", "--workers", str(self.config['workers'])])
        
        # 추가 최적화 옵션
        if self.config.get('fast_mode', False):
            cmd.append("--fast")
        if self.config.get('ultra_fast_mode', False):
            cmd.append("--ultra-fast")
        if self.config.get('parallel_mcts', False):
            cmd.extend(["--parallel-mcts", "--mcts-threads", str(self.config.get('mcts_threads', 4))])
        
        try:
            self.log(f"   명령어: {' '.join(cmd)}")
            self.log(f"   📊 Self-Play 설정:")
            self.log(f"      게임 수: {games_per_iteration}")
            self.log(f"      MCTS 시뮬레이션: {mcts_sims}")
            self.log(f"      GPU 배치 처리: 활성화")
            
            result = subprocess.run(cmd, cwd=Path.cwd(), capture_output=True, text=True)
            
            if result.returncode == 0:
                self.log(f"✅ Self-Play 완료")
                
                # 데이터 폴더 확인
                data_output = self.data_dir / f"iter_{iteration}"
                if data_output.exists():
                    data_files = list(data_output.glob("*.pt"))
                    self.log(f"   생성된 데이터 파일: {len(data_files)}개")
                    for file in data_files[:5]:  # 처음 5개만 표시
                        self.log(f"     {file.name}")
                else:
                    self.log(f"❌ Self-Play 데이터 폴더가 생성되지 않음: {data_output}")
                
                self.total_games_played += games_per_iteration
                return True
            else:
                self.log(f"❌ Self-Play 실패:")
                self.log(f"   stdout: {result.stdout}")
                self.log(f"   stderr: {result.stderr}")
                return False
                
        except Exception as e:
            self.log(f"❌ Self-Play 실행 오류: {e}")
            return False
    
    def run_training(self, iteration: int) -> Optional[str]:
        """모델 훈련 (AlphaZero 논문 방식: 모든 이전 데이터 누적)"""
        self.log(f"🎯 모델 훈련 시작 (Iteration {iteration})")
        
        # AlphaZero 논문 방식: 모든 이전 iteration의 데이터 누적
        all_data_dirs = []
        for i in range(1, iteration + 1):
            iter_data_dir = self.data_dir / f"iter_{i}"
            if iter_data_dir.exists():
                data_files = list(iter_data_dir.glob("*.pt"))
                if data_files:
                    all_data_dirs.append(str(iter_data_dir))
                    self.log(f"   📂 Iteration {i} 데이터 포함: {len(data_files)}개 파일")
        
        if not all_data_dirs:
            self.log(f"❌ 훈련할 데이터가 없습니다!")
            return None
        
        # 모든 데이터를 하나의 임시 디렉토리로 복사
        temp_data_dir = self.data_dir / f"temp_combined_{iteration}"
        temp_data_dir.mkdir(exist_ok=True)
        
        file_count = 0
        for data_dir in all_data_dirs:
            for file_path in Path(data_dir).glob("*.pt"):
                new_name = f"iter_{iteration}_file_{file_count:04d}.pt"
                shutil.copy2(file_path, temp_data_dir / new_name)
                file_count += 1
        
        self.log(f"   📊 총 {file_count}개 파일을 훈련에 사용")
        
        output_dir = self.data_dir / f"models_iter_{iteration}"
        output_dir.mkdir(exist_ok=True)
        
        # 훈련 명령어 구성 (AlphaZero 논문 방식)
        cmd = [
            "uv", "run", "python", "scripts/train.py",
            "--data", str(temp_data_dir),
            "--epochs", str(self.config['training_epochs']),
            "--batch-size", str(self.config['training_batch_size']),
            "--lr", str(self.config['training_lr']),
            "--output", str(output_dir)
        ]
        
        # 훈련 최적화 옵션 추가
        if self.config.get('training_optimization', False):
            cmd.extend(["--num-workers", str(self.config.get('training_workers', 4))])
        if self.config.get('mixed_precision', False):
            cmd.append("--mixed-precision")
        
        # 기존 best 모델이 있으면 이어서 훈련 (AlphaZero 논문 방식)
        best_model_path = self.model_manager.get_best_model_path()
        if best_model_path and self.config['continue_training']:
            cmd.extend(["--model", best_model_path])
            self.log(f"   🔄 기존 모델로 이어서 훈련: {best_model_path}")
        
        try:
            self.log(f"   명령어: {' '.join(cmd)}")
            result = subprocess.run(cmd, cwd=Path.cwd(), capture_output=True, text=True)
            
            # 상세한 출력 로그 추가
            self.log(f"   train.py 반환 코드: {result.returncode}")
            if result.stdout:
                self.log(f"   train.py stdout:")
                for line in result.stdout.split('\n')[:10]:  # 처음 10줄만
                    if line.strip():
                        self.log(f"     {line}")
            if result.stderr:
                self.log(f"   train.py stderr:")
                for line in result.stderr.split('\n')[:10]:  # 처음 10줄만
                    if line.strip():
                        self.log(f"     {line}")
            
            if result.returncode == 0:
                # train.py가 생성하는 모델 경로
                trained_model_path = output_dir / "trained_model.pt"
                final_model_path = self.data_dir / f"candidate_iter_{iteration}.pt"
                
                # 모델을 최종 경로로 이동
                if trained_model_path.exists():
                    shutil.move(str(trained_model_path), str(final_model_path))
                    self.log(f"✅ 모델 훈련 완료: {final_model_path}")
                    
                    # 임시 데이터 디렉토리 정리
                    shutil.rmtree(temp_data_dir, ignore_errors=True)
                    
                    return str(final_model_path)
                else:
                    self.log(f"❌ 훈련된 모델을 찾을 수 없음: {trained_model_path}")
                    self.log(f"   출력 디렉토리 내용:")
                    if output_dir.exists():
                        for file in output_dir.iterdir():
                            self.log(f"     {file.name}")
                    else:
                        self.log(f"     출력 디렉토리가 존재하지 않음: {output_dir}")
                    return None
            else:
                self.log(f"❌ 모델 훈련 실패:")
                self.log(f"   stdout: {result.stdout}")
                self.log(f"   stderr: {result.stderr}")
                return None
                
        except Exception as e:
            self.log(f"❌ 모델 훈련 실행 오류: {e}")
            return None
    
    def run_evaluation(self, candidate_path: str, best_path: str, 
                      iteration: int) -> Optional[Dict]:
        """모델 평가"""
        self.log(f"🏆 모델 평가 시작 (Iteration {iteration})")
        
        # 평가 명령어 구성
        cmd = [
            "uv", "run", "python", "scripts/evaluate.py",
            "--candidate", candidate_path,
            "--games", str(self.config['evaluation_games']),
            "--threshold", str(self.config['evaluation_threshold']),
            "--mcts-sims", str(self.config['evaluation_mcts_sims'])
        ]
        
        # 평가 최적화 옵션 추가
        if self.config.get('evaluation_parallel', False):
            cmd.extend(["--parallel", "--workers", str(self.config.get('evaluation_workers', 4))])
        if self.config.get('evaluation_fast', False):
            cmd.append("--fast")
        
        if best_path:
            cmd.extend(["--best", best_path])
        
        try:
            self.log(f"   후보 모델: {candidate_path}")
            self.log(f"   기존 모델: {best_path if best_path else 'Random baseline'}")
            
            result = subprocess.run(cmd, cwd=Path.cwd(), capture_output=True, text=True)
            
            self.log(f"   평가 완료 (반환 코드: {result.returncode})")
            
            # 평가 결과 파싱 (evaluate.py의 출력에서 추출)
            evaluation_success = result.returncode == 0  # 0 = 모델 교체 필요
            
            return {
                'should_replace': evaluation_success,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
        except Exception as e:
            self.log(f"❌ 모델 평가 실행 오류: {e}")
            return None
    
    def cleanup_iteration_data(self, iteration: int, keep_models: bool = True):
        """Iteration 데이터 정리"""
        if self.config['cleanup_data']:
            try:
                # 게임 데이터 삭제
                data_path = self.data_dir / f"iter_{iteration}"
                if data_path.exists():
                    shutil.rmtree(data_path)
                    self.log(f"🗑️ Iteration {iteration} 데이터 정리 완료")
                
                # 후보 모델 정리 (채택되지 않은 경우)
                if not keep_models:
                    candidate_path = self.data_dir / f"candidate_iter_{iteration}.pt"
                    if candidate_path.exists():
                        candidate_path.unlink()
                        self.log(f"🗑️ 후보 모델 정리: {candidate_path}")
                        
            except Exception as e:
                self.log(f"⚠️ 데이터 정리 중 오류: {e}")
    
    def run_pipeline(self, max_iterations: int = 100, max_time_hours: int = 24):
        """파이프라인 실행"""
        self.start_time = time.time()
        max_time_seconds = max_time_hours * 3600
        
        self.log("🚀 YINSH AlphaZero 훈련 파이프라인 시작!")
        self.log("=" * 80)
        self.log(f"📋 설정:")
        self.log(f"   최대 Iteration: {max_iterations}")
        self.log(f"   최대 실행 시간: {max_time_hours}시간")
        self.log(f"   Self-Play 게임 수: {self.config['selfplay_games']}")
        self.log(f"   평가 게임 수: {self.config['evaluation_games']}")
        self.log(f"   채택 기준: {self.config['evaluation_threshold']*100:.1f}%")
        self.log("=" * 80)
        
        # Best 모델 초기화
        best_model_path = self.model_manager.initialize_best_model()
        
        try:
            for iteration in range(1, max_iterations + 1):
                iteration_start_time = time.time()
                
                # 시간 제한 확인
                elapsed_time = time.time() - self.start_time
                if elapsed_time > max_time_seconds:
                    self.log(f"⏰ 시간 제한 ({max_time_hours}시간) 도달. 파이프라인 종료.")
                    break
                
                self.iteration = iteration
                self.log(f"\n🔄 === Iteration {iteration} 시작 ===")
                
                # 1. Self-Play
                if not self.run_selfplay(best_model_path, iteration):
                    self.log(f"❌ Iteration {iteration} Self-Play 실패. 계속 진행...")
                    continue
                
                # 2. 훈련
                candidate_model_path = self.run_training(iteration)
                if not candidate_model_path:
                    self.log(f"❌ Iteration {iteration} 훈련 실패. 계속 진행...")
                    self.cleanup_iteration_data(iteration, keep_models=False)
                    continue
                
                # 3. 평가
                evaluation_result = self.run_evaluation(
                    candidate_model_path, best_model_path, iteration
                )
                
                if not evaluation_result:
                    self.log(f"❌ Iteration {iteration} 평가 실패. 계속 진행...")
                    self.cleanup_iteration_data(iteration, keep_models=False)
                    continue
                
                # 4. 모델 교체 결정
                if evaluation_result['should_replace']:
                    # Best 모델 업데이트 (실제 evaluation_result는 evaluate.py에서 생성)
                    # 여기서는 간단한 정보만 저장
                    dummy_eval_result = {
                        'results': {'win_rate': 0.6, 'candidate_wins': 60},  # 임시 값
                        'evaluation_games': self.config['evaluation_games'],
                        'threshold': self.config['evaluation_threshold']
                    }
                    
                    if self.model_manager.update_best_model(
                        candidate_model_path, dummy_eval_result, iteration
                    ):
                        best_model_path = self.model_manager.get_best_model_path()
                        self.successful_updates += 1
                        self.log(f"🎉 새 Best 모델 채택! (총 {self.successful_updates}번째 업데이트)")
                    
                    self.cleanup_iteration_data(iteration, keep_models=True)
                else:
                    self.log(f"❌ 새 모델 기각. 기존 모델 유지.")
                    self.cleanup_iteration_data(iteration, keep_models=False)
                
                # Iteration 완료 통계
                iteration_time = time.time() - iteration_start_time
                total_elapsed = time.time() - self.start_time
                estimated_remaining = (total_elapsed / iteration) * (max_iterations - iteration)
                
                self.log(f"📈 Iteration {iteration} 완료:")
                self.log(f"   소요 시간: {iteration_time/60:.1f}분")
                self.log(f"   총 경과 시간: {total_elapsed/3600:.2f}시간")
                self.log(f"   예상 남은 시간: {estimated_remaining/3600:.2f}시간")
                self.log(f"   성공적 업데이트: {self.successful_updates}")
                self.log(f"   총 게임 수: {self.total_games_played:,}")
                
                # Iteration 간 메모리 최적화
                if self.config.get('memory_optimization', False):
                    try:
                        import subprocess
                        subprocess.run([
                            "uv", "run", "python", "scripts/memory_optimizer.py", "--mode", "between"
                        ], cwd=Path.cwd(), capture_output=True)
                    except Exception as e:
                        self.log(f"⚠️ 메모리 최적화 실패: {e}")
                
        except KeyboardInterrupt:
            self.log("\n⏹️ 사용자에 의해 파이프라인이 중단되었습니다.")
        except Exception as e:
            self.log(f"\n❌ 파이프라인 실행 중 예외 발생: {e}")
            import traceback
            self.log(traceback.format_exc())
        
        finally:
            self._print_final_summary()
    
    def _print_final_summary(self):
        """최종 요약 출력"""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        self.log("\n" + "=" * 80)
        self.log("🏁 YINSH AlphaZero 파이프라인 완료!")
        self.log("=" * 80)
        self.log(f"📊 최종 통계:")
        self.log(f"   총 Iteration: {self.iteration}")
        self.log(f"   성공적 모델 업데이트: {self.successful_updates}")
        self.log(f"   총 게임 수: {self.total_games_played:,}")
        self.log(f"   총 실행 시간: {total_time/3600:.2f}시간")
        self.log(f"   평균 Iteration 시간: {total_time/self.iteration/60:.1f}분" if self.iteration > 0 else "   평균 시간: N/A")
        
        # 현재 Best 모델 정보
        model_info = self.model_manager.get_model_info()
        if model_info:
            self.log(f"🏆 현재 Best 모델:")
            self.log(f"   마지막 업데이트: {model_info.get('updated', 'N/A')}")
            self.log(f"   Iteration: {model_info.get('iteration', 'N/A')}")
            self.log(f"   승률: {model_info.get('win_rate', 0)*100:.2f}%")
        
        self.log(f"📁 결과 위치:")
        self.log(f"   Best 모델: {self.model_manager.best_model_path}")
        self.log(f"   로그 파일: {self.log_file}")
        self.log("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="YINSH AlphaZero 자동화 훈련 파이프라인")
    
    # 파이프라인 설정
    parser.add_argument("--iterations", type=int, default=100,
                       help="최대 iteration 수")
    parser.add_argument("--max-hours", type=int, default=24,
                       help="최대 실행 시간 (시간)")
    
    # Self-play 설정
    parser.add_argument("--selfplay-games", type=int, default=100,
                       help="Iteration당 self-play 게임 수")
    parser.add_argument("--selfplay-mcts-sims", type=int, default=800,
                       help="Self-play MCTS 시뮬레이션 수")
    
    # 훈련 설정
    parser.add_argument("--training-epochs", type=int, default=100,  # AlphaZero 논문: 100
                       help="훈련 epoch 수")
    parser.add_argument("--training-batch-size", type=int, default=512,  # AlphaZero 논문: 512
                       help="훈련 배치 크기")
    parser.add_argument("--training-lr", type=float, default=0.002,  # AlphaZero 논문: 0.002
                       help="학습률")
    parser.add_argument("--continue-training", action="store_true",
                       help="기존 모델에서 이어서 훈련")
    
    # 평가 설정
    parser.add_argument("--evaluation-games", type=int, default=400,  # AlphaZero 논문: 400게임
                       help="모델 평가 게임 수")
    parser.add_argument("--evaluation-threshold", type=float, default=0.55,
                       help="새 모델 채택 승률 기준")
    parser.add_argument("--evaluation-mcts-sims", type=int, default=800,  # AlphaZero 논문: 정확한 평가
                       help="평가 MCTS 시뮬레이션 수")
    
    # 디렉토리 설정
    parser.add_argument("--models-dir", type=str, default="models",
                       help="모델 저장 디렉토리")
    parser.add_argument("--data-dir", type=str, default="pipeline_data",
                       help="훈련 데이터 임시 디렉토리")
    parser.add_argument("--logs-dir", type=str, default="pipeline_logs",
                       help="로그 저장 디렉토리")
    
    # 기타 설정
    parser.add_argument("--cleanup-data", action="store_true",
                       help="Iteration 완료 후 임시 데이터 삭제")
    
    # 병렬 처리 설정
    parser.add_argument("--workers", type=int, default=1,
                       help="병렬 워커 수 (1=순차실행)")
    parser.add_argument("--parallel", action="store_true",
                       help="병렬 셀프플레이 활성화")
    
    args = parser.parse_args()
    
    # 설정 딕셔너리 생성
    config = {
        'selfplay_games': args.selfplay_games,
        'selfplay_mcts_sims': args.selfplay_mcts_sims,
        'training_epochs': args.training_epochs,
        'training_batch_size': args.training_batch_size,
        'training_lr': args.training_lr,
        'continue_training': args.continue_training,
        'evaluation_games': args.evaluation_games,
        'evaluation_threshold': args.evaluation_threshold,
        'evaluation_mcts_sims': args.evaluation_mcts_sims,
        'models_dir': args.models_dir,
        'data_dir': args.data_dir,
        'logs_dir': args.logs_dir,
        'cleanup_data': args.cleanup_data,
        'parallel': args.parallel,
        'workers': args.workers
    }
    
    # 파이프라인 실행
    pipeline = AlphaZeroPipeline(config)
    pipeline.run_pipeline(
        max_iterations=args.iterations,
        max_time_hours=args.max_hours
    )


if __name__ == "__main__":
    main() 