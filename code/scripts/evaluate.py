#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH AlphaZero Model Evaluation Script
======================================

두 모델 간의 성능을 비교하여 새 모델이 기존 모델보다 
우수한지 평가하는 스크립트입니다.

AlphaZero 방식: 승률 55% 이상이면 새 모델을 채택합니다.
"""

import os
import sys
import argparse
import torch
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.cuda.amp import autocast

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yinsh import YinshEnv, YinshAgent, Color, config
from yinsh.game import YinshGame
from yinsh.mapper import YinshActionMapper

# selfplay.py의 정책 추출 함수 import (경로 수정)
try:
    from scripts.selfplay import extract_mcts_policy_distribution
except ImportError:
    # 상대 경로로 시도
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from selfplay import extract_mcts_policy_distribution


def _evaluate_game_batch(candidate_model_path, best_model_path, start_game, end_game, 
                        mcts_sims, device, worker_id):
    """병렬 처리를 위한 게임 배치 실행 함수"""
    import torch
    from yinsh import YinshEnv, YinshAgent, Color, config
    from yinsh.mapper import YinshActionMapper
    import time
    
    # 워커별 GPU 메모리 관리
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(0)  # 첫 번째 GPU 사용
        torch.cuda.empty_cache()
    
    print(f"🔨 Worker {worker_id}: 게임 {start_game+1}-{end_game} 시작")
    
    try:
        # 에이전트 초기화
        candidate_agent = YinshAgent(
            model_path=candidate_model_path,
            use_mcts=True,
            device=device
        )
        
        best_agent = YinshAgent(
            model_path=best_model_path,
            use_mcts=True,
            device=device
        )
        
        # MCTS 시뮬레이션 수 조정
        if hasattr(candidate_agent, 'mcts_agent') and candidate_agent.mcts_agent:
            candidate_agent.mcts_agent.mcts.num_simulations = mcts_sims
        if hasattr(best_agent, 'mcts_agent') and best_agent.mcts_agent:
            best_agent.mcts_agent.mcts.num_simulations = mcts_sims
        
        batch_results = []
        
        for game_id in range(start_game, end_game):
            game_start_time = time.time()
            
            # 흰색/검은색 번갈아가며 플레이
            if game_id % 2 == 0:
                white_agent = candidate_agent
                black_agent = best_agent
                candidate_is_white = True
            else:
                white_agent = best_agent
                black_agent = candidate_agent
                candidate_is_white = False
            
            try:
                # 게임 실행
                game_history, winner, turns = _play_single_game(
                    white_agent, black_agent, game_id, mcts_sims
                )
                
                # 결과 기록
                game_result = {
                    'game_id': game_id + 1,
                    'candidate_is_white': candidate_is_white,
                    'winner': 'WHITE' if winner == Color.WHITE else 'BLACK' if winner == Color.BLACK else 'DRAW',
                    'candidate_won': (winner == Color.WHITE and candidate_is_white) or 
                                   (winner == Color.BLACK and not candidate_is_white),
                    'game_time': time.time() - game_start_time,
                    'turns': turns,
                    'worker_id': worker_id
                }
                batch_results.append(game_result)
                
                # 에이전트 상태 리셋
                white_agent.reset_game_stats()
                black_agent.reset_game_stats()
                
            except Exception as e:
                print(f"❌ Worker {worker_id} 게임 {game_id + 1} 실패: {e}")
                continue
        
        print(f"✅ Worker {worker_id}: {len(batch_results)} 게임 완료")
        return batch_results
        
    except Exception as e:
        print(f"❌ Worker {worker_id} 초기화 실패: {e}")
        return []
    finally:
        # GPU 메모리 정리
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _play_single_game(agent1, agent2, game_id, mcts_sims):
    """단일 게임 실행 (병렬 처리용)"""
    env = YinshEnv()
    game_history = []
    turn_count = 0
    max_turns = None  # 자연 종료까지
    
    while not env.is_game_over() and (max_turns is None or turn_count < max_turns):
        current_player = agent1 if env.current_player == Color.WHITE else agent2
        
        try:
            # 현재 상태 저장
            state = env.get_state_tensor()
            
            # 액션 선택 (평가용 온도 사용)
            action, action_info = current_player.select_action(
                env, 
                temperature=config.EVALUATION_TEMPERATURE,  # 0.1
                add_noise=False  # 평가시에는 노이즈 없음
            )
            
            # 액션 실행
            env.step(action)
            
            # 게임 히스토리에 추가
            game_history.append({
                "state": state, 
                "action": action, 
                "player": env.current_player,
                "action_info": action_info
            })
            
            turn_count += 1
            
        except Exception as e:
            print(f"❌ 게임 {game_id + 1} 턴 {turn_count + 1} 실패: {e}")
            return None, None, turn_count
    
    # 게임 결과
    try:
        winner = env.get_winner()
        return game_history, winner, turn_count
    except Exception as e:
        print(f"❌ 게임 {game_id + 1} 결과 확인 실패: {e}")
        return None, None, turn_count


class ModelEvaluator:
    """모델 평가를 담당하는 클래스 (GPU 최적화)"""
    
    def __init__(self, evaluation_games: int = 100, mcts_sims: int = None, 
                 use_parallel: bool = True, num_workers: int = None, use_gpu: bool = True):
        """
        Args:
            evaluation_games: 평가에 사용할 게임 수
            mcts_sims: MCTS 시뮬레이션 수 (None이면 config에서 가져옴)
            use_parallel: 병렬 처리 사용 여부
            num_workers: 병렬 워커 수 (None이면 CPU 코어 수의 절반)
            use_gpu: GPU 사용 여부
        """
        self.evaluation_games = evaluation_games
        self.mcts_sims = mcts_sims if mcts_sims is not None else config.EVALUATION_MCTS_SIMULATIONS
        self.use_parallel = use_parallel and evaluation_games >= 4  # 최소 4게임부터 병렬화
        self.num_workers = num_workers if num_workers else max(1, cpu_count() // 2)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = 'cuda' if self.use_gpu else 'cpu'
        self.results = []
        
        print(f"🚀 ModelEvaluator 초기화:")
        print(f"   평가 게임: {self.evaluation_games}")
        print(f"   MCTS 시뮬레이션: {self.mcts_sims}")
        print(f"   병렬 처리: {self.use_parallel} ({self.num_workers} workers)")
        print(f"   GPU 사용: {self.use_gpu} ({self.device})")
        
    def evaluate_models(self, candidate_model_path: str, best_model_path: str = None,
                       threshold: float = 0.55, save_results: bool = True) -> dict:
        """
        두 모델을 평가하여 새 모델 채택 여부 결정 (최적화된 버전)
        
        Args:
            candidate_model_path: 새로운 후보 모델 경로
            best_model_path: 기존 best 모델 경로 (None이면 랜덤 초기화)
            threshold: 새 모델 채택을 위한 최소 승률 (기본 55%)
            save_results: 결과 저장 여부
            
        Returns:
            평가 결과 딕셔너리
        """
        print(f"\n🏆 모델 평가 시작 (최적화된 버전)")
        print(f"📋 설정:")
        print(f"   후보 모델: {candidate_model_path}")
        print(f"   기존 모델: {best_model_path if best_model_path else 'Random baseline'}")
        print(f"   평가 게임 수: {self.evaluation_games}")
        print(f"   MCTS 시뮬레이션: {self.mcts_sims}")
        print(f"   채택 기준: {threshold*100:.1f}% 이상 승률")
        print("=" * 60)
        
        # 병렬 처리 vs 순차 처리 결정
        if self.use_parallel:
            return self._evaluate_models_parallel(
                candidate_model_path, best_model_path, threshold, save_results
            )
        else:
            return self._evaluate_models_sequential(
                candidate_model_path, best_model_path, threshold, save_results
            )
    
    def _evaluate_models_sequential(self, candidate_model_path: str, best_model_path: str = None,
                                   threshold: float = 0.55, save_results: bool = True) -> dict:
        """순차 처리로 모델 평가 (기존 방식 개선)"""
        
        # 에이전트 생성
        try:
            print("🤖 에이전트 초기화 중...")
            candidate_agent = YinshAgent(
                model_path=candidate_model_path,
                use_mcts=True,
                device=self.device
            )
            print("   ✅ 후보 모델 로드 완료")
            
            best_agent = YinshAgent(
                model_path=best_model_path,
                use_mcts=True,
                device=self.device
            )
            print("   ✅ 기존 모델 로드 완료")
            
            # MCTS 시뮬레이션 수 조정
            if hasattr(candidate_agent, 'mcts_agent') and candidate_agent.mcts_agent:
                candidate_agent.mcts_agent.mcts.num_simulations = self.mcts_sims
            if hasattr(best_agent, 'mcts_agent') and best_agent.mcts_agent:
                best_agent.mcts_agent.mcts.num_simulations = self.mcts_sims
                
        except Exception as e:
            print(f"❌ 에이전트 초기화 실패: {e}")
            return None
        
        # 평가 게임 진행 (최적화된 버전)
        candidate_as_white_wins = 0
        candidate_as_black_wins = 0
        draws = 0
        total_game_time = 0
        
        print(f"\n🎮 {self.evaluation_games}게임 평가 시작 (최적화된 버전)...")
        
        for game_id in tqdm(range(self.evaluation_games), desc="평가 진행"):
            game_start_time = time.time()
            
            # 흰색/검은색 번갈아가며 플레이 (공정한 평가)
            if game_id % 2 == 0:
                # 후보 모델이 흰색
                white_agent = candidate_agent
                black_agent = best_agent
                candidate_is_white = True
            else:
                # 후보 모델이 검은색
                white_agent = best_agent
                black_agent = candidate_agent
                candidate_is_white = False
            
            try:
                # 최적화된 게임 진행 (selfplay와 동일한 방식)
                game_history, winner, turns = self._play_game_optimized(
                    white_agent, black_agent, game_id=game_id
                )
                
                # 정책 추출 (selfplay와 동일한 방식)
                if game_history:
                    action_mapper = YinshActionMapper()
                    for move in game_history:
                        action = move["action"]
                        action_info = move["action_info"]
                        
                        # selfplay와 동일한 정책 추출 사용
                        policy = extract_mcts_policy_distribution(action_info, action_mapper, action)
                        if policy is not None:
                            # 정책 추출 성공 (MCTS 사용)
                            pass
                        else:
                            # 정책 추출 실패 (Direct NN 사용)
                            pass
                
                # 승자 결정
                winner_value = None
                if winner == Color.WHITE:
                    winner_value = 1
                elif winner == Color.BLACK:
                    winner_value = -1
                else:
                    winner_value = 0  # 무승부
                
                # 결과 기록
                if winner_value == 0:
                    draws += 1
                elif (winner_value > 0 and candidate_is_white) or \
                     (winner_value < 0 and not candidate_is_white):
                    if candidate_is_white:
                        candidate_as_white_wins += 1
                    else:
                        candidate_as_black_wins += 1
                
                game_time = time.time() - game_start_time
                total_game_time += game_time
                
                # 게임 결과 저장
                game_result = {
                    'game_id': game_id + 1,
                    'candidate_is_white': candidate_is_white,
                    'winner': 'WHITE' if winner == Color.WHITE else 'BLACK' if winner == Color.BLACK else 'DRAW',
                    'candidate_won': (winner == Color.WHITE and candidate_is_white) or 
                                   (winner == Color.BLACK and not candidate_is_white),
                    'game_time': game_time,
                    'turns': turns
                }
                self.results.append(game_result)
                
                # 에이전트 상태 리셋
                white_agent.reset_game_stats()
                black_agent.reset_game_stats()
                
            except Exception as e:
                print(f"❌ 게임 {game_id + 1} 실패: {e}")
                continue
        
        # 결과 계산
        total_candidate_wins = candidate_as_white_wins + candidate_as_black_wins
        total_games_played = len(self.results)
        win_rate = total_candidate_wins / total_games_played if total_games_played > 0 else 0.0
        
        # 평가 결과
        evaluation_result = {
            'timestamp': datetime.now().isoformat(),
            'candidate_model': candidate_model_path,
            'best_model': best_model_path,
            'evaluation_games': total_games_played,
            'threshold': threshold,
            'results': {
                'candidate_wins': total_candidate_wins,
                'candidate_as_white_wins': candidate_as_white_wins,
                'candidate_as_black_wins': candidate_as_black_wins,
                'draws': draws,
                'win_rate': win_rate,
                'should_replace': win_rate >= threshold
            },
            'performance': {
                'total_time': total_game_time,
                'avg_game_time': total_game_time / total_games_played if total_games_played > 0 else 0,
                'mcts_simulations': self.mcts_sims
            },
            'game_details': self.results
        }
        
        # 결과 출력
        self._print_evaluation_results(evaluation_result)
        
        # 결과 저장
        if save_results:
            self._save_evaluation_results(evaluation_result)
        
        return evaluation_result
    
    def _evaluate_models_parallel(self, candidate_model_path: str, best_model_path: str = None,
                                 threshold: float = 0.55, save_results: bool = True) -> dict:
        """병렬 처리로 모델 평가 (최적화된 성능)"""
        
        print(f"⚡ 병렬 평가 시작: {self.num_workers} workers")
        
        # 게임을 워커들에게 분배
        games_per_worker = self.evaluation_games // self.num_workers
        remaining_games = self.evaluation_games % self.num_workers
        
        game_batches = []
        start_game = 0
        for i in range(self.num_workers):
            batch_size = games_per_worker + (1 if i < remaining_games else 0)
            if batch_size > 0:
                game_batches.append((start_game, start_game + batch_size))
                start_game += batch_size
        
        print(f"📊 게임 분배: {len(game_batches)} batches")
        for i, (start, end) in enumerate(game_batches):
            print(f"   Worker {i+1}: 게임 {start+1}-{end}")
        
        # 병렬 실행
        all_results = []
        total_start_time = time.time()
        
        try:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                # 각 워커에게 작업 할당
                futures = []
                for i, (start_game, end_game) in enumerate(game_batches):
                    future = executor.submit(
                        _evaluate_game_batch,
                        candidate_model_path,
                        best_model_path,
                        start_game,
                        end_game,
                        self.mcts_sims,
                        self.device,
                        i
                    )
                    futures.append(future)
                
                # 결과 수집
                for future in tqdm(as_completed(futures), total=len(futures), desc="병렬 평가"):
                    try:
                        batch_results = future.result()
                        all_results.extend(batch_results)
                    except Exception as e:
                        print(f"❌ 배치 처리 실패: {e}")
                        continue
                        
        except Exception as e:
            print(f"❌ 병렬 처리 초기화 실패: {e}")
            print("🔄 순차 처리로 전환합니다...")
            return self._evaluate_models_sequential(
                candidate_model_path, best_model_path, threshold, save_results
            )
        
        total_time = time.time() - total_start_time
        
        # 결과 집계
        candidate_as_white_wins = 0
        candidate_as_black_wins = 0
        draws = 0
        
        for result in all_results:
            if result['winner'] == 'DRAW':
                draws += 1
            elif result['candidate_won']:
                if result['candidate_is_white']:
                    candidate_as_white_wins += 1
                else:
                    candidate_as_black_wins += 1
        
        total_candidate_wins = candidate_as_white_wins + candidate_as_black_wins
        total_games_played = len(all_results)
        win_rate = total_candidate_wins / total_games_played if total_games_played > 0 else 0.0
        
        # 평가 결과
        evaluation_result = {
            'timestamp': datetime.now().isoformat(),
            'candidate_model': candidate_model_path,
            'best_model': best_model_path,
            'evaluation_games': total_games_played,
            'threshold': threshold,
            'results': {
                'candidate_wins': total_candidate_wins,
                'candidate_as_white_wins': candidate_as_white_wins,
                'candidate_as_black_wins': candidate_as_black_wins,
                'draws': draws,
                'win_rate': win_rate,
                'should_replace': win_rate >= threshold
            },
            'performance': {
                'total_time': total_time,
                'avg_game_time': total_time / total_games_played if total_games_played > 0 else 0,
                'mcts_simulations': self.mcts_sims,
                'parallel_workers': self.num_workers
            },
            'game_details': all_results
        }
        
        # 결과 출력
        self._print_evaluation_results(evaluation_result)
        
        # 결과 저장
        if save_results:
            self._save_evaluation_results(evaluation_result)
        
        return evaluation_result
    
    def _play_game_optimized(self, agent1, agent2, game_id=0):
        """최적화된 게임 실행 함수 (selfplay와 동일한 방식)"""
        env = YinshEnv()
        game_history = []
        turn_count = 0
        max_turns = None  # 자연 종료까지
        
        while not env.is_game_over() and (max_turns is None or turn_count < max_turns):
            current_player = agent1 if env.current_player == Color.WHITE else agent2
            
            try:
                # 현재 상태 저장 (selfplay와 동일)
                state = env.get_state_tensor()
                
                # 액션 선택 (평가용 온도 사용)
                action, action_info = current_player.select_action(
                    env, 
                    temperature=config.EVALUATION_TEMPERATURE,  # 0.1
                    add_noise=False  # 평가시에는 노이즈 없음
                )
                
                # 액션 실행
                env.step(action)
                
                # 게임 히스토리에 추가 (selfplay와 동일한 방식)
                game_history.append({
                    "state": state, 
                    "action": action, 
                    "player": env.current_player,
                    "action_info": action_info
                })
                
                turn_count += 1
                
            except Exception as e:
                print(f"❌ 게임 {game_id + 1} 턴 {turn_count + 1} 실패: {e}")
                return None, None, turn_count
        
        # 게임 결과
        try:
            winner = env.get_winner()
            return game_history, winner, turn_count
        except Exception as e:
            print(f"❌ 게임 {game_id + 1} 결과 확인 실패: {e}")
            return None, None, turn_count
    
    def _print_evaluation_results(self, result: dict):
        """평가 결과를 콘솔에 출력"""
        print(f"\n🏆 평가 결과")
        print("=" * 60)
        print(f"📊 게임 통계:")
        print(f"   총 게임 수: {result['evaluation_games']}")
        print(f"   후보 모델 승리: {result['results']['candidate_wins']}")
        print(f"   ├── 흰색으로 승리: {result['results']['candidate_as_white_wins']}")
        print(f"   └── 검은색으로 승리: {result['results']['candidate_as_black_wins']}")
        print(f"   무승부: {result['results']['draws']}")
        print(f"   후보 모델 승률: {result['results']['win_rate']*100:.2f}%")
        print(f"   채택 기준: {result['threshold']*100:.1f}%")
        
        print(f"\n⏱️ 성능 통계:")
        print(f"   총 소요시간: {result['performance']['total_time']:.1f}초")
        print(f"   게임당 평균시간: {result['performance']['avg_game_time']:.2f}초")
        
        # 최종 결정
        if result['results']['should_replace']:
            print(f"\n✅ 🎉 새 모델 채택!")
            print(f"   승률 {result['results']['win_rate']*100:.2f}%로 기준({result['threshold']*100:.1f}%) 달성")
        else:
            print(f"\n❌ 새 모델 기각")
            print(f"   승률 {result['results']['win_rate']*100:.2f}%로 기준({result['threshold']*100:.1f}%) 미달")
        
        print("=" * 60)
    
    def _save_evaluation_results(self, result: dict):
        """평가 결과를 파일로 저장"""
        # 결과 디렉토리 생성
        results_dir = Path("evaluation_results")
        results_dir.mkdir(exist_ok=True)
        
        # 파일명 생성 (타임스탬프 기반)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = results_dir / f"evaluation_{timestamp}.json"
        
        try:
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"💾 평가 결과 저장: {result_file}")
        except Exception as e:
            print(f"❌ 결과 저장 실패: {e}")


def main():
    parser = argparse.ArgumentParser(description="YINSH AlphaZero 모델 평가")
    parser.add_argument("--candidate", type=str, required=True, 
                       help="평가할 새 모델 경로")
    parser.add_argument("--best", type=str, default=None,
                       help="기존 best 모델 경로 (없으면 랜덤 baseline)")
    parser.add_argument("--games", type=int, default=100,
                       help="평가 게임 수")
    parser.add_argument("--threshold", type=float, default=0.55,
                       help="새 모델 채택을 위한 최소 승률 (0.0-1.0)")
    parser.add_argument("--mcts-sims", type=int, default=None,
                       help="MCTS 시뮬레이션 수 (평가용, 비워두면 config에서 가져옴)")
    parser.add_argument("--no-save", action="store_true",
                       help="결과 파일 저장 안함")
    
    # 최적화 옵션
    parser.add_argument("--no-parallel", action="store_true",
                       help="병렬 처리 비활성화 (순차 실행)")
    parser.add_argument("--workers", type=int, default=None,
                       help="병렬 워커 수 (기본: CPU 코어 수의 절반)")
    parser.add_argument("--no-gpu", action="store_true",
                       help="GPU 사용 비활성화 (CPU만 사용)")
    parser.add_argument("--fast", action="store_true",
                       help="빠른 평가 모드 (MCTS 시뮬레이션 감소)")
    
    args = parser.parse_args()
    
    # 입력 검증
    if not os.path.exists(args.candidate):
        print(f"❌ 후보 모델 파일을 찾을 수 없습니다: {args.candidate}")
        return
    
    if args.best and not os.path.exists(args.best):
        print(f"❌ 기존 모델 파일을 찾을 수 없습니다: {args.best}")
        return
    
    if not (0.0 <= args.threshold <= 1.0):
        print(f"❌ 임계값은 0.0과 1.0 사이여야 합니다: {args.threshold}")
        return
    
    # 최적화 설정
    use_parallel = not args.no_parallel
    use_gpu = not args.no_gpu
    
    # 빠른 모드 설정
    if args.fast:
        mcts_sims = min(args.mcts_sims or config.EVALUATION_MCTS_SIMULATIONS, 50)
        print("⚡ 빠른 평가 모드: MCTS 시뮬레이션 수 감소")
    else:
        mcts_sims = args.mcts_sims if args.mcts_sims is not None else config.EVALUATION_MCTS_SIMULATIONS
    
    print(f"🚀 최적화 설정:")
    print(f"   병렬 처리: {use_parallel}")
    print(f"   GPU 사용: {use_gpu}")
    print(f"   MCTS 시뮬레이션: {mcts_sims}")
    print(f"   워커 수: {args.workers or '자동'}")
    
    # 평가 실행
    evaluator = ModelEvaluator(
        evaluation_games=args.games,
        mcts_sims=mcts_sims,
        use_parallel=use_parallel,
        num_workers=args.workers,
        use_gpu=use_gpu
    )
    
    result = evaluator.evaluate_models(
        candidate_model_path=args.candidate,
        best_model_path=args.best,
        threshold=args.threshold,
        save_results=not args.no_save
    )
    
    if result:
        # 반환 코드로 결과 전달 (자동화 파이프라인용)
        if result['results']['should_replace']:
            print("🔄 반환 코드: 0 (모델 교체 필요)")
            sys.exit(0)  # 성공 - 모델 교체
        else:
            print("🔄 반환 코드: 1 (모델 유지)")
            sys.exit(1)  # 모델 교체 불필요
    else:
        print("🔄 반환 코드: 2 (평가 실패)")
        sys.exit(2)  # 평가 실패


if __name__ == "__main__":
    main() 