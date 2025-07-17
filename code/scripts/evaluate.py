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

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yinsh import YinshEnv, YinshAgent, Color, config
from yinsh.game import YinshGame


class ModelEvaluator:
    """모델 평가를 담당하는 클래스"""
    
    def __init__(self, evaluation_games: int = 100, mcts_sims: int = 400):
        """
        Args:
            evaluation_games: 평가에 사용할 게임 수
            mcts_sims: MCTS 시뮬레이션 수 (평가 시에는 빠른 진행을 위해 적게)
        """
        self.evaluation_games = evaluation_games
        self.mcts_sims = mcts_sims
        self.results = []
        
    def evaluate_models(self, candidate_model_path: str, best_model_path: str = None,
                       threshold: float = 0.55, save_results: bool = True) -> dict:
        """
        두 모델을 평가하여 새 모델 채택 여부 결정
        
        Args:
            candidate_model_path: 새로운 후보 모델 경로
            best_model_path: 기존 best 모델 경로 (None이면 랜덤 초기화)
            threshold: 새 모델 채택을 위한 최소 승률 (기본 55%)
            save_results: 결과 저장 여부
            
        Returns:
            평가 결과 딕셔너리
        """
        print(f"\n🏆 모델 평가 시작")
        print(f"📋 설정:")
        print(f"   후보 모델: {candidate_model_path}")
        print(f"   기존 모델: {best_model_path if best_model_path else 'Random baseline'}")
        print(f"   평가 게임 수: {self.evaluation_games}")
        print(f"   MCTS 시뮬레이션: {self.mcts_sims}")
        print(f"   채택 기준: {threshold*100:.1f}% 이상 승률")
        print("=" * 60)
        
        # 에이전트 생성
        try:
            print("🤖 에이전트 초기화 중...")
            candidate_agent = YinshAgent(
                model_path=candidate_model_path,
                use_mcts=True,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            print("   ✅ 후보 모델 로드 완료")
            
            best_agent = YinshAgent(
                model_path=best_model_path,
                use_mcts=True,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            print("   ✅ 기존 모델 로드 완료")
            
            # MCTS 시뮬레이션 수 조정 (평가 시에는 더 빠르게)
            if hasattr(candidate_agent, 'mcts_agent') and candidate_agent.mcts_agent:
                candidate_agent.mcts_agent.num_simulations = self.mcts_sims
            if hasattr(best_agent, 'mcts_agent') and best_agent.mcts_agent:
                best_agent.mcts_agent.num_simulations = self.mcts_sims
                
        except Exception as e:
            print(f"❌ 에이전트 초기화 실패: {e}")
            return None
        
        # 평가 게임 진행
        candidate_as_white_wins = 0
        candidate_as_black_wins = 0
        draws = 0
        total_game_time = 0
        
        print(f"\n🎮 {self.evaluation_games}게임 평가 시작...")
        
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
                # 게임 진행
                game = YinshGame(white_agent, black_agent)
                winner_value = game.play_one_game(stochastic=False)  # 평가시에는 결정적
                
                # 승자 결정
                winner = None
                if winner_value > 0:
                    winner = Color.WHITE
                elif winner_value < 0:
                    winner = Color.BLACK
                else:
                    winner = None  # 무승부
                
                # 결과 기록
                if winner is None:
                    draws += 1
                elif (winner == Color.WHITE and candidate_is_white) or \
                     (winner == Color.BLACK and not candidate_is_white):
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
                    'turns': game.moves_played if hasattr(game, 'moves_played') else 0
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
    parser.add_argument("--mcts-sims", type=int, default=400,
                       help="MCTS 시뮬레이션 수 (평가용)")
    parser.add_argument("--no-save", action="store_true",
                       help="결과 파일 저장 안함")
    
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
    
    # 평가 실행
    evaluator = ModelEvaluator(
        evaluation_games=args.games,
        mcts_sims=args.mcts_sims
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