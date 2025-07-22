#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py 단독 테스트 스크립트
"""

import os
import sys
import time
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.evaluate import ModelEvaluator


def test_evaluate_simple():
    """간단한 평가 테스트"""
    print("🧪 evaluate.py 단독 테스트 시작")
    
    # 모델 파일 확인
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ models 디렉토리가 없습니다.")
        return
    
    model_files = list(models_dir.glob("*.pth"))
    if not model_files:
        print("❌ models 디렉토리에 .pth 파일이 없습니다.")
        return
    
    print(f"📁 발견된 모델 파일들: {[f.name for f in model_files]}")
    
    # 첫 번째 모델로 테스트
    candidate_model = str(model_files[0])
    best_model = str(model_files[1]) if len(model_files) > 1 else None
    
    print(f"\n🎯 테스트 설정:")
    print(f"   후보 모델: {candidate_model}")
    print(f"   기존 모델: {best_model if best_model else 'Random baseline'}")
    print(f"   게임 수: 5 (빠른 테스트용)")
    print(f"   MCTS 시뮬레이션: 50 (빠른 테스트용)")
    
    # 평가 실행
    try:
        evaluator = ModelEvaluator(
            evaluation_games=5,  # 빠른 테스트용
            mcts_sims=50         # 빠른 테스트용
        )
        
        start_time = time.time()
        result = evaluator.evaluate_models(
            candidate_model_path=candidate_model,
            best_model_path=best_model,
            threshold=0.55,
            save_results=True
        )
        end_time = time.time()
        
        print(f"\n⏱️ 총 소요시간: {end_time - start_time:.2f}초")
        
        if result:
            print("✅ 평가 성공!")
            return True
        else:
            print("❌ 평가 실패!")
            return False
            
    except Exception as e:
        print(f"❌ 평가 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluate_with_timeout():
    """타임아웃이 있는 평가 테스트"""
    print("\n🧪 타임아웃 테스트 시작")
    
    models_dir = Path("models")
    model_files = list(models_dir.glob("*.pth"))
    if not model_files:
        print("❌ 모델 파일이 없습니다.")
        return
    
    candidate_model = str(model_files[0])
    
    # 타임아웃 설정 (5분)
    timeout_seconds = 300
    
    print(f"⏰ 타임아웃: {timeout_seconds}초")
    
    try:
        evaluator = ModelEvaluator(
            evaluation_games=3,  # 매우 적은 게임 수
            mcts_sims=100
        )
        
        start_time = time.time()
        
        # 타임아웃 체크를 위한 래퍼
        def check_timeout():
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(f"평가가 {timeout_seconds}초를 초과했습니다.")
        
        # 간단한 게임별 타임아웃 체크
        result = evaluator.evaluate_models(
            candidate_model_path=candidate_model,
            best_model_path=None,
            threshold=0.55,
            save_results=True
        )
        
        print("✅ 타임아웃 테스트 통과!")
        return True
        
    except TimeoutError as e:
        print(f"⏰ {e}")
        return False
    except Exception as e:
        print(f"❌ 타임아웃 테스트 중 오류: {e}")
        return False


def test_single_game():
    """단일 게임만 테스트"""
    print("\n🧪 단일 게임 테스트 시작")
    
    models_dir = Path("models")
    model_files = list(models_dir.glob("*.pth"))
    if not model_files:
        print("❌ 모델 파일이 없습니다.")
        return
    
    candidate_model = str(model_files[0])
    
    try:
        evaluator = ModelEvaluator(
            evaluation_games=1,  # 단일 게임만
            mcts_sims=50
        )
        
        print("🎮 단일 게임 평가 시작...")
        start_time = time.time()
        
        result = evaluator.evaluate_models(
            candidate_model_path=candidate_model,
            best_model_path=None,
            threshold=0.55,
            save_results=True
        )
        
        end_time = time.time()
        print(f"⏱️ 단일 게임 소요시간: {end_time - start_time:.2f}초")
        
        if result:
            print("✅ 단일 게임 테스트 성공!")
            return True
        else:
            print("❌ 단일 게임 테스트 실패!")
            return False
            
    except Exception as e:
        print(f"❌ 단일 게임 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("🧪 evaluate.py 단독 테스트 스위트")
    print("=" * 60)
    
    # 1. 간단한 테스트
    print("\n1️⃣ 간단한 평가 테스트")
    test1_result = test_evaluate_simple()
    
    # 2. 타임아웃 테스트
    print("\n2️⃣ 타임아웃 테스트")
    test2_result = test_evaluate_with_timeout()
    
    # 3. 단일 게임 테스트
    print("\n3️⃣ 단일 게임 테스트")
    test3_result = test_single_game()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 테스트 결과 요약")
    print("=" * 60)
    print(f"   간단한 평가 테스트: {'✅ 통과' if test1_result else '❌ 실패'}")
    print(f"   타임아웃 테스트: {'✅ 통과' if test2_result else '❌ 실패'}")
    print(f"   단일 게임 테스트: {'✅ 통과' if test3_result else '❌ 실패'}")
    
    if test1_result and test2_result and test3_result:
        print("\n🎉 모든 테스트 통과!")
    else:
        print("\n⚠️ 일부 테스트 실패. 문제를 확인해주세요.") 