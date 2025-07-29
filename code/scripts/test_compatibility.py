#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH AlphaZero Compatibility Test
==================================

개선된 코드의 호환성을 테스트하는 스크립트
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yinsh import YinshEnv, YinshAgent, YinshModel, Color, config


def test_model_compatibility():
    """모델 호환성 테스트"""
    print("🧪 모델 호환성 테스트")
    print("=" * 50)
    
    try:
        # 새 모델 생성
        print("1. 새 모델 생성 테스트...")
        model = YinshModel()
        print("   ✅ 새 모델 생성 성공")
        
        # 모델 구조 확인
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   📊 모델 파라미터: {total_params:,}")
        
        # 입력 테스트
        print("2. 모델 입력 테스트...")
        test_input = torch.randn(1, 15, 11, 11)
        policy_output, value_output = model(test_input)
        
        print(f"   📊 정책 출력 형태: {policy_output.shape}")
        print(f"   📊 가치 출력 형태: {value_output.shape}")
        print("   ✅ 모델 순전파 성공")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 모델 테스트 실패: {e}")
        return False


def test_agent_compatibility():
    """에이전트 호환성 테스트"""
    print("\n🧪 에이전트 호환성 테스트")
    print("=" * 50)
    
    try:
        # 에이전트 생성
        print("1. 에이전트 생성 테스트...")
        agent = YinshAgent(use_mcts=True, device="cpu")
        print("   ✅ 에이전트 생성 성공")
        
        # 게임 환경 테스트
        print("2. 게임 환경 테스트...")
        env = YinshEnv()
        print("   ✅ 게임 환경 생성 성공")
        
        # 액션 선택 테스트
        print("3. 액션 선택 테스트...")
        action, action_info = agent.select_action(env)
        print(f"   📊 선택된 액션: {action}")
        print(f"   📊 액션 정보: {action_info.get('method', 'unknown')}")
        print("   ✅ 액션 선택 성공")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 에이전트 테스트 실패: {e}")
        return False


def test_training_data_compatibility():
    """훈련 데이터 호환성 테스트"""
    print("\n🧪 훈련 데이터 호환성 테스트")
    print("=" * 50)
    
    try:
        # 가상 훈련 데이터 생성
        print("1. 가상 훈련 데이터 생성...")
        num_positions = 100
        states = torch.randn(num_positions, 15, 11, 11)
        policies = torch.randn(num_positions, 4000)
        values = torch.randn(num_positions)
        
        # 정책 정규화
        policy_sums = torch.sum(policies, dim=1, keepdim=True)
        policies = policies / (policy_sums + 1e-8)
        policies = torch.log(policies + 1e-8)
        
        print(f"   📊 상태 형태: {states.shape}")
        print(f"   📊 정책 형태: {policies.shape}")
        print(f"   📊 가치 형태: {values.shape}")
        print("   ✅ 훈련 데이터 생성 성공")
        
        # 모델 훈련 테스트
        print("2. 모델 훈련 테스트...")
        model = YinshModel()
        model.train()
        
        # 손실 함수
        policy_criterion = torch.nn.KLDivLoss(reduction='batchmean')
        value_criterion = torch.nn.MSELoss()
        
        # 옵티마이저
        optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
        
        # 한 배치 훈련
        batch_size = 32
        batch_states = states[:batch_size]
        batch_policies = policies[:batch_size]
        batch_values = values[:batch_size]
        
        optimizer.zero_grad()
        policy_output, value_output = model(batch_states)
        
        policy_loss = policy_criterion(policy_output, batch_policies)
        value_loss = value_criterion(value_output.squeeze(), batch_values)
        total_loss = policy_loss + value_loss
        
        total_loss.backward()
        optimizer.step()
        
        print(f"   📊 정책 손실: {policy_loss.item():.4f}")
        print(f"   📊 가치 손실: {value_loss.item():.4f}")
        print(f"   📊 총 손실: {total_loss.item():.4f}")
        print("   ✅ 모델 훈련 성공")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 훈련 데이터 테스트 실패: {e}")
        return False


def test_config_compatibility():
    """설정 호환성 테스트"""
    print("\n🧪 설정 호환성 테스트")
    print("=" * 50)
    
    try:
        # 중요 설정 확인
        print("1. 학습 설정 확인...")
        print(f"   📊 학습률: {config.LEARNING_RATE}")
        print(f"   📊 배치 크기: {config.BATCH_SIZE}")
        print(f"   📊 에포크 수: {config.EPOCHS_PER_LOOP}")
        print(f"   📊 MCTS 시뮬레이션: {config.MCTS_SIMULATIONS}")
        print(f"   📊 Selfplay 시뮬레이션: {config.SELFPLAY_MCTS_SIMULATIONS}")
        print("   ✅ 설정 확인 성공")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 설정 테스트 실패: {e}")
        return False


def main():
    """메인 테스트 함수"""
    print("🚀 YINSH AlphaZero 호환성 테스트 시작")
    print("=" * 60)
    
    tests = [
        ("모델 호환성", test_model_compatibility),
        ("에이전트 호환성", test_agent_compatibility),
        ("훈련 데이터 호환성", test_training_data_compatibility),
        ("설정 호환성", test_config_compatibility),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} 테스트 통과")
            else:
                print(f"❌ {test_name} 테스트 실패")
        except Exception as e:
            print(f"❌ {test_name} 테스트 오류: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 테스트 결과: {passed}/{total} 통과")
    
    if passed == total:
        print("🎉 모든 테스트 통과! 코드가 정상적으로 작동합니다.")
        return True
    else:
        print("⚠️ 일부 테스트 실패. 코드를 다시 검토해주세요.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)