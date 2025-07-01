#!/usr/bin/env python3
"""
학습 상태 진단 스크립트
AlphaYinsh 프로젝트의 학습이 제대로 되고 있는지 확인합니다.
"""

import torch
import numpy as np
import time
import os
import config
from model import YinshNet
from train import Trainer, YinshDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def diagnose_config():
    """현재 설정 확인"""
    print("=== 🔍 현재 설정 진단 ===")
    print(f"학습률: {config.LEARNING_RATE}")
    print(f"배치 크기: {config.BATCH_SIZE}")
    print(f"입력 형태: {config.INPUT_SHAPE}")
    print(f"출력 형태: {config.OUTPUT_SHAPE}")
    print(f"ResNet 블록 수: {config.AMOUNT_OF_RESIDUAL_BLOCKS}")
    print(f"컨볼루션 필터: {config.CONVOLUTION_FILTERS}")
    print()

def diagnose_model():
    """모델 구조 진단"""
    print("=== 🧠 모델 구조 진단 ===")
    model = YinshNet()
    
    # 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"총 파라미터 수: {total_params:,}")
    print(f"학습 가능한 파라미터 수: {trainable_params:,}")
    
    # 순전파 테스트
    test_input = torch.randn(1, *config.INPUT_SHAPE)
    try:
        with torch.no_grad():
            policy, value = model(test_input)
            print(f"Policy 출력 형태: {policy.shape}")
            print(f"Value 출력 형태: {value.shape}")
            print("✅ 모델 순전파 정상")
    except Exception as e:
        print(f"❌ 모델 순전파 오류: {e}")
    print()

def diagnose_data():
    """학습 데이터 진단"""
    print("=== 📊 학습 데이터 진단 ===")
    
    memory_files = [f for f in os.listdir(config.MEMORY_DIR) if f.endswith('.npy')]
    print(f"메모리 파일 수: {len(memory_files)}")
    
    if not memory_files:
        print("❌ 학습 데이터가 없습니다!")
        return
    
    # 첫 번째 파일 분석
    try:
        sample_data = np.load(os.path.join(config.MEMORY_DIR, memory_files[0]), allow_pickle=True)
        print(f"샘플 파일 엔트리 수: {len(sample_data)}")
        
        if len(sample_data) > 0:
            entry = sample_data[0]
            print(f"샘플 엔트리 타입: {type(entry)}")
            print(f"엔트리 길이: {len(entry) if hasattr(entry, '__len__') else 'N/A'}")
            
            if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                state, action_probs, value = entry
                print(f"State 타입: {type(state)}")
                print(f"Action probs 타입: {type(action_probs)}")
                print(f"Value: {value}")
                print("✅ 데이터 구조 정상")
            else:
                print("❌ 데이터 구조 이상")
    except Exception as e:
        print(f"❌ 데이터 로딩 오류: {e}")
    print()

def test_training_speed():
    """학습 속도 테스트"""
    print("=== ⚡ 학습 속도 테스트 ===")
    
    model = YinshNet()
    trainer = Trainer(model)
    
    # 적절한 크기의 가짜 데이터 생성
    print("가짜 데이터 생성 중...")
    fake_data = []
    for i in range(config.BATCH_SIZE * 4):  # 4 배치 분량
        state = np.random.random(config.INPUT_SHAPE).astype(np.float32)
        policy = np.random.random(config.OUTPUT_SHAPE[0]).astype(np.float32)
        policy = policy / policy.sum()  # 정규화
        value = np.float32(np.random.random() * 2 - 1)  # -1 to 1
        fake_data.append((state, policy, value))
    
    print(f"생성된 데이터: {len(fake_data)}개")
    
    # 학습 속도 측정
    start_time = time.time()
    history = trainer.train(fake_data, epochs=3)
    end_time = time.time()
    
    print(f"총 학습 시간: {end_time - start_time:.3f}초")
    print(f"평균 epoch 시간: {(end_time - start_time)/3:.3f}초")
    
    if history:
        first_loss = history[0]['total_loss']
        last_loss = history[-1]['total_loss']
        print(f"첫 번째 loss: {first_loss:.6f}")
        print(f"마지막 loss: {last_loss:.6f}")
        
        if last_loss < first_loss:
            print("✅ Loss가 감소하고 있습니다 (정상 학습)")
        elif last_loss == first_loss:
            print("⚠️ Loss가 변화하지 않습니다 (학습률 확인 필요)")
        else:
            print("❌ Loss가 증가하고 있습니다 (학습률이 너무 높을 수 있음)")
    
    print()

def test_mcts_speed():
    """MCTS 시뮬레이션 속도 테스트"""
    print("=== 🌳 MCTS 속도 테스트 ===")
    
    from agent import Agent
    from mcts import MCTS
    from yinshEnv import YinshEnv
    
    agent = Agent()
    
    # 기본 게임 상태 생성
    env = YinshEnv()
    game_state = {
        'board': env,
        'current_player': env.current_player,
        'game_phase': env.game_phase,
        'rings_placed': env.rings_placed.copy(),
        'rings_removed': env.rings_removed.copy(),
        'move_count': env.move_count
    }
    
    # MCTS 초기화
    agent.mcts = MCTS(agent, state=game_state, stochastic=False)
    
    # MCTS 시뮬레이션 속도 측정
    start_time = time.time()
    agent.run_simulations(100)  # 100 시뮬레이션
    end_time = time.time()
    
    print(f"100 시뮬레이션 시간: {end_time - start_time:.3f}초")
    print(f"시뮬레이션당 평균 시간: {(end_time - start_time)/100*1000:.1f}ms")
    
    if (end_time - start_time) < 1.0:
        print("⚠️ MCTS가 너무 빠릅니다. 게임 로직이 제대로 구현되지 않았을 수 있습니다.")
    else:
        print("✅ MCTS 속도가 적절합니다.")
    
    print()

def main():
    """메인 진단 함수"""
    print("🔍 AlphaYinsh 학습 진단을 시작합니다...\n")
    
    diagnose_config()
    diagnose_model()
    diagnose_data()
    test_training_speed()
    test_mcts_speed()
    
    print("=== 📋 진단 완료 ===")
    print("\n권장사항:")
    print("1. 학습률이 이제 0.001로 수정되었습니다 (이전: 0.2)")
    print("2. 배치 크기가 32로 조정되었습니다 (이전: 64)")
    print("3. 실제 Yinsh 게임 로직 구현을 완성해야 합니다")
    print("4. 더 많은 self-play 데이터를 생성해야 합니다")
    print("5. MCTS가 너무 빠르다면 게임 로직을 점검하세요")

if __name__ == "__main__":
    main() 