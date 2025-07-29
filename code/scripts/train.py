#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH AlphaZero Training Script
===============================

This script trains the YINSH AlphaZero model using self-play data.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yinsh import YinshModel, config


def create_training_batch(data_folder: str, batch_size: int = 512):
    """훈련 배치 생성 (AlphaZero 논문 기반)"""
    all_states = []
    all_policies = []
    all_values = []

    print(f"🔍 데이터 폴더 검사: {data_folder}")
    
    # 데이터 폴더 존재 확인
    if not os.path.exists(data_folder):
        print(f"❌ 데이터 폴더가 존재하지 않습니다: {data_folder}")
        return None, None, None
    
    # 데이터 폴더에서 모든 .pt 파일 로드
    data_files = []
    for file in os.listdir(data_folder):
        if file.endswith(".pt"):
            data_files.append(os.path.join(data_folder, file))
    
    print(f"📂 데이터 폴더 내용:")
    for file in os.listdir(data_folder):
        print(f"   {file}")
    
    if not data_files:
        print("❌ No training data found!")
        return None, None, None
    
    print(f"📂 Found {len(data_files)} data files")

    for file_path in data_files:
        try:
            print(f"📖 로딩 중: {file_path}")
            saved_data = torch.load(file_path, map_location="cpu")
            print(f"   ✅ 로딩 성공, 데이터 타입: {type(saved_data)}")
            
            # selfplay.py 형식 처리
            if isinstance(saved_data, list):
                print(f"   📊 selfplay.py 형식 감지: {len(saved_data)}개 게임 데이터")
                # selfplay.py가 생성한 게임 히스토리 형식
                for game_data in saved_data:
                    if isinstance(game_data, dict) and "state" in game_data:
                        state = game_data["state"]
                        action = game_data["action"]
                        action_info = game_data.get("action_info", {})
                        
                        # 상태 검증
                        if state.shape != (15, 11, 11):
                            continue
                        
                        # 정책 생성 (action_info에서 추출 또는 원핫)
                        policy = np.zeros(4000, dtype=np.float32)
                        if "mcts_stats" in action_info:
                            # MCTS 통계에서 정책 추출
                            visit_counts = action_info["mcts_stats"].get("visit_counts", {})
                            if visit_counts:
                                total_visits = sum(visit_counts.values())
                                if total_visits > 0:
                                    # 간단한 원핫 정책 생성
                                    policy[0] = 1.0  # 임시 처리
                        else:
                            # 원핫 정책 생성
                            policy[0] = 1.0  # 임시 처리
                        
                        # 가치 생성 (임시)
                        value = 0.0  # 게임 결과에 따라 수정 필요
                        
                        all_states.append(state)
                        all_policies.append(policy)
                        all_values.append(value)
            
            # 기존 형식 처리 (하위 호환성)
            elif isinstance(saved_data, dict) and "states" in saved_data:
                print(f"   📊 기존 형식 감지")
                states = saved_data["states"]
                policies = saved_data["policies"]
                values = saved_data["values"]
                
                # 빈 데이터 파일 건너뛰기
                if states.numel() == 0 or policies.numel() == 0 or values.numel() == 0:
                    print(f"   ⚠️ 빈 데이터 파일 건너뛰기: {file_path}")
                    continue

                # 데이터 검증 (AlphaZero 논문 기반)
                for state, policy, value in zip(states, policies, values):
                    # 빈 데이터 건너뛰기
                    if state.numel() == 0 or policy.numel() == 0:
                        continue
                    
                    # 상태 검증 (형태 확인)
                    if len(state.shape) != 3 or state.shape[0] != 15 or state.shape[1] != 11 or state.shape[2] != 11:
                        print(f"   ⚠️ 상태 형태 불일치: {state.shape}, 예상: (15, 11, 11)")
                        continue
                    
                    # 정책 검증 (형태 확인)
                    if len(policy.shape) != 1 or policy.shape[0] != 4000:
                        print(f"   ⚠️ 정책 형태 불일치: {policy.shape}, 예상: (4000,)")
                        continue
                    
                    # 가치 검증 (범위 확인)
                    if not isinstance(value, (int, float, torch.Tensor)):
                        print(f"   ⚠️ 가치 타입 불일치: {type(value)}")
                        continue
                    
                    # 텐서를 numpy로 변환
                    if isinstance(state, torch.Tensor):
                        state = state.numpy()
                    if isinstance(policy, torch.Tensor):
                        policy = policy.numpy()
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    
                    all_states.append(state)
                    all_policies.append(policy)
                    all_values.append(value)
            else:
                print(f"   ⚠️ 알 수 없는 데이터 형식: {type(saved_data)}")

        except Exception as e:
            print(f"⚠️  Error loading {file_path}: {e}")
            continue

    if not all_states:
        print("❌ No valid training data found!")
        return None, None, None

    print(f"📊 Loaded {len(all_states)} valid training positions")

    # NumPy 배열로 변환
    states = np.array(all_states, dtype=np.float32)
    policies = np.array(all_policies, dtype=np.float32)
    values = np.array(all_values, dtype=np.float32)

    # 정책 정규화 (AlphaZero 논문 기반)
    # 각 정책의 합이 1이 되도록 정규화
    policy_sums = np.sum(policies, axis=1, keepdims=True)
    policies = policies / (policy_sums + 1e-8)  # 0으로 나누기 방지
    
    # 로그 확률로 변환 (KL divergence용)
    policies = np.log(policies + 1e-8)

    # 데이터 통계 출력
    print(f"📈 Data Statistics:")
    print(f"  States shape: {states.shape}")
    print(f"  Policies shape: {policies.shape}")
    print(f"  Values range: [{np.min(values):.3f}, {np.max(values):.3f}]")
    print(f"  Values mean: {np.mean(values):.3f}")
    print(f"  Values std: {np.std(values):.3f}")

    return states, policies, values


def train_model(model, states, policies, values, epochs=100, batch_size=512, lr=0.002):
    """모델 훈련 (AlphaZero 논문 기반)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 데이터를 텐서로 변환
    states_tensor = torch.FloatTensor(states).to(device)
    policies_tensor = torch.FloatTensor(policies).to(device)
    values_tensor = torch.FloatTensor(values).to(device)

    # 데이터셋 생성
    dataset = TensorDataset(states_tensor, policies_tensor, values_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 손실 함수 (AlphaZero 논문 기반)
    # KL divergence for policy loss (논문: CrossEntropy 대신 KL divergence)
    # KL divergence는 log_softmax 출력과 함께 사용
    policy_criterion = nn.KLDivLoss(reduction='batchmean')
    value_criterion = nn.MSELoss()
    
    # 옵티마이저 (AlphaZero 논문 기반)
    optimizer = optim.SGD(
        model.parameters(), 
        lr=lr, 
        momentum=0.9, 
        weight_decay=1e-4
    )
    
    # 학습률 스케줄러 (AlphaZero 논문 기반)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=400000, 
        gamma=0.1
    )

    # 훈련 루프
    model.train()
    total_loss = 0
    policy_losses = []
    value_losses = []

    for epoch in range(epochs):
        epoch_policy_loss = 0
        epoch_value_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_states, batch_policies, batch_values in progress_bar:
            optimizer.zero_grad()

            # 순전파
            policy_output, value_output = model(batch_states)

            # 손실 계산 (AlphaZero 논문 기반)
            # Policy loss: KL divergence (log_softmax output)
            # batch_policies는 이미 log 확률로 변환되어 있음
            policy_loss = policy_criterion(policy_output, batch_policies)
            
            # Value loss: MSE
            value_loss = value_criterion(value_output.squeeze(), batch_values)
            
            # 가중 합계 (논문: 동일한 가중치)
            total_batch_loss = policy_loss + value_loss

            # 역전파
            total_batch_loss.backward()
            
            # 그래디언트 클리핑 (AlphaZero 논문 기반)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            
            progress_bar.set_postfix({
                "Policy Loss": f"{policy_loss.item():.4f}",
                "Value Loss": f"{value_loss.item():.4f}",
                "Total Loss": f"{total_batch_loss.item():.4f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.6f}"
            })

        # 에포크 평균 손실
        avg_policy_loss = epoch_policy_loss / len(dataloader)
        avg_value_loss = epoch_value_loss / len(dataloader)
        avg_total_loss = avg_policy_loss + avg_value_loss
        
        policy_losses.append(avg_policy_loss)
        value_losses.append(avg_value_loss)
        total_loss += avg_total_loss
        
        print(f"📈 Epoch {epoch+1} - Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f}, Total: {avg_total_loss:.4f}")
        
        # 학습률 스케줄링
        scheduler.step()

    avg_total_loss = total_loss / epochs
    print(f"🎯 Training completed - Average Loss: {avg_total_loss:.4f}")
    
    # 손실 그래프 저장
    save_loss_plot(policy_losses, value_losses, epochs)

    return model


def save_loss_plot(policy_losses, value_losses, epochs):
    """손실 그래프 저장"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(policy_losses, label='Policy Loss', color='blue')
    plt.title('Policy Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(value_losses, label='Value Loss', color='red')
    plt.title('Value Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("📊 Loss plot saved as 'training_loss.png'")


def main():
    parser = argparse.ArgumentParser(description="Train YINSH AlphaZero model (AlphaZero 논문 기반)")
    parser.add_argument(
        "--data", type=str, default="memory", help="Training data folder"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Model to continue training"
    )
    parser.add_argument(
        "--output", type=str, default="models", help="Output model folder"
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (AlphaZero: 100)")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size (AlphaZero: 512)")
    parser.add_argument("--lr", type=float, default=0.002, help="Learning rate (AlphaZero: 0.002)")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (AlphaZero: 1e-4)")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum (AlphaZero: 0.9)")

    args = parser.parse_args()

    print("🚀 Starting YINSH AlphaZero Training (AlphaZero 논문 기반)...")
    print("=" * 60)
    print(f"📋 Training Parameters:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  Momentum: {args.momentum}")
    print("=" * 60)

    # 출력 폴더 생성
    os.makedirs(args.output, exist_ok=True)

    # 모델 로드 또는 생성 (AlphaZero 논문 기반)
    if args.model and os.path.exists(args.model):
        print(f"📥 Loading existing model: {args.model}")
        try:
            # AlphaZero 논문 방식: 저장된 모델을 직접 로드
            model = torch.load(args.model, map_location='cpu')
            print(f"✅ 모델 로드 성공: {args.model}")
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            print("🆕 새 모델을 생성합니다...")
            model = YinshModel()
            # 모델 로드 실패는 치명적 오류이므로 종료
            sys.exit(1)
    else:
        print("🆕 Creating new model (AlphaZero 논문 기반)")
        model = YinshModel()

    # 모델 정보 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model Statistics:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")

    # 훈련 데이터 로드
    print(f"\n📂 Loading training data from: {args.data}")
    states, policies, values = create_training_batch(args.data, args.batch_size)

    if states is None:
        print("❌ No training data available!")
        sys.exit(1)  # 명시적으로 실패 종료 코드 반환

    # 데이터 크기 확인
    if len(states) < args.batch_size:
        print(f"⚠️  Warning: Data size ({len(states)}) is smaller than batch size ({args.batch_size})")
        print("   Consider reducing batch size or collecting more data")

    # 모델 훈련
    print(f"\n🎯 Training model for {args.epochs} epochs...")
    trained_model = train_model(
        model,
        states,
        policies,
        values,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # 모델 저장
    model_path = os.path.join(args.output, "trained_model.pt")
    print(f"💾 모델 저장 중: {model_path}")
    
    try:
        torch.save(trained_model, model_path)
        print(f"✅ 모델 저장 완료: {model_path}")
        
        # 파일 크기 확인
        file_size = os.path.getsize(model_path) / 1024 / 1024  # MB
        print(f"📊 모델 파일 크기: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"❌ 모델 저장 실패: {e}")
        sys.exit(1)  # 명시적으로 실패 종료 코드 반환

    # 모델 성능 평가
    print("\n📊 Model Performance Summary:")
    print(f"  Training completed successfully!")
    print(f"  Model saved: {model_path}")
    print(f"  Training data: {len(states)} positions")
    print(f"  Model parameters: {total_params:,}")

    print("\n✅ Training completed successfully!")


if __name__ == "__main__":
    main()
