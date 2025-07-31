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
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yinsh import YinshModel, config


def find_optimal_batch_size(model, sample_input_shape, device, max_batch_size=1024):
    """GPU 메모리에 맞는 최적 배치 크기 찾기"""
    model.eval()
    optimal_batch_size = 32  # 최소값
    
    print("🔍 최적 배치 크기 탐색 중...")
    
    for batch_size in [32, 64, 128, 256, 512, 1024, 2048]:
        if batch_size > max_batch_size:
            break
            
        try:
            # 테스트 데이터 생성
            test_states = torch.randn(batch_size, *sample_input_shape, device=device)
            test_policies = torch.randn(batch_size, config.POLICY_OUTPUT_SIZE, device=device)
            test_values = torch.randn(batch_size, device=device)
            
            # 메모리 사용량 확인
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated()
            
            # 순전파 테스트
            with torch.no_grad():
                policy_output, value_output = model(test_states)
                
            if device.type == 'cuda':
                memory_after = torch.cuda.memory_allocated()
                memory_used = (memory_after - memory_before) / 1024**2  # MB
                
                print(f"   배치 {batch_size}: {memory_used:.1f} MB")
                
            optimal_batch_size = batch_size
            
            # 메모리 정리
            del test_states, test_policies, test_values, policy_output, value_output
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   배치 {batch_size}: 메모리 부족")
                break
            else:
                raise e
    
    print(f"✅ 최적 배치 크기: {optimal_batch_size}")
    model.train()
    return optimal_batch_size


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
            
            # selfplay.py 새 형식 처리 (generate_training_data 결과)
            if isinstance(saved_data, dict) and "states" in saved_data:
                print(f"   📊 selfplay.py 새 형식 감지")
                states = saved_data["states"]
                policies = saved_data["policies"] 
                values = saved_data["values"]
                
                # 빈 데이터 파일 건너뛰기
                if states.numel() == 0 or policies.numel() == 0 or values.numel() == 0:
                    print(f"   ⚠️ 빈 데이터 파일 건너뛰기: {file_path}")
                    continue

                print(f"   📊 데이터 크기:")
                print(f"      상태: {states.shape}")
                print(f"      정책: {policies.shape}")
                print(f"      가치: {values.shape}")
                
                # 데이터 검증 및 추가
                for i, (state, policy, value) in enumerate(zip(states, policies, values)):
                    # 빈 데이터 건너뛰기
                    if state.numel() == 0 or policy.numel() == 0:
                        continue
                    
                    # 상태 검증 (새 게임 규칙: 6채널)
                    if len(state.shape) != 3 or state.shape != config.INPUT_SHAPE:
                        print(f"   ⚠️ 상태 형태 불일치 (position {i}): {state.shape}, 예상: {config.INPUT_SHAPE}")
                        continue
                    
                    # 정책 검증 (새 게임 규칙: 1848개 액션)
                    if len(policy.shape) != 1 or policy.shape[0] != config.POLICY_OUTPUT_SIZE:
                        print(f"   ⚠️ 정책 형태 불일치 (position {i}): {policy.shape}, 예상: ({config.POLICY_OUTPUT_SIZE},)")
                        continue
                    
                    # 가치 검증 (범위 확인)
                    if not isinstance(value, (int, float, torch.Tensor)):
                        print(f"   ⚠️ 가치 타입 불일치 (position {i}): {type(value)}")
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
                    
                print(f"   ✅ {len(all_states)} 포지션 추가됨")
            
            else:
                print(f"   ⚠️ 알 수 없는 데이터 형식: {type(saved_data)}")
                print(f"   📋 데이터 키: {list(saved_data.keys()) if isinstance(saved_data, dict) else 'N/A'}")

        except Exception as e:
            print(f"⚠️  Error loading {file_path}: {e}")
            continue

    if not all_states:
        print("❌ No valid training data found!")
        return None, None, None

    print(f"📊 Loaded {len(all_states)} valid training positions")

    # NumPy 배열로 변환
    try:
        states = np.array(all_states, dtype=np.float32)
        policies = np.array(all_policies, dtype=np.float32)
        values = np.array(all_values, dtype=np.float32)
        
        # NaN/Inf 검증
        if np.any(np.isnan(states)) or np.any(np.isinf(states)):
            print("⚠️ 상태 데이터에 NaN/Inf 값 발견, 제거 중...")
            valid_mask = ~(np.isnan(states).any(axis=(1,2,3)) | np.isinf(states).any(axis=(1,2,3)))
            states = states[valid_mask]
            policies = policies[valid_mask]
            values = values[valid_mask]
            
        if np.any(np.isnan(policies)) or np.any(np.isinf(policies)):
            print("⚠️ 정책 데이터에 NaN/Inf 값 발견, 제거 중...")
            valid_mask = ~(np.isnan(policies).any(axis=1) | np.isinf(policies).any(axis=1))
            states = states[valid_mask]
            policies = policies[valid_mask]
            values = values[valid_mask]
            
        if np.any(np.isnan(values)) or np.any(np.isinf(values)):
            print("⚠️ 가치 데이터에 NaN/Inf 값 발견, 제거 중...")
            valid_mask = ~(np.isnan(values) | np.isinf(values))
            states = states[valid_mask]
            policies = policies[valid_mask]
            values = values[valid_mask]
            
        print(f"📊 정제 후 데이터: {len(states)} positions")
        
    except Exception as e:
        print(f"❌ 배열 변환 실패: {e}")
        return None, None, None

    # 정책 정규화 (AlphaZero 논문 기반)
    # 각 정책의 합이 1이 되도록 정규화
    policy_sums = np.sum(policies, axis=1, keepdims=True)
    policies = policies / (policy_sums + 1e-8)  # 0으로 나누기 방지
    
    # MSE 손실 사용하므로 로그 변환하지 않음
    # policies는 그대로 확률 분포로 유지

    # 데이터 통계 출력
    print(f"📈 Data Statistics:")
    print(f"  States shape: {states.shape}")
    print(f"  Policies shape: {policies.shape}")
    print(f"  Values range: [{np.min(values):.3f}, {np.max(values):.3f}]")
    print(f"  Values mean: {np.mean(values):.3f}")
    print(f"  Values std: {np.std(values):.3f}")

    return states, policies, values


def train_model(model, states, policies, values, epochs=100, batch_size=512, lr=0.002, use_mixed_precision=True):
    """모델 훈련 (AlphaZero 논문 기반 + GPU 최적화)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Mixed Precision Training 설정
    use_amp = use_mixed_precision and device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    
    print(f"🚀 GPU 최적화 설정:")
    print(f"   Device: {device}")
    print(f"   Mixed Precision: {use_amp}")
    if device.type == 'cuda':
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 메모리 효율적 데이터 로딩을 위해 CPU에서 데이터셋 생성
    states_tensor = torch.FloatTensor(states)
    policies_tensor = torch.FloatTensor(policies)  
    values_tensor = torch.FloatTensor(values)

    # 데이터셋 생성
    dataset = TensorDataset(states_tensor, policies_tensor, values_tensor)
    
    # 최적화된 DataLoader 설정
    num_workers = min(8, torch.get_num_threads())  # CPU 코어에 맞춰 조정
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,  # GPU 전송 최적화
        persistent_workers=True,  # 워커 재사용으로 오버헤드 감소
        prefetch_factor=2,  # 미리 가져올 배치 수
        drop_last=True  # 마지막 불완전한 배치 제거
    )
    
    print(f"📊 DataLoader 설정:")
    print(f"   배치 크기: {batch_size}")
    print(f"   워커 수: {num_workers}")
    print(f"   Pin Memory: {device.type == 'cuda'}")
    print(f"   데이터셋 크기: {len(dataset):,}")

    # 손실 함수 (AlphaZero 논문 기반)
    # MSE for policy loss (확률 분포 대 확률 분포)
    # AlphaZero 논문에서는 실제로 MSE를 사용
    policy_criterion = nn.MSELoss()
    value_criterion = nn.MSELoss()
    
    # 옵티마이저 (AlphaZero 논문 기반)
    optimizer = optim.SGD(
        model.parameters(), 
        lr=lr, 
        momentum=0.9, 
        weight_decay=1e-4
    )
    
    # 학습률 스케줄러 (AlphaZero 논문 기반)
    # 에포크 기반으로 조정 (원래는 iteration 기반이지만 에포크로 단순화)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=max(1, epochs // 3),  # 에포크의 1/3마다 감소
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
            # GPU로 데이터 전송 (pin_memory 덕분에 빠름)
            batch_states = batch_states.to(device, non_blocking=True)
            batch_policies = batch_policies.to(device, non_blocking=True)
            batch_values = batch_values.to(device, non_blocking=True)
            
            optimizer.zero_grad()

            # Mixed Precision Training
            if use_amp:
                with autocast():
                    # 순전파
                    policy_output, value_output = model(batch_states)
                    
                    # 손실 계산
                    policy_probs = torch.softmax(policy_output, dim=1)
                    policy_loss = policy_criterion(policy_probs, batch_policies)
                    value_loss = value_criterion(value_output.squeeze(), batch_values)
                    total_batch_loss = policy_loss + value_loss
                
                # Scaled 역전파
                scaler.scale(total_batch_loss).backward()
                
                # 그래디언트 클리핑
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 옵티마이저 스텝
                scaler.step(optimizer)
                scaler.update()
            else:
                # 일반 훈련 (FP32)
                policy_output, value_output = model(batch_states)
                policy_probs = torch.softmax(policy_output, dim=1)
                policy_loss = policy_criterion(policy_probs, batch_policies)
                value_loss = value_criterion(value_output.squeeze(), batch_values)
                total_batch_loss = policy_loss + value_loss
                
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            
            # GPU 메모리 사용량 모니터링
            gpu_memory_info = ""
            if device.type == 'cuda':
                memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
                memory_cached = torch.cuda.memory_reserved() / 1024**2  # MB
                gpu_memory_info = f"GPU: {memory_used:.0f}MB"
            
            progress_bar.set_postfix({
                "Policy Loss": f"{policy_loss.item():.4f}",
                "Value Loss": f"{value_loss.item():.4f}",
                "Total Loss": f"{total_batch_loss.item():.4f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.6f}",
                "Memory": gpu_memory_info
            })

        # 에포크 평균 손실
        avg_policy_loss = epoch_policy_loss / len(dataloader)
        avg_value_loss = epoch_value_loss / len(dataloader)
        avg_total_loss = avg_policy_loss + avg_value_loss
        
        policy_losses.append(avg_policy_loss)
        value_losses.append(avg_value_loss)
        total_loss += avg_total_loss
        
        # GPU 메모리 사용량 보고
        memory_info = ""
        if device.type == 'cuda':
            memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_cached = torch.cuda.memory_reserved() / 1024**2  # MB
            memory_info = f" | GPU: {memory_used:.0f}MB"
            
            # 에포크 종료 시 캐시 정리
            if (epoch + 1) % 10 == 0:  # 10에포크마다
                torch.cuda.empty_cache()
        
        print(f"📈 Epoch {epoch+1} - Policy: {avg_policy_loss:.4f}, Value: {avg_value_loss:.4f}, Total: {avg_total_loss:.4f}{memory_info}")
        
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
    parser.add_argument("--mixed-precision", action="store_true", default=True, help="Use mixed precision training (default: True)")
    parser.add_argument("--no-mixed-precision", dest="mixed_precision", action="store_false", help="Disable mixed precision training")

    args = parser.parse_args()

    print("🚀 Starting YINSH AlphaZero Training (AlphaZero 논문 기반)...")
    print("=" * 60)
    print(f"📋 Training Parameters:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  Momentum: {args.momentum}")
    print(f"  Mixed Precision: {args.mixed_precision}")
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
    else:
        print("🆕 Creating new model (AlphaZero 논문 기반)")
        model = YinshModel()

    # 모델 정보 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model Statistics:")
    print(f"  Total Parameters: {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    
    # GPU 최적화: 동적 배치 크기 조정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda' and args.batch_size > 32:
        optimal_batch_size = find_optimal_batch_size(
            model, 
            config.INPUT_SHAPE, 
            device, 
            max_batch_size=args.batch_size
        )
        if optimal_batch_size != args.batch_size:
            print(f"🔧 배치 크기 조정: {args.batch_size} → {optimal_batch_size}")
            args.batch_size = optimal_batch_size

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
        use_mixed_precision=args.mixed_precision,
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
