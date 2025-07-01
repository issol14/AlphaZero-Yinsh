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


def create_training_batch(data_folder: str, batch_size: int = 32):
    """훈련 배치 생성"""
    all_states = []
    all_policies = []
    all_values = []

    # 데이터 폴더에서 모든 .pt 파일 로드
    for file in os.listdir(data_folder):
        if file.endswith(".pt"):
            file_path = os.path.join(data_folder, file)
            try:
                saved_data = torch.load(file_path, map_location="cpu")
                states = saved_data["states"]
                policies = saved_data["policies"]
                values = saved_data["values"]

                # 각 포지션을 튜플로 변환
                for state, policy, value in zip(states, policies, values):
                    all_states.append(state)
                    all_policies.append(policy)
                    all_values.append(value)

            except Exception as e:
                print(f"⚠️  Error loading {file}: {e}")
                continue

    if not all_states:
        print("❌ No training data found!")
        return None, None, None

    print(f"📊 Loaded {len(all_states)} training positions")

    # NumPy 배열로 변환
    states = np.array(all_states, dtype=np.float32)
    policies = np.array(all_policies, dtype=np.float32)
    values = np.array(all_values, dtype=np.float32)

    # 값들을 float로 변환 (unsqueeze 문제 해결)
    values = values.astype(np.float32)

    return states, policies, values


def train_model(model, states, policies, values, epochs=10, batch_size=32, lr=0.001):
    """모델 훈련"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 데이터를 텐서로 변환
    states_tensor = torch.FloatTensor(states).to(device)
    policies_tensor = torch.FloatTensor(policies).to(device)
    values_tensor = torch.FloatTensor(values).to(device)

    # 데이터셋 생성
    dataset = TensorDataset(states_tensor, policies_tensor, values_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 손실 함수와 옵티마이저
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 훈련 루프
    model.train()
    total_loss = 0

    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch_states, batch_policies, batch_values in progress_bar:
            optimizer.zero_grad()

            # 순전파
            policy_output, value_output = model(batch_states)

            # 손실 계산
            policy_loss = policy_criterion(policy_output, batch_policies)
            value_loss = value_criterion(value_output.squeeze(), batch_values)
            total_batch_loss = policy_loss + value_loss

            # 역전파
            total_batch_loss.backward()
            optimizer.step()

            epoch_loss += total_batch_loss.item()
            progress_bar.set_postfix(
                {
                    "Policy Loss": f"{policy_loss.item():.4f}",
                    "Value Loss": f"{value_loss.item():.4f}",
                    "Total Loss": f"{total_batch_loss.item():.4f}",
                }
            )

        avg_epoch_loss = epoch_loss / len(dataloader)
        total_loss += avg_epoch_loss
        print(f"📈 Epoch {epoch+1} - Average Loss: {avg_epoch_loss:.4f}")

    avg_total_loss = total_loss / epochs
    print(f"🎯 Training completed - Average Loss: {avg_total_loss:.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train YINSH AlphaZero model")
    parser.add_argument(
        "--data", type=str, default="memory", help="Training data folder"
    )
    parser.add_argument(
        "--model", type=str, default=None, help="Model to continue training"
    )
    parser.add_argument(
        "--output", type=str, default="models", help="Output model folder"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()

    print("🚀 Starting YINSH AlphaZero Training...")

    # 출력 폴더 생성
    os.makedirs(args.output, exist_ok=True)

    # 모델 로드 또는 생성
    if args.model and os.path.exists(args.model):
        print(f"📥 Loading existing model: {args.model}")
        model = torch.load(args.model, map_location="cpu")
    else:
        print("🆕 Creating new model")
        model = YinshModel()

    # 훈련 데이터 로드
    print(f"📂 Loading training data from: {args.data}")
    states, policies, values = create_training_batch(args.data, args.batch_size)

    if states is None:
        print("❌ No training data available!")
        return

    # 모델 훈련
    print(f"🎯 Training model for {args.epochs} epochs...")
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
    torch.save(trained_model, model_path)
    print(f"💾 Model saved to: {model_path}")

    print("✅ Training completed successfully!")


if __name__ == "__main__":
    main()
