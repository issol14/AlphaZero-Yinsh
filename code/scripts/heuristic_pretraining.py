#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Heuristic-based Pre-training for Yinsh
=====================================

완전 랜덤 모델 대신 기본적인 휴리스틱을 학습한 모델로 시작
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from yinsh import YinshEnv, YinshModel, YinshActionMapper, config

def generate_heuristic_policy(env: YinshEnv, mapper: YinshActionMapper) -> np.ndarray:
    """
    간단한 휴리스틱 기반 정책 생성
    1. 중앙에 가까운 위치 선호
    2. 상대방 마커 근처 회피
    3. 연속된 마커 형성 선호
    """
    actions = env.get_valid_actions()
    if not actions:
        # 유효한 액션이 없으면 uniform 분포
        policy = np.ones(config.POLICY_OUTPUT_SIZE) / config.POLICY_OUTPUT_SIZE
        return policy
    
    policy = np.zeros(config.POLICY_OUTPUT_SIZE)
    total_score = 0
    
    # 보드 중심점 (hex 좌표계)
    center_x, center_y = 0, 0
    
    for action in actions:
        action_idx = mapper.get_action_index(action)
        if action_idx == -1:
            continue
            
        score = 1.0  # 기본 점수
        
        # 휴리스틱 1: 중앙 선호도
        if hasattr(action, 'to_position'):
            to_x, to_y = action.to_position
            distance_to_center = abs(to_x - center_x) + abs(to_y - center_y)
            center_bonus = max(0, 10 - distance_to_center) / 10.0
            score += center_bonus
        
        # 휴리스틱 2: 가장자리 회피
        if hasattr(action, 'to_position'):
            to_x, to_y = action.to_position
            if abs(to_x) > 4 or abs(to_y) > 4:
                score *= 0.5  # 가장자리 페널티
        
        # 휴리스틱 3: 랜덤 노이즈 추가 (다양성)
        score += random.uniform(0, 0.3)
        
        policy[action_idx] = score
        total_score += score
    
    # 정규화
    if total_score > 0:
        policy = policy / total_score
    else:
        policy = np.ones(config.POLICY_OUTPUT_SIZE) / config.POLICY_OUTPUT_SIZE
    
    return policy

def generate_heuristic_training_data(num_games=100, max_moves_per_game=20):
    """휴리스틱 기반 훈련 데이터 생성"""
    print(f"🎯 휴리스틱 기반 훈련 데이터 생성 중...")
    print(f"   게임 수: {num_games}")
    print(f"   게임당 최대 수: {max_moves_per_game}")
    
    env = YinshEnv()
    mapper = YinshActionMapper()
    training_data = []
    
    for game_id in range(num_games):
        if game_id % 10 == 0:
            print(f"   진행상황: {game_id}/{num_games} 게임")
        
        env.reset()
        move_count = 0
        
        while not env.is_game_over() and move_count < max_moves_per_game:
            # 현재 상태 저장
            state = env.get_state_tensor()
            
            # 휴리스틱 정책 생성
            policy = generate_heuristic_policy(env, mapper)
            
            # 가치 추정 (게임 중간: 0, 승리 가능성에 따라 조정)
            value = 0.0
            if env.is_game_over():
                winner = env.get_winner()
                if winner == env.current_player:
                    value = 1.0
                elif winner is not None:
                    value = -1.0
                else:
                    value = 0.0
            else:
                # 간단한 보드 평가 (마커 수 기반)
                my_markers = len([pos for pos in env.marker_positions 
                                if env.board[env.hex_to_array_coords(pos)] == env.current_player])
                opponent_markers = len([pos for pos in env.marker_positions 
                                      if env.board[env.hex_to_array_coords(pos)] == env.current_player.opposite()])
                value = np.tanh((my_markers - opponent_markers) * 0.1)
            
            # 훈련 데이터에 추가
            training_data.append({
                'state': state,
                'policy': policy,
                'value': value
            })
            
            # 실제 액션 수행 (휴리스틱 정책 기반)
            actions = env.get_valid_actions()
            if actions:
                # 휴리스틱 정책에 따라 액션 선택
                action_probs = []
                for action in actions:
                    action_idx = mapper.get_action_index(action)
                    if action_idx != -1:
                        action_probs.append(policy[action_idx])
                    else:
                        action_probs.append(0.0)
                
                if sum(action_probs) > 0:
                    action_probs = np.array(action_probs)
                    action_probs = action_probs / action_probs.sum()
                    selected_idx = np.random.choice(len(actions), p=action_probs)
                    selected_action = actions[selected_idx]
                else:
                    selected_action = random.choice(actions)
                
                env.step(selected_action)
                move_count += 1
            else:
                break
    
    print(f"✅ 휴리스틱 데이터 생성 완료!")
    print(f"   총 데이터 포인트: {len(training_data)}")
    
    return training_data

def pretrain_model_with_heuristics(output_path="models/heuristic_pretrained_model.pt", 
                                 num_games=500, epochs=20):
    """휴리스틱 데이터로 모델 사전 훈련"""
    print("🧠 휴리스틱 기반 모델 사전 훈련 시작!")
    
    # 훈련 데이터 생성
    training_data = generate_heuristic_training_data(num_games=num_games)
    
    # 모델 초기화
    model = YinshModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 옵티마이저 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    policy_criterion = nn.MSELoss()
    value_criterion = nn.MSELoss()
    
    # 훈련 데이터 준비
    states = []
    policies = []
    values = []
    
    for data in training_data:
        states.append(data['state'])
        policies.append(data['policy'])
        values.append(data['value'])
    
    states = torch.stack(states).to(device)
    policies = torch.FloatTensor(policies).to(device)
    values = torch.FloatTensor(values).to(device)
    
    print(f"📊 훈련 설정:")
    print(f"   데이터 크기: {len(states)}")
    print(f"   에포크: {epochs}")
    print(f"   디바이스: {device}")
    
    # 훈련 루프
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        policy_loss_sum = 0
        value_loss_sum = 0
        
        # 배치 단위로 훈련
        batch_size = 64
        num_batches = len(states) // batch_size
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(states))
            
            batch_states = states[start_idx:end_idx]
            batch_policies = policies[start_idx:end_idx]
            batch_values = values[start_idx:end_idx]
            
            optimizer.zero_grad()
            
            # 모델 예측
            policy_pred, value_pred = model(batch_states)
            
            # 손실 계산
            policy_loss = policy_criterion(torch.softmax(policy_pred, dim=1), batch_policies)
            value_loss = value_criterion(value_pred.squeeze(), batch_values)
            
            total_loss_batch = policy_loss + value_loss
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
        
        avg_total_loss = total_loss / num_batches
        avg_policy_loss = policy_loss_sum / num_batches
        avg_value_loss = value_loss_sum / num_batches
        
        print(f"   Epoch {epoch+1}/{epochs}: "
              f"Total={avg_total_loss:.4f}, "
              f"Policy={avg_policy_loss:.4f}, "
              f"Value={avg_value_loss:.4f}")
    
    # 모델 저장
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model, output_path)
    print(f"✅ 휴리스틱 사전 훈련된 모델 저장: {output_path}")
    
    return output_path

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Heuristic-based Pre-training")
    parser.add_argument("--games", type=int, default=500, help="Number of games to generate")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--output", type=str, default="models/heuristic_pretrained_model.pt", 
                       help="Output model path")
    
    args = parser.parse_args()
    
    pretrain_model_with_heuristics(
        output_path=args.output,
        num_games=args.games,
        epochs=args.epochs
    )

if __name__ == "__main__":
    main()