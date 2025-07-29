#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified YINSH Environment Test
=================================

간소화된 YINSH 환경을 테스트하는 스크립트
"""

import os
import sys
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yinsh import YinshEnv, YinshAction, YinshModel, Color, config


def test_environment():
    """간소화된 환경 테스트"""
    print("🧪 Testing Simplified YINSH Environment")
    print("=" * 50)

    # 환경 생성
    env = YinshEnv()
    print("✅ Environment created")

    # 초기 상태 확인
    print(f"\n📋 Initial State:")
    print(f"   ├── Current Player: {env.current_player}")
    print(f"   ├── White Rings: {len(env.ring_positions[Color.WHITE])}")
    print(f"   ├── Black Rings: {len(env.ring_positions[Color.BLACK])}")
    print(f"   └── Game Over: {env.is_game_over()}")

    # 초기 링 위치 출력
    print(f"\n📍 Initial Ring Positions:")
    print(f"   ├── White Rings: {list(env.ring_positions[Color.WHITE])}")
    print(f"   └── Black Rings: {list(env.ring_positions[Color.BLACK])}")

    # 유효한 액션 확인
    valid_actions = env.get_valid_actions()
    print(f"\n🎯 Valid Actions: {len(valid_actions)}")
    
    if len(valid_actions) > 0:
        print("   ├── Sample actions:")
        for i, action in enumerate(valid_actions[:5]):
            print(f"   │   {i+1}. {action}")
        if len(valid_actions) > 5:
            print(f"   └── ... and {len(valid_actions)-5} more")
    else:
        print("   └── No valid actions!")

    # 상태 텐서 확인
    state_tensor = env.get_state_tensor()
    print(f"\n🧠 State Tensor Shape: {state_tensor.shape}")
    print(f"   ├── Channel 0 (White Rings): {np.sum(state_tensor[0])} active")
    print(f"   ├── Channel 1 (Black Rings): {np.sum(state_tensor[1])} active")
    print(f"   └── Channel 2 (Markers): {np.sum(state_tensor[2])} active")

    return env


def test_game_play(env):
    """게임 플레이 테스트"""
    print(f"\n🎮 Testing Game Play")
    print("=" * 50)

    turn_count = 0
    max_turns = 10

    while not env.is_game_over() and turn_count < max_turns:
        print(f"\n🔄 Turn {turn_count + 1}")
        print(f"   ├── Current Player: {env.current_player}")
        
        # 유효한 액션 가져오기
        valid_actions = env.get_valid_actions()
        
        if len(valid_actions) == 0:
            print("   └── No valid actions! Game might be stuck.")
            break
        
        # 첫 번째 유효한 액션 선택
        action = valid_actions[0]
        print(f"   ├── Selected Action: {action}")
        
        # 액션 실행
        success = env.step(action)
        print(f"   ├── Action Success: {success}")
        print(f"   ├── White Rings: {len(env.ring_positions[Color.WHITE])}")
        print(f"   ├── Black Rings: {len(env.ring_positions[Color.BLACK])}")
        print(f"   └── Game Over: {env.is_game_over()}")
        
        turn_count += 1

    # 게임 결과
    if env.is_game_over():
        winner = env.get_winner()
        print(f"\n🏆 Game Over!")
        print(f"   ├── Winner: {winner}")
        print(f"   ├── Total Turns: {turn_count}")
        print(f"   ├── White Rings Remaining: {len(env.ring_positions[Color.WHITE])}")
        print(f"   └── Black Rings Remaining: {len(env.ring_positions[Color.BLACK])}")
    else:
        print(f"\n⏰ Game stopped after {max_turns} turns")


def test_model():
    """간소화된 모델 테스트"""
    print(f"\n🧠 Testing Simplified Model")
    print("=" * 50)

    # 모델 생성
    model = YinshModel()
    print("✅ Model created")

    # 모델 요약
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Model Summary:")
    print(f"   ├── Total Parameters: {total_params:,}")
    print(f"   ├── Input Shape: {config.INPUT_SHAPE}")
    print(f"   └── Policy Output Size: {config.POLICY_OUTPUT_SIZE}")

    # 더미 입력으로 테스트
    dummy_input = np.random.randn(1, 3, 11, 11).astype(np.float32)
    print(f"\n🧪 Testing with dummy input: {dummy_input.shape}")
    
    try:
        policy, value = model.predict(dummy_input)
        print(f"✅ Model prediction successful:")
        print(f"   ├── Policy shape: {policy.shape}")
        print(f"   ├── Policy sum: {np.sum(policy):.4f}")
        print(f"   └── Value: {value:.4f}")
    except Exception as e:
        print(f"❌ Model prediction failed: {e}")


def test_action_mapping():
    """액션 매핑 테스트"""
    print(f"\n🔗 Testing Action Mapping")
    print("=" * 50)

    from yinsh.mapper import get_action_mapper

    mapper = get_action_mapper()
    stats = mapper.get_mapping_stats()
    
    print(f"📊 Mapping Stats:")
    print(f"   ├── Total Actions: {stats['total_actions']}")
    print(f"   ├── Policy Output Size: {stats['policy_output_size']}")
    print(f"   ├── Coverage: {stats['coverage']:.2%}")
    print(f"   └── Action Type: {stats['action_type']}")

    # 몇 개의 액션 테스트
    test_actions = [
        YinshAction(from_pos=(0, 0), to_pos=(1, 1)),
        YinshAction(from_pos=(5, 5), to_pos=(6, 6)),
        YinshAction(from_pos=(10, 10), to_pos=(9, 9)),
    ]
    
    print(f"\n🧪 Action Mapping Test:")
    for action in test_actions:
        index = mapper.get_action_index(action)
        if index is not None:
            reconstructed = mapper.get_index_action(index)
            print(f"   ✅ {action} -> {index} -> {reconstructed}")
        else:
            print(f"   ❌ {action} not found in mapping")


def main():
    """메인 테스트 함수"""
    print("🚀 Simplified YINSH Environment Test Suite")
    print("=" * 60)

    try:
        # 환경 테스트
        env = test_environment()
        
        # 게임 플레이 테스트
        test_game_play(env)
        
        # 모델 테스트
        test_model()
        
        # 액션 매핑 테스트
        test_action_mapping()
        
        print(f"\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 