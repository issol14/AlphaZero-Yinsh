"""
Policy Head와 Value Head 구체적 출력 데모
"""

from .game_state import GameState, Ring
from .neural_agent import NeuralAgent
import numpy as np


def show_policy_value_output():
    """실제 Policy Head와 Value Head 출력 보여주기"""
    print("🧠 Policy & Value Heads 구체적 출력")
    print("=" * 50)

    # 게임 상태 생성
    state = GameState()
    state.rings.append(Ring((5, 5), "white"))

    print(f"📍 Game State:")
    print(f"   Phase: {state.phase.name}")
    print(f"   Current player: {state.current_player}")

    # 상태 텐서 확인
    tensor = state.to_tensor()
    print(f"\n📊 Input: State Tensor {tensor.shape}")

    # 신경망 에이전트 예측
    agent = NeuralAgent(use_neural_network=False)
    action_probs, value = agent.predict(state)

    print(f"\n🎯 POLICY HEAD → 1000 Action Probabilities:")
    print(f"   Type: Dictionary[str, float]")
    print(f"   Total actions: {len(action_probs)}")
    print(f"   Sum of probabilities: {sum(action_probs.values()):.6f}")

    # 상위 확률 액션들
    sorted_actions = sorted(action_probs.items(), key=lambda x: x[1], reverse=True)
    print(f"   Top 8 actions:")
    for i, (action, prob) in enumerate(sorted_actions[:8]):
        print(f"     {i+1:2d}. {action[:35]:<35}: {prob:.6f}")

    print(f"\n💎 VALUE HEAD → Single Value Estimate:")
    print(f"   Type: float")
    print(f"   Value: {value:.6f}")
    print(f"   Range: [-1.0, +1.0]")
    print(f"   Meaning: {interpret_value(value)}")


def show_neural_structure():
    """신경망 구조와 출력 형태"""
    print(f"\n🏗️ Neural Network Structure")
    print("=" * 30)

    print("📥 INPUT:")
    print("   (11, 11, 11) tensor")
    print("   ↓")
    print("🔄 BACKBONE (ResNet):")
    print("   Conv layers + ResBlocks")
    print("   ↓")
    print("🎯 POLICY HEAD:")
    print("   Conv2D → Flatten → Linear")
    print("   Output: 1000 logits")
    print("   Softmax: → 1000 probabilities")
    print("   📤 [0.001, 0.003, 0.001, ..., 0.002]")
    print("       ↑ 1000개 확률값, 합계=1.0")
    print("   ↓")
    print("💎 VALUE HEAD:")
    print("   Conv2D → Flatten → Linear → Linear")
    print("   Tanh: → single value [-1, +1]")
    print("   📤 0.347")
    print("       ↑ 현재 플레이어 승률 예측")


def demonstrate_conversion():
    """Raw → Final 변환 과정"""
    print(f"\n⚙️ Raw Output → Final Output")
    print("=" * 35)

    # 가상의 raw 출력
    print("🔢 1. Raw Neural Network Output:")
    raw_logits = np.random.randn(1000)
    raw_value = np.random.randn()

    print(f"   Policy logits: {raw_logits.shape} array")
    print(f"   Values: [{raw_logits.min():.2f}, {raw_logits.max():.2f}]")
    print(f"   Example: {raw_logits[:5]}")
    print(f"   Raw value: {raw_value:.6f}")

    print(f"\n🔄 2. Apply Softmax & Tanh:")
    # Softmax
    exp_logits = np.exp(raw_logits - np.max(raw_logits))  # numerical stability
    policy_probs = exp_logits / np.sum(exp_logits)

    # Tanh
    final_value = np.tanh(raw_value)

    print(f"   Policy: softmax(logits)")
    print(f"   → {policy_probs.shape} probabilities")
    print(f"   → Sum: {policy_probs.sum():.6f}")
    print(f"   → Range: [{policy_probs.min():.8f}, {policy_probs.max():.6f}]")

    print(f"   Value: tanh({raw_value:.3f}) = {final_value:.6f}")

    print(f"\n🎮 3. Apply Game Rules:")
    print("   Filter to legal actions only")
    print("   Renormalize probabilities")

    # 예시: 7개 legal actions
    legal_indices = np.random.choice(1000, 7, replace=False)
    legal_probs = policy_probs[legal_indices]
    normalized_probs = legal_probs / legal_probs.sum()

    print(f"   Legal actions: 7 out of 1000")
    print(f"   Renormalized probabilities:")
    for i, prob in enumerate(normalized_probs):
        print(f"     Action {i+1}: {prob:.6f}")
    print(f"   New sum: {normalized_probs.sum():.6f}")


def interpret_value(value):
    """Value 해석"""
    if value > 0.5:
        return "Strong advantage for current player"
    elif value > 0.1:
        return "Slight advantage for current player"
    elif value > -0.1:
        return "Roughly equal position"
    elif value > -0.5:
        return "Slight disadvantage for current player"
    else:
        return "Strong disadvantage for current player"


def main():
    """메인 데모"""
    show_policy_value_output()
    show_neural_structure()
    demonstrate_conversion()

    print(f"\n✅ 완료!")
    print(f"\n📚 요약:")
    print("• Policy Head: 1000개 확률값 배열 (액션 선택 분포)")
    print("• Value Head: 1개 실수값 (게임 결과 예측)")
    print("• 신경망 → raw logits → softmax/tanh → 게임 규칙 적용")


if __name__ == "__main__":
    main()
