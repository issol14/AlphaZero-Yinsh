"""
Policy Head와 Value Head 구체적 출력 데모 (import 문제 해결)
"""

import numpy as np


def demonstrate_policy_value_heads():
    """Policy Head와 Value Head의 구체적인 출력 시뮬레이션"""
    print("🧠 Policy & Value Heads 구체적 출력")
    print("=" * 50)

    print("📍 Input: Game State → (11, 11, 11) tensor")
    print("   11×11 board, 11 channels (rings, markers, etc.)")

    # 가상의 신경망 출력 시뮬레이션
    print(f"\n🔄 Neural Network Forward Pass:")

    # Raw 출력
    raw_policy_logits = np.random.randn(1000)
    raw_value = np.random.randn()

    print(f"   Raw outputs from network:")
    print(
        f"   Policy logits: shape=(1000,), range=[{raw_policy_logits.min():.2f}, {raw_policy_logits.max():.2f}]"
    )
    print(f"   Value (raw): {raw_value:.6f}")

    # 1. Policy Head 처리
    print(f"\n🎯 POLICY HEAD → 1000 Action Probabilities:")

    # Softmax 적용
    exp_logits = np.exp(raw_policy_logits - np.max(raw_policy_logits))
    policy_probs = exp_logits / np.sum(exp_logits)

    print(f"   1. Apply softmax: exp(logits) / sum(exp(logits))")
    print(f"   2. Result: {policy_probs.shape} probabilities")
    print(f"   3. Sum: {policy_probs.sum():.6f} (should be 1.0)")
    print(f"   4. Range: [{policy_probs.min():.8f}, {policy_probs.max():.6f}]")

    # 상위 확률 보기
    top_indices = np.argsort(policy_probs)[-10:][::-1]
    print(f"\n   Top 10 action probabilities:")
    for i, idx in enumerate(top_indices):
        action_name = f"ACTION_{idx}"
        print(f"     {i+1:2d}. {action_name:<12}: {policy_probs[idx]:.6f}")

    # 2. Value Head 처리
    print(f"\n💎 VALUE HEAD → Single Value Estimate:")

    # Tanh 적용
    final_value = np.tanh(raw_value)

    print(f"   1. Apply tanh: tanh({raw_value:.6f})")
    print(f"   2. Result: {final_value:.6f}")
    print(f"   3. Range: [-1.0, +1.0]")
    print(f"   4. Meaning: {interpret_value(final_value)}")


def show_actual_output_format():
    """실제 출력 형태 구체적으로 보여주기"""
    print(f"\n📤 실제 출력 형태:")
    print("=" * 25)

    # Policy Head 출력 예시
    print("🎯 Policy Head Output:")
    print("   Type: numpy.ndarray or List[float]")
    print("   Shape: (1000,)")
    print("   Example:")

    sample_probs = np.random.dirichlet(np.ones(1000))  # 정확한 확률분포
    print(f"   [")
    for i in range(10):
        print(f"     {sample_probs[i]:.8f},  # Action {i}")
    print(f"     ...")
    print(f"     {sample_probs[-1]:.8f}   # Action 999")
    print(f"   ]")
    print(f"   Sum: {sample_probs.sum():.6f}")

    # Value Head 출력 예시
    print(f"\n💎 Value Head Output:")
    print("   Type: float")
    print("   Range: [-1.0, +1.0]")
    value_examples = [0.847, -0.234, 0.001, -0.891, 0.523]
    print("   Examples:")
    for val in value_examples:
        print(f"     {val:+.3f} → {interpret_value(val)}")


def demonstrate_conversion_pipeline():
    """변환 파이프라인 단계별 보여주기"""
    print(f"\n⚙️ 변환 파이프라인:")
    print("=" * 20)

    print("1️⃣ Neural Network Raw Output:")
    raw_logits = np.array([-2.1, 0.5, -0.8, 3.2, 1.1])  # 5개만 예시
    raw_value = 1.234

    print(f"   Policy logits: {raw_logits}")
    print(f"   Value (raw): {raw_value}")

    print(f"\n2️⃣ Apply Activation Functions:")

    # Softmax for policy
    exp_logits = np.exp(raw_logits - np.max(raw_logits))
    probs = exp_logits / np.sum(exp_logits)

    # Tanh for value
    value_final = np.tanh(raw_value)

    print(f"   Policy softmax: {probs}")
    print(f"   Value tanh: {value_final:.6f}")

    print(f"\n3️⃣ Game-Specific Processing:")
    print("   • Filter to legal actions only")
    print("   • Renormalize probabilities")
    print("   • Map to action strings")

    # 예시: 3개 legal actions만 유효
    legal_probs = probs[[1, 3, 4]]  # indices 1, 3, 4만 legal
    normalized = legal_probs / legal_probs.sum()

    actions = ["PLACE_RING:(5,5)", "MOVE_RING:(3,3)→(4,4)", "REMOVE_RING:(7,7)"]
    print(f"\n   Final action probabilities:")
    for action, prob in zip(actions, normalized):
        print(f"     {action:<20}: {prob:.6f}")
    print(f"   Sum: {normalized.sum():.6f}")

    print(f"\n   Final value: {value_final:.6f}")


def interpret_value(value):
    """Value 값 해석"""
    if value > 0.5:
        return "Strong advantage"
    elif value > 0.1:
        return "Slight advantage"
    elif value > -0.1:
        return "Equal position"
    elif value > -0.5:
        return "Slight disadvantage"
    else:
        return "Strong disadvantage"


def main():
    """메인 데모"""
    demonstrate_policy_value_heads()
    show_actual_output_format()
    demonstrate_conversion_pipeline()

    print(f"\n✅ Policy & Value Heads 데모 완료!")
    print(f"\n📚 핵심 요약:")
    print("• Policy Head: 1000개 확률값 → 액션 선택 분포")
    print("• Value Head: 1개 실수값 → 게임 결과 예측")
    print("• Raw logits → softmax/tanh → 게임 규칙 적용")
    print("• 실제 사용시 legal actions만 고려")


if __name__ == "__main__":
    main()
