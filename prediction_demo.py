"""
Policy Head와 Value Head로 다음 상태 예측하는 방법 데모
"""

import numpy as np


def demonstrate_next_state_prediction():
    """Policy Head와 Value Head로 다음 상태를 예측하는 과정"""
    print("🔮 Policy & Value Head로 다음 상태 예측")
    print("=" * 50)

    print("📍 현재 상황:")
    print("   게임 상태: YINSH 링 이동 단계")
    print("   현재 플레이어: White")
    print("   보드에 링 5개, 마커 10개")

    # 1. 현재 상태에서 신경망 예측
    print(f"\n🧠 1단계: 현재 상태 → 신경망 예측")
    print("   Input: (11,11,11) 현재 상태 텐서")
    print("   ↓")

    # 가상의 신경망 출력
    print("   Neural Network Output:")

    # Policy Head: 가능한 액션들과 확률
    actions_probs = {
        "MOVE_RING:(3,3)→(5,5)": 0.35,
        "MOVE_RING:(7,7)→(8,8)": 0.28,
        "MOVE_RING:(4,6)→(4,9)": 0.18,
        "MOVE_RING:(2,8)→(2,5)": 0.12,
        "MOVE_RING:(9,4)→(6,4)": 0.07,
    }

    current_value = 0.23  # Value Head

    print(f"   🎯 Policy Head (액션 선택 확률):")
    for action, prob in actions_probs.items():
        print(f"     {action:<20}: {prob:.2f}")

    print(f"   💎 Value Head (현재 상태 가치): {current_value:.2f}")
    print("      → 현재 White가 약간 유리한 상황")


def show_action_selection_process():
    """액션 선택 과정 상세히 보여주기"""
    print(f"\n🎯 2단계: 액션 선택 → 다음 상태 예측")
    print("=" * 40)

    # 각 액션별로 다음 상태가 어떻게 될지 예측
    actions = [
        {
            "action": "MOVE_RING:(3,3)→(5,5)",
            "prob": 0.35,
            "next_state_description": "링이 (5,5)로 이동, 경로에 마커 생성",
            "predicted_outcome": "상대방 차례, 마커 연결 가능성 증가",
        },
        {
            "action": "MOVE_RING:(7,7)→(8,8)",
            "prob": 0.28,
            "next_state_description": "링이 (8,8)로 이동, 코너 공격 위치",
            "predicted_outcome": "방어적 위치, 안전한 선택",
        },
        {
            "action": "MOVE_RING:(4,6)→(4,9)",
            "prob": 0.18,
            "next_state_description": "세로 라인 공격, 긴 거리 이동",
            "predicted_outcome": "공격적 선택, 위험도 중간",
        },
    ]

    print("각 액션의 다음 상태 예측:")
    print()

    for i, action_info in enumerate(actions):
        print(f"📋 Option {i+1}: {action_info['action']}")
        print(f"   확률: {action_info['prob']:.2f} (Policy Head)")
        print(f"   다음 상태: {action_info['next_state_description']}")
        print(f"   예상 결과: {action_info['predicted_outcome']}")
        print()


def demonstrate_mcts_usage():
    """MCTS에서 어떻게 사용되는지"""
    print(f"🌳 3단계: MCTS에서 활용")
    print("=" * 25)

    print("MCTS 각 단계에서의 활용:")
    print()

    print("1️⃣ Selection (선택):")
    print("   Policy Head → 탐색할 액션 우선순위 결정")
    print("   높은 확률의 액션을 더 자주 탐색")
    print("   예: MOVE_RING:(3,3)→(5,5) (35%) 먼저 탐색")

    print(f"\n2️⃣ Expansion (확장):")
    print("   선택된 액션으로 새로운 노드 생성")
    print("   실제로 다음 상태 계산:")
    print("   • 링 위치 업데이트")
    print("   • 마커 배치")
    print("   • 플레이어 교체")

    print(f"\n3️⃣ Evaluation (평가):")
    print("   새로운 상태에서 다시 신경망 실행")
    print("   Value Head → 새 상태의 가치 예측")
    print("   예: 다음 상태 가치 = -0.15 (상대방 유리)")

    print(f"\n4️⃣ Backpropagation (역전파):")
    print("   예측된 가치를 부모 노드들로 전파")
    print("   여러 시뮬레이션 결과 누적")


def show_concrete_example():
    """구체적인 예시"""
    print(f"\n🎮 구체적인 예시")
    print("=" * 20)

    print("현재 상태 A:")
    print("┌─────────────┐")
    print("│ ○   ●   ○   │  ○=White링, ●=Black링")
    print("│   •   •   • │  •=마커")
    print("│ ●   ○   ●   │")
    print("└─────────────┘")

    print(f"\n🧠 신경망 예측:")
    print("Policy Head → 'MOVE_RING:(2,2)→(4,2)' 확률 = 0.45")
    print("Value Head → 현재 상태 가치 = +0.12")

    print(f"\n➡️  액션 실행 후 다음 상태 B:")
    print("┌─────────────┐")
    print("│ ○   ●     ○ │  링이 이동됨")
    print("│   • • • • • │  경로에 마커 추가")
    print("│ ●   ○   ●   │")
    print("└─────────────┘")

    print(f"\n🔄 다음 상태에서 다시 예측:")
    print("Policy Head → Black의 액션 확률들")
    print("Value Head → 새 상태 가치 = -0.08 (White 관점)")

    print(f"\n✨ 핵심:")
    print("• Policy Head: '어떤 액션을 할지' 확률")
    print("• Value Head: '현재 상황이 얼마나 좋은지' 평가")
    print("• 액션 실행 → 실제 다음 상태 생성")
    print("• 다음 상태에서 다시 신경망 예측 반복")


def explain_prediction_vs_evaluation():
    """예측 vs 평가의 차이점"""
    print(f"\n🔍 예측 vs 평가 구분")
    print("=" * 25)

    print("❌ 흔한 오해:")
    print("   '신경망이 다음 상태를 직접 예측한다'")

    print(f"\n✅ 실제 동작:")
    print("1. Policy Head: 어떤 액션을 선택할 확률 제공")
    print("2. 게임 규칙 엔진: 선택된 액션으로 다음 상태 계산")
    print("3. Value Head: 현재 상태의 승률 평가")

    print(f"\n📊 단계별 흐름:")
    print("현재 상태 → 신경망 → 액션 확률 + 상태 가치")
    print("           ↓")
    print("        액션 선택")
    print("           ↓")
    print("        게임 규칙 적용")
    print("           ↓")
    print("        다음 상태 생성")
    print("           ↓")
    print("        다음 상태 → 신경망 → ...")


def main():
    """메인 데모"""
    demonstrate_next_state_prediction()
    show_action_selection_process()
    demonstrate_mcts_usage()
    show_concrete_example()
    explain_prediction_vs_evaluation()

    print(f"\n✅ 요약:")
    print("🎯 Policy Head → 어떤 액션을 선택할지 가이드")
    print("💎 Value Head → 현재 상태가 얼마나 좋은지 평가")
    print("⚙️  게임 엔진 → 실제 다음 상태 계산")
    print("🔄 반복 → 다음 상태에서 다시 신경망 예측")


if __name__ == "__main__":
    main()
