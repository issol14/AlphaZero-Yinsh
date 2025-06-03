"""
Value Head 학습 목표와 과정 데모
"""

import numpy as np


def explain_value_head_learning_goal():
    """Value Head의 학습 목표 설명"""
    print("🎯 Value Head 학습 목표")
    print("=" * 30)

    print("❌ 잘못된 이해:")
    print("   'Value를 항상 1에 가깝게 만드는 것이 목표'")

    print(f"\n✅ 올바른 목표:")
    print("   '실제 게임 결과를 정확히 예측하는 것이 목표'")

    print(f"\n📊 Value Head 출력의 의미:")
    print("   +1.0: 현재 플레이어가 확실히 승리")
    print("   +0.5: 현재 플레이어가 유리한 상황")
    print("    0.0: 동등한 상황 또는 무승부")
    print("   -0.5: 현재 플레이어가 불리한 상황")
    print("   -1.0: 현재 플레이어가 확실히 패배")


def show_training_examples():
    """학습 데이터 예시"""
    print(f"\n📚 학습 데이터 예시")
    print("=" * 20)

    training_examples = [
        {
            "game_id": 1,
            "turn": 15,
            "player": "White",
            "predicted_value": 0.7,
            "actual_result": "White_Win",
            "target_value": +1.0,
            "loss": "정확한 예측 (좋음)",
        },
        {
            "game_id": 2,
            "turn": 8,
            "player": "Black",
            "predicted_value": 0.3,
            "actual_result": "White_Win",
            "target_value": -1.0,
            "loss": "예측 부정확 (나쁨)",
        },
        {
            "game_id": 3,
            "turn": 22,
            "player": "White",
            "predicted_value": -0.1,
            "actual_result": "Draw",
            "target_value": 0.0,
            "loss": "거의 정확한 예측",
        },
        {
            "game_id": 4,
            "turn": 5,
            "player": "Black",
            "predicted_value": -0.6,
            "actual_result": "Black_Win",
            "target_value": +1.0,
            "loss": "완전히 틀린 예측",
        },
    ]

    print("State → Prediction → Actual → Target → Loss")
    print("-" * 55)

    for ex in training_examples:
        print(
            f"Game{ex['game_id']}, T{ex['turn']:2d}, {ex['player'][:5]}: "
            f"{ex['predicted_value']:+.1f} → {ex['actual_result'][:5]} → "
            f"{ex['target_value']:+.1f} → {ex['loss']}"
        )


def demonstrate_loss_calculation():
    """Loss 계산 과정"""
    print(f"\n📐 Loss 계산 과정")
    print("=" * 20)

    print("Loss Function: MSE (Mean Squared Error)")
    print("Loss = (predicted_value - target_value)²")

    examples = [
        {"pred": 0.7, "target": 1.0, "result": "Win"},
        {"pred": 0.3, "target": -1.0, "result": "Lose"},
        {"pred": -0.1, "target": 0.0, "result": "Draw"},
        {"pred": -0.6, "target": 1.0, "result": "Win"},
    ]

    print(f"\n계산 예시:")
    total_loss = 0
    for i, ex in enumerate(examples, 1):
        loss = (ex["pred"] - ex["target"]) ** 2
        total_loss += loss
        print(f"  Example {i}: ({ex['pred']:+.1f} - {ex['target']:+.1f})² = {loss:.3f}")

    avg_loss = total_loss / len(examples)
    print(f"  Average Loss: {avg_loss:.3f}")

    print(f"\n🎯 학습 목표: Loss를 최소화")
    print("• 정확한 예측일수록 Loss 감소")
    print("• 틀린 예측일수록 Loss 증가")


def show_value_prediction_scenarios():
    """다양한 상황에서의 Value 예측"""
    print(f"\n🎮 다양한 게임 상황에서의 Value 예측")
    print("=" * 40)

    scenarios = [
        {
            "situation": "게임 시작 (초기 상태)",
            "description": "모든 링이 배치됨, 동등한 상태",
            "ideal_value": "~0.0 (균형잡힌 상태)",
            "reasoning": "어느 쪽도 유리하지 않음",
        },
        {
            "situation": "White가 4개 연속 마커 생성",
            "description": "한 번만 더 연결하면 승리",
            "ideal_value": "~+0.8 (White 관점)",
            "reasoning": "승리에 매우 가까움",
        },
        {
            "situation": "Black이 링 3개 제거 완료",
            "description": "승리 조건 달성",
            "ideal_value": "-1.0 (White 관점)",
            "reasoning": "Black이 이미 승리",
        },
        {
            "situation": "양쪽 모두 2개씩 링 제거",
            "description": "접전 상황",
            "ideal_value": "~0.0 ~ ±0.2",
            "reasoning": "약간의 위치적 우세 반영",
        },
        {
            "situation": "White 링이 구석에 갇힘",
            "description": "이동 옵션 제한됨",
            "ideal_value": "~-0.3 (White 관점)",
            "reasoning": "전략적 불리함",
        },
    ]

    for scenario in scenarios:
        print(f"📋 {scenario['situation']}")
        print(f"   상황: {scenario['description']}")
        print(f"   이상적 Value: {scenario['ideal_value']}")
        print(f"   이유: {scenario['reasoning']}")
        print()


def explain_learning_process():
    """학습 과정 설명"""
    print(f"🔄 Value Head 학습 과정")
    print("=" * 25)

    print("1️⃣ Self-Play 게임 진행:")
    print("   • 각 턴마다 현재 상태에서 Value 예측")
    print("   • 게임 끝까지 진행")
    print("   • 최종 결과 기록 (승/패/무)")

    print(f"\n2️⃣ 학습 데이터 생성:")
    print("   • 각 상태: (state, predicted_value)")
    print("   • 최종 결과를 바탕으로 target_value 설정")
    print("   • 승리한 플레이어 관점: +1.0")
    print("   • 패배한 플레이어 관점: -1.0")
    print("   • 무승부: 0.0")

    print(f"\n3️⃣ 신경망 업데이트:")
    print("   • Loss = Σ(predicted - target)²")
    print("   • Gradient Descent로 가중치 조정")
    print("   • 예측이 실제 결과에 가까워지도록")

    print(f"\n4️⃣ 반복 학습:")
    print("   • 더 많은 self-play 게임")
    print("   • 더 정확한 Value 예측")
    print("   • 더 강한 AI")


def show_good_vs_bad_predictions():
    """좋은 예측 vs 나쁜 예측"""
    print(f"\n✅❌ 좋은 예측 vs 나쁜 예측")
    print("=" * 30)

    print("✅ 좋은 Value Head:")
    print("   • 승리 확정 상황에서 +0.9 예측")
    print("   • 패배 확정 상황에서 -0.9 예측")
    print("   • 동등한 상황에서 0.0 근처 예측")
    print("   • 실제 게임 결과와 일치")

    print(f"\n❌ 나쁜 Value Head:")
    print("   • 승리 확정인데 -0.5 예측 (과소평가)")
    print("   • 패배 확정인데 +0.7 예측 (과대평가)")
    print("   • 항상 비슷한 값만 예측 (학습 부족)")
    print("   • 실제 결과와 큰 차이")

    print(f"\n🎯 목표:")
    print("   • 정확도 향상 (실제 결과와 일치)")
    print("   • 캘리브레이션 (확률적 해석 가능)")
    print("   • 일관성 (비슷한 상황에서 비슷한 예측)")


def main():
    """메인 데모"""
    explain_value_head_learning_goal()
    show_training_examples()
    demonstrate_loss_calculation()
    show_value_prediction_scenarios()
    explain_learning_process()
    show_good_vs_bad_predictions()

    print(f"\n🎓 핵심 정리:")
    print("• Value Head 목표: 실제 게임 결과 정확 예측")
    print("• +1/-1은 확실한 승/패 상황에서만")
    print("• 학습은 예측값과 실제결과 차이 최소화")
    print("• 좋은 AI = 정확한 상황 판단 능력")


if __name__ == "__main__":
    main()
