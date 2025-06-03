"""
MCTS Backpropagation 흐름 데모
"""

import numpy as np


def demonstrate_mcts_backpropagation():
    """MCTS Backpropagation의 구체적인 흐름"""
    print("🔄 MCTS Backpropagation 흐름")
    print("=" * 40)

    print("📍 시뮬레이션 시작 상태:")
    print("Root(visits=10, value_sum=2.3)")
    print(" └─ A(visits=6, value_sum=1.8)")
    print("     └─ A1(visits=3, value_sum=0.9)")
    print("         └─ A1a(visits=1, value_sum=0.2) ← 현재 위치")

    print(f"\n🧠 1단계: Neural Network Evaluation")
    print("   A1a 상태에서 신경망 실행")
    print("   Value Head 출력: +0.7")
    print("   (현재 플레이어에게 유리한 상황)")

    print(f"\n⬆️  2단계: Backpropagation 시작")
    print("   평가된 value(+0.7)을 부모 노드들로 전파")


def show_step_by_step_backprop():
    """단계별 backpropagation 과정"""
    print(f"\n📊 단계별 Backpropagation")
    print("=" * 30)

    # 초기 상태
    nodes = {
        "A1a": {"visits": 1, "value_sum": 0.2, "avg": 0.2},
        "A1": {"visits": 3, "value_sum": 0.9, "avg": 0.3},
        "A": {"visits": 6, "value_sum": 1.8, "avg": 0.3},
        "Root": {"visits": 10, "value_sum": 2.3, "avg": 0.23},
    }

    new_value = 0.7

    print(f"🔢 새로운 평가값: {new_value}")
    print(f"전파 경로: A1a → A1 → A → Root")

    print(f"\n단계별 업데이트:")

    # A1a 업데이트
    nodes["A1a"]["visits"] += 1
    nodes["A1a"]["value_sum"] += new_value
    nodes["A1a"]["avg"] = nodes["A1a"]["value_sum"] / nodes["A1a"]["visits"]

    print(f"1️⃣ A1a 업데이트:")
    print(f"   visits: 1 → {nodes['A1a']['visits']}")
    print(f"   value_sum: 0.2 → {nodes['A1a']['value_sum']:.1f}")
    print(f"   average: 0.2 → {nodes['A1a']['avg']:.3f}")

    # A1 업데이트 (자식 관점에서 부모 관점으로 전환)
    parent_value = -new_value  # 플레이어 교체
    nodes["A1"]["visits"] += 1
    nodes["A1"]["value_sum"] += parent_value
    nodes["A1"]["avg"] = nodes["A1"]["value_sum"] / nodes["A1"]["visits"]

    print(f"\n2️⃣ A1 업데이트 (플레이어 전환):")
    print(f"   value 전환: +0.7 → -0.7 (상대방 관점)")
    print(f"   visits: 3 → {nodes['A1']['visits']}")
    print(f"   value_sum: 0.9 → {nodes['A1']['value_sum']:.1f}")
    print(f"   average: 0.3 → {nodes['A1']['avg']:.3f}")

    # A 업데이트
    grandparent_value = -parent_value  # 다시 플레이어 교체
    nodes["A"]["visits"] += 1
    nodes["A"]["value_sum"] += grandparent_value
    nodes["A"]["avg"] = nodes["A"]["value_sum"] / nodes["A"]["visits"]

    print(f"\n3️⃣ A 업데이트 (플레이어 재전환):")
    print(f"   value 재전환: -0.7 → +0.7")
    print(f"   visits: 6 → {nodes['A']['visits']}")
    print(f"   value_sum: 1.8 → {nodes['A']['value_sum']:.1f}")
    print(f"   average: 0.3 → {nodes['A']['avg']:.3f}")

    # Root 업데이트
    root_value = -grandparent_value
    nodes["Root"]["visits"] += 1
    nodes["Root"]["value_sum"] += root_value
    nodes["Root"]["avg"] = nodes["Root"]["value_sum"] / nodes["Root"]["visits"]

    print(f"\n4️⃣ Root 업데이트:")
    print(f"   value 전환: +0.7 → -0.7")
    print(f"   visits: 10 → {nodes['Root']['visits']}")
    print(f"   value_sum: 2.3 → {nodes['Root']['value_sum']:.1f}")
    print(f"   average: 0.23 → {nodes['Root']['avg']:.3f}")


def explain_value_alternation():
    """Value 부호 교체 설명"""
    print(f"\n🔄 Value 부호 교체 원리")
    print("=" * 25)

    print("핵심 원리: 플레이어 관점 전환")
    print("• 자식 노드: 현재 플레이어 관점")
    print("• 부모 노드: 상대 플레이어 관점")
    print("• 따라서 value 부호가 반대")

    print(f"\n예시:")
    print("A1a (White 차례): value = +0.7 (White 유리)")
    print("A1  (Black 차례): value = -0.7 (Black 불리 = White 유리)")
    print("A   (White 차례): value = +0.7 (White 유리)")
    print("Root(Black 차례): value = -0.7 (Black 불리 = White 유리)")

    print(f"\n🎯 결과:")
    print("모든 노드가 동일한 상황을 다른 관점에서 평가")


def show_multiple_simulations():
    """여러 시뮬레이션에서의 누적 효과"""
    print(f"\n📈 여러 시뮬레이션 누적 효과")
    print("=" * 30)

    print("Root 노드의 변화 과정:")

    simulations = [
        {"sim": 1, "path": "Root→A→A1→A1a", "value": 0.7, "visits": 11, "avg": 0.182},
        {"sim": 2, "path": "Root→A→A1→A1b", "value": -0.3, "visits": 12, "avg": 0.158},
        {"sim": 3, "path": "Root→A→A2", "value": 0.1, "visits": 13, "avg": 0.150},
        {"sim": 4, "path": "Root→B", "value": -0.8, "visits": 14, "avg": 0.093},
        {"sim": 5, "path": "Root→A→A1→A1a", "value": 0.4, "visits": 15, "avg": 0.099},
    ]

    print("Sim | Path              | Value | Visits | Avg Value")
    print("-" * 50)
    for sim in simulations:
        print(
            f" {sim['sim']:2d} | {sim['path']:<17} | {sim['value']:+.1f}  | {sim['visits']:6d} | {sim['avg']:8.3f}"
        )

    print(f"\n📊 관찰사항:")
    print("• 좋은 평가(+)가 많으면 평균 상승")
    print("• 나쁜 평가(-)가 많으면 평균 하락")
    print("• 더 많은 시뮬레이션 = 더 정확한 평가")


def demonstrate_model_mcts_loop():
    """모델 ↔ MCTS 반복 루프"""
    print(f"\n🔁 모델 ↔ MCTS 반복 루프")
    print("=" * 30)

    print("전체 흐름:")

    for i in range(1, 4):
        print(f"\n🔄 Simulation #{i}:")
        print("1️⃣ Selection: PUCT로 최적 경로 선택")
        print("2️⃣ Expansion: 새 노드 생성 (필요시)")
        print("3️⃣ Evaluation: 🧠 Neural Network 호출")
        print("   └─ Input: 현재 상태 tensor")
        print("   └─ Output: Policy + Value")
        print("4️⃣ Backpropagation: Value를 상위 노드로 전파")
        print("   └─ 각 노드의 visits++, value_sum 업데이트")

    print(f"\n⚡ 핵심 포인트:")
    print("• 각 시뮬레이션마다 Neural Network 1회 호출")
    print("• Backpropagation은 트리 내에서만 발생")
    print("• Neural Network 가중치는 변하지 않음")
    print("• 시뮬레이션 완료 후 action 선택")


def explain_two_types_of_backprop():
    """두 종류의 backpropagation 구분"""
    print(f"\n🧠 두 종류의 Backpropagation")
    print("=" * 35)

    print("1️⃣ MCTS Backpropagation (실시간):")
    print("   • 시뮬레이션 중 즉시 발생")
    print("   • 트리 노드의 통계 업데이트")
    print("   • visits, value_sum 변경")
    print("   • Neural Network 가중치 불변")

    print(f"\n2️⃣ Neural Network Backpropagation (학습):")
    print("   • 게임 완료 후 발생")
    print("   • 실제 게임 결과로 학습")
    print("   • Gradient descent")
    print("   • Neural Network 가중치 변경")

    print(f"\n🔄 시간 순서:")
    print("   게임 중: MCTS Backprop 반복")
    print("   ↓")
    print("   게임 종료: 결과 기록")
    print("   ↓")
    print("   나중에: NN Backprop으로 학습")


def show_practical_example():
    """실제 예시"""
    print(f"\n🎮 실제 게임 예시")
    print("=" * 20)

    print("YINSH 게임 턴 15:")
    print("Root 상태: White 차례, 링 4개씩")

    print(f"\n🧠 Neural Network 예측:")
    print("Value Head: +0.25 (White 약간 유리)")

    print(f"\n🌳 MCTS 50회 시뮬레이션:")
    print("Sim 1: +0.25 → Root(avg: +0.25)")
    print("Sim 2: -0.15 → Root(avg: +0.05)")
    print("Sim 3: +0.40 → Root(avg: +0.17)")
    print("...")
    print("Sim 50: +0.10 → Root(avg: +0.12)")

    print(f"\n✅ 최종 결과:")
    print("• Root의 평균 value: +0.12")
    print("• 가장 많이 방문된 액션 선택")
    print("• 다음 상태로 이동")
    print("• 새로운 Root에서 다시 MCTS 시작")


def main():
    """메인 데모"""
    demonstrate_mcts_backpropagation()
    show_step_by_step_backprop()
    explain_value_alternation()
    show_multiple_simulations()
    demonstrate_model_mcts_loop()
    explain_two_types_of_backprop()
    show_practical_example()

    print(f"\n🎓 핵심 정리:")
    print("• MCTS Backprop: 트리 내 즉시 통계 업데이트")
    print("• NN Backprop: 게임 후 가중치 학습")
    print("• Value 부호 교체: 플레이어 관점 전환")
    print("• 누적 효과: 더 정확한 상황 판단")


if __name__ == "__main__":
    main()
