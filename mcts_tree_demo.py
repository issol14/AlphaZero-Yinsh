"""
MCTS 노드 재사용과 트리 확장 데모
"""

import numpy as np


def demonstrate_mcts_tree_growth():
    """MCTS 트리가 어떻게 성장하고 노드가 재사용되는지"""
    print("🌳 MCTS 트리 성장과 노드 재사용")
    print("=" * 40)

    print("📍 시작 상태:")
    print("   Root 노드 (현재 게임 상태)")
    print("   visits: 0, value: 0")

    print(f"\n🔄 Simulation #1:")
    print("1. Policy Head: [0.4, 0.3, 0.2, 0.1] (4개 액션)")
    print("2. 가장 높은 확률 액션 A 선택")
    print("3. 새 노드 A 생성")
    print("4. Value Head: A의 가치 = +0.2")
    print("5. 백업: Root(visits:1, value:+0.2)")

    print("   트리 상태:")
    print("   Root(1) → A(1)")

    print(f"\n🔄 Simulation #2:")
    print("1. 같은 Policy Head: [0.4, 0.3, 0.2, 0.1]")
    print("2. 다시 액션 A 선택 (높은 확률)")
    print("3. 🔄 기존 노드 A 재사용!")
    print("4. A에서 Policy Head 다시 실행")
    print("5. A의 자식 노드 B 생성")
    print("6. Value Head: B의 가치 = -0.1")
    print("7. 백업: A(2, -0.1) → Root(2, +0.05)")

    print("   트리 상태:")
    print("   Root(2) → A(2) → B(1)")


def show_node_reuse_process():
    """노드 재사용 과정 상세히"""
    print(f"\n♻️  노드 재사용 과정")
    print("=" * 25)

    print("💾 메모리에 저장된 노드들:")
    print("   Root: state_hash=12345, visits=0")
    print("   A: state_hash=23456, visits=0")
    print("   B: state_hash=34567, visits=0")

    print(f"\n🔍 새 시뮬레이션에서:")
    print("1. 현재 노드에서 액션 선택")
    print("2. 다음 상태 계산")
    print("3. 상태 해시 생성: hash(다음상태)")
    print("4. 메모리에서 해시 검색:")

    print(f"\n   Case 1: 해시 발견 ✅")
    print("   → 기존 노드 재사용")
    print("   → visits, value 업데이트")

    print(f"\n   Case 2: 해시 없음 🆕")
    print("   → 새 노드 생성")
    print("   → Policy Head로 prior 설정")
    print("   → 메모리에 추가")


def demonstrate_tree_structure():
    """실제 트리 구조 예시"""
    print(f"\n🌲 MCTS 트리 구조 예시")
    print("=" * 30)

    print("Simulation 진행에 따른 트리 확장:")

    print(f"\n📊 After 1 simulation:")
    print("Root(1)")
    print(" └─ A(1) [MOVE_RING:(3,3)→(5,5)]")

    print(f"\n📊 After 5 simulations:")
    print("Root(5)")
    print(" ├─ A(3) [MOVE_RING:(3,3)→(5,5)]")
    print(" │   ├─ A1(1)")
    print(" │   └─ A2(1)")
    print(" ├─ B(1) [MOVE_RING:(7,7)→(8,8)]")
    print(" └─ C(1) [MOVE_RING:(4,6)→(4,9)]")

    print(f"\n📊 After 20 simulations:")
    print("Root(20)")
    print(" ├─ A(12) [prob=0.4, visits=12]")
    print(" │   ├─ A1(5)")
    print(" │   │   ├─ A1a(2)")
    print(" │   │   └─ A1b(3)")
    print(" │   ├─ A2(4)")
    print(" │   └─ A3(3)")
    print(" ├─ B(5) [prob=0.3, visits=5]")
    print(" │   ├─ B1(2)")
    print(" │   └─ B2(3)")
    print(" ├─ C(2) [prob=0.2, visits=2]")
    print(" └─ D(1) [prob=0.1, visits=1]")


def show_selection_with_reuse():
    """노드 재사용과 함께 selection 과정"""
    print(f"\n🎯 Selection with Node Reuse")
    print("=" * 35)

    print("PUCT 점수로 노드 선택:")

    print(f"\n🌟 Simulation #21:")
    print("1. Root에서 시작")
    print("   A: PUCT = 0.45 (visits=12, prior=0.4)")
    print("   B: PUCT = 0.52 (visits=5, prior=0.3)  ← 선택!")
    print("   C: PUCT = 0.48 (visits=2, prior=0.2)")
    print("   D: PUCT = 0.61 (visits=1, prior=0.1)")

    print(f"\n2. B 노드로 이동 (재사용)")
    print("   B1: PUCT = 0.23 (visits=2)")
    print("   B2: PUCT = 0.31 (visits=3)  ← 선택!")

    print(f"\n3. B2 노드로 이동 (재사용)")
    print("   아직 자식 없음 → 확장 단계")
    print("   Policy Head 실행 → 새 자식들 생성")


def explain_memory_efficiency():
    """메모리 효율성 설명"""
    print(f"\n💾 메모리 효율성")
    print("=" * 20)

    print("✅ 노드 재사용의 장점:")
    print("• 같은 상태 중복 계산 방지")
    print("• 누적된 통계 정보 활용")
    print("• 메모리 사용량 최적화")
    print("• 탐색 효율성 증가")

    print(f"\n🔢 예시 계산:")
    print("• 50 시뮬레이션, 평균 깊이 10")
    print("• 재사용 없으면: 500개 노드")
    print("• 재사용 있으면: ~150개 노드")
    print("• 메모리 절약: 70%")

    print(f"\n🗑️  가지치기 (Pruning):")
    print("• 게임 진행시 사용하지 않는 브랜치 제거")
    print("• 메모리 확보")
    print("• 새로운 탐색 공간 확보")


def show_actual_implementation():
    """실제 구현에서의 동작"""
    print(f"\n⚙️ 실제 구현에서의 동작")
    print("=" * 30)

    print("Python 의사코드:")
    print(
        """
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}  # action -> child_node
        self.visits = 0
        self.value_sum = 0
        
class MCTS:
    def __init__(self):
        self.node_cache = {}  # state_hash -> node
        
    def get_or_create_node(self, state):
        hash_key = state.get_hash()
        if hash_key in self.node_cache:
            return self.node_cache[hash_key]  # 재사용!
        else:
            node = MCTSNode(state)
            self.node_cache[hash_key] = node
            return node
"""
    )

    print(f"\n🔑 핵심:")
    print("• state.get_hash(): 상태를 고유 식별자로 변환")
    print("• node_cache: 해시 → 노드 매핑 딕셔너리")
    print("• 같은 해시 = 같은 상태 = 노드 재사용")


def main():
    """메인 데모"""
    demonstrate_mcts_tree_growth()
    show_node_reuse_process()
    demonstrate_tree_structure()
    show_selection_with_reuse()
    explain_memory_efficiency()
    show_actual_implementation()

    print(f"\n✅ 핵심 정리:")
    print("🌳 MCTS 트리는 시뮬레이션마다 성장")
    print("♻️  같은 상태의 노드는 재사용")
    print("📊 visits, value 등 통계 누적")
    print("🎯 Policy Head가 확장 방향 가이드")
    print("💾 메모리 효율적인 탐색")


if __name__ == "__main__":
    main()
