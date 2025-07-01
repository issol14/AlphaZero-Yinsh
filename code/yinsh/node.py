# node.py - YINSH MCTS Node

import numpy as np
from typing import Dict, Optional, List, Tuple
from .env import YinshEnv, YinshAction, Color


class YinshNode:
    """YINSH MCTS 노드"""

    def __init__(
        self,
        env: YinshEnv,
        parent: Optional["YinshNode"] = None,
        action: Optional[YinshAction] = None,
    ):
        """
        Args:
            env: 게임 환경
            parent: 부모 노드
            action: 이 노드로 이어지는 액션
        """
        self.env = env
        self.parent = parent
        self.action = action

        # MCTS 통계
        self.visit_count = 0
        self.value = None  # 평균 가치
        self.prior = 0.0  # 사전 확률

        # 자식 노드들
        self.children: Dict[YinshAction, YinshNode] = {}

        # 노드 상태
        self.is_terminal = env.is_game_over()
        self.is_expanded = False

    def add_child(self, action: YinshAction, child_env: YinshEnv) -> "YinshNode":
        """자식 노드 추가"""
        child = YinshNode(child_env, self, action)
        self.children[action] = child
        return child

    def get_ucb_value(self, c_puct: float) -> float:
        """UCB 값 계산"""
        if self.visit_count == 0:
            return float("inf")

        # Q 값 (평균 가치)
        q_value = self.value if self.value is not None else 0.0

        # UCB 항
        parent_visits = self.parent.visit_count if self.parent else 1
        ucb_term = c_puct * self.prior * np.sqrt(parent_visits) / (1 + self.visit_count)

        return q_value + ucb_term

    def select_child(self, c_puct: float) -> Tuple[YinshAction, "YinshNode"]:
        """UCB 기반으로 최적 자식 선택"""
        if not self.children:
            raise ValueError("No children to select from")

        best_ucb = float("-inf")
        best_action = None
        best_child = None

        for action, child in self.children.items():
            ucb = child.get_ucb_value(c_puct)
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, policy_probs: np.ndarray) -> None:
        """노드 확장"""
        if self.is_expanded or self.is_terminal:
            return

        # 유효한 액션들 가져오기
        valid_actions = self.env.get_valid_actions()
        if not valid_actions:
            return

        # 자식 노드들 생성
        for action in valid_actions:
            # 새 환경 생성
            new_env = self.env.copy()
            new_env.step(action)

            # 자식 노드 생성
            child = self.add_child(action, new_env)

            # 사전 확률 설정 (간단한 균등 분포)
            child.prior = 1.0 / len(valid_actions)

        self.is_expanded = True

    def backup(self, value: float) -> None:
        """가치 역전파"""
        current = self
        current_value = value

        while current is not None:
            current.visit_count += 1

            # 가치 업데이트
            if current.value is None:
                current.value = current_value
            else:
                # 방문 횟수 가중 평균
                current.value = (
                    current.value * (current.visit_count - 1) + current_value
                ) / current.visit_count

            # 다음 노드로 이동 (값은 반전)
            current = current.parent
            current_value = -current_value

    def get_action_probabilities(
        self, temperature: float = 1.0
    ) -> Dict[YinshAction, float]:
        """방문 횟수 기반 액션 확률 분포"""
        if not self.children:
            return {}

        actions = list(self.children.keys())
        visit_counts = np.array(
            [self.children[action].visit_count for action in actions]
        )

        if temperature == 0:
            # Greedy selection
            best_idx = np.argmax(visit_counts)
            probs = np.zeros(len(actions))
            probs[best_idx] = 1.0
        else:
            # Temperature scaling
            if temperature != 1.0:
                visit_counts = visit_counts ** (1.0 / temperature)

            # 정규화
            probs = visit_counts / visit_counts.sum()

        return {action: prob for action, prob in zip(actions, probs)}

    def get_best_action(self) -> Optional[YinshAction]:
        """가장 많이 방문된 액션 반환"""
        if not self.children:
            return None

        best_action = None
        best_visits = -1

        for action, child in self.children.items():
            if child.visit_count > best_visits:
                best_visits = child.visit_count
                best_action = action

        return best_action

    def get_state_hash(self) -> str:
        """상태 해시 반환 (캐싱용)"""
        return self.env.get_state_string()

    def is_leaf(self) -> bool:
        """리프 노드인지 확인"""
        return not self.children

    def get_depth(self) -> int:
        """노드 깊이 반환"""
        depth = 0
        current = self
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth

    def get_path_to_root(self) -> List[YinshAction]:
        """루트까지의 경로 반환"""
        path = []
        current = self
        while current.parent is not None:
            path.append(current.action)
            current = current.parent
        return list(reversed(path))

    def print_tree(self, max_depth: int = 3, current_depth: int = 0):
        """트리 구조 출력 (디버깅용)"""
        if current_depth > max_depth:
            return

        indent = "  " * current_depth
        action_str = str(self.action) if self.action else "ROOT"
        value_str = f"{self.value:.3f}" if self.value is not None else "None"

        print(f"{indent}{action_str} (N={self.visit_count}, V={value_str})")

        for action, child in self.children.items():
            child.print_tree(max_depth, current_depth + 1)

    def __str__(self):
        action_str = str(self.action) if self.action else "ROOT"
        value_str = f"{self.value:.3f}" if self.value is not None else "None"
        return f"YinshNode({action_str}, N={self.visit_count}, V={value_str})"

    def __repr__(self):
        return self.__str__()
