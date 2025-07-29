# node_optimized.py - AlphaZero 논문 기반 최적화된 MCTS 노드

import numpy as np
from typing import Dict, Optional, List, Tuple
from .env import YinshEnv, YinshAction, Color


class YinshNode:
    """AlphaZero 논문 기반 최적화된 YINSH MCTS 노드"""

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

        # AlphaZero MCTS 통계
        self.visit_count = 0
        self.value_sum = 0.0  # 누적 가치 (Q값 계산용)
        self.prior = 0.0      # 신경망 정책에서의 사전 확률

        # 자식 노드들
        self.children: Dict[YinshAction, YinshNode] = {}

        # 노드 상태
        self.is_terminal = env.is_game_over()
        self.is_expanded = False
        
        # 캐시된 신경망 평가 값
        self.cached_value = None

    @property
    def q_value(self) -> float:
        """Q값 (평균 가치) 반환"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def add_child(self, action: YinshAction, child_env: YinshEnv, prior: float = 0.0) -> "YinshNode":
        """자식 노드 추가"""
        child = YinshNode(child_env, self, action)
        child.prior = prior
        self.children[action] = child
        return child

    def get_ucb_value(self, c_puct: float) -> float:
        """
        AlphaZero UCB 값 계산
        UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        if self.visit_count == 0:
            return float("inf")  # First Play Urgency

        # Q값 (평균 가치)
        q_value = self.q_value

        # UCB 탐색 항
        if self.parent is None:
            parent_visits = 1
        else:
            parent_visits = self.parent.visit_count

        exploration_bonus = (
            c_puct 
            * self.prior 
            * np.sqrt(parent_visits) 
            / (1 + self.visit_count)
        )

        return q_value + exploration_bonus

    def select_best_child(self, c_puct: float) -> Tuple[YinshAction, "YinshNode"]:
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

    def expand(self, action_priors: Dict[YinshAction, float]) -> None:
        """
        신경망 정책을 기반으로 노드 확장
        
        Args:
            action_priors: 액션별 사전 확률
        """
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

            # 사전 확률 설정
            prior = action_priors.get(action, 1.0 / len(valid_actions))

            # 자식 노드 생성
            self.add_child(action, new_env, prior)

        self.is_expanded = True

    def backup(self, value: float) -> None:
        """
        AlphaZero 스타일 가치 역전파
        
        Args:
            value: 리프 노드에서 평가된 가치
        """
        current = self
        current_value = value

        while current is not None:
            # 방문 횟수 및 가치 합 업데이트
            current.visit_count += 1
            current.value_sum += current_value

            # 다음 노드로 이동 (값은 반전)
            current = current.parent
            current_value = -current_value

    def get_action_probabilities(
        self, temperature: float = 1.0
    ) -> Dict[YinshAction, float]:
        """
        방문 횟수 기반 액션 확률 분포
        
        Args:
            temperature: 선택 온도 (0=greedy, >0=stochastic)
            
        Returns:
            액션별 확률 분포
        """
        if not self.children:
            return {}

        actions = list(self.children.keys())
        visit_counts = np.array([
            self.children[action].visit_count for action in actions
        ])

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
            if visit_counts.sum() > 0:
                probs = visit_counts / visit_counts.sum()
            else:
                probs = np.ones(len(actions)) / len(actions)

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

    def get_visit_counts(self) -> Dict[YinshAction, int]:
        """자식 노드들의 방문 횟수 반환"""
        return {action: child.visit_count for action, child in self.children.items()}

    def get_q_values(self) -> Dict[YinshAction, float]:
        """자식 노드들의 Q값 반환"""
        return {action: child.q_value for action, child in self.children.items()}

    def get_state_hash(self) -> str:
        """상태 해시 반환 (캐싱용)"""
        return self.env.get_state_string()

    def is_leaf(self) -> bool:
        """리프 노드인지 확인"""
        return not self.children

    def is_root(self) -> bool:
        """루트 노드인지 확인"""
        return self.parent is None

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

    def get_subtree_size(self) -> int:
        """서브트리 크기 (노드 수) 반환"""
        size = 1  # 자기 자신
        for child in self.children.values():
            size += child.get_subtree_size()
        return size

    def get_principal_variation(self, depth: int = 5) -> List[YinshAction]:
        """
        주요 변화 (가장 많이 방문된 경로) 반환
        
        Args:
            depth: 최대 깊이
            
        Returns:
            주요 변화 액션 시퀀스
        """
        pv = []
        current = self
        
        for _ in range(depth):
            if not current.children:
                break
                
            best_action = current.get_best_action()
            if best_action is None:
                break
                
            pv.append(best_action)
            current = current.children[best_action]
            
        return pv

    def print_tree(self, max_depth: int = 3, current_depth: int = 0):
        """트리 구조 출력 (디버깅용)"""
        if current_depth > max_depth:
            return

        indent = "  " * current_depth
        action_str = str(self.action) if self.action else "ROOT"
        
        # 상세 통계 출력
        print(f"{indent}{action_str}")
        print(f"{indent}  N={self.visit_count}, Q={self.q_value:.3f}, P={self.prior:.3f}")
        
        if hasattr(self, 'cached_value') and self.cached_value is not None:
            print(f"{indent}  V={self.cached_value:.3f}")

        # 자식 노드들을 방문 횟수 순으로 정렬하여 출력
        sorted_children = sorted(
            self.children.items(), 
            key=lambda x: x[1].visit_count, 
            reverse=True
        )
        
        for action, child in sorted_children:
            child.print_tree(max_depth, current_depth + 1)

    def print_statistics(self):
        """노드 통계 출력"""
        print(f"Node Statistics:")
        print(f"  Visit Count: {self.visit_count}")
        print(f"  Q-value: {self.q_value:.4f}")
        print(f"  Prior: {self.prior:.4f}")
        print(f"  Children: {len(self.children)}")
        print(f"  Terminal: {self.is_terminal}")
        print(f"  Expanded: {self.is_expanded}")
        print(f"  Depth: {self.get_depth()}")

    def __str__(self):
        action_str = str(self.action) if self.action else "ROOT"
        return f"YinshNode({action_str}, N={self.visit_count}, Q={self.q_value:.3f})"

    def __repr__(self):
        return self.__str__() 