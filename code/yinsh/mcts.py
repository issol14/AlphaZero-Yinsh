# mcts.py - YINSH Monte Carlo Tree Search (PyTorch)

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import math
import time

from .env import YinshEnv, YinshAction, Color, GamePhase
from .node import YinshNode
from .mapper import YinshActionMapper
from . import config


class MCTS:
    """Monte Carlo Tree Search for YINSH"""

    def __init__(self, neural_network, c_puct: float = 1.0, num_simulations: int = 800):
        """
        Args:
            neural_network: PyTorch YINSH 신경망
            c_puct: UCB 탐색 상수
            num_simulations: 시뮬레이션 횟수
        """
        self.neural_network = neural_network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        
        # 액션 매퍼 초기화
        self.action_mapper = YinshActionMapper()

        # 통계
        self.nodes_expanded = 0
        self.cache_hits = 0

        # 노드 캐시 (상태 해시 -> 노드)
        self.node_cache = {}

    def search(
        self, env: YinshEnv, temperature: float = 1.0
    ) -> Tuple[YinshAction, Dict]:
        """
        MCTS 검색 실행

        Args:
            env: YINSH 게임 환경
            temperature: 액션 선택 온도

        Returns:
            best_action: 최적 액션
            stats: 검색 통계
        """
        # 루트 노드 생성
        root = YinshNode(env.copy(), None, None)

        # 시뮬레이션 실행
        start_time = time.time()
        for _ in range(self.num_simulations):
            # 환경 복사
            current_env = env.copy()
            current_node = root

            # Selection: UCB로 리프 노드까지 선택
            current_node = self._select(current_node)

            # Expansion: 리프 노드 확장
            self._expand_node(current_node)

            # Simulation: 신경망으로 평가
            if current_node.children:
                # 자식 노드 중 하나 선택
                action = np.random.choice(list(current_node.children.keys()))
                child = current_node.children[action]
                value = self._evaluate_with_neural_network(
                    child.env.get_state_tensor()
                )[1]
            else:
                # 터미널 노드인 경우
                if current_env.is_game_over():
                    winner = current_env.get_winner()
                    if winner == Color.WHITE:
                        value = 1.0 if current_env.current_player == Color.WHITE else -1.0
                    elif winner == Color.BLACK:
                        value = 1.0 if current_env.current_player == Color.BLACK else -1.0
                    else:
                        value = 0.0  # 무승부
                else:
                    value = self._evaluate_with_neural_network(
                        current_env.get_state_tensor()
                    )[1]

            # Backup: 루트까지 역전파
            self._backup(current_node, value)

        # 최종 액션 선택
        best_action, stats = self._select_action(root, temperature)

        # 통계 업데이트
        search_time = time.time() - start_time
        stats.update(
            {
                "nodes_expanded": self.nodes_expanded,
                "cache_hits": self.cache_hits,
                "search_time": search_time,
                "simulations_per_second": self.num_simulations / search_time if search_time > 0 else 0,
            }
        )

        return best_action, stats

    def _select(self, node: YinshNode) -> YinshNode:
        """UCB로 리프 노드까지 선택"""
        while node.children and not node.env.is_game_over():
            # UCB 값이 가장 높은 자식 선택
            best_ucb = float("-inf")
            best_child = None
            best_action = None

            for action, child in node.children.items():
                ucb = self._calculate_ucb(node, child, action)
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child
                    best_action = action

            if best_child is None:
                break

            # 환경 업데이트
            node = best_child

        return node

    def _calculate_ucb(
        self, parent: YinshNode, child: YinshNode, action: YinshAction
    ) -> float:
        """UCB 값 계산"""
        if child.visit_count == 0:
            return float("inf")

        # Q 값 (평균 가치)
        q_value = child.value if child.value is not None else 0.0

        # UCB 항
        ucb_term = (
            self.c_puct
            * child.prior
            * math.sqrt(parent.visit_count)
            / (1 + child.visit_count)
        )

        return q_value + ucb_term

    def _expand_node(self, node: YinshNode) -> None:
        """노드 확장"""
        if node.children or node.env.is_game_over():
            return

        # 유효한 액션들 가져오기
        valid_actions = node.env.get_valid_actions()
        if not valid_actions:
            return

        # 신경망으로 정책과 가치 예측
        state_tensor = node.env.get_state_tensor()
        policy_probs, value = self._evaluate_with_neural_network(state_tensor)

        # 자식 노드들 생성
        node.children = {}
        for action in valid_actions:
            # 새 환경 생성
            new_env = node.env.copy()
            new_env.step(action)

            # 자식 노드 생성
            child = YinshNode(new_env, node, action)

            # 사전 확률 설정 (액션 매핑 사용)
            try:
                action_index = self.action_mapper.get_action_index(action)
                if action_index is not None and action_index < len(policy_probs):
                    child.prior = policy_probs[action_index]
                else:
                    child.prior = 1.0 / len(valid_actions)  # 균등 분포
            except Exception:
                child.prior = 1.0 / len(valid_actions)  # 안전 장치

            node.children[action] = child

        # 노드 값 설정
        node.value = value
        self.nodes_expanded += 1

    def _evaluate_with_neural_network(
        self, state_tensor: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        신경망으로 정책과 가치 평가

        Returns:
            policy_probs: 액션 확률 분포 (4000,)
            value: 위치 평가값 (-1 ~ +1)
        """
        self.neural_network.eval()
        with torch.no_grad():
            # NumPy 배열을 PyTorch 텐서로 변환
            if isinstance(state_tensor, np.ndarray):
                state_tensor = torch.from_numpy(state_tensor).float()
            elif not isinstance(state_tensor, torch.Tensor):
                state_tensor = torch.FloatTensor(state_tensor)

            # 배치 차원 추가
            if len(state_tensor.shape) == 3:
                state_tensor = state_tensor.unsqueeze(0)

            # GPU로 이동 (가능한 경우)
            device = next(self.neural_network.parameters()).device
            state_tensor = state_tensor.to(device)

            # 신경망 평가
            policy_logits, value = self.neural_network(state_tensor)

            # 확률 분포로 변환
            policy_probs = torch.exp(policy_logits).squeeze().cpu().numpy()
            value_scalar = value.squeeze().item()

        return policy_probs, value_scalar

    def _backup(self, leaf_node: YinshNode, value: float) -> None:
        """
        Backup 단계: 리프 노드 값을 루트까지 역전파

        Args:
            leaf_node: 리프 노드
            value: 평가된 가치
        """
        current = leaf_node

        while current is not None:
            current.visit_count += 1

            # 가치 업데이트 (방문 횟수 가중 평균)
            if current.value is None:
                current.value = value
            else:
                current.value = (
                    current.value * (current.visit_count - 1) + value
                ) / current.visit_count

            # 다음 노드로 이동 (값은 반전)
            current = current.parent
            value = -value

    def _select_action(
        self, root: YinshNode, temperature: float
    ) -> Tuple[YinshAction, Dict]:
        """
        최종 액션 선택

        Args:
            root: 루트 노드
            temperature: 선택 온도 (0=greedy, >0=stochastic)

        Returns:
            action: 선택된 액션
            stats: 선택 통계
        """
        if not root.children:
            raise ValueError("Root node has no children!")

        # 방문 횟수 기반 확률 분포 계산
        actions = list(root.children.keys())
        visit_counts = np.array(
            [root.children[action].visit_count for action in actions]
        )

        if temperature == 0:
            # Greedy selection
            best_idx = np.argmax(visit_counts)
            selected_action = actions[best_idx]
            probs = np.zeros(len(actions))
            probs[best_idx] = 1.0
        else:
            # Stochastic selection with temperature
            if temperature != 1.0:
                visit_counts = visit_counts ** (1.0 / temperature)

            probs = visit_counts / visit_counts.sum()
            selected_idx = np.random.choice(len(actions), p=probs)
            selected_action = actions[selected_idx]

        stats = {
            "action_probabilities": {
                str(action): prob for action, prob in zip(actions, probs)
            },
            "visit_counts": {
                str(action): count for action, count in zip(actions, visit_counts)
            },
            "total_visits": root.visit_count,
        }

        return selected_action, stats

    def get_action_probabilities(self, root: YinshNode) -> Dict[YinshAction, float]:
        """
        방문 횟수 기반 액션 확률 분포 반환
        """
        if not root.children:
            return {}

        actions = list(root.children.keys())
        visit_counts = np.array(
            [root.children[action].visit_count for action in actions]
        )

        # 정규화
        probs = (
            visit_counts / visit_counts.sum()
            if visit_counts.sum() > 0
            else np.ones(len(actions)) / len(actions)
        )

        return {action: prob for action, prob in zip(actions, probs)}


class MCTSAgent:
    """MCTS 기반 YINSH 에이전트"""

    def __init__(self, neural_network, mcts_config: Optional[Dict] = None):
        """
        Args:
            neural_network: PyTorch YINSH 신경망
            mcts_config: MCTS 설정 옵션
        """
        if mcts_config is None:
            mcts_config = {
                "c_puct": config.CPUCT,
                "num_simulations": config.MCTS_SIMULATIONS,
            }

        self.mcts = MCTS(
            neural_network=neural_network,
            c_puct=mcts_config.get("c_puct", 1.0),
            num_simulations=mcts_config.get("num_simulations", 800),
        )

        self.name = "MCTS Agent"
        self.games_played = 0

    def select_action(
        self, env: YinshEnv, temperature: float = 1.0
    ) -> Tuple[YinshAction, Dict]:
        """
        MCTS를 사용하여 액션 선택

        Returns:
            action: 선택된 액션
            mcts_info: MCTS 정보 (학습용)
        """
        action, stats = self.mcts.search(env, temperature)

        mcts_info = {
            "mcts_stats": stats,
            "agent_name": self.name,
            "games_played": self.games_played,
        }

        return action, mcts_info

    def reset(self):
        """게임 종료 후 리셋"""
        self.games_played += 1
        # MCTS 통계 리셋
        self.mcts.nodes_expanded = 0
        self.mcts.cache_hits = 0
