# mcts_optimized.py - AlphaZero 논문 기반 최적화된 MCTS 구현

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import math
import time
from collections import defaultdict

from .env import YinshEnv, YinshAction, Color, GamePhase
from .node_optimized import YinshNode
from .mapper import YinshActionMapper
from . import config


class OptimizedMCTS:
    """AlphaZero 논문 기반 최적화된 MCTS"""

    def __init__(self, neural_network, c_puct: float = 2.5, num_simulations: int = 800):
        """
        Args:
            neural_network: PyTorch YINSH 신경망
            c_puct: UCB 탐색 상수 (AlphaZero 논문: 체스 2.5, 바둑 5.0)
            num_simulations: 시뮬레이션 횟수
        """
        self.neural_network = neural_network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        
        # 액션 매퍼
        self.action_mapper = YinshActionMapper()

        # 상태 캐시 최적화 (상태 해시 -> (policy, value))
        self.state_cache = {}
        self.cache_size_limit = 10000  # 캐시 크기 제한
        
        # 통계
        self.nodes_expanded = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def search(
        self, 
        root_env: YinshEnv, 
        temperature: float = 1.0,
        add_noise: bool = False
    ) -> Tuple[YinshAction, Dict]:
        """
        AlphaZero 논문 기반 MCTS 검색

        Args:
            root_env: 루트 게임 환경
            temperature: 액션 선택 온도
            add_noise: Dirichlet 노이즈 추가 여부 (셀프플레이용)

        Returns:
            best_action: 최적 액션
            stats: 검색 통계
        """
        # 루트 노드 생성 및 확장
        root = YinshNode(root_env.copy(), None, None)
        
        # 루트 노드 신경망 평가 및 확장
        self._expand_and_evaluate(root)
        
        # Dirichlet 노이즈 추가 (셀프플레이용)
        if add_noise:
            self._add_dirichlet_noise(root)

        # 시뮬레이션 실행
        start_time = time.time()
        for _ in range(self.num_simulations):
            # 1. Selection: UCB로 리프 노드까지 경로 선택
            path = self._select_path(root)
            leaf = path[-1]
            
            # 2. Expansion & Evaluation: 리프 노드 확장 및 평가
            value = self._expand_and_evaluate(leaf)
            
            # 3. Backup: 경로를 따라 값 역전파
            self._backup_path(path, value)

        # 최종 액션 선택
        best_action, stats = self._select_final_action(root, temperature)

        # 통계 업데이트
        search_time = time.time() - start_time
        stats.update({
            "nodes_expanded": self.nodes_expanded,
            "cache_hits": self.cache_hits,
            "search_time": search_time,
            "simulations_per_second": self.num_simulations / search_time if search_time > 0 else 0,
        })

        return best_action, stats

    def _select_path(self, root: YinshNode) -> List[YinshNode]:
        """
        UCB 기준으로 루트부터 리프까지의 경로 선택
        
        Returns:
            path: 루트부터 리프까지의 노드 경로
        """
        path = [root]
        current = root

        while current.children and not current.env.is_game_over():
            # UCB 값이 가장 높은 자식 선택
            best_action, best_child = self._select_best_child(current)
            path.append(best_child)
            current = best_child

        return path

    def _select_best_child(self, node: YinshNode) -> Tuple[YinshAction, YinshNode]:
        """UCB 기준으로 최고 자식 노드 선택"""
        best_ucb = float("-inf")
        best_action = None
        best_child = None

        for action, child in node.children.items():
            ucb = self._calculate_ucb(node, child)
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action
                best_child = child

        return best_action, best_child

    def _calculate_ucb(self, parent: YinshNode, child: YinshNode) -> float:
        """
        AlphaZero UCB 공식
        UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        """
        if child.visit_count == 0:
            return float("inf")  # First Play Urgency

        # Q값: 현재 플레이어 관점에서의 평균 가치
        q_value = child.value_sum / child.visit_count

        # UCB 항: 탐색 보너스
        exploration_bonus = (
            self.c_puct 
            * child.prior 
            * math.sqrt(parent.visit_count) 
            / (1 + child.visit_count)
        )

        return q_value + exploration_bonus

    def _expand_and_evaluate(self, node: YinshNode) -> float:
        """
        노드 확장 및 신경망 평가
        
        Returns:
            value: 신경망이 평가한 노드의 가치
        """
        # 이미 확장되었거나 터미널 노드인 경우
        if node.children or node.env.is_game_over():
            if node.env.is_game_over():
                # 터미널 노드 가치 계산
                return self._get_terminal_value(node.env)
            else:
                # 이미 확장된 노드는 캐시된 값 반환
                return node.cached_value if hasattr(node, 'cached_value') else 0.0

        # 상태 해시 확인 (캐싱)
        state_hash = node.env.get_state_string()
        if state_hash in self.state_cache:
            policy_probs, value = self.state_cache[state_hash]
            self.cache_hits += 1
        else:
            # 신경망으로 정책과 가치 평가
            policy_probs, value = self._evaluate_with_neural_network(node.env)
            self.state_cache[state_hash] = (policy_probs, value)

        # 노드 확장
        self._expand_node(node, policy_probs)
        
        # 값 캐싱
        node.cached_value = value
        self.nodes_expanded += 1
        
        return value

    def _expand_node(self, node: YinshNode, policy_probs: np.ndarray) -> None:
        """신경망 정책을 기반으로 노드 확장"""
        valid_actions = node.env.get_valid_actions()
        if not valid_actions:
            return

        node.children = {}
        
        for action in valid_actions:
            # 새 환경 생성 (액션 적용)
            new_env = node.env.copy()
            new_env.step(action)
            
            # 자식 노드 생성
            child = YinshNode(new_env, node, action)
            
            # 사전 확률 설정
            try:
                action_index = self.action_mapper.get_action_index(action)
                if action_index is not None and action_index < len(policy_probs):
                    child.prior = policy_probs[action_index]
                else:
                    child.prior = 1.0 / len(valid_actions)
            except Exception:
                child.prior = 1.0 / len(valid_actions)
            
            node.children[action] = child

    def _backup_path(self, path: List[YinshNode], leaf_value: float) -> None:
        """
        경로를 따라 값 역전파
        
        Args:
            path: 루트부터 리프까지의 노드 경로
            leaf_value: 리프 노드에서 평가된 가치
        """
        value = leaf_value
        
        # 경로를 역순으로 순회하며 값 업데이트
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            
            # 다음 노드는 상대방 관점이므로 값 반전
            value = -value

    def _add_dirichlet_noise(self, root: YinshNode) -> None:
        """
        루트 노드에 Dirichlet 노이즈 추가 (셀프플레이용 탐색 촉진)
        P_root = (1-ε)P + ε*Dir(α)
        """
        if not root.children:
            return

        actions = list(root.children.keys())
        alpha = config.SELFPLAY_NOISE_ALPHA
        epsilon = config.SELFPLAY_NOISE_EPSILON
        
        # Dirichlet 노이즈 생성
        noise = np.random.dirichlet([alpha] * len(actions))
        
        # 노이즈 적용
        for i, action in enumerate(actions):
            child = root.children[action]
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]

    def _select_final_action(
        self, 
        root: YinshNode, 
        temperature: float
    ) -> Tuple[YinshAction, Dict]:
        """
        최종 액션 선택 (방문 횟수 기반)
        
        Args:
            root: 루트 노드
            temperature: 선택 온도 (0=greedy, >0=stochastic)
            
        Returns:
            action: 선택된 액션
            stats: 선택 통계
        """
        if not root.children:
            raise ValueError("Root node has no children!")

        actions = list(root.children.keys())
        visit_counts = np.array([
            root.children[action].visit_count for action in actions
        ])

        if temperature == 0:
            # Greedy 선택
            best_idx = np.argmax(visit_counts)
            selected_action = actions[best_idx]
            probs = np.zeros(len(actions))
            probs[best_idx] = 1.0
        else:
            # 온도 기반 확률적 선택
            if temperature != 1.0:
                visit_counts = visit_counts ** (1.0 / temperature)
            
            probs = visit_counts / visit_counts.sum()
            selected_idx = np.random.choice(len(actions), p=probs)
            selected_action = actions[selected_idx]

        # 통계 생성
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

    def _evaluate_with_neural_network(
        self, env: YinshEnv
    ) -> Tuple[np.ndarray, float]:
        """신경망을 사용한 상태 평가 (캐시 최적화)"""
        # 상태 해시 생성
        state_hash = self._get_state_hash(env)
        
        # 캐시 확인
        if state_hash in self.state_cache:
            self.cache_hits += 1
            return self.state_cache[state_hash]
        
        self.cache_misses += 1
        
        # 신경망 평가
        state_tensor = env.get_state_tensor()
        state_tensor = torch.from_numpy(state_tensor).unsqueeze(0).to(self.neural_network.device)
        
        with torch.no_grad():
            policy, value = self.neural_network(state_tensor)
            policy = policy.cpu().numpy().squeeze()
            value = value.cpu().numpy().squeeze()
        
        # 캐시에 저장 (크기 제한 확인)
        if len(self.state_cache) < self.cache_size_limit:
            self.state_cache[state_hash] = (policy, value)
        
        return policy, value
    
    def _get_state_hash(self, env: YinshEnv) -> str:
        """상태 해시 생성 (빠른 버전)"""
        # 간단한 해시 생성 (성능 최적화)
        rings_white = sorted(env.ring_positions[Color.WHITE])
        rings_black = sorted(env.ring_positions[Color.BLACK])
        markers_white = sorted(env.marker_positions[Color.WHITE])
        markers_black = sorted(env.marker_positions[Color.BLACK])
        
        state_str = f"{env.current_player.name}_{rings_white}_{rings_black}_{markers_white}_{markers_black}"
        return str(hash(state_str))
    
    def clear_cache(self):
        """캐시 정리"""
        self.state_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_cache_stats(self) -> Dict:
        """캐시 통계 반환"""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.state_cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }


class OptimizedMCTSAgent:
    """최적화된 MCTS 기반 에이전트"""

    def __init__(self, neural_network, mcts_config: Optional[Dict] = None):
        """
        Args:
            neural_network: 신경망
            mcts_config: MCTS 설정
        """
        if mcts_config is None:
            mcts_config = {
                "c_puct": 2.5,  # AlphaZero 논문 기준 조정
                "num_simulations": config.MCTS_SIMULATIONS,
            }

        self.mcts = OptimizedMCTS(
            neural_network=neural_network,
            c_puct=mcts_config.get("c_puct", 2.5),
            num_simulations=mcts_config.get("num_simulations", 800),
        )

        self.name = "Optimized MCTS Agent"
        self.games_played = 0

    def select_action(
        self, 
        env: YinshEnv, 
        temperature: float = 1.0,
        add_noise: bool = False
    ) -> Tuple[YinshAction, Dict]:
        """
        최적화된 MCTS로 액션 선택
        
        Args:
            env: 게임 환경
            temperature: 선택 온도
            add_noise: Dirichlet 노이즈 추가 (셀프플레이용)
            
        Returns:
            action: 선택된 액션
            mcts_info: MCTS 정보
        """
        action, stats = self.mcts.search(env, temperature, add_noise)

        mcts_info = {
            "mcts_stats": stats,
            "cache_stats": self.mcts.get_cache_stats(),
            "agent_name": self.name,
            "games_played": self.games_played,
        }

        return action, mcts_info

    def reset(self):
        """게임 종료 후 리셋"""
        self.games_played += 1
        # 캐시는 유지 (성능 향상)
        # self.mcts.clear_cache()  # 필요시에만 호출 