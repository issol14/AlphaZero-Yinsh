# mcts_parallel.py - 병렬 MCTS 구현

import numpy as np
import torch
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import math
from collections import defaultdict

from .env import YinshEnv, YinshAction, Color
from .node_optimized import YinshNode
from .mapper import YinshActionMapper
from . import config


class ParallelMCTS:
    """병렬 MCTS 구현"""
    
    def __init__(self, neural_network, c_puct: float = 2.5, num_simulations: int = 800, 
                 num_threads: int = 4, batch_size: int = 32):
        """
        Args:
            neural_network: PyTorch YINSH 신경망
            c_puct: UCB 탐색 상수
            num_simulations: 시뮬레이션 횟수
            num_threads: 병렬 스레드 수
            batch_size: 신경망 배치 크기
        """
        self.neural_network = neural_network
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.num_threads = num_threads
        self.batch_size = batch_size
        
        # 액션 매퍼
        self.action_mapper = YinshActionMapper()
        
        # 상태 캐시
        self.state_cache = {}
        self.cache_size_limit = 10000
        
        # 통계
        self.nodes_expanded = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.batch_evaluations = 0
        
        # 스레드 안전을 위한 락
        self.cache_lock = threading.Lock()
        self.stats_lock = threading.Lock()
    
    def search(self, root_env: YinshEnv, temperature: float = 1.0, 
               add_noise: bool = False) -> Tuple[YinshAction, Dict]:
        """병렬 MCTS 검색"""
        # 루트 노드 생성
        root = YinshNode(root_env.copy(), None, None)
        
        # 루트 노드 확장
        self._expand_and_evaluate(root)
        
        # Dirichlet 노이즈 추가
        if add_noise:
            self._add_dirichlet_noise(root)
        
        # 병렬 시뮬레이션 실행
        start_time = time.time()
        
        # 배치 크기로 시뮬레이션 분할
        batch_simulations = []
        for i in range(0, self.num_simulations, self.batch_size):
            batch_size = min(self.batch_size, self.num_simulations - i)
            batch_simulations.append(batch_size)
        
        # 병렬 실행
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for batch_size in batch_simulations:
                future = executor.submit(self._run_batch_simulations, root, batch_size)
                futures.append(future)
            
            # 모든 배치 완료 대기
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"배치 시뮬레이션 오류: {e}")
        
        # 최종 액션 선택
        best_action, stats = self._select_final_action(root, temperature)
        
        # 통계 업데이트
        search_time = time.time() - start_time
        stats.update({
            "nodes_expanded": self.nodes_expanded,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "batch_evaluations": self.batch_evaluations,
            "search_time": search_time,
            "simulations_per_second": self.num_simulations / search_time if search_time > 0 else 0,
            "parallel_threads": self.num_threads,
            "batch_size": self.batch_size
        })
        
        return best_action, stats
    
    def _run_batch_simulations(self, root: YinshNode, batch_size: int):
        """배치 시뮬레이션 실행"""
        for _ in range(batch_size):
            # 1. Selection: UCB로 리프 노드까지 경로 선택
            path = self._select_path(root)
            leaf = path[-1]
            
            # 2. Expansion & Evaluation: 리프 노드 확장 및 평가
            value = self._expand_and_evaluate(leaf)
            
            # 3. Backup: 경로를 따라 값 역전파
            self._backup_path(path, value)
    
    def _select_path(self, root: YinshNode) -> List[YinshNode]:
        """UCB 기준으로 루트부터 리프까지의 경로 선택"""
        path = [root]
        current = root
        
        while current.children and not current.env.is_game_over():
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
        """AlphaZero UCB 공식"""
        if child.visit_count == 0:
            return float("inf")
        
        q_value = child.value_sum / child.visit_count
        exploration_bonus = (
            self.c_puct 
            * child.prior 
            * math.sqrt(parent.visit_count) 
            / (1 + child.visit_count)
        )
        
        return q_value + exploration_bonus
    
    def _expand_and_evaluate(self, node: YinshNode) -> float:
        """노드 확장 및 신경망 평가 (배치 처리 지원)"""
        # 이미 확장되었거나 터미널 노드인 경우
        if node.children or node.env.is_game_over():
            if node.env.is_game_over():
                return self._get_terminal_value(node.env)
            else:
                return node.cached_value if hasattr(node, 'cached_value') else 0.0
        
        # 상태 해시 확인 (스레드 안전)
        state_hash = self._get_state_hash(node.env)
        
        with self.cache_lock:
            if state_hash in self.state_cache:
                policy_probs, value = self.state_cache[state_hash]
                self.cache_hits += 1
            else:
                # 신경망으로 정책과 가치 평가
                policy_probs, value = self._evaluate_with_neural_network(node.env)
                self.state_cache[state_hash] = (policy_probs, value)
                self.cache_misses += 1
        
        # 노드 확장
        self._expand_node(node, policy_probs)
        
        # 값 캐싱
        node.cached_value = value
        
        with self.stats_lock:
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
        """경로를 따라 값 역전파"""
        value = leaf_value
        
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value
    
    def _evaluate_with_neural_network(self, env: YinshEnv) -> Tuple[np.ndarray, float]:
        """신경망을 사용한 상태 평가"""
        state_tensor = env.get_state_tensor()
        state_tensor = torch.from_numpy(state_tensor).unsqueeze(0).to(self.neural_network.device)
        
        with torch.no_grad():
            policy, value = self.neural_network(state_tensor)
            policy = policy.cpu().numpy().squeeze()
            value = value.cpu().numpy().squeeze()
        
        return policy, value
    
    def _get_state_hash(self, env: YinshEnv) -> str:
        """상태 해시 생성"""
        rings_white = sorted(env.ring_positions[Color.WHITE])
        rings_black = sorted(env.ring_positions[Color.BLACK])
        markers_white = sorted(env.marker_positions[Color.WHITE])
        markers_black = sorted(env.marker_positions[Color.BLACK])
        
        state_str = f"{env.current_player.name}_{rings_white}_{rings_black}_{markers_white}_{markers_black}"
        return str(hash(state_str))
    
    def _get_terminal_value(self, env: YinshEnv) -> float:
        """터미널 상태의 가치 계산"""
        if not env.is_game_over():
            return 0.0
        
        winner = env.get_winner()
        current_player = env.current_player
        
        if winner == current_player:
            return 1.0
        elif winner is None:
            return 0.0
        else:
            return -1.0
    
    def _add_dirichlet_noise(self, root: YinshNode) -> None:
        """루트 노드에 Dirichlet 노이즈 추가"""
        if not root.children:
            return
        
        actions = list(root.children.keys())
        alpha = config.SELFPLAY_NOISE_ALPHA
        epsilon = config.SELFPLAY_NOISE_EPSILON
        
        noise = np.random.dirichlet([alpha] * len(actions))
        
        for i, action in enumerate(actions):
            child = root.children[action]
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]
    
    def _select_final_action(self, root: YinshNode, temperature: float) -> Tuple[YinshAction, Dict]:
        """최종 액션 선택"""
        if not root.children:
            raise ValueError("No actions available")
        
        # 방문 횟수 기반 확률 분포
        visit_counts = np.array([child.visit_count for child in root.children.values()])
        
        if temperature == 0:
            # Greedy 선택
            best_idx = np.argmax(visit_counts)
            best_action = list(root.children.keys())[best_idx]
        else:
            # Temperature 기반 샘플링
            probs = visit_counts ** (1.0 / temperature)
            probs = probs / np.sum(probs)
            best_idx = np.random.choice(len(probs), p=probs)
            best_action = list(root.children.keys())[best_idx]
        
        # 통계 수집
        stats = {
            "visit_counts": {action: child.visit_count for action, child in root.children.items()},
            "q_values": {action: child.q_value for action, child in root.children.items()},
            "selected_action": str(best_action),
            "temperature": temperature
        }
        
        return best_action, stats
    
    def clear_cache(self):
        """캐시 정리"""
        with self.cache_lock:
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
            "total_requests": total_requests,
            "batch_evaluations": self.batch_evaluations
        }


class ParallelMCTSAgent:
    """병렬 MCTS 에이전트"""
    
    def __init__(self, neural_network, mcts_config: Optional[Dict] = None):
        """
        Args:
            neural_network: PyTorch 신경망
            mcts_config: MCTS 설정 딕셔너리
        """
        if mcts_config is None:
            mcts_config = {}
        
        self.mcts = ParallelMCTS(
            neural_network=neural_network,
            c_puct=mcts_config.get("c_puct", config.CPUCT),
            num_simulations=mcts_config.get("num_simulations", config.MCTS_SIMULATIONS),
            num_threads=mcts_config.get("num_threads", 6),
            batch_size=mcts_config.get("batch_size", 32)
        )
    
    def select_action(self, env: YinshEnv, temperature: float = 1.0, 
                     add_noise: bool = False) -> Tuple[YinshAction, Dict]:
        """액션 선택"""
        action, stats = self.mcts.search(env, temperature, add_noise)
        return action, {"method": "parallel_mcts", "mcts_stats": stats}
    
    def reset(self):
        """에이전트 상태 초기화"""
        self.mcts.clear_cache()