"""
YINSH AlphaZero 데이터 처리 유틸리티
"""

import pickle
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Any
import json
from dataclasses import dataclass
import sys
import os

# 프로젝트 루트 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from yinsh.env import YinshEnv, Color
from yinsh.mapper import YinshActionMapper

# 로컬 임포트
try:
    import training.data_config as config
except ImportError:
    # 직접 실행하는 경우
    import data_config as config

@dataclass
class GameData:
    """단일 게임의 데이터를 저장하는 클래스"""
    states: List[np.ndarray]           # 게임 중 상태들
    actions: List[int]                 # 선택된 액션들 (인덱스)
    mcts_probs: List[np.ndarray]       # MCTS 확률 분포들
    rewards: List[float]               # 각 상태의 최종 가치
    game_result: float                 # 게임 결과 (1/-1/0)
    game_length: int                   # 게임 길이
    white_agent: str                   # 백 에이전트 타입
    black_agent: str                   # 흑 에이전트 타입
    metadata: Dict[str, Any]           # 추가 메타데이터

class DataProcessor:
    """게임 데이터 처리 클래스"""
    
    def __init__(self):
        self.action_mapper = YinshActionMapper()
        self.env = YinshEnv()
        
    def process_game_result(self, game_dict: Dict[str, Any]) -> GameData:
        """
        게임 결과를 처리하여 학습용 데이터로 변환합니다.
        
        Args:
            game_dict: 게임 결과 딕셔너리
            
        Returns:
            GameData: 처리된 게임 데이터
        """
        # 기본 정보 추출
        moves_history = game_dict.get('moves_history', [])
        final_result = game_dict.get('result', 0)
        winner = game_dict.get('winner')
        
        if not moves_history:
            raise ValueError("No moves history found in game data")
            
        # 상태, 액션, 확률 추출
        states = []
        actions = []
        mcts_probs = []
        players = []
        
        for move_data in moves_history:
            # 상태 추출
            if 'state' in move_data:
                state = move_data['state']
                if isinstance(state, dict):
                    # 환경 상태로부터 텐서 생성
                    state_tensor = self._dict_to_state_tensor(state)
                else:
                    state_tensor = state
                states.append(state_tensor)
            
            # 액션 추출
            if 'action' in move_data:
                action = move_data['action']
                if hasattr(action, '__dict__'):
                    # YinshAction 객체인 경우
                    action_idx = self.action_mapper.action_to_index(action)
                else:
                    action_idx = action
                actions.append(action_idx)
            
            # MCTS 확률 추출
            if 'mcts_probs' in move_data:
                probs = move_data['mcts_probs']
                if isinstance(probs, dict):
                    # 딕셔너리 형태를 배열로 변환
                    prob_array = np.zeros(self.action_mapper.total_actions)
                    for action_str, prob in probs.items():
                        try:
                            action_obj = self._parse_action_string(action_str)
                            action_idx = self.action_mapper.action_to_index(action_obj)
                            prob_array[action_idx] = prob
                        except:
                            continue
                    mcts_probs.append(prob_array)
                else:
                    mcts_probs.append(probs)
            else:
                # MCTS 확률이 없는 경우 (Random 에이전트 등)
                prob_array = np.zeros(self.action_mapper.total_actions)
                if len(actions) > 0:
                    prob_array[actions[-1]] = 1.0  # 선택된 액션에 확률 1
                mcts_probs.append(prob_array)
            
            # 플레이어 정보
            if 'player' in move_data:
                players.append(move_data['player'])
        
        # 보상 계산 (게임 결과를 각 상태에 역전파)
        rewards = self._calculate_rewards(final_result, winner, players)
        
        # 게임 데이터 생성
        game_data = GameData(
            states=states,
            actions=actions,
            mcts_probs=mcts_probs,
            rewards=rewards,
            game_result=final_result,
            game_length=len(states),
            white_agent=game_dict.get('white_agent', 'unknown'),
            black_agent=game_dict.get('black_agent', 'unknown'),
            metadata={
                'game_id': game_dict.get('game_id'),
                'duration': game_dict.get('duration', 0),
                'final_phase': game_dict.get('final_state', {}).get('phase')
            }
        )
        
        return game_data
    
    def _dict_to_state_tensor(self, state_dict: Dict) -> np.ndarray:
        """상태 딕셔너리를 텐서로 변환합니다."""
        # 간단한 구현 - 실제로는 YinshEnv.get_state_tensor() 사용
        try:
            env = YinshEnv()
            # 상태 딕셔너리로부터 환경 복원 (간단화된 버전)
            return env.get_state_tensor()
        except:
            # 기본 빈 상태 반환
            return np.zeros((13, 11, 11))
    
    def _parse_action_string(self, action_str: str):
        """액션 문자열을 파싱하여 액션 객체를 생성합니다."""
        # 간단한 파싱 구현
        # 실제로는 더 정교한 파싱이 필요할 수 있음
        from yinsh.env import YinshAction, YinshActionType
        
        if "PLACE_RING" in action_str:
            # 예: "PLACE_RING((3, 4))"
            import re
            match = re.search(r'\((\d+),\s*(\d+)\)', action_str)
            if match:
                row, col = int(match.group(1)), int(match.group(2))
                return YinshAction(YinshActionType.PLACE_RING, position=(row, col))
        
        # 기본 액션 반환 (에러 방지)
        return YinshAction(YinshActionType.PLACE_RING, position=(5, 5))
    
    def _calculate_rewards(self, final_result: float, winner: Any, players: List) -> List[float]:
        """각 상태에 대한 보상을 계산합니다."""
        if not players:
            return []
        
        rewards = []
        for player in players:
            if winner is None:
                # 무승부
                reward = 0.0
            elif hasattr(winner, 'name'):
                # Color enum의 경우
                if hasattr(player, 'name'):
                    reward = 1.0 if player.name == winner.name else -1.0
                else:
                    reward = 0.0
            else:
                # 단순 비교
                reward = 1.0 if player == winner else -1.0
            
            rewards.append(reward)
        
        return rewards

class DatasetBuilder:
    """학습용 데이터셋 구축 클래스"""
    
    def __init__(self):
        self.processor = DataProcessor()
    
    def build_training_dataset(self, game_data_list: List[GameData]) -> Dict[str, np.ndarray]:
        """
        게임 데이터 리스트로부터 학습용 데이터셋을 구축합니다.
        
        Returns:
            Dict containing 'states', 'policy_targets', 'value_targets'
        """
        all_states = []
        all_policy_targets = []
        all_value_targets = []
        
        for game_data in game_data_list:
            # 게임 필터링
            if not self._is_valid_game(game_data):
                continue
                
            # 데이터 추가
            all_states.extend(game_data.states)
            all_policy_targets.extend(game_data.mcts_probs)
            all_value_targets.extend(game_data.rewards)
        
        return {
            'states': np.array(all_states),
            'policy_targets': np.array(all_policy_targets),
            'value_targets': np.array(all_value_targets).reshape(-1, 1)
        }
    
    def _is_valid_game(self, game_data: GameData) -> bool:
        """게임 데이터의 유효성을 검증합니다."""
        # 게임 길이 검증
        if game_data.game_length < config.MIN_GAME_LENGTH:
            return False
        if game_data.game_length > config.MAX_GAME_LENGTH_FILTER:
            return False
        
        # 액션 다양성 검증
        unique_actions = len(set(game_data.actions))
        if unique_actions < config.MIN_UNIQUE_ACTIONS:
            return False
        
        # 데이터 일관성 검증
        if len(game_data.states) != len(game_data.actions):
            return False
        if len(game_data.states) != len(game_data.mcts_probs):
            return False
        
        return True
    
    def save_dataset(self, dataset: Dict[str, np.ndarray], filepath: Path):
        """데이터셋을 파일로 저장합니다."""
        np.savez_compressed(filepath, **dataset)
        print(f"💾 Dataset saved to {filepath}")
        print(f"📊 States: {dataset['states'].shape}")
        print(f"📊 Policy targets: {dataset['policy_targets'].shape}")
        print(f"📊 Value targets: {dataset['value_targets'].shape}")
    
    def load_dataset(self, filepath: Path) -> Dict[str, np.ndarray]:
        """저장된 데이터셋을 로드합니다."""
        data = np.load(filepath)
        return {
            'states': data['states'],
            'policy_targets': data['policy_targets'],
            'value_targets': data['value_targets']
        }

def load_selfplay_games(filepath: Path) -> List[Dict[str, Any]]:
    """셀프플레이 게임 데이터를 로드합니다."""
    if not filepath.exists():
        return []
    
    with open(filepath, 'rb') as f:
        games = pickle.load(f)
    
    print(f"📂 Loaded {len(games)} games from {filepath}")
    return games

def save_selfplay_games(games: List[Dict[str, Any]], filepath: Path):
    """셀프플레이 게임 데이터를 저장합니다."""
    with open(filepath, 'wb') as f:
        pickle.dump(games, f)
    
    print(f"💾 Saved {len(games)} games to {filepath}")

def analyze_dataset(dataset: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """데이터셋을 분석하여 통계를 반환합니다."""
    stats = {
        'total_samples': len(dataset['states']),
        'state_shape': dataset['states'].shape,
        'policy_shape': dataset['policy_targets'].shape,
        'value_stats': {
            'mean': float(np.mean(dataset['value_targets'])),
            'std': float(np.std(dataset['value_targets'])),
            'min': float(np.min(dataset['value_targets'])),
            'max': float(np.max(dataset['value_targets']))
        },
        'policy_entropy': float(np.mean([
            -np.sum(p * np.log(p + 1e-8)) for p in dataset['policy_targets']
            if np.sum(p) > 0
        ]))
    }
    
    return stats

if __name__ == "__main__":
    # 설정 검증
    config.validate_config()
    print("✅ Data utilities ready") 