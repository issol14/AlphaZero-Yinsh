# mapper.py - YINSH Action Mapper (간소화 버전 - 링 이동만)

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from .env import YinshAction, YinshEnv, Color
from . import config


class YinshActionMapper:
    """YINSH 액션을 신경망 출력과 매핑하는 클래스 (간소화 버전)"""

    def __init__(self):
        self.action_to_index: Dict[str, int] = {}  # 액션 해시 -> 인덱스
        self.index_to_action: Dict[int, YinshAction] = {}
        
        # 액션 타입별 인덱스 범위 (링 이동만)
        self.action_ranges = {
            "MOVE_RING": (0, 0)
        }
        
        # 효율성을 위한 valid_points 캐시
        self._valid_points_cache = None
        
        self._build_mapping()

    def _get_valid_positions(self):
        """유효한 보드 위치들을 반환 (캐시 사용)"""
        if self._valid_points_cache is None:
            # 임시 env 인스턴스로 valid_points 가져오기
            temp_env = YinshEnv()
            self._valid_points_cache = []
            for hex_pos in temp_env.valid_points:
                array_pos = temp_env.hex_to_array_coords(hex_pos)
                self._valid_points_cache.append(array_pos)
        return self._valid_points_cache

    def _build_mapping(self):  # 총 1848개의 액션 존재
        """액션 매핑 구축 (링 이동만)"""
        index = 0

        # MOVE_RING 액션 생성
        start_move = index
        # 모든 가능한 시작 위치에서 6방향으로 최대 거리까지
        for pos in config.VALID_POINTS:
            for direction in config.VALID_HEX_DIRECTIONS:
                dx, dy = direction
                next_pos = (pos[0] + dx, pos[1] + dy)
                while next_pos in config.VALID_POINTS:
                    action = YinshAction(from_pos=pos, to_pos=next_pos)
                    action_hash = self._hash_action(action)
                    self.action_to_index[action_hash] = index
                    self.index_to_action[index] = action
                    index += 1
                    next_pos = (next_pos[0] + dx, next_pos[1] + dy)
        
        self.action_ranges["MOVE_RING"] = (start_move, index - 1)

        print(f"🔗 Action mapper built (간소화): {len(self.action_to_index)} actions mapped")
        print(f"└── MOVE_RING: {self.action_ranges['MOVE_RING'][1] - self.action_ranges['MOVE_RING'][0] + 1} actions")

    def _hash_action(self, action: YinshAction) -> str:
        """액션을 고유 해시로 변환"""
        return str(action)

    def get_action_index(self, action: YinshAction) -> Optional[int]:
        """액션을 인덱스로 변환"""
        action_hash = self._hash_action(action)
        return self.action_to_index.get(action_hash)

    def get_index_action(self, index: int) -> Optional[YinshAction]:
        """인덱스를 액션으로 변환"""
        return self.index_to_action.get(index)

    def get_valid_action_mask(self, valid_actions: List[YinshAction]) -> np.ndarray:
        """유효한 액션 마스크 생성"""
        mask = np.zeros(config.POLICY_OUTPUT_SIZE, dtype=np.float32)

        for action in valid_actions:
            index = self.get_action_index(action)
            if index is not None and index < config.POLICY_OUTPUT_SIZE:
                mask[index] = 1.0
        
        return mask

    def get_action_probabilities(
        self, policy_probs: np.ndarray, valid_actions: List[YinshAction]
    ) -> Dict[YinshAction, float]:
        """정책 확률을 유효한 액션들에 대해 정규화"""
        # 유효한 액션들의 확률만 추출
        valid_probs = {}
        total_prob = 0.0
        
        for action in valid_actions:
            index = self.get_action_index(action)
            if index is not None and index < len(policy_probs):
                prob = policy_probs[index]
                valid_probs[action] = prob
                total_prob += prob
        
        # 정규화
        if total_prob > 0:
            for action in valid_probs:
                valid_probs[action] /= total_prob
        else:
            # 균등 분포
            prob_per_action = 1.0 / len(valid_actions)
            for action in valid_actions:
                valid_probs[action] = prob_per_action
        
        return valid_probs

    def get_mapping_stats(self) -> Dict:
        """매핑 통계 반환"""
        return {
            "total_actions": len(self.action_to_index),
            "move_ring_actions": self.action_ranges["MOVE_RING"][1] - self.action_ranges["MOVE_RING"][0] + 1,
            "action_ranges": self.action_ranges
        }

    def validate_action_mapping(self, env: YinshEnv) -> Dict:
        """현재 게임 상태에서 액션 매핑 유효성 검증"""
        valid_actions = env.get_valid_actions()
        mapped_actions = 0
        unmapped_actions = []
        
        for action in valid_actions:
            index = self.get_action_index(action)
            if index is not None:
                mapped_actions += 1
            else:
                unmapped_actions.append(action)
        
        return {
            "total_valid_actions": len(valid_actions),
            "mapped_actions": mapped_actions,
            "unmapped_actions": len(unmapped_actions),
            "mapping_coverage": mapped_actions / len(valid_actions) if valid_actions else 0.0,
            "unmapped_examples": unmapped_actions[:5]  # 처음 5개만
        }


# 전역 매퍼 인스턴스
_action_mapper = None


def get_action_mapper() -> YinshActionMapper:
    """전역 액션 매퍼 인스턴스 반환"""
    global _action_mapper
    if _action_mapper is None:
        _action_mapper = YinshActionMapper()
    return _action_mapper


def create_policy_vector(actions: List[YinshAction]) -> np.ndarray:
    """액션 리스트를 정책 벡터로 변환"""
    mapper = get_action_mapper()
    policy = np.zeros(config.POLICY_OUTPUT_SIZE, dtype=np.float32)

    for action in actions:
        index = mapper.get_action_index(action)
        if index is not None and index < config.POLICY_OUTPUT_SIZE:
            policy[index] = 1.0
    
    # 정규화
    total = np.sum(policy)
    if total > 0:
        policy /= total
    
    return policy


def action_to_index(action: YinshAction) -> Optional[int]:
    """액션을 인덱스로 변환 (편의 함수)"""
    return get_action_mapper().get_action_index(action)


def index_to_action(index: int) -> Optional[YinshAction]:
    """인덱스를 액션으로 변환 (편의 함수)"""
    return get_action_mapper().get_index_action(index)


def get_valid_action_mask(valid_actions: List[YinshAction]) -> np.ndarray:
    """유효한 액션들의 마스크 반환 (편의 함수)"""
    return get_action_mapper().get_valid_action_mask(valid_actions)


def test_action_mapper():
    """액션 매퍼 테스트"""
    print("🧪 Testing Action Mapper...")
    
    mapper = get_action_mapper()
    env = YinshEnv()
    
    # 매핑 통계 출력
    stats = mapper.get_mapping_stats()
    print(f"📊 Mapping Stats: {stats}")
    
    # 검증
    validation = mapper.validate_action_mapping(env)
    print(f"✅ Validation: {validation}")
    
    # 샘플 액션 테스트
    valid_actions = env.get_valid_actions()
    if valid_actions:
        sample_action = valid_actions[0]
        index = mapper.get_action_index(sample_action)
        reconstructed_action = mapper.get_index_action(index)
        
        print(f"🔍 Sample Action Test:")
        print(f"   Original: {sample_action}")
        print(f"   Index: {index}")
        print(f"   Reconstructed: {reconstructed_action}")
        print(f"   Match: {sample_action == reconstructed_action}")
    
    print("✅ Action Mapper Test Complete!")


if __name__ == "__main__":
    test_action_mapper()
