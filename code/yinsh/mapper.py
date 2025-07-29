# mapper.py - Simplified YINSH Action Mapper (MOVE_RING only)

import numpy as np
from typing import Dict, List, Optional, Tuple
from .env import YinshAction, YinshEnv
from . import config


class YinshActionMapper:
    """간소화된 YINSH 액션을 신경망 출력과 매핑하는 클래스 (MOVE_RING만)"""

    def __init__(self):
        self.action_to_index: Dict[YinshAction, int] = {}
        self.index_to_action: Dict[int, YinshAction] = {}
        self._build_mapping()

    def _build_mapping(self):
        """액션 매핑 구축 (간소화: from_pos * to_pos = 121개)"""
        index = 0

        # 간소화된 매핑: from_pos와 to_pos를 조합하여 121개 액션
        for from_x in range(config.BOARD_SIZE):
            for from_y in range(config.BOARD_SIZE):
                for to_x in range(config.BOARD_SIZE):
                    for to_y in range(config.BOARD_SIZE):
                        if (from_x, from_y) != (to_x, to_y):  # 같은 위치로 이동 불가
                            action = YinshAction(
                                from_pos=(from_x, from_y),
                                to_pos=(to_x, to_y),
                            )
                            if index < config.POLICY_OUTPUT_SIZE:
                                self.action_to_index[action] = index
                                self.index_to_action[index] = action
                                index += 1

        print(f"🔗 Simplified action mapper built: {len(self.action_to_index)} actions mapped")

    def get_action_index(self, action: YinshAction) -> Optional[int]:
        """액션을 인덱스로 변환"""
        return self.action_to_index.get(action)

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
        """정책 확률을 액션별 확률로 변환"""
        action_probs = {}

        for action in valid_actions:
            index = self.get_action_index(action)
            if index is not None and index < len(policy_probs):
                action_probs[action] = policy_probs[index]
            else:
                action_probs[action] = 0.0

        return action_probs

    def get_mapping_stats(self) -> Dict:
        """매핑 통계 반환"""
        return {
            "total_actions": len(self.action_to_index),
            "policy_output_size": config.POLICY_OUTPUT_SIZE,
            "coverage": len(self.action_to_index) / config.POLICY_OUTPUT_SIZE,
            "action_type": "MOVE_RING_ONLY",
        }


# 전역 매퍼 인스턴스
_action_mapper = None


def get_action_mapper() -> YinshActionMapper:
    """전역 액션 매퍼 반환"""
    global _action_mapper
    if _action_mapper is None:
        _action_mapper = YinshActionMapper()
    return _action_mapper


def create_policy_vector(actions: List[YinshAction]) -> np.ndarray:
    """액션 리스트를 정책 벡터로 변환"""
    policy = np.zeros(config.POLICY_OUTPUT_SIZE, dtype=np.float32)
    mapper = get_action_mapper()
    
    for action in actions:
        index = mapper.get_action_index(action)
        if index is not None:
            policy[index] = 1.0
    
    return policy


def action_to_index(action: YinshAction) -> Optional[int]:
    """액션을 인덱스로 변환 (편의 함수)"""
    return get_action_mapper().get_action_index(action)


def index_to_action(index: int) -> Optional[YinshAction]:
    """인덱스를 액션으로 변환 (편의 함수)"""
    return get_action_mapper().get_index_action(index)


def get_valid_action_mask(valid_actions: List[YinshAction]) -> np.ndarray:
    """유효한 액션 마스크 생성 (편의 함수)"""
    return get_action_mapper().get_valid_action_mask(valid_actions)


def test_action_mapper():
    """액션 매퍼 테스트"""
    print("🧪 Testing Simplified Action Mapper...")
    
    mapper = get_action_mapper()
    stats = mapper.get_mapping_stats()
    
    print(f"📊 Mapping Stats:")
    print(f"   ├── Total Actions: {stats['total_actions']}")
    print(f"   ├── Policy Output Size: {stats['policy_output_size']}")
    print(f"   ├── Coverage: {stats['coverage']:.2%}")
    print(f"   └── Action Type: {stats['action_type']}")
    
    # 몇 개의 액션 테스트
    test_actions = [
        YinshAction(from_pos=(0, 0), to_pos=(1, 1)),
        YinshAction(from_pos=(5, 5), to_pos=(6, 6)),
        YinshAction(from_pos=(10, 10), to_pos=(9, 9)),
    ]
    
    for action in test_actions:
        index = mapper.get_action_index(action)
        if index is not None:
            reconstructed = mapper.get_index_action(index)
            print(f"✅ {action} -> {index} -> {reconstructed}")
        else:
            print(f"❌ {action} not found in mapping")


if __name__ == "__main__":
    test_action_mapper()
