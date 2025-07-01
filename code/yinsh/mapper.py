# mapper.py - YINSH Action Mapper

import numpy as np
from typing import Dict, List, Optional, Tuple
from .env import YinshAction, YinshEnv
from . import config


class YinshActionMapper:
    """YINSH 액션을 신경망 출력과 매핑하는 클래스"""

    def __init__(self):
        self.action_to_index: Dict[YinshAction, int] = {}
        self.index_to_action: Dict[int, YinshAction] = {}
        self._build_mapping()

    def _build_mapping(self):
        """액션 매핑 구축"""
        index = 0

        # 링 배치 액션 (11x11 = 121개)
        for x in range(config.BOARD_SIZE):
            for y in range(config.BOARD_SIZE):
                action = YinshAction("PLACE_RING", to_pos=(x, y))
                self.action_to_index[action] = index
                self.index_to_action[index] = action
                index += 1

        # 링 이동 액션 (시작 위치 x 도착 위치)
        # 실제로는 유효한 이동만 고려해야 하지만, 간단화를 위해 모든 가능한 조합
        for from_x in range(config.BOARD_SIZE):
            for from_y in range(config.BOARD_SIZE):
                for to_x in range(config.BOARD_SIZE):
                    for to_y in range(config.BOARD_SIZE):
                        if (from_x, from_y) != (to_x, to_y):
                            action = YinshAction(
                                "MOVE_RING",
                                from_pos=(from_x, from_y),
                                to_pos=(to_x, to_y),
                            )
                            if index < config.POLICY_OUTPUT_SIZE:
                                self.action_to_index[action] = index
                                self.index_to_action[index] = action
                                index += 1

        print(f"🔗 Action mapper built: {len(self.action_to_index)} actions mapped")

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
    mapper = get_action_mapper()
    policy = np.zeros(config.POLICY_OUTPUT_SIZE, dtype=np.float32)

    for action in actions:
        index = mapper.get_action_index(action)
        if index is not None and index < config.POLICY_OUTPUT_SIZE:
            policy[index] = 1.0 / len(actions)  # 균등 분포

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
    print("🧪 Testing YINSH Action Mapper...")

    mapper = get_action_mapper()

    # 기본 테스트
    env = YinshEnv()
    valid_actions = env.get_valid_actions()

    print(f"├── Valid actions: {len(valid_actions)}")
    print(f"├── Mapped actions: {len(mapper.action_to_index)}")
    print(f"└── Policy output size: {config.POLICY_OUTPUT_SIZE}")

    # 매핑 테스트
    for i, action in enumerate(valid_actions[:5]):  # 처음 5개만 테스트
        index = mapper.get_action_index(action)
        if index is not None:
            reconstructed_action = mapper.get_index_action(index)
            print(f"├── Action {i}: {action} -> {index} -> {reconstructed_action}")

    # 마스크 테스트
    mask = mapper.get_valid_action_mask(valid_actions)
    print(f"├── Valid mask sum: {mask.sum()}")
    print(f"└── Valid mask shape: {mask.shape}")


if __name__ == "__main__":
    test_action_mapper()
