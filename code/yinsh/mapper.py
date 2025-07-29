# mapper.py - YINSH Action Mapper (개선된 버전)

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from .env import YinshAction, YinshEnv, Color
from . import config


class YinshActionMapper:
    """YINSH 액션을 신경망 출력과 매핑하는 클래스 (개선된 버전)"""

    def __init__(self):
        self.action_to_index: Dict[str, int] = {}  # 액션 해시 -> 인덱스
        self.index_to_action: Dict[int, YinshAction] = {}
        
        # 액션 타입별 인덱스 범위
        self.action_ranges = {
            "PLACE_RING": (0, 0),
            "MOVE_RING": (0, 0), 
            "REMOVE_LINE": (0, 0)
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

    def _build_mapping(self):
        """효율적인 액션 매핑 구축"""
        index = 0

        # 1. 링 배치 액션 (최적화: valid_points 사용)
        start_place = index
        if config.USE_VALID_POINTS_ONLY:
            # 유효한 위치만 사용 (85개)
            valid_positions = self._get_valid_positions()
            for x, y in valid_positions:
                action = YinshAction("PLACE_RING", to_pos=(x, y))
                action_hash = self._hash_action(action)
                self.action_to_index[action_hash] = index
                self.index_to_action[index] = action
                index += 1
        else:
            # 기존 방식: 전체 보드 (121개)
            for x in range(config.BOARD_SIZE):
                for y in range(config.BOARD_SIZE):
                    action = YinshAction("PLACE_RING", to_pos=(x, y))
                    action_hash = self._hash_action(action)
                    self.action_to_index[action_hash] = index
                    self.index_to_action[index] = action
                    index += 1
        self.action_ranges["PLACE_RING"] = (start_place, index - 1)

        # 2. 링 이동 액션 (효율적인 매핑)
        start_move = index
        # 모든 가능한 시작 위치에서 6방향으로 최대 거리까지
        for from_x in range(config.BOARD_SIZE):
            for from_y in range(config.BOARD_SIZE):
                from_pos = (from_x, from_y)
                # 6방향으로 이동
                for direction in config.VALID_HEX_DIRECTIONS:
                    # 각 방향으로 최대 10칸까지 (보드 크기 고려)
                    for distance in range(1, config.BOARD_SIZE):
                        to_x = from_x + direction[0] * distance
                        to_y = from_y + direction[1] * distance
                        
                        # 보드 범위 체크
                        if 0 <= to_x < config.BOARD_SIZE and 0 <= to_y < config.BOARD_SIZE:
                            to_pos = (to_x, to_y)
                            action = YinshAction("MOVE_RING", from_pos=from_pos, to_pos=to_pos)
                            action_hash = self._hash_action(action)
                            self.action_to_index[action_hash] = index
                            self.index_to_action[index] = action
                            index += 1
                            
                            if index >= config.POLICY_OUTPUT_SIZE * 0.8:  # 공간 확보
                                break
                        else:
                            break
                    if index >= config.POLICY_OUTPUT_SIZE * 0.8:
                        break
                if index >= config.POLICY_OUTPUT_SIZE * 0.8:
                    break
            if index >= config.POLICY_OUTPUT_SIZE * 0.8:
                break
        self.action_ranges["MOVE_RING"] = (start_move, index - 1)

        # 3. REMOVE_LINE 액션 (라인 제거 + 링 제거)
        start_remove = index
        # 가능한 라인 위치들을 미리 생성 (실제 게임에서는 동적으로 생성됨)
        self._generate_remove_line_actions(index)
        self.action_ranges["REMOVE_LINE"] = (start_remove, min(index + 500, config.POLICY_OUTPUT_SIZE - 1))

        print(f"🔗 Action mapper built: {len(self.action_to_index)} actions mapped")
        print(f"├── PLACE_RING: {self.action_ranges['PLACE_RING'][1] - self.action_ranges['PLACE_RING'][0] + 1} actions")
        print(f"├── MOVE_RING: {self.action_ranges['MOVE_RING'][1] - self.action_ranges['MOVE_RING'][0] + 1} actions")
        print(f"└── REMOVE_LINE: {self.action_ranges['REMOVE_LINE'][1] - self.action_ranges['REMOVE_LINE'][0] + 1} actions")

    def _generate_remove_line_actions(self, start_index: int):
        """REMOVE_LINE 액션들을 생성"""
        index = start_index
        
        # 가능한 라인 패턴들 (5개 연속)
        for start_x in range(config.BOARD_SIZE):
            for start_y in range(config.BOARD_SIZE):
                for direction in config.VALID_HEX_DIRECTIONS:
                    # 5개 연속 라인 생성
                    line_positions = []
                    for i in range(config.LINE_LENGTH_TO_WIN):
                        pos_x = start_x + direction[0] * i
                        pos_y = start_y + direction[1] * i
                        if 0 <= pos_x < config.BOARD_SIZE and 0 <= pos_y < config.BOARD_SIZE:
                            line_positions.append((pos_x, pos_y))
                        else:
                            break
                    
                    # 정확히 5개일 때만 유효한 라인
                    if len(line_positions) == config.LINE_LENGTH_TO_WIN:
                        # 각 링 위치에 대해 액션 생성 (최적화: valid_points만 사용)
                        if config.USE_VALID_POINTS_ONLY:
                            valid_positions = self._get_valid_positions()
                            for ring_x, ring_y in valid_positions:
                                ring_pos = (ring_x, ring_y)
                                action = YinshAction(
                                    "REMOVE_LINE",
                                    remove_line_positions=line_positions,
                                    remove_ring_position=ring_pos
                                )
                                action_hash = self._hash_action(action)
                                self.action_to_index[action_hash] = index
                                self.index_to_action[index] = action
                                index += 1
                                
                                if index >= config.POLICY_OUTPUT_SIZE:
                                    return
                        else:
                            # 기존 방식: 전체 보드 순회
                            for ring_x in range(config.BOARD_SIZE):
                                for ring_y in range(config.BOARD_SIZE):
                                    ring_pos = (ring_x, ring_y)
                                    action = YinshAction(
                                        "REMOVE_LINE",
                                        remove_line_positions=line_positions,
                                        remove_ring_position=ring_pos
                                    )
                                    action_hash = self._hash_action(action)
                                    self.action_to_index[action_hash] = index
                                    self.index_to_action[index] = action
                                    index += 1
                                    
                                    if index >= config.POLICY_OUTPUT_SIZE:
                                        return
        
    def _hash_action(self, action: YinshAction) -> str:
        """액션을 고유 해시로 변환"""
        if action.action_type == "PLACE_RING":
            return f"PLACE_{action.to_pos[0]}_{action.to_pos[1]}"
        elif action.action_type == "MOVE_RING":
            return f"MOVE_{action.from_pos[0]}_{action.from_pos[1]}_{action.to_pos[0]}_{action.to_pos[1]}"
        elif action.action_type == "REMOVE_LINE":
            line_str = "_".join([f"{pos[0]}{pos[1]}" for pos in action.remove_line_positions])
            ring_str = f"{action.remove_ring_position[0]}{action.remove_ring_position[1]}"
            return f"REMOVE_{line_str}_RING_{ring_str}"
        else:
            return f"{action.action_type}_{hash(action)}"

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
        
        mapping_rate = mapped_actions / len(valid_actions) if valid_actions else 0
        
        return {
            "total_valid_actions": len(valid_actions),
            "mapped_actions": mapped_actions,
            "unmapped_actions": len(unmapped_actions),
            "mapping_rate": mapping_rate,
            "unmapped_sample": unmapped_actions[:3]  # 처음 3개만 샘플로
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
    """액션 매퍼 테스트 (개선된 버전)"""
    print("🧪 Testing YINSH Action Mapper (Enhanced)...")

    mapper = get_action_mapper()

    # 기본 테스트 - 링 배치 단계
    print("\n📋 Testing PLACE_RING phase:")
    env = YinshEnv()
    validation_result = mapper.validate_action_mapping(env)
    print(f"├── Valid actions: {validation_result['total_valid_actions']}")
    print(f"├── Mapped actions: {validation_result['mapped_actions']}")
    print(f"├── Mapping rate: {validation_result['mapping_rate']:.2%}")
    print(f"└── Unmapped sample: {validation_result['unmapped_sample']}")

    # 메인 게임 단계 테스트를 위해 링들을 배치
    print("\n📋 Testing MAIN_GAME phase:")
    # 빠르게 링 배치 단계를 완료
    for color in [env.current_player]:
        for i in range(config.RINGS_PER_PLAYER):
            valid_actions = env.get_valid_actions()
            if valid_actions and valid_actions[0].action_type == "PLACE_RING":
                env.step(valid_actions[0])
                # 플레이어 전환
                env.current_player = Color.BLACK if env.current_player == Color.WHITE else Color.WHITE
            else:
                break
    
    # 메인 게임 단계에서 테스트
    if env.phase.name == "MAIN_GAME":
        validation_result = mapper.validate_action_mapping(env)
        print(f"├── Valid actions: {validation_result['total_valid_actions']}")
        print(f"├── Mapped actions: {validation_result['mapped_actions']}")
        print(f"├── Mapping rate: {validation_result['mapping_rate']:.2%}")
        print(f"└── Unmapped sample: {validation_result['unmapped_sample']}")

    # 매핑 통계
    print("\n📊 Mapping Statistics:")
    stats = mapper.get_mapping_stats()
    print(f"├── Total mapped actions: {stats['total_actions']}")
    print(f"├── Policy output size: {stats['policy_output_size']}")
    print(f"├── Coverage: {stats['coverage']:.2%}")
    print(f"└── Action ranges: {stats['action_ranges']}")


if __name__ == "__main__":
    test_action_mapper()
