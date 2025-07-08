# mapper.py

from typing import List, Tuple, Dict
from yinsh_mcts.game_state import Action

# === 좌표 설정 ===
BOARD_SIZE = 11
MAX_INDEX = 200  # 모델 출력 크기와 맞춤

# === 미리 지정된 고정 액션 리스트 (index ↔ action 매핑의 기준) ===
ACTION_LOOKUP: List[Action] = []

# 우선: PLACE_RING (0 ~ 99)
for x in range(BOARD_SIZE):
    for y in range(BOARD_SIZE):
        if len(ACTION_LOOKUP) >= 100:
            break
        ACTION_LOOKUP.append(Action(action_type="PLACE_RING", to_pos=(x, y)))

# 다음: MOVE_RING (100 ~ 199)
for fx in range(5):
    for fy in range(5):
        for tx in range(5, 10):
            for ty in range(5, 10):
                if len(ACTION_LOOKUP) >= MAX_INDEX:
                    break
                ACTION_LOOKUP.append(Action(
                    action_type="MOVE_RING",
                    from_pos=(fx, fy),
                    to_pos=(tx, ty)
                ))

# === 인덱스 → 액션 ===
def index_to_action(index: int) -> Action:
    if 0 <= index < len(ACTION_LOOKUP):
        return ACTION_LOOKUP[index]
    else:
        raise IndexError(f"Invalid policy index: {index}")

# === 액션 → 인덱스 ===
def action_to_index(action: Action) -> int:
    for idx, a in enumerate(ACTION_LOOKUP):
        if a.action_type != action.action_type:
            continue
        if a.to_pos != action.to_pos:
            continue
        if a.from_pos != action.from_pos:
            continue
        return idx
    raise ValueError(f"Action not found in ACTION_LOOKUP: {action}")
