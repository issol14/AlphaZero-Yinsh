# env.py - Simplified YINSH Game Environment (MOVE_RING only)

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from enum import Enum
import copy
from . import config


class Color(Enum):
    WHITE = 1
    BLACK = -1
    EMPTY = 0


class PieceType(Enum):
    RING = "RING"
    MARKER = "MARKER"
    EMPTY = "EMPTY"


class YinshAction:
    def __init__(
        self,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
    ):
        self.from_pos = from_pos
        self.to_pos = to_pos

    def __str__(self):
        return f"MOVE_RING({self.from_pos}->{self.to_pos})"

    def __hash__(self):
        return hash((self.from_pos, self.to_pos))

    def __eq__(self, other):
        if not isinstance(other, YinshAction):
            return False
        return self.from_pos == other.from_pos and self.to_pos == other.to_pos


class YinshEnv:
    def __init__(self):
        self.board_size = config.BOARD_SIZE
        # YINSH 육각형 보드의 유효한 좌표들 (-5, -5) ~ (5, 5)
        self.valid_points: Set[Tuple[int, int]] = {
            (-5, -4), (-5, -3), (-5, -2), (-5, -1),
            (-4, -5), (-4, -4), (-4, -3), (-4, -2), (-4, -1), (-4, 0), (-4, 1),
            (-3, -5), (-3, -4), (-3, -3), (-3, -2), (-3, -1), (-3, 0), (-3, 1), (-3, 2),
            (-2, -5), (-2, -4), (-2, -3), (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-2, 3),
            (-1, -5), (-1, -4), (-1, -3), (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2), (-1, 3), (-1, 4),
            (0, -4), (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
            (1, -4), (1, -3), (1, -2), (1, -1), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
            (2, -3), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2), (2, 3),
            (3, -2), (3, -1), (3, 0), (3, 1), (3, 2),
            (4, -1), (4, 0), (4, 1),
            (5, 0)
        }

        # 게임 상태 초기화
        self.reset()

    def reset(self):
        """게임을 초기 상태로 리셋 (링들이 이미 배치된 상태)"""
        # 11x11 보드 초기화
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        
        # 초기 링 위치 설정 (간소화된 버전)
        self.ring_positions = {
            Color.WHITE: {(2, 2), (2, 8), (8, 2), (8, 8), (5, 5)},
            Color.BLACK: {(2, 5), (5, 2), (5, 8), (8, 5), (5, 5)}
        }
        
        # 링 위치를 보드에 반영
        for color, positions in self.ring_positions.items():
            for pos in positions:
                if self.is_valid_position(pos):
                    x, y = pos
                    self.board[x, y] = color.value

        self.marker_positions = {Color.WHITE: set(), Color.BLACK: set()}
        self.current_player = Color.WHITE
        self.rings_removed = {Color.WHITE: 0, Color.BLACK: 0}
        self.done = False
        self.winner = None
        self.game_history = []
        self.move_count = 0

    def hex_to_array_coords(self, hex_pos: Tuple[int, int]) -> Tuple[int, int]:
        """육각형 좌표를 배열 인덱스로 변환"""
        x, y = hex_pos
        array_x = x + 5
        array_y = y + 5
        return (array_x, array_y)

    def array_to_hex_coords(self, array_pos: Tuple[int, int]) -> Tuple[int, int]:
        """배열 인덱스를 육각형 좌표로 변환"""
        x, y = array_pos
        hex_x = x - 5
        hex_y = y - 5
        return (hex_x, hex_y)

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """유효한 보드 위치인지 확인 (육각형 보드 기준)"""
        hex_pos = self.array_to_hex_coords(pos)
        return hex_pos in self.valid_points

    def is_empty_position(self, pos: Tuple[int, int]) -> bool:
        """해당 위치가 비어있는지 확인"""
        if not self.is_valid_position(pos):
            return False
        x, y = pos
        return self.board[x, y] == 0

    def has_ring(self, pos: Tuple[int, int], color: Color) -> bool:
        """해당 위치에 특정 색깔의 링이 있는지 확인"""
        return pos in self.ring_positions[color]

    def has_marker(self, pos: Tuple[int, int], color: Color) -> bool:
        """해당 위치에 특정 색깔의 마커가 있는지 확인"""
        return pos in self.marker_positions[color]

    def _has_marker_at(self, pos: Tuple[int, int]) -> bool:
        """해당 위치에 마커가 있는지 확인 (색깔 무관)"""
        return any(pos in marker_set for marker_set in self.marker_positions.values())

    def get_valid_actions(self) -> List[YinshAction]:
        """현재 상태에서 유효한 액션들을 반환 (MOVE_RING만)"""
        actions = []

        # 현재 플레이어의 링들에 대해 가능한 이동 찾기
        for ring_pos in self.ring_positions[self.current_player]:
            for x in range(self.board_size):
                for y in range(self.board_size):
                    to_pos = (x, y)
                    if self.is_valid_ring_move(ring_pos, to_pos):
                        actions.append(YinshAction(from_pos=ring_pos, to_pos=to_pos))

        return actions

    def is_valid_ring_move(
        self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]
    ) -> bool:
        """링 이동이 유효한지 확인 (간소화된 규칙)"""
        if not self.is_valid_position(to_pos) or not self.is_empty_position(to_pos):
            return False

        # 같은 위치로 이동 불가
        if from_pos == to_pos:
            return False

        # 직선 이동만 가능
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]

        # 8방향 이동 (수평, 수직, 대각선)
        if not (dx == 0 or dy == 0 or abs(dx) == abs(dy)):
            return False

        # 경로상의 장애물 체크
        steps = max(abs(dx), abs(dy))
        step_x = 0 if dx == 0 else dx // abs(dx)
        step_y = 0 if dy == 0 else dy // abs(dy)

        for i in range(1, steps):
            check_pos = (from_pos[0] + i * step_x, from_pos[1] + i * step_y)
            
            if not self.is_valid_position(check_pos):
                return False

            # 링이 있으면 이동 불가
            if any(check_pos in ring_set for ring_set in self.ring_positions.values()):
                return False

            # 마커가 있으면 그 다음 칸까지만 이동 가능
            if any(check_pos in marker_set for marker_set in self.marker_positions.values()):
                if i == steps - 1:  # 마지막 칸이 마커 다음 칸이 아니면
                    return False

        return True

    def step(self, action: YinshAction) -> bool:
        """액션을 실행하고 게임 상태를 업데이트"""
        if action not in self.get_valid_actions():
            return False

        # 링 이동 실행
        self._move_ring(action.from_pos, action.to_pos)

        self.game_history.append(action)
        self.move_count += 1

        # 게임 종료 조건 확인
        self._check_game_end()

        # 플레이어 교체
        if not self.done:
            self.current_player = (
                Color.BLACK if self.current_player == Color.WHITE else Color.WHITE
            )

        return True

    def _move_ring(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]):
        """링 이동 실행"""
        # 링 위치 업데이트
        self.ring_positions[self.current_player].remove(from_pos)
        self.ring_positions[self.current_player].add(to_pos)

        # 보드 업데이트
        x1, y1 = from_pos
        x2, y2 = to_pos
        self.board[x1, y1] = 0
        self.board[x2, y2] = self.current_player.value

        # 경로상의 마커 뒤집기
        self._flip_markers_on_path(from_pos, to_pos)

        # 라인 완성 확인 및 제거
        self._check_and_remove_lines()

    def _flip_markers_on_path(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]):
        """이동 경로상의 마커들을 뒤집기"""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        
        if dx == 0 and dy == 0:
            return

        steps = max(abs(dx), abs(dy))
        step_x = 0 if dx == 0 else dx // abs(dx)
        step_y = 0 if dy == 0 else dy // abs(dy)

        for i in range(1, steps):
            check_pos = (from_pos[0] + i * step_x, from_pos[1] + i * step_y)
            
            # 마커가 있으면 뒤집기
            for color in [Color.WHITE, Color.BLACK]:
                if check_pos in self.marker_positions[color]:
                    self.marker_positions[color].remove(check_pos)
                    self.marker_positions[-color].add(check_pos)
                    break

    def _check_and_remove_lines(self):
        """완성된 라인 확인 및 제거"""
        # 간소화: 5개 연속 마커가 있으면 제거
        lines = self._find_lines(self.current_player)
        
        for line in lines:
            # 라인의 마커들 제거
            for pos in line:
                self.marker_positions[self.current_player].discard(pos)
                x, y = pos
                self.board[x, y] = 0

    def _find_lines(self, color: Color) -> List[List[Tuple[int, int]]]:
        """특정 색깔의 완성된 라인들 찾기"""
        lines = []
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # 수평, 수직, 대각선
        
        for pos in self.marker_positions[color]:
            for dx, dy in directions:
                line = [pos]
                
                # 한 방향으로 4개 더 확인
                for i in range(1, 5):
                    check_pos = (pos[0] + i * dx, pos[1] + i * dy)
                    if check_pos in self.marker_positions[color]:
                        line.append(check_pos)
                    else:
                        break
                
                # 반대 방향으로도 확인
                for i in range(1, 5):
                    check_pos = (pos[0] - i * dx, pos[1] - i * dy)
                    if check_pos in self.marker_positions[color]:
                        line.insert(0, check_pos)
                    else:
                        break
                
                # 5개 이상이면 라인 완성
                if len(line) >= 5:
                    lines.append(line[:5])  # 최대 5개까지만

        return lines

    def _check_game_end(self):
        """게임 종료 조건 확인"""
        # 간소화: 링이 3개 이하가 되면 게임 종료
        if len(self.ring_positions[Color.WHITE]) <= 2:
            self.done = True
            self.winner = Color.BLACK
        elif len(self.ring_positions[Color.BLACK]) <= 2:
            self.done = True
            self.winner = Color.WHITE
        elif self.move_count >= config.MAX_GAME_MOVES:
            self.done = True
            self.winner = Color.EMPTY  # 무승부

    def is_game_over(self) -> bool:
        """게임이 종료되었는지 확인"""
        return self.done

    def get_winner(self) -> Optional[Color]:
        """승자 반환"""
        return self.winner

    def get_state_tensor(self) -> np.ndarray:
        """게임 상태를 신경망 입력용 텐서로 변환"""
        # 간소화된 상태 표현: 3채널 (흰색 링, 검은색 링, 마커)
        state = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        
        # 흰색 링
        for pos in self.ring_positions[Color.WHITE]:
            x, y = pos
            state[0, x, y] = 1.0
            
        # 검은색 링
        for pos in self.ring_positions[Color.BLACK]:
            x, y = pos
            state[1, x, y] = 1.0
            
        # 마커들 (흰색 + 검은색)
        for color in [Color.WHITE, Color.BLACK]:
            for pos in self.marker_positions[color]:
                x, y = pos
                state[2, x, y] = color.value

        return state

    def get_state_string(self) -> str:
        """게임 상태를 문자열로 표현"""
        return f"Player: {self.current_player}, White Rings: {len(self.ring_positions[Color.WHITE])}, Black Rings: {len(self.ring_positions[Color.BLACK])}"

    def copy(self):
        """게임 환경 복사"""
        new_env = YinshEnv()
        new_env.board = self.board.copy()
        new_env.ring_positions = {
            color: positions.copy() for color, positions in self.ring_positions.items()
        }
        new_env.marker_positions = {
            color: positions.copy() for color, positions in self.marker_positions.items()
        }
        new_env.current_player = self.current_player
        new_env.rings_removed = self.rings_removed.copy()
        new_env.done = self.done
        new_env.winner = self.winner
        new_env.game_history = self.game_history.copy()
        new_env.move_count = self.move_count
        return new_env

    def get_turn_state(self) -> str:
        """현재 턴 상태 반환"""
        return f"Turn {self.move_count}: {self.current_player}"
