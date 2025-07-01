# env.py - YINSH Game Environment

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
        action_type: str,
        from_pos: Optional[Tuple[int, int]] = None,
        to_pos: Optional[Tuple[int, int]] = None,
        remove_positions: Optional[List[Tuple[int, int]]] = None,
    ):
        self.action_type = (
            action_type  # "PLACE_RING", "MOVE_RING", "REMOVE_MARKERS", "REMOVE_RING"
        )
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.remove_positions = remove_positions or []

    def __str__(self):
        if self.action_type == "PLACE_RING":
            return f"PLACE_RING({self.to_pos})"
        elif self.action_type == "MOVE_RING":
            return f"MOVE_RING({self.from_pos}->{self.to_pos})"
        elif self.action_type == "REMOVE_MARKERS":
            return f"REMOVE_MARKERS({self.remove_positions})"
        elif self.action_type == "REMOVE_RING":
            return f"REMOVE_RING({self.to_pos})"
        return f"{self.action_type}"

    def __hash__(self):
        return hash(
            (self.action_type, self.from_pos, self.to_pos, tuple(self.remove_positions))
        )

    def __eq__(self, other):
        if not isinstance(other, YinshAction):
            return False
        return (
            self.action_type == other.action_type
            and self.from_pos == other.from_pos
            and self.to_pos == other.to_pos
            and self.remove_positions == other.remove_positions
        )


class YinshEnv:
    def __init__(self):
        self.board_size = config.BOARD_SIZE
        # YINSH 육각형 보드의 유효한 좌표들 (-5, -5) ~ (5, 5)
        self.valid_points: Set[Tuple[int, int]] = {
            (-5, -4),
            (-5, -3),
            (-5, -2),
            (-5, -1),
            (-4, -5),
            (-4, -4),
            (-4, -3),
            (-4, -2),
            (-4, -1),
            (-4, 0),
            (-4, 1),
            (-3, -5),
            (-3, -4),
            (-3, -3),
            (-3, -2),
            (-3, -1),
            (-3, 0),
            (-3, 1),
            (-3, 2),
            (-2, -5),
            (-2, -4),
            (-2, -3),
            (-2, -2),
            (-2, -1),
            (-2, 0),
            (-2, 1),
            (-2, 2),
            (-2, 3),
            (-1, -5),
            (-1, -4),
            (-1, -3),
            (-1, -2),
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (-1, 2),
            (-1, 3),
            (-1, 4),
            (0, -4),
            (0, -3),
            (0, -2),
            (0, -1),
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (1, -4),
            (1, -3),
            (1, -2),
            (1, -1),
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (2, -3),
            (2, -2),
            (2, -1),
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (3, -2),
            (3, -1),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (3, 5),
            (4, -1),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (5, 1),
            (5, 2),
            (5, 3),
            (5, 4),
        }
        self.reset()

    def reset(self):
        """게임을 초기 상태로 리셋"""
        # 11x11 보드 초기화 (육각형 모양이지만 사각형 배열로 구현)
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.ring_positions = {Color.WHITE: set(), Color.BLACK: set()}
        self.marker_positions = {Color.WHITE: set(), Color.BLACK: set()}

        self.current_player = Color.WHITE
        self.phase = "PLACE_RINGS"  # "PLACE_RINGS", "MAIN_GAME"
        self.rings_placed = {Color.WHITE: 0, Color.BLACK: 0}
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
        # pos는 배열 인덱스 (0~10, 0~10)
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
        """현재 상태에서 유효한 액션들을 반환"""
        actions = []

        if self.phase == "PLACE_RINGS":
            # 링 배치 단계 - 마커가 있는 곳에는 링을 둘 수 없음
            for x in range(self.board_size):
                for y in range(self.board_size):
                    pos = (x, y)
                    if self.is_empty_position(pos) and not self._has_marker_at(pos):
                        actions.append(YinshAction("PLACE_RING", to_pos=pos))
        else:
            # 메인 게임 단계
            for ring_pos in self.ring_positions[self.current_player]:
                for x in range(self.board_size):
                    for y in range(self.board_size):
                        to_pos = (x, y)
                        if self.is_valid_ring_move(ring_pos, to_pos):
                            actions.append(
                                YinshAction(
                                    "MOVE_RING", from_pos=ring_pos, to_pos=to_pos
                                )
                            )

        return actions

    def is_valid_ring_move(
        self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]
    ) -> bool:
        """링 이동이 유효한지 확인 (YINSH 규칙)"""
        if not self.is_valid_position(to_pos) or not self.is_empty_position(to_pos):
            return False

        # 직선 이동만 가능
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]

        if dx == 0 and dy == 0:
            return False

        # 6방향 (수평, 수직, 대각선) 체크
        if not (dx == 0 or dy == 0 or abs(dx) == abs(dy)):
            return False

        # YINSH 이동 규칙 체크
        steps = max(abs(dx), abs(dy))
        step_x = 0 if dx == 0 else dx // abs(dx)
        step_y = 0 if dy == 0 else dy // abs(dy)

        # 경로상의 장애물 체크
        for i in range(1, steps):
            check_pos = (from_pos[0] + i * step_x, from_pos[1] + i * step_y)

            if not self.is_valid_position(check_pos):
                return False

            # 링이 있으면 그 위치까지만 이동 가능
            if any(check_pos in ring_set for ring_set in self.ring_positions.values()):
                # 링이 있는 위치까지만 이동 가능한지 확인
                if check_pos != to_pos:
                    return False
                break

            # 마커가 있으면 그 다음 칸까지 이동 가능
            if any(
                check_pos in marker_set for marker_set in self.marker_positions.values()
            ):
                # 마커가 있는 칸에 링을 둘 수 없음
                if check_pos == to_pos:
                    return False
                # 마커 다음 칸까지만 이동 가능
                if i == steps - 1:  # 마지막 칸이 마커 다음 칸이 아니면
                    return False

        return True

    def step(self, action: YinshAction) -> bool:
        """액션을 실행하고 게임 상태를 업데이트"""
        if action not in self.get_valid_actions():
            return False

        if action.action_type == "PLACE_RING":
            self._place_ring(action.to_pos)
        elif action.action_type == "MOVE_RING":
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

    def _place_ring(self, pos: Tuple[int, int]):
        """링을 배치"""
        x, y = pos
        self.board[x, y] = self.current_player.value
        self.ring_positions[self.current_player].add(pos)
        self.rings_placed[self.current_player] += 1

        # 모든 링이 배치되면 메인 게임으로 전환
        if (
            self.rings_placed[Color.WHITE] == config.RINGS_PER_PLAYER
            and self.rings_placed[Color.BLACK] == config.RINGS_PER_PLAYER
        ):
            self.phase = "MAIN_GAME"

    def _move_ring(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]):
        """링을 이동하고 경로상의 마커들을 뒤집음"""
        # 링 이동
        x_from, y_from = from_pos
        x_to, y_to = to_pos

        self.board[x_from, y_from] = 0
        self.board[x_to, y_to] = self.current_player.value

        self.ring_positions[self.current_player].remove(from_pos)
        self.ring_positions[self.current_player].add(to_pos)

        # 시작 위치에 마커 배치
        self.marker_positions[self.current_player].add(from_pos)

        # 경로상의 마커들 뒤집기
        self._flip_markers_on_path(from_pos, to_pos)

        # 라인 완성 체크
        self._check_and_remove_lines()

    def _flip_markers_on_path(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]):
        """링 이동 경로상의 마커들을 뒤집음"""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]
        steps = max(abs(dx), abs(dy))

        if steps == 0:
            return

        step_x = 0 if dx == 0 else dx // abs(dx)
        step_y = 0 if dy == 0 else dy // abs(dy)

        for i in range(1, steps):
            check_pos = (from_pos[0] + i * step_x, from_pos[1] + i * step_y)

            # 마커가 있으면 뒤집기
            for color in [Color.WHITE, Color.BLACK]:
                if check_pos in self.marker_positions[color]:
                    self.marker_positions[color].remove(check_pos)
                    other_color = Color.BLACK if color == Color.WHITE else Color.WHITE
                    self.marker_positions[other_color].add(check_pos)
                    break

    def _check_and_remove_lines(self):
        """5개 연속 라인이 있는지 체크하고 제거"""
        for color in [Color.WHITE, Color.BLACK]:
            lines = self._find_lines(color)
            if lines:
                # 첫 번째 라인 제거 (간단화)
                line = lines[0]
                for pos in line:
                    self.marker_positions[color].remove(pos)

                # 링 제거 (플레이어가 선택해야 하지만 간단화)
                if self.ring_positions[color]:
                    ring_to_remove = list(self.ring_positions[color])[0]
                    self.ring_positions[color].remove(ring_to_remove)
                    x, y = ring_to_remove
                    self.board[x, y] = 0
                    self.rings_removed[color] += 1

    def _find_lines(self, color: Color) -> List[List[Tuple[int, int]]]:
        """해당 색깔의 5개 연속 라인을 찾음"""
        lines = []
        markers = self.marker_positions[color]

        # 6방향으로 라인 체크
        directions = [(1, 0), (0, 1), (1, 1), (1, -1), (-1, 1), (-1, 0)]

        for start_pos in markers:
            for dx, dy in directions:
                line = [start_pos]
                current_pos = start_pos

                # 한 방향으로 계속 체크
                for _ in range(4):  # 5개 라인이므로 4번 더 체크
                    next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                    if next_pos in markers:
                        line.append(next_pos)
                        current_pos = next_pos
                    else:
                        break

                if len(line) >= config.LINE_LENGTH_TO_WIN:
                    lines.append(line[: config.LINE_LENGTH_TO_WIN])

        return lines

    def _check_game_end(self):
        """게임 종료 조건 체크"""
        if self.rings_removed[Color.WHITE] >= config.RINGS_TO_WIN:
            self.done = True
            self.winner = Color.WHITE
        elif self.rings_removed[Color.BLACK] >= config.RINGS_TO_WIN:
            self.done = True
            self.winner = Color.BLACK
        elif self.move_count >= config.MAX_GAME_MOVES:
            self.done = True
            self.winner = None  # 무승부

    def is_game_over(self) -> bool:
        """게임 종료 조건 확인"""
        # 링 배치 단계에서는 게임이 종료될 수 없음
        if "RING_PLACE" in self.get_turn_state():
            return False

        # 링 이동 단계 이후에만 승리 조건 확인
        if self.rings_removed[Color.WHITE] >= config.RINGS_TO_WIN:
            return True
        if self.rings_removed[Color.BLACK] >= config.RINGS_TO_WIN:
            return True

        return False

    def get_winner(self) -> Optional[Color]:
        """승자 반환"""
        if not self.is_game_over():
            return None

        white_rings = sum(1 for pos in self.ring_positions[Color.WHITE] if pos)
        black_rings = sum(1 for pos in self.ring_positions[Color.BLACK] if pos)

        if white_rings <= 2:
            return Color.BLACK  # 백이 링을 3개 제거당했으므로 흑 승리
        elif black_rings <= 2:
            return Color.WHITE  # 흑이 링을 3개 제거당했으므로 백 승리

        return None  # 무승부 (이론적으로 발생하지 않음)

    def get_state_tensor(self) -> np.ndarray:
        """현재 상태를 신경망 입력용 텐서로 변환"""
        state = np.zeros((11, self.board_size, self.board_size), dtype=np.float32)

        # 채널 0-1: 현재 플레이어의 링과 마커
        current_color = self.current_player
        for pos in self.ring_positions[current_color]:
            x, y = pos
            state[0, x, y] = 1.0
        for pos in self.marker_positions[current_color]:
            x, y = pos
            state[1, x, y] = 1.0

        # 채널 2-3: 상대 플레이어의 링과 마커
        opponent_color = Color.BLACK if current_color == Color.WHITE else Color.WHITE
        for pos in self.ring_positions[opponent_color]:
            x, y = pos
            state[2, x, y] = 1.0
        for pos in self.marker_positions[opponent_color]:
            x, y = pos
            state[3, x, y] = 1.0

        # 채널 4: 현재 플레이어 (전체 보드에 색칠)
        state[4, :, :] = 1.0 if current_color == Color.WHITE else 0.0

        # 채널 5: 게임 단계
        state[5, :, :] = 1.0 if self.phase == "PLACE_RINGS" else 0.0

        # 채널 6-7: 각 플레이어가 배치한 링 개수
        state[6, :, :] = self.rings_placed[Color.WHITE] / config.RINGS_PER_PLAYER
        state[7, :, :] = self.rings_placed[Color.BLACK] / config.RINGS_PER_PLAYER

        # 채널 8-9: 각 플레이어가 제거한 링 개수
        state[8, :, :] = self.rings_removed[Color.WHITE] / config.RINGS_TO_WIN
        state[9, :, :] = self.rings_removed[Color.BLACK] / config.RINGS_TO_WIN

        # 채널 10: 유효한 보드 위치
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.is_valid_position((x, y)):
                    state[10, x, y] = 1.0

        return state

    def get_state_string(self) -> str:
        """상태를 문자열로 변환 (해싱용)"""
        return str(
            {
                "rings_white": sorted(list(self.ring_positions[Color.WHITE])),
                "rings_black": sorted(list(self.ring_positions[Color.BLACK])),
                "markers_white": sorted(list(self.marker_positions[Color.WHITE])),
                "markers_black": sorted(list(self.marker_positions[Color.BLACK])),
                "current_player": self.current_player.value,
                "phase": self.phase,
                "rings_placed": dict(self.rings_placed),
                "rings_removed": dict(self.rings_removed),
            }
        )

    def copy(self):
        """환경의 깊은 복사본을 생성"""
        new_env = YinshEnv()
        new_env.board = self.board.copy()
        new_env.ring_positions = {
            Color.WHITE: self.ring_positions[Color.WHITE].copy(),
            Color.BLACK: self.ring_positions[Color.BLACK].copy(),
        }
        new_env.marker_positions = {
            Color.WHITE: self.marker_positions[Color.WHITE].copy(),
            Color.BLACK: self.marker_positions[Color.BLACK].copy(),
        }
        new_env.current_player = self.current_player
        new_env.phase = self.phase
        new_env.rings_placed = self.rings_placed.copy()
        new_env.rings_removed = self.rings_removed.copy()
        new_env.done = self.done
        new_env.winner = self.winner
        new_env.game_history = self.game_history.copy()
        new_env.move_count = self.move_count
        return new_env

    def get_turn_state(self) -> str:
        """현재 턴 상태 반환 (YINSH_TURN_PROCESSING.md 참고)"""
        # 간단한 구현: 링 배치 단계 -> 링 이동 단계
        white_rings = sum(1 for pos in self.ring_positions[Color.WHITE] if pos)
        black_rings = sum(1 for pos in self.ring_positions[Color.BLACK] if pos)

        if white_rings < 5 or black_rings < 5:
            # 링 배치 단계
            if self.current_player == Color.WHITE:
                return "WHITE_RING_PLACE"
            else:
                return "BLACK_RING_PLACE"
        else:
            # 링 이동 단계
            if self.current_player == Color.WHITE:
                return "WHITE_RING_MOVE"
            else:
                return "BLACK_RING_MOVE"
