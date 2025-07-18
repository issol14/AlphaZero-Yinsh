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


class GamePhase(Enum):
    PLACE_RINGS = "PLACE_RINGS"
    MAIN_GAME = "MAIN_GAME"
    LINE_REMOVAL = "LINE_REMOVAL"  # 새로 추가


class YinshAction:
    def __init__(
        self,
        action_type: str,
        from_pos: Optional[Tuple[int, int]] = None,
        to_pos: Optional[Tuple[int, int]] = None,
        remove_positions: Optional[List[Tuple[int, int]]] = None,
        remove_line_positions: Optional[List[Tuple[int, int]]] = None,  # 추가
        remove_ring_position: Optional[Tuple[int, int]] = None,  # 추가
    ):
        self.action_type = (
            action_type  # "PLACE_RING", "MOVE_RING", "REMOVE_MARKERS", "REMOVE_RING", "REMOVE_LINE"
        )
        self.from_pos = from_pos
        self.to_pos = to_pos
        self.remove_positions = remove_positions or []
        self.remove_line_positions = remove_line_positions or []  # 추가
        self.remove_ring_position = remove_ring_position  # 추가

    def __str__(self):
        if self.action_type == "PLACE_RING":
            return f"PLACE_RING({self.to_pos})"
        elif self.action_type == "MOVE_RING":
            return f"MOVE_RING({self.from_pos}->{self.to_pos})"
        elif self.action_type == "REMOVE_MARKERS":
            return f"REMOVE_MARKERS({self.remove_positions})"
        elif self.action_type == "REMOVE_RING":
            return f"REMOVE_RING({self.to_pos})"
        elif self.action_type == "REMOVE_LINE":
            return f"REMOVE_LINE(markers={self.remove_line_positions}, ring={self.remove_ring_position})"
        return f"{self.action_type}"

    def __hash__(self):
        return hash(
            (self.action_type, self.from_pos, self.to_pos, 
             tuple(self.remove_positions), tuple(self.remove_line_positions), 
             self.remove_ring_position)
        )

    def __eq__(self, other):
        if not isinstance(other, YinshAction):
            return False
        return (
            self.action_type == other.action_type
            and self.from_pos == other.from_pos
            and self.to_pos == other.to_pos
            and self.remove_positions == other.remove_positions
            and self.remove_line_positions == other.remove_line_positions
            and self.remove_ring_position == other.remove_ring_position
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
        self.phase = GamePhase.PLACE_RINGS  # GamePhase enum 사용
        self.rings_placed = {Color.WHITE: 0, Color.BLACK: 0}
        self.rings_removed = {Color.WHITE: 0, Color.BLACK: 0}
        self.done = False
        self.winner = None

        # 마커 풀 관리는 실시간 계산으로 처리 (property 사용)

        # 라인 제거 시스템 (새로 추가)
        self.pending_line_removals = []  # 제거 대기 중인 라인들
        self.line_removal_player = None  # 라인을 제거해야 하는 플레이어

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

    @property
    def markers_on_board(self) -> int:
        """현재 보드에 놓인 마커 수 (실시간 계산)"""
        return len(self.marker_positions[Color.WHITE]) + len(self.marker_positions[Color.BLACK])

    @property
    def markers_in_pool(self) -> int:
        """사용 가능한 마커 수 (실시간 계산)"""
        return config.TOTAL_MARKERS - self.markers_on_board

    def _place_marker(self, pos: Tuple[int, int], color: Color) -> bool:
        """마커 배치 (풀에서 가져오기)"""
        if self.markers_in_pool <= 0:
            return False
        self.marker_positions[color].add(pos)
        return True
    
    def _remove_markers(self, positions: List[Tuple[int, int]], color: Color):
        """마커들을 제거하고 풀로 반환"""
        for pos in positions:
            if pos in self.marker_positions[color]:
                self.marker_positions[color].remove(pos)
        self.markers_in_pool += len(positions)
        self.markers_on_board -= len(positions)

    def _is_valid_hex_direction(self, dx: int, dy: int) -> bool:
        """육각형 보드의 유효한 6방향인지 확인"""
        if dx == 0 and dy == 0:
            return False
            
        # 방향 벡터 정규화
        gcd = abs(dx) if dy == 0 else abs(dy) if dx == 0 else min(abs(dx), abs(dy))
        if gcd == 0:
            return False
            
        norm_dx = dx // gcd if dx != 0 else 0
        norm_dy = dy // gcd if dy != 0 else 0
        
        return (norm_dx, norm_dy) in config.VALID_HEX_DIRECTIONS

    def get_valid_actions(self) -> List[YinshAction]:
        """현재 상태에서 유효한 액션들을 반환"""
        actions = []

        if self.phase == GamePhase.PLACE_RINGS:
            # 링 배치 단계 - 마커가 있는 곳에는 링을 둘 수 없음
            for x in range(self.board_size):
                for y in range(self.board_size):
                    pos = (x, y)
                    if self.is_empty_position(pos) and not self._has_marker_at(pos):
                        actions.append(YinshAction("PLACE_RING", to_pos=pos))
        
        elif self.phase == GamePhase.LINE_REMOVAL:
            # 라인 제거 액션들
            if self.pending_line_removals:
                line = self.pending_line_removals[0]
                # 5개 연속 라인에서 가능한 5개 조합들 생성
                if len(line) >= config.LINE_LENGTH_TO_WIN:
                    for i in range(len(line) - config.LINE_LENGTH_TO_WIN + 1):
                        remove_positions = line[i:i+config.LINE_LENGTH_TO_WIN]
                        # 제거할 링 선택
                        for ring_pos in self.ring_positions[self.current_player]:
                            actions.append(YinshAction(
                                "REMOVE_LINE",
                                remove_line_positions=remove_positions,
                                remove_ring_position=ring_pos
                            ))
        
        elif self.phase == GamePhase.MAIN_GAME:
            # 메인 게임 단계 - 링 이동
            for ring_pos in self.ring_positions[self.current_player]:
                valid_moves = self.get_valid_ring_moves(ring_pos)
                for to_pos in valid_moves:
                    actions.append(YinshAction(
                        "MOVE_RING", 
                        from_pos=ring_pos, 
                        to_pos=to_pos
                    ))

        return actions

    def is_valid_ring_move(
        self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]
    ) -> bool:
        """링 이동이 유효한지 확인 (YINSH 규칙) - 수정된 버전"""
        if not self.is_valid_position(to_pos) or not self.is_empty_position(to_pos):
            return False

        # 직선 이동만 가능
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]

        if dx == 0 and dy == 0:
            return False

        # 올바른 6방향 체크 (수정됨)
        if not self._is_valid_hex_direction(dx, dy):
            return False

        # YINSH 이동 규칙 체크는 _find_landing_position에서 처리
        return to_pos in self.get_valid_ring_moves(from_pos)

    def _find_landing_position(self, from_pos: Tuple[int, int], direction: Tuple[int, int]) -> List[Tuple[int, int]]:
        """주어진 방향으로 링이 착지할 수 있는 위치들 찾기"""
        dx, dy = direction
        landing_positions = []
        current_pos = from_pos
        
        # 1단계: 연속된 빈 공간들 건너뛰기
        while True:
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
       
            if not self.is_valid_position(next_pos):
                break
                
            # 링이 있으면 건너뛸 수 없음
            if any(next_pos in ring_set for ring_set in self.ring_positions.values()):
                break
                
            # 마커가 있으면 마커 구간 시작
            if any(next_pos in marker_set for marker_set in self.marker_positions.values()):
                break
                
            # 빈 공간이면 착지 가능하고 계속 진행
            landing_positions.append(next_pos)
            current_pos = next_pos
        
        # 2단계: 마커들 건너뛰기
        marker_start = (current_pos[0] + dx, current_pos[1] + dy)
        if (self.is_valid_position(marker_start) and 
            any(marker_start in marker_set for marker_set in self.marker_positions.values())):
            
            # 연속된 마커들을 모두 건너뛰기
            current_pos = marker_start
            while (self.is_valid_position(current_pos) and 
                   any(current_pos in marker_set for marker_set in self.marker_positions.values())):
                current_pos = (current_pos[0] + dx, current_pos[1] + dy)
            
            # 마커 다음 첫 번째 빈 공간에 착지
            if (self.is_valid_position(current_pos) and 
                self.is_empty_position(current_pos)):
                landing_positions.append(current_pos)
        
        return landing_positions

    def get_valid_ring_moves(self, ring_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """해당 링이 이동할 수 있는 모든 위치들 반환"""
        valid_moves = []
        
        # 6방향으로 체크
        for direction in config.VALID_HEX_DIRECTIONS:
            landing_positions = self._find_landing_position(ring_pos, direction)
            valid_moves.extend(landing_positions)
            
        return valid_moves

    def step(self, action: YinshAction) -> bool:
        """액션을 실행하고 게임 상태를 업데이트"""
        if action not in self.get_valid_actions():
            return False

        if action.action_type == "PLACE_RING":
            self._place_ring(action.to_pos)
        elif action.action_type == "MOVE_RING":
            self._move_ring(action.from_pos, action.to_pos)
        elif action.action_type == "REMOVE_LINE":
            self._remove_line_and_ring(action.remove_line_positions, action.remove_ring_position)

        self.game_history.append(action)
        self.move_count += 1

        # 게임 종료 조건 확인
        self._check_game_end()

        # 플레이어 교체 (수정된 로직)
        if not self.done:
            if self.phase == GamePhase.LINE_REMOVAL:
                # 라인 제거 완료 후 원래 게임으로 복귀하고 다른 색 라인도 체크
                self.phase = GamePhase.MAIN_GAME
                self._check_for_remaining_lines()
                # 라인 제거 후에는 플레이어 변경하지 않음 (같은 플레이어 계속)
            else:
                # 일반적인 플레이어 교체
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
            self.phase = GamePhase.MAIN_GAME

    def _move_ring(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]):
        """링을 이동하고 경로상의 마커들을 뒤집음 (수정된 버전)"""
        # 링 이동
        x_from, y_from = from_pos
        x_to, y_to = to_pos

        self.board[x_from, y_from] = 0
        self.board[x_to, y_to] = self.current_player.value

        self.ring_positions[self.current_player].remove(from_pos)
        self.ring_positions[self.current_player].add(to_pos)

        # 시작 위치에 마커 배치 (마커 풀 관리)
        if not self._place_marker(from_pos, self.current_player):
            # 마커 부족 시 게임 종료 처리
            self._handle_marker_exhaustion()
            return

        # 경로상의 마커들 뒤집기
        self._flip_markers_on_path(from_pos, to_pos)

        # 5연속 라인 체크 (새로운 시스템)
        self._check_for_lines_and_handle()

    def _check_for_lines_and_handle(self):
        """5연속 라인 체크 및 처리 순서 관리"""
        white_lines = self._find_lines(Color.WHITE)
        black_lines = self._find_lines(Color.BLACK)
        
        # 양쪽 모두 5연속이 생겼을 때
        if white_lines and black_lines:
            # 방금 움직인 플레이어부터 처리
            if self.current_player == Color.WHITE:
                self._initiate_line_removal(Color.WHITE, white_lines[0])
            else:
                self._initiate_line_removal(Color.BLACK, black_lines[0])
        # 한쪽만 5연속
        elif white_lines:
            self._initiate_line_removal(Color.WHITE, white_lines[0])
        elif black_lines:
            self._initiate_line_removal(Color.BLACK, black_lines[0])

    def _initiate_line_removal(self, color: Color, line: List[Tuple[int, int]]):
        """라인 제거 프로세스 시작"""
        self.phase = GamePhase.LINE_REMOVAL
        self.line_removal_player = color
        self.pending_line_removals = [line]
        # 현재 플레이어를 라인 제거해야 하는 플레이어로 변경
        self.current_player = color

    def _handle_marker_exhaustion(self):
        """마커 51개 소진 시 처리"""
        self.done = True
        
        # 제거한 링 개수로 승자 결정
        white_removed = self.rings_removed[Color.WHITE]
        black_removed = self.rings_removed[Color.BLACK]
        
        if white_removed > black_removed:
            self.winner = Color.WHITE
        elif black_removed > white_removed:
            self.winner = Color.BLACK
        else:
            self.winner = None  # 무승부

    def _remove_line_and_ring(self, remove_positions: List[Tuple[int, int]], remove_ring_position: Optional[Tuple[int, int]]):
        """라인을 제거하고 해당 링을 제거"""
        # 라인 제거 (마커 풀 상태는 property로 자동 계산됨)
        for pos in remove_positions:
            if pos in self.marker_positions[self.current_player]:
                self.marker_positions[self.current_player].remove(pos)

        # 링 제거
        if remove_ring_position and remove_ring_position in self.ring_positions[self.current_player]:
            self.ring_positions[self.current_player].remove(remove_ring_position)
            x, y = remove_ring_position
            self.board[x, y] = 0
            self.rings_removed[self.current_player] += 1

    def _check_for_remaining_lines(self):
        """라인 제거 후에도 남아있는 라인이 있는지 확인"""
        for color in [Color.WHITE, Color.BLACK]:
            lines = self._find_lines(color)
            if lines:
                # 라인이 남아있으면 다시 라인 제거 단계로 전환
                self.phase = GamePhase.LINE_REMOVAL
                self.line_removal_player = color
                break

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

    def _find_lines(self, color: Color) -> List[List[Tuple[int, int]]]:
        """해당 색깔의 5개 연속 라인을 찾음 (수정된 버전)"""
        lines = []
        markers = self.marker_positions[color]

        # 올바른 6방향
        for start_pos in markers:
            for dx, dy in config.VALID_HEX_DIRECTIONS:
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
                    lines.append(line[:config.LINE_LENGTH_TO_WIN])

        return lines

    def _check_game_end(self):
        """게임 종료 조건 체크 (완전한 버전)"""
        # 1. 링 3개 제거 조건
        if self.rings_removed[Color.WHITE] >= config.RINGS_TO_WIN:
            self.done = True
            self.winner = Color.WHITE
            return
        elif self.rings_removed[Color.BLACK] >= config.RINGS_TO_WIN:
            self.done = True
            self.winner = Color.BLACK
            return
        
        # 2. 마커 소진 조건
        if self.markers_in_pool <= 0:
            self._handle_marker_exhaustion()
            return
        
        # 3. 최대 이동 수 초과 (무승부)
        if self.move_count >= config.MAX_GAME_MOVES:
            self.done = True
            self.winner = None
            return

    def is_game_over(self) -> bool:
        """게임 종료 조건 확인"""
        # 링 배치 단계에서는 게임이 종료될 수 없음
        if self.phase == GamePhase.PLACE_RINGS:
            return False

        # 링 이동 단계 이후에만 승리 조건 확인
        if self.rings_removed[Color.WHITE] >= config.RINGS_TO_WIN:
            return True
        if self.rings_removed[Color.BLACK] >= config.RINGS_TO_WIN:
            return True
        
        # 마커 소진 조건
        if self.markers_in_pool <= 0:
            return True

        return self.done

    def get_winner(self) -> Optional[Color]:
        """승자 반환 (수정된 버전)"""
        if not self.is_game_over():
            return None

        return self.winner  # _check_game_end에서 이미 설정됨

    def get_state_tensor(self) -> np.ndarray:
        """현재 상태를 신경망 입력용 텐서로 변환 (확장된 버전)"""
        state = np.zeros((13, self.board_size, self.board_size), dtype=np.float32)  # 11->13 채널로 확장

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

        # 채널 5: 게임 단계 (수정됨)
        phase_encoding = {
            GamePhase.PLACE_RINGS: 0.0,
            GamePhase.MAIN_GAME: 0.5,
            GamePhase.LINE_REMOVAL: 1.0
        }
        state[5, :, :] = phase_encoding[self.phase]

        # 채널 6-7: 각 플레이어가 배치한 링 개수
        state[6, :, :] = self.rings_placed[Color.WHITE] / config.RINGS_PER_PLAYER
        state[7, :, :] = self.rings_placed[Color.BLACK] / config.RINGS_PER_PLAYER

        # 채널 8-9: 각 플레이어가 제거한 링 개수
        state[8, :, :] = self.rings_removed[Color.WHITE] / config.RINGS_TO_WIN
        state[9, :, :] = self.rings_removed[Color.BLACK] / config.RINGS_TO_WIN

        # 채널 10: 마커 풀 상태 (새로 추가)
        state[10, :, :] = self.markers_in_pool / config.TOTAL_MARKERS

        # 채널 11: 보드 위 마커 수 (새로 추가)
        state[11, :, :] = self.markers_on_board / config.TOTAL_MARKERS

        # 채널 12: 유효한 보드 위치
        for x in range(self.board_size):
            for y in range(self.board_size):
                if self.is_valid_position((x, y)):
                    state[12, x, y] = 1.0

        return state

    def get_state_string(self) -> str:
        """상태를 문자열로 변환 (해싱용) - 확장된 버전"""
        return str(
            {
                "rings_white": sorted(list(self.ring_positions[Color.WHITE])),
                "rings_black": sorted(list(self.ring_positions[Color.BLACK])),
                "markers_white": sorted(list(self.marker_positions[Color.WHITE])),
                "markers_black": sorted(list(self.marker_positions[Color.BLACK])),
                "current_player": self.current_player.value,
                "phase": self.phase.value,  # GamePhase enum
                "rings_placed": dict(self.rings_placed),
                "rings_removed": dict(self.rings_removed),
                "markers_in_pool": self.markers_in_pool,  # 새로 추가
                "markers_on_board": self.markers_on_board,  # 새로 추가
                "pending_line_removals": self.pending_line_removals,  # 새로 추가
                "line_removal_player": self.line_removal_player.value if self.line_removal_player else None,  # 새로 추가
            }
        )

    def copy(self):
        """환경의 깊은 복사본을 생성 (확장된 버전)"""
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
        
        # 새로 추가된 필드들 (markers_in_pool, markers_on_board는 property로 자동 계산)
        new_env.pending_line_removals = self.pending_line_removals.copy()
        new_env.line_removal_player = self.line_removal_player
        
        new_env.game_history = self.game_history.copy()
        new_env.move_count = self.move_count
        return new_env

    def get_turn_state(self) -> str:
        """현재 턴 상태 반환 (확장된 버전)"""
        if self.phase == GamePhase.PLACE_RINGS:
            # 링 배치 단계
            if self.current_player == Color.WHITE:
                return "WHITE_RING_PLACE"
            else:
                return "BLACK_RING_PLACE"
        elif self.phase == GamePhase.LINE_REMOVAL:
            # 라인 제거 단계
            if self.current_player == Color.WHITE:
                return "WHITE_LINE_REMOVAL"
            else:
                return "BLACK_LINE_REMOVAL"
        else:
            # 링 이동 단계 (메인 게임)
            if self.current_player == Color.WHITE:
                return "WHITE_RING_MOVE"
            else:
                return "BLACK_RING_MOVE"
