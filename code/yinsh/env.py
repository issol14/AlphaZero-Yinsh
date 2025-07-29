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

        # 라인 제거 시스템 (전면 재설계)
        self.line_removal_queue = []  # 제거해야 하는 플레이어들의 순서 (Color 리스트)
        self.removable_lines = {Color.WHITE: [], Color.BLACK: []}  # 각 플레이어의 제거가능한 라인들
        self.move_player = Color.WHITE  # 실제로 움직인 플레이어 (라인 제거 후 턴 전환용)

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
        # markers_in_pool과 markers_on_board는 @property로 자동 계산되므로 수동 업데이트 불필요

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

    def generate_dynamic_place_ring_actions(self) -> List[YinshAction]:
        """동적으로 링 배치 액션 생성 (최적화된 버전)"""
        actions = []
        
        if not config.USE_DYNAMIC_ACTIONS:
            return self._get_static_place_ring_actions()
            
        # 실제로 배치 가능한 위치만 생성
        for hex_pos in self.valid_points:
            array_pos = self.hex_to_array_coords(hex_pos)
            if self.is_empty_position(array_pos) and not self._has_marker_at(array_pos):
                actions.append(YinshAction("PLACE_RING", to_pos=array_pos))
                
        return actions[:config.MAX_DYNAMIC_ACTIONS]
    
    def generate_dynamic_move_ring_actions(self) -> List[YinshAction]:
        """동적으로 링 이동 액션 생성 (현재 플레이어 소유 링만)"""
        actions = []
        
        if not config.USE_DYNAMIC_ACTIONS:
            return self._get_static_move_ring_actions()
            
        # 현재 플레이어의 링만 고려
        player_rings = self.ring_positions[self.current_player]
        
        for from_pos in player_rings:
            # 가능한 이동 위치 계산
            valid_moves = self.get_valid_ring_moves(from_pos)
            for to_pos in valid_moves:
                actions.append(YinshAction("MOVE_RING", from_pos=from_pos, to_pos=to_pos))
                    
        return actions[:config.MAX_DYNAMIC_ACTIONS]
    
    def generate_dynamic_remove_line_actions(self) -> List[YinshAction]:
        """동적으로 라인 제거 액션 생성 (현재 플레이어 소유 링과 라인만)"""
        actions = []
        
        if not config.USE_DYNAMIC_ACTIONS:
            return self._get_static_remove_line_actions()
            
        # 현재 플레이어의 제거 가능한 라인들
        current_lines = self.removable_lines[self.current_player]
        # 현재 플레이어의 링들
        player_rings = self.ring_positions[self.current_player]
        
        for line in current_lines:
            if len(line) >= config.LINE_LENGTH_TO_WIN:
                for i in range(len(line) - config.LINE_LENGTH_TO_WIN + 1):
                    remove_positions = line[i:i+config.LINE_LENGTH_TO_WIN]
                    # 실제 소유 링만 제거 후보로 사용
                    for ring_pos in player_rings:
                        actions.append(YinshAction(
                            "REMOVE_LINE",
                            remove_line_positions=remove_positions,
                            remove_ring_position=ring_pos
                        ))
                        
        return actions[:config.MAX_DYNAMIC_ACTIONS]
    
    def _get_static_place_ring_actions(self) -> List[YinshAction]:
        """정적 매핑 방식의 링 배치 액션 (호환성용)"""
        actions = []
        for x in range(self.board_size):
            for y in range(self.board_size):
                pos = (x, y)
                if self.is_empty_position(pos) and not self._has_marker_at(pos):
                    actions.append(YinshAction("PLACE_RING", to_pos=pos))
        return actions
    
    def _get_static_move_ring_actions(self) -> List[YinshAction]:
        """정적 매핑 방식의 링 이동 액션 (호환성용)"""
        # 현재 get_valid_actions의 MAIN_GAME 부분과 동일한 로직
        actions = []
        player_rings = self.ring_positions[self.current_player]
        for from_pos in player_rings:
            valid_moves = self.get_valid_ring_moves(from_pos)
            for to_pos in valid_moves:
                actions.append(YinshAction("MOVE_RING", from_pos=from_pos, to_pos=to_pos))
        return actions
    
    def _get_static_remove_line_actions(self) -> List[YinshAction]:
        """정적 매핑 방식의 라인 제거 액션 (호환성용)"""
        # 현재 get_valid_actions의 LINE_REMOVAL 부분과 동일한 로직  
        actions = []
        current_lines = self.removable_lines[self.current_player]
        for line in current_lines:
            if len(line) >= config.LINE_LENGTH_TO_WIN:
                for i in range(len(line) - config.LINE_LENGTH_TO_WIN + 1):
                    remove_positions = line[i:i+config.LINE_LENGTH_TO_WIN]
                    for ring_pos in self.ring_positions[self.current_player]:
                        actions.append(YinshAction(
                            "REMOVE_LINE",
                            remove_line_positions=remove_positions,
                            remove_ring_position=ring_pos
                        ))
        return actions

    def get_valid_actions(self) -> List[YinshAction]:
        """현재 상태에서 유효한 액션들을 반환 (최적화된 버전)"""
        
        if self.phase == GamePhase.PLACE_RINGS:
            return self.generate_dynamic_place_ring_actions()
        
        elif self.phase == GamePhase.LINE_REMOVAL:
            return self.generate_dynamic_remove_line_actions()
        
        elif self.phase == GamePhase.MAIN_GAME:
            return self.generate_dynamic_move_ring_actions()
        
        return []

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

        # 플레이어 교체 (새로운 라인 제거 플로우)
        if not self.done:
            if self.phase == GamePhase.LINE_REMOVAL:
                self._continue_line_removal_process()
            elif self.phase == GamePhase.PLACE_RINGS:
                # 링 배치 단계에서는 단순 교체
                self.current_player = (
                    Color.BLACK if self.current_player == Color.WHITE else Color.WHITE
                )
            # MAIN_GAME에서는 _move_ring이 _start_line_removal_process를 호출하므로
            # 여기서 추가 처리 불필요

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

        # 라인 제거 프로세스 시작 (새로운 플로우)
        self._start_line_removal_process()

    def _start_line_removal_process(self):
        """라인 제거 프로세스 시작 - 올바른 Yinsh 플로우 구현"""
        # 현재 움직인 플레이어 저장
        self.move_player = self.current_player
        
        # 모든 플레이어의 제거가능한 라인 체크
        self.removable_lines[Color.WHITE] = self._find_lines(Color.WHITE)
        self.removable_lines[Color.BLACK] = self._find_lines(Color.BLACK)
        
        # 제거 가능한 플레이어들을 큐에 추가 (현재 플레이어부터)
        self.line_removal_queue = []
        if self.removable_lines[self.current_player]:
            self.line_removal_queue.append(self.current_player)
        
        other_player = Color.BLACK if self.current_player == Color.WHITE else Color.WHITE
        if self.removable_lines[other_player]:
            self.line_removal_queue.append(other_player)
        
        # 제거할 라인이 있으면 LINE_REMOVAL 단계로 전환
        if self.line_removal_queue:
            self.phase = GamePhase.LINE_REMOVAL
            self.current_player = self.line_removal_queue[0]
        # 제거할 라인이 없으면 바로 다음 플레이어 턴으로
        else:
            self._end_line_removal_process()

    def _end_line_removal_process(self):
        """라인 제거 프로세스 종료 - 다음 플레이어 턴으로"""
        self.phase = GamePhase.MAIN_GAME
        self.line_removal_queue = []
        self.removable_lines = {Color.WHITE: [], Color.BLACK: []}
        
        # 다음 플레이어로 턴 전환
        self.current_player = Color.BLACK if self.move_player == Color.WHITE else Color.WHITE

    def _continue_line_removal_process(self):
        """라인 제거 프로세스 계속 진행"""
        # 현재 플레이어의 제거 가능한 라인 업데이트
        self.removable_lines[self.current_player] = self._find_lines(self.current_player)
        
        # 현재 플레이어에게 더 이상 제거할 라인이 없으면 큐에서 제거
        if not self.removable_lines[self.current_player]:
            if self.current_player in self.line_removal_queue:
                self.line_removal_queue.remove(self.current_player)
        
        # 큐에서 다음 플레이어 찾기
        if self.line_removal_queue:
            # 현재 플레이어가 여전히 큐에 있으면 계속, 아니면 다음 플레이어
            if self.current_player not in self.line_removal_queue:
                self.current_player = self.line_removal_queue[0]
        else:
            # 큐가 비었으면 모든 플레이어 다시 체크
            self._check_all_players_for_lines()
    
    def _check_all_players_for_lines(self):
        """모든 플레이어의 제거 가능한 라인 재검사"""
        # 모든 플레이어의 제거가능한 라인 다시 체크
        self.removable_lines[Color.WHITE] = self._find_lines(Color.WHITE)
        self.removable_lines[Color.BLACK] = self._find_lines(Color.BLACK)
        
        # 새로운 큐 구성 (현재 플레이어부터)
        self.line_removal_queue = []
        if self.removable_lines[self.current_player]:
            self.line_removal_queue.append(self.current_player)
        
        other_player = Color.BLACK if self.current_player == Color.WHITE else Color.WHITE
        if self.removable_lines[other_player]:
            self.line_removal_queue.append(other_player)
        
        # 더 이상 제거할 라인이 없으면 라인 제거 프로세스 종료
        if not self.line_removal_queue:
            self._end_line_removal_process()

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
        """해당 색깔의 5개 이상 연속 라인을 찾음 (중복 제거 버전)"""
        lines = []
        markers = self.marker_positions[color]
        processed_positions = set()  # 이미 처리된 위치들

        # 3방향만 사용 (6방향의 방향 쌍에서 하나씩만 선택하여 중복 방지)
        directions = [(0, 1), (1, 0), (1, 1)]  # 세로, 가로, 대각선 (각 쌍의 정방향만)
        
        for start_pos in markers:
            if start_pos in processed_positions:
                continue
                
            for dx, dy in directions:
                # 양방향으로 확장하여 최대 길이 라인 찾기
                full_line = self._find_max_line_in_direction(start_pos, dx, dy, markers)
                
                # 5개 이상일 때만 유효한 라인
                if len(full_line) >= config.LINE_LENGTH_TO_WIN:
                    lines.append(full_line)
                    # 이 라인의 모든 위치를 처리됨으로 표시
                    processed_positions.update(full_line)

        return lines
    
    def _find_max_line_in_direction(self, start_pos: Tuple[int, int], dx: int, dy: int, markers: set) -> List[Tuple[int, int]]:
        """주어진 방향에서 최대 길이 라인 찾기 (양방향 확장)"""
        line = [start_pos]
        
        # 정방향으로 확장
        current_pos = start_pos
        while True:
            next_pos = (current_pos[0] + dx, current_pos[1] + dy)
            if next_pos in markers:
                line.append(next_pos)
                current_pos = next_pos
            else:
                break
        
        # 역방향으로 확장
        current_pos = start_pos
        while True:
            prev_pos = (current_pos[0] - dx, current_pos[1] - dy)
            if prev_pos in markers:
                line.insert(0, prev_pos)  # 앞쪽에 추가
                current_pos = prev_pos
            else:
                break
        
        return line

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
        state = np.zeros((15, self.board_size, self.board_size), dtype=np.float32)  # 13->15 채널로 확장

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

        # 채널 13-14: 제거 가능한 라인 정보 (새로 추가)
        if self.phase == GamePhase.LINE_REMOVAL:
            # 채널 13: 현재 플레이어의 제거 가능한 라인들
            for line in self.removable_lines[current_color]:
                for pos in line:
                    x, y = pos
                    state[13, x, y] = 1.0
            
            # 채널 14: 상대 플레이어의 제거 가능한 라인들
            for line in self.removable_lines[opponent_color]:
                for pos in line:
                    x, y = pos
                    state[14, x, y] = 1.0

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
                "markers_in_pool": self.markers_in_pool,
                "markers_on_board": self.markers_on_board,
                "line_removal_queue": [color.value for color in self.line_removal_queue],  # 새로운 시스템
                "removable_lines_white": [sorted(line) for line in self.removable_lines[Color.WHITE]],  # 새로 추가
                "removable_lines_black": [sorted(line) for line in self.removable_lines[Color.BLACK]],  # 새로 추가
                "move_player": self.move_player.value,  # 새로 추가
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
        
        # 새로운 라인 제거 시스템 필드들 (markers_in_pool, markers_on_board는 property로 자동 계산)
        new_env.line_removal_queue = self.line_removal_queue.copy()
        new_env.removable_lines = {
            Color.WHITE: [line.copy() for line in self.removable_lines[Color.WHITE]],
            Color.BLACK: [line.copy() for line in self.removable_lines[Color.BLACK]]
        }
        new_env.move_player = self.move_player
        
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
