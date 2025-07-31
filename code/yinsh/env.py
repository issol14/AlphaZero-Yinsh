# env.py - YINSH Game Environment 

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from enum import Enum
import copy
import random
from . import config


class Color(Enum):
    WHITE = 1
    BLACK = -1
    EMPTY = 0


class PieceType(Enum):
    RING = "RING"
    MARKER = "MARKER"
    EMPTY = "EMPTY"


# 의문점: 색깔은 몰라도 되는거야? -> 몰라도 됨. 
class YinshAction:
    def __init__(
        self,
        from_pos: Optional[Tuple[int, int]] = None,
        to_pos: Optional[Tuple[int, int]] = None,
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
        return (self.from_pos == other.from_pos
            and self.to_pos == other.to_pos
        )


# 받아온 보드 상태를 기반으로 YinshEnv를 구성하는 함수가 있어야 할까? / 개발하다가 생각남. 판단 필요.
class YinshEnv:
    def __init__(self):
        self.board_size = config.BOARD_SIZE
        self.valid_points: Set[Tuple[int, int]] = config.VALID_POINTS
        
        # 게임 상태 초기화
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)  # 색 기준 0, 1, -1 만 저장된다는 가설. 다른 함수 사용성 검증과 함께 확인 필요. / 일단은 상관X / 보드가 꼭 필요한가에 대한 판단
        self.current_player = Color.WHITE
        
        # 링과 마커 위치
        self.ring_positions = {
            Color.WHITE: set(),
            Color.BLACK: set(),
        }
        self.marker_positions = {
            Color.WHITE: set(),
            Color.BLACK: set(),
        }
        
        # 게임 상태 변수들
        self.done = False  # 게임 종료 여부임. 
        self.winner = None
        
        # 게임 초기화
        self.reset()

    def reset(self):
        """게임 상태 초기화"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = Color.WHITE
        self.done = False
        self.winner = None
        
        # 링과 마커 위치 초기화
        self.ring_positions = {Color.WHITE: set(), Color.BLACK: set()}
        self.marker_positions = {Color.WHITE: set(), Color.BLACK: set()}
        
        # 링을 랜덤하게 배치
        self._place_rings_randomly()

    def _place_rings_randomly(self):
        """링 10개를 랜덤하게 배치"""
        candidate = random.sample(list(self.valid_points), 10)
        white_ring_positions = candidate[:5]
        black_ring_positions = candidate[5:]
        for pos in white_ring_positions:
            array_pos = self.hex_to_array_coords(pos)
            self.board[array_pos] = Color.WHITE.value
            self.ring_positions[Color.WHITE].add(pos)
        for pos in black_ring_positions:
            array_pos = self.hex_to_array_coords(pos)
            self.board[array_pos] = Color.BLACK.value
            self.ring_positions[Color.BLACK].add(pos)

    def hex_to_array_coords(self, hex_pos: Tuple[int, int]) -> Tuple[int, int]:
        """육각형 좌표를 배열 좌표로 변환"""
        # Hex(x,y) -> Array(-y+5, x+5)
        x, y = hex_pos
        return (-y+5, x+5)

    def array_to_hex_coords(self, array_pos: Tuple[int, int]) -> Tuple[int, int]:
        """배열 좌표를 육각형 좌표로 변환"""
        # Array(x, y) -> Hex(y-5, -x+5)
        x, y = array_pos
        return (y-5, -x+5)

    def is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        위치가 유효한지 확인 (HEX Coors 기준)
        """
        return pos in self.valid_points

    def is_empty_position(self, pos: Tuple[int, int]) -> bool:
        """위치가 비어있는지 확인"""
        # array_pos = self.hex_to_array_coords(pos)
        # return self.board[array_pos[0], array_pos[1]] == Color.EMPTY.value
        return not self._has_marker_at(pos) and not self._has_ring_at(pos)

    def has_ring(self, pos: Tuple[int, int], color: Color) -> bool:
        """위치에 해당 색의 링이 있는지 확인"""
        return pos in self.ring_positions[color]

    def has_marker(self, pos: Tuple[int, int], color: Color) -> bool:
        """위치에 해당 색의 마커가 있는지 확인"""
        return pos in self.marker_positions[color]

    def _has_marker_at(self, pos: Tuple[int, int]) -> bool:
        """위치에 마커가 있는지 확인 (색 무관)"""
        return pos in self.marker_positions[Color.WHITE] or pos in self.marker_positions[Color.BLACK]
    
    def _has_ring_at(self, pos: Tuple[int, int]) -> bool:
        """위치에 링이 있는지 확인 (색 무관)"""
        return pos in self.ring_positions[Color.WHITE] or pos in self.ring_positions[Color.BLACK]

    @property
    def markers_on_board(self) -> int:
        """현재 보드에 놓인 마커 수"""
        return len(self.marker_positions[Color.WHITE]) + len(self.marker_positions[Color.BLACK])

    @property
    def markers_in_pool(self) -> int:
        """사용 가능한 마커 수"""
        return config.TOTAL_MARKERS - self.markers_on_board

    def _place_marker(self, pos: Tuple[int, int], color: Color) -> bool:
        """
        마커 배치 함수
        - 마커를 배치하는 경우는 Move Ring 액션에서 링 자리에 마커를 놓는 경우임. 
        - 링 이동이 선행된 후, 진행되어야 함. 
        - 이미 앞에서 마커 풀이 소진되지 않았음을 검증했으므로 검증 패스
        """
        if not self.is_empty_position(pos):
            return False
        self.marker_positions[color].add(pos)

        # 보드 업데이트 수행 (보드 실용성 검증 후에 실제로 사용할지 말지 판단. )
        array_pos = self.hex_to_array_coords(pos)
        self.board[array_pos[0], array_pos[1]] = color.value

        return True


    def _is_valid_hex_direction(self, dx: int, dy: int) -> bool:
        """YINSH 육각형 보드에서 유효한 방향인지 확인"""
        return (dx, dy) in config.VALID_HEX_DIRECTIONS


    def generate_dynamic_move_ring_actions(self) -> List[YinshAction]:
        """동적으로 링 이동 액션 생성"""
        actions = []
        current_rings = self.ring_positions[self.current_player]
        
        for from_pos in current_rings:
            valid_moves = self.get_valid_ring_moves(from_pos)
            for to_pos in valid_moves:
                actions.append(YinshAction(from_pos=from_pos, to_pos=to_pos))
                
        return actions

    def get_valid_actions(self) -> List[YinshAction]: 
        """유효한 액션들 반환"""
        # 게임이 종료된 경우 빈 리스트 반환
        if self.done:
            return []
        return self.generate_dynamic_move_ring_actions()

    def is_valid_ring_move(
        self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]
    ) -> bool:
        """링 이동이 유효한지 확인"""
        # from_pos 검증 | 시작 위치에 현재 플레이어의 링이 있어야 함
        if not self.has_ring(from_pos, self.current_player):
            return False
        
        # to_pos 검증 | 목적지가 육각형 보드 위에 있어야 함. 
        if not self.is_valid_position(to_pos):
            return False
        
        # to_pos 검증 | 목적지가 비어있어야 함 / -> from_pos와 다르다는건 자동적으로 검증됨. 
        if not self.is_empty_position(to_pos):
            return False

        # 육각형 방향으로 이동해야 함
        dx, dy = self._get_direction(from_pos, to_pos)
        if not self._is_valid_hex_direction(dx, dy):
            return False
        
        # 특정 방향 탐색해서 to_pos가 있는지 확인
        if to_pos not in self._find_landing_position(from_pos, (dx, dy)):
            return False
        return True


    def _get_direction(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> Tuple[int, int]:
        """이동 방향 반환"""
        if from_pos[0] == to_pos[0]:
            dx = 0
        elif to_pos[0] > from_pos[0]: 
            dx = 1
        else:
            dx = -1
        
        if from_pos[1] == to_pos[1]:
            dy = 0
        elif to_pos[1] > from_pos[1]:
            dy = 1
        else:
            dy = -1
        return dx, dy
    

    def get_valid_ring_moves(self, from_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """특정 링의 유효한 모든 이동가능 위치들을 반환"""
        valid_moves = []
        
        # 6방향으로 이동 시도
        for direction in config.VALID_HEX_DIRECTIONS:
            landing_positions = self._find_landing_position(from_pos, direction)
            valid_moves.extend(landing_positions)
        
        return valid_moves


    def _find_landing_position(self, from_pos: Tuple[int, int], direction: Tuple[int, int]) -> List[Tuple[int, int]]:
        """특정 링의 특정 방향의 유효한 모든 이동가능 위치들을 반환"""
        landing_positions = []
        current_pos = from_pos
        
        # 빈 칸들을 지나가기
        while True:
            next_pos = (current_pos[0] + direction[0], current_pos[1] + direction[1])
            
            # 검증 : 보드 위에 있어야 함. 
            if not self.is_valid_position(next_pos):
                break

            # 검증 : next_pos에 링이 있을 경우 종료
            if self._has_ring_at(next_pos):
                break
            
            # 1. 인접한 연속된 빈칸 (왜냐면 마커를 만나지 않았음.)
            if self.is_empty_position(next_pos):
                landing_positions.append(next_pos)
                current_pos = next_pos
                continue

            if self._has_marker_at(next_pos):  # 마커를 만난 경우, 그 이후 첫 빈칸을 추가하거나 그냥 아예 추가할게 없거나 둘 중 하나임. 
                while True: 
                    next_pos = (next_pos[0] + direction[0], next_pos[1] + direction[1])

                    if not self.is_valid_position(next_pos):  # 그 마커가 벽에 붙어있는 경우 탐색 종료
                        break
                    elif self._has_ring_at(next_pos):  # 그 마커 뒤에 바로 링이 있는 경우 탐색 종료
                        break
                    elif self.is_empty_position(next_pos):  # 2. 연속된 마커 무더기 끝에 valid한 빈칸이 있는 경우, 빈칸을 추가하고 탐색 종료
                        landing_positions.append(next_pos)
                        break
                    current_pos = next_pos  # next_pos도 마커인 경우, 판단을 다음으로 유보. 
                # 위 Loop 종료 후 어떻게든 모든 경우의 수 탐색이 종료됨. 
                break
        return landing_positions

    


    def step(self, action: YinshAction) -> bool:
        """액션 실행"""
        # 게임이 이미 종료된 경우
        if self.done:
            return False
        
        # 게임 종료 조건 확인
        self._check_game_end()
        if self.done:
            return False
        
        if self.is_valid_ring_move(action.from_pos, action.to_pos):
            self._move_ring(action.from_pos, action.to_pos)
            return True
        return False

    def _move_ring(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]):
        """링을 이동하고 경로의 마커들을 뒤집기"""
        # 링 이동
        self.ring_positions[self.current_player].remove(from_pos)
        self.ring_positions[self.current_player].add(to_pos)
        
        # 보드 업데이트
        from_array_pos = self.hex_to_array_coords(from_pos)
        to_array_pos = self.hex_to_array_coords(to_pos)
        self.board[from_array_pos[0], from_array_pos[1]] = Color.EMPTY.value
        self.board[to_array_pos[0], to_array_pos[1]] = self.current_player.value
        
        # 시작 위치에 마커 배치
        self._place_marker(from_pos, self.current_player)
        
        # 경로의 마커들 뒤집기
        self._flip_markers_on_path(from_pos, to_pos)
        
        # 게임 종료 확인
        self._check_game_end()
        if self.done == True:
            return
        
        # 게임이 종료되지 않은 경우에만 플레이어 변경
        if not self.done:
            self.current_player = Color.BLACK if self.current_player == Color.WHITE else Color.WHITE

    def _get_path_positions(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """이동 경로의 위치들 반환"""
        dx, dy = self._get_direction(from_pos, to_pos)
        if not self._is_valid_hex_direction(dx, dy):
            return []
        path_positions = []
        next_pos = (from_pos[0] + dx, from_pos[1] + dy)
        while next_pos != to_pos:
            path_positions.append(next_pos)
            next_pos = (next_pos[0] + dx, next_pos[1] + dy)
        return path_positions
    

    def _flip_markers_on_path(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]):
        """이동 경로의 마커들 뒤집기"""
        path_positions = self._get_path_positions(from_pos, to_pos)
        
        for pos in path_positions:
            if pos in self.marker_positions[Color.WHITE]:
                self.marker_positions[Color.WHITE].remove(pos)
                self.marker_positions[Color.BLACK].add(pos)
                array_pos = self.hex_to_array_coords(pos)
                self.board[array_pos[0], array_pos[1]] = Color.BLACK.value
            elif pos in self.marker_positions[Color.BLACK]:
                self.marker_positions[Color.BLACK].remove(pos)
                self.marker_positions[Color.WHITE].add(pos)
                array_pos = self.hex_to_array_coords(pos)
                self.board[array_pos[0], array_pos[1]] = Color.WHITE.value




    def _check_for_five_in_line(self, color: Color) -> bool: 
        """
        전체에서 특정 색깔 마커에 대한 5개 연속 라인 존재 유무 판별
        1. X축방향
        2. Y축방향
        3. 대각선 방향
        """
        axis_line_lists = [config.X_POINTS_LIST, config.Y_POINTS_LIST, config.DIAGONAL_POINTS_LIST]
        for points_list in axis_line_lists:
            for line in points_list:
                if self._check_for_five_in_a_line(line, color):
                    return True
        return False
    
    def _check_for_five_in_a_line(self, line: List[Tuple[int, int]], color: Color) -> bool:
        """
        한 줄에서 특정 색깔 마커에 대한 5개 연속 라인 존재 유무 판별
        """
        line_str = ''.join(list(map(lambda x: "1" if self.has_marker(x, color) else "0", line)))
        if "11111" in line_str:
            return True
        else:
            return False





    def _check_game_end(self):
        """게임 종료 조건 확인"""
        # 5개 연속 라인 확인    
        is_black_marker_five_in_line = self._check_for_five_in_line(Color.BLACK)
        is_white_marker_five_in_line = self._check_for_five_in_line(Color.WHITE)

        # 게임 종료 여부 및 승자 판별
        if is_black_marker_five_in_line and is_white_marker_five_in_line:
            self.winner = None
            self.done = True
            return
        elif is_black_marker_five_in_line:
            self.winner = Color.BLACK
            self.done = True
            return
        elif is_white_marker_five_in_line:
            self.winner = Color.WHITE
            self.done = True
            return
        
        # 마커 소진 조건
        if self.markers_in_pool <= 0:
            self.winner = None  
            self.done = True
            return
        

    def is_game_over(self) -> bool:
        """게임이 종료되었는지 확인"""
        return self.done

    def get_winner(self) -> Optional[Color]:
        """승자를 반환"""
        return self.winner

    def get_state_tensor(self) -> np.ndarray:
        """현재 상태를 신경망 입력용 텐서로 변환 (간소화된 6채널)"""
        state = np.zeros((6, self.board_size, self.board_size), dtype=np.float32)

        # 채널 0-1: 현재 플레이어의 링과 마커
        current_color = self.current_player
        for pos in self.ring_positions[current_color]:
            x, y = self.hex_to_array_coords(pos)
            state[0, x, y] = 1.0
        for pos in self.marker_positions[current_color]:
            x, y = self.hex_to_array_coords(pos)
            state[1, x, y] = 1.0

        # 채널 2-3: 상대 플레이어의 링과 마커
        opponent_color = Color.BLACK if current_color == Color.WHITE else Color.WHITE
        for pos in self.ring_positions[opponent_color]:
            x, y = self.hex_to_array_coords(pos)
            state[2, x, y] = 1.0
        for pos in self.marker_positions[opponent_color]:
            x, y = self.hex_to_array_coords(pos)
            state[3, x, y] = 1.0

        # 채널 4: 유효한 보드 위치
        for pos in config.VALID_POINTS:
            x, y = self.hex_to_array_coords(pos)
            state[4, x, y] = 1.0

        # 채널 5: 현재 플레이어 (전체 보드에 색칠)
        state[5, :, :] = 1.0 if current_color == Color.WHITE else 0.0

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
                "markers_on_board": self.markers_on_board,
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
        new_env.done = self.done
        new_env.winner = self.winner
        return new_env
