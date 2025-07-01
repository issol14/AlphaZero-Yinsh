import config
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
import copy

logging.basicConfig(level=logging.INFO, format=' %(message)s')

# Yinsh 게임에서 사용되는 색상 정의
class Color:
    WHITE = 1
    BLACK = -1
    NONE = 0

# Yinsh 게임의 턴 상태 정의
class GameTurnState:
    WHITE_RING_PLACE = "WHITE_RING_PLACE"
    BLACK_RING_PLACE = "BLACK_RING_PLACE"
    WHITE_RING_MOVE = "WHITE_RING_MOVE"
    WHITE_MARKER_REMOVE = "WHITE_MARKER_REMOVE"
    WHITE_RING_REMOVE = "WHITE_RING_REMOVE"
    BLACK_RING_MOVE = "BLACK_RING_MOVE"
    BLACK_MARKER_REMOVE = "BLACK_MARKER_REMOVE"
    BLACK_RING_REMOVE = "BLACK_RING_REMOVE"
    WHITE_WIN = "WHITE_WIN"
    BLACK_WIN = "BLACK_WIN"
    DRAW = "DRAW"

class YinshAction:
    """Yinsh 게임의 액션을 표현하는 클래스"""
    def __init__(self, action_type: str, **kwargs):
        self.action_type = action_type
        self.params = kwargs
        
    def __repr__(self):
        return f"YinshAction({self.action_type}, {self.params})"
        
    def to_dict(self):
        return {"type": self.action_type, **self.params}

class YinshEnv:
    def __init__(self, state_string: str = None):
        """
        Yinsh 게임 환경 초기화 - 기존 코드베이스 호환
        """
        # 보드 크기 설정 (11x11)
        self.board_size = config.n
        self.center = 5  # 중앙 좌표 (0,0)을 (5,5)로 변환
        
        # 유효한 보드 좌표 설정 (육각형 모양)
        self.valid_points = {
            (-5, -4), (-5, -3), (-5, -2), (-5, -1),
            (-4, -5), (-4, -4), (-4, -3), (-4, -2), (-4, -1), (-4, 0), (-4, 1),
            (-3, -5), (-3, -4), (-3, -3), (-3, -2), (-3, -1), (-3, 0), (-3, 1), (-3, 2),
            (-2, -5), (-2, -4), (-2, -3), (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-2, 3),
            (-1, -5), (-1, -4), (-1, -3), (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2), (-1, 3), (-1, 4),
            (0, -4), (0, -3), (0, -2), (0, -1), (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
            (1, -4), (1, -3), (1, -2), (1, -1), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
            (2, -3), (2, -2), (2, -1), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5),
            (3, -2), (3, -1), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5),
            (4, -1), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5),
            (5, 1), (5, 2), (5, 3), (5, 4),
        }
        
        # 이동 방향 정의 (6방향 - 육각형 보드)
        self.directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1)]
        
        # 상태 문자열로부터 초기화
        if state_string:
            self.load_from_state_string(state_string)
        else:
            self.reset()

    def reset(self):
        """
        게임 상태 초기화
        """
        # 보드 상태 초기화
        self.white_rings = set()  # 백색 링 위치
        self.black_rings = set()  # 흑색 링 위치
        self.white_markers = set()  # 백색 마커 위치
        self.black_markers = set()  # 흑색 마커 위치
        
        # 게임 상태
        self.turn_state = GameTurnState.WHITE_RING_PLACE
        self.current_player = Color.WHITE
        self.winner = None
        self.done = False
        
        # 링 제거 카운터 (승리 조건: 3개 제거)
        self.removed_rings = {Color.WHITE: 0, Color.BLACK: 0}
        
        # 현재 턴 (체스와 호환을 위해)
        self.turn = True  # True = 백색 턴, False = 흑색 턴
        
        # Game 클래스 호환 속성들
        self.rings_placed = [0, 0]  # [White, Black] 배치된 링 개수
        self.rings_removed = [0, 0]  # [White, Black] 제거된 링 개수 (호환성)
        self.move_count = 0
        self.game_phase = 'placement'

    def get_state_string(self) -> str:
        """현재 상태를 문자열로 변환"""
        # 간단한 상태 문자열 형식
        white_rings_str = ';'.join([f"{x},{y}" for x, y in self.white_rings])
        black_rings_str = ';'.join([f"{x},{y}" for x, y in self.black_rings])
        white_markers_str = ';'.join([f"{x},{y}" for x, y in self.white_markers])
        black_markers_str = ';'.join([f"{x},{y}" for x, y in self.black_markers])
        
        state_parts = [
            white_rings_str or "EMPTY",
            black_rings_str or "EMPTY", 
            white_markers_str or "EMPTY",
            black_markers_str or "EMPTY",
            self.turn_state,
            str(self.current_player),
            str(self.removed_rings[Color.WHITE]),
            str(self.removed_rings[Color.BLACK]),
            str(self.turn)
        ]
        
        return "|".join(state_parts)

    def load_from_state_string(self, state_string: str):
        """문자열 상태로부터 게임 상태 복원"""
        try:
            parts = state_string.split("|")
            if len(parts) != 9:
                self.reset()
                return
                
            # 링과 마커 위치 복원
            self.white_rings = set()
            self.black_rings = set()
            self.white_markers = set()
            self.black_markers = set()
            
            if parts[0] != "EMPTY":
                for pos_str in parts[0].split(';'):
                    x, y = map(int, pos_str.split(','))
                    self.white_rings.add((x, y))
                    
            if parts[1] != "EMPTY":
                for pos_str in parts[1].split(';'):
                    x, y = map(int, pos_str.split(','))
                    self.black_rings.add((x, y))
                    
            if parts[2] != "EMPTY":
                for pos_str in parts[2].split(';'):
                    x, y = map(int, pos_str.split(','))
                    self.white_markers.add((x, y))
                    
            if parts[3] != "EMPTY":
                for pos_str in parts[3].split(';'):
                    x, y = map(int, pos_str.split(','))
                    self.black_markers.add((x, y))
            
            # 게임 상태 복원
            self.turn_state = parts[4]
            self.current_player = int(parts[5])
            self.removed_rings = {
                Color.WHITE: int(parts[6]),
                Color.BLACK: int(parts[7])
            }
            self.turn = parts[8] == "True"
            
            # 게임 종료 체크
            if self.removed_rings[Color.WHITE] >= 3:
                self.winner = Color.WHITE
                self.done = True
            elif self.removed_rings[Color.BLACK] >= 3:
                self.winner = Color.BLACK
                self.done = True
                
        except Exception as e:
            logging.warning(f"Failed to load state from string: {e}")
            self.reset()

    @staticmethod
    def state_to_input(state_string: str) -> np.ndarray:
        """
        상태 문자열을 신경망 입력으로 변환 (ChessEnv과 동일한 인터페이스)
        """
        # 임시 환경 생성하여 상태 복원
        temp_env = YinshEnv(state_string)
        return temp_env._state_to_input_array()

    def _state_to_input_array(self) -> np.ndarray:
        """
        현재 게임 상태를 신경망 입력으로 변환
        
        입력 형태 (6개 채널, config.py에 맞춤):
        0: 흰색 링 위치 
        1: 검은색 링 위치
        2: 흰색 마커 위치  
        3: 검은색 마커 위치
        4: 현재 턴 (1=백색 턴, 0=흑색 턴)
        5: 게임 페이즈 (0=링배치, 1=링이동, 2=마커제거, 3=링제거)
        """
        input_state = np.zeros((self.board_size, self.board_size, config.amount_of_input_planes), dtype=np.float32)
        
        # 링과 마커 위치 설정
        for pos in self.white_rings:
            if pos in self.valid_points:
                x, y = self.transform_coord(pos[0], pos[1])
                input_state[x, y, 0] = 1
                
        for pos in self.black_rings:
            if pos in self.valid_points:
                x, y = self.transform_coord(pos[0], pos[1])
                input_state[x, y, 1] = 1
                
        for pos in self.white_markers:
            if pos in self.valid_points:
                x, y = self.transform_coord(pos[0], pos[1])
                input_state[x, y, 2] = 1
                
        for pos in self.black_markers:
            if pos in self.valid_points:
                x, y = self.transform_coord(pos[0], pos[1])
                input_state[x, y, 3] = 1
        
        # 현재 턴 정보
        input_state[:, :, 4] = 1 if self.current_player == Color.WHITE else 0
        
        # 게임 단계 정보
        if "RING_PLACE" in self.turn_state:
            phase = 0
        elif "RING_MOVE" in self.turn_state:
            phase = 1
        elif "MARKER_REMOVE" in self.turn_state:
            phase = 2
        elif "RING_REMOVE" in self.turn_state:
            phase = 3
        else:
            phase = 4  # 게임 종료
            
        input_state[:, :, 5] = phase
        
        return input_state.reshape((1, *config.INPUT_SHAPE)).astype(bool)

    def transform_coord(self, x: int, y: int) -> Tuple[int, int]:
        """게임 좌표 (-5,-5) ~ (5,5)를 배열 인덱스 (0,0) ~ (10,10)로 변환"""
        return x + self.center, y + self.center

    def reverse_transform_coord(self, x: int, y: int) -> Tuple[int, int]:
        """배열 인덱스 (0,0) ~ (10,10)를 게임 좌표 (-5,-5) ~ (5,5)로 변환"""
        return x - self.center, y - self.center

    @property
    def board(self):
        """체스 보드와 호환성을 위한 속성"""
        return self
        
    def get_valid_moves(self) -> List[YinshAction]:
        """현재 상태에서 가능한 모든 합법적 행동 반환"""
        return self.legal_moves
    
    def is_game_over(self) -> bool:
        """게임 종료 여부 확인"""
        return self.done
    
    def make_move(self, action) -> bool:
        """행동 실행 및 성공 여부 반환"""
        try:
            self.step(action)
            return True
        except Exception as e:
            logging.warning(f"Invalid move: {e}")
            return False
    
    @property
    def legal_moves(self) -> List[YinshAction]:
        """현재 상태에서 가능한 모든 합법적 행동 반환"""
        if self.done:
            return []
            
        legal_moves = []
        
        # 링 배치 단계
        if "RING_PLACE" in self.turn_state:
            occupied_positions = self.white_rings | self.black_rings | self.white_markers | self.black_markers
            for pos in self.valid_points:
                if pos not in occupied_positions:
                    legal_moves.append(YinshAction("place_ring", position=pos))
        
        # 링 이동 단계
        elif "RING_MOVE" in self.turn_state:
            current_rings = self.white_rings if self.current_player == Color.WHITE else self.black_rings
            
            for ring_pos in current_rings:
                # 각 방향으로 이동 가능한 위치 계산
                for direction in self.directions:
                    for steps in range(1, 11):  # 최대 10칸
                        new_pos = (ring_pos[0] + direction[0] * steps, ring_pos[1] + direction[1] * steps)
                        
                        # 보드 범위 벗어나면 중단
                        if new_pos not in self.valid_points:
                            break
                            
                        # 다른 링이나 마커가 있으면 중단
                        if (new_pos in self.white_rings or new_pos in self.black_rings or 
                            new_pos in self.white_markers or new_pos in self.black_markers):
                            break
                            
                        legal_moves.append(YinshAction("move_ring", from_pos=ring_pos, to_pos=new_pos))
        
        # 마커 제거 단계
        elif "MARKER_REMOVE" in self.turn_state:
            color = Color.WHITE if "WHITE" in self.turn_state else Color.BLACK
            consecutive_markers = self.check_five_consecutive(color)
            
            for segment in consecutive_markers:
                legal_moves.append(YinshAction("remove_markers", positions=segment))
                
        # 링 제거 단계
        elif "RING_REMOVE" in self.turn_state:
            current_rings = self.white_rings if "WHITE" in self.turn_state else self.black_rings
            
            for ring_pos in current_rings:
                legal_moves.append(YinshAction("remove_ring", position=ring_pos))
                
        return legal_moves

    @staticmethod
    def estimate_winner(board, rings_removed=None) -> float:
        """
        게임 승자 추정 (ChessEnv과 동일한 인터페이스)
        """
        env = board  # board는 실제로 YinshEnv 인스턴스
        
        if env.done:
            if env.winner == Color.WHITE:
                return 0.9
            elif env.winner == Color.BLACK:
                return -0.9
            else:
                return 0
        
        # 제거된 링 개수로 우세 판단
        if rings_removed is not None:
            white_removed = rings_removed[0] if len(rings_removed) > 0 else 0
            black_removed = rings_removed[1] if len(rings_removed) > 1 else 0
        else:
            white_removed = env.removed_rings.get(Color.WHITE, 0)
            black_removed = env.removed_rings.get(Color.BLACK, 0)
        
        # 링 개수로 우세 판단
        white_rings = len(env.white_rings)
        black_rings = len(env.black_rings)
        
        # 점수 계산
        score = (white_removed - black_removed) * 0.3
        score += (black_rings - white_rings) * 0.1  # 링이 적을수록 유리
        
        return np.tanh(score)

    @staticmethod
    def get_piece_amount(board) -> int:
        """보드 위 기물 개수 (ChessEnv과 호환)"""
        env = board
        return len(env.white_rings) + len(env.black_rings) + len(env.white_markers) + len(env.black_markers)

    def step(self, action: YinshAction):
        """
        행동 실행 (ChessEnv과 동일한 인터페이스)
        """
        if self.done:
            return self
            
        try:
            action_dict = action.to_dict()
            self._execute_action(action_dict)
        except Exception as e:
            logging.warning(f"Invalid action: {e}")
            
        return self

    def _execute_action(self, action: Dict[str, Any]):
        """내부 행동 실행 로직"""
        action_type = action["type"]
        
        if action_type == "place_ring":
            self._place_ring(action["position"])
        elif action_type == "move_ring":
            self._move_ring(action["from_pos"], action["to_pos"])
        elif action_type == "remove_markers":
            self._remove_markers(action["positions"])
        elif action_type == "remove_ring":
            self._remove_ring(action["position"])
        else:
            raise ValueError(f"Invalid action type: {action_type}")

    def _place_ring(self, position: Tuple[int, int]):
        """링 배치 실행"""
        if position not in self.valid_points:
            raise ValueError("Invalid position")
            
        occupied = self.white_rings | self.black_rings | self.white_markers | self.black_markers
        if position in occupied:
            raise ValueError("Position already occupied")
        
        # 링 추가
        if self.current_player == Color.WHITE:
            self.white_rings.add(position)
            self.rings_placed[0] += 1
        else:
            self.black_rings.add(position)
            self.rings_placed[1] += 1
            
        # 턴 상태 업데이트
        white_ring_count = len(self.white_rings)
        black_ring_count = len(self.black_rings)
        
        # 이동 카운터 업데이트
        self.move_count += 1
        
        if white_ring_count == 5 and black_ring_count == 5:
            self.turn_state = GameTurnState.WHITE_RING_MOVE
            self.current_player = Color.WHITE
            self.turn = True
            self.game_phase = 'movement'
        else:
            if self.current_player == Color.WHITE:
                self.turn_state = GameTurnState.BLACK_RING_PLACE
                self.current_player = Color.BLACK
                self.turn = False
            else:
                self.turn_state = GameTurnState.WHITE_RING_PLACE
                self.current_player = Color.WHITE
                self.turn = True

    def _move_ring(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]):
        """링 이동 실행"""
        current_rings = self.white_rings if self.current_player == Color.WHITE else self.black_rings
        
        if from_pos not in current_rings:
            raise ValueError("No ring at from_position")
            
        if to_pos not in self.valid_points:
            raise ValueError("Invalid to_position")
            
        occupied = self.white_rings | self.black_rings | self.white_markers | self.black_markers
        if to_pos in occupied:
            raise ValueError("To_position already occupied")
        
        # 링 이동
        current_rings.remove(from_pos)
        current_rings.add(to_pos)
        
        # 원래 위치에 마커 추가
        if self.current_player == Color.WHITE:
            self.white_markers.add(from_pos)
        else:
            self.black_markers.add(from_pos)
            
        # 이동 경로상 마커 뒤집기
        self._flip_markers_on_path(from_pos, to_pos)
        
        # 이동 카운터 업데이트
        self.move_count += 1
        
        # 5개 연속 마커 확인 및 다음 상태 결정
        self._check_and_update_after_move()

    def _flip_markers_on_path(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]):
        """이동 경로상의 마커 뒤집기"""
        diff = (to_pos[0] - from_pos[0], to_pos[1] - from_pos[1])
        
        # 방향과 스텝 수 계산
        direction = None
        steps = None
        
        for d in self.directions:
            if d[0] == 0:  # 세로 방향
                if diff[0] == 0 and diff[1] != 0 and diff[1] % d[1] == 0:
                    k = diff[1] // d[1]
                    if k > 0:
                        direction = d
                        steps = k
                        break
            elif d[1] == 0:  # 가로 방향
                if diff[1] == 0 and diff[0] != 0 and diff[0] % d[0] == 0:
                    k = diff[0] // d[0]
                    if k > 0:
                        direction = d
                        steps = k
                        break
            else:  # 대각선 방향
                if diff[0] != 0 and diff[1] != 0:
                    if (diff[0] % d[0] == 0 and diff[1] % d[1] == 0 and 
                        diff[0] // d[0] == diff[1] // d[1] and diff[0] // d[0] > 0):
                        direction = d
                        steps = diff[0] // d[0]
                        break
        
        if direction and steps:
            for i in range(1, steps):
                intermediate_pos = (from_pos[0] + direction[0] * i, from_pos[1] + direction[1] * i)
                
                # 마커 뒤집기
                if intermediate_pos in self.white_markers:
                    self.white_markers.remove(intermediate_pos)
                    self.black_markers.add(intermediate_pos)
                elif intermediate_pos in self.black_markers:
                    self.black_markers.remove(intermediate_pos)
                    self.white_markers.add(intermediate_pos)

    def _check_and_update_after_move(self):
        """이동 후 5개 연속 마커 확인 및 상태 업데이트"""
        white_consecutive = self.check_five_consecutive(Color.WHITE)
        black_consecutive = self.check_five_consecutive(Color.BLACK)
        
        # 무승부 조건 체크 (마커 51개 이상)
        total_markers = len(self.white_markers) + len(self.black_markers)
        if total_markers >= 51:
            white_rings = len(self.white_rings)
            black_rings = len(self.black_rings)
            
            if white_rings < black_rings:
                self.winner = Color.WHITE
            elif black_rings < white_rings:
                self.winner = Color.BLACK
            else:
                self.winner = None  # 무승부
                
            self.done = True
            return
        
        # 5개 연속 마커 처리
        if self.current_player == Color.WHITE and white_consecutive:
            self.turn_state = GameTurnState.WHITE_MARKER_REMOVE
        elif self.current_player == Color.BLACK and black_consecutive:
            self.turn_state = GameTurnState.BLACK_MARKER_REMOVE
        elif self.current_player == Color.WHITE and black_consecutive:
            self.turn_state = GameTurnState.BLACK_MARKER_REMOVE
            self.current_player = Color.BLACK
            self.turn = False
        elif self.current_player == Color.BLACK and white_consecutive:
            self.turn_state = GameTurnState.WHITE_MARKER_REMOVE
            self.current_player = Color.WHITE
            self.turn = True
        else:
            # 다음 플레이어 턴
            if self.current_player == Color.WHITE:
                self.turn_state = GameTurnState.BLACK_RING_MOVE
                self.current_player = Color.BLACK
                self.turn = False
            else:
                self.turn_state = GameTurnState.WHITE_RING_MOVE
                self.current_player = Color.WHITE
                self.turn = True

    def _remove_markers(self, positions: List[Tuple[int, int]]):
        """마커 제거 실행"""
        if len(positions) != 5:
            raise ValueError("Must remove exactly 5 markers")
            
        # 연속성 및 색상 검증
        color = Color.WHITE if "WHITE" in self.turn_state else Color.BLACK
        marker_set = self.white_markers if color == Color.WHITE else self.black_markers
        
        for pos in positions:
            if pos not in marker_set:
                raise ValueError(f"No marker at position {pos}")
                
        # 마커 제거
        for pos in positions:
            marker_set.remove(pos)
            
        # 링 제거 단계로 전환
        if color == Color.WHITE:
            self.turn_state = GameTurnState.WHITE_RING_REMOVE
        else:
            self.turn_state = GameTurnState.BLACK_RING_REMOVE

    def _remove_ring(self, position: Tuple[int, int]):
        """링 제거 실행"""
        color = Color.WHITE if "WHITE" in self.turn_state else Color.BLACK
        ring_set = self.white_rings if color == Color.WHITE else self.black_rings
        
        if position not in ring_set:
            raise ValueError("No ring at position")
            
        # 링 제거
        ring_set.remove(position)
        self.removed_rings[color] += 1
        
        # 호환성을 위한 리스트 형태 업데이트
        if color == Color.WHITE:
            self.rings_removed[0] += 1
        else:
            self.rings_removed[1] += 1
        
        # 승리 조건 체크 (3개 제거하면 승리)
        if self.removed_rings[color] >= 3:
            self.winner = color
            self.done = True
            return
            
        # 계속 플레이 - 5개 연속 마커 재확인
        white_consecutive = self.check_five_consecutive(Color.WHITE)
        black_consecutive = self.check_five_consecutive(Color.BLACK)
        
        if color == Color.WHITE and white_consecutive:
            self.turn_state = GameTurnState.WHITE_MARKER_REMOVE
        elif color == Color.BLACK and black_consecutive:
            self.turn_state = GameTurnState.BLACK_MARKER_REMOVE
        elif color == Color.WHITE and black_consecutive:
            self.turn_state = GameTurnState.BLACK_MARKER_REMOVE
            self.current_player = Color.BLACK
            self.turn = False
        elif color == Color.BLACK and white_consecutive:
            self.turn_state = GameTurnState.WHITE_MARKER_REMOVE
            self.current_player = Color.WHITE
            self.turn = True
        else:
            # 다음 플레이어 턴
            if color == Color.WHITE:
                self.turn_state = GameTurnState.BLACK_RING_MOVE
                self.current_player = Color.BLACK
                self.turn = False
            else:
                self.turn_state = GameTurnState.WHITE_RING_MOVE
                self.current_player = Color.WHITE
                self.turn = True

    def check_five_consecutive(self, color: int) -> List[List[Tuple[int, int]]]:
        """5개 연속 마커 확인"""
        markers = self.white_markers if color == Color.WHITE else self.black_markers
        marker_positions = set(markers)
        
        # 중복 방지를 위해 4방향만 검사
        directions = [(1, 0), (0, 1), (1, -1), (1, 1)]
        consecutive_segments = []
        
        for pos in marker_positions:
            for d in directions:
                # 시작점 확인 (이전 위치에 마커가 없어야 함)
                prev_pos = (pos[0] - d[0], pos[1] - d[1])
                if prev_pos in marker_positions:
                    continue
                    
                # 연속 마커 체인 찾기
                chain = []
                current = pos
                while current in marker_positions:
                    chain.append(current)
                    current = (current[0] + d[0], current[1] + d[1])
                    
                # 5개 이상 연속된 마커가 있으면 모든 5개 조합 추가
                if len(chain) >= 5:
                    for i in range(len(chain) - 4):
                        segment = chain[i:i+5]
                        consecutive_segments.append(segment)
                        
        return consecutive_segments

    def __str__(self) -> str:
        """보드 상태 문자열 표현"""
        board_str = "   "
        for x in range(-5, 6):
            board_str += f"{x:2d} "
        board_str += "\n"
        
        for y in range(-5, 6):
            board_str += f"{y:2d} "
            for x in range(-5, 6):
                pos = (x, y)
                if pos in self.white_rings:
                    board_str += "WR "
                elif pos in self.black_rings:
                    board_str += "BR "
                elif pos in self.white_markers:
                    board_str += "wm "
                elif pos in self.black_markers:
                    board_str += "bm "
                elif pos in self.valid_points:
                    board_str += ".  "
                else:
                    board_str += "   "
            board_str += "\n"
            
        board_str += f"\nTurn: {self.turn_state}\n"
        board_str += f"Current Player: {'WHITE' if self.current_player == Color.WHITE else 'BLACK'}\n"
        board_str += f"Removed Rings: W={self.removed_rings[Color.WHITE]}, B={self.removed_rings[Color.BLACK]}\n"
        
        return board_str 

    def get_winner(self) -> int:
        """게임 승자 반환 (Game 클래스 호환)"""
        if self.winner == Color.WHITE:
            return 1
        elif self.winner == Color.BLACK:
            return -1
        else:
            return 0 