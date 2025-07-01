import config
from re import A
import sys
import os

# Add chess directory to path and ensure it's prioritized
chess_path = os.path.join(os.path.dirname(__file__), '..', 'chess')
if chess_path not in sys.path:
    sys.path.insert(0, chess_path)

# Import from chess directory explicitly
import yinsh as chess_yinsh
from yinsh import Move
import numpy as np

import time
import logging

logging.basicConfig(level=logging.INFO, format=' %(message)s')


class YinshEnv:
    def __init__(self, fen: str = chess_yinsh.Board.starting_fen):
        """
        Initialize the YINSH environment

        YINSH: 
        환경 초기화. 
        FEN을 받아서 초기화
        """
        # the yinsh board
        self.fen = fen
        self.reset()

    def reset(self):
        """
        Reset everything
        """
        self.board = chess_yinsh.Board(self.fen)

    @staticmethod
    def state_to_input(fen: str) -> np.ndarray:
        """
        Convert board to a state that is interpretable by the model

        YINSH: 
        fen으로 받아서 상태를 만들어줌. 
        
        YINSH INPUT SHAPE 만들기:
        - 85개 위치를 11x11 격자로 표현 (유효하지 않은 위치는 0으로 마스킹)
        - 각 플레이어의 링과 마커를 별도 채널로 표현
        - 게임 단계, 턴, 제거된 링 개수 등을 전역 정보로 포함
        """
        board = chess_yinsh.Board(fen)

        # YINSH는 11x11 격자로 표현 (85개 유효 위치)
        board_size = 11
        
        # 1. 턴 정보 (1x11x11)
        is_white_turn = np.ones((board_size, board_size)) if board.turn == chess_yinsh.WHITE else np.zeros((board_size, board_size))

        # 2. 게임 단계 정보 (5x11x11) - 각 단계별로 채널 분리
        phase_placement = np.ones((board_size, board_size)) if board.phase == chess_yinsh.GamePhase.PLACEMENT else np.zeros((board_size, board_size))
        phase_main = np.ones((board_size, board_size)) if board.phase == chess_yinsh.GamePhase.MAIN else np.zeros((board_size, board_size))
        phase_marker_remove = np.ones((board_size, board_size)) if board.phase == chess_yinsh.GamePhase.MARKER_REMOVE else np.zeros((board_size, board_size))
        phase_ring_remove = np.ones((board_size, board_size)) if board.phase == chess_yinsh.GamePhase.RING_REMOVE else np.zeros((board_size, board_size))
        phase_ended = np.ones((board_size, board_size)) if board.phase == chess_yinsh.GamePhase.ENDED else np.zeros((board_size, board_size))

        # 3. 제거된 링 개수 정보 (2x11x11)
        white_rings_removed = np.full((board_size, board_size), board.rings_removed[chess_yinsh.WHITE] / 3.0)  # 정규화 (최대 3개)
        black_rings_removed = np.full((board_size, board_size), board.rings_removed[chess_yinsh.BLACK] / 3.0)

        # 4. 보드 상태 배열 생성 - 각 플레이어의 링과 마커 (4x11x11)
        white_rings = np.zeros((board_size, board_size))
        black_rings = np.zeros((board_size, board_size))
        white_markers = np.zeros((board_size, board_size))
        black_markers = np.zeros((board_size, board_size))

        # 좌표 변환을 위한 함수 (YINSH 좌표를 11x11 격자 좌표로 변환)
        def yinsh_to_grid(x, y):
            grid_x = x + 5  # -5~5 -> 0~10
            grid_y = y + 5  # -5~5 -> 0~10
            return grid_x, grid_y

        # 보드의 각 위치를 확인하여 배열에 설정
        for pos in range(len(chess_yinsh.POSITION_LIST)):
            x, y = chess_yinsh.position_to_coord(pos)
            grid_x, grid_y = yinsh_to_grid(x, y)
            
            if 0 <= grid_x < board_size and 0 <= grid_y < board_size:
                # 링 정보
                if pos in board.rings:
                    if board.rings[pos] == chess_yinsh.WHITE:
                        white_rings[grid_y, grid_x] = 1.0
                    else:
                        black_rings[grid_y, grid_x] = 1.0
                
                # 마커 정보
                if pos in board.markers:
                    if board.markers[pos] == chess_yinsh.WHITE:
                        white_markers[grid_y, grid_x] = 1.0
                    else:
                        black_markers[grid_y, grid_x] = 1.0

        # 5. 유효 위치 마스크 (1x11x11)
        valid_positions = np.zeros((board_size, board_size))
        for pos in range(len(chess_yinsh.POSITION_LIST)):
            x, y = chess_yinsh.position_to_coord(pos)
            grid_x, grid_y = yinsh_to_grid(x, y)
            if 0 <= grid_x < board_size and 0 <= grid_y < board_size:
                valid_positions[grid_y, grid_x] = 1.0

        # 모든 채널을 결합
        channels = [
            is_white_turn,           # 1 channel
            phase_placement,         # 5 channels
            phase_main,
            phase_marker_remove,
            phase_ring_remove,
            phase_ended,
            white_rings_removed,     # 2 channels
            black_rings_removed,
            white_rings,            # 4 channels
            black_rings,
            white_markers,
            black_markers,
            valid_positions         # 1 channel
        ]
        
        # (13, 11, 11) 형태로 결합
        result = np.array(channels)
        
        # (1, 11, 11, 13) 형태로 변환 (배치 차원 추가 및 채널을 마지막으로)
        result = result.transpose(1, 2, 0)  # (11, 11, 13)
        result = np.expand_dims(result, axis=0)  # (1, 11, 11, 13)
        
        # memory management
        del board
        return result.astype(np.float32)

    @staticmethod
    def estimate_winner(board) -> float:
        """
        Estimate the winner of the current node.
        
        YINSH: 
        제거한 링의 개수로 승부 판단. 
        3개 링을 먼저 제거하는 플레이어가 승리.
        """
        white_rings_removed = board.rings_removed[chess_yinsh.WHITE]
        black_rings_removed = board.rings_removed[chess_yinsh.BLACK]
        
        # 게임이 끝난 경우
        if white_rings_removed >= 3:
            logging.debug("White wins")
            return 1.0
        elif black_rings_removed >= 3:
            logging.debug("Black wins")
            return -1.0
        
        # 게임이 진행 중인 경우 - 제거된 링 개수 차이로 추정
        score_diff = white_rings_removed - black_rings_removed
        
        if score_diff > 0:
            logging.debug(f"White advantage (rings removed: W:{white_rings_removed}, B:{black_rings_removed})")
            return 0.3 * score_diff
        elif score_diff < 0:
            logging.debug(f"Black advantage (rings removed: W:{white_rings_removed}, B:{black_rings_removed})")
            return 0.3 * score_diff
        else:
            logging.debug("Equal position")
            return 0.0

    @staticmethod
    def get_piece_amount(board) -> int:
        """
        Get total number of pieces on the board (rings + markers)
        """
        return len(board.rings) + len(board.markers)

    def get_legal_moves(self) -> list:
        """
        Get list of legal moves in current position
        """
        return list(self.board.generate_legal_moves())

    def get_legal_actions(self) -> list:
        """
        Get list of legal action indices for neural network
        """
        legal_moves = self.get_legal_moves()
        legal_actions = []
        
        for move in legal_moves:
            action = chess_yinsh.encode_move(move, self.board)
            if action is not None:
                legal_actions.append(action)
        
        return legal_actions

    def action_to_move(self, action: int) -> Move:
        """
        Convert action index to Move object
        """
        return chess_yinsh.decode_move(action, self.board)

    def move_to_action(self, move: Move) -> int:
        """
        Convert Move object to action index
        """
        return chess_yinsh.encode_move(move, self.board)

    def get_action_probabilities(self, probabilities: np.ndarray) -> dict:
        """
        Map neural network output probabilities to legal moves
        
        Args:
            probabilities: Neural network output vector
            
        Returns:
            Dict mapping move UCI strings to probabilities
        """
        if len(probabilities.shape) > 1:
            probabilities = probabilities.flatten()
        
        legal_moves = self.get_legal_moves()
        action_probs = {}
        
        for move in legal_moves:
            action = self.move_to_action(move)
            if action is not None and 0 <= action < len(probabilities):
                action_probs[move.uci()] = probabilities[action]
        
        return action_probs

    def is_game_over(self) -> bool:
        """
        Check if game is over
        """
        return self.board.is_game_over()

    def get_result(self) -> str:
        """
        Get game result
        """
        return self.board.result()

    def get_winner(self) -> int:
        """
        Get winner as integer (-1: black wins, 0: draw, 1: white wins)
        """
        result = self.get_result()
        if result == "1-0":
            return 1
        elif result == "0-1":
            return -1
        else:
            return 0

    def copy(self):
        """
        Create a copy of the environment
        """
        new_env = YinshEnv()
        new_env.board = self.board.copy()
        return new_env

    def __str__(self):
        """
        Print the board
        """
        return str(self.board)

    def step(self, action: Move):
        """
        Perform a step in the game
        """
        self.board.push(action)
        return self.board 