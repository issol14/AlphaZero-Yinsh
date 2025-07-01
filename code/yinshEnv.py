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
        - 85개 위치를 직접 1차원 벡터로 표현 (11x11 격자는 부정확함)
        - 각 플레이어의 링과 마커를 별도 채널로 표현
        - 게임 단계, 턴, 제거된 링 개수 등을 전역 정보로 포함
        """
        board = chess_yinsh.Board(fen)

        # YINSH는 85개 유효 위치를 직접 사용
        board_size = 85
        
        # 1. 보드 상태 배열 생성 - 각 플레이어의 링과 마커 (4 x 85)
        white_rings = np.zeros(board_size, dtype=np.float32)
        black_rings = np.zeros(board_size, dtype=np.float32)
        white_markers = np.zeros(board_size, dtype=np.float32)
        black_markers = np.zeros(board_size, dtype=np.float32)

        # 보드의 각 위치를 확인하여 배열에 설정
        for pos in range(board_size):
            # 링 정보
            if pos in board.rings:
                if board.rings[pos] == chess_yinsh.WHITE:
                    white_rings[pos] = 1.0
                else:
                    black_rings[pos] = 1.0
            
            # 마커 정보
            if pos in board.markers:
                if board.markers[pos] == chess_yinsh.WHITE:
                    white_markers[pos] = 1.0
                else:
                    black_markers[pos] = 1.0

        # 2. 전역 정보 (스칼라 값들을 85차원 벡터로 확장)
        
        # 턴 정보 (1 x 85)
        is_white_turn = np.full(board_size, 1.0 if board.turn == chess_yinsh.WHITE else 0.0, dtype=np.float32)

        # 게임 단계 정보 (5 x 85) - 각 단계별로 채널 분리
        phase_placement = np.full(board_size, 1.0 if board.phase == chess_yinsh.GamePhase.PLACEMENT else 0.0, dtype=np.float32)
        phase_main = np.full(board_size, 1.0 if board.phase == chess_yinsh.GamePhase.MAIN else 0.0, dtype=np.float32)
        phase_marker_remove = np.full(board_size, 1.0 if board.phase == chess_yinsh.GamePhase.MARKER_REMOVE else 0.0, dtype=np.float32)
        phase_ring_remove = np.full(board_size, 1.0 if board.phase == chess_yinsh.GamePhase.RING_REMOVE else 0.0, dtype=np.float32)
        phase_ended = np.full(board_size, 1.0 if board.phase == chess_yinsh.GamePhase.ENDED else 0.0, dtype=np.float32)

        # 제거된 링 개수 정보 (2 x 85)
        white_rings_removed = np.full(board_size, board.rings_removed[chess_yinsh.WHITE] / 3.0, dtype=np.float32)  # 정규화 (최대 3개)
        black_rings_removed = np.full(board_size, board.rings_removed[chess_yinsh.BLACK] / 3.0, dtype=np.float32)

        # 배치된 링 개수 정보 (2 x 85) - placement 단계에서 중요
        white_rings_placed = np.full(board_size, board.rings_placed[chess_yinsh.WHITE] / 5.0, dtype=np.float32)  # 정규화 (최대 5개)
        black_rings_placed = np.full(board_size, board.rings_placed[chess_yinsh.BLACK] / 5.0, dtype=np.float32)

        # 남은 마커 개수 정보 (1 x 85)
        markers_available = np.full(board_size, board.markers_available / 51.0, dtype=np.float32)  # 정규화 (최대 51개)

        # 모든 채널을 결합 (14 x 85)
        channels = [
            white_rings,            # 1 channel
            black_rings,            # 1 channel  
            white_markers,          # 1 channel
            black_markers,          # 1 channel
            is_white_turn,          # 1 channel
            phase_placement,        # 5 channels
            phase_main,
            phase_marker_remove,
            phase_ring_remove,
            phase_ended,
            white_rings_removed,    # 2 channels
            black_rings_removed,
            white_rings_placed,     # 2 channels
            black_rings_placed,
            markers_available       # 1 channel
        ]
        
        # (14, 85) 형태로 결합
        result = np.array(channels, dtype=np.float32)
        
        # (1, 85, 14) 형태로 변환 (배치 차원 추가 및 채널을 마지막으로)
        result = result.transpose(1, 0)  # (85, 14)
        result = np.expand_dims(result, axis=0)  # (1, 85, 14)
        
        # memory management
        del board
        return result

    @staticmethod
    def estimate_winner(board) -> float:
        """
        Estimate the winner of the current node.
        
        YINSH: 
        제거한 링의 개수, 보드 위 링 개수, 5개 연속 마커 가능성 등을 종합적으로 고려하여 승부 판단.
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
        
        # 게임이 진행 중인 경우 - 여러 요소를 종합적으로 고려
        score = 0.0
        
        # 1. 제거된 링 개수 차이 (가장 중요한 요소)
        rings_diff = white_rings_removed - black_rings_removed
        score += rings_diff * 0.4
        
        # 2. 보드 위의 링 개수 차이 (더 많은 링 = 더 많은 이동 옵션)
        white_rings_on_board = sum(1 for color in board.rings.values() if color == chess_yinsh.WHITE)
        black_rings_on_board = sum(1 for color in board.rings.values() if color == chess_yinsh.BLACK)
        rings_on_board_diff = white_rings_on_board - black_rings_on_board
        score += rings_on_board_diff * 0.1
        
        # 3. 5개 연속 마커 만들기 가능성
        white_potential_lines = len(board.check_five_in_row(chess_yinsh.WHITE))
        black_potential_lines = len(board.check_five_in_row(chess_yinsh.BLACK))
        potential_diff = white_potential_lines - black_potential_lines
        score += potential_diff * 0.2
        
        # 4. 보드 위 마커 개수 차이 (더 많은 마커 = 더 많은 제어)
        white_markers_on_board = sum(1 for color in board.markers.values() if color == chess_yinsh.WHITE)
        black_markers_on_board = sum(1 for color in board.markers.values() if color == chess_yinsh.BLACK)
        markers_diff = white_markers_on_board - black_markers_on_board
        score += markers_diff * 0.05
        
        # 5. 게임 단계에 따른 가중치 조정
        if board.phase == chess_yinsh.GamePhase.PLACEMENT:
            # 배치 단계에서는 위치 선점이 중요
            score *= 0.5
        elif board.phase in [chess_yinsh.GamePhase.MARKER_REMOVE, chess_yinsh.GamePhase.RING_REMOVE]:
            # 마커/링 제거 단계에서는 현재 턴 플레이어가 유리
            if board.turn == chess_yinsh.WHITE:
                score += 0.3
            else:
                score -= 0.3
        
        # 점수를 -1에서 1 사이로 정규화
        score = max(-1.0, min(1.0, score))
        
        logging.debug(f"Position evaluation: score={score:.3f}, rings_removed(W:{white_rings_removed}, B:{black_rings_removed}), "
                     f"rings_on_board(W:{white_rings_on_board}, B:{black_rings_on_board}), "
                     f"potential_lines(W:{white_potential_lines}, B:{black_potential_lines})")
        
        return score

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
        move = chess_yinsh.decode_move(action, self.board)
        if move is None:
            raise ValueError(f"Invalid action {action} for current board state")
        return move

    def move_to_action(self, move: Move) -> int:
        """
        Convert Move object to action index
        """
        action = chess_yinsh.encode_move(move, self.board)
        if action is None:
            raise ValueError(f"Invalid move {move.uci()} for current board state")
        return action

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