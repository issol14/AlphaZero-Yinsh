import config
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO, format=' %(message)s')


class YinshEnv:
    def __init__(self, board_state: str = None):
        """
        Initialize the Yinsh environment
        """
        self.board_size = config.BOARD_SIZE
        self.rings_per_player = config.RINGS_PER_PLAYER
        self.markers_to_win = config.MARKERS_TO_WIN
        
        self.reset()

    def reset(self):
        """
        Reset everything to initial game state
        """
        # Initialize empty board
        self.board = np.zeros((self.board_size, self.board_size, 4))  # rings_p1, rings_p2, markers_p1, markers_p2
        self.current_player = 0  # 0 for player 1, 1 for player 2
        self.game_phase = "placement"  # "placement" or "movement"
        self.rings_placed = [0, 0]  # rings placed by each player
        self.rings_removed = [0, 0]  # rings removed by each player
        self.move_count = 0

    @staticmethod
    def state_to_input(board_state: np.ndarray, current_player: int, game_phase: str) -> np.ndarray:
        """
        Convert board to a state that is interpretable by the model
        """
        # TODO: Implement Yinsh board state to neural network input conversion
        # This should create input planes for:
        # - Player 1 rings
        # - Player 2 rings  
        # - Player 1 markers
        # - Player 2 markers
        # - Valid positions
        # - Current player indicator
        
        input_state = np.zeros(config.INPUT_SHAPE)
        
        # Placeholder implementation
        # input_state[:, :, 0] = board_state[:, :, 0]  # P1 rings
        # input_state[:, :, 1] = board_state[:, :, 1]  # P2 rings
        # input_state[:, :, 2] = board_state[:, :, 2]  # P1 markers
        # input_state[:, :, 3] = board_state[:, :, 3]  # P2 markers
        # input_state[:, :, 4] = get_valid_positions()  # Valid positions
        # input_state[:, :, 5] = np.ones((config.n, config.n)) * current_player  # Current player
        
        return input_state.reshape((1, *config.INPUT_SHAPE)).astype(bool)

    @staticmethod
    def estimate_winner(board_state: np.ndarray, rings_removed: list) -> float:
        """
        Estimate the winner of the current state
        Returns: 1 if player 1 wins, -1 if player 2 wins, 0 for draw/unknown
        """
        # TODO: Implement Yinsh winner estimation
        # Check if either player has removed 3 rings
        if rings_removed[0] >= config.MARKERS_TO_WIN:
            return 1.0
        elif rings_removed[1] >= config.MARKERS_TO_WIN:
            return -1.0
        
        # TODO: Add heuristic evaluation based on board position
        # For now, return 0 (draw/unknown)
        return 0.0

    def is_valid_position(self, row: int, col: int) -> bool:
        """
        Check if position is valid on Yinsh hexagonal board
        """
        # TODO: Implement Yinsh hex board validity check
        # Yinsh has a specific hexagonal shape
        return 0 <= row < self.board_size and 0 <= col < self.board_size

    def get_valid_moves(self) -> list:
        """
        Get all valid moves for the current game state
        """
        # TODO: Implement Yinsh move generation
        valid_moves = []
        
        if self.game_phase == "placement":
            # Ring placement phase
            # valid_moves = self._get_ring_placement_moves()
            pass
        else:
            # Ring movement phase
            # valid_moves = self._get_ring_movement_moves()
            pass
        
        return valid_moves

    def make_move(self, move) -> bool:
        """
        Apply a move to the game state
        Returns True if move was successful, False otherwise
        """
        # TODO: Implement move execution
        # This should:
        # 1. Validate the move
        # 2. Update board state
        # 3. Check for line formations
        # 4. Handle ring removal if lines are formed
        # 5. Switch players
        
        self.move_count += 1
        return True

    def is_game_over(self) -> bool:
        """
        Check if the game is over
        """
        # Game ends when a player removes 3 rings
        return (self.rings_removed[0] >= self.markers_to_win or 
                self.rings_removed[1] >= self.markers_to_win or
                self.move_count >= config.MAX_GAME_MOVES)

    def get_winner(self) -> int:
        """
        Get the winner of the game
        Returns: 1 if player 1 wins, -1 if player 2 wins, 0 for draw
        """
        if self.rings_removed[0] >= self.markers_to_win:
            return 1
        elif self.rings_removed[1] >= self.markers_to_win:
            return -1
        else:
            return 0

    def __str__(self):
        """
        String representation of the board
        """
        # TODO: Implement board display
        return f"Yinsh Board - Player: {self.current_player}, Phase: {self.game_phase}" 