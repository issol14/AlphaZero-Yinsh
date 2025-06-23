from dataclasses import dataclass
from enum import Enum
from typing import Tuple
import numpy as np
import config


class YinshActionType(Enum):
    """
    Types of actions in Yinsh
    """
    PLACE_RING = 0
    MOVE_RING = 1
    REMOVE_MARKERS = 2


class YinshDirection(Enum):
    """
    Six directions on hexagonal board
    """
    NORTH = 0
    NORTHEAST = 1
    SOUTHEAST = 2
    SOUTH = 3
    SOUTHWEST = 4
    NORTHWEST = 5


@dataclass
class YinshMove:
    """
    Represents a move in Yinsh
    """
    action_type: YinshActionType
    from_pos: Tuple[int, int] = None
    to_pos: Tuple[int, int] = None
    positions: list = None  # For marker removal


class Mapping:
    """
    The mapper handles conversion between Yinsh moves and neural network outputs
    """
    
    @staticmethod
    def move_to_action_index(move: YinshMove) -> int:
        """
        Convert a Yinsh move to action index for neural network
        """
        # TODO: Implement Yinsh move to action index conversion
        # This should map moves to the output vector indices
        
        if move.action_type == YinshActionType.PLACE_RING:
            # Ring placement: position on board
            if move.to_pos:
                row, col = move.to_pos
                return row * config.BOARD_SIZE + col
        
        elif move.action_type == YinshActionType.MOVE_RING:
            # Ring movement: from_pos + to_pos encoded
            if move.from_pos and move.to_pos:
                from_row, from_col = move.from_pos
                to_row, to_col = move.to_pos
                # Encode as offset from place_ring actions
                return (config.BOARD_SIZE * config.BOARD_SIZE + 
                       from_row * config.BOARD_SIZE + from_col)
        
        elif move.action_type == YinshActionType.REMOVE_MARKERS:
            # Marker removal: encoded differently
            # Offset from other action types
            return 2 * config.BOARD_SIZE * config.BOARD_SIZE
        
        return 0
    
    @staticmethod
    def action_index_to_move(action_index: int) -> YinshMove:
        """
        Convert action index back to Yinsh move
        """
        # TODO: Implement action index to Yinsh move conversion
        
        board_positions = config.BOARD_SIZE * config.BOARD_SIZE
        
        if action_index < board_positions:
            # Ring placement
            row = action_index // config.BOARD_SIZE
            col = action_index % config.BOARD_SIZE
            return YinshMove(YinshActionType.PLACE_RING, to_pos=(row, col))
        
        elif action_index < 2 * board_positions:
            # Ring movement
            adjusted_index = action_index - board_positions
            row = adjusted_index // config.BOARD_SIZE
            col = adjusted_index % config.BOARD_SIZE
            return YinshMove(YinshActionType.MOVE_RING, from_pos=(row, col))
        
        else:
            # Marker removal
            return YinshMove(YinshActionType.REMOVE_MARKERS)
    
    @staticmethod
    def get_valid_positions_mask(board_state: np.ndarray) -> np.ndarray:
        """
        Get mask of valid positions on the Yinsh board
        """
        # TODO: Implement Yinsh hex board valid positions
        # Yinsh has a specific hexagonal shape
        
        mask = np.zeros((config.BOARD_SIZE, config.BOARD_SIZE))
        
        # Placeholder: simple rectangular mask
        # In reality, Yinsh board has hexagonal shape
        for row in range(config.BOARD_SIZE):
            for col in range(config.BOARD_SIZE):
                # TODO: Implement proper hex board geometry
                mask[row, col] = 1
        
        return mask
    
    @staticmethod
    def hex_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """
        Calculate distance between two positions on hex grid
        """
        # TODO: Implement proper hexagonal distance calculation
        x1, y1 = pos1
        x2, y2 = pos2
        return abs(x1 - x2) + abs(y1 - y2)  # Placeholder Manhattan distance
    
    @staticmethod
    def get_line_positions(start: Tuple[int, int], direction: YinshDirection, length: int) -> list:
        """
        Get positions in a line from start position in given direction
        """
        # TODO: Implement hexagonal line calculation
        positions = []
        # This should calculate positions along hex grid lines
        return positions 