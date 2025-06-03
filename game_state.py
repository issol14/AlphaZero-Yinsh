"""
YINSH Game State Representation

This module defines the core data structures for representing YINSH game states,
including board positions, rings, markers, and actions.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Literal
from enum import Enum
from dataclasses import dataclass
import copy

# Type definitions
Coordinate = Tuple[int, int]
PlayerColor = Literal["white", "black"]


class GamePhase(Enum):
    """Game phases in YINSH"""

    WHITE_RING_PLACE = 0
    BLACK_RING_PLACE = 1
    WHITE_RING_MOVE = 2
    BLACK_RING_MOVE = 3
    WHITE_MARKER_REMOVE = 4
    BLACK_MARKER_REMOVE = 5
    WHITE_RING_REMOVE = 6
    BLACK_RING_REMOVE = 7


@dataclass
class Ring:
    """Represents a ring on the board"""

    position: Coordinate
    color: PlayerColor


@dataclass
class Marker:
    """Represents a marker on the board"""

    position: Coordinate
    color: PlayerColor


@dataclass
class Action:
    """Represents a game action"""

    action_type: str  # 'PLACE_RING', 'MOVE_RING', 'REMOVE_MARKERS', 'REMOVE_RING'
    to_pos: Optional[Coordinate] = None
    from_pos: Optional[Coordinate] = None
    markers_to_remove: Optional[List[Coordinate]] = None
    ring_to_remove: Optional[Coordinate] = None


class GameState:
    """
    Represents the complete state of a YINSH game

    This includes the board configuration, current phase, player turn,
    and additional game information needed for AI decision making.
    """

    def __init__(self):
        self.rings: List[Ring] = []
        self.markers: List[Marker] = []
        self.phase: GamePhase = GamePhase.WHITE_RING_PLACE
        self.current_player: PlayerColor = "white"
        self.removed_rings: Dict[PlayerColor, int] = {"white": 0, "black": 0}
        self.board_size = 11  # Standard YINSH board is 11x11

    def copy(self) -> "GameState":
        """Create a deep copy of this game state"""
        new_state = GameState()
        new_state.rings = [Ring(r.position, r.color) for r in self.rings]
        new_state.markers = [Marker(m.position, m.color) for m in self.markers]
        new_state.phase = self.phase
        new_state.current_player = self.current_player
        new_state.removed_rings = self.removed_rings.copy()
        return new_state

    def get_ring_at(self, position: Coordinate) -> Optional[Ring]:
        """Get ring at specific position"""
        for ring in self.rings:
            if ring.position == position:
                return ring
        return None

    def get_marker_at(self, position: Coordinate) -> Optional[Marker]:
        """Get marker at specific position"""
        for marker in self.markers:
            if marker.position == position:
                return marker
        return None

    def is_position_occupied(self, position: Coordinate) -> bool:
        """Check if position is occupied by ring or marker"""
        return (
            self.get_ring_at(position) is not None
            or self.get_marker_at(position) is not None
        )

    def get_player_rings(self, player: PlayerColor) -> List[Ring]:
        """Get all rings belonging to a player"""
        return [ring for ring in self.rings if ring.color == player]

    def get_player_markers(self, player: PlayerColor) -> List[Marker]:
        """Get all markers belonging to a player"""
        return [marker for marker in self.markers if marker.color == player]

    def to_tensor(self) -> np.ndarray:
        """
        Convert game state to tensor representation for neural network input

        Returns 11x11x11 tensor with the following channels:
        0: White rings (1 where white ring exists, 0 elsewhere)
        1: Black rings (1 where black ring exists, 0 elsewhere)
        2: White markers (1 where white marker exists, 0 elsewhere)
        3: Black markers (1 where black marker exists, 0 elsewhere)
        4: Valid positions mask (1 where moves are possible, 0 elsewhere)
        5: White removable markers (1 where white can remove markers)
        6: Black removable markers (1 where black can remove markers)
        7: Current phase (GamePhase value normalized to [0,1])
        8: Current player (1 for white, -1 for black)
        9: White removed rings count (0-3 normalized to [0,1])
        10: Black removed rings count (0-3 normalized to [0,1])
        """
        tensor = np.zeros((self.board_size, self.board_size, 11), dtype=np.float32)

        # Channel 0: White rings
        for ring in self.rings:
            if ring.color == "white":
                tensor[ring.position[0], ring.position[1], 0] = 1.0

        # Channel 1: Black rings
        for ring in self.rings:
            if ring.color == "black":
                tensor[ring.position[0], ring.position[1], 1] = 1.0

        # Channel 2: White markers
        for marker in self.markers:
            if marker.color == "white":
                tensor[marker.position[0], marker.position[1], 2] = 1.0

        # Channel 3: Black markers
        for marker in self.markers:
            if marker.color == "black":
                tensor[marker.position[0], marker.position[1], 3] = 1.0

        # Channel 4: Valid positions (this would need game rules to compute)
        # For now, mark all empty positions as valid
        for i in range(self.board_size):
            for j in range(self.board_size):
                if not self.is_position_occupied((i, j)):
                    tensor[i, j, 4] = 1.0

        # Channels 5-6: Removable markers (would need game logic)
        # TODO: Implement based on YINSH rules for 5-in-a-row detection

        # Channel 7: Current phase (normalized)
        phase_value = self.phase.value / 7.0  # Normalize to [0,1]
        tensor[:, :, 7] = phase_value

        # Channel 8: Current player
        player_value = 1.0 if self.current_player == "white" else -1.0
        tensor[:, :, 8] = player_value

        # Channel 9: White removed rings (normalized)
        white_removed = self.removed_rings["white"] / 3.0
        tensor[:, :, 9] = white_removed

        # Channel 10: Black removed rings (normalized)
        black_removed = self.removed_rings["black"] / 3.0
        tensor[:, :, 10] = black_removed

        return tensor

    def is_terminal(self) -> bool:
        """Check if game is in terminal state (someone won)"""
        return self.removed_rings["white"] >= 3 or self.removed_rings["black"] >= 3

    def get_winner(self) -> Optional[PlayerColor]:
        """Get winner if game is terminal"""
        if self.removed_rings["white"] >= 3:
            return "white"
        elif self.removed_rings["black"] >= 3:
            return "black"
        else:
            return None

    def __repr__(self) -> str:
        return f"GameState(phase={self.phase}, player={self.current_player}, rings={len(self.rings)}, markers={len(self.markers)})"
