"""
YINSH board game implementation following chess library conventions.

YINSH is an abstract strategy game where players place rings and markers
on a hexagonal board to form lines of 5 markers in their color.
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import enum
import re
import typing

from typing import ClassVar, Dict, Iterator, List, Literal, Optional, Set, Tuple, Union

# Type aliases
Color: typing.TypeAlias = bool
WHITE: Color = True
BLACK: Color = False
COLORS: List[Color] = [WHITE, BLACK]

# YINSH board valid positions (from API implementation)
# Using (x, y) coordinate system with center at (0, 0)
YINSH_POSITIONS = {
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

# Create position mapping
POSITION_LIST = sorted(YINSH_POSITIONS, key=lambda p: (p[1], p[0]))  # Sort by y, then x
POSITION_TO_INDEX = {pos: i for i, pos in enumerate(POSITION_LIST)}
INDEX_TO_POSITION = {i: pos for i, pos in enumerate(POSITION_LIST)}

Position: typing.TypeAlias = int

def coord_to_position(x: int, y: int) -> Optional[Position]:
    """Convert (x, y) coordinate to position index."""
    return POSITION_TO_INDEX.get((x, y))

def position_to_coord(position: Position) -> Tuple[int, int]:
    """Convert position index to (x, y) coordinate."""
    return INDEX_TO_POSITION[position]

def position_name(position: Position) -> str:
    """Get position name like 'e4' from position number."""
    x, y = position_to_coord(position)
    # Convert to a simple naming scheme (x=0, y=0 becomes 'f6')
    file_letter = chr(ord('a') + x + 5)
    rank_number = y + 6
    return f"{file_letter}{rank_number}"

def parse_position(name: str) -> Position:
    """Parse position name like 'e4' to position number."""
    if len(name) < 2:
        raise ValueError(f"Invalid position name: {name}")
    
    file_letter = name[0].lower()
    rank_str = name[1:]
    
    try:
        x = ord(file_letter) - ord('a') - 5
        y = int(rank_str) - 6
        
        pos = coord_to_position(x, y)
        if pos is None:
            raise ValueError(f"Position not on board: {name}")
        return pos
    except (ValueError, IndexError):
        raise ValueError(f"Invalid position name: {name}")

# Direction vectors for YINSH hexagonal board (6 directions)
# Based on actual YINSH board coordinate system:
# 12 o'clock (North) = +y axis, 4 o'clock (East) = +x axis
DIRECTIONS = [
    (0, 1),   # 12시 (북쪽)
    (1, 1),   # 2시 (북동쪽)  
    (1, 0),   # 4시 (동쪽)
    (0, -1),  # 6시 (남쪽)
    (-1, -1), # 8시 (남서쪽)
    (-1, 0),  # 10시 (서쪽)
]

# Canonical directions to avoid checking duplicate lines (3 directions only)
# When checking for lines, we only need to check in 3 directions since
# checking both directions of a line would be redundant
CANONICAL_DIRECTIONS = [
    (0, 1),   # 12시-6시 (North-South lines)
    (1, 1),   # 2시-8시 (Northeast-Southwest lines)  
    (1, 0),   # 4시-10시 (East-West lines)
]

# Game piece types
PieceType: typing.TypeAlias = int
RING: PieceType = 1
MARKER: PieceType = 2

class GamePhase(enum.Enum):
    """Game phases in YINSH"""
    PLACEMENT = "placement"      # Ring placement phase
    MAIN = "main"               # Main game phase (ring movement)
    MARKER_REMOVE = "marker_remove"  # Removing 5 consecutive markers
    RING_REMOVE = "ring_remove"      # Removing a ring after marker removal
    ENDED = "ended"             # Game ended

class InvalidMoveError(ValueError):
    """Raised when move notation is not syntactically valid"""

class IllegalMoveError(ValueError):
    """Raised when the attempted move is illegal in the current position"""

@dataclasses.dataclass
class Piece:
    """A YINSH piece with type and color."""
    
    piece_type: PieceType
    color: Color
    
    def symbol(self) -> str:
        """Get symbol for the piece."""
        if self.piece_type == RING:
            return "R" if self.color else "r"
        elif self.piece_type == MARKER:
            return "M" if self.color else "m"
        return "?"
    
    def __str__(self) -> str:
        return self.symbol()

@dataclasses.dataclass(unsafe_hash=True)
class Move:
    """
    Represents a move in YINSH.
    
    For placement phase: only to_position is used
    For main phase: from_position and to_position for ring movement
    For marker removal: positions contains list of 5 positions to remove
    For ring removal: to_position indicates which ring to remove
    """
    
    from_position: Optional[Position] = None
    to_position: Optional[Position] = None
    positions: Optional[List[Position]] = None  # For marker removal
    
    def __bool__(self) -> bool:
        return self.to_position is not None or bool(self.positions)
    
    def uci(self) -> str:
        """Get UCI notation for the move."""
        if not self:
            return "0000"
        
        if self.positions:
            # Marker removal move
            pos_names = [position_name(pos) for pos in self.positions]
            return f"x{'-'.join(pos_names)}"
        elif self.from_position is None:
            # Placement or ring removal move
            return f"@{position_name(self.to_position)}"
        else:
            # Ring movement
            return f"{position_name(self.from_position)}{position_name(self.to_position)}"
    
    @classmethod
    def from_uci(cls, uci: str) -> Move:
        """Parse UCI notation to create a move."""
        if uci == "0000":
            return cls()
        
        if uci.startswith("x"):
            # Marker removal move
            pos_names = uci[1:].split("-")
            positions = [parse_position(name) for name in pos_names]
            return cls(positions=positions)
        elif uci.startswith("@"):
            # Placement or ring removal move
            pos_name = uci[1:]
            return cls(to_position=parse_position(pos_name))
        elif len(uci) >= 4:
            # Ring movement - handle variable length position names
            mid = len(uci) // 2
            from_name = uci[:mid]
            to_name = uci[mid:]
            return cls(
                from_position=parse_position(from_name),
                to_position=parse_position(to_name)
            )
        else:
            raise InvalidMoveError(f"Invalid UCI: {uci}")
    
    @classmethod
    def null(cls) -> Move:
        """Create a null move."""
        return cls()

class Board:
    """
    YINSH board implementation.
    
    The board has 85 intersections arranged in a hexagonal pattern.
    Players place rings and markers on these intersections.
    """
    
    aliases: ClassVar[List[str]] = ["YINSH", "Yinsh"]
    uci_variant: ClassVar[Optional[str]] = "yinsh"
    starting_fen: ClassVar[str] = "85/WHITE/PLACEMENT/0-0/1"
    
    def __init__(self, fen: Optional[str] = None) -> None:
        # Board state
        self.rings: Dict[Position, Color] = {}
        self.markers: Dict[Position, Color] = {}
        
        # Game state
        self.turn: Color = WHITE
        self.phase: GamePhase = GamePhase.PLACEMENT
        self.rings_placed: Dict[Color, int] = {WHITE: 0, BLACK: 0}
        self.rings_removed: Dict[Color, int] = {WHITE: 0, BLACK: 0}
        self.markers_available: int = 51
        
        # Current five-in-row lines that need to be processed
        self.pending_five_lines: Dict[Color, List[List[Position]]] = {WHITE: [], BLACK: []}
        
        # Move tracking
        self.move_stack: List[Move] = []
        self.fullmove_number: int = 1
        
        if fen:
            self.set_fen(fen)
        else:
            self.reset()
    
    def reset(self) -> None:
        """Reset to starting position."""
        self.rings.clear()
        self.markers.clear()
        self.turn = WHITE
        self.phase = GamePhase.PLACEMENT
        self.rings_placed = {WHITE: 0, BLACK: 0}
        self.rings_removed = {WHITE: 0, BLACK: 0}
        self.markers_available = 51
        self.pending_five_lines = {WHITE: [], BLACK: []}
        self.move_stack.clear()
        self.fullmove_number = 1
    
    def copy(self) -> Board:
        """Create a copy of the board."""
        board = Board()
        board.rings = self.rings.copy()
        board.markers = self.markers.copy()
        board.turn = self.turn
        board.phase = self.phase
        board.rings_placed = self.rings_placed.copy()
        board.rings_removed = self.rings_removed.copy()
        board.markers_available = self.markers_available
        board.pending_five_lines = {
            WHITE: [line.copy() for line in self.pending_five_lines[WHITE]],
            BLACK: [line.copy() for line in self.pending_five_lines[BLACK]]
        }
        board.move_stack = self.move_stack.copy()
        board.fullmove_number = self.fullmove_number
        return board
    
    def piece_at(self, position: Position) -> Optional[Piece]:
        """Get piece at position."""
        if position in self.rings:
            return Piece(RING, self.rings[position])
        elif position in self.markers:
            return Piece(MARKER, self.markers[position])
        return None
    
    def is_empty(self, position: Position) -> bool:
        """Check if position is empty."""
        return position not in self.rings and position not in self.markers
    
    def get_line_positions(self, start_pos: Position, direction: Tuple[int, int], max_length: int = 11) -> List[Position]:
        """Get positions in a line from start position in given direction."""
        positions = []
        x, y = position_to_coord(start_pos)
        dx, dy = direction
        
        for i in range(max_length):
            current_x = x + dx * i
            current_y = y + dy * i
            pos = coord_to_position(current_x, current_y)
            if pos is None:
                break
            positions.append(pos)
        
        return positions
    
    def get_path_between(self, from_pos: Position, to_pos: Position) -> Optional[List[Position]]:
        """Get path between two positions if they're in a straight line in hexagonal directions."""
        from_x, from_y = position_to_coord(from_pos)
        to_x, to_y = position_to_coord(to_pos)
        
        diff_x = to_x - from_x
        diff_y = to_y - from_y
        
        # If positions are the same, return single position
        if diff_x == 0 and diff_y == 0:
            return [from_pos]
        
        # Check if the difference vector is a multiple of one of the 6 hexagonal directions
        for dx, dy in DIRECTIONS:
            if dx == 0 and dy == 0:
                continue
                
            # Check if diff is a multiple of this direction
            # We need to find a scaling factor k such that (diff_x, diff_y) = k * (dx, dy)
            if dx == 0:
                if diff_x != 0:
                    continue
                if dy == 0:
                    continue
                if diff_y % dy != 0:
                    continue
                k = diff_y // dy
            elif dy == 0:
                if diff_y != 0:
                    continue
                if diff_x % dx != 0:
                    continue
                k = diff_x // dx
            else:
                # Both dx and dy are non-zero
                if diff_x % dx != 0 or diff_y % dy != 0:
                    continue
                k_x = diff_x // dx
                k_y = diff_y // dy
                if k_x != k_y:
                    continue
                k = k_x
            
            # If k is positive, we found a valid direction
            if k > 0:
                path = []
                for i in range(k + 1):
                    x = from_x + dx * i
                    y = from_y + dy * i
                    pos = coord_to_position(x, y)
                    if pos is None:
                        return None  # Path goes off the board
                    path.append(pos)
                return path
        
        return None  # No valid straight line path found
    
    def check_five_in_row(self, color: Color) -> List[List[Position]]:
        """Check for lines of 5+ consecutive markers of the same color using hexagonal directions."""
        lines = []
        checked = set()
        
        # Get all marker positions for this color
        marker_positions = set()
        for pos, marker_color in self.markers.items():
            if marker_color == color:
                marker_positions.add(pos)
        
        # Use canonical directions to avoid duplicates (3 directions for hexagonal board)
        # We only check in 3 directions since the opposite directions would create duplicate lines
        
        # Check each marker position as potential start of a line
        for start_pos in marker_positions:
            if start_pos in checked:
                continue
                
            x, y = position_to_coord(start_pos)
            
            for dx, dy in CANONICAL_DIRECTIONS:
                # Check if this is actually the start of the line (no marker behind it)
                prev_x, prev_y = x - dx, y - dy
                prev_pos = coord_to_position(prev_x, prev_y)
                if prev_pos is not None and prev_pos in marker_positions:
                    continue  # Not the start of the line
                
                # Build consecutive chain in this direction
                chain = []
                current_x, current_y = x, y
                
                while True:
                    current_pos = coord_to_position(current_x, current_y)
                    if current_pos is None or current_pos not in marker_positions:
                        break
                    chain.append(current_pos)
                    current_x += dx
                    current_y += dy
                
                # If we have 5+ consecutive markers, create all possible 5-marker segments
                if len(chain) >= 5:
                    for i in range(len(chain) - 4):
                        segment = chain[i:i+5]
                        lines.append(segment)
                        checked.update(segment)
        
        return lines
    
    def generate_placement_moves(self) -> Iterator[Move]:
        """Generate legal placement moves."""
        if self.phase != GamePhase.PLACEMENT:
            return
        
        if self.rings_placed[self.turn] >= 5:
            return
        
        # Can place ring on any empty position
        for pos in range(len(POSITION_LIST)):
            if self.is_empty(pos):
                yield Move(to_position=pos)
    
    def generate_ring_moves(self) -> Iterator[Move]:
        """Generate legal ring movement moves."""
        if self.phase != GamePhase.MAIN:
            return
        
        # Find rings belonging to current player
        player_rings = [pos for pos, color in self.rings.items() if color == self.turn]
        
        for ring_pos in player_rings:
            # Generate moves for this ring in all 6 hexagonal directions
            for direction in DIRECTIONS:
                positions = self.get_line_positions(ring_pos, direction)
                
                # Skip the ring's current position
                for i, pos in enumerate(positions[1:], 1):
                    if pos in self.rings:
                        # Blocked by another ring
                        break
                    
                    if pos not in self.markers:
                        # Empty position - valid destination
                        yield Move(from_position=ring_pos, to_position=pos)
                    else:
                        # Marker found - must jump over consecutive markers
                        # Find end of marker sequence
                        jump_end = None
                        for j in range(i, len(positions)):
                            if positions[j] not in self.markers:
                                jump_end = positions[j]
                                break
                            elif positions[j] in self.rings:
                                # Blocked by ring after markers
                                break
                        
                        if jump_end is not None and self.is_empty(jump_end):
                            yield Move(from_position=ring_pos, to_position=jump_end)
                        break
    
    def generate_marker_removal_moves(self) -> Iterator[Move]:
        """Generate legal marker removal moves."""
        if self.phase != GamePhase.MARKER_REMOVE:
            return
        
        # Return all possible 5-consecutive marker lines for current player
        lines = self.check_five_in_row(self.turn)
        for line in lines:
            yield Move(positions=line)
    
    def generate_ring_removal_moves(self) -> Iterator[Move]:
        """Generate legal ring removal moves."""
        if self.phase != GamePhase.RING_REMOVE:
            return
        
        # Can remove any ring belonging to current player
        player_rings = [pos for pos, color in self.rings.items() if color == self.turn]
        for ring_pos in player_rings:
            yield Move(to_position=ring_pos)
    
    def generate_legal_moves(self) -> Iterator[Move]:
        """Generate all legal moves for current position."""
        if self.phase == GamePhase.PLACEMENT:
            yield from self.generate_placement_moves()
        elif self.phase == GamePhase.MAIN:
            yield from self.generate_ring_moves()
        elif self.phase == GamePhase.MARKER_REMOVE:
            yield from self.generate_marker_removal_moves()
        elif self.phase == GamePhase.RING_REMOVE:
            yield from self.generate_ring_removal_moves()
    
    def is_legal(self, move: Move) -> bool:
        """Check if move is legal."""
        for legal_move in self.generate_legal_moves():
            if (legal_move.from_position == move.from_position and
                legal_move.to_position == move.to_position and
                legal_move.positions == move.positions):
                return True
        return False
    
    def flip_markers_on_path(self, from_pos: Position, to_pos: Position) -> None:
        """Flip all markers on the path between two positions."""
        path = self.get_path_between(from_pos, to_pos)
        if path is None:
            return
        
        # Skip first (ring starting position) and last (ring destination)
        for pos in path[1:-1]:
            if pos in self.markers:
                # Flip marker color
                self.markers[pos] = not self.markers[pos]
    
    def push(self, move: Move) -> None:
        """Make a move."""
        if not self.is_legal(move):
            raise IllegalMoveError(f"Illegal move: {move.uci()}")
        
        self.move_stack.append(move)
        
        if self.phase == GamePhase.PLACEMENT:
            # Place ring
            if move.to_position is not None:
                self.rings[move.to_position] = self.turn
                self.rings_placed[self.turn] += 1
            
            # Check if placement phase is over
            if all(count >= 5 for count in self.rings_placed.values()):
                self.phase = GamePhase.MAIN
        
        elif self.phase == GamePhase.MAIN:
            # Ring movement
            if move.from_position is not None and move.to_position is not None:
                # Place marker at ring's current position
                self.markers[move.from_position] = self.turn
                self.markers_available -= 1
                
                # Move ring
                ring_color = self.rings[move.from_position]
                del self.rings[move.from_position]
                self.rings[move.to_position] = ring_color
                
                # Flip markers on path
                self.flip_markers_on_path(move.from_position, move.to_position)
                
                # Check for five in a row for current player first
                current_player_lines = self.check_five_in_row(self.turn)
                if current_player_lines:
                    self.phase = GamePhase.MARKER_REMOVE
                    # Don't switch turns yet - same player removes markers
                    return
                
                # Check for five in a row for opponent
                opponent_lines = self.check_five_in_row(not self.turn)
                if opponent_lines:
                    # Switch to opponent for marker removal
                    self.turn = not self.turn
                    if self.turn == WHITE:
                        self.fullmove_number += 1
                    self.phase = GamePhase.MARKER_REMOVE
                    return
        
        elif self.phase == GamePhase.MARKER_REMOVE:
            # Remove 5 consecutive markers
            if move.positions and len(move.positions) == 5:
                for pos in move.positions:
                    if pos in self.markers:
                        del self.markers[pos]
                        self.markers_available += 1
                
                # Move to ring removal phase
                self.phase = GamePhase.RING_REMOVE
                # Don't switch turns yet - same player removes a ring
                return
        
        elif self.phase == GamePhase.RING_REMOVE:
            # Remove ring
            if move.to_position is not None and move.to_position in self.rings:
                if self.rings[move.to_position] == self.turn:
                    del self.rings[move.to_position]
                    self.rings_removed[self.turn] += 1
                
                # Check for more five-in-row lines
                lines = self.check_five_in_row(self.turn)
                if lines:
                    self.phase = GamePhase.MARKER_REMOVE
                    # Don't switch turns - continue removing
                    return
                else:
                    # Back to main game
                    self.phase = GamePhase.MAIN
        
        # Switch turns
        self.turn = not self.turn
        if self.turn == WHITE:
            self.fullmove_number += 1
        
        # Check for game end (3 rings removed = win)
        if any(count >= 3 for count in self.rings_removed.values()):
            self.phase = GamePhase.ENDED
    
    def pop(self) -> Move:
        """Undo last move."""
        if not self.move_stack:
            raise IndexError("No moves to undo")
        
        # This is a simplified undo - in practice you'd need to store more state
        move = self.move_stack.pop()
        
        # Switch back turn
        if self.turn == WHITE:
            self.fullmove_number -= 1
        self.turn = not self.turn
        
        return move
    
    def is_game_over(self) -> bool:
        """Check if game is over."""
        return (self.phase == GamePhase.ENDED or 
                any(count >= 3 for count in self.rings_removed.values()) or
                self.markers_available <= 0)
    
    def result(self) -> str:
        """Get game result."""
        if not self.is_game_over():
            return "*"
        
        if self.rings_removed[WHITE] >= 3:
            return "1-0"
        elif self.rings_removed[BLACK] >= 3:
            return "0-1"
        else:
            return "1/2-1/2"
    
    def fen(self) -> str:
        """Get FEN representation."""
        # Create position string
        pieces = ["." for _ in range(len(POSITION_LIST))]
        for pos, color in self.rings.items():
            pieces[pos] = "R" if color == WHITE else "r"
        for pos, color in self.markers.items():
            pieces[pos] = "M" if color == WHITE else "m"
        
        pieces_str = "".join(pieces)
        turn_str = "WHITE" if self.turn == WHITE else "BLACK"
        phase_str = self.phase.value.upper()
        rings_str = f"{self.rings_removed[WHITE]}-{self.rings_removed[BLACK]}"
        
        return f"{pieces_str}/{turn_str}/{phase_str}/{rings_str}/{self.fullmove_number}"
    
    def set_fen(self, fen: str) -> None:
        """Set position from FEN."""
        parts = fen.split("/")
        if len(parts) != 5:
            raise ValueError("Invalid FEN format")
        
        # Reset board
        self.reset()
        
        # Parse pieces
        pieces_str = parts[0]
        for i, symbol in enumerate(pieces_str):
            if i >= len(POSITION_LIST):
                break
            if symbol == "R":
                self.rings[i] = WHITE
            elif symbol == "r":
                self.rings[i] = BLACK
            elif symbol == "M":
                self.markers[i] = WHITE
            elif symbol == "m":
                self.markers[i] = BLACK
        
        # Parse turn
        self.turn = WHITE if parts[1] == "WHITE" else BLACK
        
        # Parse phase
        phase_str = parts[2].lower()
        if phase_str == "placement":
            self.phase = GamePhase.PLACEMENT
        elif phase_str == "main":
            self.phase = GamePhase.MAIN
        elif phase_str == "marker_remove":
            self.phase = GamePhase.MARKER_REMOVE
        elif phase_str == "ring_remove":
            self.phase = GamePhase.RING_REMOVE
        elif phase_str == "ended":
            self.phase = GamePhase.ENDED
        
        # Parse rings removed
        rings_parts = parts[3].split("-")
        self.rings_removed[WHITE] = int(rings_parts[0])
        self.rings_removed[BLACK] = int(rings_parts[1])
        
        # Parse move number
        self.fullmove_number = int(parts[4])
    
    def __str__(self) -> str:
        """String representation of the board."""
        lines = []
        lines.append("YINSH Board:")
        lines.append(f"Turn: {'WHITE' if self.turn else 'BLACK'}")
        lines.append(f"Phase: {self.phase.value}")
        lines.append(f"Rings removed: W:{self.rings_removed[WHITE]} B:{self.rings_removed[BLACK]}")
        lines.append(f"Rings placed: W:{self.rings_placed[WHITE]} B:{self.rings_placed[BLACK]}")
        lines.append(f"Total pieces: {len(self.rings)} rings, {len(self.markers)} markers")
        lines.append("")
        
        # Create 11x11 grid representation
        lines.append("Board layout (● = valid position, R/r = rings, M/m = markers):")
        for y in range(5, -6, -1):  # y: 5 to -5
            line = f"y={y:2d}: "
            for x in range(-5, 6):  # x: -5 to 5
                pos = coord_to_position(x, y)
                if pos is None:
                    line += "·"
                else:
                    piece = self.piece_at(pos)
                    if piece:
                        line += piece.symbol()
                    else:
                        line += "●"
                line += " "
            lines.append(line)
        
        lines.append(f"     x: {' '.join(f'{i:1d}' for i in range(-5, 6))}")
        
        return "\n".join(lines)

# Legal move generators
class LegalMoveGenerator:
    def __init__(self, board: Board) -> None:
        self.board = board
    
    def __iter__(self) -> Iterator[Move]:
        return self.board.generate_legal_moves()
    
    def __bool__(self) -> bool:
        return any(self.board.generate_legal_moves())
    
    def count(self) -> int:
        return len(list(self.board.generate_legal_moves()))

# Add legal_moves property to Board class
Board.legal_moves = property(lambda self: LegalMoveGenerator(self))

# Move encoding/decoding functions for neural network
def encode_move(move: Move, board: Board) -> int:
    """
    Encode a YINSH move to an integer action for neural network.
    
    Encoding scheme:
    - Ring placement: position (0-84)
    - Ring movement: from_pos * 85 + to_pos + 85 (85-7309)
    - Marker removal: 7310 + legal_move_index (7310+)
    - Ring removal: position + 85 (85-169, but distinguished by game phase)
    
    Returns action index (0 to action_space_size-1)
    """
    if not move:
        return 0  # Null move
    
    if move.positions:
        # Marker removal - find index among legal marker removal moves
        legal_marker_moves = list(board.generate_marker_removal_moves())
        for i, legal_move in enumerate(legal_marker_moves):
            if legal_move.positions == move.positions:
                return 7310 + i
        return None  # Invalid marker removal move
    elif move.from_position is None:
        # Ring placement or ring removal
        if board.phase == GamePhase.PLACEMENT:
            return move.to_position  # 0-84
        elif board.phase == GamePhase.RING_REMOVE:
            return 85 + move.to_position  # 85-169
        return None
    else:
        # Ring movement - from_pos * 85 + to_pos + 85
        if 0 <= move.from_position < 85 and 0 <= move.to_position < 85:
            return 85 + move.from_position * 85 + move.to_position  # 85-7309
        return None

def decode_move(action: int, board: Board) -> Move:
    """
    Decode an integer action to a YINSH move.
    
    Returns the corresponding Move object, or None if invalid.
    """
    if action == 0:
        return Move.null()
    
    if action < 85:
        # Ring placement (only valid in placement phase)
        if board.phase == GamePhase.PLACEMENT:
            return Move(to_position=action)
        return None
    elif action < 170:
        # Ring removal (only valid in ring removal phase) 
        if board.phase == GamePhase.RING_REMOVE:
            return Move(to_position=action - 85)
        return None
    elif action < 7310:
        # Ring movement (only valid in main phase)
        if board.phase == GamePhase.MAIN:
            action -= 85
            from_pos = action // 85
            to_pos = action % 85
            if 0 <= from_pos < 85 and 0 <= to_pos < 85:
                return Move(from_position=from_pos, to_position=to_pos)
        return None
    else:
        # Marker removal (only valid in marker removal phase)
        if board.phase == GamePhase.MARKER_REMOVE:
            legal_marker_moves = list(board.generate_marker_removal_moves())
            move_index = action - 7310
            if 0 <= move_index < len(legal_marker_moves):
                return legal_marker_moves[move_index]
        return None

def get_action_space_size() -> int:
    """
    Get the size of the action space for YINSH.
    
    Returns:
        int: Total number of possible actions
    """
    # Ring placement: 85 (0-84)
    # Ring removal: 85 (85-169) 
    # Ring movement: 85 * 85 = 7225 (170-7309)
    # Marker removal: ~500 (estimated max, 7310+)
    return 7895  # Total actions

def move_to_action_probabilities(moves: List[Move], board: Board) -> Dict[int, float]:
    """
    Convert a list of legal moves to action probabilities.
    
    Args:
        moves: List of legal Move objects
        board: Current board state
        
    Returns:
        Dict mapping action indices to equal probabilities
    """
    if not moves:
        return {}
    
    prob = 1.0 / len(moves)
    action_probs = {}
    
    for move in moves:
        action = encode_move(move, board)
        if action is not None:
            action_probs[action] = prob
    
    return action_probs

def get_legal_action_mask(board: Board) -> List[bool]:
    """
    Get a boolean mask of legal actions for the current board state.
    
    Args:
        board: Current board state
        
    Returns:
        List of booleans indicating which actions are legal
    """
    action_space_size = get_action_space_size()
    mask = [False] * action_space_size
    
    for move in board.generate_legal_moves():
        action = encode_move(move, board)
        if action is not None and 0 <= action < action_space_size:
            mask[action] = True
    
    return mask 