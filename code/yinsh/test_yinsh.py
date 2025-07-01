#!/usr/bin/env python3
"""
Test script for YINSH board game implementation.
"""

import sys
import os
import unittest

# Add chess module to path
sys.path.insert(0, os.path.dirname(__file__))

# Import from current yinsh module
from . import Board, Move, WHITE, BLACK, GamePhase, DIRECTIONS, CANONICAL_DIRECTIONS, coord_to_position, position_to_coord

def test_basic_functionality():
    """Test basic YINSH functionality."""
    print("=== YINSH 보드게임 테스트 ===")
    print()
    
    # Create new board
    board = Board()
    print("1. 새 보드 생성:")
    print(board)
    print()
    
    # Test placement phase
    print("2. 링 배치 단계 테스트:")
    print(f"현재 턴: {'WHITE' if board.turn else 'BLACK'}")
    print(f"게임 단계: {board.phase.value}")
    
    # Get legal moves
    legal_moves = list(board.legal_moves)
    print(f"가능한 수: {len(legal_moves)}개")
    
    # Place some rings
    if legal_moves:
        move = legal_moves[0]
        print(f"첫 번째 수 실행: {move.uci()}")
        board.push(move)
        
        print(f"수 실행 후 턴: {'WHITE' if board.turn else 'BLACK'}")
        print(f"WHITE 링 배치 수: {board.rings_placed[WHITE]}")
        print(f"BLACK 링 배치 수: {board.rings_placed[BLACK]}")
    print()
    
    # Test a few more moves
    print("3. 몇 수 더 진행:")
    for i in range(9):  # Total 10 rings to place
        legal_moves = list(board.legal_moves)
        if legal_moves:
            move = legal_moves[min(i % len(legal_moves), len(legal_moves)-1)]
            print(f"수 {i+2}: {move.uci()}")
            board.push(move)
    
    print(f"링 배치 완료 후 게임 단계: {board.phase.value}")
    print(f"WHITE 링: {board.rings_placed[WHITE]}, BLACK 링: {board.rings_placed[BLACK]}")
    print()
    
    # Test main phase
    if board.phase == GamePhase.MAIN:
        print("4. 메인 게임 단계:")
        print(board)
        print()
        
        # Try some ring moves
        legal_moves = list(board.legal_moves)
        print(f"가능한 링 이동: {len(legal_moves)}개")
        
        if legal_moves:
            move = legal_moves[0]
            print(f"링 이동: {move.uci()}")
            print(f"이동 전 마커 수: {len(board.markers)}")
            board.push(move)
            print(f"이동 후 마커 수: {len(board.markers)}")
            print()
            print(board)
    
    print()
    print("5. 게임 상태:")
    print(f"게임 종료 여부: {board.is_game_over()}")
    print(f"결과: {board.result()}")
    print(f"FEN: {board.fen()}")

def test_move_parsing():
    """Test move parsing functionality."""
    print("\n=== 이동 표기법 테스트 ===")
    
    # Test UCI parsing
    test_ucis = ["@f6", "f6g7", "0000"]
    
    for uci in test_ucis:
        try:
            move = Move.from_uci(uci)
            print(f"UCI '{uci}' -> Move(from={move.from_position}, to={move.to_position})")
            print(f"다시 UCI로: {move.uci()}")
        except Exception as e:
            print(f"UCI '{uci}' 파싱 실패: {e}")
    print()

def test_coordinate_system():
    """Test hexagonal coordinate system."""
    print("=== 육각형 좌표계 테스트 ===")
    
    from . import position_name, parse_position, YINSH_POSITIONS
    
    print(f"총 보드 위치: {len(YINSH_POSITIONS)}")
    
    # Test position naming
    for i in range(min(10, len(YINSH_POSITIONS))):
        name = position_name(i)
        try:
            parsed_back = parse_position(name)
            print(f"위치 {i} -> '{name}' -> {parsed_back} {'✓' if parsed_back == i else '✗'}")
        except Exception as e:
            print(f"위치 {i} -> '{name}' -> 오류: {e}")
    print()

def test_hexagonal_directions():
    """Test hexagonal directions by examining actual board positions."""
    print("=== 육각형 방향 검증 ===")
    
    from . import YINSH_POSITIONS, DIRECTIONS, coord_to_position, position_to_coord
    
    # Test center position (0, 0)
    center = (0, 0)
    if center in YINSH_POSITIONS:
        print(f"중심점 {center} 존재 확인")
        
        # Test each direction from center
        for i, direction in enumerate(DIRECTIONS):
            dx, dy = direction
            new_pos = (center[0] + dx, center[1] + dy)
            
            exists = new_pos in YINSH_POSITIONS
            print(f"방향 {i+1}: {direction} -> {new_pos} {'✓' if exists else '✗'}")
            
            if not exists:
                print(f"  ❌ 경고: 방향 {direction}는 유효하지 않습니다!")
    
    print()
    
    # Test a few other positions for consistency
    test_positions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    
    for test_pos in test_positions:
        if test_pos in YINSH_POSITIONS:
            print(f"위치 {test_pos}에서 인접 위치들:")
            valid_neighbors = 0
            
            for direction in DIRECTIONS:
                dx, dy = direction
                neighbor = (test_pos[0] + dx, test_pos[1] + dy)
                exists = neighbor in YINSH_POSITIONS
                
                if exists:
                    valid_neighbors += 1
                    
            print(f"  유효한 인접 위치: {valid_neighbors}/6")
            if valid_neighbors < 4:  # Edge positions might have fewer neighbors
                print(f"  ⚠️  가장자리 위치이거나 방향 설정에 문제가 있을 수 있습니다")
    
    print()

def test_actual_neighbor_analysis():
    """Analyze actual neighbors to determine correct hexagonal directions."""
    print("=== 실제 인접 위치 분석 ===")
    
    from . import YINSH_POSITIONS
    
    # Find actual neighbors of center position (0,0)
    center = (0, 0)
    if center not in YINSH_POSITIONS:
        print("중심점 (0,0)이 없습니다!")
        return
        
    # Check all possible adjacent positions (8 directions)
    all_directions = [
        (1, 0), (-1, 0),      # horizontal
        (0, 1), (0, -1),      # vertical  
        (1, 1), (-1, -1),     # diagonal 1
        (1, -1), (-1, 1)      # diagonal 2
    ]
    
    print(f"중심점 {center}의 주변 위치 분석:")
    actual_neighbors = []
    
    for direction in all_directions:
        dx, dy = direction
        neighbor = (center[0] + dx, center[1] + dy)
        exists = neighbor in YINSH_POSITIONS
        
        if exists:
            actual_neighbors.append(direction)
            print(f"  {direction} -> {neighbor} ✓")
        else:
            print(f"  {direction} -> {neighbor} ✗")
    
    print(f"\n실제 인접 방향들: {actual_neighbors}")
    print(f"총 인접 위치 개수: {len(actual_neighbors)}")
    
    # Check if this forms a valid hexagon
    if len(actual_neighbors) == 6:
        print("✓ 6개 방향 확인됨 - 육각형 구조")
    else:
        print(f"❌ {len(actual_neighbors)}개 방향 - 육각형이 아님")
    
    return actual_neighbors

class TestYinshDirections(unittest.TestCase):
    """Test hexagonal direction system"""
    
    def test_six_directions(self):
        """Test that we have exactly 6 directions for hexagonal movement"""
        self.assertEqual(len(DIRECTIONS), 6)
        
    def test_canonical_directions(self):
        """Test that canonical directions cover all needed lines"""
        expected_canonical = {
            (0, 1),   # 12시-6시 라인
            (1, 1),   # 2시-8시 라인  
            (1, 0),   # 4시-10시 라인
        }
        
        actual_canonical = set(CANONICAL_DIRECTIONS)
        self.assertEqual(actual_canonical, expected_canonical)
        
    def test_direction_vectors(self):
        """Test that direction vectors are valid hexagonal directions"""
        expected_directions = {
            (0, 1),   # 12시 북쪽
            (1, 1),   # 2시 북동쪽  
            (1, 0),   # 4시 동쪽
            (0, -1),  # 6시 남쪽
            (-1, -1), # 8시 남서쪽
            (-1, 0),  # 10시 서쪽
        }
        
        actual_directions = set(DIRECTIONS)
        self.assertEqual(actual_directions, expected_directions)

class TestYinshBoard(unittest.TestCase):
    """Test YINSH board functionality"""
    
    def setUp(self):
        self.board = Board()
    
    def test_board_initialization(self):
        """Test that board initializes correctly"""
        self.assertIsNotNone(self.board)
        self.assertEqual(len(self.board.rings), 0)
        self.assertEqual(len(self.board.markers), 0)
    
    def test_get_path_between_same_position(self):
        """Test path between same position"""
        path = self.board.get_path_between(0, 0)
        self.assertEqual(path, [0])
    
    def test_get_path_between_valid_directions(self):
        """Test path calculation for valid hexagonal directions"""
        # Test a few known valid positions and directions
        if coord_to_position(0, 0) is not None and coord_to_position(1, 0) is not None:
            pos_center = coord_to_position(0, 0)
            pos_east = coord_to_position(1, 0)
            
            if pos_center is not None and pos_east is not None:
                path = self.board.get_path_between(pos_center, pos_east)
                self.assertIsNotNone(path)
                self.assertEqual(len(path), 2)
                self.assertEqual(path[0], pos_center)
                self.assertEqual(path[1], pos_east)
    
    def test_generate_placement_moves(self):
        """Test placement move generation"""
        moves = list(self.board.generate_placement_moves())
        # Should be able to place on any empty position (85 total)
        self.assertTrue(len(moves) > 0)
        self.assertTrue(len(moves) <= 85)
    
    def test_check_five_in_row_empty_board(self):
        """Test five-in-row detection on empty board"""
        lines_white = self.board.check_five_in_row(True)
        lines_black = self.board.check_five_in_row(False)
        self.assertEqual(len(lines_white), 0)
        self.assertEqual(len(lines_black), 0)

class TestYinshMoves(unittest.TestCase):
    """Test YINSH move functionality"""
    
    def test_move_creation(self):
        """Test move object creation"""
        # Placement move
        placement_move = Move(to_position=0)
        self.assertEqual(placement_move.to_position, 0)
        self.assertIsNone(placement_move.from_position)
        
        # Ring movement
        ring_move = Move(from_position=0, to_position=1)
        self.assertEqual(ring_move.from_position, 0)
        self.assertEqual(ring_move.to_position, 1)
        
        # Marker removal
        marker_move = Move(positions=[0, 1, 2, 3, 4])
        self.assertEqual(len(marker_move.positions), 5)

if __name__ == "__main__":
    test_coordinate_system()
    test_move_parsing()
    test_basic_functionality()
    test_hexagonal_directions()
    test_actual_neighbor_analysis()
    unittest.main() 