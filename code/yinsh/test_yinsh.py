#!/usr/bin/env python3
"""
Test script for YINSH board game implementation.
"""

import sys
import os

# Add chess module to path
sys.path.insert(0, os.path.dirname(__file__))

from chess.yinsh import Board, Move, WHITE, BLACK, GamePhase

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
    
    from chess.yinsh import position_name, parse_position, YINSH_POSITIONS
    
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

if __name__ == "__main__":
    test_coordinate_system()
    test_move_parsing()
    test_basic_functionality() 