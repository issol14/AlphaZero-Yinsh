#!/usr/bin/env python3
"""
Test script for YINSH board game implementation (code_old version).
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import YINSH modules
import yinsh
from yinsh import Board, Move, WHITE, BLACK, GamePhase
import yinshEnv
import config


def test_basic_yinsh_game():
    """기본 YINSH 게임 테스트"""
    print("=== YINSH 보드게임 테스트 (code_old) ===")
    print()

    # 1. 새 보드 생성
    board = Board()
    print("1. 새 보드 생성:")
    print(board)
    print()

    # 2. 링 배치 단계 테스트
    print("2. 링 배치 단계 테스트:")
    print(f"현재 턴: {'WHITE' if board.turn else 'BLACK'}")
    print(f"게임 단계: {board.phase.value}")

    # 가능한 수 확인
    legal_moves = list(board.legal_moves)
    print(f"가능한 수: {len(legal_moves)}개")

    # 몇 수 진행
    if legal_moves:
        for i in range(min(10, len(legal_moves))):
            move = legal_moves[i]
            print(f"수 {i+1}: {move.uci()}")
            board.push(move)
            print(f"  턴: {'WHITE' if board.turn else 'BLACK'}")
            print(
                f"  WHITE 링: {board.rings_placed[WHITE]}, BLACK 링: {board.rings_placed[BLACK]}"
            )

    print(f"링 배치 완료 후 게임 단계: {board.phase.value}")
    print()

    # 3. 메인 게임 단계 테스트
    if board.phase == GamePhase.MAIN:
        print("3. 메인 게임 단계:")
        print(board)

        # 링 이동 테스트
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
    print("4. 게임 상태:")
    print(f"게임 종료 여부: {board.is_game_over()}")
    print(f"결과: {board.result()}")
    print(f"FEN: {board.fen()}")


def test_yinsh_env():
    """YINSH 환경 래퍼 테스트"""
    print("\n=== YINSH 환경 래퍼 테스트 ===")

    # 환경 생성
    env = yinshEnv.YinshEnv()
    print("1. YINSH 환경 생성 완료")

    # 초기 상태
    print(f"2. 초기 상태:")
    print(f"  턴: {'WHITE' if env.board.turn else 'BLACK'}")
    print(f"  게임 단계: {env.board.phase.value}")
    print(f"  링 개수: {len(env.board.rings)}")
    print(f"  마커 개수: {len(env.board.markers)}")

    # 가능한 액션
    legal_moves = env.get_legal_moves()
    print(f"3. 가능한 액션: {len(legal_moves)}개")

    # 상태 인코딩 테스트
    state_tensor = yinshEnv.YinshEnv.state_to_input(env.board.fen())
    print(f"4. 상태 텐서 형태: {state_tensor.shape}")
    print(f"   예상 형태: (1, 11, 11, 13)")

    # 승리 추정 테스트
    winner_estimate = yinshEnv.YinshEnv.estimate_winner(env.board)
    print(f"5. 승리 추정: {winner_estimate}")

    # 몇 수 진행
    print("6. 몇 수 진행:")
    for i in range(min(5, len(legal_moves))):
        move = legal_moves[i]
        print(f"  수 {i+1}: {move.uci()}")
        env.step(move)

        # 상태 업데이트
        state_tensor = yinshEnv.YinshEnv.state_to_input(env.board.fen())
        winner_estimate = yinshEnv.YinshEnv.estimate_winner(env.board)
        print(f"    상태 형태: {state_tensor.shape}, 승리 추정: {winner_estimate}")


def test_config():
    """설정 파일 테스트"""
    print("\n=== 설정 파일 테스트 ===")

    print(f"1. MCTS 설정:")
    print(f"  시뮬레이션 수: {config.SIMULATIONS_PER_MOVE}")
    print(f"  C_base: {config.C_base}")
    print(f"  C_init: {config.C_init}")

    print(f"2. 신경망 입력:")
    print(f"  YINSH 보드 크기: {config.yinsh_board_size}")
    print(f"  입력 채널 수: {config.yinsh_input_planes}")
    print(f"  입력 형태: {config.YINSH_INPUT_SHAPE}")

    print(f"3. 신경망 출력:")
    print(f"  액션 공간 크기: {config.YINSH_ACTION_SPACE_SIZE}")
    print(f"  출력 형태: {config.YINSH_OUTPUT_SHAPE}")

    print(f"4. 학습 설정:")
    print(f"  학습률: {config.LEARNING_RATE}")
    print(f"  배치 크기: {config.BATCH_SIZE}")
    print(f"  컨볼루션 필터: {config.CONVOLUTION_FILTERS}")
    print(f"  Residual 블록: {config.AMOUNT_OF_RESIDUAL_BLOCKS}")


def test_move_encoding():
    """액션 인코딩 테스트"""
    print("\n=== 액션 인코딩 테스트 ===")

    board = Board()

    # 링 배치 액션 테스트
    placement_moves = list(board.generate_placement_moves())
    if placement_moves:
        move = placement_moves[0]
        action = yinsh.encode_move(move, board)
        print(f"1. 링 배치 액션:")
        print(f"  이동: {move.uci()}")
        print(f"  액션 인덱스: {action}")
        print(f"  예상 범위: 0-84")

    # 링 이동 액션 테스트 (링 배치 후)
    for _ in range(10):  # 링 배치
        legal_moves = list(board.legal_moves)
        if legal_moves:
            board.push(legal_moves[0])

    if board.phase == GamePhase.MAIN:
        ring_moves = list(board.generate_ring_moves())
        if ring_moves:
            move = ring_moves[0]
            action = yinsh.encode_move(move, board)
            print(f"2. 링 이동 액션:")
            print(f"  이동: {move.uci()}")
            print(f"  액션 인덱스: {action}")
            print(f"  예상 범위: 85-7224")


if __name__ == "__main__":
    print("🎮 YINSH AlphaZero (code_old) 테스트 시작")
    print("=" * 60)

    try:
        test_config()
        test_basic_yinsh_game()
        test_yinsh_env()
        test_move_encoding()

        print("\n✅ 모든 테스트 완료!")
        print("\n📋 실행 가능한 명령어:")
        print("  python test_yinsh_old.py          # 전체 테스트")
        print("  python -c 'import yinsh; print(yinsh.Board())'  # 간단한 보드 출력")

    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()
