#!/usr/bin/env python3
"""
Simple YINSH game runner (code_old version).
"""

import sys
import os
import random

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from yinsh import Board, Move, WHITE, BLACK, GamePhase


def play_random_game():
    """랜덤하게 YINSH 게임 진행"""
    print("🎮 YINSH 랜덤 게임 시작")
    print("=" * 50)

    board = Board()
    move_count = 0

    while not board.is_game_over() and move_count < 100:
        # 현재 상태 출력
        print(f"\n--- 수 {move_count + 1} ---")
        print(f"턴: {'WHITE' if board.turn else 'BLACK'}")
        print(f"게임 단계: {board.phase.value}")
        print(
            f"WHITE 링: {board.rings_placed[WHITE]}, BLACK 링: {board.rings_placed[BLACK]}"
        )
        print(
            f"WHITE 제거: {board.rings_removed[WHITE]}, BLACK 제거: {board.rings_removed[BLACK]}"
        )

        # 가능한 수 확인
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            print("더 이상 가능한 수가 없습니다!")
            break

        # 랜덤하게 수 선택
        move = random.choice(legal_moves)
        print(f"선택된 수: {move.uci()}")

        # 수 실행
        board.push(move)
        move_count += 1

        # 간단한 보드 출력 (처음 몇 수만)
        if move_count <= 5:
            print("보드 상태:")
            print(board)

    # 게임 결과
    print(f"\n🎯 게임 종료!")
    print(f"총 수: {move_count}")
    print(f"결과: {board.result()}")
    print(f"WHITE 제거된 링: {board.rings_removed[WHITE]}")
    print(f"BLACK 제거된 링: {board.rings_removed[BLACK]}")


def play_interactive_game():
    """사람과 상호작용하는 YINSH 게임"""
    print("🎮 YINSH 인터랙티브 게임")
    print("=" * 50)
    print("명령어:")
    print("  'board' - 보드 상태 출력")
    print("  'moves' - 가능한 수 출력")
    print("  'uci' - UCI 표기법으로 수 입력 (예: @f6, f6g7)")
    print("  'quit' - 게임 종료")
    print()

    board = Board()

    while not board.is_game_over():
        # 현재 상태
        print(f"\n--- 턴: {'WHITE' if board.turn else 'BLACK'} ---")
        print(f"게임 단계: {board.phase.value}")
        print(
            f"WHITE 링: {board.rings_placed[WHITE]}, BLACK 링: {board.rings_placed[BLACK]}"
        )
        print(
            f"WHITE 제거: {board.rings_removed[WHITE]}, BLACK 제거: {board.rings_removed[BLACK]}"
        )

        # 사용자 입력
        command = input("\n명령어 입력: ").strip().lower()

        if command == "quit":
            print("게임을 종료합니다.")
            break
        elif command == "board":
            print(board)
        elif command == "moves":
            legal_moves = list(board.legal_moves)
            print(f"가능한 수 ({len(legal_moves)}개):")
            for i, move in enumerate(legal_moves[:10]):  # 처음 10개만 출력
                print(f"  {i+1}. {move.uci()}")
            if len(legal_moves) > 10:
                print(f"  ... (총 {len(legal_moves)}개)")
        elif command.startswith("uci "):
            uci = command[4:].strip()
            try:
                move = Move.from_uci(uci)
                if board.is_legal(move):
                    board.push(move)
                    print(f"수 실행: {move.uci()}")
                else:
                    print(f"잘못된 수입니다: {uci}")
            except Exception as e:
                print(f"수 파싱 오류: {e}")
        else:
            print(
                "알 수 없는 명령어입니다. 'board', 'moves', 'uci <수>', 'quit' 중 하나를 입력하세요."
            )

    # 게임 결과
    print(f"\n🎯 게임 종료!")
    print(f"결과: {board.result()}")


def main():
    """메인 함수"""
    print("🎮 YINSH 게임 실행기 (code_old)")
    print("=" * 50)
    print("1. 랜덤 게임")
    print("2. 인터랙티브 게임")
    print("3. 보드 상태만 출력")

    choice = input("\n선택 (1-3): ").strip()

    if choice == "1":
        play_random_game()
    elif choice == "2":
        play_interactive_game()
    elif choice == "3":
        board = Board()
        print(board)
    else:
        print("잘못된 선택입니다.")


if __name__ == "__main__":
    main()
