#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH Human vs AlphaZero Script (Rewritten for New Rules)
=======================================================

새 게임 규칙:
- 링 배치는 랜덤으로 자동 완료
- MOVE_RING 액션만 사용
- 5개 연속 마커 완성 시 즉시 게임 종료 및 승부 결정
- 마커/링 제거 과정 없음

사용법:
$ python code/human_vs_ai.py --model best_model.pt --human-color white

옵션:
- `--model`        : 학습 완료된 모델(.pt) 경로 (필수)
- `--human-color`  : white | black (사람 플레이어 색상)
- `--mcts-sims`    : AI의 MCTS 시뮬레이션 수 (기본 800)
- `--show-coord`   : 좌표계 안내 출력
- `--debug`        : 디버그 정보 출력
"""

import os
import sys
import argparse
import time
import torch
from pathlib import Path
from typing import Optional, Tuple

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yinsh import YinshEnv, YinshAgent, Color, display_board, display_compact_board
from yinsh.env import YinshAction

def print_welcome():
    """게임 시작 환영 메시지"""
    print("🎮 YINSH Human vs AlphaZero")
    print("=" * 50)
    print("📋 새로운 게임 규칙:")
    print("   • 링 배치: 자동으로 랜덤 배치 (10개)")
    print("   • 액션: 링 이동(MOVE_RING)만 가능")
    print("   • 승리: 5개 연속 마커 완성 시 즉시 승리")
    print("   • 제거: 마커/링 제거 과정 없음")
    print("=" * 50)

def print_coordinate_help():
    """좌표계 안내"""
    print("\n🗺️  좌표계 안내:")
    print("   • Hex 좌표계 사용: (q, r) 형식")
    print("   • q: 가로 축 (-5 ~ 5)")
    print("   • r: 세로 축 (-5 ~ 5)")
    print("   • 보드 중앙이 (0, 0)")
    print("\n💡 입력 형식:")
    print("   • 링 이동: move <q1> <r1> <q2> <r2>")
    print("   • 예시:")
    print("     move 0 0 2 1    (중앙에서 우상단으로)")
    print("     move -1 2 1 0   (좌하단에서 우측으로)")
    print("\n🎮 기타 명령어:")
    print("   • help     - 이 도움말 표시")
    print("   • status   - 상세 게임 상태")
    print("   • debug    - 디버그 정보")
    print("   • resign   - 기권")
    print()

def print_game_status(env: YinshEnv, turn: int):
    """게임 상태 정보 출력"""
    print(f"\n🎯 Turn {turn} | {env.current_player.name} 차례")
    print(f"⚪ 흰색: 링 {len(env.ring_positions[Color.WHITE])}개, 마커 {len(env.marker_positions[Color.WHITE])}개")
    print(f"⚫ 검은색: 링 {len(env.ring_positions[Color.BLACK])}개, 마커 {len(env.marker_positions[Color.BLACK])}개")
    print(f"🎪 마커 풀: {env.markers_in_pool}개 남음")
    print(f"🎮 게임 상태: {'진행 중' if not env.done else '종료'}")

def parse_human_input(user_input: str) -> Tuple[str, Optional[YinshAction]]:
    """
    사용자 입력을 파싱하여 명령어와 액션 반환
    
    Returns:
        (command, action) tuple
        command: "move", "help", "status", "resign", "debug", "invalid"
        action: YinshAction 객체 (move 명령어인 경우)
    """
    tokens = user_input.strip().lower().split()
    if not tokens:
        return "invalid", None
    
    command = tokens[0]
    
    if command in ["help", "status", "resign", "debug"]:
        return command, None
    
    if command == "move" and len(tokens) == 5:
        try:
            q1, r1, q2, r2 = map(int, tokens[1:5])
            action = YinshAction(from_pos=(q1, r1), to_pos=(q2, r2))
            return "move", action
        except ValueError:
            return "invalid", None
    
    return "invalid", None

def validate_human_action(env: YinshEnv, action: YinshAction) -> Tuple[bool, str]:
    """
    사용자 액션 유효성 검사
    
    Returns:
        (is_valid, error_message)
    """
    try:
        # 1. 시작 위치에 현재 플레이어의 링이 있는지 확인
        if action.from_pos not in env.ring_positions[env.current_player]:
            return False, f"시작 위치 {action.from_pos}에 {env.current_player.name} 링이 없습니다."
        
        # 2. 목적지가 유효한 위치인지 확인
        if not env.is_valid_position(action.to_pos):
            return False, f"목적지 {action.to_pos}는 유효하지 않은 위치입니다."
        
        # 3. 링 이동이 가능한지 확인
        if not env.is_valid_ring_move(action.from_pos, action.to_pos):
            return False, f"링을 {action.from_pos}에서 {action.to_pos}로 이동할 수 없습니다."
        
        return True, ""
        
    except Exception as e:
        return False, f"액션 검증 중 오류: {e}"

def execute_human_turn(env: YinshEnv) -> bool:
    """
    사용자 턴 실행
    
    Returns:
        게임을 계속할지 여부 (False면 resign)
    """
    while True:
        try:
            user_input = input("💡 명령어 입력: ").strip()
            command, action = parse_human_input(user_input)
            
            if command == "help":
                print_coordinate_help()
                continue
            
            elif command == "status":
                print(f"\n📊 상세 상태:")
                print(f"   현재 플레이어: {env.current_player.name}")
                print(f"   게임 종료: {env.done}")
                if env.done and env.winner:
                    print(f"   승자: {env.winner.name}")
                print(f"   유효한 액션 수: {len(env.get_valid_actions())}")
                continue
            
            elif command == "resign":
                print(f"\n🏳️ {env.current_player.name} 플레이어가 기권했습니다!")
                return False
            
            elif command == "debug":
                print(f"\n🔍 디버그 정보:")
                print(f"   링 위치 (WHITE): {list(env.ring_positions[Color.WHITE])}")
                print(f"   링 위치 (BLACK): {list(env.ring_positions[Color.BLACK])}")
                print(f"   마커 위치 (WHITE): {list(env.marker_positions[Color.WHITE])}")
                print(f"   마커 위치 (BLACK): {list(env.marker_positions[Color.BLACK])}")
                continue
            
            elif command == "move" and action:
                # 액션 유효성 검사
                is_valid, error_msg = validate_human_action(env, action)
                if not is_valid:
                    print(f"❌ {error_msg}")
                    continue
                
                # 액션 실행
                success = env.step(action)
                if success:
                    print(f"✅ 링 이동: {action.from_pos} → {action.to_pos}")
                    return True
                else:
                    print("❌ 액션 실행에 실패했습니다. 다시 시도해주세요.")
                    continue
            
            else:
                print("❌ 잘못된 입력입니다. 'help'로 사용법을 확인하세요.")
                continue
                
        except KeyboardInterrupt:
            print(f"\n🛑 게임이 중단되었습니다.")
            return False
        except Exception as e:
            print(f"❌ 입력 처리 중 오류: {e}")
            continue

def execute_ai_turn(env: YinshEnv, ai_agent: YinshAgent, mcts_sims: int, debug: bool = False) -> bool:
    """
    AI 턴 실행
    
    Returns:
        게임을 계속할지 여부
    """
    try:
        print(f"🤖 AI가 수를 계산 중... (MCTS {mcts_sims} simulations)")
        start_time = time.time()
        
        # AI 액션 선택
        action, action_info = ai_agent.select_action(env)
        thinking_time = time.time() - start_time
        
        if action is None:
            print("❌ AI가 유효한 액션을 찾지 못했습니다.")
            return False
        
        # 액션 실행
        success = env.step(action)
        if not success:
            print("❌ AI 액션 실행에 실패했습니다.")
            return False
        
        # 결과 출력
        print(f"🤖 AI 액션: {action.from_pos} → {action.to_pos} ({thinking_time:.2f}초)")
        
        if debug and action_info:
            if "mcts_stats" in action_info:
                stats = action_info["mcts_stats"]
                if "visit_counts" in stats:
                    max_visits = max(stats["visit_counts"].values()) if stats["visit_counts"] else 0
                    print(f"   🌳 MCTS 최대 방문 횟수: {max_visits}")
                if "root_value" in stats:
                    print(f"   📊 루트 노드 가치: {stats['root_value']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ AI 턴 실행 중 오류: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        return False

def print_game_result(env: YinshEnv, human_color: Color):
    """게임 결과 출력"""
    print("\n" + "=" * 60)
    print("🎮 게임 종료!")
    print("=" * 60)
    
    try:
        winner = env.get_winner()
        
        if winner is None:
            print("🏁 무승부!")
            if env.markers_in_pool <= 0:
                print("   📦 마커가 모두 소진되었습니다.")
        elif winner == human_color:
            print("🏆 축하합니다! 당신이 승리했습니다!")
            print("   🎯 5개 연속 마커를 완성했습니다!")
        else:
            print("🤖 AI 승리!")
            print("   🎯 AI가 5개 연속 마커를 완성했습니다!")
        
        # 최종 통계
        print(f"\n📊 최종 게임 통계:")
        print(f"   ⚪ 흰색 - 링: {len(env.ring_positions[Color.WHITE])}개, 마커: {len(env.marker_positions[Color.WHITE])}개")
        print(f"   ⚫ 검은색 - 링: {len(env.ring_positions[Color.BLACK])}개, 마커: {len(env.marker_positions[Color.BLACK])}개")
        print(f"   🎪 남은 마커: {env.markers_in_pool}개")
        
    except Exception as e:
        print(f"❌ 게임 결과 확인 중 오류: {e}")

def main():
    parser = argparse.ArgumentParser(description="YINSH Human vs AlphaZero (새 규칙)")
    parser.add_argument("--model", type=str, required=True, help="학습된 모델 경로 (.pt)")
    parser.add_argument("--human-color", type=str, choices=["white", "black"], default="white", help="사람 플레이어 색상")
    parser.add_argument("--mcts-sims", type=int, default=800, help="AI의 MCTS 시뮬레이션 수")
    parser.add_argument("--show-coord", action="store_true", help="좌표계 안내 출력")
    parser.add_argument("--debug", action="store_true", help="디버그 정보 출력")
    args = parser.parse_args()
    
    # 환영 메시지
    print_welcome()
    
    # 설정 정보 출력
    print(f"📋 게임 설정:")
    print(f"   🤖 AI 모델: {args.model}")
    print(f"   👤 사람 색상: {args.human_color.upper()}")
    print(f"   🌳 MCTS 시뮬레이션: {args.mcts_sims}")
    print(f"   💻 디바이스: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"   🔍 디버그: {'활성화' if args.debug else '비활성화'}")
    
    # 모델 경로 검증
    if not Path(args.model).exists():
        print(f"\n❌ 모델 파일을 찾을 수 없습니다: {args.model}")
        print("   모델 경로를 확인해주세요.")
        return 1
    
    # AI 에이전트 초기화
    try:
        print(f"\n🤖 AI 에이전트 초기화 중...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ai_agent = YinshAgent(
            model_path=args.model,
            use_mcts=True,
            device=device
        )
        
        # MCTS 시뮬레이션 수 설정
        if hasattr(ai_agent, 'mcts_agent') and ai_agent.mcts_agent:
            ai_agent.mcts_agent.num_simulations = args.mcts_sims
        
        print(f"   ✅ AI 에이전트 초기화 완료")
        print(f"   🧠 모델: {args.model}")
        print(f"   🌳 MCTS: {args.mcts_sims} simulations")
        
    except Exception as e:
        print(f"\n❌ AI 에이전트 초기화 실패: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    # 게임 환경 초기화
    try:
        print(f"\n🎮 게임 환경 초기화 중...")
        env = YinshEnv()
        print(f"   ✅ 게임 환경 초기화 완료")
        print(f"   🎯 링 배치 완료: 각 색상 5개씩 랜덤 배치")
        
    except Exception as e:
        print(f"\n❌ 게임 환경 초기화 실패: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    
    # 플레이어 색상 설정
    human_color = Color.WHITE if args.human_color == "white" else Color.BLACK
    ai_color = Color.BLACK if human_color == Color.WHITE else Color.WHITE
    
    print(f"\n👥 플레이어 설정:")
    print(f"   👤 사람: {human_color.name}")
    print(f"   🤖 AI: {ai_color.name}")
    
    if args.show_coord:
        print_coordinate_help()
    
    # 게임 루프
    turn = 0
    print(f"\n🚀 게임 시작!")
    
    while not env.is_game_over():
        turn += 1
        current_player = env.current_player
        
        # 게임 상태 출력
        print("\n" + "-" * 60)
        print_game_status(env, turn)
        
        # 보드 출력
        display_board(env, f"Turn {turn} - {current_player.name} 차례")
        
        # 턴 실행
        if current_player == human_color:
            # 사람 턴
            print(f"\n👤 {human_color.name} 플레이어 차례")
            if not execute_human_turn(env):
                # 기권 또는 중단
                break
        else:
            # AI 턴
            print(f"\n🤖 {ai_color.name} AI 차례")
            if not execute_ai_turn(env, ai_agent, args.mcts_sims, args.debug):
                # AI 오류
                break
        
        # 게임이 끝났는지 확인
        if env.is_game_over():
            break
    
    # 최종 보드 출력
    print("\n" + "-" * 60)
    display_board(env, "최종 보드 상태")
    
    # 게임 결과 출력
    print_game_result(env, human_color)
    
    print(f"\n🎮 게임을 플레이해주셔서 감사합니다!")
    return 0

if __name__ == "__main__":
    exit(main())