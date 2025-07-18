#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH Board Display Utilities
============================

YinshEnv의 보드 상태를 시각적으로 출력하는 유틸리티 함수들
기존 로직에 영향을 주지 않는 독립적인 모듈
"""

from typing import Optional, List, Tuple
from .env import YinshEnv, Color, GamePhase


def display_board(env: YinshEnv, title: Optional[str] = None, 
                  show_coordinates: bool = False) -> None:
    """
    YinshEnv의 현재 보드 상태를 콘솔에 출력
    
    Args:
        env: YinshEnv 인스턴스
        title: 출력할 제목 (옵션)
        show_coordinates: 좌표 표시 여부
    
    기호:
        흰색 마커: m    검은색 마커: M
        흰색 링: r      검은색 링: R  
        이동 불가능: -   이동 가능: .
    """
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    
    # 게임 상태 정보 출력
    print(f"🎮 게임 단계: {env.phase.name}")
    print(f"🎯 현재 플레이어: {env.current_player.name}")
    print(f"⚪ 흰색 링: 배치됨 {env.rings_placed[Color.WHITE]}/5, 제거됨 {env.rings_removed[Color.WHITE]}/3")
    print(f"⚫ 검은색 링: 배치됨 {env.rings_placed[Color.BLACK]}/5, 제거됨 {env.rings_removed[Color.BLACK]}/3")
    print(f"🎪 마커 풀: {env.markers_in_pool}개 남음")
    
    if env.phase == GamePhase.LINE_REMOVAL and env.pending_line_removals:
        print(f"🔴 라인 제거 대기 중: {len(env.pending_line_removals[0])}개 연속 마커")
    
    print()
    
    # 보드 문자 배열 생성
    board_display = []
    for x in range(env.board_size):
        row = []
        for y in range(env.board_size):
            pos = (x, y)
            char = get_position_char(env, pos)
            row.append(char)
        board_display.append(row)
    
    # 보드 출력
    if show_coordinates:
        # 상단 좌표 (Y축)
        print("    ", end="")
        for y in range(env.board_size):
            print(f"{y:2}", end="")
        print()
    
    for x in range(env.board_size):
        if show_coordinates:
            print(f"{x:2}: ", end="")  # 좌측 좌표 (X축)
        else:
            print("  ", end="")
        
        for y in range(env.board_size):
            char = board_display[x][y]
            print(f"{char} ", end="")
        print()
    
    print()
    
    # 범례 출력
    print("📖 범례:")
    print("  r = 흰색 링    R = 검은색 링")
    print("  m = 흰색 마커  M = 검은색 마커") 
    print("  . = 빈 공간    - = 무효 위치")


def get_position_char(env: YinshEnv, pos: Tuple[int, int]) -> str:
    """
    특정 위치의 표시 문자를 반환
    
    Args:
        env: YinshEnv 인스턴스
        pos: 보드 위치 (x, y)
        
    Returns:
        해당 위치를 나타내는 문자
    """
    # 1. 유효하지 않은 위치
    if not env.is_valid_position(pos):
        return '-'
    
    # 2. 링 확인 (링이 마커보다 우선)
    if pos in env.ring_positions[Color.WHITE]:
        return 'r'
    elif pos in env.ring_positions[Color.BLACK]:
        return 'R'
    
    # 3. 마커 확인
    elif pos in env.marker_positions[Color.WHITE]:
        return 'm'
    elif pos in env.marker_positions[Color.BLACK]:
        return 'M'
    
    # 4. 빈 공간
    else:
        return '.'


def display_game_summary(env: YinshEnv) -> None:
    """게임 요약 정보 출력"""
    print(f"\n📊 게임 요약:")
    print(f"   턴 수: {env.move_count}")
    print(f"   게임 단계: {env.phase.name}")
    
    if env.is_game_over():
        winner = env.get_winner()
        if winner:
            print(f"   🏆 승자: {winner.name}")
        else:
            print(f"   🤝 무승부")
    
    # 각 플레이어 통계
    for color in [Color.WHITE, Color.BLACK]:
        symbol = "⚪" if color == Color.WHITE else "⚫"
        print(f"   {symbol} {color.name}:")
        print(f"      링: {len(env.ring_positions[color])}개 (제거됨 {env.rings_removed[color]}개)")
        print(f"      마커: {len(env.marker_positions[color])}개")


def display_valid_actions_summary(env: YinshEnv, max_actions: int = 10) -> None:
    """유효한 액션들의 요약 출력"""
    valid_actions = env.get_valid_actions()
    
    print(f"\n🎯 유효한 액션: {len(valid_actions)}개")
    
    if len(valid_actions) == 0:
        print("   ❌ 유효한 액션이 없습니다!")
        return
    
    # 액션 타입별 개수 계산
    action_counts = {}
    for action in valid_actions:
        action_type = action.action_type
        action_counts[action_type] = action_counts.get(action_type, 0) + 1
    
    for action_type, count in action_counts.items():
        print(f"   📌 {action_type}: {count}개")
    
    # 처음 몇 개 액션 상세 표시
    if len(valid_actions) > 0:
        print(f"\n   처음 {min(max_actions, len(valid_actions))}개 액션:")
        for i, action in enumerate(valid_actions[:max_actions]):
            print(f"      {i+1}. {action}")


def display_action_details(action, action_info: dict = None) -> None:
    """선택된 액션의 상세 정보 출력"""
    print(f"\n🎯 선택된 액션: {action}")
    
    if action_info:
        print(f"   방법: {action_info.get('method', 'Unknown')}")
        if 'temperature' in action_info:
            print(f"   온도: {action_info['temperature']}")
        if 'mcts_stats' in action_info:
            mcts_stats = action_info['mcts_stats']
            if 'simulations_run' in mcts_stats:
                print(f"   MCTS 시뮬레이션: {mcts_stats['simulations_run']}회")
            if 'max_visits' in mcts_stats:
                print(f"   최대 방문 수: {mcts_stats['max_visits']}")


def display_compact_board(env: YinshEnv) -> str:
    """
    컴팩트한 한 줄 보드 표현 반환
    로그에 간단히 기록할 때 사용
    """
    white_rings = len(env.ring_positions[Color.WHITE])
    black_rings = len(env.ring_positions[Color.BLACK])
    white_markers = len(env.marker_positions[Color.WHITE])
    black_markers = len(env.marker_positions[Color.BLACK])
    
    return (f"[{env.phase.name[:4]} {env.current_player.name[0]} | "
            f"R:{white_rings}/{black_rings} M:{white_markers}/{black_markers} "
            f"Removed:{env.rings_removed[Color.WHITE]}/{env.rings_removed[Color.BLACK]}]")


def save_board_to_text(env: YinshEnv, filename: str, title: Optional[str] = None) -> None:
    """보드 상태를 텍스트 파일로 저장"""
    import io
    import sys
    from contextlib import redirect_stdout
    
    # stdout을 문자열로 캡처
    captured_output = io.StringIO()
    
    with redirect_stdout(captured_output):
        display_board(env, title, show_coordinates=True)
        display_game_summary(env)
        display_valid_actions_summary(env)
    
    # 파일로 저장
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(captured_output.getvalue())
        print(f"💾 보드 상태 저장: {filename}")
    except Exception as e:
        print(f"❌ 파일 저장 실패: {e}")


# 테스트 함수
def test_board_display():
    """보드 출력 기능 테스트"""
    print("🧪 보드 출력 기능 테스트")
    
    # 환경 생성 및 초기화
    env = YinshEnv()
    
    # 초기 상태 출력
    display_board(env, "초기 게임 상태", show_coordinates=True)
    
    # 몇 개 링 배치해보기
    if env.get_valid_actions():
        action = env.get_valid_actions()[0]
        env.step(action)
        display_board(env, f"첫 번째 액션 후: {action}")
    
    print("✅ 테스트 완료")


if __name__ == "__main__":
    test_board_display() 