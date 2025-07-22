#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH Human vs AlphaZero Script (Enhanced)
=========================================

사용법:
$ python code/human_vs_ai.py --model best_model.pt --human-color white

- `--model`        : 학습 완료된 모델(.pt) 경로
- `--human-color`  : white | black  (사람이 먼저 두면 white)
- `--mcts-sims`    : AI 의 MCTS 시뮬레이션 수 (기본 800)
- `--show-coord`   : 좌표계 안내 표 출력
"""

import os
import sys
import argparse
import time
import torch
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yinsh import YinshEnv, YinshAgent, Color, display_board, display_action_details
from yinsh.mapper import YinshActionMapper

def print_coord_help():
    """
    간단 좌표계 안내:
    - 보드 교차점을   (q, r)  큐브 좌표계의  q(가로) / r(세로) 로 표기한다고 가정
    - Ring 배치 단계 :  place q r
    - 이동 단계     :  move q_from r_from q_to r_to
    예)  move 0 0  0 3   (링 (0,0) → (0,3) 로 직선 이동)
    """
    print("\n🗺️  좌표 입력 예시")
    print("   링 배치 :  place  0  2")
    print("   링 이동 :  move  1  0   3  0")
    print("   help 입력으로 언제든 재확인\n")

def parse_user_input(user_str: str, action_mapper: YinshActionMapper):
    """사용자 문자열을 YinshEnv 액션 객체로 변환한다."""
    from yinsh.env import YinshAction  # 직접 생성 위해 import
    tokens = user_str.strip().lower().split()
    if not tokens:
        raise ValueError("빈 입력")

    cmd = tokens[0]
    if cmd == "place" and len(tokens) == 3:
        q, r = int(tokens[1]), int(tokens[2])
        return YinshAction("PLACE_RING", to_pos=(q, r))

    if cmd == "move" and len(tokens) == 5:
        q1, r1, q2, r2 = map(int, tokens[1:5])
        return YinshAction("MOVE_RING", from_pos=(q1, r1), to_pos=(q2, r2))

    raise ValueError("지원하지 않는 입력 형식")

def main():
    parser = argparse.ArgumentParser(description="YINSH AlphaZero와 대국 (Human vs AI)")
    parser.add_argument("--model", type=str, required=True, help="학습된 모델 경로 (.pt)")
    parser.add_argument("--human-color", type=str, choices=["white", "black"], default="white", help="사람 색상 (white|black)")
    parser.add_argument("--mcts-sims", type=int, default=800, help="AI의 MCTS 시뮬레이션 수")
    parser.add_argument("--show-coord", action="store_true", help="좌표계 안내 출력")
    args = parser.parse_args()

    print("🎮 YINSH Human vs AlphaZero (강화 버전)")
    print(f"📋 설정:")
    print(f"   모델: {args.model}")
    print(f"   사람 색상: {args.human_color}")
    print(f"   MCTS 시뮬레이션: {args.mcts_sims}")
    print(f"   좌표 안내: {'출력' if args.show_coord else '미출력'}")
    print(f"   디바이스: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 50)

    # 모델 경로 검증
    if not Path(args.model).exists():
        print(f"❌ 모델 파일을 찾을 수 없습니다: {args.model}")
        return

    # 디바이스
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 에이전트 생성 (selfplay.py/evaluate.py와 동일하게)
    try:
        print("🤖 AI 에이전트 초기화 중...")
        ai_agent = YinshAgent(
            model_path=args.model,
            use_mcts=True,
            device=device
        )
        print("   ✅ AI 에이전트 초기화 완료")
        if hasattr(ai_agent, 'mcts_agent') and ai_agent.mcts_agent:
            ai_agent.mcts_agent.num_simulations = args.mcts_sims
    except Exception as e:
        print(f"❌ AI 에이전트 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 환경 및 보조 객체
    try:
        env = YinshEnv()
        mapper = YinshActionMapper()
    except Exception as e:
        print(f"❌ 환경 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    human_color = Color.WHITE if args.human_color == "white" else Color.BLACK
    ai_color = Color.BLACK if human_color == Color.WHITE else Color.WHITE
    print(f"\n당신: {human_color.name}")
    print(f"AI  : {ai_color.name}")
    if args.show_coord:
        print_coord_help()

    # 게임 루프
    turn = 0
    while not env.is_game_over():
        turn += 1
        current_color = env.current_player
        print("\n" + "-" * 50)
        display_board(env, f"Turn {turn} — {current_color.name} 차례")

        if current_color == human_color:
            # ─────────── 인간 플레이어 ───────────
            while True:
                try:
                    user_in = input("💡 입력 (help, resign): ").strip()
                    if user_in == "help":
                        print_coord_help()
                        continue
                    if user_in == "resign":
                        env.force_resign(current_color)
                        print(f"\n🏳️  {current_color.name} 플레이어가 기권했습니다!")
                        break

                    action = parse_user_input(user_in, mapper)
                    env.step(action)
                    display_action_details(action, {"method": "human"})
                    break
                except Exception as e:
                    print(f"❌ 입력 오류: {e}  (help 로 안내 확인)")
                    continue
        else:
            # ─────────── AI 플레이어 ───────────
            try:
                t0 = time.time()
                action, info = ai_agent.select_action(env)
                env.step(action)
                t_used = time.time() - t0
                print(f"🤖 AI 액션: {action}  ({t_used:.2f}s, sims={args.mcts_sims})")
                if info and "mcts_stats" in info:
                    visits = max(info["mcts_stats"].get("visit_counts", {}).values() or [0])
                    print(f"    최고 방문 횟수: {visits}")
            except Exception as e:
                print(f"❌ AI 액션 오류: {e}")
                import traceback
                traceback.print_exc()
                print("AI가 수를 두지 못했습니다. 게임을 종료합니다.")
                break

    # 게임 종료 처리
    try:
        winner = env.get_winner()
        if winner is None:
            print("\n🏁 무승부!")
        elif winner == human_color:
            print("\n🏆 당신이 승리했습니다!")
        else:
            print("\n🤖 AI 승리... 더 연습해보세요!")
    except Exception as e:
        print(f"❌ 게임 결과 확인 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
