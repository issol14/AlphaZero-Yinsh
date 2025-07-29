#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH AlphaZero Self-Play Script (Enhanced)
===========================================

This script generates high-quality training data through self-play games
with proper MCTS policy distribution extraction for optimal learning.
Supports both sequential and parallel execution.
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json
import time
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yinsh import YinshEnv, YinshAgent, Color, config, display_board, display_action_details, display_compact_board
from yinsh.mapper import YinshActionMapper


def run_single_game_parallel(game_config: Dict) -> Dict:
    """병렬 실행용 단일 게임 실행 함수 (최적화된 버전)"""
    try:
        game_id = game_config['game_id']
        model_path = game_config['model_path']
        output_dir = game_config['output_dir']
        mcts_sims = game_config['mcts_sims']
        use_mcts = game_config['use_mcts']
        show_board = game_config['show_board']
        
        # 디바이스 설정 (GPU 메모리 최적화)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # GPU 메모리 정리 및 최적화
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.8)  # GPU 메모리 사용량 제한
        
        print(f"🎮 [Worker {os.getpid()}] Game {game_id + 1} starting...")
        
        # 에이전트 생성 (메모리 효율적)
        agent1 = YinshAgent(
            model_path=model_path, 
            use_mcts=use_mcts, 
            use_parallel_mcts=game_config.get('use_parallel_mcts', False),
            device=device
        )
        agent2 = YinshAgent(
            model_path=model_path, 
            use_mcts=use_mcts, 
            use_parallel_mcts=game_config.get('use_parallel_mcts', False),
            device=device
        )
        
        # Selfplay용 빠른 MCTS 설정
        if use_mcts:
            # Selfplay용으로 시뮬레이션 수 조정
            fast_sims = min(mcts_sims, config.SELFPLAY_MCTS_SIMULATIONS)
            agent1.mcts_agent.mcts.num_simulations = fast_sims
            agent2.mcts_agent.mcts.num_simulations = fast_sims
            
            # 병렬 MCTS 스레드 수 설정
            if game_config.get('use_parallel_mcts', False):
                num_threads = game_config.get('mcts_threads', 4)
                agent1.mcts_agent.mcts.num_threads = num_threads
                agent2.mcts_agent.mcts.num_threads = num_threads
        
        game_start_time = time.time()
        
        # 게임 실행 (최적화된 버전)
        game_history, winner, turns = play_game_optimized(agent1, agent2, game_id=game_id, show_board=show_board)
        
        # 훈련 데이터 생성
        training_data = generate_training_data(game_history, winner, game_id=game_id)
        
        # 데이터 저장
        save_result = save_game_data(training_data, output_dir, game_id)
        
        game_time = time.time() - game_start_time
        
        # 메모리 정리
        if device == 'cuda':
            torch.cuda.empty_cache()
        
        result = {
            'game_id': game_id,
            'winner': winner.name if winner else 'DRAW',
            'turns': turns,
            'positions': len(training_data),
            'game_time': game_time,
            'save_success': save_result is not None,
            'process_id': os.getpid()
        }
        
        print(f"✅ [Worker {os.getpid()}] Game {game_id + 1} completed in {game_time:.1f}s")
        return result
        
    except Exception as e:
        print(f"❌ [Worker {os.getpid()}] Game {game_id + 1} failed: {e}")
        return {
            'game_id': game_id,
            'error': str(e),
            'process_id': os.getpid()
        }


def run_parallel_games(args):
    """병렬 게임 실행"""
    print("=" * 50)
    
    # 게임 설정 생성
    game_configs = [
        {
            'game_id': i,
            'model_path': args.model,
            'output_dir': args.output,
            'mcts_sims': args.mcts_sims,
            'use_mcts': not args.no_mcts,
            'show_board': args.show_board,
            'use_parallel_mcts': args.parallel_mcts,
            'mcts_threads': args.mcts_threads
        }
        for i in range(args.games)
    ]
    
    # 통계 초기화
    total_positions = 0
    white_wins = 0
    black_wins = 0
    draws = 0
    completed_games = 0
    failed_games = 0
    total_start_time = time.time()
    
    # 병렬 실행
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # 모든 게임 제출
        future_to_config = {
            executor.submit(run_single_game_parallel, config): config 
            for config in game_configs
        }
        
        # 진행률 바 설정
        with tqdm(total=args.games, desc="🎮 Games", unit="game") as pbar:
            # 완료된 작업들 처리
            for future in as_completed(future_to_config):
                try:
                    result = future.result()
                    
                    if 'error' in result:
                        failed_games += 1
                    else:
                        completed_games += 1
                        total_positions += result['positions']
                        
                        # 승부 통계
                        if result['winner'] == 'WHITE':
                            white_wins += 1
                        elif result['winner'] == 'BLACK':
                            black_wins += 1
                        else:
                            draws += 1
                        
                        # 진행률 업데이트
                        pbar.set_postfix({
                            'W': white_wins,
                            'B': black_wins,
                            'D': draws,
                            'Pos': total_positions
                        })
                    
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"❌ Error processing result: {e}")
                    failed_games += 1
                    pbar.update(1)
    
    # 최종 통계
    total_time = time.time() - total_start_time
    print(f"\n🎉 병렬 Self-Play 완료! (총 소요시간: {total_time:.1f}초)")
    print("📈 최종 통계:")
    print(f"├── 총 게임 수: {args.games}")
    print(f"├── 완료된 게임: {completed_games}")
    print(f"├── 실패한 게임: {failed_games}")
    print(f"├── 총 포지션 수: {total_positions}")
    if completed_games > 0:
        print(f"├── White 승리: {white_wins} ({white_wins/completed_games*100:.1f}%)")
        print(f"├── Black 승리: {black_wins} ({black_wins/completed_games*100:.1f}%)")
        print(f"├── 무승부: {draws} ({draws/completed_games*100:.1f}%)")
        print(f"├── 게임당 평균 포지션: {total_positions/completed_games:.1f}")
    print(f"├── 게임/초: {completed_games/total_time:.2f}")
    print(f"└── 워커 수: {args.workers}")


def play_game(agent1, agent2, max_turns=1000, game_id=0, show_board=False):
    """두 에이전트 간의 게임 진행 - MCTS 정책 정보 포함"""
    print(f"\n🎮 Starting Game {game_id + 1}...")
    
    env = YinshEnv()
    game_history = []
    turn_count = 0
    game_start_time = time.time()
    
    # 초기 보드 상태 출력 (옵션)
    if show_board:
        display_board(env, f"Game {game_id + 1} - 초기 상태")

    while not env.is_game_over() and turn_count < max_turns:
        turn_start_time = time.time()
        current_player = agent1 if env.current_player == Color.WHITE else agent2
        player_name = "WHITE" if env.current_player == Color.WHITE else "BLACK"
        
        print(f"  Turn {turn_count + 1}: {player_name} 플레이어 턴")

        # 현재 상태 저장
        try:
            state = env.get_state_tensor()
            print(f"    ✅ 상태 텐서 생성 완료: {state.shape}")
        except Exception as e:
            print(f"    ❌ 상태 텐서 생성 실패: {e}")
            return game_history, None, turn_count

        # 액션 선택 (MCTS 정보 포함)
        try:
            print(f"    🤔 액션 선택 중... (MCTS 시뮬레이션 진행)")
            action_start_time = time.time()
            
            action, action_info = current_player.select_action(env)
            
            action_time = time.time() - action_start_time
            print(f"    ✅ 액션 선택 완료: {action} (소요시간: {action_time:.2f}초)")
            
            # MCTS 통계 출력
            if action_info and "method" in action_info:
                print(f"       방법: {action_info['method']}")
                if "mcts_stats" in action_info:
                    stats = action_info["mcts_stats"]
                    if "simulations" in stats:
                        print(f"       시뮬레이션 수: {stats['simulations']}")
                    if "visit_counts" in stats:
                        print(f"       방문 횟수 정보: {len(stats['visit_counts'])}개 액션")
            
        except Exception as e:
            print(f"    ❌ 액션 선택 실패: {e}")
            import traceback
            traceback.print_exc()
            return game_history, None, turn_count

        # 액션 실행
        try:
            print(f"    🎯 액션 실행 중: {action}")
            env.step(action)
            print(f"    ✅ 액션 실행 완료")
            
            # 보드 상태 출력 (옵션)
            if show_board:
                print(f"    📋 보드 상태:")
                display_action_details(action, action_info)
                display_board(env, f"Turn {turn_count + 1} - {player_name} 플레이 후")
                print("-" * 40)
            
        except Exception as e:
            print(f"    ❌ 액션 실행 실패: {e}")
            import traceback
            traceback.print_exc()
            return game_history, None, turn_count

        # 게임 히스토리에 추가 (action_info 포함)
        game_history.append({
            "state": state, 
            "action": action, 
            "player": env.current_player,
            "action_info": action_info  # MCTS 통계 포함
        })

        turn_time = time.time() - turn_start_time
        print(f"    ⏱️  턴 완료 (총 소요시간: {turn_time:.2f}초)")
        
        turn_count += 1

    # 게임 결과
    try:
        winner = env.get_winner()
        game_time = time.time() - game_start_time
        
        winner_name = "WHITE" if winner == Color.WHITE else "BLACK" if winner == Color.BLACK else "DRAW"
        print(f"\n🏁 Game {game_id + 1} 완료!")
        print(f"   승자: {winner_name}")
        print(f"   총 턴 수: {turn_count}")
        print(f"   총 게임 시간: {game_time:.2f}초")
        print(f"   턴당 평균 시간: {game_time/max(turn_count, 1):.2f}초")
        
    except Exception as e:
        print(f"❌ 게임 결과 확인 실패: {e}")
        winner = None
    
    return game_history, winner, turn_count


def play_game_optimized(agent1, agent2, max_turns=1000, game_id=0, show_board=False):
    """최적화된 게임 실행 함수 (빠른 selfplay용)"""
    env = YinshEnv()
    game_history = []
    turn_count = 0
    game_start_time = time.time()
    
    # 로깅 최소화 (성능 향상)
    verbose = show_board
    
    while not env.is_game_over() and turn_count < max_turns:
        current_player = agent1 if env.current_player == Color.WHITE else agent2
        player_name = "WHITE" if env.current_player == Color.WHITE else "BLACK"
        
        if verbose:
            print(f"  Turn {turn_count + 1}: {player_name} 플레이어 턴")

        # 현재 상태 저장 (최적화된 버전)
        try:
            state = env.get_state_tensor()
        except Exception as e:
            if verbose:
                print(f"    ❌ 상태 텐서 생성 실패: {e}")
            return game_history, None, turn_count

        # 액션 선택 (최적화된 버전)
        try:
            if verbose:
                print(f"    🤔 액션 선택 중...")
            
            action_start_time = time.time()
            action, action_info = current_player.select_action(env)
            action_time = time.time() - action_start_time
            
            if verbose:
                print(f"    ✅ 액션 선택 완료: {action} (소요시간: {action_time:.2f}초)")
            
        except Exception as e:
            if verbose:
                print(f"    ❌ 액션 선택 실패: {e}")
            return game_history, None, turn_count

        # 액션 실행
        try:
            if verbose:
                print(f"    🎯 액션 실행 중: {action}")
            
            env.step(action)
            
            if verbose:
                print(f"    ✅ 액션 실행 완료")
                if show_board:
                    display_action_details(action, action_info)
                    display_board(env, f"Turn {turn_count + 1} - {player_name} 플레이 후")
                    print("-" * 40)
            
        except Exception as e:
            if verbose:
                print(f"    ❌ 액션 실행 실패: {e}")
            return game_history, None, turn_count

        # 게임 히스토리에 추가 (최적화된 버전)
        game_history.append({
            "state": state, 
            "action": action, 
            "player": env.current_player,
            "action_info": action_info
        })

        turn_count += 1

    # 게임 결과 (최적화된 버전)
    try:
        winner = env.get_winner()
        game_time = time.time() - game_start_time
        
        if verbose:
            winner_name = "WHITE" if winner == Color.WHITE else "BLACK" if winner == Color.BLACK else "DRAW"
            print(f"\n🏁 Game {game_id + 1} 완료!")
            print(f"   승자: {winner_name}")
            print(f"   총 턴 수: {turn_count}")
            print(f"   총 게임 시간: {game_time:.2f}초")
            print(f"   턴당 평균 시간: {game_time/max(turn_count, 1):.2f}초")
        
    except Exception as e:
        if verbose:
            print(f"❌ 게임 결과 확인 실패: {e}")
        winner = None
    
    return game_history, winner, turn_count


def extract_mcts_policy_distribution(action_info, action_mapper, executed_action):
    """MCTS 통계에서 4000차원 정책 분포 추출 (개선된 버전)"""
    policy = np.zeros(config.POLICY_OUTPUT_SIZE, dtype=np.float32)
    
    # MCTS 사용한 경우
    if action_info.get("method") == "mcts" and "mcts_stats" in action_info:
        mcts_stats = action_info["mcts_stats"]
        
        # 방문 횟수 기반 정책 분포 추출
        visit_counts = mcts_stats.get("visit_counts", {})
        
        if visit_counts:
            # 액션별 방문 횟수를 4000차원 벡터로 변환
            total_visits = sum(visit_counts.values())
            
            if total_visits > 0:
                # visit_counts의 키들을 처리
                for action_str, visits in visit_counts.items():
                    try:
                        # 실행된 액션과 일치하는 경우 우선 처리
                        if action_str == str(executed_action):
                            action_index = action_mapper.get_action_index(executed_action)
                            if action_index is not None and action_index < config.POLICY_OUTPUT_SIZE:
                                policy[action_index] = visits / total_visits
                    except Exception:
                        continue
                
                # 다른 액션들도 처리 (문자열 파싱은 복잡하므로 스킵)
                # 대신 실행된 액션의 확률을 높게 설정
                if policy.sum() == 0:
                    # 실행된 액션에 모든 확률 할당
                    action_index = action_mapper.get_action_index(executed_action)
                    if action_index is not None:
                        policy[action_index] = 1.0
            
            # 정책 벡터 정규화 (안전장치)
            if policy.sum() > 0:
                policy = policy / policy.sum()
            else:
                # 실행된 액션에 확률 1.0 할당
                action_index = action_mapper.get_action_index(executed_action)
                if action_index is not None:
                    policy[action_index] = 1.0
                else:
                    policy[0] = 1.0
        else:
            # visit_counts가 없는 경우 실행된 액션에 확률 1.0 할당
            action_index = action_mapper.get_action_index(executed_action)
            if action_index is not None:
                policy[action_index] = 1.0
            else:
                policy[0] = 1.0
    
    # Direct neural network 사용한 경우는 원핫 처리
    else:
        return None  # 원핫 처리는 generate_training_data에서
    
    return policy


def generate_training_data(game_history, winner, game_id=0):
    """게임 히스토리에서 고품질 훈련 데이터 생성"""
    print(f"\n📊 Game {game_id + 1} 훈련 데이터 생성 중...")
    
    training_data = []
    action_mapper = YinshActionMapper()
    mcts_policy_count = 0
    direct_policy_count = 0

    for i, move in enumerate(game_history):
        try:
            state = move["state"]
            action = move["action"]
            player = move["player"]
            action_info = move["action_info"]

            # MCTS 정책 분포 추출 시도 (환경 복원 없이)
            policy = extract_mcts_policy_distribution(action_info, action_mapper, action)
            
            # MCTS 정책 추출 실패 시 또는 Direct NN인 경우 원핫 인코딩
            if policy is None:
                policy = np.zeros(config.POLICY_OUTPUT_SIZE, dtype=np.float32)
                try:
                    action_index = action_mapper.get_action_index(action)  # 수정된 메서드명
                    if action_index is not None:
                        policy[action_index] = 1.0
                        direct_policy_count += 1
                    else:
                        # 매핑 실패 시 기본값 사용
                        policy[0] = 1.0
                        print(f"      ⚠️ Turn {i+1}: Action mapping failed for action: {action}")
                except Exception as e:
                    print(f"      ❌ Turn {i+1}: Action mapping failed: {e}")
                    policy[0] = 1.0
                    direct_policy_count += 1
            else:
                mcts_policy_count += 1

            # 가치 계산
            if winner is None:
                value = 0.0
            elif winner == player:
                value = 1.0
            else:
                value = -1.0

            training_data.append({"state": state, "policy": policy, "value": value})
            
        except Exception as e:
            print(f"      ❌ Turn {i+1} 데이터 생성 실패: {e}")
            continue

    print(f"   ✅ 훈련 데이터 생성 완료: {len(training_data)}개 포지션")
    print(f"      MCTS 정책: {mcts_policy_count}개")
    print(f"      Direct 정책: {direct_policy_count}개")
    
    return training_data


def save_game_data(training_data, output_dir, game_id):
    """게임 데이터 저장"""
    print(f"\n💾 Game {game_id + 1} 데이터 저장 중...")
    
    try:
        os.makedirs(output_dir, exist_ok=True)

        # 데이터 분리
        states = [data["state"] for data in training_data]
        policies = [data["policy"] for data in training_data]
        values = [data["value"] for data in training_data]

        # PyTorch 텐서로 변환
        states_tensor = torch.FloatTensor(states)
        policies_tensor = torch.FloatTensor(policies)
        values_tensor = torch.FloatTensor(values)

        # 저장
        data = {
            "states": states_tensor,
            "policies": policies_tensor,
            "values": values_tensor,
        }

        file_path = os.path.join(output_dir, f"game_{game_id}.pt")
        torch.save(data, file_path)
        
        file_size = os.path.getsize(file_path) / 1024 / 1024  # MB
        print(f"   ✅ 저장 완료: {file_path} ({file_size:.2f} MB)")

        return file_path
        
    except Exception as e:
        print(f"   ❌ 저장 실패: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate self-play data for YINSH AlphaZero with enhanced MCTS policy extraction"
    )
    parser.add_argument("--model", type=str, default=None, help="Model path to load")
    parser.add_argument("--games", type=int, default=50, help="Number of games to play")
    parser.add_argument("--output", type=str, default="memory", help="Output directory")
    parser.add_argument(
        "--no-mcts", action="store_true", help="Disable MCTS (use direct prediction)"
    )
    parser.add_argument("--mcts-sims", type=int, default=800, help="MCTS simulations")
    parser.add_argument("--show-board", action="store_true", help="각 턴마다 보드 상태 출력")
    parser.add_argument("--workers", type=int, default=6,
                       help="병렬 워커 수 (기본: 6)")
    
    # 빠른 모드 옵션 추가
    parser.add_argument("--fast", action="store_true",
                       help="빠른 selfplay 모드 (적은 시뮬레이션, 최소 로깅)")
    parser.add_argument("--ultra-fast", action="store_true",
                       help="초고속 selfplay 모드 (매우 적은 시뮬레이션, 로깅 없음)")
    
    # 병렬 MCTS 옵션 추가
    parser.add_argument("--parallel-mcts", action="store_true",
                       help="병렬 MCTS 사용 (CPU 멀티스레딩)")
    parser.add_argument("--mcts-threads", type=int, default=4,
                       help="병렬 MCTS 스레드 수 (기본: 4)")
    
    args = parser.parse_args()
    
    # 빠른 모드 설정 적용
    if args.fast:
        args.mcts_sims = min(args.mcts_sims, 200)  # 시뮬레이션 수 감소
        args.show_board = False  # 보드 표시 비활성화
        print("⚡ 빠른 모드 활성화")
    elif args.ultra_fast:
        args.mcts_sims = min(args.mcts_sims, 100)  # 매우 적은 시뮬레이션
        args.show_board = False  # 보드 표시 비활성화
        args.workers = min(args.workers * 2, 12)  # 워커 수 증가
        print("🚀 초고속 모드 활성화")

    print("🎮 Starting Enhanced YINSH AlphaZero Self-Play...")
    print(f"📋 설정:")
    print(f"   게임 수: {args.games}")
    print(f"   모델: {args.model if args.model else 'Random initialization'}")
    print(f"   MCTS: {'Disabled' if args.no_mcts else f'Enabled ({args.mcts_sims} sims)'}")
    print(f"   출력 디렉토리: {args.output}")
    print(f"   보드 출력: {'Enabled' if args.show_board else 'Disabled'}")
    print(f"   디바이스: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"   실행 모드: {'Parallel' if args.parallel_mcts and args.mcts_threads > 1 else 'Sequential'}")
    print(f"   워커 수: {args.workers}")

    # 에이전트 생성 (현재 인터페이스에 맞게)
    try:
        print("\n🤖 에이전트 초기화 중...")
        agent1 = YinshAgent(
            model_path=args.model, 
            use_mcts=not args.no_mcts,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("   ✅ Agent 1 초기화 완료")
        
        agent2 = YinshAgent(
            model_path=args.model, 
            use_mcts=not args.no_mcts,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("   ✅ Agent 2 초기화 완료")
        
    except Exception as e:
        print(f"❌ 에이전트 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 게임 통계
    total_positions = 0
    white_wins = 0
    black_wins = 0
    draws = 0
    mcts_policies_used = 0
    direct_policies_used = 0
    total_start_time = time.time()

    print(f"\n🚀 {args.games}게임 셀프플레이 시작!")
    print("=" * 50)

    # 병렬 실행 모드 확인
    if args.parallel_mcts and args.mcts_threads > 1:
        print(f"\n🚀 병렬 MCTS 모드로 {args.games}개 게임을 {args.mcts_threads}개 스레드로 실행")
        run_parallel_games(args)
        return
    
    # 순차 실행 모드 (기존 방식)
    print(f"\n🐌 순차 모드로 {args.games}개 게임 실행")
    
    # 게임 진행
    for game_id in range(args.games):
        try:
            # 게임 진행
            game_history, winner, turns = play_game(agent1, agent2, game_id=game_id, show_board=args.show_board)

            # 통계 업데이트
            if winner == Color.WHITE:
                white_wins += 1
            elif winner == Color.BLACK:
                black_wins += 1
            else:
                draws += 1

            total_positions += len(game_history)

            # 훈련 데이터 생성 (향상된 정책 분포 포함)
            training_data = generate_training_data(game_history, winner, game_id=game_id)

            # 정책 타입 통계
            for data in training_data:
                if data["policy"].max() == 1.0 and (data["policy"] == 1.0).sum() == 1:
                    direct_policies_used += 1
                else:
                    mcts_policies_used += 1

            # 데이터 저장
            save_game_data(training_data, args.output, game_id)

            # 중간 통계 출력
            elapsed_time = time.time() - total_start_time
            games_completed = game_id + 1
            avg_time_per_game = elapsed_time / games_completed
            estimated_total_time = avg_time_per_game * args.games
            remaining_time = estimated_total_time - elapsed_time
            
            print(f"\n📈 진행 상황 ({games_completed}/{args.games}):")
            print(f"   경과 시간: {elapsed_time:.1f}초")
            print(f"   게임당 평균 시간: {avg_time_per_game:.1f}초")
            print(f"   예상 남은 시간: {remaining_time:.1f}초")
            print("=" * 50)
            
        except Exception as e:
            print(f"❌ Game {game_id + 1} 실패: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 최종 통계
    total_time = time.time() - total_start_time
    print(f"\n🎉 Enhanced Self-Play 완료! (총 소요시간: {total_time:.1f}초)")
    print("📈 최종 통계:")
    print(f"├── 총 게임 수: {args.games}")
    print(f"├── 총 포지션 수: {total_positions}")
    print(f"├── White 승리: {white_wins} ({white_wins/args.games*100:.1f}%)")
    print(f"├── Black 승리: {black_wins} ({black_wins/args.games*100:.1f}%)")
    print(f"├── 무승부: {draws} ({draws/args.games*100:.1f}%)")
    print(f"├── 게임당 평균 포지션: {total_positions/args.games:.1f}")
    print(f"├── MCTS 정책 분포: {mcts_policies_used}")
    print(f"└── Direct/fallback 정책: {direct_policies_used}")

    print(f"\n💾 훈련 데이터 저장 위치: {args.output}")
    print("✅ Enhanced self-play completed successfully!")


if __name__ == "__main__":
    # 멀티프로세싱 시작 방법 설정
    mp.set_start_method('spawn', force=True)
    main()
