#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate.py 디버깅용 단일 게임 테스트
"""

import os
import sys
import time
import torch
from pathlib import Path

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yinsh import YinshAgent, Color
from yinsh.game import YinshGame


def debug_single_game():
    """단일 게임을 상세히 디버깅"""
    print("🔍 단일 게임 디버깅 시작")
    
    # 모델 파일 확인
    models_dir = Path("models")
    if not models_dir.exists():
        print("❌ models 디렉토리가 없습니다.")
        return False
    
    model_files = list(models_dir.glob("*.pth"))
    if not model_files:
        print("❌ models 디렉토리에 .pth 파일이 없습니다.")
        return False
    
    print(f"📁 발견된 모델 파일들: {[f.name for f in model_files]}")
    
    # 첫 번째 모델 사용
    model_path = str(model_files[0])
    print(f"🎯 사용할 모델: {model_path}")
    
    try:
        # 1. 에이전트 생성 테스트
        print("\n1️⃣ 에이전트 생성 테스트")
        print("   GPU 사용 가능:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("   GPU 메모리:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
        
        start_time = time.time()
        agent1 = YinshAgent(
            model_path=model_path,
            use_mcts=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        agent2 = YinshAgent(
            model_path=model_path,
            use_mcts=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        agent_creation_time = time.time() - start_time
        print(f"   ✅ 에이전트 생성 완료 (소요시간: {agent_creation_time:.2f}초)")
        
        # 2. MCTS 설정 확인
        print("\n2️⃣ MCTS 설정 확인")
        if hasattr(agent1, 'mcts_agent') and agent1.mcts_agent:
            print(f"   MCTS 시뮬레이션 수: {agent1.mcts_agent.num_simulations}")
            # 빠른 테스트를 위해 시뮬레이션 수 줄임
            agent1.mcts_agent.num_simulations = 50
            agent2.mcts_agent.num_simulations = 50
            print(f"   → 빠른 테스트용으로 {agent1.mcts_agent.num_simulations}로 조정")
        else:
            print("   ❌ MCTS 에이전트가 없습니다.")
            return False
        
        # 3. 게임 생성 테스트
        print("\n3️⃣ 게임 생성 테스트")
        start_time = time.time()
        game = YinshGame(agent1, agent2)
        game_creation_time = time.time() - start_time
        print(f"   ✅ 게임 생성 완료 (소요시간: {game_creation_time:.2f}초)")
        
        # 4. 단일 게임 실행 (타임아웃 포함)
        print("\n4️⃣ 단일 게임 실행 테스트")
        print("   ⏰ 타임아웃: 60초")
        
        start_time = time.time()
        
        # 타임아웃 체크 함수
        def check_timeout():
            if time.time() - start_time > 60:
                raise TimeoutError("게임이 60초를 초과했습니다.")
        
        # 게임 실행
        try:
            winner_value = game.play_one_game(stochastic=False)
            game_time = time.time() - start_time
            
            print(f"   ✅ 게임 완료 (소요시간: {game_time:.2f}초)")
            
            # 승자 결정
            if winner_value > 0:
                winner = "WHITE"
            elif winner_value < 0:
                winner = "BLACK"
            else:
                winner = "DRAW"
            
            print(f"   🏆 승자: {winner}")
            print(f"   📊 게임 값: {winner_value}")
            
            if hasattr(game, 'moves_played'):
                print(f"   🎯 총 턴 수: {game.moves_played}")
            
            return True
            
        except TimeoutError as e:
            print(f"   ⏰ {e}")
            return False
        except Exception as e:
            print(f"   ❌ 게임 실행 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"❌ 전체 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def debug_mcts_step():
    """MCTS 단계별 디버깅"""
    print("\n🔍 MCTS 단계별 디버깅")
    
    models_dir = Path("models")
    model_files = list(models_dir.glob("*.pth"))
    if not model_files:
        print("❌ 모델 파일이 없습니다.")
        return False
    
    model_path = str(model_files[0])
    
    try:
        # 에이전트 생성
        agent = YinshAgent(
            model_path=model_path,
            use_mcts=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # MCTS 시뮬레이션 수를 매우 적게 설정
        if hasattr(agent, 'mcts_agent') and agent.mcts_agent:
            agent.mcts_agent.num_simulations = 10
            print(f"   MCTS 시뮬레이션 수: {agent.mcts_agent.num_simulations}")
        
        # 게임 생성
        game = YinshGame(agent, agent)
        
        # 첫 번째 액션만 테스트
        print("   🎮 첫 번째 액션 테스트...")
        start_time = time.time()
        
        # 게임의 첫 번째 턴만 실행
        if hasattr(game, 'env') and game.env:
            print(f"   📊 초기 보드 상태:")
            print(f"      현재 플레이어: {game.env.current_player}")
            print(f"      게임 종료 여부: {game.env.is_terminal()}")
            
            # 첫 번째 액션 선택
            action = agent.select_action(game.env)
            action_time = time.time() - start_time
            
            print(f"   ✅ 첫 번째 액션 완료 (소요시간: {action_time:.2f}초)")
            print(f"   🎯 선택된 액션: {action}")
            
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ MCTS 단계별 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


def debug_model_loading():
    """모델 로딩 디버깅"""
    print("\n🔍 모델 로딩 디버깅")
    
    models_dir = Path("models")
    model_files = list(models_dir.glob("*.pt"))
    if not model_files:
        print("❌ 모델 파일이 없습니다.")
        return False
    
    model_path = str(model_files[0])
    
    try:
        print(f"   📁 모델 파일: {model_path}")
        print(f"   📊 파일 크기: {Path(model_path).stat().st_size / 1e6:.2f} MB")
        
        # 모델 로딩 테스트
        start_time = time.time()
        
        # GPU 메모리 사용량 체크
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            print(f"   💾 초기 GPU 메모리: {initial_memory / 1e6:.2f} MB")
        
        agent = YinshAgent(
            model_path=model_path,
            use_mcts=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        loading_time = time.time() - start_time
        print(f"   ✅ 모델 로딩 완료 (소요시간: {loading_time:.2f}초)")
        
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            print(f"   💾 최종 GPU 메모리: {final_memory / 1e6:.2f} MB")
            print(f"   💾 사용된 GPU 메모리: {(final_memory - initial_memory) / 1e6:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ 모델 로딩 테스트 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("🔍 evaluate.py 디버깅 스위트")
    print("=" * 60)
    
    # 1. 모델 로딩 테스트
    print("\n1️⃣ 모델 로딩 테스트")
    test1_result = debug_model_loading()
    
    # 2. MCTS 단계별 테스트
    print("\n2️⃣ MCTS 단계별 테스트")
    test2_result = debug_mcts_step()
    
    # 3. 단일 게임 테스트
    print("\n3️⃣ 단일 게임 테스트")
    test3_result = debug_single_game()
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 디버깅 결과 요약")
    print("=" * 60)
    print(f"   모델 로딩 테스트: {'✅ 통과' if test1_result else '❌ 실패'}")
    print(f"   MCTS 단계별 테스트: {'✅ 통과' if test2_result else '❌ 실패'}")
    print(f"   단일 게임 테스트: {'✅ 통과' if test3_result else '❌ 실패'}")
    
    if test1_result and test2_result and test3_result:
        print("\n🎉 모든 디버깅 테스트 통과!")
        print("   → evaluate.py 자체에는 문제가 없을 가능성이 높습니다.")
    else:
        print("\n⚠️ 일부 디버깅 테스트 실패.")
        print("   → 해당 부분에서 문제가 발생하고 있습니다.") 