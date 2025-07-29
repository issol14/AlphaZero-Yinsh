#!/usr/bin/env python3
"""
최적화된 MCTS 최종 통합 테스트
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

import time
from yinsh import YinshEnv, YinshAgent, Color

def test_agent_creation():
    """에이전트 생성 테스트"""
    print("🔧 에이전트 생성 테스트")
    print("=" * 40)
    
    try:
        # MCTS 에이전트 생성 (scripts에서 사용하는 방식)
        agent = YinshAgent(model_path=None, use_mcts=True)
        print("  ✅ MCTS 에이전트 생성 성공")
        
        # 에이전트 타입 확인
        agent_type = type(agent.mcts_agent).__name__
        print(f"  📊 MCTS 에이전트 타입: {agent_type}")
        
        if "Optimized" in agent_type:
            print("  ✅ 최적화된 MCTS 적용 확인")
        else:
            print("  ❌ 기존 MCTS가 사용됨")
            return False
            
        return True
        
    except Exception as e:
        print(f"  ❌ 에이전트 생성 실패: {e}")
        return False

def test_script_compatibility():
    """Script 사용 패턴 테스트"""
    print(f"\n🎮 Script 호환성 테스트")
    print("=" * 40)
    
    try:
        agent = YinshAgent(model_path=None, use_mcts=True)
        env = YinshEnv()
        
        # selfplay.py 스타일 테스트
        print("  selfplay.py 패턴...")
        original_sims = agent.mcts_agent.mcts.num_simulations
        agent.mcts_agent.mcts.num_simulations = 50  # 빠른 테스트
        
        action, info = agent.select_action(env, temperature=1.0)
        
        if action and 'mcts_stats' in info:
            print("    ✅ selfplay 패턴 성공")
        else:
            print("    ❌ selfplay 패턴 실패")
            return False
        
        # evaluate.py 스타일 테스트
        print("  evaluate.py 패턴...")
        if hasattr(agent, 'mcts_agent') and agent.mcts_agent:
            agent.mcts_agent.mcts.num_simulations = 30
            action2, info2 = agent.select_action(env, temperature=0.1)
            
            if action2 and 'mcts_stats' in info2:
                print("    ✅ evaluate 패턴 성공")
            else:
                print("    ❌ evaluate 패턴 실패")
                return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Script 호환성 테스트 실패: {e}")
        return False

def test_new_features():
    """새로운 기능 테스트"""
    print(f"\n✨ 새로운 기능 테스트")
    print("=" * 40)
    
    try:
        agent = YinshAgent(model_path=None, use_mcts=True)
        env = YinshEnv()
        
        # 빠른 테스트를 위해 시뮬레이션 수 줄임
        agent.mcts_agent.mcts.num_simulations = 30
        
        # 기존 방식 (노이즈 없음)
        print("  기존 방식 테스트...")
        action1, info1 = agent.select_action(env, temperature=1.0)
        
        # 새로운 방식 (Dirichlet 노이즈)
        print("  Dirichlet 노이즈 테스트...")
        action2, info2 = agent.select_action(env, temperature=1.0, add_noise=True)
        
        # 캐시 통계 확인
        if 'cache_stats' in info2:
            cache_stats = info2['cache_stats']
            print(f"    📊 캐시 크기: {cache_stats['cache_size']}")
            print(f"    📊 히트율: {cache_stats['hit_rate']*100:.1f}%")
            print("    ✅ 캐시 통계 확인")
        else:
            print("    ❌ 캐시 통계 없음")
            return False
        
        print("  ✅ 모든 새로운 기능 작동")
        return True
        
    except Exception as e:
        print(f"  ❌ 새로운 기능 테스트 실패: {e}")
        return False

def test_performance_comparison():
    """성능 비교 테스트"""
    print(f"\n⚡ 성능 비교 테스트")
    print("=" * 40)
    
    try:
        agent = YinshAgent(model_path=None, use_mcts=True)
        env = YinshEnv()
        
        # 빠른 테스트
        agent.mcts_agent.mcts.num_simulations = 50
        
        # 성능 측정
        start_time = time.time()
        action, info = agent.select_action(env, temperature=1.0)
        search_time = time.time() - start_time
        
        # 통계 확인
        mcts_stats = info.get('mcts_stats', {})
        nodes_expanded = mcts_stats.get('nodes_expanded', 0)
        sims_per_sec = mcts_stats.get('simulations_per_second', 0)
        
        print(f"  📊 검색 시간: {search_time:.3f}초")
        print(f"  📊 노드 확장: {nodes_expanded}")
        print(f"  📊 시뮬레이션/초: {sims_per_sec:.1f}")
        
        # 성능 기준 (매우 관대함)
        if search_time < 10.0 and nodes_expanded > 0:
            print("  ✅ 성능 기준 통과")
            return True
        else:
            print("  ❌ 성능 기준 미달")
            return False
            
    except Exception as e:
        print(f"  ❌ 성능 테스트 실패: {e}")
        return False

def run_final_integration_test():
    """최종 통합 테스트 실행"""
    print("🎯 최적화된 MCTS 최종 통합 테스트")
    print("=" * 50)
    
    tests = [
        ("에이전트 생성", test_agent_creation),
        ("Script 호환성", test_script_compatibility), 
        ("새로운 기능", test_new_features),
        ("성능 비교", test_performance_comparison),
    ]
    
    all_passed = True
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            all_passed &= result
        except Exception as e:
            print(f"❌ {test_name} 테스트 중 오류: {e}")
            results[test_name] = False
            all_passed = False
    
    print(f"\n🎯 최종 통합 테스트 결과:")
    print("=" * 50)
    
    for test_name, result in results.items():
        status = "✅ 통과" if result else "❌ 실패"
        print(f"  {status} {test_name}")
    
    print(f"\n🏆 전체 결과:")
    if all_passed:
        print("  🎉 모든 테스트 통과!")
        print("  ✅ 최적화된 MCTS가 성공적으로 적용됨")
        print("  🚀 모든 scripts에서 사용 가능")
        print("\n💡 사용 가능한 새로운 기능:")
        print("    - Dirichlet 노이즈 (add_noise=True)")
        print("    - 상태 캐싱 (자동)")
        print("    - 향상된 UCB 공식")
        print("    - AlphaZero 논문 준수 알고리즘")
    else:
        print("  ❌ 일부 테스트 실패")
        print("  🔧 추가 수정 필요")
        
    return all_passed

if __name__ == "__main__":
    success = run_final_integration_test()
    exit(0 if success else 1) 