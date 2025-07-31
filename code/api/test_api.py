#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH API 테스트 스크립트
========================

Stateless AI API 테스트용
"""

import os
import sys
import json
import requests
import time

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yinsh import YinshEnv, Color
sys.path.append(os.path.join(os.path.dirname(__file__)))
from compression import yinsh_state_to_compressed, compressed_to_yinsh_state

def create_test_game_state():
    """테스트용 게임 상태 생성"""
    env = YinshEnv()
    
    # 게임이 초기화된 상태 (링이 랜덤 배치됨)
    print("🎮 테스트 게임 상태 생성")
    print(f"   현재 플레이어: {env.current_player.name}")
    print(f"   링 개수: W{len(env.ring_positions[Color.WHITE])}/B{len(env.ring_positions[Color.BLACK])}")
    print(f"   마커 개수: W{len(env.marker_positions[Color.WHITE])}/B{len(env.marker_positions[Color.BLACK])}")
    
    # 환경 상태를 딕셔너리로 변환
    state_dict = {
        "current_player": env.current_player.name,
        "rings_white": list(env.ring_positions[Color.WHITE]),
        "rings_black": list(env.ring_positions[Color.BLACK]),
        "markers_white": list(env.marker_positions[Color.WHITE]),
        "markers_black": list(env.marker_positions[Color.BLACK])
    }
    
    return state_dict

def test_compression():
    """압축/해제 테스트"""
    print("\n🧪 압축/해제 테스트")
    
    # 테스트 상태 생성
    state_dict = create_test_game_state()
    
    # 압축
    compressed = yinsh_state_to_compressed(state_dict)
    print(f"   압축된 상태 길이: {len(compressed)} bytes")
    
    # 해제
    decompressed = compressed_to_yinsh_state(compressed)
    print(f"   해제 성공: {decompressed['current_player']}")
    
    return compressed

def test_api_request(compressed_state: str, server_url: str = "http://127.0.0.1:8000"):
    """API 요청 테스트"""
    print(f"\n🌐 API 요청 테스트 ({server_url})")
    
    try:
        start_time = time.time()
        
        # API 요청
        response = requests.post(
            f"{server_url}/process-board-state",
            json={"compressed_state": compressed_state},
            timeout=30
        )
        
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ 요청 성공 (소요시간: {request_time:.3f}초)")
            print(f"   📊 응답 키: {list(result.keys())}")
            
            # AI 액션 정보가 있으면 출력
            if "ai_action" in result:
                ai_action = result["ai_action"]
                print(f"   🤖 AI 액션: {ai_action['action_type']}")
                if "thinking_time" in result:
                    print(f"   ⏱️ AI 추론 시간: {result['thinking_time']:.3f}초")
            
            return result
        else:
            print(f"   ❌ 요청 실패: {response.status_code}")
            print(f"   에러: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print("   ❌ 서버 연결 실패. 서버가 실행 중인지 확인하세요.")
        return None
    except Exception as e:
        print(f"   ❌ 요청 실패: {e}")
        return None

def test_multiple_requests(num_requests: int = 5):
    """여러 요청 테스트 (동시성 테스트)"""
    print(f"\n🔁 다중 요청 테스트 ({num_requests}회)")
    
    # 여러 다른 게임 상태 생성
    test_states = []
    for i in range(num_requests):
        state_dict = create_test_game_state()
        compressed = yinsh_state_to_compressed(state_dict)
        test_states.append(compressed)
    
    results = []
    total_start_time = time.time()
    
    for i, compressed_state in enumerate(test_states):
        print(f"   요청 {i+1}/{num_requests}")
        result = test_api_request(compressed_state)
        if result:
            results.append(result)
        time.sleep(0.1)  # 짧은 대기
    
    total_time = time.time() - total_start_time
    success_rate = len(results) / num_requests * 100
    
    print(f"\n📈 다중 요청 결과:")
    print(f"   총 요청: {num_requests}회")
    print(f"   성공: {len(results)}회")
    print(f"   성공률: {success_rate:.1f}%")
    print(f"   총 소요시간: {total_time:.3f}초")
    print(f"   평균 요청 시간: {total_time/num_requests:.3f}초")

def main():
    """메인 테스트 함수"""
    print("🚀 YINSH Stateless AI API 테스트 시작")
    print("=" * 50)
    
    # 1. 압축/해제 테스트
    compressed_state = test_compression()
    
    # 2. 단일 API 요청 테스트
    result = test_api_request(compressed_state)
    
    if result:
        # 3. 다중 요청 테스트
        test_multiple_requests(3)
    
    print("\n✅ 테스트 완료!")

if __name__ == "__main__":
    main()