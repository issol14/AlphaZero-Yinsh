#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH Frontend API Client Example
=================================

프론트엔드 호환 API 테스트 클라이언트
"""

import requests
import json
import time
from typing import Dict, Any

class YinshAPIClient:
    """YINSH 프론트엔드 API 클라이언트"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """서버 상태 확인"""
        url = f"{self.base_url}/health"
        
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def process_board_state(self, compressed_state: str) -> Dict[str, Any]:
        """tesee.py 호환 보드 상태 처리"""
        url = f"{self.base_url}/process-board-state/"
        
        response = self.session.post(url, json=compressed_state)
        response.raise_for_status()
        return response.json()

def demo_api():
    """API 테스트"""
    client = YinshAPIClient()
    
    print("🔧 YINSH 프론트엔드 API 테스트")
    print("=" * 50)
    
    # 서버 상태 확인
    try:
        health = client.health_check()
        print(f"✅ 서버 상태: {health['status']}")
        print(f"   업타임: {health.get('uptime', 'N/A')}초")
        print(f"   서비스 사용 가능: {health.get('service_available', False)}")
    except Exception as e:
        print(f"❌ 서버 연결 실패: {e}")
        return
    
    # 샘플 압축된 상태 (빈 보드)
    sample_compressed_state = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
    
    try:
        response = client.process_board_state(sample_compressed_state)
        print(f"\n✅ AI 추론 성공!")
        print(f"   처리된 상태 길이: {len(response['compressed_state'])}")
        print(f"   응답 형식: {type(response)}")
    except Exception as e:
        print(f"❌ AI 추론 실패: {e}")
    
    print("\n🎉 테스트 완료!")

if __name__ == "__main__":
    demo_api() 