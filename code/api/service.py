#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH Frontend Service
======================

프론트엔드와의 통신을 위한 서비스 클래스
"""

import os
import sys
import time
import torch
from typing import Optional, Tuple

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yinsh import YinshEnv, YinshAgent, Color
from compression import (
    yinsh_state_to_compressed, compressed_to_yinsh_state,
    validate_compressed_data
)

class FrontendService:
    """프론트엔드 서비스 클래스 - Stateless AI Agent"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Args:
            model_path: AI 모델 경로. None이면 랜덤 모델 사용
        """
        self.ai_agent = None
        self.model_path = model_path or "/home/mori/lab/AlphaZero-Yinsh/code/api/model/best_model.pt"
        self._load_ai_agent()
    
    def _load_ai_agent(self):
        """AI 에이전트 로드 (한 번만 로드, 모든 요청에서 재사용)"""
        print("🤖 AI 에이전트 로딩 중...")
        
        try:
            # Stateless AI 에이전트 생성
            self.ai_agent = YinshAgent(
                model_path=self.model_path if os.path.exists(self.model_path) else None,
                use_mcts=True,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            print(f"   ✅ AI 에이전트 로드 완료 (Model: {self.model_path})")
            print(f"   📊 디바이스: {self.ai_agent.device}")
            print(f"   🌳 MCTS: {'활성화' if self.ai_agent.use_mcts else '비활성화'}")
        except Exception as e:
            print(f"   ❌ AI 에이전트 로드 실패: {e}")
            # 백업으로 랜덤 모델 사용
            try:
                self.ai_agent = YinshAgent(
                    model_path=None,  # 랜덤 모델
                    use_mcts=True,
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
                print("   🔄 백업 랜덤 모델로 초기화 완료")
            except Exception as e2:
                print(f"   ❌ 백업 모델 로드도 실패: {e2}")
                self.ai_agent = None
    
    def process_board_state(self, compressed_state: str) -> Tuple[str, Optional[dict], Optional[float]]:
        """
        Stateless AI 추론: 현재 보드 상태만을 기반으로 최적 액션 결정
        
        Args:
            compressed_state: 압축된 보드 상태 (프론트엔드 형식)
            
        Returns:
            (AI 액션 적용 후 압축 상태, AI 액션 정보, 추론 시간)
        """
        if not validate_compressed_data(compressed_state):
            raise ValueError("Invalid compressed data format")
        
        # 압축된 상태를 YINSH 상태로 변환
        yinsh_state = compressed_to_yinsh_state(compressed_state)
        
        # 임시 YinshEnv 생성 및 상태 복원 (Stateless)
        temp_env = YinshEnv()
        self._restore_environment_state(temp_env, yinsh_state)
        
        print(f"🎯 AI 추론 요청:")
        print(f"   현재 플레이어: {temp_env.current_player.name}")
        print(f"   링 개수: W{len(temp_env.ring_positions[Color.WHITE])}/B{len(temp_env.ring_positions[Color.BLACK])}")
        print(f"   마커 개수: W{len(temp_env.marker_positions[Color.WHITE])}/B{len(temp_env.marker_positions[Color.BLACK])}")
        print(f"   게임 종료: {temp_env.done}")
        
        # AI 에이전트로 최적 액션 추론
        try:
            if not self.ai_agent:
                raise ValueError("AI 에이전트가 로드되지 않았습니다")
            
            # 게임이 이미 종료된 경우
            if temp_env.is_game_over():
                print("   ⚠️ 게임이 이미 종료되었습니다")
                return compressed_state, None, 0.0
            
            start_time = time.time()
            
            # Stateless AI 액션 선택 (현재 상태만 고려)
            action, action_info = self.ai_agent.select_action(temp_env)
            
            thinking_time = time.time() - start_time
            
            if action is None:
                print("   ❌ AI가 유효한 액션을 찾지 못했습니다")
                return compressed_state, None, thinking_time
            
            print(f"   ✅ AI 액션 선택: {action} (추론 시간: {thinking_time:.3f}초)")
            
            # 액션 실행하여 새 상태 생성
            success = temp_env.step(action)
            if not success:
                print("   ❌ 액션 실행 실패")
                return compressed_state, None, thinking_time
            
            # AI 액션을 프론트엔드 형식으로 변환
            ai_action = self._convert_action_to_dict(action)
            
            # 업데이트된 환경 상태를 압축된 형태로 변환
            updated_state = self._extract_environment_state(temp_env)
            updated_compressed = yinsh_state_to_compressed(updated_state)
            
            print(f"   📤 응답 준비 완료")
            
            return updated_compressed, ai_action, thinking_time
            
        except Exception as e:
            print(f"❌ AI 추론 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            # 오류 시 원본 상태 반환
            return compressed_state, None, None
    
    def _restore_environment_state(self, env: YinshEnv, yinsh_state: dict):
        """압축된 상태를 YinshEnv에 복원 (새 게임 규칙 적용)"""
        # 현재 플레이어 설정
        env.current_player = Color.WHITE if yinsh_state["current_player"] == "WHITE" else Color.BLACK
        
        # 링 위치 복원 (hex 좌표를 내부에서 사용)
        env.ring_positions[Color.WHITE] = set()
        env.ring_positions[Color.BLACK] = set()
        
        for hex_pos in yinsh_state["rings_white"]:
            # hex 좌표를 튜플로 변환
            if isinstance(hex_pos, list):
                hex_pos = tuple(hex_pos)
            env.ring_positions[Color.WHITE].add(hex_pos)
            
        for hex_pos in yinsh_state["rings_black"]:
            # hex 좌표를 튜플로 변환
            if isinstance(hex_pos, list):
                hex_pos = tuple(hex_pos)
            env.ring_positions[Color.BLACK].add(hex_pos)
        
        # 마커 위치 복원 (hex 좌표를 내부에서 사용)
        env.marker_positions[Color.WHITE] = set()
        env.marker_positions[Color.BLACK] = set()
        
        for hex_pos in yinsh_state["markers_white"]:
            # hex 좌표를 튜플로 변환
            if isinstance(hex_pos, list):
                hex_pos = tuple(hex_pos)
            env.marker_positions[Color.WHITE].add(hex_pos)
            
        for hex_pos in yinsh_state["markers_black"]:
            # hex 좌표를 튜플로 변환
            if isinstance(hex_pos, list):
                hex_pos = tuple(hex_pos)
            env.marker_positions[Color.BLACK].add(hex_pos)
        
        # 마커 풀 크기 계산 (새 게임 규칙)
        total_markers_on_board = len(env.marker_positions[Color.WHITE]) + len(env.marker_positions[Color.BLACK])
        
        # 보드 상태 업데이트 (array 좌표계로 변환)
        env.board.fill(0)  # 초기화
        
        # 링 위치를 보드에 반영
        for hex_pos in env.ring_positions[Color.WHITE]:
            array_pos = env.hex_to_array_coords(hex_pos)
            env.board[array_pos[0], array_pos[1]] = Color.WHITE.value
            
        for hex_pos in env.ring_positions[Color.BLACK]:
            array_pos = env.hex_to_array_coords(hex_pos)
            env.board[array_pos[0], array_pos[1]] = Color.BLACK.value
        
        # 마커 위치를 보드에 반영 (마커는 다른 값으로 구분 필요시)
        # 현재 구현에서는 링과 마커가 같은 보드를 사용하므로 생략
        
        # 게임 종료 상태 확인 (새 게임 규칙: 5연속 즉시 승리)
        env._check_game_end()
    
    def _extract_environment_state(self, env: YinshEnv) -> dict:
        """YinshEnv 상태를 압축 형식으로 추출 (새 게임 규칙 적용)"""
        # hex 좌표는 이미 올바른 형식으로 저장되어 있음
        rings_white = list(env.ring_positions[Color.WHITE])
        rings_black = list(env.ring_positions[Color.BLACK])
        markers_white = list(env.marker_positions[Color.WHITE])
        markers_black = list(env.marker_positions[Color.BLACK])
        
        return {
            "current_player": env.current_player.name,
            "rings_white": rings_white,
            "rings_black": rings_black,
            "markers_white": markers_white,
            "markers_black": markers_black
        }
    
    def _convert_action_to_dict(self, action) -> dict:
        """YINSH 액션을 프론트엔드 형식으로 변환 (새 게임 규칙)"""
        # 새 게임 규칙에서는 MOVE_RING만 사용
        if hasattr(action, 'from_pos') and hasattr(action, 'to_pos'):
            # from_pos와 to_pos는 이미 hex 좌표
            return {
                "action_type": "move_ring",
                "from_pos": {"q": action.from_pos[0], "r": action.from_pos[1]},
                "to_pos": {"q": action.to_pos[0], "r": action.to_pos[1]}
            }
        else:
            return {"action_type": "unknown", "error": "Invalid action format"} 