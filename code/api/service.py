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

from yinsh import YinshEnv, YinshAgent, Color, GamePhase
from compression import (
    yinsh_state_to_compressed, compressed_to_yinsh_state,
    validate_compressed_data
)

class FrontendService:
    """프론트엔드 서비스 클래스"""
    
    def __init__(self):
        self.ai_agent = None
        self._load_ai_agent()
    
    def _load_ai_agent(self):
        """AI 에이전트 로드"""
        print("🤖 AI 에이전트 로딩 중...")
        
        try:
            # 기본 에이전트 생성 (모델 없이 랜덤 초기화)
            self.ai_agent = YinshAgent(
                model_path="/home/mori/lab/AlphaZero-Yinsh/code/api/model/best_model.pt",
                use_mcts=True,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            print("   ✅ AI 에이전트 로드 완료")
        except Exception as e:
            print(f"   ❌ AI 에이전트 로드 실패: {e}")
            self.ai_agent = None
    
    def process_board_state(self, compressed_state: str) -> Tuple[str, Optional[dict], Optional[float]]:
        """
        프론트엔드 호환용 보드 상태 처리 - AI 추론 수행
        
        Args:
            compressed_state: 압축된 보드 상태
            
        Returns:
            (처리된 압축 상태, AI 액션, 생각 시간)
        """
        if not validate_compressed_data(compressed_state):
            raise ValueError("Invalid compressed data format")
        
        # 압축된 상태를 YINSH 상태로 변환
        yinsh_state = compressed_to_yinsh_state(compressed_state)
        
        # YinshEnv 인스턴스 생성 및 상태 복원
        temp_env = YinshEnv()
        self._restore_environment_state(temp_env, yinsh_state)
        
        # AI 에이전트로 다음 행동 선택
        try:
            if not self.ai_agent:
                raise ValueError("AI 에이전트가 로드되지 않았습니다")
            
            start_time = time.time()
            
            # AI 액션 선택
            action, action_info = self.ai_agent.select_action(temp_env)
            
            thinking_time = time.time() - start_time
            
            # 액션 실행
            temp_env.step(action)
            
            # AI 액션을 딕셔너리로 변환
            ai_action = self._convert_action_to_dict(action)
            
            # 업데이트된 환경 상태를 압축된 형태로 변환
            updated_state = self._extract_environment_state(temp_env)
            
            return yinsh_state_to_compressed(updated_state), ai_action, thinking_time
            
        except Exception as e:
            print(f"AI 추론 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            # 오류 시 원본 상태 반환
            return compressed_state, None, None
    
    def _restore_environment_state(self, env: YinshEnv, yinsh_state: dict):
        """압축된 상태를 YinshEnv에 복원"""
        # 현재 플레이어 설정
        env.current_player = Color.WHITE if yinsh_state["current_player"] == "WHITE" else Color.BLACK
        
        # 링 위치 복원 (hex 좌표를 array 좌표로 변환)
        env.ring_positions[Color.WHITE] = set()
        env.ring_positions[Color.BLACK] = set()
        
        for hex_pos in yinsh_state["rings_white"]:
            array_pos = env.hex_to_array_coords(hex_pos)
            env.ring_positions[Color.WHITE].add(array_pos)
            
        for hex_pos in yinsh_state["rings_black"]:
            array_pos = env.hex_to_array_coords(hex_pos)
            env.ring_positions[Color.BLACK].add(array_pos)
        
        # 마커 위치 복원 (hex 좌표를 array 좌표로 변환)
        env.marker_positions[Color.WHITE] = set()
        env.marker_positions[Color.BLACK] = set()
        
        for hex_pos in yinsh_state["markers_white"]:
            array_pos = env.hex_to_array_coords(hex_pos)
            env.marker_positions[Color.WHITE].add(array_pos)
            
        for hex_pos in yinsh_state["markers_black"]:
            array_pos = env.hex_to_array_coords(hex_pos)
            env.marker_positions[Color.BLACK].add(array_pos)
        
        # 게임 단계 결정
        env.phase = self._determine_game_phase(env)
        
        # 링 배치 수 업데이트
        env.rings_placed[Color.WHITE] = len(env.ring_positions[Color.WHITE])
        env.rings_placed[Color.BLACK] = len(env.ring_positions[Color.BLACK])
        
        # 게임 종료 상태 확인
        env._check_game_end()
    
    def _extract_environment_state(self, env: YinshEnv) -> dict:
        """YinshEnv 상태를 딕셔너리로 추출"""
        # array 좌표를 hex 좌표로 변환
        rings_white = [env.array_to_hex_coords(pos) for pos in env.ring_positions[Color.WHITE]]
        rings_black = [env.array_to_hex_coords(pos) for pos in env.ring_positions[Color.BLACK]]
        markers_white = [env.array_to_hex_coords(pos) for pos in env.marker_positions[Color.WHITE]]
        markers_black = [env.array_to_hex_coords(pos) for pos in env.marker_positions[Color.BLACK]]
        
        return {
            "current_player": env.current_player.name,
            "rings_white": rings_white,
            "rings_black": rings_black,
            "markers_white": markers_white,
            "markers_black": markers_black
        }
    
    def _determine_game_phase(self, env: YinshEnv) -> GamePhase:
        """게임 단계 결정"""
        total_rings = len(env.ring_positions[Color.WHITE]) + len(env.ring_positions[Color.BLACK])
        
        if total_rings < 10:
            return GamePhase.PLACE_RINGS
        elif env.markers_on_board == 0:
            return GamePhase.MAIN_GAME
        else:
            return GamePhase.LINE_REMOVAL
    
    def _convert_action_to_dict(self, action) -> dict:
        """YINSH 액션을 딕셔너리로 변환"""
        if action.action_type == "PLACE_RING":
            hex_pos = self._array_to_hex_coords(action.to_pos)
            return {
                "action_type": "place_ring",
                "to_pos": {"q": hex_pos[0], "r": hex_pos[1]}
            }
        elif action.action_type == "MOVE_RING":
            from_hex = self._array_to_hex_coords(action.from_pos)
            to_hex = self._array_to_hex_coords(action.to_pos)
            return {
                "action_type": "move_ring",
                "from_pos": {"q": from_hex[0], "r": from_hex[1]},
                "to_pos": {"q": to_hex[0], "r": to_hex[1]}
            }
        elif action.action_type == "REMOVE_LINE":
            remove_positions = [self._array_to_hex_coords(pos) for pos in action.remove_line_positions]
            remove_ring = self._array_to_hex_coords(action.remove_ring_position) if action.remove_ring_position else None
            return {
                "action_type": "remove_line",
                "remove_positions": [{"q": pos[0], "r": pos[1]} for pos in remove_positions],
                "remove_ring": {"q": remove_ring[0], "r": remove_ring[1]} if remove_ring else None
            }
        else:
            return {"action_type": "unknown"}
    
    def _array_to_hex_coords(self, array_pos):
        """배열 좌표를 육각형 좌표로 변환"""
        x, y = array_pos
        hex_x = x - 5
        hex_y = y - 5
        return (hex_x, hex_y) 