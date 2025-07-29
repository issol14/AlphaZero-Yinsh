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
        
        # 임시 환경 생성하여 상태 복원
        temp_env = YinshEnv()
        yinsh_state = compressed_to_yinsh_state(compressed_state)
        
        # 환경 상태 복원
        temp_env.current_player = Color.WHITE if yinsh_state["current_player"] == "WHITE" else Color.BLACK
        temp_env.ring_positions[Color.WHITE] = set(yinsh_state["rings_white"])
        temp_env.ring_positions[Color.BLACK] = set(yinsh_state["rings_black"])
        temp_env.marker_positions[Color.WHITE] = set(yinsh_state["markers_white"])
        temp_env.marker_positions[Color.BLACK] = set(yinsh_state["markers_black"])
        
        # 게임 단계 및 기타 상태 복원
        temp_env.phase = self._determine_game_phase(temp_env)
        temp_env._update_game_state()
        
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
            updated_state = {
                "current_player": temp_env.current_player.name,
                "rings_white": list(temp_env.ring_positions[Color.WHITE]),
                "rings_black": list(temp_env.ring_positions[Color.BLACK]),
                "markers_white": list(temp_env.marker_positions[Color.WHITE]),
                "markers_black": list(temp_env.marker_positions[Color.BLACK])
            }
            
            return yinsh_state_to_compressed(updated_state), ai_action, thinking_time
            
        except Exception as e:
            print(f"AI 추론 중 오류 발생: {e}")
            # 오류 시 원본 상태 반환
            return compressed_state, None, None
    
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
            return {
                "action_type": "place_ring",
                "to_pos": {"q": action.to_pos[0], "r": action.to_pos[1]}
            }
        elif action.action_type == "MOVE_RING":
            return {
                "action_type": "move_ring",
                "from_pos": {"q": action.from_pos[0], "r": action.from_pos[1]},
                "to_pos": {"q": action.to_pos[0], "r": action.to_pos[1]}
            }
        elif action.action_type == "REMOVE_LINE":
            return {
                "action_type": "remove_line",
                "remove_positions": [{"q": pos[0], "r": pos[1]} for pos in action.remove_positions]
            }
        else:
            return {"action_type": "unknown"} 