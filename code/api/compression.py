#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH Board State Compression Utilities
======================================

프론트엔드와의 통신을 위한 보드 상태 압축/해제 유틸리티
"""

import base64
import numpy as np
from typing import Dict, Tuple, List

# ============================================================================
# 상수 정의
# ============================================================================

BOARD_STATE_EMPTY = 0
BOARD_STATE_WHITE_RING = 1
BOARD_STATE_BLACK_RING = 2
BOARD_STATE_WHITE_MARKER = 3
BOARD_STATE_BLACK_MARKER = 4

# 85개 유효한 좌표 (YINSH 육각형 보드)
COORDINATE_TO_ID = {
    (-5, -4): 0, (-5, -3): 1, (-5, -2): 2, (-5, -1): 3,
    (-4, -5): 4, (-4, -4): 5, (-4, -3): 6, (-4, -2): 7, (-4, -1): 8, (-4, 0): 9, (-4, 1): 10,
    (-3, -5): 11, (-3, -4): 12, (-3, -3): 13, (-3, -2): 14, (-3, -1): 15, (-3, 0): 16, (-3, 1): 17, (-3, 2): 18,
    (-2, -5): 19, (-2, -4): 20, (-2, -3): 21, (-2, -2): 22, (-2, -1): 23, (-2, 0): 24, (-2, 1): 25, (-2, 2): 26, (-2, 3): 27,
    (-1, -5): 28, (-1, -4): 29, (-1, -3): 30, (-1, -2): 31, (-1, -1): 32, (-1, 0): 33, (-1, 1): 34, (-1, 2): 35, (-1, 3): 36, (-1, 4): 37,
    (0, -4): 38, (0, -3): 39, (0, -2): 40, (0, -1): 41, (0, 0): 42, (0, 1): 43, (0, 2): 44, (0, 3): 45, (0, 4): 46,
    (1, -4): 47, (1, -3): 48, (1, -2): 49, (1, -1): 50, (1, 0): 51, (1, 1): 52, (1, 2): 53, (1, 3): 54, (1, 4): 55, (1, 5): 56,
    (2, -3): 57, (2, -2): 58, (2, -1): 59, (2, 0): 60, (2, 1): 61, (2, 2): 62, (2, 3): 63, (2, 4): 64, (2, 5): 65,
    (3, -2): 66, (3, -1): 67, (3, 0): 68, (3, 1): 69, (3, 2): 70, (3, 3): 71, (3, 4): 72, (3, 5): 73,
    (4, -1): 74, (4, 0): 75, (4, 1): 76, (4, 2): 77, (4, 3): 78, (4, 4): 79, (4, 5): 80,
    (5, 1): 81, (5, 2): 82, (5, 3): 83, (5, 4): 84,
}

# ID to Coordinate mapping
ID_TO_COORDINATE = {v: k for k, v in COORDINATE_TO_ID.items()}

# ============================================================================
# 압축/해제 함수들
# ============================================================================

def yinsh_state_to_compressed(env_state: Dict) -> str:
    """
    YINSH 환경 상태를 압축된 문자열로 변환
    
    Args:
        env_state: YINSH 환경의 상태 딕셔너리
        
    Returns:
        압축된 base64 문자열
    """
    compressed = bytearray(32)
    
    # 턴 정보 설정 (0=white, 1=black)
    current_player = env_state.get("current_player", "WHITE")
    turn_bit = 1 if current_player == "BLACK" else 0
    compressed[0] |= (turn_bit & 1) << 7
    
    # 링과 마커 위치 압축
    bit_position = 1
    
    # 각 위치별 상태 설정
    for pos_id in range(85):
        coord = ID_TO_COORDINATE[pos_id]
        state_value = BOARD_STATE_EMPTY
        
        # 링 위치 확인
        if coord in env_state.get("rings_white", []):
            state_value = BOARD_STATE_WHITE_RING
        elif coord in env_state.get("rings_black", []):
            state_value = BOARD_STATE_BLACK_RING
        # 마커 위치 확인
        elif coord in env_state.get("markers_white", []):
            state_value = BOARD_STATE_WHITE_MARKER
        elif coord in env_state.get("markers_black", []):
            state_value = BOARD_STATE_BLACK_MARKER
        
        # 3비트로 상태 인코딩
        for bit_offset in range(3):
            if state_value & (1 << bit_offset):
                byte_index = bit_position // 8
                bit_index = bit_position % 8
                compressed[byte_index] |= 1 << (7 - bit_index)
            bit_position += 1
    
    return base64.b64encode(bytes(compressed)).decode('utf-8')

def compressed_to_yinsh_state(compressed_data: str) -> Dict:
    """
    압축된 문자열을 YINSH 환경 상태로 변환
    
    Args:
        compressed_data: 압축된 base64 문자열
        
    Returns:
        YINSH 환경 상태 딕셔너리
    """
    compressed = base64.b64decode(compressed_data)
    if len(compressed) != 32:
        raise ValueError("Invalid compressed data length")
    
    # 턴 정보 추출
    turn_bit = (compressed[0] >> 7) & 1
    current_player = "BLACK" if turn_bit == 1 else "WHITE"
    
    # 링과 마커 위치 추출
    rings_white = []
    rings_black = []
    markers_white = []
    markers_black = []
    
    bit_position = 1
    for pos_id in range(85):
        coord = ID_TO_COORDINATE[pos_id]
        state_value = _get_state_at_pos(compressed, pos_id)
        
        if state_value == BOARD_STATE_WHITE_RING:
            rings_white.append(coord)
        elif state_value == BOARD_STATE_BLACK_RING:
            rings_black.append(coord)
        elif state_value == BOARD_STATE_WHITE_MARKER:
            markers_white.append(coord)
        elif state_value == BOARD_STATE_BLACK_MARKER:
            markers_black.append(coord)
    
    return {
        "current_player": current_player,
        "rings_white": rings_white,
        "rings_black": rings_black,
        "markers_white": markers_white,
        "markers_black": markers_black
    }

def compressed_to_neural_input(compressed_data: str) -> np.ndarray:
    """
    압축된 상태를 신경망 입력용 11x11x13 텐서로 변환
    
    Args:
        compressed_data: 압축된 base64 문자열
        
    Returns:
        11x11x13 numpy 배열
    """
    compressed = base64.b64decode(compressed_data)
    if len(compressed) != 32:
        raise ValueError("Invalid compressed data length")
    
    # 턴 정보 추출
    turn_bit = (compressed[0] >> 7) & 1
    current_turn = "white" if turn_bit == 0 else "black"
    
    # 11x11x13 배열 초기화
    state = np.zeros((11, 11, 13), dtype=np.float32)
    
    # 링과 마커 개수 계산
    white_rings = sum(1 for i in range(85) if _get_state_at_pos(compressed, i) == BOARD_STATE_WHITE_RING)
    black_rings = sum(1 for i in range(85) if _get_state_at_pos(compressed, i) == BOARD_STATE_BLACK_RING)
    white_markers = sum(1 for i in range(85) if _get_state_at_pos(compressed, i) == BOARD_STATE_WHITE_MARKER)
    black_markers = sum(1 for i in range(85) if _get_state_at_pos(compressed, i) == BOARD_STATE_BLACK_MARKER)
    board_markers = white_markers + black_markers
    
    # 게임 단계 계산
    game_phase = 0.0 if white_rings + black_rings < 10 else (0.5 if board_markers == 0 else 1.0)
    
    # 정규화된 값들
    white_rings_placed = min(white_rings / 5, 1.0)
    black_rings_placed = min(black_rings / 5, 1.0)
    white_rings_removed = min((5 - white_rings) / 3, 1.0) if white_rings < 5 else 1.0
    black_rings_removed = min((5 - black_rings) / 3, 1.0) if black_rings < 5 else 1.0
    markers_remaining = min((51 - board_markers) / 51, 1.0)
    markers_on_board = min(board_markers / 51, 1.0)
    
    # 전역 채널 값 설정
    state[:, :, 4] = 1.0 if current_turn == "white" else 0.0  # 채널 4: 현재 플레이어
    state[:, :, 5] = game_phase  # 채널 5: 게임 단계
    state[:, :, 6] = white_rings_placed  # 채널 6: 흰색 링 배치
    state[:, :, 7] = black_rings_placed  # 채널 7: 검은색 링 배치
    state[:, :, 8] = white_rings_removed  # 채널 8: 흰색 링 제거
    state[:, :, 9] = black_rings_removed  # 채널 9: 검은색 링 제거
    state[:, :, 10] = markers_remaining  # 채널 10: 남은 마커
    state[:, :, 11] = markers_on_board  # 채널 11: 보드 위 마커
    
    # 85개 위치를 11x11 그리드에 매핑
    for pos_id in range(85):
        x = pos_id // 11
        y = pos_id % 11
        state_value = _get_state_at_pos(compressed, pos_id)
        
        # 채널 0-3: 위치별 상태
        if state_value == BOARD_STATE_WHITE_RING:
            state[x, y, 0] = 1.0 if current_turn == "white" else 0.0  # 채널 0: 현재 플레이어 링
            state[x, y, 2] = 1.0 if current_turn != "white" else 0.0  # 채널 2: 상대방 링
        elif state_value == BOARD_STATE_BLACK_RING:
            state[x, y, 0] = 1.0 if current_turn == "black" else 0.0  # 채널 0: 현재 플레이어 링
            state[x, y, 2] = 1.0 if current_turn != "black" else 0.0  # 채널 2: 상대방 링
        elif state_value == BOARD_STATE_WHITE_MARKER:
            state[x, y, 1] = 1.0 if current_turn == "white" else 0.0  # 채널 1: 현재 플레이어 마커
            state[x, y, 3] = 1.0 if current_turn != "white" else 0.0  # 채널 3: 상대방 마커
        elif state_value == BOARD_STATE_BLACK_MARKER:
            state[x, y, 1] = 1.0 if current_turn == "black" else 0.0  # 채널 1: 현재 플레이어 마커
            state[x, y, 3] = 1.0 if current_turn != "black" else 0.0  # 채널 3: 상대방 마커
        
        state[x, y, 12] = 1.0  # 채널 12: 유효한 위치
    
    return state

def neural_output_to_compressed(neural_output: np.ndarray, current_turn: str) -> str:
    """
    신경망 출력을 압축된 상태로 변환
    
    Args:
        neural_output: 11x11x13 신경망 출력
        current_turn: 현재 턴 ("white" 또는 "black")
        
    Returns:
        압축된 base64 문자열
    """
    compressed = bytearray(32)
    
    # 턴 비트 설정
    turn_bit = 1 if current_turn == "black" else 0
    compressed[0] |= (turn_bit & 1) << 7
    
    bit_position = 1
    for pos_id in range(85):
        x = pos_id // 11
        y = pos_id % 11
        state_value = BOARD_STATE_EMPTY
        
        # 채널 0-3에서 상태 결정
        if neural_output[x, y, 0] > 0.5:  # 현재 플레이어 링
            state_value = BOARD_STATE_WHITE_RING if current_turn == "white" else BOARD_STATE_BLACK_RING
        elif neural_output[x, y, 1] > 0.5:  # 현재 플레이어 마커
            state_value = BOARD_STATE_WHITE_MARKER if current_turn == "white" else BOARD_STATE_BLACK_MARKER
        elif neural_output[x, y, 2] > 0.5:  # 상대방 링
            state_value = BOARD_STATE_BLACK_RING if current_turn == "white" else BOARD_STATE_WHITE_RING
        elif neural_output[x, y, 3] > 0.5:  # 상대방 마커
            state_value = BOARD_STATE_BLACK_MARKER if current_turn == "white" else BOARD_STATE_WHITE_MARKER
        
        # 3비트로 상태 인코딩
        for bit_offset in range(3):
            if state_value & (1 << bit_offset):
                byte_index = bit_position // 8
                bit_index = bit_position % 8
                compressed[byte_index] |= 1 << (7 - bit_index)
            bit_position += 1
    
    return base64.b64encode(bytes(compressed)).decode('utf-8')

# ============================================================================
# 헬퍼 함수들
# ============================================================================

def _get_state_at_pos(compressed: bytes, pos_id: int) -> int:
    """압축된 데이터에서 특정 위치의 상태 추출"""
    bit_position = 1 + pos_id * 3
    state = 0
    for bit_offset in range(3):
        byte_index = bit_position // 8
        bit_index = bit_position % 8
        if compressed[byte_index] & (1 << (7 - bit_index)):
            state |= 1 << bit_offset
        bit_position += 1
    return state

def validate_compressed_data(compressed_data: str) -> bool:
    """압축된 데이터 유효성 검사"""
    try:
        compressed = base64.b64decode(compressed_data)
        return len(compressed) == 32
    except:
        return False 