#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH Frontend API Models
=========================

프론트엔드와의 통신을 위한 API 모델들
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel

# ============================================================================
# API 응답 모델
# ============================================================================

class ProcessBoardStateResponse(BaseModel):
    """보드 상태 처리 응답"""
    compressed_state: str
    ai_action: Optional[Dict[str, Any]] = None
    thinking_time: Optional[float] = None
    success: Optional[bool] = True 