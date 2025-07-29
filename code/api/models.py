#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH Frontend API Models
=========================

프론트엔드와의 통신을 위한 API 모델들
"""

from typing import Optional
from pydantic import BaseModel

# ============================================================================
# tesee.py 호환 API 모델
# ============================================================================

class ProcessBoardStateResponse(BaseModel):
    """tesee.py 호환 응답"""
    compressed_state: str 