#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YINSH Frontend API Server
=========================

프론트엔드와의 통신을 위한 FastAPI 서버
"""

import os
import sys
import time
import asyncio
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from models import ProcessBoardStateResponse
from service import FrontendService

# ============================================================================
# 서버 생명주기 관리
# ============================================================================

frontend_service: Optional[FrontendService] = None
server_start_time: Optional[datetime] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 실행"""
    global frontend_service, server_start_time
    
    # 서버 시작
    print("🚀 YINSH Frontend API 서버 시작 중...")
    server_start_time = datetime.now()
    
    # 프론트엔드 서비스 초기화
    frontend_service = FrontendService()
    
    print("✅ 서버 시작 완료!")
    yield
    
    # 서버 종료
    print("🛑 서버 종료 중...")
    if frontend_service:
        del frontend_service
    print("✅ 서버 종료 완료!")

# ============================================================================
# FastAPI 앱 설정
# ============================================================================

app = FastAPI(
    title="YINSH Frontend API",
    description="프론트엔드와의 통신을 위한 YINSH API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# 의존성 주입
# ============================================================================

def get_frontend_service() -> FrontendService:
    """프론트엔드 서비스 의존성"""
    if frontend_service is None:
        raise HTTPException(status_code=503, detail="서비스가 초기화되지 않았습니다")
    return frontend_service

# ============================================================================
# 기본 엔드포인트
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """루트 엔드포인트"""
    return {
        "message": "YINSH Frontend API 서버",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """헬스 체크"""
    uptime = (datetime.now() - server_start_time).total_seconds() if server_start_time else 0
    
    return {
        "status": "healthy",
        "uptime": uptime,
        "service_available": frontend_service is not None
    }

# ============================================================================
# 프론트엔드 호환 API 엔드포인트
# ============================================================================

@app.post("/process-board-state/", response_model=ProcessBoardStateResponse, tags=["Frontend"])
async def process_board_state(compressed_state: str):
    """tesee.py 호환 보드 상태 처리 - AI 추론 수행"""
    try:
        frontend_service = get_frontend_service()
        processed_state, ai_action, thinking_time = frontend_service.process_board_state(compressed_state)
        
        return {"compressed_state": processed_state}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="내부 서버 오류")

# ============================================================================
# 예외 처리
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP 예외 처리"""
    return {
        "success": False,
        "error": exc.detail,
        "status_code": exc.status_code
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """일반 예외 처리"""
    return {
        "success": False,
        "error": "내부 서버 오류",
        "status_code": 500
    }

# ============================================================================
# 서버 실행
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 