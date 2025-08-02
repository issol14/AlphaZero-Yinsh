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
from pydantic import BaseModel

# 프로젝트 루트 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    
    # 프론트엔드 서비스 초기화 (Stateless AI)
    model_path = os.environ.get("YINSH_MODEL_PATH", "/home/mori/lab/AlphaZero-Yinsh/main_experiment/model/best_model.pt")
    frontend_service = FrontendService(model_path=model_path)
    
    print(f"🤖 AI 모델: {model_path}")
    print("🎯 Stateless AI 서비스 준비 완료")
    
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
    allow_origins=["https://df978b75027c.ngrok-free.app", "https://yinsh.io", "yinsh.io", "demo.yinsh.io", "https://demo.yinsh.io"],
    # allow_origins=["*"],
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


class ProcessBoardStateRequest(BaseModel):
    """보드 상태 처리 요청 모델"""
    compressed_state: str
@app.post("/process-board-state", response_model=ProcessBoardStateResponse, tags=["Frontend"])
async def process_board_state(request: ProcessBoardStateRequest):
    """Stateless AI 보드 상태 처리 - 현재 상태만으로 최적 액션 추론"""
    try:
        request_start_time = time.time()
        frontend_service = get_frontend_service()
        
        print(f"\n🎯 새 AI 추론 요청 (요청 시간: {datetime.now().strftime('%H:%M:%S')})")
        
        # Stateless AI 추론 수행
        processed_state, ai_action, thinking_time = frontend_service.process_board_state(
            request.compressed_state
        )
        
        total_time = time.time() - request_start_time
        
        print(f"   📤 응답 전송 완료 (총 소요시간: {total_time:.3f}초)")
        
        # ProcessBoardStateResponse 모델 사용
        response = ProcessBoardStateResponse(
            compressed_state=processed_state,
            ai_action=ai_action,
            thinking_time=thinking_time,
            success=True
        )
        
        return response
        
    except ValueError as e:
        print(f"❌ 요청 검증 실패: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ AI 추론 실패: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="AI 추론 중 오류 발생")
    
@app.post("/test")
async def test():
    return {"message": "Hello, World!"}

# ============================================================================
# 예외 처리
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """HTTP 예외 처리"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """일반 예외 처리"""
    print(f"❌ 서버 예외 발생: {exc}")
    import traceback
    traceback.print_exc()
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "내부 서버 오류",
            "status_code": 500
        }
    )

# ============================================================================
# 서버 실행
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8018,
        # reload=True,
        log_level="info"
    ) 