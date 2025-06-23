# Yinsh AI 웹앱 연동 가이드

이 가이드는 AlphaYinsh AI를 웹애플리케이션과 연동하는 방법을 설명합니다.

## 설치 및 설정

### 1. 의존성 설치

```bash
cd AlphaYinsh
pip install -r requirements.txt
```

### 2. AI 서버 실행

```bash
# 기본 실행 (모델 없이)
python web_server.py

# 모델과 함께 실행
python web_server.py --model models/model.h5

# 커스텀 호스트/포트로 실행
python web_server.py --host 0.0.0.0 --port 8000

# 디버그 모드로 실행
python web_server.py --debug
```

서버가 실행되면 `http://localhost:5000`에서 API에 접근할 수 있습니다.

## API 엔드포인트

### 기본 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/health` | 서버 상태 확인 |
| POST | `/api/new_game` | 새 게임 시작 |
| GET | `/api/game_state/<game_id>` | 게임 상태 조회 |
| POST | `/api/make_move` | 플레이어 수 적용 |
| POST | `/api/ai_move` | AI 수 요청 |
| POST | `/api/update_game_state` | 게임 상태 업데이트 |
| POST | `/api/get_valid_moves` | 유효한 수 조회 |
| POST | `/api/analyze_position` | 포지션 분석 |

### 상세 API 문서

#### 1. 서버 상태 확인
```javascript
GET /api/health

// 응답
{
    "status": "healthy",
    "agent_loaded": true,
    "version": "1.0.0"
}
```

#### 2. 새 게임 시작
```javascript
POST /api/new_game
Content-Type: application/json

{
    "game_id": "my_game_123"  // 선택사항
}

// 응답
{
    "success": true,
    "game_id": "my_game_123",
    "game_state": {
        "board": [...],
        "current_player": 0,
        "game_phase": "placement",
        "rings_placed": [0, 0],
        "rings_removed": [0, 0],
        "move_count": 0,
        "is_game_over": false,
        "winner": null
    }
}
```

#### 3. AI 수 요청
```javascript
POST /api/ai_move
Content-Type: application/json

{
    "game_id": "my_game_123",
    "simulations": 100  // 선택사항, 기본값: 100
}

// 응답
{
    "success": true,
    "ai_move": "PLACE_RING (5, 5)",
    "confidence": 0.85,
    "game_state": {...},
    "move_info": {
        "visits": 42,
        "value": 0.12,
        "prior": 0.05
    }
}
```

#### 4. 포지션 분석
```javascript
POST /api/analyze_position
Content-Type: application/json

{
    "game_id": "my_game_123",
    "depth": 50  // 선택사항, 시뮬레이션 수
}

// 응답
{
    "success": true,
    "position_evaluation": 0.15,
    "total_simulations": 50,
    "top_moves": [
        {
            "move": "PLACE_RING (5, 5)",
            "visits": 25,
            "win_rate": 0.6,
            "probability": 0.5,
            "prior": 0.1
        },
        // ... 최대 5개 수
    ]
}
```

## JavaScript 클라이언트 사용법

### 1. 클라이언트 초기화

```javascript
// 브라우저에서
const yinshAI = new YinshAIClient('http://localhost:5000');

// Node.js에서
const YinshAIClient = require('./web_client_example.js');
const yinshAI = new YinshAIClient('http://localhost:5000');
```

### 2. 기본 사용 예제

```javascript
async function playGame() {
    // 서버 상태 확인
    const health = await yinshAI.checkHealth();
    if (health.status !== 'healthy') {
        console.error('AI 서버가 준비되지 않았습니다');
        return;
    }

    // 새 게임 시작
    const newGame = await yinshAI.newGame();
    if (!newGame.success) {
        console.error('게임 시작 실패:', newGame.error);
        return;
    }

    console.log('게임 시작됨:', newGame.game_state);

    // 게임 루프
    while (true) {
        const gameState = await yinshAI.getGameState();
        
        if (gameState.game_state.is_game_over) {
            console.log('게임 종료! 승자:', gameState.game_state.winner);
            break;
        }

        // 현재 플레이어가 AI인 경우
        if (gameState.game_state.current_player === 1) {
            const aiMove = await yinshAI.getAIMove(100);
            if (aiMove.success) {
                console.log('AI 수:', aiMove.ai_move);
            }
        } else {
            // 인간 플레이어 차례
            console.log('당신의 차례입니다');
            // 웹앱에서 사용자 입력 처리
        }
    }
}
```

### 3. 웹앱과의 연동

```javascript
// React 컴포넌트 예제
import React, { useState, useEffect } from 'react';

function YinshGame() {
    const [yinshAI] = useState(new YinshAIClient('http://localhost:5000'));
    const [gameState, setGameState] = useState(null);
    const [isAIThinking, setIsAIThinking] = useState(false);

    useEffect(() => {
        initializeGame();
    }, []);

    const initializeGame = async () => {
        const result = await yinshAI.newGame();
        if (result.success) {
            setGameState(result.game_state);
        }
    };

    const handlePlayerMove = async (move) => {
        const result = await yinshAI.makeMove(move);
        if (result.success) {
            setGameState(result.game_state);
            
            // AI 차례라면 AI 수 요청
            if (result.game_state.current_player === 1) {
                await getAIMove();
            }
        }
    };

    const getAIMove = async () => {
        setIsAIThinking(true);
        const result = await yinshAI.getAIMove(100);
        if (result.success) {
            setGameState(result.game_state);
        }
        setIsAIThinking(false);
    };

    const analyzePosition = async () => {
        const analysis = await yinshAI.analyzePosition(50);
        if (analysis.success) {
            console.log('포지션 분석:', analysis);
            // 분석 결과를 UI에 표시
        }
    };

    return (
        <div>
            {/* 게임 보드 렌더링 */}
            {gameState && <YinshBoard gameState={gameState} onMove={handlePlayerMove} />}
            
            {/* AI 상태 표시 */}
            {isAIThinking && <div>AI가 수를 계산중입니다...</div>}
            
            {/* 분석 버튼 */}
            <button onClick={analyzePosition}>포지션 분석</button>
        </div>
    );
}
```

## 게임 상태 형식

```javascript
{
    "board": [          // 11x11x4 배열 (rings_p1, rings_p2, markers_p1, markers_p2)
        [...],
        // 11개 행
    ],
    "current_player": 0,    // 0: 플레이어1, 1: 플레이어2
    "game_phase": "placement",  // "placement" 또는 "movement"
    "rings_placed": [2, 1],     // 각 플레이어가 놓은 링 수
    "rings_removed": [0, 0],    // 각 플레이어가 제거한 링 수
    "move_count": 5,            // 총 수 카운트
    "is_game_over": false,      // 게임 종료 여부
    "winner": null              // 승자 (1, -1, 0 또는 null)
}
```

## 이동 형식

현재 구현에서는 이동을 문자열로 표현합니다:

```javascript
// 링 배치
"PLACE_RING (row, col)"

// 링 이동
"MOVE_RING (from_row, from_col) to (to_row, to_col)"

// 마커 제거
"REMOVE_MARKERS [(row1, col1), (row2, col2), ...]"
```

## 오류 처리

모든 API 응답은 다음 형식을 따릅니다:

```javascript
// 성공
{
    "success": true,
    "data": {...}
}

// 실패
{
    "success": false,
    "error": "오류 메시지"
}
```

## 성능 최적화

1. **시뮬레이션 수 조절**: AI 수 요청 시 `simulations` 매개변수로 계산 시간 조절
2. **게임 세션 관리**: 동일한 `game_id`를 사용하여 게임 상태 유지
3. **비동기 처리**: AI 계산 중 UI 블로킹을 방지하기 위해 비동기 처리

## 문제 해결

### 1. CORS 오류
웹앱이 다른 도메인에서 실행되는 경우, 서버에서 CORS가 활성화되어 있는지 확인하세요.

### 2. 연결 오류
- AI 서버가 실행 중인지 확인
- 방화벽 설정 확인
- 포트 번호가 올바른지 확인

### 3. 성능 문제
- 시뮬레이션 수를 줄여보세요 (기본값: 100)
- 서버 리소스 확인
- 모델 로딩 여부 확인

## 추가 기능

향후 추가될 수 있는 기능들:

- WebSocket 지원 (실시간 통신)
- 게임 기록 저장/불러오기
- 다양한 AI 난이도 설정
- 멀티플레이어 지원
- 게임 리플레이 기능

## 지원

문제가 발생하거나 추가 기능이 필요한 경우, GitHub 이슈를 생성해주세요. 