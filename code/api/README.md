# YINSH 프론트엔드 API 서버

## 📋 개요

프론트엔드와의 통신을 위한 YINSH AI 추론 서버입니다. 압축된 보드 상태를 받아서 AI의 다음 행동을 반환합니다.

## 🚀 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements_online.txt
```

### 2. 서버 실행
```bash
python server.py
```

서버는 `http://localhost:8000`에서 실행됩니다.

## 📚 API 문서

서버 실행 후 `http://localhost:8000/docs`에서 자동 생성된 API 문서를 확인할 수 있습니다.

## 🔧 주요 기능

### 프론트엔드 호환 API
- `POST /process-board-state/` - 압축된 보드 상태 처리 및 AI 추론

### 헬스 체크
- `GET /` - 서버 상태 확인
- `GET /health` - 상세 헬스 체크

## 🎮 사용 예제

### Python 클라이언트
```python
from client import YinshAPIClient

client = YinshAPIClient("http://localhost:8000")

# AI 추론 요청
compressed_state = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="
result = client.process_board_state(compressed_state)

print(f"AI 행동: {result['ai_action']}")
print(f"추론 시간: {result['ai_thinking_time']}초")
```

### JavaScript 클라이언트
```javascript
// AI 추론 요청
const processBoardState = async (compressedState) => {
  const response = await fetch('/process-board-state/', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      compressed_state: compressedState
    })
  });
  return response.json();
};

// 사용 예제
const result = await processBoardState("AAAA...");
console.log("AI 행동:", result.ai_action);
console.log("추론 시간:", result.ai_thinking_time);
```

## 🔄 압축 상태 형식

프론트엔드와의 통신을 위해 32바이트 압축 형식을 사용합니다:

- **1비트**: 턴 정보 (0=white, 1=black)
- **255비트**: 85개 위치 × 3비트 (링/마커 상태)
- **Base64 인코딩**: 최종 32바이트를 base64로 인코딩

### 상태 값
- `0`: 빈 위치
- `1`: 흰색 링
- `2`: 검은색 링  
- `3`: 흰색 마커
- `4`: 검은색 마커

## 🤖 AI 추론

- **모델**: 기본 랜덤 초기화 (학습된 모델 사용 가능)
- **알고리즘**: MCTS (Monte Carlo Tree Search)
- **응답 시간**: 일반적으로 1-3초

## 📊 API 응답 형식

```json
{
  "success": true,
  "compressed_state": "AAAA...",
  "ai_action": {
    "action_type": "place_ring",
    "to_pos": {"q": 0, "r": 2}
  },
  "ai_thinking_time": 1.23,
  "message": "AI가 다음 행동을 선택했습니다"
}
```

## 🐳 Docker 배포

```bash
# 이미지 빌드
docker build -t yinsh-frontend .

# 컨테이너 실행
docker run -p 8000:8000 yinsh-frontend
```

## 🔧 개발

### 테스트 실행
```bash
python client.py
```

### 로그 확인
서버는 실시간으로 AI 추론 과정을 로그로 출력합니다.

## 📝 주의사항

1. **모델 파일**: `models/` 디렉토리에 학습된 모델 파일을 배치하면 더 강력한 AI 사용 가능
2. **메모리**: GPU 사용 시 충분한 VRAM이 필요합니다
3. **네트워크**: 프로덕션 배포 시 CORS 설정을 조정하세요

## 🆘 문제 해결

### 일반적인 오류

1. **모델 로드 실패**: 모델 파일 경로 확인
2. **CUDA 오류**: GPU 메모리 부족 시 CPU 모드 사용
3. **압축 데이터 오류**: 32바이트 base64 형식 확인

### 디버깅

```bash
# 상세 로그로 실행
python server.py --log-level debug

# 클라이언트 테스트
python client.py
``` 