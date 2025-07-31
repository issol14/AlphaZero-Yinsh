# YINSH AlphaZero (간소화 버전)

YINSH 게임을 위한 AlphaZero 기반 강화학습 구현체입니다. **간소화된 버전**으로, 5개 연속 마커가 생기면 즉시 승리하는 규칙으로 변경되었습니다.

## 🎯 프로젝트 개요

이 프로젝트는 DeepMind의 AlphaZero 알고리즘을 YINSH 보드게임에 적용한 것입니다. **간소화된 버전**에서는 다음과 같은 변경사항이 있습니다:

### 주요 변경사항
- **승리 조건**: 5개 연속 마커가 생기면 즉시 승리 (기존: 3개 링 제거)
- **게임 단계**: 링 배치 단계 제거, 메인 게임만 진행
- **액션 공간**: 링 이동만 고려 (라인 제거 액션 제거)
- **신경망 입력**: 6채널로 간소화
- **액션 공간**: 1000차원으로 축소

### 신경망 입력 채널 (6채널)

| 채널 번호 | 의미 | 값 |
| ----- | ---------------- | ------------------------------------ |
| 0 | 현재 플레이어의 링 위치 | 1.0 (링 위치), 0.0 (그 외) |
| 1 | 현재 플레이어의 마커 위치 | 1.0 (마커 위치), 0.0 (그 외) |
| 2 | 상대 플레이어의 링 위치 | 1.0 (링 위치), 0.0 (그 외) |
| 3 | 상대 플레이어의 마커 위치 | 1.0 (마커 위치), 0.0 (그 외) |
| 4 | 유효한 보드 위치 | 1.0 (유효 위치), 0.0 (무효 위치) |
| 5 | 현재 플레이어 표시 | 1.0 (흰색), 0.0 (검은색) |

## 🏗️ 아키텍처

### 핵심 컴포넌트

```
yinsh/
├── env.py          # YINSH 게임 환경 (간소화된 규칙)
├── model.py        # 신경망 모델 (6채널 입력, 1000차원 출력)
├── mcts.py         # MCTS 알고리즘
├── node.py         # MCTS 노드 클래스
├── agent.py        # AI 에이전트 (MCTS + 신경망)
├── mapper.py       # 액션 매핑 (링 이동만)
├── game.py         # 게임 로직 및 상태 관리
├── config.py       # 설정 및 하이퍼파라미터
└── utils.py        # 유틸리티 함수들
```

### 신경망 아키텍처

```
Input: YINSH 게임 상태 (6채널 x 11x11 보드)
├── Convolutional Layers (ResNet 스타일)
├── Policy Head (1000차원 액션 확률 분포)
└── Value Head (게임 상태 가치)
```

### 게임 규칙 (간소화)

1. **게임 시작**: 링 10개가 랜덤하게 배치된 상태에서 시작
2. **턴 진행**: 플레이어가 번갈아가며 링을 이동
3. **마커 뒤집기**: 링 이동 경로의 모든 마커 뒤집기
4. **승리 조건**: 5개 연속 마커가 생기면 즉시 승리
5. **마커 소진**: 마커가 소진되면 더 많은 마커를 가진 플레이어 승리

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 의존성 설치
pip install -r requirements.txt
```

### 2. 간소화 버전 테스트

```bash
# 간소화된 버전 테스트
python test_simplified.py

# 빠른 시작 (기본 설정으로 학습)
python quick_start.py

# 병렬 AlphaZero 파이프라인 (권장) ⚡
python scripts/run_alphazero.py --quick
```

### 3. Selfplay 실행

```bash
# 빠른 selfplay (간소화된 버전)
python scripts/selfplay.py --model models/best_model.pt --games 100 --mcts-sims 200 --fast

# 초고속 selfplay (매우 적은 시뮬레이션, 로깅 없음)
python scripts/selfplay.py --model models/best_model.pt --games 50 --mcts-sims 100 --silent
```

## 📊 성능 최적화

### 병렬 처리
- **Self-Play 병렬화**: CPU 코어별 워커 분산
- **MCTS 병렬화**: 멀티스레딩으로 탐색 속도 향상
- **배치 처리**: GPU 메모리 효율적 활용

### 메모리 최적화
- **동적 액션 생성**: 유효한 액션만 생성
- **상태 캐싱**: 중복 계산 방지
- **효율적 매핑**: valid_points만 사용

### 학습 최적화
- **AlphaZero 논문 기반**: 검증된 하이퍼파라미터
- **정규화**: 드롭아웃, L2 정규화
- **학습률 스케줄링**: 점진적 감소

## 🎮 게임 플레이

### 사람 vs AI

```bash
# 사람 vs AI 게임
python human_vs_ai.py --model models/best_model.pt
```

### AI vs AI

```bash
# AI vs AI 게임 시각화
python scripts/play_game.py --model1 models/model1.pt --model2 models/model2.pt
```

## 📈 학습 모니터링

### TensorBoard 로그

```bash
# TensorBoard 실행
tensorboard --logdir logs/

# 브라우저에서 확인
# http://localhost:6006
```

### 학습 곡선

```bash
# 손실 그래프 생성
python scripts/plot_loss.py

# 성능 평가 그래프
python scripts/plot_evaluation.py
```

## 🔧 설정 및 하이퍼파라미터

### 주요 설정 (config.py)

```python
# 신경망 설정
INPUT_SHAPE = (6, 11, 11)  # 6채널 입력
POLICY_OUTPUT_SIZE = 1000   # 1000차원 액션 공간

# MCTS 설정
MCTS_SIMULATIONS = 800      # 시뮬레이션 수
CPUCT = 2.5                 # UCB 탐색 상수

# 학습 설정
BATCH_SIZE = 512            # 배치 크기
LEARNING_RATE = 0.002       # 학습률
MEMORY_SIZE = 1000000       # 메모리 크기 (1M 포지션)
```

## 🧪 테스트

### 단위 테스트

```bash
# 환경 테스트
python -m pytest tests/test_env.py

# 모델 테스트
python -m pytest tests/test_model.py

# MCTS 테스트
python -m pytest tests/test_mcts.py
```

### 통합 테스트

```bash
# 전체 시스템 테스트
python test_simplified.py

# 성능 벤치마크
python scripts/benchmark.py
```

## 📝 변경사항

### 간소화된 버전의 주요 변경사항

1. **게임 규칙 간소화**
   - 링 배치 단계 제거
   - 라인 제거 단계 제거
   - 5개 연속 마커 즉시 승리

2. **신경망 간소화**
   - 입력 채널: 15 → 6
   - 액션 공간: 4000 → 1000
   - 잔차 블록: 24 → 20

3. **액션 공간 최적화**
   - 링 이동 액션만 고려
   - 동적 액션 생성 최적화
   - 메모리 사용량 감소

## 🤝 기여

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 🙏 감사의 말

- DeepMind의 AlphaZero 논문
- YINSH 보드게임 규칙
- PyTorch 커뮤니티
- 모든 기여자들

---

**간소화된 버전**은 학습 속도와 메모리 효율성을 크게 향상시켰으며, 복잡한 게임 규칙을 단순화하여 AlphaZero 알고리즘의 핵심을 더 명확하게 보여줍니다. 