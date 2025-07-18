# YINSH AlphaZero

YINSH 게임을 위한 AlphaZero 기반 강화학습 구현체입니다. PyTorch를 사용하여 구현되었습니다.

## 🎯 프로젝트 개요

이 프로젝트는 DeepMind의 AlphaZero 알고리즘을 YINSH 보드게임에 적용한 것입니다. YINSH는 추상 전략 보드게임으로, 체스와는 다른 고유한 규칙을 가지고 있습니다.

### 주요 특징
- **Monte Carlo Tree Search (MCTS)** 기반 의사결정
- **Deep Neural Network**를 통한 정책 및 가치 예측
- **Self-Play**를 통한 지속적인 학습
- **PyTorch** 기반 구현

## 🏗️ 아키텍처

### 핵심 컴포넌트

```
yinsh/
├── env.py          # YINSH 게임 환경 (규칙, 상태 관리)
├── model.py        # 신경망 모델 (정책 + 가치 네트워크)
├── mcts.py         # MCTS 알고리즘
├── node.py         # MCTS 노드 클래스
├── agent.py        # AI 에이전트 (MCTS + 신경망)
├── mapper.py       # 액션 매핑 (게임 액션 ↔ 신경망 출력)
├── game.py         # 게임 로직 및 상태 관리
├── config.py       # 설정 및 하이퍼파라미터
└── utils.py        # 유틸리티 함수들
```

### 신경망 아키텍처

```
Input: YINSH 게임 상태 (19x19 보드)
├── Convolutional Layers (ResNet 스타일)
├── Policy Head (액션 확률 분포)
└── Value Head (게임 상태 가치)
```

### MCTS 프로세스

1. **Selection**: UCB1 공식을 사용하여 최적의 노드 선택
2. **Expansion**: 새로운 자식 노드 생성
3. **Simulation**: 신경망을 사용한 정책 및 가치 예측
4. **Backpropagation**: 결과를 루트까지 역전파

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

### 2. 기본 실행

```bash
# 빠른 시작 (기본 설정으로 학습)
python quick_start.py

# 병렬 AlphaZero 파이프라인 (권장) ⚡
python scripts/run_alphazero.py --quick

# 디버그 모드 (상세한 게임 진행 확인)
python debug_game.py

# 병렬 성능 테스트
python test_parallel.py
```

## 📚 학습 가이드

### 학습 프로세스

1. **Self-Play**: 현재 모델로 게임 진행 및 데이터 생성
2. **Training**: 생성된 데이터로 신경망 학습
3. **Evaluation**: 새로운 모델 성능 평가
4. **Iteration**: 1-3 과정 반복

### 학습 스크립트

```bash
# Self-Play 실행 (순차)
python scripts/selfplay.py --games 100 --model models/best_model.pt

# Self-Play 실행 (병렬) ⚡ 새로운 기능!
python scripts/selfplay.py --games 100 --parallel --workers 6

# 모델 학습
python scripts/train.py --data memory/ --epochs 10 --batch-size 32

# 모델 평가
python scripts/evaluate.py --candidate models/candidate.pt --best models/best.pt
```

### 주요 하이퍼파라미터

```python
# config.py에서 설정 가능
MCTS_SIMULATIONS = 800      # MCTS 시뮬레이션 횟수
LEARNING_RATE = 0.001       # 학습률
BATCH_SIZE = 32            # 배치 크기
EPOCHS = 10                # 에포크 수
SELF_PLAY_GAMES = 100      # Self-Play 게임 수
```

### 병렬 처리 설정 ⚡

```bash
# 병렬 Self-Play 옵션
--parallel              # 병렬 모드 활성화
--workers 6             # 워커 프로세스 수 (기본: CPU 코어 * 0.75)

# 사용 예시
python scripts/selfplay.py --games 50 --parallel --workers 4
python scripts/run_alphazero.py --demo  # 기본적으로 병렬 사용
```

## 🎮 Self-Play 시스템

### Self-Play 프로세스

1. **게임 초기화**: 빈 YINSH 보드로 시작
2. **턴 진행**: 
   - 링 배치 단계 (5개 링 배치)
   - 링 이동 단계 (링 이동 + 마커 뒤집기)
3. **게임 종료**: 3개 링 제거 또는 더 이상 유효한 이동이 없을 때
4. **데이터 저장**: 게임 상태, 액션, 결과를 메모리에 저장

### 데이터 형식

```python
training_data = {
    'state': game_state,           # 게임 상태 (19x19 보드)
    'policy': action_probabilities, # 액션 확률 분포
    'value': game_result,          # 게임 결과 (-1, 0, 1)
    'actions': taken_actions       # 실제 선택된 액션들
}
```

## 🎯 YINSH 게임 규칙

### 기본 규칙
- **보드**: 19x19 육각형 격자
- **목표**: 상대방보다 먼저 3개의 링을 제거
- **게임 단계**:
  1. **링 배치**: 각 플레이어가 5개의 링을 보드에 배치
  2. **링 이동**: 링을 이동하고 경로의 마커를 뒤집기

### 특수 규칙
- **마커 뒤집기**: 링이 이동할 때 경로의 모든 마커를 뒤집음
- **링 제거**: 5개의 마커가 연속으로 같은 색이 되면 링 제거
- **블로킹**: 링이 다른 링을 건너뛸 수 없음

## 📁 프로젝트 구조

```
chess-deep-rl/code/
├── README.md              # 이 파일
├── requirements.txt       # Python 의존성
├── quick_start.py        # 빠른 시작 스크립트
├── debug_game.py         # 디버그용 게임 실행
├── yinsh/                # 핵심 YINSH 패키지
│   ├── __init__.py
│   ├── env.py            # 게임 환경
│   ├── model.py          # 신경망 모델
│   ├── mcts.py           # MCTS 알고리즘
│   ├── agent.py          # AI 에이전트
│   ├── mapper.py         # 액션 매핑
│   ├── game.py           # 게임 로직
│   ├── config.py         # 설정
│   └── utils.py          # 유틸리티
├── scripts/              # 실행 스크립트
│   ├── train.py          # 학습 스크립트
│   └── selfplay.py       # Self-Play 스크립트
├── models/               # 학습된 모델 저장
├── memory/               # Self-Play 데이터 저장
└── plots/                # 학습 그래프 저장
```

## 🔧 문제 해결

### 일반적인 문제들

1. **Import 오류**: `pip install -r requirements.txt` 실행
2. **CUDA 오류**: CPU 모드로 실행하거나 CUDA 설치 확인
3. **메모리 부족**: 배치 크기 줄이기
4. **학습 데이터 없음**: Self-Play 먼저 실행

### 디버깅

```bash
# 상세한 게임 진행 확인
python debug_game.py

# 학습 과정 상세 확인
python quick_start.py --verbose
```

## 📊 학습 모니터링

### 학습 과정 확인
- `memory/` 폴더: Self-Play 데이터 저장
- `models/` 폴더: 학습된 모델 저장
- `plots/` 폴더: 학습 그래프 저장

## 🎮 게임 규칙 요약

### YINSH 기본 규칙
1. **링 배치**: 각 플레이어가 5개의 링을 보드에 배치
2. **링 이동**: 링을 이동하고 경로의 마커를 뒤집기
3. **링 제거**: 5개의 마커가 연속으로 같은 색이 되면 링 제거
4. **승리 조건**: 상대방보다 먼저 3개의 링을 제거

### 보드 표시
- 🔲: 빈 공간 (유효한 위치)
- 🔳: 빈 공간 (유효하지 않은 위치)
- ⚪: 흰색 링
- ⚫: 검은색 링
- 🔵: 흰색 마커
- 🔴: 검은색 마커

## 📈 성능 최적화

### 학습 최적화 팁

1. **데이터 품질**: 충분한 Self-Play 게임 수 확보
2. **모델 크기**: 하드웨어에 맞는 모델 크기 선택
3. **학습률**: 점진적으로 학습률 조정
4. **정규화**: 과적합 방지를 위한 정규화 적용

### 하드웨어 권장사항

- **CPU**: 멀티코어 프로세서 (8코어 이상 권장)
- **RAM**: 16GB 이상
- **GPU**: CUDA 지원 GPU (선택사항, 학습 속도 향상)

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 문의

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 생성해 주세요. 