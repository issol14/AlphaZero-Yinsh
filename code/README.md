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

## 📚 AlphaZero 학습 완전 가이드

### 학습 프로세스

1. **Self-Play**: 현재 모델로 게임 진행 및 데이터 생성
2. **Training**: 생성된 데이터로 신경망 학습
3. **Evaluation**: 새로운 모델 성능 평가
4. **Iteration**: 1-3 과정 반복

## 🚀 **실행 스크립트 완전 가이드**

### **1. 간편 실행 스크립트 (권장 시작점)**

#### **`scripts/run_alphazero.py`** - 올인원 런처
```bash
# 📌 기본 사용법
python scripts/run_alphazero.py [모드]

# 📌 사용 가능한 모드들
--demo        # 데모 모드 (빠른 체험)
--quick       # 빠른 모드 (적당한 규모) 
--full        # 풀 모드 (완전한 학습)
--custom      # 커스텀 모드 (사용자 설정)
--status      # 현재 시스템 상태 보기
--evaluate    # 단일 모델 평가
```

**각 모드별 자동 설정:**
- **데모 모드**: 5 iterations, 10게임/iter, 4 workers, 2시간 제한
- **퀵 모드**: 20 iterations, 50게임/iter, 6 workers, 8시간 제한  
- **풀 모드**: 100 iterations, 100게임/iter, 8 workers, 24시간 제한

### **2. 셀프플레이 데이터 생성**

#### **`scripts/selfplay.py`** - 훈련 데이터 생성
```bash
python scripts/selfplay.py [옵션들]

# 📌 필수 옵션
--games INT              # 플레이할 게임 수 (기본: 50)

# 📌 모델 관련
--model PATH             # 로드할 모델 경로 (기본: None=랜덤)
--no-mcts               # MCTS 비활성화 (직접 예측)
--mcts-sims INT         # MCTS 시뮬레이션 수 (기본: 800)

# 📌 출력 관련  
--output DIR            # 출력 디렉토리 (기본: "memory")
--show-board           # 각 턴마다 보드 상태 출력

# 📌 병렬 처리 (NEW!)
--parallel             # 병렬 실행 모드 활성화
--workers INT          # 병렬 워커 수 (기본: 1=순차실행)
```

**사용 예시:**
```bash
# 순차 실행 (기본)
python scripts/selfplay.py --games 50 --mcts-sims 800

# 병렬 실행 (6배 빠름!)
python scripts/selfplay.py --games 50 --parallel --workers 6

# 기존 모델 사용
python scripts/selfplay.py --games 100 --model models/best_model.pt --parallel --workers 8

# 빠른 테스트 (MCTS 적게)
python scripts/selfplay.py --games 20 --mcts-sims 200 --parallel --workers 4
```

### **3. 모델 훈련**

#### **`scripts/train.py`** - 신경망 훈련
```bash
python scripts/train.py [옵션들]

# 📌 데이터 관련
--data DIR              # 훈련 데이터 폴더 (기본: "memory")
--model PATH            # 이어서 훈련할 모델 경로 (기본: None=새 모델)

# 📌 출력 관련
--output DIR            # 모델 저장 폴더 (기본: "models")

# 📌 훈련 하이퍼파라미터
--epochs INT            # 훈련 에포크 수 (기본: 10)
--batch-size INT        # 배치 크기 (기본: 32)
--lr FLOAT              # 학습률 (기본: 0.001)
```

**사용 예시:**
```bash
# 기본 훈련
python scripts/train.py --data memory --epochs 10

# 기존 모델에서 이어서 훈련
python scripts/train.py --data memory --model models/checkpoint.pt --epochs 5

# 하이퍼파라미터 조정
python scripts/train.py --data memory --epochs 20 --batch-size 64 --lr 0.0005
```

### **4. 모델 평가**

#### **`scripts/evaluate.py`** - 모델 성능 평가
```bash
python scripts/evaluate.py [옵션들]

# 📌 필수 옵션
--candidate PATH        # 평가할 새 모델 경로 (필수)

# 📌 비교 모델
--best PATH             # 기존 best 모델 경로 (기본: None=랜덤 baseline)

# 📌 평가 설정
--games INT             # 평가 게임 수 (기본: 100)
--threshold FLOAT       # 새 모델 채택 최소 승률 (기본: 0.55)
--mcts-sims INT         # MCTS 시뮬레이션 수 (기본: 400)

# 📌 기타
--no-save              # 결과 파일 저장 안함
```

**사용 예시:**
```bash
# 새 모델 vs 기존 모델
python scripts/evaluate.py --candidate models/new_model.pt --best models/best_model.pt

# 새 모델 vs 랜덤 (baseline 테스트)
python scripts/evaluate.py --candidate models/new_model.pt --games 50

# 빠른 평가 (적은 게임, 적은 MCTS)
python scripts/evaluate.py --candidate models/test.pt --games 20 --mcts-sims 200
```

### **5. 완전 자동화 파이프라인**

#### **`scripts/pipeline.py`** - 완전 AlphaZero 루프
```bash
python scripts/pipeline.py [옵션들]

# 📌 파이프라인 제어
--iterations INT        # 최대 iteration 수 (기본: 100)
--max-hours INT         # 최대 실행 시간 (기본: 24시간)

# 📌 셀프플레이 설정
--selfplay-games INT    # Iteration당 게임 수 (기본: 100)
--selfplay-mcts-sims INT # MCTS 시뮬레이션 수 (기본: 800)

# 📌 훈련 설정
--training-epochs INT   # 훈련 에포크 수 (기본: 10)
--training-batch-size INT # 배치 크기 (기본: 32)
--training-lr FLOAT     # 학습률 (기본: 0.001)
--continue-training     # 기존 모델에서 이어서 훈련

# 📌 평가 설정
--evaluation-games INT  # 모델 평가 게임 수 (기본: 100)
--evaluation-threshold FLOAT # 승률 기준 (기본: 0.55)
--evaluation-mcts-sims INT # 평가 MCTS 시뮬레이션 (기본: 400)

# 📌 디렉토리 설정
--models-dir DIR        # 모델 저장 디렉토리 (기본: "models")
--data-dir DIR          # 훈련 데이터 임시 디렉토리 (기본: "pipeline_data")
--logs-dir DIR          # 로그 저장 디렉토리 (기본: "pipeline_logs")

# 📌 기타 설정
--cleanup-data         # Iteration 완료 후 임시 데이터 삭제

# 📌 병렬 처리 (NEW!)
--parallel             # 병렬 셀프플레이 활성화
--workers INT          # 병렬 워커 수 (기본: 1)
```

**사용 예시:**
```bash
# 기본 파이프라인 (순차 실행)
python scripts/pipeline.py --iterations 20 --max-hours 8

# 병렬 파이프라인 (권장)
python scripts/pipeline.py \
    --iterations 20 \
    --selfplay-games 50 \
    --parallel \
    --workers 6 \
    --max-hours 8 \
    --cleanup-data

# 빠른 프로토타이핑
python scripts/pipeline.py \
    --iterations 5 \
    --selfplay-games 20 \
    --selfplay-mcts-sims 400 \
    --training-epochs 5 \
    --parallel \
    --workers 4

# 풀 스케일 훈련
python scripts/pipeline.py \
    --iterations 100 \
    --selfplay-games 100 \
    --selfplay-mcts-sims 800 \
    --training-epochs 10 \
    --evaluation-games 100 \
    --parallel \
    --workers 8 \
    --max-hours 24 \
    --continue-training \
    --cleanup-data
```

## 🎯 **권장 실행 시퀀스**

### **초보자용 (처음 시작)**
```bash
# 1. 빠른 데모 체험
python scripts/run_alphazero.py --demo

# 2. 성능 테스트
python test_parallel.py

# 3. 적당한 규모 학습  
python scripts/run_alphazero.py --quick
```

### **고급 사용자용 (커스터마이징)**
```bash
# 1. 셀프플레이 데이터 생성
python scripts/selfplay.py --games 100 --parallel --workers 8 --output custom_data

# 2. 모델 훈련
python scripts/train.py --data custom_data --epochs 15 --batch-size 64

# 3. 모델 평가
python scripts/evaluate.py --candidate models/trained_model.pt --games 100

# 4. 완전 파이프라인 (커스텀 설정)
python scripts/pipeline.py \
    --iterations 50 \
    --selfplay-games 80 \
    --parallel \
    --workers 6 \
    --training-epochs 12 \
    --evaluation-threshold 0.60
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

## ⚡ **성능 최적화 가이드**

### **병렬 처리 최적화:**
```bash
# CPU 코어별 권장 워커 수:
# 4코어: --workers 3
# 8코어: --workers 6 
# 16코어: --workers 12
# 24코어: --workers 18

# 사용 예시
python scripts/selfplay.py --games 50 --parallel --workers 4
python scripts/run_alphazero.py --demo  # 기본적으로 병렬 사용
```

### **메모리/시간 최적화:**
- **빠른 테스트**: `--mcts-sims 200`, `--games 20`
- **균형잡힌 설정**: `--mcts-sims 400`, `--games 50` 
- **최고 품질**: `--mcts-sims 800`, `--games 100`

### **GPU 메모리 최적화:**
- **작은 GPU (4-8GB)**: `--batch-size 16`, `--workers 4`
- **중간 GPU (8-16GB)**: `--batch-size 32`, `--workers 6`
- **큰 GPU (16GB+)**: `--batch-size 64`, `--workers 8`

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
├── test_parallel.py      # 병렬 성능 테스트
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
│   ├── run_alphazero.py  # 올인원 런처 (권장)
│   ├── pipeline.py       # 완전 자동화 파이프라인
│   ├── selfplay.py       # Self-Play (병렬 지원)
│   ├── train.py          # 모델 훈련
│   └── evaluate.py       # 모델 평가
├── models/               # 학습된 모델 저장
├── memory/               # Self-Play 데이터 저장
├── pipeline_data/        # 파이프라인 임시 데이터
├── pipeline_logs/        # 파이프라인 로그
└── plots/                # 학습 그래프 저장
```

## 🗂️ **결과 파일 구조**

실행 후 생성되는 파일들:
```
models/
├── best_model.pt              # 현재 최고 모델
├── best_model_info.json       # 모델 메타데이터
└── history/                   # 모델 히스토리

memory/ (또는 지정 디렉토리)
├── game_0.pt                  # 게임별 훈련 데이터
├── game_1.pt
└── ...

pipeline_logs/
├── pipeline_YYYYMMDD_HHMMSS.log # 파이프라인 실행 로그

evaluation_results/
├── eval_YYYYMMDD_HHMMSS.json  # 평가 결과
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