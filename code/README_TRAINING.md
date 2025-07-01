# YINSH AlphaZero Training Guide

## 🚀 Quick Start

### 1. 환경 설정
```bash
# 가상환경 활성화
cd chess-deep-rl/code
yinsh_env\Scripts\activate  # Windows
# source yinsh_env/bin/activate  # Linux/Mac

# 필요한 패키지 설치
pip install -r requirements.txt
```

### 2. 빠른 시작 (권장)
```bash
# 기본 설정으로 즉시 훈련 시작
python quick_start.py
```

### 3. 전체 훈련
```bash
# 완전한 훈련 (시간이 오래 걸림)
python main.py --iterations 100 --games-per-iteration 50 --epochs 10
```

## 📋 Training Scripts

### `quick_start.py` - 빠른 시작
- **목적**: 즉시 훈련을 시작하고 싶을 때
- **설정**: 적은 반복과 게임 수로 빠른 결과 확인
- **사용법**: `python quick_start.py`

### `main.py` - 완전한 훈련
- **목적**: 완전한 AlphaZero 훈련
- **설정**: 모든 하이퍼파라미터 조정 가능
- **사용법**: `python main.py [options]`

### `demo.py` - 모델 데모
- **목적**: 훈련된 모델 테스트 및 시연
- **사용법**: `python demo.py --model path/to/model.pt --games 5`

## ⚙️ Configuration Options

### main.py 옵션
```bash
python main.py \
    --iterations 100 \           # 훈련 반복 횟수
    --games-per-iteration 50 \   # 반복당 게임 수
    --epochs 10 \                # 훈련 에포크 수
    --batch-size 32 \            # 배치 크기
    --lr 0.001 \                 # 학습률
    --eval-games 20 \            # 평가 게임 수
    --output-dir training_output \ # 출력 디렉토리
    --continue-from model.pt     # 체크포인트에서 재시작
```

### demo.py 옵션
```bash
python demo.py \
    --model model.pt \           # 모델 파일 경로
    --games 5 \                  # 데모 게임 수
    --max-turns 50 \             # 게임당 최대 턴 수
    --mcts-sims 100 \            # MCTS 시뮬레이션 수
    --output demo_output         # 출력 디렉토리
```

## 📊 Training Process

### 1. Self-Play Phase
- 현재 모델로 자기 자신과 게임
- MCTS를 사용한 액션 선택
- 게임 결과를 훈련 데이터로 변환

### 2. Training Phase
- 수집된 데이터로 신경망 훈련
- Policy Loss + Value Loss
- Adam 옵티마이저 사용

### 3. Evaluation Phase
- 훈련된 모델 vs 랜덤 에이전트
- 승률 측정으로 성능 평가

### 4. Iteration
- 위 과정을 반복하여 모델 개선

## 📁 Output Structure

```
training_output/
├── models/              # 훈련된 모델들
│   ├── model_iter_1.pt
│   ├── model_iter_2.pt
│   └── ...
├── data/                # 훈련 데이터
│   ├── iter_1/
│   ├── iter_2/
│   └── ...
├── logs/                # 로그 파일들
│   └── training_*.log
└── training_history.json # 훈련 히스토리
```

## 🔧 Advanced Usage

### 체크포인트에서 재시작
```bash
python main.py --continue-from training_output/models/model_iter_50.pt
```

### 하이퍼파라미터 튜닝
```bash
# 높은 품질 훈련
python main.py \
    --iterations 200 \
    --games-per-iteration 100 \
    --epochs 20 \
    --batch-size 64 \
    --lr 0.0005

# 빠른 실험
python main.py \
    --iterations 10 \
    --games-per-iteration 20 \
    --epochs 5 \
    --batch-size 16 \
    --lr 0.002
```

### GPU 사용
```bash
# CUDA가 설치된 경우 자동으로 GPU 사용
# CPU 강제 사용
export CUDA_VISIBLE_DEVICES=""
python main.py
```

## 📈 Monitoring Training

### 로그 확인
```bash
# 실시간 로그 모니터링
tail -f training_output/logs/training_*.log
```

### 성능 추적
```bash
# 훈련 히스토리 확인
cat training_output/training_history.json | jq '.[] | {iteration, evaluation}'
```

### 모델 테스트
```bash
# 특정 반복의 모델 테스트
python demo.py --model training_output/models/model_iter_50.pt --games 10
```

## 🎯 Expected Results

### 초기 단계 (1-10 반복)
- 랜덤 수준의 플레이
- 승률: 40-60%

### 중간 단계 (10-50 반복)
- 기본 전략 학습
- 승률: 60-80%

### 고급 단계 (50+ 반복)
- 복잡한 전략 학습
- 승률: 80-95%

## ⚠️ Troubleshooting

### 메모리 부족
```bash
# 배치 크기 줄이기
python main.py --batch-size 16
```

### 훈련이 느림
```bash
# 게임 수 줄이기
python main.py --games-per-iteration 20
```

### 모델이 개선되지 않음
```bash
# 학습률 조정
python main.py --lr 0.0005
```

### GPU 오류
```bash
# CPU 사용
export CUDA_VISIBLE_DEVICES=""
python main.py
```

## 🏆 Best Practices

1. **시작**: `quick_start.py`로 빠른 테스트
2. **확장**: `main.py`로 완전한 훈련
3. **모니터링**: 로그와 히스토리 확인
4. **테스트**: `demo.py`로 모델 평가
5. **백업**: 정기적으로 체크포인트 저장

## 📚 Additional Resources

- [YINSH Game Rules](https://en.wikipedia.org/wiki/YINSH)
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

**Happy Training! 🎮🤖** 