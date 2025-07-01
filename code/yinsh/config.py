# config.py - YINSH AlphaZero Configuration

import os

# ==== 게임 설정 ====
BOARD_SIZE = 11  # YINSH는 11x11 육각형 보드
INPUT_SHAPE = (
    11,
    11,
    11,
)  # (channels, height, width) - 11 layers for different game states
POLICY_OUTPUT_SIZE = 200  # 가능한 액션 수 (링 배치 + 링 이동)

# ==== MCTS 설정 ====
MCTS_SIMULATIONS = 400  # 수당 MCTS 시뮬레이션 횟수
CPUCT = 1.0  # UCB 탐색 상수
MAX_GAME_MOVES = 200  # 최대 게임 턴 수

# ==== 신경망 설정 ====
LEARNING_RATE = 0.001
BATCH_SIZE = 32
AMOUNT_OF_RESIDUAL_BLOCKS = 5
CONVOLUTION_FILTERS = 64
L2_REGULARIZATION = 1e-4

# ==== 학습 설정 ====
EPOCHS_PER_TRAINING = 10
GAMES_PER_ITERATION = 100
EVALUATION_GAMES = 20
MODEL_SELECTION_THRESHOLD = 0.55  # 새 모델이 이길 확률

# ==== 디렉토리 설정 ====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
LOSS_PLOTS_FOLDER = os.path.join(BASE_DIR, "plots")

# 디렉토리 생성
os.makedirs(MEMORY_DIR, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(LOSS_PLOTS_FOLDER, exist_ok=True)

# ==== YINSH 게임 규칙 설정 ====
RINGS_PER_PLAYER = 5  # 각 플레이어당 링 개수
RINGS_TO_WIN = 3  # 승리에 필요한 링 제거 개수
LINE_LENGTH_TO_WIN = 5  # 연속된 마커 5개로 라인 완성

# ==== 로깅 설정 ====
LOG_LEVEL = "INFO"
SAVE_GAMES = True
TENSORBOARD_LOG = True

# ==== PyTorch 설정 ====
DEVICE = "auto"  # "cuda", "cpu", "auto"
NUM_WORKERS = 4  # 데이터 로더 워커 수
PIN_MEMORY = True  # CUDA 메모리 핀닝

# ==== 셀프플레이 설정 ====
SELFPLAY_TEMPERATURE = 1.0  # 셀프플레이 온도
SELFPLAY_NOISE_ALPHA = 0.3  # Dirichlet 노이즈 알파
SELFPLAY_NOISE_EPSILON = 0.25  # Dirichlet 노이즈 엡실론

# ==== 평가 설정 ====
EVALUATION_TEMPERATURE = 0.1  # 평가 시 온도
EVALUATION_SIMULATIONS = 800  # 평가 시 시뮬레이션 수
