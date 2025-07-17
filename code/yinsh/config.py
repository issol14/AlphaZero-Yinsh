# config.py - YINSH AlphaZero Configuration

# ==== 게임 환경 설정 ====
BOARD_SIZE = 11  # 게임 보드 크기 (11x11)
TOTAL_MARKERS = 51  # 마커 풀 크기 (YINSH 공식 규칙)
MAX_RINGS = 5  # 플레이어당 링 개수
RINGS_PER_PLAYER = 5  # 각 플레이어의 링 개수 (YINSH 규칙)
RINGS_TO_WIN = 3  # 승리에 필요한 제거된 링 개수 (YINSH 규칙)
LINE_LENGTH_TO_WIN = 5  # 라인을 이루기 위한 마커 개수 (YINSH 규칙)

# ==== 유효한 육각형 방향 ====
VALID_HEX_DIRECTIONS = [
    (0, 1), (0, -1),    # 세로
    (1, 0), (-1, 0),    # 가로 
    (1, 1), (-1, -1),   # 대각선
]

# ==== 신경망 설정 ====
INPUT_SHAPE = (13, 11, 11)  # 입력 텐서 형태 (13채널, 11x11)
POLICY_OUTPUT_SIZE = 4000   # 액션 공간 크기 (대폭 확장)

# 모델 아키텍처
AMOUNT_OF_RESIDUAL_BLOCKS = 5
CONVOLUTION_FILTERS = 64
DENSE_LAYERS = [256]

# ==== MCTS 설정 ====
MCTS_SIMULATIONS = 800
CPUCT = 1.0  # UCT 탐색 계수

# ==== 학습 설정 ====
LEARNING_RATE = 0.001
L2_REGULARIZATION = 1e-4
MOMENTUM = 0.9
BATCH_SIZE = 32

# 훈련 에포크 및 스케줄링
TRAINING_LOOPS = 100
EPOCHS_PER_LOOP = 10
LEARNING_RATE_DECAY = 0.95
LEARNING_RATE_DECAY_STEPS = 10

# 메모리 관리
MEMORY_SIZE = 100000
MEMORY_MINIMUM_SIZE = 10000

# ==== 파일 및 디렉토리 설정 ====
MODEL_FOLDER = "models"
MEMORY_DIR = "memory" 
LOSS_PLOTS_FOLDER = "plots"
LOG_FOLDER = "logs"

# 체크포인트 및 저장
CHECKPOINT_FREQUENCY = 10  # 몇 루프마다 체크포인트 저장
MODEL_SAVE_FREQUENCY = 5   # 몇 루프마다 모델 저장

# ==== 게임 플레이 설정 ====
MAX_GAME_MOVES = 200  # 게임당 최대 이동 수
GAME_TIMEOUT = 600    # 게임 타임아웃 (초)

# ==== 로깅 설정 ====
VERBOSE = True
LOG_LEVEL = "INFO"
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
EVALUATION_GAMES = 10  # 평가 게임 수

# ==== 게임 플레이 설정 ====
MAX_GAME_MOVES = 200  # 게임당 최대 이동 수
GAME_TIMEOUT = 600    # 게임 타임아웃 (초)

# ==== 디버깅 설정 ====
DEBUG_MODE = False
SAVE_GAME_STATES = True
VISUALIZE_STATES = False
