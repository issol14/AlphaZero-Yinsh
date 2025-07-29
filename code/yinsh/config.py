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
INPUT_SHAPE = (15, 11, 11)  # 입력 텐서 형태 (15채널, 11x11)
POLICY_OUTPUT_SIZE = 4000   # 액션 공간 크기 (대폭 확장)

# ==== 액션 공간 최적화 설정 ====
USE_DYNAMIC_ACTIONS = True  # 동적 액션 생성 사용 여부
USE_VALID_POINTS_ONLY = True  # valid_points만 사용하여 액션 생성
MAX_DYNAMIC_ACTIONS = 200  # 동적으로 생성되는 최대 액션 수
ENABLE_ACTION_MASKING = True  # 신경망에서 유효하지 않은 액션 마스킹

# 모델 아키텍처 (AlphaZero 논문 기반 개선)
AMOUNT_OF_RESIDUAL_BLOCKS = 24  # 5 -> 24로 증가 (체스: 20, 바둑: 40)
CONVOLUTION_FILTERS = 256  # 64 -> 128로 증가 (논문: 256)
DENSE_LAYERS = [256]

# ==== MCTS 설정 ====
MCTS_SIMULATIONS = 800
# Selfplay 최적화 설정
SELFPLAY_MCTS_SIMULATIONS = 400  # Selfplay용 빠른 시뮬레이션
EVALUATION_MCTS_SIMULATIONS = 400  # 평가용 빠른 시뮬레이션 (selfplay와 동일)
CPUCT = 2.5  # UCT 탐색 계수 (AlphaZero 논문: 체스 2.5, 바둑 5.0)

# ==== 학습 설정 (AlphaZero 논문 기반) ====
LEARNING_RATE = 0.002  # 논문: 0.002 (기존 0.001에서 증가)
L2_REGULARIZATION = 1e-4  # 논문: 1e-4
MOMENTUM = 0.9  # 논문: 0.9
BATCH_SIZE = 512  # 논문: 512 (기존 32에서 대폭 증가)

# 훈련 에포크 및 스케줄링 (AlphaZero 논문 기반)
TRAINING_LOOPS = 100
EPOCHS_PER_LOOP = 100  # 논문: 100 에포크 (기존 10에서 증가)
LEARNING_RATE_DECAY = 0.1  # 논문: 0.1 (기존 0.95에서 변경)
LEARNING_RATE_DECAY_STEPS = 400000  # 논문: 400k 스텝마다 감소

# 손실 함수 가중치 (AlphaZero 논문 기반)
POLICY_LOSS_WEIGHT = 1.0  # 정책 손실 가중치
VALUE_LOSS_WEIGHT = 1.0   # 가치 손실 가중치

# 정규화 설정
DROPOUT_RATE = 0.3  # 드롭아웃 비율
WEIGHT_DECAY = 1e-4  # L2 정규화

# 메모리 관리 (AlphaZero 논문 기반)
MEMORY_SIZE = 1000000  # 논문: 1M 포지션 (기존 100k에서 증가)
MEMORY_MINIMUM_SIZE = 100000  # 최소 메모리 크기

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

# ==== 셀플레이 설정 ====
SELFPLAY_TEMPERATURE = 1.0  # 셀플레이 온도
SELFPLAY_NOISE_ALPHA = 0.3  # Dirichlet 노이즈 알파
SELFPLAY_NOISE_EPSILON = 0.25  # Dirichlet 노이즈 엡실론

# ==== 평가 설정 ====
EVALUATION_TEMPERATURE = 0.1  # 평가 시 온도 (논문과 동일)
EVALUATION_SIMULATIONS = 800  # 평가 시 시뮬레이션 수 (논문: 정확한 평가)
EVALUATION_GAMES = 400  # 평가 게임 수 (논문: 400게임 토너먼트)
EVALUATION_THRESHOLD = 0.55  # 새 모델 채택 기준 (논문: 55%)

# ==== 디버깅 설정 ====
DEBUG_MODE = False
SAVE_GAME_STATES = True
VISUALIZE_STATES = False
