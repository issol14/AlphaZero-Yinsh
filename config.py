# config file: includes parameters for the model and the mcts tree
import os


# ============= MCTS 설정값 =============
SIMULATIONS_PER_MOVE = 400
# UCT 공식에서 사용되는 탐색 계수 (AlphaZero 식으로 적용)
C_base = 19652
C_init = 1.25

# Dirichlet noise (초기 exploration 강제)
DIRICHLET_NOISE = 0.25

# 게임 최대 수 (YINSH 게임은 보통 85턴 이내 종료)
MAX_GAME_MOVES = 85


# ============= YINSH 규칙 기반 입력 설정 =============

# YINSH 보드 크기: 5링 보드 기준, 11x11 hex 좌표 체계로 처리하되 유효 좌표만 사용
# 이를 위한 평면 보드 변환 (패딩 포함 11x11, 실제 유효한 hex 좌표는 61개)

BOARD_SIZE = 11  # (입력은 square-grid로 변환하되 실제 playable 위치만 마스크)

# 각 입력 채널 구성:
# - 내 링 위치 (1 plane)
# - 상대 링 위치 (1)
# - 내 마커 위치 (1)
# - 상대 마커 위치 (1)
# - 현재 플레이어 표시 (1)
# 총 5개의 plane을 갖는 2D 입력

INPUT_PLANES = 5
INPUT_SHAPE = (INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)  # channel-first (NCHW)

# ============= 출력 설정 =============

# 가능한 행동:
# YINSH의 한 턴은 [링 선택 → 이동 경로 선택 → 마커 제거]의 복합 행동
# 행동 공간을 이산화하여:
# (1) 링 선택 (최대 5개)
# (2) 방향 선택 (6방향)
# (3) 거리 선택 (최대 6칸 정도까지 가능)
# 따라서 정책 차원은: 5 * 6 * 6 = 180 가지

# 또는 self-play 구조에 따라 Action Encoder에 따라 변경 가능
# 여기선 간단하게 policy head는 180차원 출력 벡터로 정의

POLICY_OUTPUT_DIM = 180
VALUE_OUTPUT_DIM = 1

OUTPUT_SHAPE = (POLICY_OUTPUT_DIM, VALUE_OUTPUT_DIM)

# ============= 신경망 구조 설정 =============

AMOUNT_OF_RESIDUAL_BLOCKS = 10
CONVOLUTION_FILTERS = 128
LEARNING_RATE = 0.001
