"""
YINSH AlphaZero 셀프플레이 데이터 생성 설정
"""

import os
from pathlib import Path

# ================================
# 데이터 생성 설정
# ================================

# 생성할 게임 수
TOTAL_GAMES = 1000

# 에이전트 조합 (비율로 설정)
AGENT_COMBINATIONS = {
    'random_vs_random': 0.6,     # 60% - 빠른 데이터 생성
    'ai_vs_random': 0.3,         # 30% - AI 학습 데이터
    'random_vs_ai': 0.1,         # 10% - 균형 잡힌 데이터
}

# MCTS 설정 (초기 단계이므로 적은 시뮬레이션)
MCTS_SIMULATIONS = {
    'initial': 200,              # 초기 AI 시뮬레이션 수
    'random': 0,                 # Random 에이전트는 MCTS 없음
}

# 병렬 처리 설정
NUM_PROCESSES = 4                # CPU 코어별 프로세스 수
GAMES_PER_PROCESS = TOTAL_GAMES // NUM_PROCESSES

# 데이터 저장 설정
DATA_DIR = Path("training/data")
BATCH_SIZE = 32                  # 데이터 로딩 배치 크기
MAX_GAME_LENGTH = 200            # 최대 게임 길이

# 파일 경로 설정
SELFPLAY_DATA_FILE = DATA_DIR / "selfplay_games.pkl"
PROCESSED_DATA_FILE = DATA_DIR / "training_data.npz"
STATS_FILE = DATA_DIR / "data_stats.json"

# ================================
# 데이터 품질 설정
# ================================

# 게임 필터링 기준
MIN_GAME_LENGTH = 20             # 너무 짧은 게임 제외
MAX_GAME_LENGTH_FILTER = 500     # 너무 긴 게임 제외

# 액션 다양성 기준
MIN_UNIQUE_ACTIONS = 10          # 최소 고유 액션 수

# 온도 설정 (탐험 vs 활용)
TEMPERATURE_SCHEDULE = {
    'early_game': 1.0,           # 게임 초기 - 높은 탐험
    'mid_game': 0.5,             # 게임 중반 - 균형
    'late_game': 0.1,            # 게임 후반 - 높은 활용
}

# ================================
# 유틸리티 함수
# ================================

def create_data_directories():
    """필요한 데이터 디렉토리들을 생성합니다."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
def get_temperature(move_number: int, total_moves: int) -> float:
    """게임 진행도에 따른 온도 값을 반환합니다."""
    progress = move_number / max(total_moves, 1)
    
    if progress < 0.3:
        return TEMPERATURE_SCHEDULE['early_game']
    elif progress < 0.7:
        return TEMPERATURE_SCHEDULE['mid_game']
    else:
        return TEMPERATURE_SCHEDULE['late_game']

def validate_config():
    """설정 값들의 유효성을 검증합니다."""
    # 에이전트 조합 비율 합이 1.0인지 확인
    total_ratio = sum(AGENT_COMBINATIONS.values())
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Agent combination ratios must sum to 1.0, got {total_ratio}")
    
    # 디렉토리 생성
    create_data_directories()
    
    print("✅ Data configuration validated successfully")
    return True

if __name__ == "__main__":
    validate_config()
    print(f"📊 Will generate {TOTAL_GAMES} games with {NUM_PROCESSES} processes")
    print(f"📁 Data will be saved to: {DATA_DIR}") 