"""
YINSH AlphaZero Implementation
==============================

A PyTorch-based implementation of AlphaZero for the YINSH board game.

Modules:
--------
- env: YINSH game environment
- model: Neural network model
- agent: AlphaZero agent
- mcts: Monte Carlo Tree Search
- node: MCTS node implementation
- mapper: Action mapping utilities
- utils: Utility functions
- config: Configuration constants
"""

from .env import YinshEnv, YinshAction, Color, GamePhase
from .model import YinshNet as YinshModel
from .agent import YinshAgent
from .mcts import MCTS as YinshMCTS
from .node import YinshNode
from .mapper import YinshActionMapper
from .board_display import display_board, display_compact_board, display_action_details
from .utils import *
from . import config

__version__ = "1.0.0"
__author__ = "YINSH AlphaZero Team"

# 주요 클래스들
__all__ = [
    # 환경
    "YinshEnv",
    "YinshAction",
    "Color",
    "GamePhase",
    # 모델
    "YinshModel",
    # 에이전트
    "YinshAgent",
    # MCTS
    "YinshMCTS",
    "YinshNode",
    # 매퍼
    "YinshActionMapper",
    # 설정
    "config",
]

# 패키지 정보
PACKAGE_INFO = {
    "name": "yinsh_alphazero",
    "version": __version__,
    "description": "AlphaZero implementation for YINSH board game",
    "author": __author__,
    "modules": ["env", "model", "agent", "mcts", "node", "mapper", "utils", "config"],
}


def get_package_info():
    """패키지 정보 반환"""
    return PACKAGE_INFO.copy()


def test_package():
    """패키지 테스트"""
    print("🧪 Testing YINSH AlphaZero Package...")

    try:
        # 환경 테스트
        from .env import YinshEnv

        env = YinshEnv()
        print("✅ Environment imported successfully")

        # 모델 테스트
        from .model import YinshNet

        model = YinshNet()
        print("✅ Model imported successfully")

        # 에이전트 테스트
        from .agent import YinshAgent

        agent = YinshAgent(model_path=None)
        print("✅ Agent imported successfully")

        # MCTS 테스트
        from .mcts import MCTS as YinshMCTS

        mcts = YinshMCTS(model)
        print("✅ MCTS imported successfully")

        # 매퍼 테스트
        from .mapper import YinshActionMapper

        mapper = YinshActionMapper()
        print("✅ Action mapper imported successfully")

        print("🎉 All modules imported successfully!")

    except Exception as e:
        print(f"❌ Import error: {e}")
        raise


if __name__ == "__main__":
    test_package()
