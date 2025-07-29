# agent.py - YINSH Neural Agent (PyTorch)

import torch
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, List, Optional
from pathlib import Path

from .model import YinshNet, YinshModelBuilder
from .mcts_optimized import OptimizedMCTSAgent as MCTSAgent
from .mcts_parallel import ParallelMCTSAgent as ParallelMCTSAgent
from .env import YinshEnv, YinshAction, Color, GamePhase
from .mapper import YinshActionMapper
from . import config


class YinshAgent:
    """YINSH 게임용 신경망 에이전트 (AlphaZero 스타일)"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_mcts: bool = True,
        use_parallel_mcts: bool = False,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_path: 사전 훈련된 모델 경로
            use_mcts: MCTS 사용 여부
            use_parallel_mcts: 병렬 MCTS 사용 여부
            device: PyTorch 디바이스 ('cuda', 'cpu', 또는 None=자동선택)
        """
        # 디바이스 설정
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"🔧 YINSH Agent initializing on device: {self.device}")

        # 모델 초기화
        self.model_builder = YinshModelBuilder()
        if model_path and Path(model_path).exists():
            self.neural_network = self.model_builder.load_model(model_path)
            print(f"✅ Loaded model from {model_path}")
        else:
            self.neural_network = self.model_builder.build_model()
            print(f"🔄 Created new random model")

        # MCTS 에이전트 설정
        self.use_mcts = use_mcts
        self.use_parallel_mcts = use_parallel_mcts
        
        if use_mcts:
            if use_parallel_mcts:
                # 병렬 MCTS 설정
                mcts_config = {
                    "c_puct": config.CPUCT,
                    "num_simulations": config.MCTS_SIMULATIONS,
                    "num_threads": 4,  # CPU 코어 수에 맞게 조정
                    "batch_size": 32
                }
                self.mcts_agent = ParallelMCTSAgent(self.neural_network, mcts_config)
                print(f"🌳 Parallel MCTS enabled with {config.MCTS_SIMULATIONS} simulations, {mcts_config['num_threads']} threads")
            else:
                # 일반 MCTS 설정
                mcts_config = {
                    "c_puct": config.CPUCT,
                    "num_simulations": config.MCTS_SIMULATIONS,
                }
                self.mcts_agent = MCTSAgent(self.neural_network, mcts_config)
                print(f"🌳 MCTS enabled with {config.MCTS_SIMULATIONS} simulations")
        else:
            self.mcts_agent = None
            print(f"🚀 Direct neural network prediction (no MCTS)")

        # 액션 매퍼
        self.action_mapper = YinshActionMapper()

        # 통계
        self.games_played = 0
        self.training_step = 0

        # 옵티마이저 (훈련용)
        self.optimizer = optim.Adam(
            self.neural_network.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.L2_REGULARIZATION,
        )

        # 손실 함수
        self.value_loss_fn = torch.nn.MSELoss()
        self.policy_loss_fn = torch.nn.KLDivLoss(reduction="batchmean")

    def select_action(
        self, env: YinshEnv, temperature: float = 1.0, add_noise: bool = False
    ) -> Tuple[YinshAction, Dict]:
        """
        게임 상태에서 액션 선택

        Args:
            env: YINSH 게임 환경
            temperature: 선택 온도 (0=greedy, 1=stochastic)
            add_noise: 루트 노드에 Dirichlet 노이즈 추가 여부

        Returns:
            selected_action: 선택된 액션
            action_info: 액션 선택 정보 (MCTS 통계 등)
        """
        if self.use_mcts:
            # MCTS 기반 액션 선택
            action, mcts_info = self.mcts_agent.select_action(env, temperature)

            action_info = {
                "method": "mcts",
                "temperature": temperature,
                "add_noise": add_noise,
                **mcts_info,
            }
        else:
            # 직접 신경망 예측 기반 액션 선택
            action, direct_info = self._select_action_direct(env, temperature)

            action_info = {
                "method": "direct",
                "temperature": temperature,
                **direct_info,
            }

        return action, action_info

    def _select_action_direct(
        self, env: YinshEnv, temperature: float
    ) -> Tuple[YinshAction, Dict]:
        """
        MCTS 없이 직접 신경망 예측으로 액션 선택
        """
        # 상태 텐서 생성
        state_tensor = env.get_state_tensor()

        # 신경망 예측
        policy_probs, value = self.predict(state_tensor)

        # 유효한 액션들 가져오기
        valid_actions = env.get_valid_actions()

        # 유효한 액션들을 인덱스로 변환하여 마스킹
        valid_indices = []
        for action in valid_actions:
            try:
                idx = self.action_mapper.get_action_index(action)
                if idx is not None:
                    valid_indices.append(idx)
            except:
                continue
        
        if not valid_indices:
            # 매핑 가능한 유효 액션이 없으면 랜덤 선택
            selected_action = np.random.choice(valid_actions)
            action_prob = 1.0 / len(valid_actions)
        else:
            # 유효한 인덱스에 대해서만 확률 추출
            valid_probs = policy_probs[valid_indices]
            
            # 온도 적용
            if temperature != 1.0 and temperature > 0:
                valid_probs = np.power(valid_probs, 1.0 / temperature)

            # 정규화
            if valid_probs.sum() > 0:
                valid_probs = valid_probs / valid_probs.sum()
            else:
                valid_probs = np.ones(len(valid_probs)) / len(valid_probs)

            # 액션 선택
            selected_idx_in_valid = np.random.choice(len(valid_probs), p=valid_probs)
            selected_action_idx = valid_indices[selected_idx_in_valid]
            
            try:
                selected_action = self.action_mapper.index_to_action(selected_action_idx)
                action_prob = valid_probs[selected_idx_in_valid]
                
                # 선택된 액션이 실제 유효한지 확인
                if selected_action not in valid_actions:
                    selected_action = np.random.choice(valid_actions)
                    action_prob = 1.0 / len(valid_actions)
            except:
                selected_action = np.random.choice(valid_actions)
                action_prob = 1.0 / len(valid_actions)

        direct_info = {
            "value": value,
            "action_probability": action_prob,
            "valid_actions_count": len(valid_actions),
            "agent_name": "Direct Neural Agent",
        }

        return selected_action, direct_info

    def predict(self, state_tensor: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        신경망으로 정책과 가치 예측

        Args:
            state_tensor: (13, 11, 11) 게임 상태 텐서

        Returns:
            policy_probs: (4000,) 액션 확률 분포
            value: 위치 평가값 (-1 ~ +1)
        """
        return self.neural_network.predict(state_tensor)

    def train_step(
        self,
        states: List[np.ndarray],
        target_policies: List[np.ndarray],
        target_values: List[float],
    ) -> Dict[str, float]:
        """
        단일 훈련 스텝 수행

        Args:
            states: 게임 상태들
            target_policies: 타겟 정책 분포들 (MCTS 결과)
            target_values: 타겟 가치들 (게임 결과)

        Returns:
            훈련 손실 정보
        """
        if len(states) == 0:
            return {"total_loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0}

        # 배치 텐서 생성
        batch_states = torch.stack([torch.FloatTensor(s) for s in states]).to(self.device)
        batch_policies = torch.FloatTensor(np.array(target_policies)).to(self.device)
        batch_values = torch.FloatTensor(target_values).unsqueeze(1).to(self.device)

        # Forward pass
        self.optimizer.zero_grad()
        policy_logits, value_pred = self.neural_network(batch_states)

        # 손실 계산
        policy_loss = self.policy_loss_fn(policy_logits, batch_policies)
        value_loss = self.value_loss_fn(value_pred, batch_values)
        total_loss = policy_loss + value_loss

        # Backward pass
        total_loss.backward()
        self.optimizer.step()

        self.training_step += 1

        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }

    def save_model(self, save_path: str) -> None:
        """모델 저장"""
        self.model_builder.save_model(self.neural_network, save_path)

    def load_model(self, model_path: str) -> None:
        """모델 로드"""
        self.neural_network = self.model_builder.load_model(model_path)

    def set_learning_rate(self, lr: float) -> None:
        """학습률 설정"""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        total_params = sum(p.numel() for p in self.neural_network.parameters())
        trainable_params = sum(p.numel() for p in self.neural_network.parameters() if p.requires_grad)
        
        model_info = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "use_mcts": self.use_mcts,
            "device": str(self.device),
            "games_played": self.games_played,
            "training_step": self.training_step,
        }

        return model_info

    def reset_game_stats(self) -> None:
        """게임 통계 리셋"""
        self.games_played += 1
        if self.mcts_agent:
            self.mcts_agent.reset()


class RandomAgent:
    """랜덤 액션 선택 에이전트 (테스트용)"""

    def __init__(self):
        self.name = "Random Agent"
        self.games_played = 0

    def select_action(self, env: YinshEnv, **kwargs) -> Tuple[YinshAction, Dict]:
        """랜덤 액션 선택"""
        valid_actions = env.get_valid_actions()

        if not valid_actions:
            raise ValueError("No valid actions available")

        selected_action = np.random.choice(valid_actions)

        action_info = {
            "method": "random",
            "agent_name": self.name,
            "valid_actions_count": len(valid_actions),
            "action_probability": 1.0 / len(valid_actions),
        }

        return selected_action, action_info

    def reset(self):
        """게임 종료 후 리셋"""
        self.games_played += 1


def create_agent(agent_type: str = "neural", **kwargs) -> object:
    """
    에이전트 생성 팩토리 함수

    Args:
        agent_type: 에이전트 타입 ("neural", "mcts", "random")
        **kwargs: 에이전트 생성 인자

    Returns:
        생성된 에이전트
    """
    if agent_type == "neural":
        return YinshAgent(use_mcts=False, **kwargs)
    elif agent_type == "mcts":
        return YinshAgent(use_mcts=True, **kwargs)
    elif agent_type == "random":
        return RandomAgent()  # RandomAgent는 kwargs 필요 없음
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
