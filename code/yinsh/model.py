# model.py - Simplified YINSH Neural Network Model (PyTorch)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from . import config


class ResidualBlock(nn.Module):
    """잔차 블록 (ResNet 스타일)"""

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class YinshNet(nn.Module):
    """간소화된 YINSH 게임용 신경망 (MOVE_RING만)"""

    def __init__(
        self,
        input_channels: int = 3,  # 간소화: 흰색 링, 검은색 링, 마커
        board_size: int = 11,
        num_res_blocks: int = 5,
        num_filters: int = 64,
        policy_output_dim: int = 121,  # 간소화: 11x11 = 121 (from_pos * to_pos)
    ):
        super(YinshNet, self).__init__()

        self.board_size = board_size
        self.policy_output_dim = policy_output_dim

        # 초기 합성곱층
        self.conv1 = nn.Conv2d(
            input_channels, num_filters, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_filters)

        # 잔차 블록들
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # 정책 헤드 (Policy Head)
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, policy_output_dim)

        # 가치 헤드 (Value Head)
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: (batch_size, 3, 11, 11) - 간소화된 YINSH 게임 상태 텐서

        Returns:
            policy_logits: (batch_size, 121) - 액션 확률 분포 (log softmax)
            value: (batch_size, 1) - 위치 평가값 (-1 ~ +1)
        """
        # 백본 네트워크
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_blocks(x)

        # 정책 헤드
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)  # flatten
        p = F.log_softmax(self.policy_fc(p), dim=1)

        # 가치 헤드
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)  # flatten
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

    def predict(self, state_tensor):
        """
        단일 상태에 대한 예측

        Args:
            state_tensor: (3, 11, 11) 또는 (1, 3, 11, 11) - NumPy 배열 또는 PyTorch 텐서

        Returns:
            policy_probs: (121,) - 액션 확률 분포
            value: float - 위치 평가값
        """
        self.eval()
        with torch.no_grad():
            # NumPy 배열을 PyTorch 텐서로 변환
            if isinstance(state_tensor, np.ndarray):
                state_tensor = torch.from_numpy(state_tensor).float()
            elif not isinstance(state_tensor, torch.Tensor):
                state_tensor = torch.FloatTensor(state_tensor)

            # 배치 차원 추가
            if len(state_tensor.shape) == 3:
                state_tensor = state_tensor.unsqueeze(0)

            policy_logits, value = self.forward(state_tensor)
            policy_probs = torch.exp(policy_logits).squeeze().cpu().numpy()
            value_scalar = value.squeeze().item()

        return policy_probs, value_scalar


class YinshModelBuilder:
    """YINSH 모델 생성 및 관리 클래스"""

    def __init__(self):
        self.config = config

    def build_model(self) -> YinshNet:
        """새로운 YINSH 모델 생성"""
        model = YinshNet(
            input_channels=3,  # 간소화된 입력
            board_size=self.config.BOARD_SIZE,
            num_res_blocks=self.config.AMOUNT_OF_RESIDUAL_BLOCKS,
            num_filters=self.config.CONVOLUTION_FILTERS,
            policy_output_dim=121,  # 간소화된 액션 공간
        )
        return model

    def load_model(self, model_path: str) -> YinshNet:
        """저장된 모델 로드"""
        model = self.build_model()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        return model

    def save_model(self, model: YinshNet, model_path: str):
        """모델 저장"""
        torch.save(model.state_dict(), model_path)

    def get_model_summary(self, model: YinshNet):
        """모델 요약 정보"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "input_channels": 3,
            "board_size": self.config.BOARD_SIZE,
            "policy_output_dim": 121,
            "residual_blocks": self.config.AMOUNT_OF_RESIDUAL_BLOCKS,
            "filters": self.config.CONVOLUTION_FILTERS,
        }


def create_yinsh_model() -> YinshNet:
    """간소화된 YINSH 모델 생성"""
    builder = YinshModelBuilder()
    return builder.build_model()


def load_yinsh_model(model_path: str) -> YinshNet:
    """저장된 간소화된 YINSH 모델 로드"""
    builder = YinshModelBuilder()
    return builder.load_model(model_path)
