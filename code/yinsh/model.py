# model.py - YINSH Neural Network Model (PyTorch)

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
    """YINSH 게임용 신경망 (AlphaZero 스타일)"""

    def __init__(
        self,
        input_channels: int = 13,  # 11 -> 13으로 확장
        board_size: int = 11,
        num_res_blocks: int = 5,
        num_filters: int = 64,
        policy_output_dim: int = 4000,
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
            x: (batch_size, 13, 11, 11) - YINSH 게임 상태 텐서

        Returns:
            policy_logits: (batch_size, 4000) - 액션 확률 분포 (log softmax)
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
            state_tensor: (13, 11, 11) 또는 (1, 13, 11, 11) - NumPy 배열 또는 PyTorch 텐서

        Returns:
            policy_probs: (4000,) - 액션 확률 분포
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
            
            # 모델과 같은 디바이스로 이동
            device = next(self.parameters()).device
            state_tensor = state_tensor.to(device)

            policy_logits, value = self.forward(state_tensor)
            policy_probs = torch.exp(policy_logits).squeeze().cpu().numpy()
            value_scalar = value.squeeze().item()

        return policy_probs, value_scalar


class YinshModelBuilder:
    """YINSH 모델 생성 및 관리 클래스"""

    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_model(self) -> YinshNet:
        """새로운 YINSH 모델 생성"""
        model = YinshNet(
            input_channels=13,  # 확장된 채널 수
            board_size=config.BOARD_SIZE,
            num_res_blocks=config.AMOUNT_OF_RESIDUAL_BLOCKS,
            num_filters=config.CONVOLUTION_FILTERS,
            policy_output_dim=config.POLICY_OUTPUT_SIZE,
        )

        model = model.to(self.device)
        self.model = model
        return model

    def load_model(self, model_path: str) -> YinshNet:
        """저장된 모델 로드"""
        model = self.build_model()

        try:
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"✅ Model loaded from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔄 Using random initialized model")

        model.eval()
        self.model = model
        return model

    def save_model(self, model: YinshNet, model_path: str):
        """모델 저장"""
        try:
            torch.save(model.state_dict(), model_path)
            print(f"💾 Model saved to {model_path}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")

    def get_model_summary(self, model: YinshNet):
        """모델 구조 요약"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(
            f"""
🧠 YINSH Neural Network Summary:
├── Input Shape: {config.INPUT_SHAPE}
├── Policy Output: {config.POLICY_OUTPUT_SIZE}
├── Board Size: {config.BOARD_SIZE}x{config.BOARD_SIZE}
├── Residual Blocks: {config.AMOUNT_OF_RESIDUAL_BLOCKS}
├── Filters: {config.CONVOLUTION_FILTERS}
├── Total Parameters: {total_params:,}
├── Trainable Parameters: {trainable_params:,}
└── Device: {self.device}
        """
        )


def create_yinsh_model() -> YinshNet:
    """편의 함수: 새로운 YINSH 모델 생성"""
    builder = YinshModelBuilder()
    return builder.build_model()


def load_yinsh_model(model_path: str) -> YinshNet:
    """편의 함수: YINSH 모델 로드"""
    builder = YinshModelBuilder()
    return builder.load_model(model_path)


# 입력 채널 설명
CHANNEL_DESCRIPTIONS = {
    0: "흰 링 위치 (1=흰 링, 0=그 외)",
    1: "검은 링 위치 (1=검 링, 0=그 외)",
    2: "흰 마커 위치 (1=흰 마커, 0=그 외)",
    3: "검은 마커 위치 (1=검 마커, 0=그 외)",
    4: "유효 칸 마스크 (1=플레이가능점, 0=불가점)",
    5: "흰색 제거가능한 마커",
    6: "검은색 제거가능한 마커",
    7: "현재 단계 (GameTurnState)",
    8: "현재 플레이어 (white=+1, black=-1)",
    9: "흰 플레이어 회수 링 수 (0~3)",
    10: "검 플레이어 회수 링 수 (0~3)",
    11: "흰 마커 풀 잔여 개수 (정규화)",
    12: "검은 마커 풀 잔여 개수 (정규화)",
}
