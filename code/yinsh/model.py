# model.py - YINSH Neural Network Model (PyTorch) - 간소화 버전

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from . import config


class ResidualBlock(nn.Module):
    """잔차 블록 (AlphaZero 논문 기반)"""

    def __init__(self, channels, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class YinshNet(nn.Module):
    """YINSH 게임용 신경망 (AlphaZero 논문 기반) - 간소화 버전"""

    def __init__(
        self,
        input_channels: int = 6,  # 6채널로 간소화
        board_size: int = 11,
        num_res_blocks: int = 20,  # AlphaZero 논문: 체스 20, 바둑 40
        num_filters: int = 256,  # AlphaZero 논문: 256
        policy_output_dim: int = 1848,  # 링 이동 액션만 고려하므로 크기 축소
        dropout_rate: float = 0.3,
    ):
        super(YinshNet, self).__init__()

        self.board_size = board_size
        self.policy_output_dim = policy_output_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 기본값

        # 초기 합성곱층
        self.conv1 = nn.Conv2d(
            input_channels, num_filters, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_filters)

        # 잔차 블록들 (AlphaZero 논문 기반)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_filters, dropout_rate) for _ in range(num_res_blocks)]
        )

        # 정책 헤드 (Policy Head) - AlphaZero 논문 기반
        self.policy_conv = nn.Conv2d(num_filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, policy_output_dim)
        self.policy_dropout = nn.Dropout(dropout_rate)

        # 가치 헤드 (Value Head) - AlphaZero 논문 기반
        self.value_conv = nn.Conv2d(num_filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
        self.value_dropout = nn.Dropout(dropout_rate)

    def to(self, device):
        """디바이스 이동 시 device 속성도 업데이트"""
        self.device = device
        return super().to(device)

    def forward(self, x):
        """
        Forward pass (AlphaZero 논문 기반)

        Args:
            x: (batch_size, 6, 11, 11) - YINSH 게임 상태 텐서 (간소화)

        Returns:
            policy_logits: (batch_size, 1848) - 액션 확률 분포 (log softmax)
            value: (batch_size, 1) - 위치 평가값 (-1 ~ +1)
        """
        # 백본 네트워크
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_blocks(x)

        # 정책 헤드 (AlphaZero 논문 기반)
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)  # flatten
        p = self.policy_dropout(p)
        p = F.log_softmax(self.policy_fc(p), dim=1)

        # 가치 헤드 (AlphaZero 논문 기반)
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)  # flatten
        # v = self.value_dropout(v)
        v = F.relu(self.value_fc1(v))
        v = self.value_dropout(v)
        v = torch.tanh(self.value_fc2(v))  # -1 ~ +1 범위

        return p, v

    def predict(self, state_tensor):
        """
        단일 상태에 대한 예측

        Args:
            state_tensor: (6, 11, 11) - 게임 상태 텐서

        Returns:
            policy_probs: (1848,) - 액션 확률 분포
            value: float - 상태 평가값
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
    """YINSH 모델 빌더"""

    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.default_config = {
            'input_channels': 6,
            'board_size': 11,
            'num_res_blocks': 20,
            'num_filters': 256,
            'policy_output_dim': 1848,
            'dropout_rate': 0.3
        }

    def build_model(self) -> YinshNet:
        """새로운 YINSH 모델 생성 (AlphaZero 논문 기반)"""
        model = YinshNet(
            input_channels=self.default_config['input_channels'],
            board_size=self.default_config['board_size'],
            num_res_blocks=self.default_config['num_res_blocks'],
            num_filters=self.default_config['num_filters'],
            policy_output_dim=self.default_config['policy_output_dim'],
            dropout_rate=self.default_config['dropout_rate']
        )
        model = model.to(self.device)
        self.model = model
        return model

    def load_model(self, model_path: str) -> YinshNet:
        """저장된 모델 로드"""
        model = self.build_model()
        
        try:
            loaded_data = torch.load(model_path, map_location=self.device)
            
            # 모델 전체가 로드된 경우 state_dict 추출
            if isinstance(loaded_data, dict):
                state_dict = loaded_data
            else:
                # 모델 전체가 로드된 경우
                print("⚠️  전체 모델이 로드되었습니다. state_dict로 변환합니다.")
                if hasattr(loaded_data, 'state_dict'):
                    state_dict = loaded_data.state_dict()
                else:
                    raise ValueError("로드된 데이터가 state_dict도 모델도 아닙니다.")
            
            # 구조 체크
            current_state_dict = model.state_dict()
            if state_dict['conv1.weight'].shape != current_state_dict['conv1.weight'].shape:
                print(f"⚠️ 모델 구조 불일치!")
                print(f"   기존: {state_dict['conv1.weight'].shape}")
                print(f"   현재: {current_state_dict['conv1.weight'].shape}")
                print("🔄 새 모델로 초기화합니다.")
                
                # 필요시 파일도 덮어쓰기
                self.save_model(model, model_path)
                return model

            # 구조가 일치하면 정상 로드
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
            print(f"✅ 모델 저장 완료: {model_path}")
        except Exception as e:
            print(f"❌ 모델 저장 실패: {e}")

    def get_model_summary(self, model: YinshNet):
        """모델 요약 정보"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary = {
            'input_channels': model.conv1.in_channels,
            'board_size': model.board_size,
            'num_res_blocks': len(model.res_blocks),
            'num_filters': model.conv1.out_channels,
            'policy_output_dim': model.policy_output_dim,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'device': model.device
        }
        # 
        #         print(
        #             f"""
        # 🧠 YINSH Neural Network Summary (Ring Place 제거):
        # ├── Input Shape: (13, 11, 11)  # Ring Place 관련 채널 제거
        # ├── Policy Output: {config.POLICY_OUTPUT_SIZE}
        # ├── Board Size: {config.BOARD_SIZE}x{config.BOARD_SIZE}
        # ├── Residual Blocks: {config.AMOUNT_OF_RESIDUAL_BLOCKS}
        # ├── Filters: {config.CONVOLUTION_FILTERS}
        # ├── Total Parameters: {total_params:,}
        # ├── Trainable Parameters: {trainable_params:,}
        # └── Device: {self.device}
        #         """
        #         )
        return summary


def create_yinsh_model() -> YinshNet:
    """YINSH 모델 생성 (편의 함수)"""
    builder = YinshModelBuilder()
    return builder.build_model()


def load_yinsh_model(model_path: str) -> YinshNet:
    """YINSH 모델 로드 (편의 함수)"""
    builder = YinshModelBuilder()
    return builder.load_model(model_path)

