# utils.py - YINSH Utility Functions

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any
from functools import wraps
import matplotlib.pyplot as plt
from datetime import datetime

from .env import YinshEnv, YinshAction
from . import config


def time_function(func):
    """함수 실행 시간 측정 데코레이터"""

    @wraps(func)
    def wrap_func(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️ {func.__name__} took {end_time - start_time:.2f} seconds")
        return result

    return wrap_func


def save_state_to_image(state_tensor: torch.Tensor, path: str, name: str = "state"):
    """게임 상태를 이미지로 저장"""
    if not isinstance(state_tensor, torch.Tensor):
        state_tensor = torch.FloatTensor(state_tensor)

    # 상태 텐서를 시각화 가능한 형태로 변환
    state_np = state_tensor.cpu().numpy()

    # 채널별로 시각화
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    channel_names = [
        "White Rings",
        "White Markers",
        "Black Rings",
        "Black Markers",
        "Current Player",
        "Game Phase",
        "White Rings Placed",
        "Black Rings Placed",
        "White Rings Removed",
        "Black Rings Removed",
        "Valid Positions",
    ]

    for i in range(min(len(channel_names), len(axes) - 1)):
        if i < state_np.shape[0]:
            im = axes[i].imshow(state_np[i], cmap="viridis")
            axes[i].set_title(channel_names[i])
            plt.colorbar(im, ax=axes[i])

    # 마지막 축은 비워둠
    axes[-1].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"💾 State visualization saved to {path}")


def save_policy_to_image(policy_probs: np.ndarray, path: str, name: str = "policy"):
    """정책 분포를 이미지로 저장"""
    plt.figure(figsize=(12, 6))

    # 정책 분포 플롯
    plt.subplot(1, 2, 1)
    plt.bar(range(len(policy_probs)), policy_probs)
    plt.title(f"{name} Policy Distribution")
    plt.xlabel("Action Index")
    plt.ylabel("Probability")
    plt.yscale("log")

    # 상위 액션들
    plt.subplot(1, 2, 2)
    top_indices = np.argsort(policy_probs)[-10:]  # 상위 10개
    top_probs = policy_probs[top_indices]
    plt.bar(range(len(top_indices)), top_probs)
    plt.title("Top 10 Actions")
    plt.xlabel("Action Index")
    plt.ylabel("Probability")

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"💾 Policy visualization saved to {path}")


def create_training_batch(
    states: List[torch.Tensor], policies: List[np.ndarray], values: List[float]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    훈련 배치 생성

    Args:
        states: 상태 텐서 리스트
        policies: 정책 분포 리스트
        values: 가치 리스트

    Returns:
        배치 텐서들
    """
    # 상태 배치
    state_batch = torch.stack(states)

    # 정책 배치
    policy_batch = torch.FloatTensor(np.array(policies))

    # 가치 배치
    value_batch = torch.FloatTensor(values).unsqueeze(1)

    return state_batch, policy_batch, value_batch


def calculate_accuracy(
    predicted_policies: torch.Tensor, target_policies: torch.Tensor
) -> float:
    """
    정책 예측 정확도 계산

    Args:
        predicted_policies: 예측된 정책 (logits)
        target_policies: 타겟 정책 (one-hot)

    Returns:
        정확도
    """
    predicted_actions = torch.argmax(predicted_policies, dim=1)
    target_actions = torch.argmax(target_policies, dim=1)

    correct = (predicted_actions == target_actions).float().mean()
    return correct.item()


def calculate_value_accuracy(
    predicted_values: torch.Tensor, target_values: torch.Tensor, threshold: float = 0.1
) -> float:
    """
    가치 예측 정확도 계산

    Args:
        predicted_values: 예측된 가치
        target_values: 타겟 가치
        threshold: 정확도 임계값

    Returns:
        정확도
    """
    diff = torch.abs(predicted_values - target_values)
    correct = (diff < threshold).float().mean()
    return correct.item()


def get_model_summary(model: torch.nn.Module) -> Dict:
    """
    모델 요약 정보 반환

    Args:
        model: PyTorch 모델

    Returns:
        모델 정보 딕셔너리
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # float32 기준
        "device": next(model.parameters()).device,
    }


def save_training_history(history: List[Dict], path: str):
    """
    훈련 히스토리 저장

    Args:
        history: 훈련 히스토리 리스트
        path: 저장 경로
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # numpy 배열을 리스트로 변환
    serializable_history = []
    for entry in history:
        serializable_entry = {}
        for key, value in entry.items():
            if isinstance(value, np.ndarray):
                serializable_entry[key] = value.tolist()
            elif isinstance(value, torch.Tensor):
                serializable_entry[key] = value.cpu().numpy().tolist()
            else:
                serializable_entry[key] = value
        serializable_history.append(serializable_entry)

    with open(path, "w") as f:
        json.dump(serializable_history, f, indent=2)

    print(f"💾 Training history saved to {path}")


def load_training_history(path: str) -> List[Dict]:
    """
    훈련 히스토리 로드

    Args:
        path: 파일 경로

    Returns:
        훈련 히스토리 리스트
    """
    with open(path, "r") as f:
        history = json.load(f)

    return history


def create_directory_structure():
    """
    필요한 디렉토리 구조 생성
    """
    directories = [
        config.MEMORY_DIR,
        config.MODEL_FOLDER,
        config.LOSS_PLOTS_FOLDER,
        "logs",
        "checkpoints",
        "evaluation_results",
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")


def get_device_info() -> Dict:
    """
    현재 디바이스 정보 반환

    Returns:
        디바이스 정보 딕셔너리
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info.update(
            {
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_memory_allocated": torch.cuda.memory_allocated(0)
                / (1024**3),  # GB
                "cuda_memory_reserved": torch.cuda.memory_reserved(0) / (1024**3),  # GB
            }
        )

    return info


def print_device_info():
    """
    디바이스 정보 출력
    """
    info = get_device_info()

    print("🖥️ Device Information:")
    for key, value in info.items():
        if isinstance(value, float):
            print(f"├── {key}: {value:.3f}")
        else:
            print(f"├── {key}: {value}")
    print("└──")


def validate_yinsh_action(action: YinshAction, env: YinshEnv) -> bool:
    """
    YINSH 액션 유효성 검증

    Args:
        action: 검증할 액션
        env: YINSH 환경

    Returns:
        유효성 여부
    """
    try:
        valid_actions = env.get_valid_actions()
        return action in valid_actions
    except Exception:
        return False


def format_time(seconds: float) -> str:
    """
    초를 읽기 쉬운 시간 형식으로 변환

    Args:
        seconds: 초

    Returns:
        포맷된 시간 문자열
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def print_training_stats(epoch: int, total_epochs: int, losses: Dict[str, float]):
    """
    훈련 통계 출력

    Args:
        epoch: 현재 에포크
        total_epochs: 전체 에포크 수
        losses: 손실 정보
    """
    print(f"📚 Epoch {epoch}/{total_epochs}")
    for loss_name, loss_value in losses.items():
        print(f"├── {loss_name}: {loss_value:.4f}")
    print("└──")


def create_progress_bar(iterable, desc: str = "Progress"):
    """
    진행률 바 생성 (tqdm 래퍼)

    Args:
        iterable: 반복 가능한 객체
        desc: 설명

    Returns:
        tqdm 객체
    """
    try:
        from tqdm import tqdm

        return tqdm(iterable, desc=desc)
    except ImportError:
        # tqdm이 없으면 기본 반복자 반환
        return iterable
