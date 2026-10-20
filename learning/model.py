"""轻量 MLP 失败预测器 (PyTorch), 实现 FailurePredictor 接口。"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from interfaces import FailurePrediction, FailurePredictor, STATE_FEATURE_DIM


class MLP(nn.Module):
    """22 -> 隐藏层 -> 1 的 MLP。

    输出为单值 logit (未过 sigmoid), 训练用 BCEWithLogitsLoss (可与 pos_weight 配
    合处理类不平衡)。
    """

    def __init__(
        self,
        input_dim: int = STATE_FEATURE_DIM,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dims = list(hidden_dims) if hidden_dims is not None else [64, 32]
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.hidden_dims = list(hidden_dims)
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPFailurePredictor(FailurePredictor):
    """封装 MLP + 归一化统计 + 阈值, 实现 FailurePredictor.predict。

    predict(features (22,)) -> FailurePrediction(prob, prediction, confidence)。
    confidence = 2*|prob-0.5| (clip 到 [0,1]), 与 eval.learning_baselines.LearnedPredictor 一致。
    """

    def __init__(
        self,
        model: nn.Module,
        mean: np.ndarray,
        std: np.ndarray,
        threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, -1)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, -1)
        self.threshold = float(threshold)
        self.device = device

    def predict(self, features: np.ndarray) -> FailurePrediction:
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != STATE_FEATURE_DIM:
            raise ValueError(f"特征维度须为 {STATE_FEATURE_DIM}, 得到 {x.shape[1]}")
        x = (x - self.mean) / self.std
        self.model.eval()
        with torch.no_grad():
            t = torch.from_numpy(x).to(self.device)
            prob = float(torch.sigmoid(self.model(t)).item())
        prob = float(np.clip(prob, 0.0, 1.0))
        return FailurePrediction(
            prob=prob,
            prediction=prob >= self.threshold,
            confidence=float(np.clip(abs(prob - 0.5) * 2.0, 0.0, 1.0)),
        )


def predict_probs_batch(
    model: nn.Module,
    X: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: str = "cpu",
    batch_size: int = 4096,
) -> np.ndarray:
    """批量前向, 返回 sigmoid 概率数组 (N,)。不依赖 MLPFailurePredictor, 供训练/评估用。"""
    X = np.asarray(X, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32).reshape(1, -1)
    std = np.asarray(std, dtype=np.float32).reshape(1, -1)
    Xn = (X - mean) / std
    model.eval()
    outs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, Xn.shape[0], batch_size):
            chunk = torch.from_numpy(Xn[i:i + batch_size]).to(device)
            outs.append(torch.sigmoid(model(chunk)).cpu().numpy())
    if not outs:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(outs).astype(np.float32)
