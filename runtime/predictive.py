"""runtime/predictive.py — 学习失败预测器的运行时集成 (M4-B)。

把 M4 训练出的 FailurePredictor 接入恢复编排，实现「预判性恢复」：
在执行 attempt 期间每步调用 predictor.predict(extract_state_features(obs))，
当预测失败风险高 (prob >= threshold 且 confidence >= min_confidence) 时返回
PredictiveTrigger，由 executor 提前中止当前 attempt，交 RecoveryRunner 走既有
规则恢复 (RETRY/REOBSERVE/ABORT) —— max_retries 安全约束不变。

关键约束:
- 学习预测只影响「是否提前重试」，绝不直接发底层指令 (不产动作、不改 env)。
- 推理异常 / 输入无效 / 置信度不足 / 非有限概率 -> 回退规则式行为 (返回 None 并记录 error)。
- b1 = 纯规则恢复 (guard 关闭)；b2 = 规则恢复 + 学习预判提前触发。
- 学习模型不得绕过安全控制: 触发走的是同一 RecoveryManager.decide，重试预算 (max_retries)
  与 env 终止检查强制生效；本模块无任何 env/动作写权限。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional

import numpy as np

from interfaces import (
    FailurePrediction,
    FailurePredictor,
    RobotObservation,
    STATE_FEATURE_DIM,
    extract_state_features,
)


class RecoveryMode(str, Enum):
    B1_RULE = "b1"           # 纯规则恢复 (现有 RecoveryRunner 行为)
    B2_PREDICTIVE = "b2"     # 规则恢复 + 学习预判提前触发


@dataclass
class PredictiveTrigger:
    """预测到高风险时交回 executor 的提前触发信号 (只描述「是否提前重试」，无动作)。"""

    step: int
    prob: float
    confidence: float
    threshold: float


@dataclass
class PredictionRecord:
    """单次预测的结构化记录 (进 trace)。"""

    step: int
    prob: float
    confidence: float
    prediction: bool
    triggered: bool
    threshold: Optional[float] = None
    error: Optional[str] = None


@dataclass
class PredictiveConfig:
    """学习预判配置 (全部可覆盖)。"""

    mode: RecoveryMode = RecoveryMode.B1_RULE
    threshold: Optional[float] = None   # None -> 用 predictor.threshold (默认 0.5)
    min_confidence: float = 0.0         # confidence 低于此值不触发 (回退规则)


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ValueError(f"invalid numeric value: {v!r}")


class PredictiveGuard:
    """学习预测器的运行时守卫：逐步预测 + 有界触发 + 安全回退。

    step_hook 签名匹配 executor 的 step_hook 约定：返回 None (不触发) 或
    PredictiveTrigger (提前中止当前 attempt)。任何异常/无效输入都不向上抛出，
    而是记录 error 并返回 None —— 即回退到规则式恢复，不影响安全。
    """

    def __init__(
        self,
        predictor: FailurePredictor,
        *,
        mode: RecoveryMode = RecoveryMode.B2_PREDICTIVE,
        threshold: Optional[float] = None,
        min_confidence: float = 0.0,
        feature_fn: Callable[[RobotObservation], np.ndarray] = extract_state_features,
    ) -> None:
        self.predictor = predictor
        self.mode = RecoveryMode(mode)
        self.feature_fn = feature_fn
        self.min_confidence = float(min_confidence)
        self.threshold = (
            float(threshold) if threshold is not None
            else float(getattr(predictor, "threshold", 0.5))
        )
        self._records: List[PredictionRecord] = []
        self._attempt: List[PredictionRecord] = []

    # ------------------------------------------------------------------ #
    # 状态 / 记录
    # ------------------------------------------------------------------ #
    @property
    def enabled(self) -> bool:
        return self.mode is RecoveryMode.B2_PREDICTIVE

    @property
    def records(self) -> List[PredictionRecord]:
        """本 run 内已 drain 的全部预测记录 (跨 attempt 汇总)。"""
        return list(self._records)

    def reset(self) -> None:
        self._records = []
        self._attempt = []

    def begin_attempt(self) -> None:
        self._attempt = []

    def drain_attempt(self) -> List[PredictionRecord]:
        """取回本次 attempt 的预测记录并清空 attempt 缓冲 (追加进 run 汇总)。"""
        recs = list(self._attempt)
        self._records.extend(recs)
        self._attempt = []
        return recs

    # ------------------------------------------------------------------ #
    # 逐步预测 (executor step_hook)
    # ------------------------------------------------------------------ #
    def step_hook(self, obs: RobotObservation) -> Optional[PredictiveTrigger]:
        if not self.enabled:
            return None
        step = int(getattr(obs, "step", -1))
        rec = PredictionRecord(
            step=step, prob=0.0, confidence=0.0, prediction=False,
            triggered=False, threshold=self.threshold,
        )
        try:
            features = np.asarray(self.feature_fn(obs), dtype=np.float32).reshape(-1)
            if features.shape != (STATE_FEATURE_DIM,):
                raise ValueError(
                    f"feature dim must be {STATE_FEATURE_DIM}, got {features.shape}"
                )
            if not np.all(np.isfinite(features)):
                raise ValueError("features contain non-finite values")
            pred: FailurePrediction = self.predictor.predict(features)
            prob = _to_float(getattr(pred, "prob", float("nan")))
            if not np.isfinite(prob) or not (0.0 <= prob <= 1.0):
                raise ValueError(f"invalid prob {prob}")
            confidence = _to_float(getattr(pred, "confidence", 0.0))
            if not np.isfinite(confidence) or not (0.0 <= confidence <= 1.0):
                raise ValueError(f"invalid confidence {confidence}")
        except Exception as exc:  # noqa: BLE001 - 运行时兜底: 回退规则
            rec.error = f"{type(exc).__name__}: {exc}"
            self._attempt.append(rec)
            return None

        rec.prob = prob
        rec.confidence = confidence
        rec.prediction = bool(prob >= self.threshold)
        rec.triggered = bool(rec.prediction and confidence >= self.min_confidence)
        self._attempt.append(rec)

        if rec.triggered:
            return PredictiveTrigger(
                step=step, prob=prob, confidence=confidence, threshold=self.threshold,
            )
        return None


def load_predictor_from_checkpoint(
    checkpoint_path: str,
    config_path: Optional[str] = None,
) -> FailurePredictor:
    """从 M4 checkpoint 重建 MLPFailurePredictor (懒加载 torch/learning，避免硬依赖)。

    checkpoint dict 约定 (见 learning/evaluate.py):
      state_dict / mean / std / threshold / hidden_dims / input_dim / dropout。
    config_path 仅作为缺失字段的回退 (mean/std/threshold 等通常已存进 checkpoint)。
    """
    import json
    import warnings

    import torch

    from learning.model import MLP, MLPFailurePredictor

    # checkpoint 内含 numpy 数组 (mean/std)，weights_only=True 无法加载；
    # 抑制 torch.load 的 FutureWarning (checkpoint 来自本仓库可信来源)。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg: dict = {}
    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    mean = np.asarray(ckpt.get("mean", cfg.get("mean")), dtype=np.float32)
    std = np.asarray(ckpt.get("std", cfg.get("std")), dtype=np.float32)
    threshold = float(ckpt.get("threshold", cfg.get("threshold", 0.5)))
    hidden_dims = list(ckpt.get("hidden_dims", cfg.get("hidden_dims", [32])))
    input_dim = int(ckpt.get("input_dim", cfg.get("input_dim", STATE_FEATURE_DIM)))
    dropout = float(ckpt.get("dropout", cfg.get("dropout", 0.0)))

    model = MLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return MLPFailurePredictor(model, mean, std, threshold=threshold, device="cpu")
