"""三类失败预测基线, 统一 FailurePredictor 接口 (M4-D)。

1. MajorityClassPredictor : 预测主导类 (拟合 train 帧标签的 prevalence)。
2. RuleBasedPredictor     : 复用 runtime.failure_detector.FailureDetector 的可观测
                            启发式 (GRIP_FAIL/COLLISION/DROP), 封装成 FailurePredictor。
3. LearnedPredictor       : 学习预测器占位 (待 Agent C 实现), 可注入 predict_fn。

三者统一经 eval.learning_metrics.compute_prediction_metrics 评估 (帧级指标一致)。
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from interfaces import (
    EpisodeRecord, FailurePrediction, FailurePredictor,
    PolicyObservation, GroundTruth, RobotObservation, STATE_FEATURE_DIM,
)
from runtime.failure_detector import FailureDetector, FailureDetectorConfig

# 特征索引 (与 interfaces.learning.STATE_FEATURE_NAMES 一致)。
_QPOS = slice(0, 7)
_QVEL = slice(7, 14)
_GRIP = 14
_TCP = slice(15, 22)


class MajorityClassPredictor(FailurePredictor):
    """主导类基线: 恒输出 train 帧标签的 prevalence, prediction = prob >= threshold。"""

    def __init__(self, prob: float, threshold: float = 0.5) -> None:
        self.prob = float(prob)
        self.threshold = float(threshold)

    @classmethod
    def fit(cls, episodes: Sequence[EpisodeRecord], threshold: float = 0.5) -> "MajorityClassPredictor":
        labels = [f.label for ep in episodes for f in ep.frames]
        if not labels:
            raise ValueError("无法拟合 majority 基线: train 集无任何帧")
        n_pos = sum(labels)
        return cls(prob=n_pos / len(labels), threshold=threshold)

    def predict(self, features: np.ndarray) -> FailurePrediction:
        return FailurePrediction(
            prob=self.prob,
            prediction=self.prob >= self.threshold,
            confidence=1.0,
        )


class RuleBasedPredictor(FailurePredictor):
    """规则式基线: 复用 FailureDetector 的可观测启发式 (GRIP_FAIL/COLLISION/DROP)。

    每次 predict 把当前特征帧重构为 RobotObservation 追加到内部历史, 调用
    FailureDetector.detect_policy_anomaly (只读 policy 字段, 绝不读 truth)。
    输出硬 0/1 (规则命中 -> prob=1.0)。跨 episode 评估前须 reset() 清空历史。
    """

    def __init__(
        self,
        config: Optional[FailureDetectorConfig] = None,
        history_limit: int = 200,
    ) -> None:
        self.detector = FailureDetector(config)
        self.history_limit = int(history_limit)
        self._history: List[RobotObservation] = []

    def reset(self) -> None:
        self._history = []

    def predict(self, features: np.ndarray) -> FailurePrediction:
        obs = self._features_to_obs(features)
        self._history.append(obs)
        if len(self._history) > self.history_limit:
            self._history = self._history[-self.history_limit:]
        event = self.detector.detect_policy_anomaly(self._history)
        if event is not None:
            return FailurePrediction(prob=1.0, prediction=True, confidence=1.0)
        return FailurePrediction(prob=0.0, prediction=False, confidence=1.0)

    def _features_to_obs(self, features: np.ndarray) -> RobotObservation:
        f = np.asarray(features, dtype=np.float32).reshape(-1)
        if f.shape[0] != STATE_FEATURE_DIM:
            raise ValueError(f"特征维度须为 {STATE_FEATURE_DIM}, 得到 {f.shape[0]}")
        policy = PolicyObservation(
            qpos=f[_QPOS].copy(),
            qvel=f[_QVEL].copy(),
            tcp_pose=f[_TCP].copy(),
            gripper=float(f[_GRIP]),
            images=None,
        )
        truth = GroundTruth(object_poses={}, success=False, is_grasped=False)
        step = len(self._history)
        return RobotObservation(policy=policy, truth=truth, step=step, timestamp=float(step) * 0.05)


class LearnedPredictor(FailurePredictor):
    """学习预测器占位 (待 Agent C)。可注入 predict_fn(features)->FailurePrediction|float。"""

    def __init__(
        self,
        predict_fn: Optional[Callable[[np.ndarray], object]] = None,
        threshold: float = 0.5,
    ) -> None:
        self._predict_fn = predict_fn
        self.threshold = float(threshold)

    def predict(self, features: np.ndarray) -> FailurePrediction:
        if self._predict_fn is None:
            raise NotImplementedError(
                "学习预测器待 Agent C 实现: 注入 predict_fn 或子类化 LearnedPredictor"
            )
        out = self._predict_fn(features)
        if isinstance(out, FailurePrediction):
            return out
        prob = float(out)
        return FailurePrediction(
            prob=prob,
            prediction=prob >= self.threshold,
            confidence=float(np.clip(abs(prob - 0.5) * 2.0, 0.0, 1.0)),
        )


def build_default_baselines(train_episodes: Sequence[EpisodeRecord]) -> Dict[str, FailurePredictor]:
    """构建三类基线 (majority / rule / learned 占位), 统一 FailurePredictor 接口。"""
    return {
        "majority": MajorityClassPredictor.fit(train_episodes),
        "rule": RuleBasedPredictor(),
        "learned": LearnedPredictor(),
    }


def evaluate_baselines(
    predictors: Dict[str, FailurePredictor],
    episodes: Sequence[EpisodeRecord],
    *,
    threshold: Optional[float] = None,
) -> Dict[str, dict]:
    """统一评估多基线: 返回每个基线的 raw counts + 聚合指标 (to_dict)。

    对 rule 类有状态基线, 每个 baseline 评估前独立 reset (若支持)。
    """
    from .learning_metrics import compute_prediction_metrics

    results: Dict[str, dict] = {}
    for name, pred in predictors.items():
        reset = getattr(pred, "reset", None)
        if callable(reset):
            reset()
        try:
            m = compute_prediction_metrics(episodes, pred, threshold=threshold)
            results[name] = m.to_dict()
        except NotImplementedError as e:
            results[name] = {"error": "not_implemented", "detail": str(e)}
    return results
