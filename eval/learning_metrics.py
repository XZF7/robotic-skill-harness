"""帧级失败预测指标 (M4-D 学习评估协议)。

所有指标的分母在 DEFINITIONS 显式定义, 返回结果同时携带 numerators 与
denominators, 绝不隐式。帧级指标跨所有 episode 展平计算; 时间性指标
(prediction_latency / lead_time) 仅在失败 episode 上度量。

阈值规则:
- threshold=None: 直接采用 predictor 返回的 FailurePrediction.prediction (bool)。
- threshold=float: 由 prob >= threshold 重新二值化 (供阈值扫描/学习预测器调参)。
PR-AUC 恒用 prob (分数) 计算, 不依赖二值化阈值。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from interfaces import (
    EpisodeRecord, FailurePredictor, DEFAULT_PREDICTION_HORIZON,
)

# 每个指标 -> 分母的精确文字定义。
DEFINITIONS: Dict[str, str] = {
    "precision": "TP / (TP+FP), 分母=预测为正的帧数; 无正预测时为 0.0",
    "recall": "TP / (TP+FN), 分母=真实正样本帧数(label=1); 无正样本时为 0.0",
    "f1": "2*precision*recall/(precision+recall); precision+recall=0 时为 0.0",
    "pr_auc": "average precision: 按 prob 降序的 PR 曲线下面积(梯形积分); 无正样本时为 None(未定义)",
    "fp_rate": "FP / (FP+TN), 分母=真实负样本帧数(label=0); 无负样本时为 0.0",
    "fn_rate": "FN / (FN+TP), 分母=真实正样本帧数(label=1); 无正样本时为 0.0",
    "prediction_latency_steps": "sum(max(0, first_pred_step - first_label_pos_step)) / "
        "num_failure_episodes_with_fire, 分母=预测器至少报警一次的失败 episode 数",
    "lead_time_steps": "sum(t_end - first_pred_step) / num_failure_episodes_with_fire, "
        "分母=预测器至少报警一次的失败 episode 数; t_end=该 episode 最大 step, 越大越早预警",
}


def average_precision(y_true: Sequence[int], y_score: Sequence[float]) -> Optional[float]:
    """手写 average precision (PR-AUC), 不依赖 sklearn。

    按分数降序排列, 对每个正样本计算 (recall 增量) * (该点及之后的最大 precision)。
    无正样本时返回 None (未定义)。与 sklearn.metrics.average_precision_score 一致。
    """
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_score = np.asarray(y_score, dtype=np.float64).reshape(-1)
    if y_true.shape[0] == 0 or y_true.shape[0] != y_score.shape[0]:
        raise ValueError(f"y_true/y_score 长度不一致: {y_true.shape[0]} vs {y_score.shape[0]}")
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return None

    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    tp = np.cumsum(y_true)
    fp = np.cumsum(1.0 - y_true)
    denom = tp + fp  # = rank, 恒 >= 1
    precision = tp / denom
    recall = tp / n_pos

    # 单调化 precision (取每个 recall 水平之后的最大 precision)。
    prec = precision.copy()
    for i in range(len(prec) - 1, 0, -1):
        prec[i - 1] = max(prec[i - 1], prec[i])

    ap = 0.0
    prev_recall = 0.0
    for i in range(len(y_true)):
        if y_true[i]:
            ap += (recall[i] - prev_recall) * prec[i]
            prev_recall = recall[i]
    return float(ap)


@dataclass
class PredictionMetrics:
    """帧级失败预测指标 + 显式 numerator/denominator。"""

    num_frames: int
    num_positive: int
    num_negative: int
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    pr_auc: Optional[float]
    fp_rate: float
    fn_rate: float
    prediction_latency_mean_steps: Optional[float]
    lead_time_mean_steps: Optional[float]
    num_failure_episodes: int
    num_failure_episodes_with_fire: int
    num_missed_failures: int
    latency_steps: List[float] = field(default_factory=list)
    lead_time_steps: List[float] = field(default_factory=list)
    numerators: Dict[str, int] = field(default_factory=dict)
    denominators: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "counts": {
                "num_frames": self.num_frames,
                "num_positive": self.num_positive,
                "num_negative": self.num_negative,
                "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
                "num_failure_episodes": self.num_failure_episodes,
                "num_failure_episodes_with_fire": self.num_failure_episodes_with_fire,
                "num_missed_failures": self.num_missed_failures,
            },
            "metrics": {
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1,
                "pr_auc": self.pr_auc,
                "fp_rate": self.fp_rate,
                "fn_rate": self.fn_rate,
                "prediction_latency_mean_steps": self.prediction_latency_mean_steps,
                "lead_time_mean_steps": self.lead_time_mean_steps,
            },
            "latency_steps": list(self.latency_steps),
            "lead_time_steps": list(self.lead_time_steps),
            "numerators": self.numerators,
            "denominators": self.denominators,
            "definitions": DEFINITIONS,
        }


def compute_prediction_metrics(
    episodes: Sequence[EpisodeRecord],
    predictor: FailurePredictor,
    *,
    horizon: int = DEFAULT_PREDICTION_HORIZON,
    threshold: Optional[float] = None,
) -> PredictionMetrics:
    """在给定 episodes 上评估 predictor 的帧级失败预测指标。

    threshold=None 用 predictor.prediction; 否则 prob >= threshold 二值化。
    horizon 仅用于文档/语义一致性 (labels 已由 label_failure 生成)。
    """
    y_true: List[int] = []
    y_pred: List[int] = []
    y_score: List[float] = []

    latency_vals: List[float] = []
    lead_vals: List[float] = []
    num_failure_episodes = 0
    num_failure_episodes_with_fire = 0

    for ep in episodes:
        if not ep.frames:
            continue
        # 每个 episode 是独立轨迹: 有状态预测器 (如规则式历史 / 循环学习器) 必须按
        # episode 重置, 防止跨轨迹历史泄漏 (如 DROP 检测跨 episode 误报)。
        reset = getattr(predictor, "reset", None)
        if callable(reset):
            reset()
        labels = [f.label for f in ep.frames]
        preds: List[int] = []
        scores: List[float] = []
        for f in ep.frames:
            p = predictor.predict(f.features)
            prob = float(p.prob)
            scores.append(prob)
            if threshold is None:
                preds.append(1 if bool(p.prediction) else 0)
            else:
                preds.append(1 if prob >= threshold else 0)
        y_true.extend(labels)
        y_pred.extend(preds)
        y_score.extend(scores)

        if (not ep.outcome) and any(labels):
            num_failure_episodes += 1
            t_label = min(f.step for f in ep.frames if f.label == 1)
            fired = [f.step for f, b in zip(ep.frames, preds) if b]
            if fired:
                t_pred = min(fired)
                num_failure_episodes_with_fire += 1
                latency_vals.append(float(max(0, t_pred - t_label)))
                t_end = max(f.step for f in ep.frames)
                lead_vals.append(float(t_end - t_pred))
            # else: 该失败 episode 未被报警 -> num_missed_failures

    n = len(y_true)
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    n_pos = tp + fn
    n_neg = tn + fp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / n_pos if n_pos else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fp_rate = fp / n_neg if n_neg else 0.0
    fn_rate = fn / n_pos if n_pos else 0.0
    pr_auc = average_precision(y_true, y_score) if n_pos else None
    latency_mean = (sum(latency_vals) / len(latency_vals)) if latency_vals else None
    lead_mean = (sum(lead_vals) / len(lead_vals)) if lead_vals else None

    return PredictionMetrics(
        num_frames=n,
        num_positive=n_pos,
        num_negative=n_neg,
        tp=tp, fp=fp, tn=tn, fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        pr_auc=pr_auc,
        fp_rate=fp_rate,
        fn_rate=fn_rate,
        prediction_latency_mean_steps=latency_mean,
        lead_time_mean_steps=lead_mean,
        num_failure_episodes=num_failure_episodes,
        num_failure_episodes_with_fire=num_failure_episodes_with_fire,
        num_missed_failures=num_failure_episodes - num_failure_episodes_with_fire,
        latency_steps=latency_vals,
        lead_time_steps=lead_vals,
        numerators={
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "sum_latency": int(sum(latency_vals)),
            "sum_lead_time": int(sum(lead_vals)),
        },
        denominators={
            "precision": tp + fp,
            "recall": n_pos,
            "fp_rate": n_neg,
            "fn_rate": n_pos,
            "prediction_latency_steps": num_failure_episodes_with_fire,
            "lead_time_steps": num_failure_episodes_with_fire,
        },
    )
