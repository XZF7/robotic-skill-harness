"""帧级失败预测指标测试: 显式分母 / raw counts / PR-AUC / latency / lead time。"""
import numpy as np
import pytest

from interfaces import (
    EpisodeRecord, FrameRecord, FailurePrediction, FailurePredictor,
)
from eval.learning_metrics import (
    average_precision, compute_prediction_metrics, DEFINITIONS,
)


def _episode(eid, seed, outcome, labels):
    frames = [
        FrameRecord(step=s, features=np.zeros(22, dtype=np.float32), label=int(l))
        for s, l in enumerate(labels)
    ]
    return EpisodeRecord(
        episode_id=eid, seed=seed, fault_type=None, outcome=outcome, frames=frames,
    )


class FixedPredictor(FailurePredictor):
    """按调用顺序返回预定的 (prob, prediction)。"""

    def __init__(self, preds):
        self.preds = list(preds)
        self.i = 0

    def predict(self, features):
        prob, pred = self.preds[self.i % len(self.preds)]
        self.i += 1
        return FailurePrediction(prob=float(prob), prediction=bool(pred))


class ConstantPredictor(FailurePredictor):
    def __init__(self, prob, pred):
        self.prob = float(prob)
        self.pred = bool(pred)

    def predict(self, features):
        return FailurePrediction(prob=self.prob, prediction=self.pred)


# labels [0,0,0,1,1] (失败 episode), 预测 [F,F,T,T,T] -> tp=2 fp=1 tn=2 fn=0
FAILED_EP = _episode("e", 0, outcome=False, labels=[0, 0, 0, 1, 1])


class TestAveragePrecision:
    def test_known_value(self):
        y_true = [0, 0, 1, 1]
        y_score = [0.1, 0.4, 0.35, 0.8]
        ap = average_precision(y_true, y_score)
        assert ap == pytest.approx(5 / 6, abs=1e-9)  # 0.83333...

    def test_perfect_ap(self):
        ap = average_precision([0, 0, 1, 1], [0.1, 0.2, 0.9, 0.95])
        assert ap == pytest.approx(1.0)

    def test_no_positive_is_none(self):
        assert average_precision([0, 0, 0], [0.1, 0.2, 0.3]) is None

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            average_precision([], [])


class TestConfusionAndRates:
    def test_exact_counts(self):
        pred = FixedPredictor([(0.1, False), (0.2, False), (0.8, True), (0.9, True), (0.95, True)])
        m = compute_prediction_metrics([FAILED_EP], pred)
        assert (m.tp, m.fp, m.tn, m.fn) == (2, 1, 2, 0)
        assert m.num_positive == 2 and m.num_negative == 3 and m.num_frames == 5
        assert m.precision == pytest.approx(2 / 3)
        assert m.recall == pytest.approx(1.0)
        assert m.f1 == pytest.approx(0.8)
        assert m.fp_rate == pytest.approx(1 / 3)
        assert m.fn_rate == pytest.approx(0.0)
        assert m.pr_auc == pytest.approx(1.0)

    def test_denominators_explicit(self):
        pred = FixedPredictor([(0.1, False), (0.2, False), (0.8, True), (0.9, True), (0.95, True)])
        m = compute_prediction_metrics([FAILED_EP], pred)
        d = m.denominators
        assert d["precision"] == m.tp + m.fp == 3
        assert d["recall"] == m.num_positive == 2
        assert d["fp_rate"] == m.num_negative == 3
        assert d["fn_rate"] == m.num_positive == 2
        # 每个聚合指标都有显式定义。
        for k in ("precision", "recall", "f1", "pr_auc", "fp_rate", "fn_rate",
                  "prediction_latency_steps", "lead_time_steps"):
            assert k in DEFINITIONS, f"{k} 缺少分母定义"

    def test_never_fires(self):
        pred = ConstantPredictor(prob=0.0, pred=False)
        m = compute_prediction_metrics([FAILED_EP], pred)
        assert (m.tp, m.fp, m.tn, m.fn) == (0, 0, 3, 2)
        assert m.precision == 0.0 and m.recall == 0.0 and m.f1 == 0.0
        assert m.fp_rate == 0.0 and m.fn_rate == 1.0
        assert m.num_missed_failures == 1
        assert m.num_failure_episodes_with_fire == 0
        assert m.prediction_latency_mean_steps is None
        assert m.lead_time_mean_steps is None

    def test_always_fires(self):
        pred = ConstantPredictor(prob=1.0, pred=True)
        m = compute_prediction_metrics([FAILED_EP], pred)
        assert (m.tp, m.fp, m.tn, m.fn) == (2, 3, 0, 0)
        assert m.precision == pytest.approx(2 / 5)
        assert m.recall == pytest.approx(1.0)
        assert m.fp_rate == pytest.approx(1.0)
        assert m.fn_rate == pytest.approx(0.0)


class TestLatencyAndLeadTime:
    def test_late_fire(self):
        # 只在 step4 报警 (label onset 在 step3) -> latency=1, lead=0。
        pred = FixedPredictor([(0.0, False), (0.0, False), (0.0, False), (0.0, False), (0.9, True)])
        m = compute_prediction_metrics([FAILED_EP], pred)
        assert m.latency_steps == [1]
        assert m.lead_time_steps == [0]
        assert m.prediction_latency_mean_steps == 1.0
        assert m.lead_time_mean_steps == 0.0

    def test_early_fire(self):
        # 在 step0 就报警 -> latency 夹到 0, lead=4。
        pred = FixedPredictor([(0.9, True), (0.0, False), (0.0, False), (0.0, False), (0.0, False)])
        m = compute_prediction_metrics([FAILED_EP], pred)
        assert m.latency_steps == [0]
        assert m.lead_time_steps == [4]
        assert m.lead_time_mean_steps == 4.0

    def test_success_episode_not_counted(self):
        ok = _episode("ok", 1, outcome=True, labels=[0, 0, 0, 0])
        pred = ConstantPredictor(prob=0.0, pred=False)
        m = compute_prediction_metrics([FAILED_EP, ok], pred)
        assert m.num_failure_episodes == 1  # 只统计失败 episode


class TestThresholdOverride:
    def test_threshold_rebinarizes(self):
        # predictor 自带 prediction=False, 但 prob 高; threshold=0.5 时重新二值化。
        pred = ConstantPredictor(prob=0.9, pred=False)
        m = compute_prediction_metrics([FAILED_EP], pred, threshold=0.5)
        assert (m.tp, m.fp, m.tn, m.fn) == (2, 3, 0, 0)
