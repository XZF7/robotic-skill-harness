"""三类基线统一接口测试: majority / 规则式 / 学习式占位。"""
import numpy as np
import pytest

from interfaces import FailurePrediction, FailurePredictor
from eval.learning_baselines import (
    MajorityClassPredictor, RuleBasedPredictor, LearnedPredictor,
    build_default_baselines, evaluate_baselines,
)
from eval.learning_metrics import compute_prediction_metrics
from eval.learning_synthetic import make_synthetic_episode, make_synthetic_episodes


class TestMajority:
    def test_fit_prevalence_and_predicts_majority(self):
        # 3 success (全 0 标签) + 1 failed -> 正样本占少数 -> 预测 0。
        eps = [
            make_synthetic_episode("s0", 0, None, failed=False),
            make_synthetic_episode("s1", 1, None, failed=False),
            make_synthetic_episode("s2", 2, None, failed=False),
            make_synthetic_episode("f0", 3, None, failed=True),
        ]
        labels = [f.label for ep in eps for f in ep.frames]
        p = MajorityClassPredictor.fit(eps)
        assert p.prob == pytest.approx(sum(labels) / len(labels))
        out = p.predict(np.zeros(22, dtype=np.float32))
        assert isinstance(out, FailurePrediction)
        assert out.prediction is False  # 主导类=负类

    def test_fit_empty_raises(self):
        with pytest.raises(ValueError):
            MajorityClassPredictor.fit([])


class TestRuleBaseline:
    def test_is_failure_predictor(self):
        r = RuleBasedPredictor()
        assert isinstance(r, FailurePredictor)

    def test_detects_gripper_close_failure(self):
        r = RuleBasedPredictor()
        r.reset()
        failed = make_synthetic_episode("f", 5, None, failed=True, failure_mode="gripper_close")
        m = compute_prediction_metrics([failed], r)
        assert m.num_failure_episodes_with_fire >= 1
        assert m.tp >= 1
        assert m.tp + m.fp >= 1  # 至少报警一次

    def test_no_false_positive_on_success(self):
        r = RuleBasedPredictor()
        r.reset()
        ok = make_synthetic_episode("ok", 6, None, failed=False)
        m = compute_prediction_metrics([ok], r)
        assert m.tp + m.fp == 0  # 成功 episode 全程不报警
        assert m.fp == 0

    def test_reset_clears_history(self):
        r = RuleBasedPredictor()
        r.reset()
        ok = make_synthetic_episode("ok", 7, None, failed=False)
        compute_prediction_metrics([ok], r)
        r.reset()
        ok2 = make_synthetic_episode("ok2", 8, None, failed=False)
        m2 = compute_prediction_metrics([ok2], r)
        assert m2.fp == 0  # 历史已清, 不受前一个 episode 影响

    def test_bad_feature_dim_raises(self):
        r = RuleBasedPredictor()
        with pytest.raises(ValueError):
            r.predict(np.zeros(5, dtype=np.float32))


class TestLearnedPlaceholder:
    def test_placeholder_raises(self):
        p = LearnedPredictor()
        with pytest.raises(NotImplementedError):
            p.predict(np.zeros(22, dtype=np.float32))

    def test_injected_predict_fn(self):
        p = LearnedPredictor(predict_fn=lambda f: 0.9)
        out = p.predict(np.zeros(22, dtype=np.float32))
        assert isinstance(out, FailurePrediction)
        assert out.prob == pytest.approx(0.9)
        assert out.prediction is True

    def test_injected_predictor_object(self):
        p = LearnedPredictor(predict_fn=lambda f: FailurePrediction(prob=0.3, prediction=False))
        out = p.predict(np.zeros(22, dtype=np.float32))
        assert out.prediction is False and out.prob == pytest.approx(0.3)


class TestEvaluateBaselines:
    def test_uniform_interface(self):
        train = make_synthetic_episodes(seeds=list(range(10)), fault_types=["a", "b"], failed_frac=0.5)
        test = make_synthetic_episodes(seeds=list(range(10, 20)), fault_types=["a", "b"], failed_frac=0.5)
        preds = build_default_baselines(train)
        assert set(preds) == {"majority", "rule", "learned"}
        results = evaluate_baselines(preds, test)
        # majority 与 rule 有完整指标; learned 占位返回 not_implemented。
        assert "counts" in results["majority"]
        assert "metrics" in results["rule"]
        assert results["learned"].get("error") == "not_implemented"

    def test_all_baselines_same_shape(self):
        train = make_synthetic_episodes(seeds=[0, 1, 2, 3], fault_types=["a"], failed_frac=0.5)
        test = make_synthetic_episodes(seeds=[4, 5], fault_types=["a"], failed_frac=0.5)
        results = evaluate_baselines(build_default_baselines(train), test)
        for name in ("majority", "rule"):
            assert results[name]["counts"]["num_frames"] == sum(len(e.frames) for e in test)
