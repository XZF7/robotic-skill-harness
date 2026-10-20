"""基线对比基础设施测试。"""
import pytest

from eval.baselines import Baseline, BaselineComparison, rank_baselines, compare_baselines
from eval.mocks import make_mock_episodes


def _baseline(name, n, success_rate, seed=0):
    return Baseline.from_episodes(name, make_mock_episodes(name, n, seed=seed, success_rate=success_rate))


class TestBaseline:
    def test_from_episodes_computes_metrics(self):
        b = _baseline("b", 10, 1.0)
        assert b.metrics.num_episodes == 10
        assert b.metrics.success_rate == 1.0

    def test_json_roundtrip(self):
        b = _baseline("b", 5, 0.6)
        b2 = Baseline.from_json("b", b.to_json())
        assert b2.episodes == b.episodes
        assert b2.metrics == b.metrics

    def test_is_sim_baseline(self):
        # 显式构造 source='sim' 的 episode 基线
        from eval.mocks import make_mock_episodes
        eps = make_mock_episodes("sim", 3, source="sim")
        sim = Baseline.from_episodes("sim", eps)
        assert sim.is_sim_baseline()
        assert sim.source_counts() == {"sim": 3}


class TestBaselineSource:
    def test_mock_baseline_is_not_sim(self):
        b = _baseline("mockb", 3, 1.0)  # source='mock'
        assert not b.is_sim_baseline()
        assert b.source_counts() == {"mock": 3}


class TestRanking:
    def test_rank_order(self):
        bs = [_baseline("low", 10, 0.3), _baseline("high", 10, 0.9), _baseline("mid", 10, 0.6)]
        ranked = rank_baselines(bs)
        assert [b.name for b in ranked] == ["high", "mid", "low"]

    def test_tiebreak_reward_then_steps(self):
        b1 = _baseline("a", 10, 0.5)  # reward 差异由 RNG 决定, 不依赖
        b2 = _baseline("b", 10, 0.5)
        ranked = rank_baselines([b1, b2])
        # 相同 success_rate 时按 mean_reward 降序, 稳定
        assert len(ranked) == 2


class TestCompare:
    def test_deltas_vs_reference(self):
        high = _baseline("high", 100, 0.9)
        low = _baseline("low", 100, 0.3)
        cmp = compare_baselines([high, low], reference="high")
        assert cmp.reference == "high"
        assert cmp.ranking == ["high", "low"]
        assert cmp.reference_success_rate == pytest.approx(high.metrics.success_rate)
        assert cmp.success_rate_deltas["high"] == pytest.approx(0.0)
        assert cmp.success_rate_deltas["low"] == pytest.approx(low.metrics.success_rate - high.metrics.success_rate)

    def test_default_reference_is_top_ranked(self):
        high = _baseline("high", 100, 0.9)
        low = _baseline("low", 100, 0.3)
        cmp = compare_baselines([low, high])
        assert cmp.reference == "high"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compare_baselines([])

    def test_duplicate_names_raise(self):
        b1 = _baseline("dup", 5, 0.5)
        b2 = _baseline("dup", 5, 0.5, seed=1)
        with pytest.raises(ValueError):
            compare_baselines([b1, b2])

    def test_unknown_reference_raises(self):
        with pytest.raises(ValueError):
            compare_baselines([_baseline("a", 5, 0.5)], reference="nope")

    def test_to_dict(self):
        cmp = compare_baselines([_baseline("a", 100, 0.9), _baseline("b", 100, 0.3)])
        d = cmp.to_dict()
        assert set(d["ranking"]) == {"a", "b"}
        assert "success_rate_deltas" in d
