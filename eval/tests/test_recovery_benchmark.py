"""B0/B1 matched 恢复对比 smoke test (M3-D)。"""
import pytest

from eval.recovery_benchmark import (
    match_key, RecoveryBenchmarkConfig, RecoveryComparison,
    compare_b0_b1, label_condition,
)
from eval.mocks import make_mock_fault_episode, make_mock_fault_episodes


def _b0(seeds, fault_type="grasp_perturb", privileged_access=False):
    """B0: 无恢复基线, 全部失败, 未检测。"""
    return make_mock_fault_episodes(
        "PickCube-v1", fault_type, seeds,
        baseline="b0", injected=True, success=False,
        recovered=False, attempts=1, detected=False,
        privileged_access=privileged_access,
    )


def _b1(seeds, recovered_seeds, fault_type="grasp_perturb", privileged_access=False):
    """B1: 规则恢复, 全部检测到, recovered_seeds 中的 seed 恢复成功。"""
    eps = []
    for seed in seeds:
        eps.append(make_mock_fault_episode(
            "PickCube-v1", seed=seed, fault_type=fault_type, injected=True,
            success=(seed in recovered_seeds), recovered=(seed in recovered_seeds),
            attempts=(2 if seed in recovered_seeds else 3),
            detected=True, baseline="b1", privileged_access=privileged_access,
        ))
    return eps


class TestMatchKey:
    def test_same_config_same_key(self):
        a = make_mock_fault_episode("t", seed=1, fault_type="f", fault_params={"d": 0.1})
        b = make_mock_fault_episode("t", seed=1, fault_type="f", fault_params={"d": 0.1})
        assert match_key(a) == match_key(b)

    def test_different_seed_or_params_diff_key(self):
        base = dict(task_id="t", seed=1, fault_type="f", fault_params={"d": 0.1})
        a = make_mock_fault_episode(**base)
        b = make_mock_fault_episode(**{**base, "seed": 2})
        c = make_mock_fault_episode(**{**base, "fault_params": {"d": 0.2}})
        d = make_mock_fault_episode(**{**base, "fault_type": "object_displacement"})
        assert match_key(a) != match_key(b)
        assert match_key(a) != match_key(c)
        assert match_key(a) != match_key(d)

    def test_privileged_in_key(self):
        a = make_mock_fault_episode("t", seed=1, fault_type="f", privileged_access=True)
        b = make_mock_fault_episode("t", seed=1, fault_type="f", privileged_access=False)
        assert match_key(a) != match_key(b)


class TestCompareB0B1:
    def test_matched_comparison_and_deltas(self):
        seeds = [0, 1, 2, 3]
        b0 = _b0(seeds)
        b1 = _b1(seeds, recovered_seeds={0, 1})
        cmp = compare_b0_b1(b0, b1)

        assert cmp.matched_pairs == 4
        assert cmp.b0_unmatched == 0 and cmp.b1_unmatched == 0
        assert cmp.privileged_access is False

        assert cmp.b0.task_success_rate == pytest.approx(0.0)
        assert cmp.b1.task_success_rate == pytest.approx(0.5)
        assert cmp.b0.recovery_success_rate == pytest.approx(0.0)
        assert cmp.b1.recovery_success_rate == pytest.approx(0.5)
        assert cmp.b0.failure_detection_rate == pytest.approx(0.0)
        assert cmp.b1.failure_detection_rate == pytest.approx(1.0)

        assert cmp.deltas["task_success_rate"] == pytest.approx(0.5)
        assert cmp.deltas["recovery_success_rate"] == pytest.approx(0.5)
        assert cmp.deltas["failure_detection_rate"] == pytest.approx(1.0)
        # B1 有恢复尝试, B0 无 -> avg_retry_count 差值不可比 (B0 None)
        assert cmp.deltas["avg_retry_count"] is None

    def test_unmatched_episodes_counted_not_dropped(self):
        seeds = [0, 1]
        b0 = _b0(seeds)
        b1 = _b1(seeds, recovered_seeds={0}) + make_mock_fault_episodes(
            "PickCube-v1", "grasp_perturb", [99], baseline="b1",
            injected=True, success=False, recovered=False, attempts=1,
            detected=False, privileged_access=False,
        )
        cmp = compare_b0_b1(b0, b1)
        assert cmp.matched_pairs == 2
        assert cmp.b1_unmatched == 1

    def test_cross_privileged_raises(self):
        b0 = _b0([0, 1], privileged_access=True)
        b1 = _b1([0, 1], recovered_seeds={0}, privileged_access=False)
        with pytest.raises(ValueError, match="privileged_access"):
            compare_b0_b1(b0, b1)

    def test_oracle_vs_observational_label(self):
        assert label_condition(True) == "oracle"
        assert label_condition(False) == "observational"
        assert label_condition(None) == "unlabeled"

    def test_per_fault_type_breakdown(self):
        b0 = _b0([0, 1], "grasp_perturb") + _b0([0, 1], "object_displacement")
        b1 = _b1([0, 1], {0}, "grasp_perturb") + _b1([0, 1], {0}, "object_displacement")
        cmp = compare_b0_b1(b0, b1)
        assert set(cmp.per_fault_type.keys()) == {"grasp_perturb", "object_displacement"}
        assert cmp.per_fault_type["grasp_perturb"].matched_pairs == 2


class TestConfig:
    def test_valid_and_iter_cases(self):
        cfg = RecoveryBenchmarkConfig(
            name="m3", fault_scenarios=[("grasp_perturb", {"d": 0.1})],
            seeds=list(range(10)), privileged_access=False,
        )
        assert cfg.total_cases == 2 * 1 * 10 * 1  # b0+b1, 1 scenario, 10 seeds
        cases = list(cfg.iter_cases())
        assert len(cases) == 20
        assert cases[0][0] == "b0" and cases[0][2] == {"d": 0.1} and cases[0][3] == 0
        assert cases[-1][0] == "b1" and cases[-1][3] == 9

    @pytest.mark.parametrize("bad", [
        dict(name=""),
        dict(fault_scenarios=[]),
        dict(seeds=[]),
        dict(fault_scenarios=[("", {})]),
        dict(conditions=("b2",)),
        dict(episodes_per_seed=0),
    ])
    def test_invalid(self, bad):
        defaults = dict(name="m3", fault_scenarios=[("f", {})], seeds=[0, 1])
        defaults.update(bad)
        with pytest.raises(ValueError):
            RecoveryBenchmarkConfig(**defaults)
