"""故障注入 episode 消费 + 恢复指标计算 smoke test (M3-D)。

所有分母显式断言 (m.denominators / m.numerators), 不靠隐式假设。
"""
import pytest

from interfaces import (
    EpisodeResult, FaultRecord, RecoveryResult, RecoveryDecision, RecoveryAction,
    SkillResult, SkillStatus, FailureEvent, FailureType,
)
from eval.fault_injection import (
    DEFINITIONS, RecoveryMetrics, compute_recovery_metrics,
    attach_fault, attach_recovery, attach_privileged, attach_baseline,
    extract_fault, extract_privileged, extract_seed, extract_baseline,
    failure_manifested, is_valid_injected_failure, failure_detected, recovery_attempted,
    split_by_privileged,
)
from eval.serialization import (
    fault_record_to_dict, fault_record_from_dict,
    recovery_result_to_dict, recovery_result_from_dict,
    recovery_decision_to_dict, recovery_decision_from_dict,
)
from eval.mocks import make_mock_fault_episode


# ---- 序列化 round-trip -----------------------------------------------------

class TestContractSerialization:
    def test_fault_record_roundtrip(self):
        f = FaultRecord(fault_type="object_displacement", injected=True, step=5,
                        params={"delta": 0.05}, detail={"reason": "ok"})
        assert fault_record_from_dict(fault_record_to_dict(f)) == f

    def test_recovery_result_roundtrip(self):
        sr = SkillResult(SkillStatus.SUCCESS, "pick", 10)
        r = RecoveryResult(
            attempts=3,
            final_skill_result=sr,
            decisions=[RecoveryDecision(RecoveryAction.RETRY, retry_count=1, reason="grip_fail")],
            recovered=True, aborted=False,
        )
        assert recovery_result_from_dict(recovery_result_to_dict(r)) == r

    def test_recovery_decision_roundtrip(self):
        d = RecoveryDecision(RecoveryAction.ABORT, retry_count=2, reason="stuck")
        assert recovery_decision_from_dict(recovery_decision_to_dict(d)) == d


# ---- 挂载/提取 -------------------------------------------------------------

class TestAttachExtract:
    def test_attach_extract_fault(self):
        ep = EpisodeResult(task_id="t", success=False, steps=10, reward=0.0)
        attach_fault(ep, FaultRecord("grasp_perturb", True, step=3, params={"a": 1}))
        f = extract_fault(ep)
        assert f.fault_type == "grasp_perturb" and f.injected and f.step == 3
        assert f.params == {"a": 1}

    def test_extract_accepts_fault_record_object(self):
        ep = EpisodeResult(task_id="t", success=False, steps=10, reward=0.0)
        ep.metrics["fault"] = FaultRecord("grasp_perturb", True, step=3)
        assert extract_fault(ep).fault_type == "grasp_perturb"

    def test_extract_fault_none(self):
        assert extract_fault(EpisodeResult(task_id="t", success=False, steps=1, reward=0.0)) is None

    def test_extract_privileged_and_seed(self):
        ep = EpisodeResult(task_id="t", success=False, steps=1, reward=0.0,
                           metrics={"seed": 7})
        attach_privileged(ep, True)
        assert extract_privileged(ep) is True
        assert extract_seed(ep) == 7

    def test_attach_baseline_validates(self):
        ep = EpisodeResult(task_id="t", success=False, steps=1, reward=0.0)
        with pytest.raises(ValueError):
            attach_baseline(ep, "b2")


# ---- 谓词 ------------------------------------------------------------------

class TestPredicates:
    def test_failure_manifested_via_success_false(self):
        ep = EpisodeResult(task_id="t", success=False, steps=1, reward=0.0)
        assert failure_manifested(ep)

    def test_failure_manifested_via_failure_event(self):
        ep = EpisodeResult(task_id="t", success=True, steps=1, reward=1.0,
                           failures=[FailureEvent(FailureType.DROP, 5, 0.25)])
        assert failure_manifested(ep)

    def test_failure_manifested_via_skill_status(self):
        ep = EpisodeResult(task_id="t", success=True, steps=1, reward=1.0,
                           skill_results=[SkillResult(SkillStatus.TIMEOUT, "pick", 1)])
        assert failure_manifested(ep)

    def test_no_failure_when_clean_success(self):
        ep = EpisodeResult(task_id="t", success=True, steps=1, reward=1.0,
                           skill_results=[SkillResult(SkillStatus.SUCCESS, "pick", 1)])
        assert not failure_manifested(ep)

    def test_valid_injected_failure_requires_injected_and_manifested(self):
        # 注入成功 + 失败真实发生 -> True
        ep = EpisodeResult(task_id="t", success=False, steps=1, reward=0.0)
        attach_fault(ep, FaultRecord("grasp_perturb", True))
        assert is_valid_injected_failure(ep)
        # 注入未成功 (injected=False) -> False
        ep2 = EpisodeResult(task_id="t", success=False, steps=1, reward=0.0)
        attach_fault(ep2, FaultRecord("grasp_perturb", False))
        assert not is_valid_injected_failure(ep2)
        # 注入成功但失败未发生 (success=True) -> False
        ep3 = EpisodeResult(task_id="t", success=True, steps=1, reward=1.0)
        attach_fault(ep3, FaultRecord("grasp_perturb", True))
        assert not is_valid_injected_failure(ep3)

    def test_failure_detected_inferred_from_recovery(self):
        ep = EpisodeResult(task_id="t", success=False, steps=1, reward=0.0)
        sr = SkillResult(SkillStatus.FAILURE, "pick", 1)
        attach_recovery(ep, RecoveryResult(
            attempts=2, final_skill_result=sr,
            decisions=[RecoveryDecision(RecoveryAction.RETRY, 1)], recovered=False))
        assert failure_detected(ep)
        assert recovery_attempted(ep)

    def test_failure_detected_explicit_flag_wins(self):
        ep = EpisodeResult(task_id="t", success=False, steps=1, reward=0.0,
                           metrics={"detected": False})
        # 无 recovery 时显式 detected=False -> False
        assert not failure_detected(ep)

    def test_b0_never_detected(self):
        ep = make_mock_fault_episode("t", seed=0, fault_type="f", baseline="b0",
                                     detected=False, attempts=1, success=False)
        assert not failure_detected(ep)


# ---- compute_recovery_metrics (显式分母) -----------------------------------

def _scenario_episodes():
    """构造一组 10 个 episode, 已知各标志, 用于验证指标与分母。

    布局 (全部 fault_type='grasp_perturb', injected=True, privileged_access=False):
    - seed 0,1: 注入但任务仍成功 (无失败发生) -> 不计入 real_failures
    - seed 2,3: 失败发生但未被检测 (detected=False, attempts=1) -> real failure, 未检测, 未恢复
    - seed 4,5,6,7: 失败发生, 检测到, 恢复成功 (recovered=True, attempts=2)
    - seed 8,9: 失败发生, 检测到, 恢复失败 (recovered=False, attempts=3, aborted)
    """
    eps = []
    # 注入但成功
    for seed in (0, 1):
        eps.append(make_mock_fault_episode(
            "PickCube-v1", seed=seed, fault_type="grasp_perturb", injected=True,
            success=True, attempts=1, recovered=False, detected=False,
            baseline="b1", privileged_access=False))
    # 失败未检测
    for seed in (2, 3):
        eps.append(make_mock_fault_episode(
            "PickCube-v1", seed=seed, fault_type="grasp_perturb", injected=True,
            success=False, attempts=1, recovered=False, detected=False,
            baseline="b1", privileged_access=False))
    # 失败检测 + 恢复成功
    for seed in (4, 5, 6, 7):
        eps.append(make_mock_fault_episode(
            "PickCube-v1", seed=seed, fault_type="grasp_perturb", injected=True,
            success=False, attempts=2, recovered=True, detected=True,
            baseline="b1", privileged_access=False))
    # 失败检测 + 恢复失败
    for seed in (8, 9):
        eps.append(make_mock_fault_episode(
            "PickCube-v1", seed=seed, fault_type="grasp_perturb", injected=True,
            success=False, attempts=3, recovered=False, detected=True,
            baseline="b1", privileged_access=False))
    return eps


class TestComputeRecoveryMetrics:
    def test_metric_values_and_denominators(self):
        eps = _scenario_episodes()
        m = compute_recovery_metrics(eps)

        assert m.num_episodes == 10
        assert m.privileged_access is False

        # 指标值
        assert m.task_success_rate == pytest.approx(2 / 10)
        assert m.verified_fault_injection_rate == pytest.approx(1.0)
        assert m.failure_detection_rate == pytest.approx(6 / 8)
        assert m.recovery_success_rate == pytest.approx(4 / 8)
        assert m.avg_retry_count == pytest.approx(8 / 6)  # retries = attempts-1: 4*1 + 2*2 = 8
        assert m.recovery_latency_mean_steps == pytest.approx(40.0)
        assert m.mean_episode_length == pytest.approx(80.0)
        assert m.failure_category_counts == {"grip_fail": 8}
        assert m.fault_type_counts == {"grasp_perturb": 10}

        # 显式分母
        assert m.denominators["task_success_rate"] == 10
        assert m.denominators["verified_fault_injection_rate"] == 10
        assert m.denominators["failure_detection_rate"] == 8
        assert m.denominators["recovery_success_rate"] == 8
        assert m.denominators["avg_retry_count"] == 6
        assert m.denominators["mean_episode_length"] == 10
        assert m.denominators["mean_cumulative_reward"] == 10

        # 分子 (numerators)
        assert m.numerators["num_success"] == 2
        assert m.numerators["num_injected"] == 10
        assert m.numerators["num_detected"] == 6
        assert m.numerators["num_real_failures"] == 8
        assert m.numerators["num_recovered"] == 4
        assert m.numerators["num_recovery_attempted"] == 6

    def test_empty_returns_zero_metrics(self):
        m = compute_recovery_metrics([])
        assert m.num_episodes == 0
        assert m.task_success_rate == 0.0
        assert m.verified_fault_injection_rate is None
        assert m.failure_detection_rate is None
        assert m.recovery_success_rate is None

    def test_no_injection_attempted_vfir_none(self):
        ep = EpisodeResult(task_id="t", success=True, steps=10, reward=1.0)
        m = compute_recovery_metrics([ep])
        assert m.verified_fault_injection_rate is None
        assert m.denominators["verified_fault_injection_rate"] == 0

    def test_recovery_success_only_on_injected_failures(self):
        # 3 个失败场景: 1 个注入未成功 (injected=False), 1 个注入成功但未失败, 1 个注入成功且失败
        eps = [
            make_mock_fault_episode("t", seed=0, fault_type="f", injected=False,
                                    success=False, recovered=True, attempts=2, detected=True,
                                    baseline="b1", privileged_access=False),
            make_mock_fault_episode("t", seed=1, fault_type="f", injected=True,
                                    success=True, recovered=False, attempts=1, detected=False,
                                    baseline="b1", privileged_access=False),
            make_mock_fault_episode("t", seed=2, fault_type="f", injected=True,
                                    success=False, recovered=True, attempts=2, detected=True,
                                    baseline="b1", privileged_access=False),
        ]
        m = compute_recovery_metrics(eps)
        # 只有 seed=2 是有效注入失败场景 -> RSR 分母=1, 分子=1 (recovered)
        assert m.denominators["recovery_success_rate"] == 1
        assert m.numerators["num_injected_failure_scenarios"] == 1
        assert m.recovery_success_rate == pytest.approx(1.0)
        # 注入尝试 = 3 (三个都带 FaultRecord), 注入成功 = 2
        assert m.denominators["verified_fault_injection_rate"] == 3
        assert m.numerators["num_injected"] == 2
        assert m.verified_fault_injection_rate == pytest.approx(2 / 3)

    def test_to_dict_includes_definitions(self):
        d = compute_recovery_metrics(_scenario_episodes()).to_dict()
        assert "definitions" in d and "denominators" in d and "numerators" in d
        for name in DEFINITIONS:
            assert name in d["definitions"]
        # 每个数值指标都有显式分母定义
        for name in ("task_success_rate", "verified_fault_injection_rate",
                     "failure_detection_rate", "recovery_success_rate",
                     "avg_retry_count", "recovery_latency_mean_steps"):
            assert name in d["denominators"]


# ---- oracle / 观测型基线区分 -----------------------------------------------

class TestPrivilegedSplit:
    def test_split_by_privileged(self):
        oracle = [make_mock_fault_episode("t", seed=i, fault_type="f", privileged_access=True)
                  for i in range(3)]
        obs = [make_mock_fault_episode("t", seed=i, fault_type="f", privileged_access=False)
               for i in range(2)]
        groups = split_by_privileged(oracle + obs)
        assert len(groups[True]) == 3 and len(groups[False]) == 2

    def test_mixed_privileged_raises_in_compute(self):
        mixed = [
            make_mock_fault_episode("t", seed=0, fault_type="f", privileged_access=True),
            make_mock_fault_episode("t", seed=1, fault_type="f", privileged_access=False),
        ]
        with pytest.raises(ValueError, match="privileged_access"):
            compute_recovery_metrics(mixed)

    def test_explicit_privileged_overrides(self):
        eps = [make_mock_fault_episode("t", seed=0, fault_type="f", privileged_access=True)]
        m = compute_recovery_metrics(eps, privileged_access=False)
        assert m.privileged_access is False
