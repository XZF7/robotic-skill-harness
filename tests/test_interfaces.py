"""契约单元测试: 验证接口定义一致性、可构造性、语义区分。"""
import numpy as np
import pytest

from interfaces import (
    PolicyObservation, GroundTruth, RobotObservation,
    SkillStatus, SkillResult, RobotSkill,
    EpisodeInfo, StepResult, SimulatorAdapter,
    FailureType, FailureEvent,
    RecoveryAction, RecoveryDecision, RecoveryResult,
    FaultRecord,
    EpisodeResult, EpisodeMetrics,
)


def _obs():
    policy = PolicyObservation(
        qpos=np.zeros(7), qvel=np.zeros(7),
        tcp_pose=np.zeros(7), gripper=0.0,
    )
    truth = GroundTruth(object_poses={}, success=False, is_grasped=False)
    return RobotObservation(policy=policy, truth=truth, step=0, timestamp=0.0)


class TestEnums:
    def test_skill_status_values(self):
        assert {s.value for s in SkillStatus} == {"success", "failure", "timeout", "aborted"}

    def test_failure_type_values(self):
        assert {f.value for f in FailureType} == {
            "drop", "collision", "grip_fail", "motion_fail", "timeout",
            "invalid_obs", "recovery_exhausted", "unknown",
        }


class TestObservation:
    def test_policy_truth_separation(self):
        policy = PolicyObservation(qpos=np.zeros(7), qvel=np.zeros(7), tcp_pose=np.zeros(7), gripper=0.0)
        truth = GroundTruth(object_poses={"cube": np.zeros(7)}, success=True, is_grasped=True)
        obs = RobotObservation(policy=policy, truth=truth, step=3, timestamp=0.15)
        assert obs.policy is policy and obs.truth is truth
        assert not hasattr(obs.policy, "success")

    def test_tcp_pose_dim(self):
        assert _obs().policy.tcp_pose.shape == (7,)


class TestABC:
    def test_robot_skill_requires_result(self):
        class IncompleteSkill(RobotSkill):
            skill_id = "incomplete"
            def reset(self, obs): pass
            def act(self, obs): return np.zeros(8)
            @property
            def is_done(self): return False
        with pytest.raises(TypeError):
            IncompleteSkill()

    def test_simulator_adapter_abstract(self):
        with pytest.raises(TypeError):
            SimulatorAdapter()

    def test_robot_skill_abstract(self):
        with pytest.raises(TypeError):
            RobotSkill()

    def test_privileged_access_default_false(self):
        class PolicySkill(RobotSkill):
            skill_id = "policy"
            def reset(self, obs): pass
            def act(self, obs): return np.zeros(8)
            @property
            def is_done(self): return False
            def result(self, final_obs=None):
                return SkillResult(SkillStatus.SUCCESS, "policy", 0)
        assert PolicySkill().privileged_access is False


class TestEpisodeInfo:
    def test_three_flags_independent(self):
        e = EpisodeInfo(terminated=False, truncated=True, success=False)
        assert not e.terminated and e.truncated and not e.success

    def test_defaults(self):
        e = EpisodeInfo(terminated=False, truncated=False, success=False)
        assert e.reward == 0.0 and e.info == {}


class TestSkillResult:
    def test_construct_with_failure(self):
        f = FailureEvent(type=FailureType.GRIP_FAIL, step=10, timestamp=1.0)
        r = SkillResult(status=SkillStatus.FAILURE, skill_id="pick", steps=10, failure=f)
        assert r.status is SkillStatus.FAILURE and r.failure.type is FailureType.GRIP_FAIL


class TestEpisodeResult:
    def test_source_distinguishes_mock_sim(self):
        mock = EpisodeResult(task_id="PickCube-v1", success=True, steps=10, reward=1.0, source="mock")
        sim = EpisodeResult(task_id="PickCube-v1", success=True, steps=10, reward=1.0, source="sim")
        assert mock.source == "mock" and sim.source == "sim"

    def test_default_source_is_mock(self):
        r = EpisodeResult(task_id="x", success=False, steps=0, reward=0.0)
        assert r.source == "mock"


class TestRecovery:
    def test_actions(self):
        assert {a.value for a in RecoveryAction} == {"retry", "reobserve", "abort"}

    def test_decision(self):
        d = RecoveryDecision(action=RecoveryAction.RETRY, retry_count=1, reason="grip_fail")
        assert d.action is RecoveryAction.RETRY and d.retry_count == 1

    def test_recovery_result_holds_final_skill_result(self):
        sr = SkillResult(SkillStatus.SUCCESS, "pick", 10)
        r = RecoveryResult(attempts=2, final_skill_result=sr, recovered=True)
        assert r.attempts == 2 and r.recovered and r.final_skill_result is sr


class TestFaultRecord:
    def test_record(self):
        f = FaultRecord(fault_type="object_displacement", injected=True,
                        step=5, params={"delta": 0.05})
        assert f.injected and f.fault_type == "object_displacement" and f.step == 5

    def test_not_injected(self):
        f = FaultRecord(fault_type="grasp_perturb", injected=False,
                        detail={"reason": "no collision"})
        assert not f.injected and f.detail["reason"]


class TestMetrics:
    def test_metrics(self):
        m = EpisodeMetrics(num_episodes=10, success_rate=0.8, mean_steps=42.0)
        assert 0.0 <= m.success_rate <= 1.0
