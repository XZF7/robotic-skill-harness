"""契约单元测试: 验证接口定义一致性、可构造性、语义区分。"""
import numpy as np
import pytest

from interfaces import (
    PolicyObservation, GroundTruth, RobotObservation,
    SkillStatus, SkillResult, RobotSkill,
    EpisodeInfo, StepResult, SimulatorAdapter,
    FailureType, FailureEvent,
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
        assert {f.value for f in FailureType} == {"drop", "collision", "grip_fail", "motion_fail", "timeout", "unknown"}


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


class TestMetrics:
    def test_metrics(self):
        m = EpisodeMetrics(num_episodes=10, success_rate=0.8, mean_steps=42.0)
        assert 0.0 <= m.success_rate <= 1.0
