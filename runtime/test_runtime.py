"""runtime 单元测试：成功 / 非法转移 / 失败 / 超时，全部基于 mock 仿真接口。

不依赖 sim/ 与 Agent A；自行构造 mock 的 RobotObservation / SkillResult / FailureEvent。
关键验证: executor 的成败判定来自 skill.result()，而非 truth.success。
"""
from __future__ import annotations

import io
import json

import numpy as np
import pytest

from interfaces import (
    EpisodeInfo,
    FailureEvent,
    FailureType,
    GroundTruth,
    PolicyObservation,
    RobotObservation,
    RobotSkill,
    SkillResult,
    SkillStatus,
    SimulatorAdapter,
    StepResult,
)
from runtime import (
    DuplicateSkillError,
    ExecutionLogger,
    InvalidTransitionError,
    SkillExecutor,
    SkillNotFoundError,
    SkillRegistry,
    TaskState,
    TaskStateMachine,
    skill_status_to_task_state,
)


# --------------------------------------------------------------------------
# mock 构造器
# --------------------------------------------------------------------------
def make_obs(step=0, timestamp=0.0, success=False, failure_type=None):
    policy = PolicyObservation(
        qpos=np.zeros(7), qvel=np.zeros(7), tcp_pose=np.zeros(7), gripper=0.0,
    )
    contact = {}
    if failure_type is not None:
        contact["failure_type"] = failure_type
    truth = GroundTruth(
        object_poses={}, success=success, is_grasped=False, contact=contact,
    )
    return RobotObservation(policy=policy, truth=truth, step=step, timestamp=timestamp)


class CounterSkill(RobotSkill):
    """act 计数到 done_after 次后 is_done=True；result() 返回配置好的状态。"""

    skill_id = "counter"

    def __init__(self, done_after=3, status=SkillStatus.SUCCESS, failure_type=None):
        self.done_after = done_after
        self.status = status
        self.failure_type = failure_type
        self.count = 0

    def reset(self, obs):
        self.count = 0

    def act(self, obs):
        self.count += 1
        return np.zeros(8)

    @property
    def is_done(self):
        return self.count >= self.done_after

    def result(self, final_obs=None):
        failure = (
            FailureEvent(self.failure_type, self.count, 0.0)
            if self.failure_type is not None else None
        )
        return SkillResult(self.status, self.skill_id, self.count, failure=failure)


class NeverDoneSkill(RobotSkill):
    skill_id = "never"

    def reset(self, obs):
        pass

    def act(self, obs):
        return np.zeros(8)

    @property
    def is_done(self):
        return False

    def result(self, final_obs=None):
        return SkillResult(SkillStatus.FAILURE, self.skill_id, 0)


class RaisingSkill(RobotSkill):
    skill_id = "raising"

    def reset(self, obs):
        pass

    def act(self, obs):
        raise RuntimeError("motion planning failed")

    @property
    def is_done(self):
        return False

    def result(self, final_obs=None):
        return SkillResult(SkillStatus.FAILURE, self.skill_id, 0)


class MockAdapter(SimulatorAdapter):
    """脚本化 step：产出递增 step/timestamp，按脚本置 success / failure_type。"""

    def __init__(self, success_from_step=None, failure_type=None):
        self.n = 0
        self.success_from_step = success_from_step
        self.failure_type = failure_type
        self.actions = []

    def reset(self, *, seed=None, options=None):
        self.n = 0
        return make_obs(step=0, timestamp=0.0)

    def step(self, action):
        self.actions.append(action)
        self.n += 1
        success = False
        if self.success_from_step is not None and self.n >= self.success_from_step:
            success = True
        obs = make_obs(
            step=self.n, timestamp=0.05 * self.n,
            success=success, failure_type=self.failure_type,
        )
        info = EpisodeInfo(terminated=success, truncated=False, success=success)
        return StepResult(observation=obs, reward=0.0, episode=info)

    def close(self):
        pass

    @property
    def num_envs(self):
        return 1

    @property
    def action_space(self):
        return None

    @property
    def max_episode_steps(self):
        return 200


# --------------------------------------------------------------------------
# SkillRegistry
# --------------------------------------------------------------------------
class TestRegistry:
    def test_register_lookup_list(self):
        reg = SkillRegistry()
        s = CounterSkill()
        assert reg.register(s) == "counter"
        assert reg.lookup("counter") is s
        assert reg.list_skills() == ["counter"]
        assert "counter" in reg and len(reg) == 1

    def test_duplicate_raises(self):
        reg = SkillRegistry()
        reg.register(CounterSkill())
        with pytest.raises(DuplicateSkillError):
            reg.register(CounterSkill())

    def test_lookup_missing_raises(self):
        reg = SkillRegistry()
        with pytest.raises(SkillNotFoundError):
            reg.lookup("nope")

    def test_unregister(self):
        reg = SkillRegistry()
        reg.register(CounterSkill())
        assert reg.unregister("counter") is not None
        assert "counter" not in reg


# --------------------------------------------------------------------------
# SkillExecutor
# --------------------------------------------------------------------------
class TestExecutor:
    def test_success(self):
        ex = SkillExecutor()
        skill = CounterSkill(done_after=3, status=SkillStatus.SUCCESS)
        adapter = MockAdapter(success_from_step=3)
        result = ex.execute(skill, adapter.reset(), max_steps=10, adapter=adapter)
        assert result.status is SkillStatus.SUCCESS
        assert result.skill_id == "counter"
        assert result.steps == 3
        assert result.failure is None

    def test_success_not_from_truth(self):
        """truth 全程 False，但技能自报 SUCCESS -> 结果仍 SUCCESS (executor 不看 truth)。"""
        ex = SkillExecutor()
        skill = CounterSkill(done_after=3, status=SkillStatus.SUCCESS)
        adapter = MockAdapter(success_from_step=None)  # truth 恒 False
        result = ex.execute(skill, adapter.reset(), max_steps=10, adapter=adapter)
        assert result.status is SkillStatus.SUCCESS
        assert result.steps == 3

    def test_failure(self):
        ex = SkillExecutor()
        skill = CounterSkill(done_after=3, status=SkillStatus.FAILURE,
                             failure_type=FailureType.GRIP_FAIL)
        adapter = MockAdapter(success_from_step=None)
        result = ex.execute(skill, adapter.reset(), max_steps=10, adapter=adapter)
        assert result.status is SkillStatus.FAILURE
        assert result.steps == 3
        assert result.failure is not None
        assert result.failure.type is FailureType.GRIP_FAIL

    def test_failure_not_from_truth(self):
        """truth 从第 1 步起 True，但技能自报 FAILURE -> 结果仍 FAILURE。"""
        ex = SkillExecutor()
        skill = CounterSkill(done_after=3, status=SkillStatus.FAILURE,
                             failure_type=FailureType.GRIP_FAIL)
        adapter = MockAdapter(success_from_step=1)  # truth 从第 1 步 True，env 提前终止
        result = ex.execute(skill, adapter.reset(), max_steps=10, adapter=adapter)
        assert result.status is SkillStatus.FAILURE
        assert result.failure.type is FailureType.GRIP_FAIL

    def test_timeout(self):
        ex = SkillExecutor()
        result = ex.execute(NeverDoneSkill(), make_obs(), max_steps=5)
        assert result.status is SkillStatus.TIMEOUT
        assert result.steps == 5
        assert result.failure is not None
        assert result.failure.type is FailureType.TIMEOUT

    def test_abort(self):
        ex = SkillExecutor()
        result = ex.execute(NeverDoneSkill(), make_obs(), max_steps=100, abort_check=lambda: True)
        assert result.status is SkillStatus.ABORTED
        assert result.steps == 0

    def test_act_exception_becomes_motion_fail(self):
        ex = SkillExecutor()
        result = ex.execute(RaisingSkill(), make_obs(), max_steps=10)
        assert result.status is SkillStatus.FAILURE
        assert result.failure.type is FailureType.MOTION_FAIL
        assert result.steps == 0

    def test_negative_budget_raises(self):
        ex = SkillExecutor()
        with pytest.raises(ValueError):
            ex.execute(NeverDoneSkill(), make_obs(), max_steps=-1)


# --------------------------------------------------------------------------
# TaskStateMachine
# --------------------------------------------------------------------------
class TestStateMachine:
    def test_success_path(self):
        sm = TaskStateMachine()
        assert sm.state is TaskState.IDLE
        sm.start()
        assert sm.state is TaskState.RUNNING
        sm.apply_skill_result(SkillResult(SkillStatus.SUCCESS, "x", 1))
        assert sm.state is TaskState.SUCCESS
        assert sm.is_terminal()

    def test_illegal_idle_to_success(self):
        sm = TaskStateMachine()
        with pytest.raises(InvalidTransitionError):
            sm.transition_to(TaskState.SUCCESS)

    def test_illegal_running_to_idle(self):
        sm = TaskStateMachine()
        sm.start()
        with pytest.raises(InvalidTransitionError):
            sm.transition_to(TaskState.IDLE)

    def test_illegal_terminal_to_running(self):
        sm = TaskStateMachine()
        sm.start()
        sm.transition_to(TaskState.FAILURE)
        with pytest.raises(InvalidTransitionError):
            sm.transition_to(TaskState.RUNNING)

    def test_status_to_state_mapping(self):
        assert skill_status_to_task_state(SkillStatus.SUCCESS) is TaskState.SUCCESS
        assert skill_status_to_task_state(SkillStatus.FAILURE) is TaskState.FAILURE
        assert skill_status_to_task_state(SkillStatus.TIMEOUT) is TaskState.TIMEOUT
        assert skill_status_to_task_state(SkillStatus.ABORTED) is TaskState.ABORTED

    def test_timeout_and_failure_paths(self):
        for status, target in (
            (SkillStatus.TIMEOUT, TaskState.TIMEOUT),
            (SkillStatus.FAILURE, TaskState.FAILURE),
        ):
            sm = TaskStateMachine()
            sm.start()
            sm.apply_skill_result(SkillResult(status, "x", 1))
            assert sm.state is target and sm.is_terminal()

    def test_reset(self):
        sm = TaskStateMachine()
        sm.start()
        sm.transition_to(TaskState.SUCCESS)
        sm.reset()
        assert sm.state is TaskState.IDLE and sm.history == []


# --------------------------------------------------------------------------
# ExecutionLogger
# --------------------------------------------------------------------------
class TestLogger:
    def test_json_lines_success(self):
        buf = io.StringIO()
        logger = ExecutionLogger(stream=buf)
        logger.log_result(SkillResult(SkillStatus.SUCCESS, "pick", 3, message="ok"))
        rec = json.loads(buf.getvalue().strip())
        assert rec["skill_id"] == "pick"
        assert rec["status"] == "success"
        assert rec["steps"] == 3
        assert rec["failure_type"] is None
        assert isinstance(rec["timestamp"], float)

    def test_json_lines_failure_type(self):
        buf = io.StringIO()
        logger = ExecutionLogger(stream=buf)
        result = SkillResult(
            SkillStatus.FAILURE, "pick", 2,
            failure=FailureEvent(FailureType.DROP, 2, 0.1),
        )
        logger.log_result(result)
        rec = json.loads(buf.getvalue().strip())
        assert rec["status"] == "failure"
        assert rec["failure_type"] == "drop"


# --------------------------------------------------------------------------
# 端到端：Registry + Executor + StateMachine + Logger
# --------------------------------------------------------------------------
class TestEndToEnd:
    def test_full_pipeline_success(self):
        reg = SkillRegistry()
        reg.register(CounterSkill(done_after=2, status=SkillStatus.SUCCESS))
        ex = SkillExecutor()
        adapter = MockAdapter(success_from_step=2)

        sm = TaskStateMachine()
        buf = io.StringIO()
        logger = ExecutionLogger(stream=buf)

        skill = reg.lookup("counter")
        sm.start()
        result = ex.execute(skill, adapter.reset(), max_steps=10, adapter=adapter)
        sm.apply_skill_result(result)
        logger.log_result(result)

        assert sm.state is TaskState.SUCCESS
        assert result.status is SkillStatus.SUCCESS
        rec = json.loads(buf.getvalue().strip())
        assert rec["status"] == "success" and rec["steps"] == 2
