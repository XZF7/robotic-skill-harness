"""runtime/test_harness.py — TaskRunner 集成测试。

覆盖:
- 真实 PickCube 成功路径 (经 executor 而非直接调 baseline 脚本)。
- 超时 (max_steps 很小 -> TIMEOUT)。
- 取消 (abort_check -> ABORTED)。
- 异常 (skill.act 抛异常 -> FAILURE/MOTION_FAIL)。
- 失败 (技能自报 FAILURE)。
- 单一权威 SkillResult (executor 返回的那个, skill.result() 不被二次调用改变结果)。
- 结构化日志完整 trace (skill_id/status/steps/failure_type)。
- EpisodeResult(source="sim") 供 Agent D 消费。
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
    ExecutionLogger,
    SkillExecutor,
    TaskRunner,
    TaskState,
)
from sim import PickCubeAdapter, PickCubeSkill


# --------------------------------------------------------------------------
# mock 构造器 (与 runtime/test_runtime.py 同风格)
# --------------------------------------------------------------------------
def make_obs(step=0, timestamp=0.0, success=False):
    policy = PolicyObservation(
        qpos=np.zeros(7), qvel=np.zeros(7), tcp_pose=np.zeros(7), gripper=0.0,
    )
    truth = GroundTruth(object_poses={}, success=success, is_grasped=False)
    return RobotObservation(policy=policy, truth=truth, step=step, timestamp=timestamp)


class CounterSkill(RobotSkill):
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


class CountingSkill(RobotSkill):
    """委托到内部技能但统计 result() 调用次数 (验证不被二次调用)。"""

    def __init__(self, inner: RobotSkill):
        self.inner = inner
        self.result_calls = 0

    @property
    def skill_id(self):
        return self.inner.skill_id

    def reset(self, obs):
        return self.inner.reset(obs)

    def act(self, obs):
        return self.inner.act(obs)

    @property
    def is_done(self):
        return self.inner.is_done

    def result(self, final_obs=None):
        self.result_calls += 1
        return self.inner.result(final_obs)


class MockAdapter(SimulatorAdapter):
    def __init__(self):
        self.n = 0

    def reset(self, *, seed=None, options=None):
        self.n = 0
        return make_obs(step=0, timestamp=0.0)

    def step(self, action):
        self.n += 1
        obs = make_obs(step=self.n, timestamp=0.05 * self.n)
        info = EpisodeInfo(terminated=False, truncated=False, success=False)
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


def _run_mock(skill, *, max_steps=None, abort_check=None):
    """用 mock adapter 组一个 runner 跑一局，返回 (TaskRunResult, 日志记录)。"""
    adapter = MockAdapter()
    buf = io.StringIO()
    runner = TaskRunner(
        adapter, skill, task_id="mock-task", logger=ExecutionLogger(stream=buf),
    )
    run = runner.run(seed=0, max_steps=max_steps, abort_check=abort_check)
    rec = json.loads(buf.getvalue().strip().splitlines()[-1])
    return run, rec


# --------------------------------------------------------------------------
# 真实 PickCube 成功路径 (经 executor)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pickcube_runner():
    adapter = PickCubeAdapter(max_episode_steps=200)
    skill = PickCubeSkill(adapter)
    buf = io.StringIO()
    runner = TaskRunner(
        adapter, skill, task_id="PickCube-v1", logger=ExecutionLogger(stream=buf),
    )
    yield runner, buf
    adapter.close()


class TestRealPickCubeSuccess:
    def test_success_via_executor(self, pickcube_runner):
        runner, buf = pickcube_runner
        run = runner.run(seed=0, max_steps=200)

        # (a) 唯一权威 SkillResult 来自 executor (不二次判定)
        assert run.skill_result.status is SkillStatus.SUCCESS
        assert run.skill_result.skill_id == "pick_cube_mplib"
        assert run.skill_result.steps > 0

        # (b) 最终 EpisodeInfo (env 特权真值)
        assert run.episode is not None
        assert run.episode.success is True
        assert run.episode.terminated is True
        assert run.episode.truncated is False

        # (c) TaskState
        assert run.task_state is TaskState.SUCCESS

        # EpisodeResult(source="sim") 供 Agent D 消费
        er = run.to_episode_result("PickCube-v1")
        assert er.source == "sim"
        assert er.success is True          # env 真值
        assert len(er.skill_results) == 1  # 只有一个权威 SkillResult
        assert er.skill_results[0] is run.skill_result
        assert er.metrics["skill_status"] == "success"
        assert er.metrics["terminated"] is True

        # 结构化日志完整 trace
        lines = buf.getvalue().strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["skill_id"] == "pick_cube_mplib"
        assert rec["status"] == "success"
        assert rec["steps"] > 0
        assert rec["failure_type"] is None
        assert rec["task_id"] == "PickCube-v1"
        assert rec["task_state"] == "success"
        assert rec["env_success"] is True
        assert rec["terminated"] is True


# --------------------------------------------------------------------------
# 超时 / 取消 / 异常 / 失败
# --------------------------------------------------------------------------
class TestTerminalPaths:
    def test_timeout(self):
        run, rec = _run_mock(NeverDoneSkill(), max_steps=3)
        assert run.skill_result.status is SkillStatus.TIMEOUT
        assert run.skill_result.steps == 3
        assert run.skill_result.failure is not None
        assert run.skill_result.failure.type is FailureType.TIMEOUT
        assert run.task_state is TaskState.TIMEOUT
        assert rec["status"] == "timeout"
        assert rec["failure_type"] == "timeout"

    def test_abort(self):
        run, rec = _run_mock(NeverDoneSkill(), max_steps=100, abort_check=lambda: True)
        assert run.skill_result.status is SkillStatus.ABORTED
        assert run.skill_result.steps == 0
        assert run.skill_result.failure is None
        assert run.task_state is TaskState.ABORTED
        assert rec["status"] == "aborted"

    def test_act_exception_becomes_motion_fail(self):
        run, rec = _run_mock(RaisingSkill(), max_steps=10)
        assert run.skill_result.status is SkillStatus.FAILURE
        assert run.skill_result.failure is not None
        assert run.skill_result.failure.type is FailureType.MOTION_FAIL
        assert run.skill_result.steps == 0
        assert run.task_state is TaskState.FAILURE
        assert rec["status"] == "failure"
        assert rec["failure_type"] == "motion_fail"

    def test_failure_path(self):
        run, rec = _run_mock(
            CounterSkill(done_after=2, status=SkillStatus.FAILURE,
                         failure_type=FailureType.GRIP_FAIL),
            max_steps=10,
        )
        assert run.skill_result.status is SkillStatus.FAILURE
        assert run.skill_result.failure is not None
        assert run.skill_result.failure.type is FailureType.GRIP_FAIL
        assert run.skill_result.steps == 2
        assert run.task_state is TaskState.FAILURE
        assert rec["status"] == "failure"
        assert rec["failure_type"] == "grip_fail"


# --------------------------------------------------------------------------
# 单一权威结果 + 日志 trace
# --------------------------------------------------------------------------
class TestSingleAuthoritativeResult:
    def test_result_not_recomputed_by_harness(self):
        inner = CounterSkill(done_after=2, status=SkillStatus.SUCCESS)
        skill = CountingSkill(inner)
        adapter = MockAdapter()
        buf = io.StringIO()
        runner = TaskRunner(
            adapter, skill, task_id="mock-task", logger=ExecutionLogger(stream=buf),
        )
        run = runner.run(seed=0, max_steps=10)

        # executor 只调用一次 result()；harness 未二次调用
        assert skill.result_calls == 1

        # 权威结果 = executor 产出的那个，且与技能自报一致
        assert run.skill_result.status is SkillStatus.SUCCESS
        assert run.skill_result.skill_id == "counter"
        assert run.skill_result.steps == 2

        # 手动再调 result() 不改变已返回的权威结果 (幂等)
        again = skill.result()
        assert again.status is run.skill_result.status
        assert run.skill_result.status is SkillStatus.SUCCESS
        assert run.skill_result.steps == 2

        # 只有一条 trace
        assert len(buf.getvalue().strip().splitlines()) == 1

    def test_log_trace_fields_complete(self):
        run, rec = _run_mock(
            CounterSkill(done_after=2, status=SkillStatus.SUCCESS), max_steps=10,
        )
        for field in ("skill_id", "status", "steps", "failure_type", "message"):
            assert field in rec
        assert rec["skill_id"] == "counter"
        assert rec["status"] == "success"
        assert rec["steps"] == 2
        assert rec["failure_type"] is None
        assert rec["task_id"] == "mock-task"
        assert rec["task_state"] == "success"
