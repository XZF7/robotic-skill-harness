"""runtime/test_recovery.py — 失败检测 + 恢复 (M3-B) 单元/集成测试。

用 mock 故障 (FaultSchedule + FaultyAdapter + FaultySkill) 模拟:
掉落/抓取失败/超时/无效观测/碰撞/终止, 全部只用 obs.policy 可观测信号。

覆盖:
1. FailureDetector: INVALID_OBS / COLLISION / GRIP_FAIL / DROP 检测, 绝不读 truth。
2. RecoveryManager: 有界重试 RETRY / REOBSERVE / ABORT, 耗尽 -> ABORT, env 终止 -> ABORT。
3. RecoveryStateMachine: 状态转移 + 决策序列。
4. RecoveryRunner 集成:
   - 检测 -> 重试 -> 成功 (recovered=True)。
   - 检测 -> 重试耗尽 -> ABORT (recovery_exhausted)。
   - terminated 后不重试 (单次尝试即 ABORT)。
   - invalid_obs -> REOBSERVE -> 成功。
   - 纯 policy 检测 (技能不自报具体类型) 端到端。
   - 每次尝试恰好一个权威 SkillResult。
   - 结构化 trace 完整 (skill_attempt / recovery_decision / recovery_result)。
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
    FailureDetector,
    FailureDetectorConfig,
    InvalidTransitionError,
    RecoveryManager,
    RecoveryManagerConfig,
    RecoveryRunner,
    RecoveryState,
    RecoveryStateMachine,
)

# --------------------------------------------------------------------------
# mock 构造器
# --------------------------------------------------------------------------
def make_obs(step=0, timestamp=0.0, gripper=0.5, qpos=None, qvel=None, success=False):
    policy = PolicyObservation(
        qpos=np.zeros(7) if qpos is None else np.asarray(qpos, dtype=float),
        qvel=np.zeros(7) if qvel is None else np.asarray(qvel, dtype=float),
        tcp_pose=np.zeros(7),
        gripper=float(gripper),
    )
    truth = GroundTruth(object_poses={}, success=success, is_grasped=False)
    return RobotObservation(policy=policy, truth=truth, step=step, timestamp=timestamp)


FAULT_TO_TYPE = {
    "grip_fail": FailureType.GRIP_FAIL,
    "drop": FailureType.DROP,
    "collision": FailureType.COLLISION,
    "invalid_obs": FailureType.INVALID_OBS,
    "terminate_drop": FailureType.DROP,
    "motion_fail": FailureType.MOTION_FAIL,
}


class FaultSchedule:
    """按尝试次数注入故障。faults[i] = 故障名 或 None (该次成功)。"""

    def __init__(self, faults, steps_per_attempt=4):
        self.faults = list(faults)
        self.steps_per_attempt = steps_per_attempt
        self.attempt = -1
        self.step_in_attempt = 0

    def begin_attempt(self):
        self.attempt += 1
        self.step_in_attempt = 0

    def current(self):
        if 0 <= self.attempt < len(self.faults):
            return self.faults[self.attempt]
        return None  # 超出 schedule -> 成功

    def advance_step(self):
        self.step_in_attempt += 1
        return self.step_in_attempt


class FaultySkill(RobotSkill):
    """mock 抓取技能: 步数驱动, 结果由 schedule 决定 (可关闭自报具体类型)。"""

    skill_id = "faulty_pick"
    privileged_access = False  # 观测型策略: 不读 truth

    def __init__(self, schedule, self_report=True):
        self.schedule = schedule
        self.self_report = self_report
        self._steps = 0

    def reset(self, obs):
        self.schedule.begin_attempt()
        self._steps = 0

    def act(self, obs):
        # 只用 policy 观测 (演示不读 truth): 这里仅计数推进。
        self.schedule.advance_step()
        self._steps = self.schedule.step_in_attempt
        fault = self.schedule.current()
        if fault == "timeout":
            return np.zeros(8)  # 永不 is_done -> executor TIMEOUT
        return np.zeros(8)

    @property
    def is_done(self):
        fault = self.schedule.current()
        if fault == "timeout":
            return False
        return self._steps >= self.schedule.steps_per_attempt

    def result(self, final_obs=None):
        fault = self.schedule.current()
        steps = self._steps
        if fault is None:
            return SkillResult(SkillStatus.SUCCESS, self.skill_id, steps)
        if fault == "timeout":
            return SkillResult(SkillStatus.FAILURE, self.skill_id, steps)
        if self.self_report:
            ftype = FAULT_TO_TYPE.get(fault, FailureType.UNKNOWN)
            return SkillResult(
                SkillStatus.FAILURE, self.skill_id, steps,
                failure=FailureEvent(ftype, steps, 0.0, detail={"source": "skill", "fault": fault}),
            )
        # 不自报具体类型 -> 迫使 detector 走 policy 启发式。
        return SkillResult(SkillStatus.FAILURE, self.skill_id, steps)


class FaultyAdapter(SimulatorAdapter):
    """脚本化 mock 环境: 按 schedule 注入可观测故障 (只改 obs.policy)。"""

    def __init__(self, schedule):
        self.schedule = schedule
        self.n = 0

    def reset(self, *, seed=None, options=None):
        self.n = 0
        return make_obs(step=0, timestamp=0.0, gripper=1.0)

    def step(self, action):
        self.n += 1
        k = self.schedule.step_in_attempt  # skill.act 已 advance
        fault = self.schedule.current()
        final = k >= self.schedule.steps_per_attempt
        terminate = fault == "terminate_drop" and final

        if fault == "drop":
            g = 0.0 if final else 0.5
        elif fault in ("grip_fail", "terminate_drop"):
            g = 0.0 if final else 1.0
        else:
            g = 0.5 if final else 1.0

        qpos = np.zeros(7)
        qvel = np.zeros(7)
        if fault == "invalid_obs" and final:
            qpos = np.full(7, np.nan)
        if fault == "collision" and final:
            qvel[0] = 100.0

        obs = make_obs(self.n, 0.05 * self.n, gripper=g, qpos=qpos, qvel=qvel)
        info = EpisodeInfo(terminated=terminate, truncated=False, success=False)
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


def _run(schedule, *, max_retries=2, max_steps=10, self_report=True):
    """组一个 RecoveryRunner 跑一局, 返回 (RecoveryRunResult, 日志记录列表)。"""
    adapter = FaultyAdapter(schedule)
    skill = FaultySkill(schedule, self_report=self_report)
    buf = io.StringIO()
    runner = RecoveryRunner(
        adapter, skill, task_id="mock-recovery",
        manager=RecoveryManager(max_retries=max_retries),
        logger=ExecutionLogger(stream=buf),
    )
    run = runner.run(seed=0, max_steps=max_steps)
    records = [json.loads(l) for l in buf.getvalue().strip().splitlines() if l.strip()]
    return run, records


def _events(records):
    return [r["event"] for r in records]


# --------------------------------------------------------------------------
# FailureDetector 单元测试 (只用 policy, 不读 truth)
# --------------------------------------------------------------------------
class TestFailureDetector:
    def test_invalid_obs_nan_qpos(self):
        det = FailureDetector()
        obs = make_obs(qpos=np.full(7, np.nan))
        ev = det.validate_observation(obs)
        assert ev is not None and ev.type is FailureType.INVALID_OBS

    def test_invalid_obs_gripper_out_of_range(self):
        det = FailureDetector()
        obs = make_obs(gripper=2.5)
        ev = det.validate_observation(obs)
        assert ev is not None and ev.type is FailureType.INVALID_OBS

    def test_valid_obs_passes(self):
        det = FailureDetector()
        assert det.validate_observation(make_obs()) is None

    def test_collision_qvel_spike(self):
        det = FailureDetector()
        obs = make_obs(qvel=np.array([100.0, 0, 0, 0, 0, 0, 0]))
        ev = det.detect_policy_anomaly([obs])
        assert ev is not None and ev.type is FailureType.COLLISION

    def test_grip_fail_closed_on_nothing(self):
        det = FailureDetector()
        obs = make_obs(gripper=0.0)
        ev = det.detect_policy_anomaly([obs])
        assert ev is not None and ev.type is FailureType.GRIP_FAIL

    def test_drop_requires_history(self):
        det = FailureDetector()
        history = [
            make_obs(step=1, gripper=0.5),  # 夹持
            make_obs(step=2, gripper=0.5),
            make_obs(step=3, gripper=0.0),  # 滑脱
        ]
        ev = det.detect_policy_anomaly(history)
        assert ev is not None and ev.type is FailureType.DROP

    def test_success_returns_none(self):
        det = FailureDetector()
        result = SkillResult(SkillStatus.SUCCESS, "x", 4)
        assert det.detect(result, make_obs()) is None

    def test_aborted_returns_none(self):
        det = FailureDetector()
        result = SkillResult(SkillStatus.ABORTED, "x", 0)
        assert det.detect(result, make_obs()) is None

    def test_does_not_read_truth(self):
        """truth.success=True 但 policy 显示抓取失败 -> 仍判失败 (证明不读 truth)。"""
        det = FailureDetector()
        obs = make_obs(gripper=0.0, success=True)  # truth 成功
        result = SkillResult(SkillStatus.FAILURE, "x", 3)  # 技能自报失败
        ev = det.detect(result, obs)
        assert ev is not None
        assert ev.type in (FailureType.GRIP_FAIL, FailureType.UNKNOWN)

    def test_skill_reported_failure_canonicalized(self):
        det = FailureDetector()
        result = SkillResult(
            SkillStatus.FAILURE, "x", 3,
            failure=FailureEvent(FailureType.DROP, 3, 0.1, detail={"a": 1}),
        )
        ev = det.detect(result, make_obs(gripper=0.5))
        assert ev is not None and ev.type is FailureType.DROP
        assert ev.detail["source"] == "skill"
        assert ev.detail["a"] == 1  # 原 detail 保留


# --------------------------------------------------------------------------
# RecoveryManager 单元测试
# --------------------------------------------------------------------------
class TestRecoveryManager:
    def test_retry_before_budget(self):
        mgr = RecoveryManager(max_retries=2)
        d = mgr.decide(FailureEvent(FailureType.DROP, 3, 0.1), retry_count=0)
        assert d.action.value == "retry"
        assert d.retry_count == 0

    def test_reobserve_on_invalid_obs(self):
        mgr = RecoveryManager(max_retries=2)
        d = mgr.decide(FailureEvent(FailureType.INVALID_OBS, 3, 0.1), retry_count=0)
        assert d.action.value == "reobserve"

    def test_abort_on_recovery_exhausted_type(self):
        mgr = RecoveryManager(max_retries=2)
        d = mgr.decide(FailureEvent(FailureType.RECOVERY_EXHAUSTED, 3, 0.1), retry_count=0)
        assert d.action.value == "abort"

    def test_abort_when_budget_exhausted(self):
        mgr = RecoveryManager(max_retries=2)
        d = mgr.decide(FailureEvent(FailureType.DROP, 3, 0.1), retry_count=2)
        assert d.action.value == "abort"
        assert "exhausted" in d.reason

    def test_abort_when_env_ended(self):
        mgr = RecoveryManager(max_retries=2)
        d = mgr.decide(FailureEvent(FailureType.DROP, 3, 0.1), retry_count=0, env_ended=True)
        assert d.action.value == "abort"
        assert "episode ended" in d.reason

    def test_negative_max_retries_raises(self):
        with pytest.raises(ValueError):
            RecoveryManager(max_retries=-1)


# --------------------------------------------------------------------------
# RecoveryStateMachine 单元测试
# --------------------------------------------------------------------------
class TestRecoveryStateMachine:
    def test_recover_then_succeed(self):
        sm = RecoveryStateMachine()
        sm.start()
        assert sm.state is RecoveryState.RUNNING
        sm.mark_failed()
        assert sm.state is RecoveryState.RECOVERING
        sm.mark_retrying()
        assert sm.state is RecoveryState.RUNNING
        sm.mark_succeeded()
        assert sm.state is RecoveryState.SUCCEEDED
        assert sm.is_terminal()

    def test_abort_from_recovering(self):
        sm = RecoveryStateMachine()
        sm.start()
        sm.mark_failed()
        sm.mark_aborted()
        assert sm.state is RecoveryState.ABORTED

    def test_decisions_recorded(self):
        from interfaces import RecoveryDecision, RecoveryAction
        sm = RecoveryStateMachine()
        sm.start()
        d = RecoveryDecision(RecoveryAction.RETRY, 0, "retry")
        sm.mark_failed(d)
        sm.mark_retrying()
        sm.mark_aborted(RecoveryDecision(RecoveryAction.ABORT, 1, "exhausted"))
        assert [x.action for x in sm.decisions] == ["retry", "abort"]

    def test_illegal_transition_raises(self):
        sm = RecoveryStateMachine()
        with pytest.raises(InvalidTransitionError):
            sm.transition_to(RecoveryState.SUCCEEDED)  # IDLE -> SUCCEEDED 非法


# --------------------------------------------------------------------------
# RecoveryRunner 集成测试
# --------------------------------------------------------------------------
class TestRecoveryIntegration:
    def test_detect_retry_success(self):
        """检测 -> 重试 -> 成功。"""
        run, records = _run(FaultSchedule(["grip_fail", None]))
        r = run.recovery
        assert r.attempts == 2
        assert r.recovered is True
        assert r.aborted is False
        assert r.final_skill_result.status is SkillStatus.SUCCESS
        assert [d.action.value for d in r.decisions] == ["retry"]
        assert run.task_state == "succeeded"

    def test_detect_retry_exhausted_abort(self):
        """检测 -> 重试耗尽 -> ABORT。"""
        run, records = _run(FaultSchedule(["grip_fail", "grip_fail", "grip_fail"]), max_retries=2)
        r = run.recovery
        assert r.attempts == 3  # 1 + max_retries
        assert r.recovered is False
        assert r.aborted is True
        assert r.final_skill_result.status is SkillStatus.FAILURE
        assert [d.action.value for d in r.decisions] == ["retry", "retry", "abort"]
        assert r.decisions[-1].reason == "recovery exhausted (retry budget reached)"
        assert run.task_state == "aborted"

    def test_terminated_no_retry(self):
        """terminated 后不重试 (同 episode 无法恢复)。"""
        run, records = _run(FaultSchedule(["terminate_drop"]))
        r = run.recovery
        assert r.attempts == 1
        assert r.aborted is True
        assert r.recovered is False
        assert [d.action.value for d in r.decisions] == ["abort"]
        assert "episode ended" in r.decisions[0].reason

    def test_invalid_obs_reobserve_success(self):
        """无效观测 -> REOBSERVE -> 成功。"""
        run, records = _run(FaultSchedule(["invalid_obs", None]))
        r = run.recovery
        assert r.attempts == 2
        assert r.recovered is True
        assert [d.action.value for d in r.decisions] == ["reobserve"]
        assert r.final_skill_result.status is SkillStatus.SUCCESS

    def test_policy_only_detection_end_to_end(self):
        """技能不自报具体类型 -> detector 纯 policy 检测 GRIP_FAIL -> 重试成功。"""
        run, records = _run(FaultSchedule(["grip_fail", None]), self_report=False)
        r = run.recovery
        assert r.attempts == 2
        assert r.recovered is True
        assert [d.action.value for d in r.decisions] == ["retry"]

    def test_policy_only_drop_detection(self):
        """纯 policy 历史检测 DROP -> 重试成功。"""
        run, records = _run(FaultSchedule(["drop", None]), self_report=False)
        r = run.recovery
        assert r.attempts == 2
        assert r.recovered is True

    def test_timeout_retry_then_success(self):
        """超时 -> RETRY -> 成功。"""
        run, records = _run(FaultSchedule(["timeout", None]), max_steps=5)
        r = run.recovery
        assert r.attempts == 2
        assert r.recovered is True
        assert run.attempt_results[0].status is SkillStatus.TIMEOUT
        assert run.attempt_results[1].status is SkillStatus.SUCCESS
        assert [d.action.value for d in r.decisions] == ["retry"]

    def test_each_attempt_one_skill_result(self):
        """每次尝试恰好一个权威 SkillResult。"""
        run, records = _run(FaultSchedule(["grip_fail", "grip_fail", "grip_fail"]), max_retries=2)
        r = run.recovery
        assert len(run.attempt_results) == r.attempts == 3
        assert all(isinstance(sr, SkillResult) for sr in run.attempt_results)
        # 最终结果 = 最后一次尝试的权威结果
        assert r.final_skill_result is run.attempt_results[-1]

    def test_trace_complete(self):
        """结构化 trace 完整: skill_attempt / recovery_decision / recovery_result。"""
        run, records = _run(FaultSchedule(["grip_fail", None]))
        ev = _events(records)
        assert ev.count("skill_attempt") == 2
        assert ev.count("recovery_decision") == 1
        assert ev.count("recovery_result") == 1
        # 每条 skill_attempt 含权威字段
        for rec in records:
            if rec["event"] == "skill_attempt":
                for f in ("skill_id", "status", "steps", "failure_type"):
                    assert f in rec, rec
        summary = next(r for r in records if r["event"] == "recovery_result")
        assert summary["attempts"] == 2
        assert summary["recovered"] is True
        assert summary["aborted"] is False
        assert summary["final_status"] == "success"

    def test_attempt_records_carry_failure_type(self):
        run, records = _run(FaultSchedule(["grip_fail", None]))
        first = next(r for r in records if r["event"] == "skill_attempt" and r["attempt"] == 1)
        assert first["failure_type"] == "grip_fail"
        assert first["status"] == "failure"
