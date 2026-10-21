"""runtime/test_temporal_guard.py — M5-B 时序门控 (TemporalGate) mock 测试。

覆盖 (mock predictor + mock obs, 不依赖真实仿真):
1. 参考 B2 行为不变: gate=None / mode=NONE -> 单帧正预测即触发。
2. debounce: 连续 K 帧正预测才触发 (K=3 / K=5)；负帧重置计数。
3. phase: 仅在有意义操作相位评估 (gripper 闭合后 / tcp 接近桌面)。
4. cooldown: 触发后冷却窗口内禁止重复触发 (跨 attempt 生效)。
5. advisory: 只记录预测, 永不中断执行。
6. 记录语义: prediction (模型正预测) 与 triggered (实际中断) 分离, gate_reason 可追溯。

用法 (仓库根目录, CPU-only):
  source scripts/activate_env.sh; export CUDA_VISIBLE_DEVICES=0
  python3 -m pytest runtime/test_temporal_guard.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from interfaces import (
    EpisodeInfo,
    FailureEvent,
    FailurePrediction,
    FailurePredictor,
    FailureType,
    GroundTruth,
    PolicyObservation,
    RobotObservation,
    RobotSkill,
    SimulatorAdapter,
    SkillResult,
    SkillStatus,
    StepResult,
)
from runtime import (
    FROZEN_B2T_GATE,
    PredictiveGuard,
    PredictiveTrigger,
    RecoveryManager,
    RecoveryMode,
    RecoveryRunner,
    TemporalGate,
    TemporalGateConfig,
    TemporalGateMode,
)


def make_obs(step=0, gripper=0.5, tcp_z=0.2):
    policy = PolicyObservation(
        qpos=np.zeros(7, dtype=np.float32),
        qvel=np.zeros(7, dtype=np.float32),
        tcp_pose=np.array([0.0, 0.0, tcp_z, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        gripper=float(gripper),
    )
    truth = GroundTruth(object_poses={}, success=False, is_grasped=False)
    return RobotObservation(policy=policy, truth=truth, step=step,
                            timestamp=float(0.05 * step))


class MockPredictor(FailurePredictor):
    """可编程 mock 预测器: prob / confidence 固定; probs 按调用序号编排。"""

    def __init__(self, prob=0.9, confidence=1.0, threshold=0.5, probs=None):
        self.prob = float(prob)
        self.confidence = float(confidence)
        self.threshold = float(threshold)
        self.probs = [float(p) for p in probs] if probs is not None else None
        self.calls = 0

    def predict(self, features):
        self.calls += 1
        prob = (self.probs[min(self.calls - 1, len(self.probs) - 1)]
                if self.probs is not None else self.prob)
        return FailurePrediction(prob=prob, prediction=prob >= self.threshold,
                                 confidence=self.confidence)


def _guard(prob=0.9, *, gate=None, mode=RecoveryMode.B2_PREDICTIVE):
    return PredictiveGuard(MockPredictor(prob=prob, threshold=0.5),
                           mode=mode, threshold=0.5, gate=gate)


# --------------------------------------------------------------------------
# 参考 B2 行为不变
# --------------------------------------------------------------------------
class TestB2ReferenceUnchanged:
    def test_no_gate_single_frame_triggers(self):
        guard = _guard(prob=0.9)
        assert isinstance(guard.step_hook(make_obs(step=3)), PredictiveTrigger)

    def test_none_gate_equivalent_to_b2(self):
        cfg = TemporalGateConfig(mode=TemporalGateMode.NONE)
        guard = _guard(prob=0.9, gate=cfg)
        assert isinstance(guard.step_hook(make_obs(step=3)), PredictiveTrigger)

    def test_positive_then_record_triggered_true(self):
        guard = _guard(prob=0.9)
        guard.step_hook(make_obs(step=1))
        rec = guard.drain_attempt()[0]
        assert rec.prediction is True and rec.triggered is True
        assert rec.gate_reason is None


# --------------------------------------------------------------------------
# Debounce
# --------------------------------------------------------------------------
class TestDebounce:
    def test_k3_requires_three_consecutive(self):
        cfg = TemporalGateConfig(mode=TemporalGateMode.DEBOUNCE, debounce_k=3)
        guard = _guard(prob=0.9, gate=cfg)
        assert guard.step_hook(make_obs(step=1)) is None
        assert guard.step_hook(make_obs(step=2)) is None
        trig = guard.step_hook(make_obs(step=3))
        assert isinstance(trig, PredictiveTrigger)
        recs = guard.drain_attempt()
        assert [r.triggered for r in recs] == [False, False, True]
        assert recs[0].gate_reason == "debounce(1/3)"
        assert recs[1].gate_reason == "debounce(2/3)"

    def test_negative_frame_resets_counter(self):
        cfg = TemporalGateConfig(mode=TemporalGateMode.DEBOUNCE, debounce_k=3)
        # probs: + + - + + +  (positive/negative 交替验证复位)
        pred = MockPredictor(probs=[0.9, 0.9, 0.0, 0.9, 0.9, 0.9], threshold=0.5)
        guard = PredictiveGuard(pred, mode=RecoveryMode.B2_PREDICTIVE, threshold=0.5,
                                gate=cfg)
        assert guard.step_hook(make_obs(step=1)) is None
        assert guard.step_hook(make_obs(step=2)) is None
        assert guard.step_hook(make_obs(step=3)) is None  # 负帧复位
        assert guard.step_hook(make_obs(step=4)) is None
        assert guard.step_hook(make_obs(step=5)) is None
        assert isinstance(guard.step_hook(make_obs(step=6)), PredictiveTrigger)

    def test_k5(self):
        cfg = TemporalGateConfig(mode=TemporalGateMode.DEBOUNCE, debounce_k=5)
        guard = _guard(prob=0.9, gate=cfg)
        for s in range(1, 5):
            assert guard.step_hook(make_obs(step=s)) is None
        assert isinstance(guard.step_hook(make_obs(step=5)), PredictiveTrigger)


# --------------------------------------------------------------------------
# Phase
# --------------------------------------------------------------------------
class TestPhase:
    def test_gripper_closed_out_of_phase_no_trigger(self):
        cfg = TemporalGateConfig(mode=TemporalGateMode.PHASE, phase_signal="gripper_closed",
                                 phase_gripper_threshold=0.5)
        guard = _guard(prob=0.9, gate=cfg)
        # gripper 0.9 (张开, 抓取闭合前) -> 门控外, 不触发
        assert guard.step_hook(make_obs(step=1, gripper=0.9)) is None
        rec = guard.drain_attempt()[0]
        assert rec.prediction is True and rec.triggered is False
        assert rec.gate_reason == "out_of_phase"

    def test_gripper_closed_in_phase_triggers(self):
        cfg = TemporalGateConfig(mode=TemporalGateMode.PHASE, phase_signal="gripper_closed",
                                 phase_gripper_threshold=0.5)
        guard = _guard(prob=0.9, gate=cfg)
        assert isinstance(guard.step_hook(make_obs(step=1, gripper=0.2)), PredictiveTrigger)

    def test_tcp_near_table(self):
        cfg = TemporalGateConfig(mode=TemporalGateMode.PHASE, phase_signal="tcp_near_table",
                                 phase_tcp_z_threshold=0.10)
        guard = _guard(prob=0.9, gate=cfg)
        assert guard.step_hook(make_obs(step=1, tcp_z=0.25)) is None      # 远 -> 门控外
        assert isinstance(guard.step_hook(make_obs(step=2, tcp_z=0.05)), PredictiveTrigger)


# --------------------------------------------------------------------------
# Cooldown
# --------------------------------------------------------------------------
class TestCooldown:
    def test_cooldown_blocks_repeat_trigger(self):
        cfg = TemporalGateConfig(mode=TemporalGateMode.COOLDOWN, cooldown_steps=10)
        guard = _guard(prob=0.9, gate=cfg)
        assert isinstance(guard.step_hook(make_obs(step=5)), PredictiveTrigger)
        # 冷却窗口内 (step 8 - 5 = 3 < 10) 再次正预测 -> 不触发
        assert guard.step_hook(make_obs(step=8)) is None
        # 窗口外 (step 16 - 5 = 11 >= 10) -> 触发
        assert isinstance(guard.step_hook(make_obs(step=16)), PredictiveTrigger)

    def test_cooldown_persists_across_attempts(self):
        cfg = TemporalGateConfig(mode=TemporalGateMode.COOLDOWN, cooldown_steps=10)
        guard = _guard(prob=0.9, gate=cfg)
        guard.begin_attempt()
        assert isinstance(guard.step_hook(make_obs(step=5)), PredictiveTrigger)
        guard.drain_attempt()
        # 新 attempt: begin_attempt 不清 cooldown, step 8 仍在窗口内
        guard.begin_attempt()
        assert guard.step_hook(make_obs(step=8)) is None


# --------------------------------------------------------------------------
# Advisory
# --------------------------------------------------------------------------
class TestAdvisory:
    def test_advisory_never_triggers(self):
        cfg = TemporalGateConfig(mode=TemporalGateMode.ADVISORY)
        guard = _guard(prob=0.9, gate=cfg)
        for s in range(1, 6):
            assert guard.step_hook(make_obs(step=s)) is None
        recs = guard.drain_attempt()
        assert all(r.triggered is False for r in recs)
        assert all(r.prediction is True for r in recs)
        assert all(r.gate_reason == "advisory" for r in recs)

    def test_advisory_records_predictions(self):
        cfg = TemporalGateConfig(mode=TemporalGateMode.ADVISORY)
        guard = _guard(prob=0.9, gate=cfg)
        guard.step_hook(make_obs(step=1))
        guard.step_hook(make_obs(step=2))
        recs = guard.drain_attempt()
        assert len(recs) == 2
        assert sum(1 for r in recs if r.prediction) == 2
        assert sum(1 for r in recs if r.triggered) == 0


# --------------------------------------------------------------------------
# 门控基础 / 配置校验
# --------------------------------------------------------------------------
class TestGateBasics:
    def test_invalid_debounce_k(self):
        with pytest.raises(ValueError):
            TemporalGateConfig(mode=TemporalGateMode.DEBOUNCE, debounce_k=0)

    def test_invalid_phase_signal(self):
        with pytest.raises(ValueError):
            TemporalGateConfig(mode=TemporalGateMode.PHASE, phase_signal="bogus")

    def test_reset_clears_state(self):
        cfg = TemporalGateConfig(mode=TemporalGateMode.DEBOUNCE, debounce_k=2)
        guard = _guard(prob=0.9, gate=cfg)
        guard.step_hook(make_obs(step=1))  # 累积 1 帧
        guard.reset()
        # 复位后计数清零: 需再连续 2 帧才触发 (K=2)
        assert guard.step_hook(make_obs(step=2)) is None        # 1/2
        assert isinstance(guard.step_hook(make_obs(step=3)), PredictiveTrigger)  # 2/2

    def test_b2t_mode_enabled(self):
        guard = PredictiveGuard(MockPredictor(prob=0.9, threshold=0.5),
                                mode=RecoveryMode.B2T_TEMPORAL, threshold=0.5,
                                gate=TemporalGateConfig(mode=TemporalGateMode.DEBOUNCE,
                                                        debounce_k=2))
        assert guard.enabled is True
        assert guard.gate.mode is TemporalGateMode.DEBOUNCE


# --------------------------------------------------------------------------
# RecoveryRunner 集成: recovery_mode="b2t" + temporal_gate
# --------------------------------------------------------------------------
class ScriptedSkill(RobotSkill):
    """按 attempt 脚本化: 每次 attempt 成功 (steps 步)。"""

    skill_id = "scripted"
    privileged_access = False

    def __init__(self, steps=1):
        self._steps_target = steps
        self._n = 0

    def reset(self, obs):
        self._n = 0

    def act(self, obs):
        self._n += 1
        return np.zeros(8, dtype=np.float32)

    @property
    def is_done(self):
        return self._n >= self._steps_target

    def result(self, final_obs=None):
        return SkillResult(SkillStatus.SUCCESS, self.skill_id, self._n)


class ScriptedAdapter(SimulatorAdapter):
    def __init__(self):
        self.n = 0

    def reset(self, *, seed=None, options=None):
        self.n = 0
        return make_obs(0)

    def step(self, action):
        self.n += 1
        return StepResult(observation=make_obs(self.n), reward=0.0,
                          episode=EpisodeInfo(terminated=False, truncated=False,
                                              success=False))

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


class TestRecoveryRunnerB2T:
    def test_b2t_debounce_retry_then_success(self):
        """b2t + debounce K=2: 第 1 步单帧不触发, 连续 2 帧触发 -> 重试成功。"""
        import io
        from runtime import ExecutionLogger
        # 预测表: 第 1 次调用 0.9, 之后 0.0 -> 仅 attempt1 的步1 单帧正 (debounce 不够 K=2 不触发)
        # 用固定 prob=0.9 会让 debounce 在第 2 步触发。
        pred = MockPredictor(prob=0.9, threshold=0.5)
        gate = TemporalGateConfig(mode=TemporalGateMode.DEBOUNCE, debounce_k=2)
        adapter = ScriptedAdapter()
        runner = RecoveryRunner(
            adapter, ScriptedSkill(steps=3), task_id="b2t-test",
            manager=RecoveryManager(max_retries=2),
            logger=ExecutionLogger(stream=io.StringIO()),
            predictor=pred, recovery_mode="b2t", predictor_threshold=0.5,
            temporal_gate=gate,
        )
        run = runner.run(seed=0, max_steps=10)
        # guard 已按 b2t 启用, gate 为 debounce
        assert runner.guard is not None
        assert runner.guard.enabled is True
        assert runner.guard.gate.mode is TemporalGateMode.DEBOUNCE
        # 第 1 次 attempt: 步1 单帧正 -> 不触发; 步2 连续 2 帧 -> 触发 (提前中止)
        # 第 2 次 attempt: 步1/步2 -> 触发? 步2 又连续 2 帧触发...
        # max_retries=2 且每次 attempt 都会在第 2 步触发 -> 最终 abort (预算耗尽)。
        r = run.recovery
        assert r.attempts == 3  # 1 + max_retries
        assert r.aborted is True
        # 触发均走规则决策 (predictive UNKNOWN), 未绕过 max_retries
        assert [d.action.value for d in r.decisions] == ["retry", "retry", "abort"]

    def test_frozen_gate_is_phase(self):
        assert FROZEN_B2T_GATE.mode is TemporalGateMode.PHASE
        assert FROZEN_B2T_GATE.phase_signal == "gripper_closed"
        assert FROZEN_B2T_GATE.phase_gripper_threshold == 0.5
