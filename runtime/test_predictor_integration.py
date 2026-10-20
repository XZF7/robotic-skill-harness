"""runtime/test_predictor_integration.py — 学习失败预测器运行时集成 (M4-B) 测试。

覆盖 (mock predictor + mock adapter, 不依赖真实仿真):
1. PredictiveGuard: 推理异常 -> fallback；低置信度 -> 不触发；无效观测 -> 不触发；
   b1 关闭；b2 高风险高置信 -> 触发。
2. RecoveryRunner 集成 (b1/b2 切换):
   - b2 预判触发 -> 提前中止 attempt -> 规则 RETRY -> 重试成功 (recovered)。
   - 预判触发仍受 max_retries 约束 (安全不绕过)。
   - 推理异常 -> 回退规则 (不触发, 正常完成)。
   - 超时 (TIMEOUT) 路径与预判共存 (不干扰)。
   - 模型 prediction/confidence/是否触发 进结构化 trace。
3. 真实 checkpoint 预测器 (从 data/m4/checkpoints 重建) + 真实 PickCube 集成。

用法 (仓库根目录, CPU-only):
  source scripts/activate_env.sh; export CUDA_VISIBLE_DEVICES=2
  python3 -m pytest runtime/test_predictor_integration.py -v
"""
from __future__ import annotations

import io
import json
import os
import sys

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
    ExecutionLogger,
    PredictiveGuard,
    PredictiveTrigger,
    RecoveryManager,
    RecoveryMode,
    RecoveryRunner,
    load_predictor_from_checkpoint,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --------------------------------------------------------------------------
# mock 构造器
# --------------------------------------------------------------------------
def make_obs(step=0, timestamp=None, gripper=0.5, qpos=None, qvel=None):
    policy = PolicyObservation(
        qpos=np.zeros(7, dtype=np.float32) if qpos is None else np.asarray(qpos, dtype=np.float32),
        qvel=np.zeros(7, dtype=np.float32) if qvel is None else np.asarray(qvel, dtype=np.float32),
        tcp_pose=np.zeros(7, dtype=np.float32),
        gripper=float(gripper),
    )
    truth = GroundTruth(object_poses={}, success=False, is_grasped=False)
    return RobotObservation(
        policy=policy, truth=truth, step=step,
        timestamp=float(timestamp if timestamp is not None else 0.05 * step),
    )


class MockPredictor(FailurePredictor):
    """可编程 mock 预测器。

    prob / confidence 固定；raise_ 让 predict 抛异常 (测 fallback)；
    probs: 可选 1-indexed 每步概率表 (probs[i-1] = 第 i 次调用返回的概率, 越界取末项)；
           缺省则每次返回 prob。用 probs 可精确编排「某一步触发、其余步不触发」。
    """

    def __init__(self, prob=0.9, confidence=1.0, threshold=0.5, raise_=False,
                 probs=None):
        self.prob = float(prob)
        self.confidence = float(confidence)
        self.threshold = float(threshold)
        self.raise_ = bool(raise_)
        self.probs = [float(p) for p in probs] if probs is not None else None
        self.calls = 0

    def predict(self, features):
        self.calls += 1
        if self.raise_:
            raise RuntimeError("predictor boom")
        if self.probs is not None:
            prob = self.probs[min(self.calls - 1, len(self.probs) - 1)]
        else:
            prob = self.prob
        return FailurePrediction(
            prob=prob,
            prediction=prob >= self.threshold,
            confidence=self.confidence,
        )


class ScriptedSkill(RobotSkill):
    """按 attempt 脚本化技能: plans[i] = {"kind": success|timeout, "steps": int}。"""

    skill_id = "scripted_pick"
    privileged_access = False

    def __init__(self, plans):
        self.plans = list(plans)
        self.attempt = -1
        self._steps = 0

    def reset(self, obs):
        self.attempt += 1
        self._steps = 0

    def _plan(self):
        return self.plans[self.attempt % len(self.plans)]

    def act(self, obs):
        self._steps += 1
        return np.zeros(8, dtype=np.float32)

    @property
    def is_done(self):
        p = self._plan()
        if p["kind"] == "timeout":
            return False
        return self._steps >= p.get("steps", 1)

    def result(self, final_obs=None):
        p = self._plan()
        if p["kind"] == "success":
            return SkillResult(SkillStatus.SUCCESS, self.skill_id, self._steps)
        return SkillResult(SkillStatus.FAILURE, self.skill_id, self._steps,
                           failure=FailureEvent(FailureType.UNKNOWN, self._steps, 0.0,
                                                detail={"fault": p.get("kind")}))


class MockAdapter(SimulatorAdapter):
    """脚本化 mock 环境: 逐步推进 step, 永不终止。"""

    def __init__(self):
        self.n = 0

    def reset(self, *, seed=None, options=None):
        self.n = 0
        return make_obs(0)

    def step(self, action):
        self.n += 1
        obs = make_obs(self.n)
        return StepResult(
            observation=obs, reward=0.0,
            episode=EpisodeInfo(terminated=False, truncated=False, success=False),
        )

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


def _run(skill, *, mode="b2", predictor=None, max_retries=2, max_steps=10,
         min_confidence=0.0, threshold=None):
    adapter = MockAdapter()
    buf = io.StringIO()
    runner = RecoveryRunner(
        adapter, skill, task_id="pred-integration",
        manager=RecoveryManager(max_retries=max_retries),
        logger=ExecutionLogger(stream=buf),
        predictor=predictor,
        recovery_mode=mode,
        predictor_threshold=threshold,
        predictor_min_confidence=min_confidence,
    )
    run = runner.run(seed=0, max_steps=max_steps)
    records = [json.loads(l) for l in buf.getvalue().strip().splitlines() if l.strip()]
    return run, records


def _events(records):
    return [r["event"] for r in records]


# --------------------------------------------------------------------------
# PredictiveGuard 单元测试
# --------------------------------------------------------------------------
class TestPredictiveGuard:
    def test_b2_high_risk_high_confidence_triggers(self):
        guard = PredictiveGuard(MockPredictor(prob=0.9, confidence=0.9),
                                mode=RecoveryMode.B2_PREDICTIVE, threshold=0.5,
                                min_confidence=0.5)
        trig = guard.step_hook(make_obs(step=3))
        assert isinstance(trig, PredictiveTrigger)
        assert trig.prob == pytest.approx(0.9)
        rec = guard.drain_attempt()[0]
        assert rec.prediction is True and rec.triggered is True

    def test_low_confidence_does_not_trigger(self):
        guard = PredictiveGuard(MockPredictor(prob=0.9, confidence=0.1),
                                mode=RecoveryMode.B2_PREDICTIVE, threshold=0.5,
                                min_confidence=0.5)
        trig = guard.step_hook(make_obs())
        assert trig is None
        rec = guard.drain_attempt()[0]
        assert rec.prediction is True and rec.triggered is False
        assert rec.prob == pytest.approx(0.9) and rec.confidence == pytest.approx(0.1)

    def test_inference_exception_falls_back(self):
        guard = PredictiveGuard(MockPredictor(raise_=True),
                                mode=RecoveryMode.B2_PREDICTIVE)
        trig = guard.step_hook(make_obs())
        assert trig is None
        rec = guard.drain_attempt()[0]
        assert rec.error is not None and "predictor boom" in rec.error
        assert rec.triggered is False

    def test_invalid_observation_nan_falls_back(self):
        guard = PredictiveGuard(MockPredictor(prob=0.9),
                                mode=RecoveryMode.B2_PREDICTIVE)
        obs = make_obs(qpos=np.full(7, np.nan))
        trig = guard.step_hook(obs)
        assert trig is None
        rec = guard.drain_attempt()[0]
        assert rec.error is not None and rec.triggered is False

    def test_invalid_observation_none_fields_falls_back(self):
        guard = PredictiveGuard(MockPredictor(prob=0.9),
                                mode=RecoveryMode.B2_PREDICTIVE)
        obs = make_obs()
        obs.policy.qpos = None
        trig = guard.step_hook(obs)
        assert trig is None
        rec = guard.drain_attempt()[0]
        assert rec.error is not None

    def test_b1_mode_never_predicts(self):
        guard = PredictiveGuard(MockPredictor(prob=0.9),
                                mode=RecoveryMode.B1_RULE)
        assert guard.step_hook(make_obs()) is None
        assert guard.records == [] and guard.drain_attempt() == []

    def test_below_threshold_does_not_trigger(self):
        guard = PredictiveGuard(MockPredictor(prob=0.2, confidence=0.9),
                                mode=RecoveryMode.B2_PREDICTIVE, threshold=0.5)
        assert guard.step_hook(make_obs()) is None
        rec = guard.drain_attempt()[0]
        assert rec.prediction is False and rec.triggered is False


# --------------------------------------------------------------------------
# RecoveryRunner 集成 (b1/b2)
# --------------------------------------------------------------------------
class TestPredictiveIntegration:
    def test_predictive_trigger_retry_success(self):
        """b2: 预判触发 -> 提前中止 attempt -> 规则 RETRY -> 重试成功 (recovered)。"""
        # 第 1 步高风险触发, 之后低风险 -> 第 1 次尝试被预判中止, 第 2 次成功。
        pred = MockPredictor(probs=[0.9, 0.0, 0.0, 0.0])
        run, records = _run(ScriptedSkill([{"kind": "success", "steps": 2}] * 2),
                            predictor=pred, max_retries=2)
        r = run.recovery
        assert r.recovered is True
        assert r.attempts == 2
        assert r.final_skill_result.status is SkillStatus.SUCCESS
        assert [d.action.value for d in r.decisions] == ["retry"]
        # 第一次尝试被预判提前中止 (predictive UNKNOWN failure)
        assert run.attempt_results[0].failure is not None
        assert run.attempt_results[0].failure.detail.get("predictive") is True
        assert run.attempt_results[0].status is SkillStatus.FAILURE

    def test_predictive_trigger_respects_max_retries(self):
        """预判触发不绕过安全: max_retries 仍强制, 耗尽 -> ABORT。"""
        pred = MockPredictor(prob=0.9, confidence=1.0)  # 每次都触发
        run, records = _run(ScriptedSkill([{"kind": "success", "steps": 1}] * 5),
                            predictor=pred, max_retries=1)
        r = run.recovery
        assert r.attempts == 2  # 1 + max_retries
        assert r.aborted is True and r.recovered is False
        assert [d.action.value for d in r.decisions] == ["retry", "abort"]
        assert r.decisions[-1].reason == "recovery exhausted (retry budget reached)"

    def test_b1_vs_b2_switching(self):
        """b1: 纯规则 (预测器被忽略, 无预测记录)；b2: 预判触发并记录。"""
        pred_b1 = MockPredictor(probs=[0.9, 0.9, 0.9])
        run_b1, rec_b1 = _run(ScriptedSkill([{"kind": "success", "steps": 1}]),
                               mode="b1", predictor=pred_b1)
        assert "predictive_prediction" not in _events(rec_b1)
        assert run_b1.recovery.attempts == 1
        assert run_b1.recovery.recovered is False
        summary_b1 = next(r for r in rec_b1 if r["event"] == "recovery_result")
        assert summary_b1["predictive"]["mode"] == "b1"
        assert summary_b1["predictive"]["num_predictions"] == 0

        pred_b2 = MockPredictor(probs=[0.9, 0.0, 0.0, 0.0])
        run_b2, rec_b2 = _run(ScriptedSkill([{"kind": "success", "steps": 2}] * 2),
                              mode="b2", predictor=pred_b2)
        assert "predictive_prediction" in _events(rec_b2)
        assert run_b2.recovery.attempts == 2 and run_b2.recovery.recovered is True
        summary_b2 = next(r for r in rec_b2 if r["event"] == "recovery_result")
        assert summary_b2["predictive"]["mode"] == "b2"
        assert summary_b2["predictive"]["num_predictions"] == 3  # 1 (触发) + 2 (成功)
        assert summary_b2["predictive"]["num_triggers"] == 1

    def test_inference_exception_falls_back_to_rule(self):
        """推理异常 -> 回退规则: 不触发, 正常完成 (单次成功)。"""
        pred = MockPredictor(raise_=True)
        run, records = _run(ScriptedSkill([{"kind": "success", "steps": 2}]),
                            predictor=pred, max_retries=2)
        r = run.recovery
        assert r.attempts == 1 and r.recovered is False
        assert r.final_skill_result.status is SkillStatus.SUCCESS
        summary = next(x for x in records if x["event"] == "recovery_result")
        assert summary["predictive"]["num_triggers"] == 0
        assert summary["predictive"]["num_fallbacks"] == 2  # 每步一次异常
        assert all(x["error"] is not None
                   for x in records if x["event"] == "predictive_prediction")

    def test_timeout_path_with_predictor(self):
        """超时 (TIMEOUT) 与预判共存: 低风险预测不干扰 TIMEOUT -> RETRY -> ABORT。"""
        pred = MockPredictor(prob=0.0, confidence=1.0)
        run, records = _run(ScriptedSkill([{"kind": "timeout", "steps": 1}] * 3),
                            predictor=pred, max_retries=1, max_steps=3)
        r = run.recovery
        assert r.attempts == 2
        assert all(sr.status is SkillStatus.TIMEOUT for sr in run.attempt_results)
        assert r.aborted is True
        summary = next(x for x in records if x["event"] == "recovery_result")
        assert summary["predictive"]["num_triggers"] == 0
        assert summary["predictive"]["num_predictions"] == 6  # 3 步 x 2 次尝试

    def test_predictions_recorded_in_trace(self):
        """模型 prediction/confidence/是否触发 全部进结构化 trace。"""
        pred = MockPredictor(probs=[0.9, 0.0, 0.0], confidence=0.8)
        run, records = _run(ScriptedSkill([{"kind": "success", "steps": 2}] * 2),
                            predictor=pred, max_retries=2)
        preds = [r for r in records if r["event"] == "predictive_prediction"]
        assert len(preds) == 3
        first = preds[0]
        assert first["triggered"] is True and first["prediction"] is True
        assert first["prob"] == pytest.approx(0.9)
        assert first["confidence"] == pytest.approx(0.8)
        assert all(p["triggered"] is False for p in preds[1:])
        # 恢复决策带 predictive 标记
        dec = next(r for r in records if r["event"] == "recovery_decision")
        assert dec["predictive_triggered"] is True
        # skill_attempt 带预测摘要
        attempts = [r for r in records if r["event"] == "skill_attempt"]
        assert attempts[0]["predictive_triggered"] is True
        assert attempts[0]["num_predictions"] == 1
        assert attempts[1]["num_predictions"] == 2

    def test_invalid_recovery_mode_raises(self):
        pred = MockPredictor()
        adapter = MockAdapter()
        with pytest.raises(ValueError):
            RecoveryRunner(adapter, ScriptedSkill([{"kind": "success", "steps": 1}]),
                           task_id="x", predictor=pred, recovery_mode="b3")


# --------------------------------------------------------------------------
# 真实 checkpoint 预测器 (从 data/m4/checkpoints 重建)
# --------------------------------------------------------------------------
def _ckpt_paths():
    return (os.path.join(_REPO, "data", "m4", "checkpoints", "model.pt"),
            os.path.join(_REPO, "data", "m4", "checkpoints", "config.json"))


class TestRealCheckpointPredictor:
    def test_load_from_checkpoint_and_predict_valid(self):
        """从 checkpoint 重建 MLPFailurePredictor, 对合成特征产出有效概率。"""
        ckpt, cfg = _ckpt_paths()
        predictor = load_predictor_from_checkpoint(ckpt, cfg)
        assert isinstance(predictor, FailurePredictor)
        feats = np.zeros(22, dtype=np.float32)
        pred = predictor.predict(feats)
        assert np.isfinite(pred.prob) and 0.0 <= pred.prob <= 1.0
        assert np.isfinite(pred.confidence) and 0.0 <= pred.confidence <= 1.0
        # 归一化 mean/std 正确应用 (非零均值特征仍产出有限值)
        feats2 = np.concatenate([np.zeros(7), np.zeros(7), [0.5], np.zeros(7)]).astype(np.float32)
        pred2 = predictor.predict(feats2)
        assert np.isfinite(pred2.prob)

    def test_real_pickcube_integration(self):
        """真实 PickCube (ManiSkill3) + 真实 checkpoint 预测器, b2 全链路。"""
        mani = pytest.importorskip("mani_skill")
        mani.envs  # noqa: B018  触发环境注册
        from sim.pickcube_adapter import PickCubeAdapter
        from sim.pick_skill import PickCubeSkill

        ckpt, cfg = _ckpt_paths()
        predictor = load_predictor_from_checkpoint(ckpt, cfg)

        adapter = PickCubeAdapter(max_episode_steps=120)
        skill = PickCubeSkill(adapter)
        buf = io.StringIO()
        runner = RecoveryRunner(
            adapter, skill, task_id="pickcube-real",
            manager=RecoveryManager(max_retries=2),
            logger=ExecutionLogger(stream=buf),
            predictor=predictor,
            recovery_mode="b2",
            predictor_min_confidence=0.0,
        )
        try:
            run = runner.run(seed=0, max_steps=120)
        finally:
            runner.close()

        records = [json.loads(l) for l in buf.getvalue().strip().splitlines() if l.strip()]
        preds = [r for r in records if r["event"] == "predictive_prediction"]
        assert len(preds) > 0  # 真实执行期间确有逐步预测
        assert all(np.isfinite(r["prob"]) and np.isfinite(r["confidence"]) for r in preds)
        summary = next(r for r in records if r["event"] == "recovery_result")
        assert summary["predictive"]["mode"] == "b2"
        assert summary["predictive"]["num_predictions"] == len(preds)
        # 结果字段完整 (真实 env 成功/终止真值)
        assert "final_status" in summary and "env_success" in summary
