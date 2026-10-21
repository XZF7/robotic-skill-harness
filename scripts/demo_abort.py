#!/usr/bin/env python3
"""Demo 3 — 不可恢复失败 (object_displacement_large 0.5m)，安全中止。

PickCubeSkill + RecoveryRunner (B1)。在第 10 步把 cube 位移 0.5m，超出可达范围：
首次尝试预规划轨迹抓空 -> 重规划时运动规划失败 -> 有界重试 (max_retries=2) 耗尽 ->
RecoveryManager 决策 ABORT (RECOVERY_EXHAUSTED)。不把 reset 当作恢复。

运行:
    source scripts/activate_env.sh && export CUDA_VISIBLE_DEVICES=2
    python scripts/demo_abort.py

退出码 0 表示演示正常结束。
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from interfaces import SimulatorAdapter
from sim import PickCubeAdapter, PickCubeSkill
from sim.fault_injection import FaultInjector, FAULT_OBJECT_DISPLACEMENT
from runtime import RecoveryRunner, RecoveryManager, ExecutionLogger
from runtime.failure_detector import FailureDetector


class FaultInjectingAdapter(SimulatorAdapter):
    """每步调用 FaultInjector.after_step。"""

    def __init__(self, inner, injector):
        self._inner = inner
        self._injector = injector

    def reset(self, *, seed=None, options=None):
        return self._inner.reset(seed=seed, options=options)

    def step(self, action):
        res = self._inner.step(action)
        self._injector.after_step(res.observation)
        return res

    def close(self):
        self._inner.close()

    @property
    def num_envs(self):
        return self._inner.num_envs

    @property
    def action_space(self):
        return self._inner.action_space

    @property
    def max_episode_steps(self):
        return self._inner.max_episode_steps

    @property
    def env(self):
        return self._inner.env


class TracingAdapter(SimulatorAdapter):
    """逐步骤打印观测状态 (透传)。"""

    def __init__(self, inner, label="step"):
        self._inner = inner
        self._label = label

    def reset(self, *, seed=None, options=None):
        obs = self._inner.reset(seed=seed, options=options)
        print(f"[{self._label}] env reset (唯一一次, 恢复期间不 reset env)")
        return obs

    def step(self, action):
        res = self._inner.step(action)
        o = res.observation
        print(f"[{self._label} {o.step:4d}] gripper={o.policy.gripper:.2f} "
              f"is_grasped={bool(o.truth.is_grasped)} success={bool(o.truth.success)}")
        return res

    def close(self):
        self._inner.close()

    @property
    def num_envs(self):
        return self._inner.num_envs

    @property
    def action_space(self):
        return self._inner.action_space

    @property
    def max_episode_steps(self):
        return self._inner.max_episode_steps

    @property
    def env(self):
        return self._inner.env


class TracingSkill:
    """包装 PickCubeSkill，在每次 skill.reset (重规划) 时打印 attempt 边界。"""

    skill_id = "pick_cube_mplib"
    privileged_access = True

    def __init__(self, inner):
        self._inner = inner
        self._attempt = 0

    def reset(self, obs):
        self._attempt += 1
        print(f"\n[skill] attempt {self._attempt}: skill.reset -> 重新规划 "
              f"(reach->grasp->close->goal)，不 reset env")
        self._inner.reset(obs)

    def act(self, obs):
        return self._inner.act(obs)

    @property
    def is_done(self):
        return self._inner.is_done

    @property
    def status(self):
        return self._inner.status

    def result(self, final_obs=None):
        return self._inner.result(final_obs)

    @property
    def attempt(self):
        return self._attempt


class InstrumentedDetector(FailureDetector):
    """记录每次尝试检测到的失败类型与最终抓手开度。"""

    def __init__(self):
        super().__init__()
        self.observations = []

    def detect(self, result, final_obs=None, obs_history=None):
        ev = super().detect(result, final_obs, obs_history=obs_history)
        final_gripper = None
        if final_obs is not None:
            final_gripper = float(final_obs.policy.gripper)
        self.observations.append({
            "detected": ev.type.value if ev is not None else None,
            "final_gripper": final_gripper,
        })
        return ev


def main():
    seed = 0
    max_steps = 300
    fault_step = 10
    fault_params = {"delta": [0.5, 0.0, 0.0]}
    print("=" * 68)
    print("Demo 3 — 不可恢复失败 (object_displacement_large 0.5m)，安全中止")
    print(f"  seed={seed}  fault=object_displacement@step{fault_step} delta={fault_params['delta']}")
    print("  max_retries=2  recovery_mode=b1 (FailureDetector + RecoveryManager)")
    print("=" * 68)

    adapter = PickCubeAdapter(max_episode_steps=max_steps)
    injector = FaultInjector(adapter, seed=seed)
    injector.schedule(FAULT_OBJECT_DISPLACEMENT, fault_step,
                      params=fault_params, duration=1)
    faulting = FaultInjectingAdapter(adapter, injector)
    tracing = TracingAdapter(faulting)
    skill = TracingSkill(PickCubeSkill(tracing))
    detector = InstrumentedDetector()
    manager = RecoveryManager(max_retries=2)
    runner = RecoveryRunner(
        tracing, skill, task_id="PickCube-v1",
        manager=manager, detector=detector, recovery_mode="b1",
        logger=ExecutionLogger(stream=io.StringIO()),
    )
    try:
        result = runner.run(seed=seed, max_steps=max_steps)
    finally:
        runner.close()

    print("-" * 68)
    print("[故障注入核实]")
    for rec in injector.records:
        print(f"  injected={rec.injected}  type={rec.fault_type}  step={rec.step}  "
              f"distance={rec.params.get('distance'):.3f}m")

    print("\n[逐次尝试]")
    for i, ar in enumerate(result.attempt_results, start=1):
        ftype = ar.failure.type.value if ar.failure is not None else None
        print(f"  attempt {i}: status={ar.status.value}  steps={ar.steps}  "
              f"failure={ftype}  msg={ar.message!r}")
        if i - 1 < len(detector.observations):
            ob = detector.observations[i - 1]
            print(f"             detector -> {ob['detected']}  "
                  f"final_gripper={ob['final_gripper'] if ob['final_gripper'] is not None else 'n/a'}")
        else:
            print(f"             detector -> not invoked (attempt succeeded, no failure to detect)")

    print("\n[恢复决策序列]")
    for d in result.recovery.decisions:
        exhausted = d.detail.get("exhausted", False)
        tag = "  <-- RECOVERY_EXHAUSTED" if exhausted else ""
        print(f"  retry_count={d.retry_count}  action={d.action.value}  "
              f"failure_type={d.detail.get('failure_type')}  reason={d.reason!r}{tag}")

    rec = result.recovery
    print("-" * 68)
    print(f"[RESULT] attempts={rec.attempts}  recovered={rec.recovered}  "
          f"aborted={rec.aborted}  final_status={rec.final_skill_result.status.value}")
    if rec.aborted:
        print("[ABORT] 规划失败 -> 有界重试 max_retries=2 耗尽 -> RECOVERY_EXHAUSTED -> "
              "安全中止 (未把 reset 当作恢复)。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
