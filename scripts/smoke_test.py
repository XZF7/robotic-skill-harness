#!/usr/bin/env python3
"""轻量冒烟测试 (smoke test)：1 故障 x 1 种子 x B0/B1。

比全量基准便宜得多 (2 个 episode，约 20 秒，CPU-only)。验证整条链路可用:
  - B0 (TaskRunner, 无恢复): object_displacement 0.07m -> 首次尝试应失败。
  - B1 (RecoveryRunner, 规则恢复): 同故障 -> 应恢复成功 (recovered=True)。

运行:
    source scripts/activate_env.sh && export CUDA_VISIBLE_DEVICES=2
    python scripts/smoke_test.py

通过条件: 故障注入核实 (injected=True) 且 B1 恢复成功。
退出码 0 = 通过；非 0 = 失败。
"""
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from interfaces import SimulatorAdapter
from sim import PickCubeAdapter, PickCubeSkill
from sim.fault_injection import FaultInjector, FAULT_OBJECT_DISPLACEMENT
from runtime import TaskRunner, RecoveryRunner, RecoveryManager, ExecutionLogger


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


def _mk(seed):
    adapter = PickCubeAdapter(max_episode_steps=300)
    injector = FaultInjector(adapter, seed=seed)
    injector.schedule(FAULT_OBJECT_DISPLACEMENT, 10,
                      params={"delta": [0.07, 0.0, 0.0]}, duration=1)
    return adapter, injector, FaultInjectingAdapter(adapter, injector)


def run_b0(seed):
    adapter, injector, fadapter = _mk(seed)
    runner = TaskRunner(fadapter, PickCubeSkill(fadapter), task_id="PickCube-v1",
                        logger=ExecutionLogger(stream=io.StringIO()))
    try:
        r = runner.run(seed=seed, max_steps=300)
    finally:
        runner.close()
    return {
        "success": bool(r.episode.success) if r.episode is not None else False,
        "steps": r.skill_result.steps,
        "status": r.skill_result.status.value,
        "injected": bool(injector.records[0].injected) if injector.records else False,
    }


def run_b1(seed):
    adapter, injector, fadapter = _mk(seed)
    manager = RecoveryManager(max_retries=2)
    runner = RecoveryRunner(fadapter, PickCubeSkill(fadapter), task_id="PickCube-v1",
                            manager=manager, recovery_mode="b1",
                            logger=ExecutionLogger(stream=io.StringIO()))
    try:
        r = runner.run(seed=seed, max_steps=300)
    finally:
        runner.close()
    return {
        "success": bool(r.recovery.final_skill_result.status.value == "success"),
        "attempts": r.recovery.attempts,
        "recovered": r.recovery.recovered,
        "aborted": r.recovery.aborted,
        "injected": bool(injector.records[0].injected) if injector.records else False,
    }


def main():
    seed = 0
    b0 = run_b0(seed)
    b1 = run_b1(seed)
    passed = bool(b1["injected"] and b1["recovered"] and b1["success"])
    report = {"fault": "object_displacement", "seed": seed, "b0": b0, "b1": b1,
              "passed": passed}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("\nSMOKE %s" % ("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
