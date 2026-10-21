#!/usr/bin/env python3
"""Demo 1 — 正常抓取 (无故障)。

PickCubeSkill + TaskRunner，无故障注入。逐步骤打印观测状态，最后 SUCCESS。

运行:
    source scripts/activate_env.sh && export CUDA_VISIBLE_DEVICES=2
    python scripts/demo_normal.py

退出码 0 表示演示正常结束。
"""
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from interfaces import SimulatorAdapter
from sim import PickCubeAdapter, PickCubeSkill
from runtime import TaskRunner, ExecutionLogger


class TracingAdapter(SimulatorAdapter):
    """逐步骤打印观测状态，不改动被装饰 adapter 的行为 (纯透传 + 打印)。"""

    def __init__(self, inner, label="step"):
        self._inner = inner
        self._label = label

    def reset(self, *, seed=None, options=None):
        obs = self._inner.reset(seed=seed, options=options)
        print(f"[{self._label}] env reset  episode 开始 (step={obs.step})")
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


def main():
    seed = 0
    max_steps = 200
    print("=" * 68)
    print("Demo 1 — 正常抓取 (PickCubeSkill + TaskRunner, 无故障)")
    print(f"  seed={seed}  max_steps={max_steps}  skill=pick_cube_mplib (B0 oracle 基线)")
    print("=" * 68)

    adapter = PickCubeAdapter(max_episode_steps=max_steps)
    tracing = TracingAdapter(adapter)
    skill = PickCubeSkill(tracing)
    # 静默 JSON logger，控制台只保留清晰的逐步 trace。
    runner = TaskRunner(
        tracing, skill, task_id="PickCube-v1",
        logger=ExecutionLogger(stream=io.StringIO()),
    )
    try:
        result = runner.run(seed=seed, max_steps=max_steps)
    finally:
        runner.close()

    sr = result.skill_result
    ep = result.episode
    print("-" * 68)
    print("[RESULT] skill_status=%s  steps=%d  task_state=%s"
          % (sr.status.value, sr.steps, result.task_state.value))
    print("         env_success=%s  total_reward=%.3f"
          % (bool(ep.success) if ep is not None else None, result.total_reward))
    if sr.status.value == "success":
        print("[SUCCESS] 正常抓取完成：reach -> grasp -> close -> move-to-goal，无故障。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
