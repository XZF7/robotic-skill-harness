#!/usr/bin/env python3
"""录制演示视频（需 GPU + 可用 Vulkan 渲染后端）。

本机为 CPU-only headless 且 GPU 均被其他任务占用，默认渲染模式未在本机验证。
在有空闲 GPU 的机器上运行（渲染后端自动遵循 CUDA_VISIBLE_DEVICES）:

    source scripts/activate_env.sh
    export CUDA_VISIBLE_DEVICES=<空闲 GPU id>
    python scripts/record_demos.py --scenario normal     # -> videos/demo_normal.mp4
    python scripts/record_demos.py --scenario recovery   # -> videos/demo_recovery.mp4
    python scripts/record_demos.py --scenario abort      # -> videos/demo_abort.mp4
    python scripts/record_demos.py --all                 # 三个都录

本机干跑（不渲染，验证场景逻辑）:
    python scripts/record_demos.py --all --headless

机器人行为与 scripts/demo_*.py 完全一致，仅额外逐帧抓取 render_rgb_array()。
"""
import argparse
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from interfaces import SimulatorAdapter
from sim import PickCubeAdapter, PickCubeSkill
from sim.fault_injection import FaultInjector, FAULT_OBJECT_DISPLACEMENT
from runtime import TaskRunner, RecoveryRunner, RecoveryManager, ExecutionLogger
from runtime.failure_detector import FailureDetector
from mani_skill.utils.visualization.misc import images_to_video


class FaultInjectingAdapter(SimulatorAdapter):
    """每步调用 FaultInjector.after_step，让调度故障在正确步数真实注入。"""

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


class RecordingAdapter(SimulatorAdapter):
    """最外层包装：reset/step 之后抓一帧 render_rgb_array()。"""

    def __init__(self, inner, capture=True):
        self._inner = inner
        self.frames = []
        self._capture_enabled = capture

    def _capture(self):
        if not self._capture_enabled:
            return
        try:
            img = self._inner.env.unwrapped.render_rgb_array()
            if img is not None:
                if getattr(img, "ndim", None) == 4:  # (num_envs,H,W,3) -> (H,W,3)
                    img = img[0]
                self.frames.append(img)
        except Exception as exc:  # 渲染失败不阻断执行, 如实告警
            print(f"[record] render_rgb_array failed: {exc}")

    def reset(self, *, seed=None, options=None):
        obs = self._inner.reset(seed=seed, options=options)
        self._capture()
        return obs

    def step(self, action):
        res = self._inner.step(action)
        self._capture()
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


def _adapter(max_steps, headless):
    if headless:
        return PickCubeAdapter(max_episode_steps=max_steps)  # render_mode/backend 默认 none
    return PickCubeAdapter(
        render_mode="rgb_array", render_backend="gpu", max_episode_steps=max_steps
    )


def _logger():
    return ExecutionLogger(stream=io.StringIO())  # 静默 JSON logger


def run_normal(headless):
    adapter = _adapter(200, headless)
    rec = RecordingAdapter(adapter, capture=not headless)
    skill = PickCubeSkill(rec)
    runner = TaskRunner(rec, skill, task_id="PickCube-v1", logger=_logger())
    try:
        result = runner.run(seed=0, max_steps=200)
        print(f"[normal] status={result.skill_result.status.value} "
              f"steps={result.skill_result.steps}")
    finally:
        runner.close()
    return rec.frames


def run_fault(name, delta, headless):
    adapter = _adapter(300, headless)
    injector = FaultInjector(adapter, seed=0)
    injector.schedule(FAULT_OBJECT_DISPLACEMENT, 10, params={"delta": delta}, duration=1)
    faulting = FaultInjectingAdapter(adapter, injector)
    rec = RecordingAdapter(faulting, capture=not headless)
    skill = PickCubeSkill(rec)
    detector = FailureDetector()
    manager = RecoveryManager(max_retries=2)
    runner = RecoveryRunner(
        rec, skill, task_id="PickCube-v1", detector=detector, manager=manager,
        recovery_mode="b1", logger=_logger(),
    )
    try:
        result = runner.run(seed=0, max_steps=300)
        r = result.recovery
        print(f"[{name}] attempts={r.attempts} recovered={r.recovered} "
              f"aborted={r.aborted} final={r.final_skill_result.status.value}")
    finally:
        runner.close()
    return rec.frames


def save(frames, name):
    os.makedirs("videos", exist_ok=True)
    images_to_video(frames, output_dir="videos", video_name=f"demo_{name}.mp4", fps=10)
    print(f"[record] saved videos/demo_{name}.mp4 ({len(frames)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["normal", "recovery", "abort"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--headless", action="store_true",
                    help="不渲染，仅验证场景逻辑（本机干跑用）")
    args = ap.parse_args()

    scenarios = ["normal", "recovery", "abort"] if args.all else [args.scenario]
    if scenarios == [None]:
        ap.error("需要 --scenario 或 --all")

    for sc in scenarios:
        if sc == "normal":
            frames = run_normal(args.headless)
        elif sc == "recovery":
            frames = run_fault("recovery", [0.07, 0.0, 0.0], args.headless)
        else:
            frames = run_fault("abort", [0.5, 0.0, 0.0], args.headless)
        if args.headless:
            print(f"[record] {sc}: headless 干跑完成，抓帧数={len(frames)}（不渲染）")
        elif frames:
            save(frames, sc)
        else:
            print(f"[record] {sc}: 未抓取到任何帧（渲染不可用？）")


if __name__ == "__main__":
    main()
