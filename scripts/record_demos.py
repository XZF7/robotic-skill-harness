"""录制演示视频（需 GPU + 可用 Vulkan 渲染后端）。

本机为 CPU-only headless 且 4× GPU 均被其他任务占用，此脚本【未在本机验证】——
录制前请在有健康 GPU 的机器上运行，见 docs/DEMO_GUIDE.md §4。

用法（GPU 机器上）:
    source scripts/activate_env.sh
    export CUDA_VISIBLE_DEVICES=<空闲 GPU id>
    python scripts/record_demos.py --scenario normal
    python scripts/record_demos.py --scenario recovery
    python scripts/record_demos.py --scenario abort
    python scripts/record_demos.py --all          # 三个都录

输出: videos/demo_{scenario}.mp4
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from sim import PickCubeAdapter, PickCubeSkill
from sim.fault_injection import FaultInjector, FAULT_OBJECT_DISPLACEMENT
from runtime import TaskRunner, RecoveryRunner
from runtime.recovery import RecoveryManager


class RecordingAdapter:
    """包装 PickCubeAdapter: 每步之后抓一帧 render_rgb_array()。"""

    def __init__(self, adapter):
        self._inner = adapter
        self.frames = []

    def _capture(self):
        try:
            img = self._inner.env.unwrapped.render_rgb_array()
            if img is not None:
                self.frames.append(img)
        except Exception as exc:  # 渲染失败不阻断执行, 但如实告警
            print(f"[record] render_rgb_array failed: {exc}")

    def reset(self, *, seed=None, options=None):
        obs = self._inner.reset(seed=seed, options=options)
        self._capture()
        return obs

    def step(self, action):
        res = self._inner.step(action)
        self._capture()
        return res

    def __getattr__(self, name):
        return getattr(self._inner, name)


def run_normal():
    adapter = PickCubeAdapter(render_mode="rgb_array", render_backend="gpu")
    rec = RecordingAdapter(adapter)
    skill = PickCubeSkill(rec)
    runner = TaskRunner(rec, skill, task_id="PickCube-v1")
    try:
        r = runner.run(seed=0, max_steps=300)
        print(f"[normal] status={r.skill_result.status} steps={r.skill_result.steps}")
    finally:
        runner.close()
    return rec.frames


def run_fault(scenario, fault_delta):
    adapter = PickCubeAdapter(render_mode="rgb_array", render_backend="gpu")
    rec = RecordingAdapter(adapter)
    injector = FaultInjector(adapter, seed=0)
    injector.schedule(FAULT_OBJECT_DISPLACEMENT, 10, params={"delta": fault_delta}, duration=1)

    class FaultingRecordingAdapter(RecordingAdapter):
        def step(self, action):
            res = super().step(action)
            injector.after_step(res.observation)
            return res

    fadapter = FaultingRecordingAdapter(adapter)
    fadapter.frames = rec.frames
    skill = PickCubeSkill(fadapter)
    manager = RecoveryManager(max_retries=2)
    runner = RecoveryRunner(fadapter, skill, task_id="PickCube-v1", manager=manager, recovery_mode="b1")
    try:
        r = runner.run(seed=0, max_steps=300)
        print(f"[{scenario}] attempts={r.recovery.attempts} recovered={r.recovery.recovered} "
              f"aborted={r.recovery.aborted} final={r.recovery.final_skill_result.status}")
    finally:
        runner.close()
    return rec.frames


def save(frames, name):
    from mani_skill.utils.visualization.misc import images_to_video
    os.makedirs("videos", exist_ok=True)
    images_to_video(frames, output_dir="videos", video_name=f"demo_{name}.mp4", fps=10)
    print(f"[record] saved videos/demo_{name}.mp4 ({len(frames)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["normal", "recovery", "abort"])
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    scenarios = []
    if args.all:
        scenarios = ["normal", "recovery", "abort"]
    elif args.scenario:
        scenarios = [args.scenario]
    else:
        ap.error("need --scenario or --all")

    for sc in scenarios:
        if sc == "normal":
            frames = run_normal()
        elif sc == "recovery":
            frames = run_fault("recovery", [0.07, 0.0, 0.0])
        else:  # abort
            frames = run_fault("abort", [0.5, 0.0, 0.0])
        if frames:
            save(frames, sc)
        else:
            print(f"[record] {sc}: no frames captured (rendering unavailable?)")


if __name__ == "__main__":
    main()
