"""runtime/verify_temporal_guard.py — M5-B 时序门控变体的真实 PickCube 选参 (轻量)。

在【验证种子】上用真实 checkpoint + 真实 PickCube 对比:
  b2 (参考) vs debounce K=3 / K=5 vs phase (gripper_closed) vs cooldown。
指标: task_success / recovery(attempts) / num_triggers / false-trigger (对比 b1 首试成功)。

用法 (仓库根目录, CPU-only):
  source scripts/activate_env.sh; export CUDA_VISIBLE_DEVICES=2
  python3 runtime/verify_temporal_guard.py --faults object_displacement --seeds 40 41 42 43 44 \
      --out logs/m5_temporal_guard.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch

from interfaces import SimulatorAdapter
from sim import PickCubeAdapter, PickCubeSkill
from sim.fault_injection import (
    FaultInjector, FAULT_OBJECT_DISPLACEMENT, FAULT_GRASP_PERTURB,
)
from runtime import (
    ExecutionLogger, RecoveryRunner, SkillExecutor, RecoveryManager,
    TemporalGateConfig, TemporalGateMode,
)
from learning.model import MLP, MLPFailurePredictor

FAULTS = {
    "object_displacement": (FAULT_OBJECT_DISPLACEMENT, 10, {"delta": [0.07, 0.0, 0.0]}, 1),
    "grasp_perturb": (FAULT_GRASP_PERTURB, 25, {"scale": 0.25}, 1),
    "object_displacement_large": (FAULT_OBJECT_DISPLACEMENT, 10, {"delta": [0.5, 0.0, 0.0]}, 1),
}

CKPT = os.path.join(_REPO, "data", "m4", "checkpoints", "model.pt")
PREDICTOR_THRESHOLD = 0.5  # M4 决定: 0.5 比 0.35 稳健

# 变体定义: name -> (recovery_mode, gate_config 或 None)
VARIANTS = {
    "b2": ("b2", None),
    "debounce_k3": ("b2t", TemporalGateConfig(mode=TemporalGateMode.DEBOUNCE, debounce_k=3)),
    "debounce_k5": ("b2t", TemporalGateConfig(mode=TemporalGateMode.DEBOUNCE, debounce_k=5)),
    "phase": ("b2t", TemporalGateConfig(mode=TemporalGateMode.PHASE,
                                         phase_signal="gripper_closed",
                                         phase_gripper_threshold=0.5)),
    "cooldown": ("b2t", TemporalGateConfig(mode=TemporalGateMode.COOLDOWN, cooldown_steps=10)),
}


def load_predictor(path=CKPT, threshold=PREDICTOR_THRESHOLD):
    ckpt = torch.load(path, map_location="cpu")
    model = MLP(input_dim=ckpt["input_dim"], hidden_dims=ckpt["hidden_dims"],
                dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    return MLPFailurePredictor(model, mean=ckpt["mean"], std=ckpt["std"],
                               threshold=threshold)


class FaultInjectingAdapter(SimulatorAdapter):
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


def _mk(seed, ftype, fstep, fparams, fdur):
    adapter = PickCubeAdapter(max_episode_steps=300)
    injector = FaultInjector(adapter, seed=seed)
    injector.schedule(ftype, fstep, params=fparams, duration=fdur)
    return adapter, injector, FaultInjectingAdapter(adapter, injector)


def run_variant(seed, ftype, fstep, fparams, fdur, predictor, threshold, gate):
    """跑一次带故障的 episode (recovery_mode=b2/b2t + 可选时序门控)。"""
    adapter, injector, fadapter = _mk(seed, ftype, fstep, fparams, fdur)
    skill = PickCubeSkill(fadapter)
    executor = SkillExecutor()
    manager = RecoveryManager(max_retries=2)
    mode = "b2" if gate is None else "b2t"
    runner = RecoveryRunner(
        fadapter, skill, task_id="PickCube-v1",
        executor=executor, manager=manager, recovery_mode=mode,
        logger=ExecutionLogger(stream=io.StringIO()),
        predictor=predictor, predictor_threshold=threshold,
        temporal_gate=gate,
    )
    try:
        r = runner.run(seed=seed, max_steps=300)
    finally:
        runner.close()
    sr = r.recovery.final_skill_result
    recs = runner.guard.records if runner.guard else []
    triggers = [rec.step for rec in recs if rec.triggered]
    return {
        "success": bool(r.episode.success) if r.episode is not None else False,
        "steps": sr.steps,
        "attempts": r.recovery.attempts,
        "recovered": r.recovery.recovered,
        "aborted": r.recovery.aborted,
        "num_predictions": len(recs),
        "num_triggers": len(triggers),
        "trigger_steps": triggers,
        "fault_injected": bool(injector.records and injector.records[0].injected),
    }


def run_b1(seed, ftype, fstep, fparams, fdur):
    """纯规则恢复参考 (用于 false-trigger 判定)。"""
    adapter, injector, fadapter = _mk(seed, ftype, fstep, fparams, fdur)
    skill = PickCubeSkill(fadapter)
    executor = SkillExecutor()
    manager = RecoveryManager(max_retries=2)
    runner = RecoveryRunner(fadapter, skill, task_id="PickCube-v1",
                            executor=executor, manager=manager, recovery_mode="b1",
                            logger=ExecutionLogger(stream=io.StringIO()))
    try:
        r = runner.run(seed=seed, max_steps=300)
    finally:
        runner.close()
    return {
        "success": bool(r.episode.success) if r.episode is not None else False,
        "attempts": r.recovery.attempts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faults", nargs="*", default=["object_displacement"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[40, 41, 42, 43, 44])
    ap.add_argument("--out", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS.keys()))
    args = ap.parse_args()

    predictor = load_predictor(threshold=args.threshold)
    details = []
    b1_by_key = {}

    for fname in args.faults:
        ftype, fstep, fparams, fdur = FAULTS[fname]
        for seed in args.seeds:
            b1 = run_b1(seed, ftype, fstep, fparams, fdur)
            b1_by_key[(fname, seed)] = b1
            for vname in args.variants:
                mode, gate = VARIANTS[vname]
                d = run_variant(seed, ftype, fstep, fparams, fdur,
                                predictor, args.threshold, gate)
                d.update({"fault": fname, "seed": seed, "variant": vname})
                details.append(d)
                print(f"[{fname} seed={seed} {vname}] success={d['success']} "
                      f"attempts={d['attempts']} triggers={d['num_triggers']} "
                      f"trigger_steps={d['trigger_steps']} injected={d['fault_injected']}")

    # 汇总 + false-trigger 判定
    summary = {}
    for vname in args.variants:
        eps = [d for d in details if d["variant"] == vname]
        n = len(eps)
        false_trigger = 0
        for d in eps:
            key = (d["fault"], d["seed"])
            b1 = b1_by_key[key]
            # 误触发: 变体触发但 matched b1 首次尝试即成功 (无需恢复)
            if d["num_triggers"] > 0 and b1["attempts"] == 1:
                false_trigger += 1
        summary[vname] = {
            "episodes": n,
            "task_success": sum(1 for d in eps if d["success"]),
            "success_rate": round(sum(1 for d in eps if d["success"]) / n, 4),
            "recovered_episodes": sum(1 for d in eps if d["recovered"]),
            "aborted_episodes": sum(1 for d in eps if d["aborted"]),
            "total_triggers": sum(d["num_triggers"] for d in eps),
            "total_predictions": sum(d["num_predictions"] for d in eps),
            "false_trigger_episodes": false_trigger,
        }

    report = {
        "faults": args.faults,
        "seeds": args.seeds,
        "predictor_threshold": args.threshold,
        "summary": summary,
        "b1_reference": {f"{f}:{s}": b for (f, s), b in sorted(b1_by_key.items())},
        "details": details,
    }
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[report saved] {args.out}")


if __name__ == "__main__":
    main()
