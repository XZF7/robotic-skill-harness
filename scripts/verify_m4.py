"""M4 端到端验证: B0(基线) vs B1(规则恢复) vs B2(规则+学习预判) matched 对比。

用法:
  python scripts/verify_m4.py --smoke
  python scripts/verify_m4.py --faults object_displacement grasp_perturb object_displacement_large --seeds 0 1 2 3 4
  python scripts/verify_m4.py --faults object_displacement grasp_perturb object_displacement_large \
                             --seeds 0 1 2 3 4 5 6 7 8 9 --out logs/m4_benchmark.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from interfaces import EpisodeResult, SimulatorAdapter
from sim import PickCubeAdapter, PickCubeSkill
from sim.fault_injection import (
    FaultInjector, FAULT_OBJECT_DISPLACEMENT, FAULT_GRASP_PERTURB,
)
from runtime import TaskRunner, RecoveryRunner, SkillExecutor
from runtime.recovery import RecoveryManager
from learning.model import MLP, MLPFailurePredictor
from eval import (
    attach_fault, attach_recovery, attach_privileged, attach_baseline,
    compute_recovery_metrics, require_sim,
)

FAULTS = {
    "object_displacement": (FAULT_OBJECT_DISPLACEMENT, 10, {"delta": [0.07, 0.0, 0.0]}, 1),
    "grasp_perturb": (FAULT_GRASP_PERTURB, 25, {"scale": 0.25}, 1),
    "object_displacement_large": (FAULT_OBJECT_DISPLACEMENT, 10, {"delta": [0.5, 0.0, 0.0]}, 1),
}

CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "m4", "checkpoints", "model.pt")
PREDICTOR_THRESHOLD = 0.5  # C 推荐: val 过小, 0.5 比 0.35 稳健


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


class AccumulatingExecutor(SkillExecutor):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.episode_reward = 0.0

    def execute(self, *a, **k):
        r = super().execute(*a, **k)
        self.episode_reward += self.total_reward
        return r


def _make_episode(task_id, seed, baseline, privileged, fault_records, recovery,
                  skill_result, episode, total_reward):
    ep = EpisodeResult(
        task_id=task_id, success=bool(episode.success) if episode is not None else False,
        steps=skill_result.steps, reward=total_reward,
        skill_results=[skill_result],
        failures=[skill_result.failure] if skill_result.failure is not None else [],
        metrics={}, source="sim",
    )
    if fault_records:
        ep = attach_fault(ep, fault_records[0])
    if recovery is not None:
        ep = attach_recovery(ep, recovery)
    ep = attach_privileged(ep, privileged)
    ep = attach_baseline(ep, baseline)
    ep.metrics["seed"] = seed
    return ep


def _mk(seed, ftype, fstep, fparams, fdur):
    adapter = PickCubeAdapter(max_episode_steps=300)
    injector = FaultInjector(adapter, seed=seed)
    injector.schedule(ftype, fstep, params=fparams, duration=fdur)
    return adapter, injector, FaultInjectingAdapter(adapter, injector)


def run_b0(seed, ftype, fstep, fparams, fdur):
    adapter, injector, fadapter = _mk(seed, ftype, fstep, fparams, fdur)
    skill = PickCubeSkill(fadapter)
    runner = TaskRunner(fadapter, skill, task_id="PickCube-v1")
    try:
        r = runner.run(seed=seed, max_steps=300)
    finally:
        runner.close()
    return _make_episode("PickCube-v1", seed, "b0", True, injector.records,
                         None, r.skill_result, r.episode, r.total_reward), 0


def run_b1(seed, ftype, fstep, fparams, fdur):
    adapter, injector, fadapter = _mk(seed, ftype, fstep, fparams, fdur)
    skill = PickCubeSkill(fadapter)
    executor = AccumulatingExecutor()
    manager = RecoveryManager(max_retries=2)
    runner = RecoveryRunner(fadapter, skill, task_id="PickCube-v1",
                            executor=executor, manager=manager, recovery_mode="b1")
    try:
        r = runner.run(seed=seed, max_steps=300)
    finally:
        runner.close()
    sr = r.recovery.final_skill_result
    return _make_episode("PickCube-v1", seed, "b1", True, injector.records,
                         r.recovery, sr, r.episode, executor.episode_reward), 0


def run_b2(seed, ftype, fstep, fparams, fdur, predictor, threshold):
    adapter, injector, fadapter = _mk(seed, ftype, fstep, fparams, fdur)
    skill = PickCubeSkill(fadapter)
    executor = AccumulatingExecutor()
    manager = RecoveryManager(max_retries=2)
    runner = RecoveryRunner(fadapter, skill, task_id="PickCube-v1",
                            executor=executor, manager=manager, recovery_mode="b2",
                            predictor=predictor, predictor_threshold=threshold)
    try:
        r = runner.run(seed=seed, max_steps=300)
    finally:
        runner.close()
    sr = r.recovery.final_skill_result
    num_triggers = sum(1 for rec in runner.guard.records if rec.triggered) if runner.guard else 0
    num_preds = len(runner.guard.records) if runner.guard else 0
    ep = _make_episode("PickCube-v1", seed, "b2", True, injector.records,
                       r.recovery, sr, r.episode, executor.episode_reward)
    return ep, {"num_predictions": num_preds, "num_triggers": num_triggers}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--faults", nargs="*", default=["object_displacement"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    if args.smoke:
        args.faults = ["object_displacement"]
        args.seeds = [0]

    predictor = load_predictor(threshold=args.threshold)
    b0_eps, b1_eps, b2_eps = [], [], []
    details = []
    for fname in args.faults:
        ftype, fstep, fparams, fdur = FAULTS[fname]
        for seed in args.seeds:
            b0, _ = run_b0(seed, ftype, fstep, fparams, fdur)
            b1, _ = run_b1(seed, ftype, fstep, fparams, fdur)
            b2, b2meta = run_b2(seed, ftype, fstep, fparams, fdur, predictor, args.threshold)
            b0_eps.append(b0); b1_eps.append(b1); b2_eps.append(b2)
            details.append({
                "fault": fname, "seed": seed,
                "b0": {"success": b0.success, "steps": b0.steps},
                "b1": {"success": b1.success, "steps": b1.steps,
                       "attempts": b1.metrics.get("recovery", {}).get("attempts")},
                "b2": {"success": b2.success, "steps": b2.steps,
                       "attempts": b2.metrics.get("recovery", {}).get("attempts"),
                       **b2meta},
            })

    require_sim(b0_eps + b1_eps + b2_eps, context="M4")
    m0 = compute_recovery_metrics(b0_eps)
    m1 = compute_recovery_metrics(b1_eps)
    m2 = compute_recovery_metrics(b2_eps)

    # 误触发: B2 预判触发但 matched B1 (纯规则) 首次尝试即成功 (无需恢复)
    false_trigger_eps = 0
    for d in details:
        if d["b2"]["num_triggers"] > 0 and d["b1"]["attempts"] == 1:
            false_trigger_eps += 1
    total_triggers = sum(d["b2"]["num_triggers"] for d in details)

    report = {
        "faults": args.faults, "seeds": args.seeds,
        "predictor_threshold": args.threshold,
        "details": details,
        "b0": m0.to_dict(), "b1": m1.to_dict(), "b2": m2.to_dict(),
        "false_recovery_trigger_episodes": false_trigger_eps,
        "total_predictive_triggers": total_triggers,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[report saved] {args.out}")


if __name__ == "__main__":
    main()
