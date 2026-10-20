"""M3-C 端到端验证: 真实故障注入 + 失败检测 + 规则恢复 + B0/B1 评估。

路径: FaultInjector -> FaultInjectingAdapter -> (TaskRunner[B0] | RecoveryRunner[B1])
      -> EpisodeResult(source=sim) -> eval.compute_recovery_metrics + compare_b0_b1

用法:
  python scripts/verify_m3c.py --smoke                              # 冒烟: 1 故障 x 1 种子
  python scripts/verify_m3c.py --faults object_displacement --seeds 0 1 2
  python scripts/verify_m3c.py --faults object_displacement grasp_perturb gripper_drop \
                               --seeds 0 1 2 3 4 5 6 7 8 9 --out logs/m3c_recovery.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interfaces import EpisodeResult, SimulatorAdapter  # noqa: E402
from sim import PickCubeAdapter, PickCubeSkill  # noqa: E402
from sim.fault_injection import (  # noqa: E402
    FaultInjector,
    FAULT_OBJECT_DISPLACEMENT,
    FAULT_GRASP_PERTURB,
    FAULT_GRIPPER_DROP,
)
from runtime import TaskRunner, RecoveryRunner, SkillExecutor  # noqa: E402
from runtime.recovery import RecoveryManager  # noqa: E402
from eval import (  # noqa: E402
    attach_fault, attach_recovery, attach_privileged, attach_baseline,
    compute_recovery_metrics, compare_b0_b1, require_sim,
)

# (fault_type, 注入时机 step, 默认参数)
# (fault_type, 注入时机 step, 参数, 持续步数 duration)
FAULTS = {
    "object_displacement": (FAULT_OBJECT_DISPLACEMENT, 10, {"delta": [0.07, 0.0, 0.0]}, 1),
    "grasp_perturb": (FAULT_GRASP_PERTURB, 25, {"scale": 0.25}, 1),
    "gripper_drop": (FAULT_GRIPPER_DROP, 45, {}, 6),
    "object_displacement_large": (FAULT_OBJECT_DISPLACEMENT, 10, {"delta": [0.5, 0.0, 0.0]}, 1),
}


class FaultInjectingAdapter(SimulatorAdapter):
    """包装 SimulatorAdapter: 每步后触发 FaultInjector 的预定故障。"""

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
    """跨多次 execute() 累计 reward (B1 多次尝试的 episode 累计奖励)。"""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.episode_reward = 0.0

    def execute(self, *a, **k):
        r = super().execute(*a, **k)
        self.episode_reward += self.total_reward
        return r


def _make_episode(task_id, seed, baseline, privileged, fault_records,
                  recovery, skill_result, episode, total_reward):
    ep = EpisodeResult(
        task_id=task_id,
        success=bool(episode.success) if episode is not None else False,
        steps=skill_result.steps,
        reward=total_reward,
        skill_results=[skill_result],
        failures=[skill_result.failure] if skill_result.failure is not None else [],
        metrics={},
        source="sim",
    )
    if fault_records:
        ep = attach_fault(ep, fault_records[0])
    if recovery is not None:
        ep = attach_recovery(ep, recovery)
    ep = attach_privileged(ep, privileged)
    ep = attach_baseline(ep, baseline)
    ep.metrics["seed"] = seed
    return ep


def run_b0(seed, fault_type, fault_step, fault_params, fault_duration=1):
    adapter = PickCubeAdapter(max_episode_steps=300)
    injector = FaultInjector(adapter, seed=seed)
    injector.schedule(fault_type, fault_step, params=fault_params, duration=fault_duration)
    fadapter = FaultInjectingAdapter(adapter, injector)
    skill = PickCubeSkill(fadapter)
    runner = TaskRunner(fadapter, skill, task_id="PickCube-v1")
    try:
        r = runner.run(seed=seed, max_steps=300)
    finally:
        runner.close()
    return _make_episode("PickCube-v1", seed, "b0", True, injector.records,
                         None, r.skill_result, r.episode, r.total_reward)


def run_b1(seed, fault_type, fault_step, fault_params, fault_duration=1):
    adapter = PickCubeAdapter(max_episode_steps=300)
    injector = FaultInjector(adapter, seed=seed)
    injector.schedule(fault_type, fault_step, params=fault_params, duration=fault_duration)
    fadapter = FaultInjectingAdapter(adapter, injector)
    skill = PickCubeSkill(fadapter)
    executor = AccumulatingExecutor()
    manager = RecoveryManager(max_retries=2)
    runner = RecoveryRunner(fadapter, skill, task_id="PickCube-v1",
                            executor=executor, manager=manager)
    try:
        r = runner.run(seed=seed, max_steps=300)
    finally:
        runner.close()
    sr = r.recovery.final_skill_result
    return _make_episode("PickCube-v1", seed, "b1", True, injector.records,
                         r.recovery, sr, r.episode, executor.episode_reward)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--faults", nargs="*", default=["object_displacement"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.smoke:
        args.faults = ["object_displacement"]
        args.seeds = [0]

    b0_eps, b1_eps = [], []
    details = []
    for fname in args.faults:
        ftype, fstep, fparams, fdur = FAULTS[fname]
        for seed in args.seeds:
            b0 = run_b0(seed, ftype, fstep, fparams, fdur)
            b1 = run_b1(seed, ftype, fstep, fparams, fdur)
            b0_eps.append(b0)
            b1_eps.append(b1)
            details.append({
                "fault": fname, "seed": seed,
                "b0": {"success": b0.success, "steps": b0.steps,
                       "status": b0.skill_results[0].status.value},
                "b1": {"success": b1.success, "steps": b1.steps,
                       "status": b1.skill_results[0].status.value,
                       "attempts": b1.metrics.get("recovery", {}).get("attempts"),
                       "recovered": b1.metrics.get("recovery", {}).get("recovered"),
                       "aborted": b1.metrics.get("recovery", {}).get("aborted")},
            })

    require_sim(b0_eps + b1_eps, context="M3-C")  # 全 sim，不应抛
    m_b0 = compute_recovery_metrics(b0_eps)
    m_b1 = compute_recovery_metrics(b1_eps)
    comp = compare_b0_b1(b0_eps, b1_eps)

    report = {
        "faults": args.faults,
        "seeds": args.seeds,
        "details": details,
        "b0": m_b0.to_dict(),
        "b1": m_b1.to_dict(),
        "comparison": comp.to_dict(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n[report saved] {args.out}")


if __name__ == "__main__":
    main()
