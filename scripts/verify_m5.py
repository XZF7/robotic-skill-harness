"""M5 扩展评估基准 (Agent D): B0/B1/B2 在 held-out 种子 100-129 x 3 故障上 matched 对比。

用法:
  python scripts/verify_m5.py --tag pilot --seeds 100 101 102 \
      --faults object_displacement grasp_perturb object_displacement_large
  python scripts/verify_m5.py --tag primary --seeds $(seq 100 129) \
      --faults object_displacement grasp_perturb object_displacement_large

运行说明 (重要):
- 本脚本支持断点续跑: 每完成一个 episode 即追加写入 logs/m5_<tag>_episodes.jsonl,
  重跑同 tag 会自动跳过已完成 (fault, seed, method) 单元。
- ManiSkill/mplib 在单个进程内连续运行 ~90 episode 后会触发 pybind11 GIL 崩溃
  (segfault, rc=139), 并可能静默污染崩溃前最后几个 episode (planning 失败 → step=0)。
  因此 30 种子 (270 episode) 必须分块运行, 每块一个全新 python 进程 (推荐 5 种子/块):
    for start in 100 105 110 115 120 125; do
      python scripts/verify_m5.py --tag primary --seeds $(seq $start $((start+4))) \
        --faults object_displacement grasp_perturb object_displacement_large
    done
"""
import argparse
import json
import math
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from interfaces import EpisodeResult, SimulatorAdapter
from sim import PickCubeAdapter, PickCubeSkill
from sim.fault_injection import (
    FaultInjector, FAULT_OBJECT_DISPLACEMENT, FAULT_GRASP_PERTURB,
)
from runtime import TaskRunner, RecoveryRunner, SkillExecutor, FROZEN_B2T_GATE
from runtime.recovery import RecoveryManager
from learning.model import MLP, MLPFailurePredictor
from eval import (
    attach_fault, attach_recovery, attach_privileged, attach_baseline,
    compute_recovery_metrics, require_sim,
    extract_fault, failure_manifested, is_valid_injected_failure,
    recovery_attempted,
)

FAULTS = {
    "object_displacement": (FAULT_OBJECT_DISPLACEMENT, 10, {"delta": [0.07, 0.0, 0.0]}, 1),
    "grasp_perturb": (FAULT_GRASP_PERTURB, 25, {"scale": 0.25}, 1),
    "object_displacement_large": (FAULT_OBJECT_DISPLACEMENT, 10, {"delta": [0.5, 0.0, 0.0]}, 1),
}

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_DEFAULT_CKPTS = [
    os.path.join(_REPO, "data", "m4", "checkpoints", "model.pt"),
]

PREDICTOR_THRESHOLD = 0.5  # 冻结协议: B2 threshold=0.5


# --------------------------------------------------------------------------- #
# 统计工具
# --------------------------------------------------------------------------- #
def wilson_ci(k, n, z=1.959963984540054):
    """Wilson score 95% CI for success rate. 返回 (p, lo, hi)。n=0 -> (None,None,None)。"""
    if n <= 0:
        return (None, None, None)
    k = float(k)
    n = float(n)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def paired_bootstrap(diffs, n_boot=1000, rng=None):
    """按配对单元重采样 (with replacement) 的均值分布 + 95% CI (2.5/97.5 分位)。

    diffs: list[float] 每个 (fault,seed) 单元的方法间差值。
    返回 dict(mean, std, ci_lo, ci_hi, distribution(list), n_units, n_boot)。
    """
    diffs = [float(d) for d in diffs]
    n = len(diffs)
    rng = rng or np.random.default_rng(0)
    if n == 0:
        return {"mean": None, "std": None, "ci_lo": None, "ci_hi": None,
                "distribution": [], "n_units": 0, "n_boot": n_boot}
    arr = np.asarray(diffs, dtype=np.float64)
    means = np.empty(n_boot, dtype=np.float64)
    idx = np.arange(n)
    for b in range(n_boot):
        sample = arr[rng.integers(0, n, size=n)]
        means[b] = sample.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if n > 1 else 0.0,
        "ci_lo": float(lo), "ci_hi": float(hi),
        "distribution": [float(x) for x in means],
        "n_units": n, "n_boot": n_boot,
    }


# --------------------------------------------------------------------------- #
# 预测器加载
# --------------------------------------------------------------------------- #
def resolve_ckpt(path=None):
    if path:
        return path
    for p in _DEFAULT_CKPTS:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"未找到 checkpoint, 已尝试 {_DEFAULT_CKPTS!r}; 用 --ckpt 显式指定"
    )


def load_predictor(path=None, threshold=PREDICTOR_THRESHOLD):
    path = resolve_ckpt(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        ckpt = torch.load(path, map_location="cpu")
    model = MLP(input_dim=ckpt["input_dim"], hidden_dims=ckpt["hidden_dims"],
                dropout=ckpt["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return MLPFailurePredictor(model, mean=ckpt["mean"], std=ckpt["std"],
                               threshold=threshold)


# --------------------------------------------------------------------------- #
# 适配器/执行器装饰
# --------------------------------------------------------------------------- #
class FaultInjectingAdapter(SimulatorAdapter):
    """包装 adapter: 每步调用 injector.after_step, 并记录 max_step_seen (eligibility 判定)。"""

    def __init__(self, inner, injector):
        self._inner = inner
        self._injector = injector
        self.max_step_seen = 0

    def reset(self, *, seed=None, options=None):
        obs = self._inner.reset(seed=seed, options=options)
        self.max_step_seen = max(self.max_step_seen, int(getattr(obs, "step", 0)))
        return obs

    def step(self, action):
        res = self._inner.step(action)
        self.max_step_seen = max(self.max_step_seen, int(getattr(res.observation, "step", 0)))
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
    """跨 attempt 累计 reward (RecoveryRunner 多次 execute 会各自 reset total_reward)。"""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.episode_reward = 0.0

    def execute(self, *a, **k):
        r = super().execute(*a, **k)
        self.episode_reward += self.total_reward
        return r


# --------------------------------------------------------------------------- #
# 运行 B0/B1/B2
# --------------------------------------------------------------------------- #
def _make_episode(seed, baseline, fault_records, recovery, skill_result,
                  episode, total_reward):
    ep = EpisodeResult(
        task_id="PickCube-v1",
        success=bool(episode.success) if episode is not None else False,
        steps=skill_result.steps,
        reward=float(total_reward),
        skill_results=[skill_result],
        failures=[skill_result.failure] if skill_result.failure is not None else [],
        metrics={},
        source="sim",
    )
    if fault_records:
        ep = attach_fault(ep, fault_records[0])
    if recovery is not None:
        ep = attach_recovery(ep, recovery)
    ep = attach_privileged(ep, True)
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
    ep = _make_episode(seed, "b0", injector.records, None, r.skill_result,
                       r.episode, r.total_reward)
    return ep, {"max_step_seen": fadapter.max_step_seen}


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
    first_fail_step = None
    for _sr in r.attempt_results:
        if _sr.failure is not None:
            first_fail_step = int(_sr.failure.step)
            break
    ep = _make_episode(seed, "b1", injector.records, r.recovery, sr,
                       r.episode, executor.episode_reward)
    return ep, {"max_step_seen": fadapter.max_step_seen,
                "first_failure_step": first_fail_step}


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
    first_fail_step = None
    for _sr in r.attempt_results:
        if _sr.failure is not None:
            first_fail_step = int(_sr.failure.step)
            break
    ep = _make_episode(seed, "b2", injector.records, r.recovery, sr,
                       r.episode, executor.episode_reward)
    records = runner.guard.records if runner.guard else []
    triggers = [rec for rec in records if rec.triggered]
    meta = {
        "max_step_seen": fadapter.max_step_seen,
        "num_predictions": len(records),
        "num_triggers": len(triggers),
        "first_trigger_step": int(triggers[0].step) if triggers else None,
        "first_failure_step": first_fail_step,
    }
    return ep, meta


def run_b2t(seed, ftype, fstep, fparams, fdur, predictor, threshold):
    adapter, injector, fadapter = _mk(seed, ftype, fstep, fparams, fdur)
    skill = PickCubeSkill(fadapter)
    executor = AccumulatingExecutor()
    manager = RecoveryManager(max_retries=2)
    runner = RecoveryRunner(fadapter, skill, task_id="PickCube-v1",
                            executor=executor, manager=manager, recovery_mode="b2t",
                            predictor=predictor, predictor_threshold=threshold,
                            temporal_gate=FROZEN_B2T_GATE)
    try:
        r = runner.run(seed=seed, max_steps=300)
    finally:
        runner.close()
    sr = r.recovery.final_skill_result
    first_fail_step = None
    for _sr in r.attempt_results:
        if _sr.failure is not None:
            first_fail_step = int(_sr.failure.step)
            break
    ep = _make_episode(seed, "b2t", injector.records, r.recovery, sr,
                       r.episode, executor.episode_reward)
    records = runner.guard.records if runner.guard else []
    triggers = [rec for rec in records if rec.triggered]
    meta = {
        "max_step_seen": fadapter.max_step_seen,
        "num_predictions": len(records),
        "num_triggers": len(triggers),
        "first_trigger_step": int(triggers[0].step) if triggers else None,
        "first_failure_step": first_fail_step,
    }
    return ep, meta


# --------------------------------------------------------------------------- #
# episode 记录 (每 episode 一行)
# --------------------------------------------------------------------------- #
def _first_failure_step(ep):
    if ep.failures:
        return min(int(f.step) for f in ep.failures)
    return None


def _recovery_dict(ep):
    from eval.fault_injection import _normalize_recovery
    return _normalize_recovery(ep) or {}


def build_record(seed, fname, ftype, fstep, method, ep, meta, wall_clock):
    f = extract_fault(ep)
    rec = _recovery_dict(ep)
    detected_step = meta.get("first_failure_step")
    if detected_step is None:
        detected_step = _first_failure_step(ep)
    latency = None
    if detected_step is not None and f is not None and f.step >= 0:
        latency = detected_step - f.step
    attempts = int(rec.get("attempts", 1))
    return {
        "seed": int(seed),
        "fault": fname,
        "fault_type": ftype,
        "fault_step": int(fstep),
        "method": method,
        "success": bool(ep.success),
        "steps": int(ep.steps),
        "reward": float(ep.reward),
        "wall_clock": float(wall_clock),
        "eligible": bool(meta["max_step_seen"] >= fstep),
        "attempted": f is not None,
        "verified": bool(f.injected) if f is not None else False,
        "manifested": bool(failure_manifested(ep)),
        "injected": bool(f.injected) if f is not None else False,
        "recovery_attempted": bool(recovery_attempted(ep)),
        "recovery_recovered": bool(rec.get("recovered", False)),
        "recovery_aborted": bool(rec.get("aborted", False)),
        "attempts": attempts,
        "retries": max(0, attempts - 1),
        "num_predictions": int(meta.get("num_predictions", 0)),
        "num_triggers": int(meta.get("num_triggers", 0)),
        "first_trigger_step": meta.get("first_trigger_step"),
        "first_failure_step": detected_step,
        "fault_detection_latency": latency,
        "max_env_step": int(meta["max_step_seen"]),
    }


# --------------------------------------------------------------------------- #
# 每方法汇总
# --------------------------------------------------------------------------- #
def _pct(num, den):
    return round(100.0 * num / den, 4) if den else None


def method_summary(records, method):
    recs = [r for r in records if r["method"] == method]
    n = len(recs)
    num_success = sum(1 for r in recs if r["success"])
    num_eligible = sum(1 for r in recs if r["eligible"])
    num_attempted = sum(1 for r in recs if r["attempted"])
    num_verified = sum(1 for r in recs if r["verified"])
    num_manifested = sum(1 for r in recs if r["manifested"])
    num_inj_man = sum(1 for r in recs if r["verified"] and r["manifested"])
    num_recovered = sum(1 for r in recs if r["recovery_recovered"])
    num_aborted = sum(1 for r in recs if r["recovery_aborted"])
    num_rec_attempted = sum(1 for r in recs if r["recovery_attempted"])
    sum_retries = sum(r["retries"] for r in recs)
    sum_reward = sum(r["reward"] for r in recs)
    sum_steps = sum(r["steps"] for r in recs)
    sum_wall = sum(r["wall_clock"] for r in recs)
    lat = [r["fault_detection_latency"] for r in recs
           if r["fault_detection_latency"] is not None]

    p, lo, hi = wilson_ci(num_success, n)

    return {
        "method": method,
        "num_episodes": n,
        "task_success_rate": {
            "value": round(num_success / n, 6) if n else None,
            "pct": _pct(num_success, n),
            "numerator": num_success, "denominator": n,
            "wilson_ci_lo": round(lo, 6) if lo is not None else None,
            "wilson_ci_hi": round(hi, 6) if hi is not None else None,
        },
        "eligibility": {
            "eligible": {"numerator": num_eligible, "denominator": n, "pct": _pct(num_eligible, n)},
            "attempted": {"numerator": num_attempted, "denominator": n, "pct": _pct(num_attempted, n)},
            "verified": {"numerator": num_verified, "denominator": num_attempted, "pct": _pct(num_verified, num_attempted)},
            "manifested": {"numerator": num_manifested, "denominator": n, "pct": _pct(num_manifested, n)},
        },
        "verified_fault_injection_rate": {
            "value": round(num_verified / num_attempted, 6) if num_attempted else None,
            "pct": _pct(num_verified, num_attempted),
            "numerator": num_verified, "denominator": num_attempted,
        },
        "recovery_success_rate": {
            "value": round(num_recovered / num_inj_man, 6) if num_inj_man else None,
            "pct": _pct(num_recovered, num_inj_man),
            "numerator": num_recovered, "denominator": num_inj_man,
        },
        "avg_retries": {
            "value": round(sum_retries / num_rec_attempted, 6) if num_rec_attempted else None,
            "numerator": sum_retries, "denominator": num_rec_attempted,
        },
        "abort_rate": {
            "value": round(num_aborted / n, 6) if n else None,
            "pct": _pct(num_aborted, n),
            "numerator": num_aborted, "denominator": n,
        },
        "cumulative_reward": {
            "total": round(sum_reward, 6), "mean": round(sum_reward / n, 6) if n else None,
            "denominator": n,
        },
        "total_episode_steps": {
            "total": sum_steps, "mean": round(sum_steps / n, 4) if n else None,
            "denominator": n,
        },
        "wall_clock": {
            "total": round(sum_wall, 4), "mean": round(sum_wall / n, 4) if n else None,
            "denominator": n,
        },
        "fault_detection_latency": {
            "value": round(sum(lat) / len(lat), 4) if lat else None,
            "numerator": round(sum(lat), 4), "denominator": len(lat),
        },
    }


# --------------------------------------------------------------------------- #
# 配对分析 + bootstrap
# --------------------------------------------------------------------------- #
def eligible_subset_summary(records):
    """fault-eligible 子集 (eligible=True) 的每方法主指标 + 恢复成功率 (协议 §6)。"""
    out = {}
    for m in sorted({r["method"] for r in records}):
        recs = [r for r in records if r["method"] == m and r["eligible"]]
        n = len(recs)
        ns = sum(1 for r in recs if r["success"])
        p, lo, hi = wilson_ci(ns, n)
        num_rec = sum(1 for r in recs if r["recovery_recovered"])
        num_ijm = sum(1 for r in recs if r["verified"] and r["manifested"])
        out[m] = {
            "num_eligible_episodes": n,
            "task_success_rate": {
                "value": round(ns / n, 6) if n else None,
                "pct": _pct(ns, n),
                "numerator": ns, "denominator": n,
                "wilson_ci_lo": round(lo, 6) if lo is not None else None,
                "wilson_ci_hi": round(hi, 6) if hi is not None else None,
            },
            "recovery_success_rate": {
                "value": round(num_rec / num_ijm, 6) if num_ijm else None,
                "pct": _pct(num_rec, num_ijm),
                "numerator": num_rec, "denominator": num_ijm,
            },
        }
    return out


def paired_analysis(records, n_boot=1000, rng=None):
    """按 (fault, seed) 配对单元计算 B1-B0 / B2-B0 / B2-B1 的任务成功差异 + bootstrap。"""
    units = {}
    for r in records:
        key = (r["fault"], r["seed"])
        units.setdefault(key, {})[r["method"]] = r

    diffs_ba, diffs_ca, diffs_cb = [], [], []
    steps_ba, steps_ca = [], []
    reward_ba, reward_ca = [], []
    matched_units = []
    unmatched = []
    for key in sorted(units.keys()):
        u = units[key]
        if set(u) >= {"b0", "b1", "b2"}:
            matched_units.append(key)
            diffs_ba.append(int(u["b1"]["success"]) - int(u["b0"]["success"]))
            diffs_ca.append(int(u["b2"]["success"]) - int(u["b0"]["success"]))
            diffs_cb.append(int(u["b2"]["success"]) - int(u["b1"]["success"]))
            steps_ba.append(u["b1"]["steps"] - u["b0"]["steps"])
            steps_ca.append(u["b2"]["steps"] - u["b0"]["steps"])
            reward_ba.append(u["b1"]["reward"] - u["b0"]["reward"])
            reward_ca.append(u["b2"]["reward"] - u["b0"]["reward"])
        else:
            unmatched.append({"fault": key[0], "seed": key[1], "methods": sorted(u.keys())})

    def bd(d):
        return paired_bootstrap(d, n_boot=n_boot, rng=rng)

    return {
        "n_matched_units": len(matched_units),
        "n_unmatched_units": len(unmatched),
        "unmatched": unmatched,
        "paired_task_success_difference": {
            "b1_minus_b0": {
                "paired_mean": round(sum(diffs_ba) / len(diffs_ba), 6) if diffs_ba else None,
                "n_pairs": len(diffs_ba),
                "raw_diff_sum": sum(diffs_ba),
                "bootstrap": bd(diffs_ba),
            },
            "b2_minus_b0": {
                "paired_mean": round(sum(diffs_ca) / len(diffs_ca), 6) if diffs_ca else None,
                "n_pairs": len(diffs_ca),
                "raw_diff_sum": sum(diffs_ca),
                "bootstrap": bd(diffs_ca),
            },
            "b2_minus_b1": {
                "paired_mean": round(sum(diffs_cb) / len(diffs_cb), 6) if diffs_cb else None,
                "n_pairs": len(diffs_cb),
                "raw_diff_sum": sum(diffs_cb),
                "bootstrap": bd(diffs_cb),
            },
        },
        "paired_steps_difference": {
            "b1_minus_b0": bd(steps_ba),
            "b2_minus_b0": bd(steps_ca),
        },
        "paired_reward_difference": {
            "b1_minus_b0": bd(reward_ba),
            "b2_minus_b0": bd(reward_ca),
        },
    }


# --------------------------------------------------------------------------- #
# false recovery trigger (B2 专有) + early warning lead time
# --------------------------------------------------------------------------- #
def false_trigger_and_lead_time(records):
    """false_recovery_trigger_rate + early_warning_lead_time (B2 vs matched B1)。"""
    units = {}
    for r in records:
        units.setdefault((r["fault"], r["seed"]), {})[r["method"]] = r

    num_false_trigger = 0
    num_inj_man = 0
    leads = []
    lead_details = []
    for key, u in sorted(units.items()):
        if "b1" not in u or "b2" not in u:
            continue
        b1, b2 = u["b1"], u["b2"]
        # 分母: 注入且 manifested 的失败场景 (以 B1 为 ground-truth 参考)
        if b1["verified"] and b1["manifested"]:
            num_inj_man += 1
        # 分子: B2 预判触发 且 matched B1 首次尝试即成功 (无需恢复)
        if b2["num_triggers"] > 0 and b1["attempts"] == 1 and b1["success"]:
            num_false_trigger += 1
        # early warning lead time: B1 检测到失败的步 - B2 首次预判触发步
        if b2["first_trigger_step"] is not None and b1["first_failure_step"] is not None:
            lead = b1["first_failure_step"] - b2["first_trigger_step"]
            leads.append(lead)
            lead_details.append({"fault": key[0], "seed": key[1],
                                 "b1_first_failure_step": b1["first_failure_step"],
                                 "b2_first_trigger_step": b2["first_trigger_step"],
                                 "lead_time": lead})

    return {
        "false_recovery_trigger_rate": {
            "value": round(num_false_trigger / num_inj_man, 6) if num_inj_man else None,
            "pct": _pct(num_false_trigger, num_inj_man),
            "numerator": num_false_trigger, "denominator": num_inj_man,
            "note": "分子=B2 预判触发且 matched B1 首次尝试即成功; 分母=注入且 manifested 失败场景 (B1 参考)。冻结协议 §7。",
        },
        "early_warning_lead_time": {
            "value": round(sum(leads) / len(leads), 4) if leads else None,
            "numerator": round(sum(leads), 4), "denominator": len(leads),
            "n_pairs_with_trigger_and_failure": len(leads),
            "details": lead_details,
        },
    }


# --------------------------------------------------------------------------- #
# 图表
# --------------------------------------------------------------------------- #
def make_figure(report, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ["b0", "b1", "b2"]
    rates = [report["methods"][m]["task_success_rate"]["value"] for m in methods]
    lo = [report["methods"][m]["task_success_rate"]["wilson_ci_lo"] for m in methods]
    hi = [report["methods"][m]["task_success_rate"]["wilson_ci_hi"] for m in methods]
    err_lo = [r - l if (r is not None and l is not None) else 0 for r, l in zip(rates, lo)]
    err_hi = [h - r if (h is not None and r is not None) else 0 for h, r in zip(hi, rates)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(methods))

    ax = axes[0]
    ax.bar(x, [r if r is not None else 0 for r in rates], yerr=[err_lo, err_hi],
           capsize=5, color=["#8a8a8a", "#4c86c6", "#c65f4c"], alpha=0.9)
    for i, r in enumerate(rates):
        ax.text(i, r + 0.02 if r is not None else 0.02, f"{r:.3f}" if r is not None else "N/A",
                ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["B0", "B1", "B2"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("task success rate")
    ax.set_title("Task success rate (95% Wilson CI)")

    ax = axes[1]
    pdiff = report["paired"]["paired_task_success_difference"]
    colors = ["#4c86c6", "#c65f4c", "#6b5f9e"]
    labels = ["B1-B0", "B2-B0", "B2-B1"]
    for i, (key, color, lab) in enumerate(zip(
            ["b1_minus_b0", "b2_minus_b0", "b2_minus_b1"], colors, labels)):
        dist = pdiff[key]["bootstrap"]["distribution"]
        if dist:
            ax.hist(dist, bins=25, alpha=0.45, color=color, label=lab, density=True)
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("paired task-success difference")
    ax.set_ylabel("density (bootstrap)")
    ax.set_title("Paired diff bootstrap distribution (1000 resamples)")
    ax.legend()

    ax = axes[2]
    rec = [report["methods"][m]["recovery_success_rate"]["value"] for m in methods]
    ax.bar(x, [r if r is not None else 0 for r in rec], color=["#8a8a8a", "#4c86c6", "#c65f4c"],
           alpha=0.9)
    for i, r in enumerate(rec):
        ax.text(i, (r if r is not None else 0) + 0.02, f"{r:.3f}" if r is not None else "N/A",
                ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(["B0", "B1", "B2"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("recovery success rate")
    ax.set_title("Recovery success rate (on injected+manifested)")

    fig.suptitle(f"M5 benchmark: {report['tag']} "
                 f"({report['num_episodes']} episodes, {report['paired']['n_matched_units']} matched units)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def _episodes_jsonl(out_dir, tag):
    return os.path.join(out_dir, f"m5_{tag}_episodes.jsonl")


def _errors_jsonl(out_dir, tag):
    return os.path.join(out_dir, f"m5_{tag}_errors.jsonl")


def _load_jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    return out


def run_all(faults, seeds, methods, predictor, threshold, tag, out_dir, n_boot, make_figs):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(12345)
    t0 = time.time()

    # 断点续跑: 载入已完成的 episode (JSONL), 跳过 (fault, seed, method) 已完成单元。
    records = _load_jsonl(_episodes_jsonl(out_dir, tag))
    errors = _load_jsonl(_errors_jsonl(out_dir, tag))
    done_units = {(r["fault"], int(r["seed"]), r["method"]) for r in records}
    new_count = 0
    jl_path = _episodes_jsonl(out_dir, tag)
    err_path = _errors_jsonl(out_dir, tag)

    with open(jl_path, "a", encoding="utf-8") as jl_fh, \
         open(err_path, "a", encoding="utf-8") as err_fh:
        for fname in faults:
            ftype, fstep, fparams, fdur = FAULTS[fname]
            for seed in seeds:
                for method in methods:
                    if (fname, seed, method) in done_units:
                        continue
                    tic = time.perf_counter()
                    try:
                        if method == "b0":
                            ep, meta = run_b0(seed, ftype, fstep, fparams, fdur)
                        elif method == "b1":
                            ep, meta = run_b1(seed, ftype, fstep, fparams, fdur)
                        elif method == "b2":
                            ep, meta = run_b2(seed, ftype, fstep, fparams, fdur,
                                              predictor, threshold)
                        elif method == "b2t":
                            ep, meta = run_b2t(seed, ftype, fstep, fparams, fdur,
                                               predictor, threshold)
                        else:
                            raise ValueError(f"unknown method {method!r}")
                        wc = time.perf_counter() - tic
                        rec = build_record(seed, fname, ftype, fstep, method,
                                           ep, meta, wc)
                        records.append(rec)
                        jl_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        jl_fh.flush()
                        new_count += 1
                    except Exception as exc:  # noqa: BLE001 - 单 episode 失败不中断整体
                        wc = time.perf_counter() - tic
                        err = {"fault": fname, "seed": seed, "method": method,
                               "error": f"{type(exc).__name__}: {exc}"}
                        errors.append(err)
                        err_fh.write(json.dumps(err, ensure_ascii=False) + "\n")
                        err_fh.flush()
                        print(f"[ERROR] fault={fname} seed={seed} method={method}: "
                              f"{type(exc).__name__}: {exc}", file=sys.stderr)
                    print(f"[{time.time()-t0:7.1f}s] {fname} seed={seed} {method} done "
                          f"(wall {wc:.2f}s)", flush=True)

    methods_present = sorted({r["method"] for r in records})
    summaries = {m: method_summary(records, m) for m in methods_present}
    paired = paired_analysis(records, n_boot=n_boot, rng=rng)
    ft = false_trigger_and_lead_time(records)

    report = {
        "tag": tag,
        "num_episodes": len(records),
        "num_errors": len(errors),
        "errors": errors,
        "faults": faults,
        "seeds": seeds,
        "method_list": methods_present,
        "predictor_threshold": threshold,
        "wall_clock_total": round(time.time() - t0, 3),
        "methods": summaries,
        "eligible_subset": eligible_subset_summary(records),
        "paired": paired,
        "false_recovery_trigger_rate": ft["false_recovery_trigger_rate"],
        "early_warning_lead_time": ft["early_warning_lead_time"],
    }

    agg_json = os.path.join(out_dir, f"m5_{tag}.json")
    ep_json = os.path.join(out_dir, f"m5_{tag}_episodes.json")
    ep_csv = os.path.join(out_dir, f"m5_{tag}_episodes.csv")
    fig_png = os.path.join(out_dir, f"m5_{tag}_figure.png")

    with open(agg_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    with open(ep_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    if records:
        import csv
        cols = list(records[0].keys())
        with open(ep_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in records:
                w.writerow(r)

    if make_figs and records:
        make_figure(report, fig_png)
        report["figure"] = fig_png

    print("\n" + "=" * 70)
    print(f"M5 benchmark [{tag}]: {len(records)} episodes ({new_count} new), "
          f"{len(errors)} errors, wall {report['wall_clock_total']:.1f}s")
    for m in methods_present:
        s = summaries[m]
        tsr = s["task_success_rate"]
        print(f"  {m}: success {tsr['numerator']}/{tsr['denominator']} = "
              f"{tsr['pct']}%  (Wilson 95% [{tsr['wilson_ci_lo']}, {tsr['wilson_ci_hi']}])  "
              f"recovery {s['recovery_success_rate']['pct']}%  "
              f"abort {s['abort_rate']['pct']}%  avg_retries {s['avg_retries']['value']}")
    pd_ = paired["paired_task_success_difference"]
    for key, lab in [("b1_minus_b0", "B1-B0"), ("b2_minus_b0", "B2-B0"),
                     ("b2_minus_b1", "B2-B1")]:
        d = pd_[key]
        b = d["bootstrap"]
        print(f"  paired {lab}: mean {d['paired_mean']}  "
              f"bootstrap 95% CI [{b['ci_lo']}, {b['ci_hi']}]  (n_units={b['n_units']})")
    ftr = ft["false_recovery_trigger_rate"]
    print(f"  false_recovery_trigger_rate: {ftr['numerator']}/{ftr['denominator']} = {ftr['pct']}%")
    print("  fault-eligible subset (eligible=True):")
    es = report["eligible_subset"]
    for m in methods_present:
        tsr = es[m]["task_success_rate"]
        print(f"    {m}: success {tsr['numerator']}/{tsr['denominator']} = {tsr['pct']}%  "
              f"(Wilson 95% [{tsr['wilson_ci_lo']}, {tsr['wilson_ci_hi']}])")
    print(f"  outputs: {agg_json}, {ep_json}, {ep_csv}" + (f", {fig_png}" if make_figs and records else ""))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="1 fault x 1 seed x 3 methods")
    ap.add_argument("--tag", default="primary")
    ap.add_argument("--faults", nargs="*",
                    default=["object_displacement", "grasp_perturb", "object_displacement_large"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[100])
    ap.add_argument("--methods", nargs="*", default=["b0", "b1", "b2"])
    ap.add_argument("--threshold", type=float, default=PREDICTOR_THRESHOLD)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--bootstrap-n", type=int, default=1000)
    ap.add_argument("--out-dir", default=os.path.join(_REPO, "logs"))
    ap.add_argument("--no-figs", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.tag = "smoke"
        args.faults = ["object_displacement"]
        args.seeds = [100]
        args.methods = ["b0", "b1", "b2"]

    predictor = load_predictor(path=args.ckpt, threshold=args.threshold)
    run_all(args.faults, args.seeds, args.methods, predictor, args.threshold,
            args.tag, args.out_dir, args.bootstrap_n, not args.no_figs)


if __name__ == "__main__":
    main()
