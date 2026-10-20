"""M4-A 数据采集管线: 采集 PickCube 成功/失败轨迹 (B0 式故障注入、无恢复)。

产出 interfaces.learning (M4 学习契约) 兼容的数据集:
- 每帧提取 22 维状态特征 (extract_state_features, 只读 obs.policy, 绝不读 obs.truth)。
- 失败标签由完整 episode 离线生成 (label_failure, 失败 episode 最后 H 步内帧标 1)。
- 元数据 (seed/fault_type/fault_params/outcome/injection 信息) 与模型输入特征
  (frames[].features) 严格分离。

场景:
- normal              无故障 (oracle 基线, 期望成功)
- object_displacement 一次性位移 (多种 delta x 多种注入时机)
- grasp_perturb       抓取阶段关节扰动 (多种 scale x 多种时机)
- object_impulse      一次性线速度冲量 (多种 speed x 多种时机)

用法:
    source scripts/activate_env.sh
    export CUDA_VISIBLE_DEVICES=2       # 不能为空字符串 (SAPIEN 段错误)
    python -m sim.collect_dataset --pilot            # ~20 episode pilot
    python -m sim.collect_dataset --full             # ~150 episode 全量
    python -m sim.collect_dataset --verify-only      # 只校验已保存数据集

输出 (data/m4/, data/ 已被 .gitignore 排除; 仅 manifest.json 例外提交):
    dataset.npz        特征矩阵 + 标签 + episode 索引 (训练友好; 原始数据不提交)
    episodes_meta.json 每 episode 完整元数据 + 注入核实记录 (原始数据不提交)
    manifest.json      配置 + 种子 + 每场景 episode 数 + 类别分布 (提交)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

# 兼容 `python sim/collect_dataset.py` 与 `python -m sim.collect_dataset` 两种运行方式
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from interfaces import (  # noqa: E402
    DEFAULT_PREDICTION_HORIZON,
    EpisodeRecord,
    FrameRecord,
    SimulatorAdapter,
    STATE_FEATURE_DIM,
    STATE_FEATURE_NAMES,
    extract_state_features,
    label_failure,
)
from sim import PickCubeAdapter, PickCubeSkill  # noqa: E402
from sim.fault_injection import (  # noqa: E402
    FAULT_GRASP_PERTURB,
    FAULT_OBJECT_DISPLACEMENT,
    FAULT_OBJECT_IMPULSE,
    FaultInjector,
)

DEFAULT_OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "m4")

# 退化 episode 判定: 正常 pick 需 >=47 步 (规划失败=0 步; cube 恰好生成在 goal 处=1 步平凡成功)。
# 低于此步数的 episode 视为退化, 换种子确定性重试 (seed+1000*attempt)。
MIN_VALID_STEPS = 10
MAX_SEED_RETRIES = 5



# --------------------------------------------------------------------------- #
# 故障注入适配器 (对齐 scripts/verify_m3c.py 的 FaultInjectingAdapter 模式)
# --------------------------------------------------------------------------- #
class FaultInjectingAdapter(SimulatorAdapter):
    """每步后触发 FaultInjector 预定故障的适配器包装。"""

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


# --------------------------------------------------------------------------- #
# 确定性调度构建 (configs x seeds; seed 决定 FaultInjector rng 复现)
# --------------------------------------------------------------------------- #
def _expand(scenario: str, fault_type: Optional[str], configs: List[Tuple[dict, int]],
            seeds: List[int], duration: int = 1) -> List[dict]:
    """configs: [(fault_params, fault_step), ...]; 每个 seed 按 seed % len(configs) 取配置。"""
    specs = []
    for seed in seeds:
        params, step = configs[seed % len(configs)]
        specs.append(dict(
            scenario=scenario, fault_type=fault_type,
            fault_params=dict(params), fault_step=step, duration=duration, seed=seed,
        ))
    return specs


def build_pilot_schedule() -> List[dict]:
    """~20 episode pilot: 正常 + 各故障 (多幅度/时机), 用于验证数据正确性。"""
    specs: List[dict] = []
    specs += [dict(scenario="normal", fault_type=None, fault_params={},
                   fault_step=None, duration=1, seed=s) for s in range(4)]
    specs += _expand("object_displacement", FAULT_OBJECT_DISPLACEMENT, [
        ({"delta": [0.06, 0.0, 0.0]}, 12),
        ({"delta": [0.10, 0.0, 0.0]}, 12),
        ({"delta": [0.15, 0.0, 0.0]}, 12),
        ({"delta": [0.10, 0.0, 0.0]}, 20),
        ({"delta": [0.15, 0.0, 0.0]}, 25),
        ({"delta": [0.10, 0.0, 0.0]}, 45),
    ], list(range(6)))
    specs += _expand("grasp_perturb", FAULT_GRASP_PERTURB, [
        ({"scale": 0.20, "joints": [0, 1, 3]}, 28),
        ({"scale": 0.30, "joints": [0, 1, 3]}, 28),
        ({"scale": 0.40, "joints": [0, 1, 3]}, 28),
        ({"scale": 0.30, "joints": [0, 1, 3]}, 34),
        ({"scale": 0.40, "joints": [0, 1, 3]}, 38),
        ({"scale": 0.30, "joints": [0, 1, 3]}, 25),
    ], list(range(6)))
    specs += _expand("object_impulse", FAULT_OBJECT_IMPULSE, [
        ({"velocity": [0.5, 0.0, 0.0]}, 15),
        ({"velocity": [0.8, 0.0, 0.0]}, 15),
        ({"velocity": [1.0, 0.0, 0.0]}, 15),
        ({"velocity": [0.8, 0.0, 0.0]}, 25),
    ], list(range(4)))
    return specs


def build_full_schedule() -> List[dict]:
    """~150 episode 全量: 60 normal + 30 disp + 30 impulse + 30 grasp_perturb。

    平衡策略: normal 全部成功; object_displacement / object_impulse 可靠失败;
    grasp_perturb 混合 (扰动部分被吸收 -> 成功, 部分抓空 -> 失败), 整体趋向各半。
    """
    specs: List[dict] = []
    specs += [dict(scenario="normal", fault_type=None, fault_params={},
                   fault_step=None, duration=1, seed=s) for s in range(60)]
    specs += _expand("object_displacement", FAULT_OBJECT_DISPLACEMENT, [
        ({"delta": [0.06, 0.0, 0.0]}, 12),
        ({"delta": [0.10, 0.0, 0.0]}, 12),
        ({"delta": [0.15, 0.0, 0.0]}, 12),
        ({"delta": [0.10, 0.0, 0.0]}, 20),
        ({"delta": [0.15, 0.0, 0.0]}, 25),
        ({"delta": [0.10, 0.0, 0.0]}, 45),
    ], list(range(30)))
    specs += _expand("object_impulse", FAULT_OBJECT_IMPULSE, [
        ({"velocity": [0.5, 0.0, 0.0]}, 15),
        ({"velocity": [0.8, 0.0, 0.0]}, 15),
        ({"velocity": [1.0, 0.0, 0.0]}, 15),
        ({"velocity": [0.8, 0.0, 0.0]}, 25),
        ({"velocity": [1.0, 0.0, 0.0]}, 30),
        ({"velocity": [0.6, 0.0, 0.0]}, 20),
    ], list(range(30)))
    specs += _expand("grasp_perturb", FAULT_GRASP_PERTURB, [
        ({"scale": 0.20, "joints": [0, 1, 3]}, 28),
        ({"scale": 0.30, "joints": [0, 1, 3]}, 28),
        ({"scale": 0.40, "joints": [0, 1, 3]}, 28),
        ({"scale": 0.30, "joints": [0, 1, 3]}, 34),
        ({"scale": 0.40, "joints": [0, 1, 3]}, 38),
        ({"scale": 0.30, "joints": [0, 1, 3]}, 25),
    ], list(range(30)))
    return specs


# --------------------------------------------------------------------------- #
# 单 episode 采集
# --------------------------------------------------------------------------- #
def run_one_episode(adapter: PickCubeAdapter, spec: dict, horizon: int,
                    max_episode_steps: int, episode_id: str) -> Tuple[EpisodeRecord, List]:
    """跑一个 B0 式 episode (故障注入、无恢复)，返回 (EpisodeRecord, injector.records)。"""
    seed = spec["seed"]
    fault_type = spec["fault_type"]
    fault_params = spec["fault_params"]
    fault_step = spec["fault_step"]
    duration = spec.get("duration", 1)

    injector = FaultInjector(adapter, seed=seed)
    if fault_type is not None:
        injector.schedule(fault_type, fault_step, params=fault_params, duration=duration)
    fadapter = FaultInjectingAdapter(adapter, injector)

    obs = fadapter.reset(seed=seed)
    skill = PickCubeSkill(fadapter)
    skill.reset(obs)

    # 每帧提取状态特征 (只读 obs.policy); 标签后离线统一标注
    frames: List[FrameRecord] = [
        FrameRecord(step=int(obs.step), features=extract_state_features(obs), label=0)
    ]

    last_obs = obs
    terminated = False
    truncated = False
    while len(frames) - 1 < max_episode_steps:
        if skill.is_done:
            break
        action = skill.act(obs)
        res = fadapter.step(action)
        obs = res.observation
        frames.append(FrameRecord(
            step=int(obs.step),
            features=extract_state_features(obs),
            label=0,
        ))
        last_obs = obs
        terminated = bool(res.episode.terminated)
        truncated = bool(res.episode.truncated)
        if terminated or truncated:
            break

    # 离线标注: 任务成功与否用特权 truth (oracle 基线可用); t_end = 最后帧 step
    success = bool(last_obs.truth.success)
    steps_list = [f.step for f in frames]
    t_end = int(last_obs.step)
    labels = label_failure(steps_list, failed=(not success), t_end=t_end, horizon=horizon)
    for f, lab in zip(frames, labels):
        f.label = int(lab)

    # 故障参数元数据追加注入时机/持续步数 (仍属 fault 描述, 不进 frames[].features)
    meta_params = dict(fault_params)
    if fault_type is not None:
        meta_params["injection_step"] = int(fault_step)
        meta_params["duration"] = int(duration)

    ep = EpisodeRecord(
        episode_id=episode_id,
        seed=seed,
        fault_type=fault_type,
        fault_params=meta_params,
        outcome=success,
        frames=frames,
    )
    return ep, injector.records


# --------------------------------------------------------------------------- #
# 序列化
# --------------------------------------------------------------------------- #
def _fault_record_to_dict(rec) -> dict:
    return {
        "fault_type": rec.fault_type,
        "injected": bool(rec.injected),
        "step": int(rec.step),
        "params": rec.params,
        "detail": rec.detail,
    }


def _episode_meta_dict(ep: EpisodeRecord, records: List, frame_start: int,
                       skill_status: str, n_steps: int) -> dict:
    return {
        "episode_id": ep.episode_id,
        "seed": ep.seed,
        "fault_type": ep.fault_type,
        "fault_params": ep.fault_params,
        "outcome": bool(ep.outcome),
        "num_frames": len(ep.frames),
        "n_steps": int(n_steps),
        "t_end": int(ep.frames[-1].step) if ep.frames else 0,
        "frame_start": int(frame_start),
        "frame_end": int(frame_start + len(ep.frames)),  # 排他上界
        "n_positive_frames": int(sum(f.label for f in ep.frames)),
        "skill_status": skill_status,
        "injections": [_fault_record_to_dict(r) for r in records],
    }


def save_dataset(episodes: List[EpisodeRecord], injections: List[List], skill_statuses: List[str],
                 n_steps_list: List[int], config: dict, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)

    # 数值数组 (模型输入特征 + 标签 + episode 索引)
    features = np.concatenate([np.stack([f.features for f in ep.frames]) for ep in episodes], axis=0)
    labels = np.concatenate([np.array([f.label for f in ep.frames], dtype=np.int64) for ep in episodes])
    ep_index = np.concatenate([np.full(len(ep.frames), i, dtype=np.int64) for i, ep in enumerate(episodes)])
    npz_path = os.path.join(out_dir, "dataset.npz")
    np.savez_compressed(npz_path, features=features, labels=labels, episode_index=ep_index)

    # 每 episode 完整元数据 (含注入核实记录)
    metas = []
    frame_start = 0
    for i, (ep, recs) in enumerate(zip(episodes, injections)):
        metas.append(_episode_meta_dict(ep, recs, frame_start, skill_statuses[i], n_steps_list[i]))
        frame_start += len(ep.frames)
    meta_path = os.path.join(out_dir, "episodes_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metas, f, indent=2, ensure_ascii=False, default=str)

    # 类别分布 + 场景汇总
    n_success = sum(1 for ep in episodes if ep.outcome)
    n_failure = len(episodes) - n_success
    n_positive = int(labels.sum())

    scenario_groups: Dict[str, dict] = {}
    for i, ep in enumerate(episodes):
        g = scenario_groups.setdefault(config["schedule"][i]["scenario"], {
            "num_episodes": 0, "seeds": [], "success_count": 0, "failure_count": 0,
            "injection_attempted": 0, "injection_success": 0,
            "param_variants": [],
        })
        g["num_episodes"] += 1
        g["seeds"].append(ep.seed)
        if ep.outcome:
            g["success_count"] += 1
        else:
            g["failure_count"] += 1
        recs = injections[i]
        if ep.fault_type is not None:
            g["injection_attempted"] += len(recs)
            g["injection_success"] += sum(1 for r in recs if r.injected)
        variant = {"fault_params": ep.fault_params, "fault_type": ep.fault_type}
        if variant not in g["param_variants"]:
            g["param_variants"].append(variant)
    for g in scenario_groups.values():
        g["seeds"].sort()

    manifest = {
        "dataset": "m4_pickcube_failure_prediction",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "mode": config["mode"],
            "horizon": config["horizon"],
            "max_episode_steps": config["max_episode_steps"],
            "feature_dim": STATE_FEATURE_DIM,
            "feature_names": STATE_FEATURE_NAMES,
            "num_episodes": len(episodes),
            "num_frames": int(len(features)),
        },
        "schedule": config["schedule"],  # 完整复现配置 (scenario/fault_type/params/step/seed)
        "scenarios": scenario_groups,
        "class_distribution": {
            "success_episodes": int(n_success),
            "failure_episodes": int(n_failure),
            "success_rate": float(n_success / len(episodes)) if episodes else 0.0,
            "positive_frame_fraction": float(n_positive / len(labels)) if len(labels) else 0.0,
        },
        "files": {
            "features": "dataset.npz",
            "episodes_meta": "episodes_meta.json",
        },
    }
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

    return manifest


# --------------------------------------------------------------------------- #
# 校验 (加载回读, 断言数据正确性)
# --------------------------------------------------------------------------- #
def verify_saved(out_dir: str, horizon: int) -> dict:
    npz_path = os.path.join(out_dir, "dataset.npz")
    meta_path = os.path.join(out_dir, "episodes_meta.json")
    manifest_path = os.path.join(out_dir, "manifest.json")

    data = np.load(npz_path)
    features, labels, ep_index = data["features"], data["labels"], data["episode_index"]
    with open(meta_path, "r", encoding="utf-8") as f:
        metas = json.load(f)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    checks: Dict[str, bool] = {}
    # 1. 特征形状/dtype/有限性
    checks["feature_shape_(N,22)"] = features.ndim == 2 and features.shape[1] == STATE_FEATURE_DIM
    checks["feature_dtype_float32"] = features.dtype == np.float32
    checks["features_finite"] = bool(np.all(np.isfinite(features)))
    # 2. 标签 ∈ {0,1} 且 int
    checks["labels_int"] = np.issubdtype(labels.dtype, np.integer)
    checks["labels_in_0_1"] = bool(np.all((labels == 0) | (labels == 1)))
    # 3. episode 索引连续覆盖 0..E-1
    n_eps = len(metas)
    checks["episode_index_contiguous"] = bool(np.array_equal(
        np.unique(ep_index), np.arange(n_eps, dtype=np.int64)))
    checks["num_rows_match"] = len(features) == len(labels) == len(ep_index)
    # 4. 每 episode 帧边界一致 + 成功全 0 / 失败存在正帧且正帧落在最后 horizon 窗口
    boundary_ok, label_consistency_ok, leak_ok = True, True, True
    frame_start = 0
    for m in metas:
        n = m["num_frames"]
        if m["frame_start"] != frame_start or m["frame_end"] != frame_start + n:
            boundary_ok = False
        ep_labels = labels[frame_start:frame_start + n]
        if m["outcome"]:
            if int(ep_labels.sum()) != 0:
                label_consistency_ok = False
        else:
            if int(ep_labels.sum()) == 0:
                label_consistency_ok = False  # 失败 episode 应至少有正帧
            # 正帧必须落在 t_end - horizon 窗口内
            pos_idx = np.where(ep_labels == 1)[0]
            for pi in pos_idx:
                if (m["t_end"] - pi) > horizon:
                    label_consistency_ok = False
        # 5. 泄漏防护: 特征维度固定 22, 不随元数据变化 (schema 层面由 extract_state_features 保证)
        if n == 0 or ep_labels.shape[0] != n:
            leak_ok = False
        frame_start += n
    checks["frame_boundaries_consistent"] = boundary_ok
    checks["label_consistency"] = label_consistency_ok
    # 6. 注入成功率 (故障 episode 至少一次注入且全部核实注入成功)
    injection_attempted = 0
    injection_success = 0
    for m in metas:
        for r in m["injections"]:
            injection_attempted += 1
            if r["injected"]:
                injection_success += 1
    checks["injection_success_rate"] = (injection_attempted == 0) or (injection_success == injection_attempted)

    n_steps = [m["n_steps"] for m in metas]
    report = {
        "checks": checks,
        "num_episodes": n_eps,
        "num_frames": int(len(features)),
        "feature_shape": list(features.shape),
        "positive_frame_fraction": float(labels.sum() / len(labels)) if len(labels) else 0.0,
        "success_episodes": sum(1 for m in metas if m["outcome"]),
        "failure_episodes": sum(1 for m in metas if not m["outcome"]),
        "min_n_steps": int(min(n_steps)) if n_steps else 0,
        "max_n_steps": int(max(n_steps)) if n_steps else 0,
        "degenerate_episodes": sum(1 for s in n_steps if s < MIN_VALID_STEPS),
        "injection_attempted": injection_attempted,
        "injection_success": injection_success,
        "manifest_num_episodes": manifest["config"]["num_episodes"],
    }
    return report


# --------------------------------------------------------------------------- #
# 采集主流程
# --------------------------------------------------------------------------- #
def collect(schedule: List[dict], mode: str, horizon: int, max_episode_steps: int,
            out_dir: str, min_valid_steps: int = MIN_VALID_STEPS,
            max_retries: int = MAX_SEED_RETRIES) -> dict:
    episodes: List[EpisodeRecord] = []
    injections: List[List] = []
    skill_statuses: List[str] = []
    n_steps_list: List[int] = []

    adapter = PickCubeAdapter(max_episode_steps=max_episode_steps)
    n_retries = 0
    try:
        for i, spec in enumerate(schedule):
            spec.setdefault("seed_requested", spec["seed"])
            episode_id = f"{spec['scenario']}-{i:04d}"
            ep = None
            records: List = []
            n_steps = 0
            # 退化 episode (规划失败 / cube 生成在 goal 处的平凡成功) -> 换种子确定性重试
            for attempt in range(max_retries + 1):
                spec["seed"] = spec["seed_requested"] + 1000 * attempt
                ep, records = run_one_episode(adapter, spec, horizon, max_episode_steps, episode_id)
                n_steps = len(ep.frames) - 1
                if n_steps >= min_valid_steps:
                    break
                n_retries += 1
                print(f"    [retry] {episode_id} seed={spec['seed_requested']} "
                      f"attempt={attempt + 1} steps={n_steps} 退化 -> 换种子 "
                      f"{spec['seed_requested'] + 1000 * (attempt + 1)}", flush=True)
            n_steps_list.append(n_steps)
            skill_statuses.append("success" if ep.outcome else "failure")
            episodes.append(ep)
            injections.append(records)
            inj_flag = "injected" if (records and records[0].injected) else ("no-fault" if not records else "NOT-injected")
            print(f"[{i:3d}/{len(schedule)}] {episode_id:24s} seed={ep.seed:2d} "
                  f"outcome={'OK' if ep.outcome else 'FAIL':5s} frames={len(ep.frames):3d} "
                  f"steps={n_steps:3d} fault={str(ep.fault_type):18s} inj={inj_flag}",
                  flush=True)
        print(f"[collect] 退化 episode 重试 {n_retries} 次 (min_valid_steps={min_valid_steps})", flush=True)
    finally:
        adapter.close()

    config = {
        "mode": mode,
        "horizon": horizon,
        "max_episode_steps": max_episode_steps,
        "schedule": schedule,
    }
    manifest = save_dataset(episodes, injections, skill_statuses, n_steps_list, config, out_dir)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="~20 episode pilot")
    ap.add_argument("--full", action="store_true", help="~150 episode 全量")
    ap.add_argument("--verify-only", action="store_true", help="只校验已保存数据集")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--horizon", type=int, default=DEFAULT_PREDICTION_HORIZON)
    ap.add_argument("--max-episode-steps", type=int, default=200)
    args = ap.parse_args()

    if args.verify_only:
        report = verify_saved(args.out_dir, args.horizon)
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        return

    mode = "full" if args.full else "pilot"
    schedule = build_full_schedule() if args.full else build_pilot_schedule()
    print(f"=== M4-A 数据采集 ({mode}) ===  共 {len(schedule)} episode, horizon={args.horizon}, "
          f"max_steps={args.max_episode_steps}", flush=True)
    manifest = collect(schedule, mode, args.horizon, args.max_episode_steps, args.out_dir)

    print("\n=== 保存后校验 ===")
    report = verify_saved(args.out_dir, args.horizon)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))

    print("\n=== 类别分布 ===")
    cd = manifest["class_distribution"]
    print(f"成功 {cd['success_episodes']} / 失败 {cd['failure_episodes']}  "
          f"(成功率 {cd['success_rate']:.2%}); 正帧占比 {cd['positive_frame_fraction']:.2%}")
    print(f"manifest: {os.path.join(args.out_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
