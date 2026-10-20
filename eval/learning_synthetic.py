"""合成 EpisodeRecord 生成器 (M4-D 开发期测试用)。

生成确定性的 EpisodeRecord (含帧级 features/labels), 不依赖 Agent A 的真实数据。
特征只占 22 维 proprioception schema (qpos/qvel/gripper/tcp_pose), 标签用
interfaces.learning.label_failure 离线生成。

failure_mode 控制失败 episode 的可观测信号 (供规则式基线检测):
- "gripper_close": 失败 episode 末端 gripper 闭合到 0 (触发 GRIP_FAIL/DROP 启发式)。
- "qvel_spike"   : 失败 episode 末端 qvel 突增 (触发 COLLISION 启发式)。
- "none"         : 失败 episode 特征无异常 (规则式基线无法检测)。
"""
from __future__ import annotations

import random
from typing import List, Optional

import numpy as np

from interfaces import (
    FrameRecord, EpisodeRecord, label_failure, DEFAULT_PREDICTION_HORIZON,
    STATE_FEATURE_DIM,
)

_GRIP_IDX = 14
_QVEL_START, _QVEL_END = 7, 14


def _uniform_vec(rng: random.Random, lo: float, hi: float, n: int) -> np.ndarray:
    return np.array([rng.uniform(lo, hi) for _ in range(n)], dtype=np.float32)


def _base_features(seed: int, step: int, rng: random.Random) -> np.ndarray:
    """确定性基础特征: 22 维, 全部有限, 无异常信号。"""
    f = np.zeros(STATE_FEATURE_DIM, dtype=np.float32)
    rng.seed(seed * 100003 + step * 7919)  # 每帧确定性
    f[0:7] = np.linspace(0.0, 0.6, 7, dtype=np.float32) + 0.01 * _uniform_vec(rng, -1, 1, 7)
    f[7:14] = 0.05 * _uniform_vec(rng, -1, 1, 7)  # qvel 小, 无碰撞
    f[14] = 0.5  # gripper 中间开度
    f[15:18] = np.array([0.3, 0.0, 0.4], dtype=np.float32) + 0.01 * _uniform_vec(rng, -1, 1, 3)
    f[18:22] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # 单位四元数
    return f.astype(np.float32)


def _apply_failure_signal(
    f: np.ndarray,
    step: int,
    t_end: int,
    horizon: int,
    mode: str,
) -> np.ndarray:
    """在失败 episode 的末端 (最后 horizon 步) 注入可观测异常信号。"""
    if mode == "none":
        return f
    in_window = (t_end - step) <= horizon
    if not in_window:
        return f
    if mode == "gripper_close":
        # 末端 gripper 直接闭合到 0 (模拟「物体掉落 -> 抓手夹空闭合」, 触发 GRIP_FAIL/DROP)。
        f[_GRIP_IDX] = 0.0
    elif mode == "qvel_spike":
        # 末端 qvel 突增超阈值 (COLLISION: > 20 rad/s)
        f[_QVEL_START:_QVEL_END] = 25.0
    else:
        raise ValueError(f"未知 failure_mode={mode!r}")
    return f


def make_synthetic_episode(
    episode_id: str,
    seed: int,
    fault_type: Optional[str],
    *,
    n_frames: int = 50,
    failed: bool = True,
    horizon: int = DEFAULT_PREDICTION_HORIZON,
    failure_mode: str = "gripper_close",
    fault_params: Optional[dict] = None,
    rng: Optional[random.Random] = None,
) -> EpisodeRecord:
    """合成单个 EpisodeRecord。t_end = n_frames-1 (最后一个 step)。"""
    if n_frames < 1:
        raise ValueError(f"n_frames 须 >= 1, 得到 {n_frames}")
    rng = rng or random.Random(seed)
    steps = list(range(n_frames))
    t_end = n_frames - 1
    labels = label_failure(steps, failed=failed, t_end=t_end, horizon=horizon)
    frames: List[FrameRecord] = []
    for s in steps:
        f = _base_features(seed, s, rng)
        if failed:
            f = _apply_failure_signal(f, s, t_end, horizon, failure_mode)
        frames.append(FrameRecord(step=s, features=f, label=int(labels[s])))
    return EpisodeRecord(
        episode_id=episode_id,
        seed=seed,
        fault_type=fault_type,
        fault_params=dict(fault_params or {}),
        outcome=(not failed),
        frames=frames,
    )


def make_synthetic_episodes(
    *,
    seeds: List[int],
    fault_types: List[Optional[str]],
    failed_frac: float = 0.6,
    n_frames: int = 50,
    horizon: int = DEFAULT_PREDICTION_HORIZON,
    failure_mode: str = "gripper_close",
    base_seed: int = 0,
) -> List[EpisodeRecord]:
    """批量合成 episodes: 每个 seed 一个 episode, fault_types 循环分配。

    失败与否由 seed 确定性决定 (约 failed_frac 比例失败), 保证可复现。
    """
    if not seeds:
        return []
    if not fault_types:
        raise ValueError("fault_types 不能为空")
    out: List[EpisodeRecord] = []
    for i, seed in enumerate(seeds):
        rng = random.Random(base_seed * 1000003 + seed)
        failed = rng.random() < failed_frac
        ft = fault_types[i % len(fault_types)]
        out.append(make_synthetic_episode(
            episode_id=f"ep_{seed}",
            seed=seed,
            fault_type=ft,
            n_frames=n_frames,
            failed=failed,
            horizon=horizon,
            failure_mode=failure_mode,
            rng=rng,
        ))
    return out
