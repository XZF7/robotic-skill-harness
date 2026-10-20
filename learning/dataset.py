"""数据集加载 (M4-A): 读 dataset.npz + episodes_meta.json, 构造 EpisodeRecord。

归一化严格防泄漏: mean/std 只从 train 集计算, 再应用到 val/test。
step = episode 内帧的局部索引 (0-based), 与离线 label_failure 的 t_end 语义一致
(t_end = num_frames - 1, 即最后一个 step)。
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import numpy as np

from interfaces import EpisodeRecord, FrameRecord, STATE_FEATURE_DIM

# 仓库内默认数据目录 (data/m4)。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(_REPO_ROOT, "data", "m4")


def load_episodes(
    npz_path: Optional[str] = None,
    meta_path: Optional[str] = None,
) -> List[EpisodeRecord]:
    """加载全部 episode, 按 meta 的 frame_start/frame_end 切片构造 FrameRecord。"""
    npz_path = npz_path or os.path.join(DEFAULT_DATA_DIR, "dataset.npz")
    meta_path = meta_path or os.path.join(DEFAULT_DATA_DIR, "episodes_meta.json")

    d = np.load(npz_path)
    features = d["features"].astype(np.float32)
    labels = d["labels"].astype(np.int64)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    episodes: List[EpisodeRecord] = []
    for e in meta:
        fs, fe = int(e["frame_start"]), int(e["frame_end"])
        n = fe - fs
        if n != int(e["num_frames"]):
            raise ValueError(
                f"episode {e['episode_id']}: frame_start/frame_end 与 num_frames 不一致 "
                f"({n} vs {e['num_frames']})"
            )
        frames: List[FrameRecord] = []
        for j in range(n):
            frames.append(FrameRecord(
                step=j,
                features=features[fs + j].copy(),
                label=int(labels[fs + j]),
            ))
        episodes.append(EpisodeRecord(
            episode_id=e["episode_id"],
            seed=int(e["seed"]),
            fault_type=e.get("fault_type"),
            fault_params=dict(e.get("fault_params") or {}),
            outcome=bool(e["outcome"]),
            frames=frames,
        ))
    return episodes


def episodes_to_arrays(
    episodes: List[EpisodeRecord],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """展平 episodes 的帧 -> (X, y, ep_ids)。

    X: (N, 22) float32; y: (N,) int64; ep_ids: (N,) int64 (帧所属 episode 在输入列表中的索引)。
    """
    xs: List[np.ndarray] = []
    ys: List[int] = []
    eids: List[int] = []
    for i, ep in enumerate(episodes):
        for f in ep.frames:
            xs.append(np.asarray(f.features, dtype=np.float32).reshape(-1))
            ys.append(int(f.label))
            eids.append(i)
    if not xs:
        return (
            np.zeros((0, STATE_FEATURE_DIM), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.stack(xs).astype(np.float32),
        np.asarray(ys, dtype=np.int64),
        np.asarray(eids, dtype=np.int64),
    )


def compute_normalization_stats(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """从给定特征计算 per-feature mean/std (22,)。std 用 eps 防除零。"""
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """应用 (X - mean) / std。"""
    X = np.asarray(X, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32).reshape(1, -1)
    std = np.asarray(std, dtype=np.float32).reshape(1, -1)
    return (X - mean) / std
