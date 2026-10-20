"""episode 级 train/val/test 切分 (M4-D 学习评估协议)。

切分保证 (防泄漏, verify_split 强制校验):
1. episode 原子性   : 每个 EpisodeRecord 整体进入唯一 split, 绝不帧级切分
                       (同一 episode 的所有帧同属一个 split)。
2. held-out seeds   : 切分在 seed 粒度进行 —— 相同 seed 的 episode 全部同 split,
                       train/val/test 三者的 seed 集合两两不相交 (test seed 绝不在 train/val)。
3. fault_type 分层  : 按 fault_type 分层切分, 每个 split 内 fault_type 分布尽量均衡。
4. held-out 扰动配置: held_out_fault_types 指定的 fault_type 的 episode 全部进 test
                       (train/val 完全不含这些扰动类型), 用于测试对新扰动类型的泛化。

所有切分决策确定性 (给定 seed)。
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from interfaces import EpisodeRecord

# fault_type=None (正常执行) 的内部 strata 键, 仅用于分组/排序, 报告时还原。
_NONE_FAULT = "__none__"


def _fault_key(fault_type: Optional[str]) -> str:
    return _NONE_FAULT if fault_type is None else fault_type


def _primary_fault_type(eps: List[EpisodeRecord]) -> str:
    """一个 seed 组的主 fault_type (最多者, 平局取字典序最小)。"""
    cnt = Counter(_fault_key(ep.fault_type) for ep in eps)
    return cnt.most_common(1)[0][0]


@dataclass
class LearningSplit:
    """episode 级切分结果。train/val/test 各自持有完整 EpisodeRecord。"""

    train: List[EpisodeRecord]
    val: List[EpisodeRecord]
    test: List[EpisodeRecord]
    held_out_fault_types: List[str] = field(default_factory=list)

    @property
    def train_seeds(self) -> List[int]:
        return sorted({ep.seed for ep in self.train})

    @property
    def val_seeds(self) -> List[int]:
        return sorted({ep.seed for ep in self.val})

    @property
    def test_seeds(self) -> List[int]:
        return sorted({ep.seed for ep in self.test})

    def summary(self) -> dict:
        """raw counts 报告: 每个 split 的 episode 数/帧数/正样本帧数/fault_type 分布, 以及 seeds。"""
        def _cnt(eps: List[EpisodeRecord]) -> dict:
            frames = sum(len(ep.frames) for ep in eps)
            pos = sum(1 for ep in eps for f in ep.frames if f.label == 1)
            ft = dict(Counter(ep.fault_type for ep in eps))
            return {
                "num_episodes": len(eps),
                "num_frames": frames,
                "num_positive_frames": pos,
                "fault_type_counts": ft,
            }
        return {
            "train": _cnt(self.train),
            "val": _cnt(self.val),
            "test": _cnt(self.test),
            "seeds": {"train": self.train_seeds, "val": self.val_seeds, "test": self.test_seeds},
            "held_out_fault_types": list(self.held_out_fault_types),
            "test_seeds_disjoint_from_train_val": self._seeds_disjoint(),
        }

    def to_dict(self) -> dict:
        return {
            "train_episode_ids": [ep.episode_id for ep in self.train],
            "val_episode_ids": [ep.episode_id for ep in self.val],
            "test_episode_ids": [ep.episode_id for ep in self.test],
            "seeds": {"train": self.train_seeds, "val": self.val_seeds, "test": self.test_seeds},
            "held_out_fault_types": list(self.held_out_fault_types),
            "summary": self.summary(),
        }

    def _seeds_disjoint(self) -> bool:
        t, v, tr = set(self.test_seeds), set(self.val_seeds), set(self.train_seeds)
        return not (t & tr) and not (t & v) and not (v & tr)


def _flatten(seeds: List[int], by_seed: Dict[int, List[EpisodeRecord]]) -> List[EpisodeRecord]:
    out: List[EpisodeRecord] = []
    for s in sorted(seeds):
        out.extend(sorted(by_seed[s], key=lambda e: e.episode_id))
    return out


def split_episodes(
    episodes: List[EpisodeRecord],
    *,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
    held_out_fault_types: Optional[List[str]] = None,
) -> LearningSplit:
    """episode 级 train/val/test 切分 (确定性)。

    held_out_fault_types: 这些 fault_type 的 episode 全部进 test (held-out 扰动配置)。
    val_frac/test_frac: 常规 (非 held-out) episode 的目标比例 (train = 1 - val - test)。
    """
    if not episodes:
        raise ValueError("episodes 不能为空")
    if val_frac < 0.0 or test_frac < 0.0 or (val_frac + test_frac) >= 1.0:
        raise ValueError(
            f"val_frac/test_frac 须 >=0 且 val+test < 1.0, 得到 val={val_frac} test={test_frac}"
        )

    ids = [ep.episode_id for ep in episodes]
    dup = [i for i, c in Counter(ids).items() if c > 1]
    if dup:
        raise ValueError(f"episode_id 重复: {dup}")

    held_out = set(held_out_fault_types or [])

    # 按 seed 分组 (seed 原子性)。
    by_seed: Dict[int, List[EpisodeRecord]] = {}
    for ep in episodes:
        by_seed.setdefault(ep.seed, []).append(ep)

    # 判定每个 seed 组是否为 held-out 扰动配置。
    held_seeds: List[int] = []
    normal_seeds: List[int] = []
    for s, eps in by_seed.items():
        fts = {ep.fault_type for ep in eps}
        is_held = {ft in held_out for ft in fts}
        if len(is_held) > 1:
            raise ValueError(
                f"seed={s} 的 episode 混合 held-out 与非 held-out fault_type {fts}; "
                "同一 seed 的 fault_type 必须一致 (seed 原子性)"
            )
        if is_held == {True}:
            held_seeds.append(s)
        else:
            normal_seeds.append(s)

    # 常规 seed 按主 fault_type 分层。
    rng = random.Random(seed)
    strata: Dict[str, List[int]] = {}
    for s in normal_seeds:
        strata.setdefault(_primary_fault_type(by_seed[s]), []).append(s)

    train_seeds: List[int] = []
    val_seeds: List[int] = []
    test_seeds: List[int] = []
    for pk in sorted(strata):
        gs = strata[pk]
        rng.shuffle(gs)
        n = len(gs)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        if n_test + n_val >= n:
            # 该层 seed 太少: 优先保 val, 再 test, 至少留 1 个给 train。
            n_val = min(n_val, n - 1)
            n_test = min(n_test, n - 1 - n_val)
        test_seeds.extend(gs[:n_test])
        val_seeds.extend(gs[n_test:n_test + n_val])
        train_seeds.extend(gs[n_test + n_val:])

    # held-out 扰动配置全部进 test。
    test_seeds.extend(sorted(held_seeds))

    # 全局保证: test_frac>0 时 test 至少 1 个 seed。
    if test_frac > 0 and not test_seeds:
        if train_seeds:
            moved = train_seeds[-1]
            train_seeds = train_seeds[:-1]
            test_seeds = [moved]
        else:
            raise ValueError("无可用 seed 构成非空 test 集 (seed 太少)")

    split = LearningSplit(
        train=_flatten(train_seeds, by_seed),
        val=_flatten(val_seeds, by_seed),
        test=_flatten(test_seeds, by_seed),
        held_out_fault_types=list(held_out),
    )
    verify_split(split)
    return split


def verify_split(split: LearningSplit) -> None:
    """校验切分无泄漏; 违反时抛 AssertionError。"""
    def ids(eps): return [ep.episode_id for ep in eps]
    def seeds(eps): return set(ep.seed for ep in eps)

    # episode 原子性: 每个 episode 恰好出现一次, 不跨 split 重复。
    seen = Counter(ids(split.train) + ids(split.val) + ids(split.test))
    dup = [i for i, c in seen.items() if c > 1]
    assert not dup, f"episode 跨 split 重复: {dup}"

    ts, vs, trs = seeds(split.test), seeds(split.val), seeds(split.train)
    assert not (ts & trs), "test seed 泄漏进 train"
    assert not (ts & vs), "test seed 泄漏进 val"
    assert not (vs & trs), "val seed 泄漏进 train"

    # held-out fault_type 只出现在 test。
    for ft in split.held_out_fault_types:
        for ep in split.train + split.val:
            assert ep.fault_type != ft, f"held-out fault_type {ft!r} 泄漏进 train/val"
