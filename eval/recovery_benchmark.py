"""B0 (无恢复基线) vs B1 (同基线 + 规则恢复) 的 matched 恢复对比 (M3-D)。

matched 键 = (task_id, seed, fault_type, frozen fault params) + privileged_access。
B0 与 B1 必须用相同 seed / 任务配置 / 故障参数 / 特权访问级别, 逐对对齐后对比。
未匹配的 episode 单独计数并计入对比结果 (不静默丢弃)。

不把 mock 结果冒充 benchmark: 本模块只负责『对齐 + 计算 + 对比』,
上报前仍需调用 eval.source.require_sim 守卫 (语义保留)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from interfaces import EpisodeResult

from .fault_injection import (
    RecoveryMetrics,
    compute_recovery_metrics,
    extract_fault,
    extract_privileged,
    extract_seed,
    extract_baseline,
    _B0, _B1,
)
from .source import validate_source


def _freeze(value) -> object:
    """把 dict/list 递归冻结为可哈希的不可变结构 (用于 matched key)。"""
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _side_privileged(eps: List[EpisodeResult]) -> Optional[bool]:
    """一侧 (B0 或 B1) 的 privileged_access; 未标注返回 None, 混标抛错。"""
    flags = {extract_privileged(ep) for ep in eps}
    flags.discard(None)
    if not flags:
        return None
    if len(flags) > 1:
        raise ValueError(
            f"一侧内 privileged_access 不一致 {sorted(flags)}; "
            "oracle 与观测型 episode 不得混在同一侧"
        )
    return flags.pop()


def match_key(ep: EpisodeResult) -> Tuple:
    """返回 B0/B1 matched 对齐键: (task_id, seed, fault_type, frozen_params, privileged_access)。"""
    f = extract_fault(ep)
    fault_type = f.fault_type if f is not None else None
    params = _freeze(f.params) if f is not None and f.params else None
    return (ep.task_id, extract_seed(ep), fault_type, params, extract_privileged(ep))


@dataclass
class RecoveryBenchmarkConfig:
    """一次恢复 benchmark 的配置: 故障场景 + 种子 + 条件 + 特权访问级别。"""
    name: str
    fault_scenarios: List[Tuple[str, dict]]   # [(fault_type, params), ...]
    seeds: List[int]                          # 每场景每条件的种子 (默认 10 个)
    conditions: Tuple[str, ...] = (_B0, _B1)  # 对比的条件 (baseline 名)
    privileged_access: bool = False           # 该基准的策略类型 (oracle True / 观测 False)
    episodes_per_seed: int = 1

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("RecoveryBenchmarkConfig.name 不能为空")
        if not self.fault_scenarios:
            raise ValueError("fault_scenarios 不能为空")
        for ft, p in self.fault_scenarios:
            if not isinstance(ft, str) or not ft:
                raise ValueError(f"fault_type 必须为非空字符串, 得到 {ft!r}")
            if not isinstance(p, dict):
                raise ValueError(f"fault params 必须为 dict, 得到 {p!r}")
        if not self.seeds:
            raise ValueError("seeds 不能为空")
        if any(not isinstance(s, int) for s in self.seeds):
            raise ValueError("seeds 必须为 int 列表")
        if self.episodes_per_seed < 1:
            raise ValueError("episodes_per_seed 必须 >= 1")
        if not set(self.conditions).issubset({_B0, _B1}):
            raise ValueError(f"conditions 仅允许 b0/b1, 得到 {self.conditions!r}")

    def iter_cases(self):
        """按 (condition, fault_type, seed, episode_index) 展开所有要执行的 case。"""
        for cond in self.conditions:
            for fault_type, params in self.fault_scenarios:
                for seed in self.seeds:
                    for i in range(self.episodes_per_seed):
                        yield (cond, fault_type, params, seed, i)

    @property
    def total_cases(self) -> int:
        return (
            len(self.conditions)
            * len(self.fault_scenarios)
            * len(self.seeds)
            * self.episodes_per_seed
        )


@dataclass
class RecoveryComparison:
    """B0 vs B1 matched 对比结果。

    - 每侧一个 RecoveryMetrics (B0/B1)。
    - deltas: 数值指标 B1 - B0 (None 指标不参与差值)。
    - matched_pairs / b0_unmatched / b1_unmatched: 对齐质量 (显式, 不静默丢弃)。
    """
    b0: RecoveryMetrics
    b1: RecoveryMetrics
    matched_pairs: int
    b0_unmatched: int
    b1_unmatched: int
    privileged_access: Optional[bool]
    deltas: Dict[str, Optional[float]] = field(default_factory=dict)
    per_fault_type: Dict[str, "RecoveryComparison"] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "privileged_access": self.privileged_access,
            "matched_pairs": self.matched_pairs,
            "b0_unmatched": self.b0_unmatched,
            "b1_unmatched": self.b1_unmatched,
            "b0": self.b0.to_dict(),
            "b1": self.b1.to_dict(),
            "deltas": dict(self.deltas),
        }


_NUMERIC_METRICS = (
    "task_success_rate",
    "verified_fault_injection_rate",
    "failure_detection_rate",
    "recovery_success_rate",
    "avg_retry_count",
    "recovery_latency_mean_steps",
    "mean_episode_length",
    "mean_cumulative_reward",
)


def compare_b0_b1(
    b0_episodes: List[EpisodeResult],
    b1_episodes: List[EpisodeResult],
    *,
    b0_name: str = "b0",
    b1_name: str = "b1",
) -> RecoveryComparison:
    """对齐 B0/B1 episode (matched 键) 并计算恢复指标 + 差值。

    - 仅对比真正匹配的 pair (B0 与 B1 各取同一 matched 键)。
    - 未匹配的 episode 计数 b0_unmatched / b1_unmatched, 不参与指标 (也不静默丢弃)。
    """
    b0 = list(b0_episodes)
    b1 = list(b1_episodes)
    validate_source(b0 + b1)

    # B0 与 B1 必须在同一 privileged_access 级别对比 (oracle vs oracle, 观测 vs 观测)
    b0_priv = _side_privileged(b0)
    b1_priv = _side_privileged(b1)
    if b0_priv != b1_priv:
        raise ValueError(
            f"B0 与 B1 privileged_access 不一致 (b0={b0_priv!r}, b1={b1_priv!r}); "
            "oracle(True) 与观测型(False)基线必须分开对比"
        )

    b0_by_key: Dict[Tuple, List[EpisodeResult]] = {}
    for ep in b0:
        b0_by_key.setdefault(match_key(ep), []).append(ep)
    b1_by_key: Dict[Tuple, List[EpisodeResult]] = {}
    for ep in b1:
        b1_by_key.setdefault(match_key(ep), []).append(ep)

    keys = set(b0_by_key) & set(b1_by_key)
    b0_matched = [ep for k in keys for ep in b0_by_key[k]]
    b1_matched = [ep for k in keys for ep in b1_by_key[k]]
    b0_unmatched = len(b0) - len(b0_matched)
    b1_unmatched = len(b1) - len(b1_matched)
    matched_pairs = min(
        sum(len(b0_by_key[k]) for k in keys),
        sum(len(b1_by_key[k]) for k in keys),
    )

    privs = {extract_privileged(ep) for ep in (b0_matched + b1_matched)}
    privs.discard(None)
    if len(privs) > 1:
        raise ValueError(
            f"matched pair 跨了不同 privileged_access {sorted(privs)}; "
            "oracle 与观测型基线必须分开对比"
        )
    priv = privs.pop() if privs else None

    b0_metrics = compute_recovery_metrics(b0_matched)
    b1_metrics = compute_recovery_metrics(b1_matched)

    deltas: Dict[str, Optional[float]] = {}
    for name in _NUMERIC_METRICS:
        v0 = getattr(b0_metrics, name)
        v1 = getattr(b1_metrics, name)
        deltas[name] = (round(v1 - v0, 10) if (v0 is not None and v1 is not None) else None)

    # per fault_type 分解
    per_ft: Dict[str, RecoveryComparison] = {}
    ft_set = {extract_fault(ep).fault_type for ep in (b0_matched + b1_matched)
              if extract_fault(ep) is not None}
    for ft in sorted(ft_set):
        b0_ft = [ep for ep in b0_matched if extract_fault(ep) is not None and extract_fault(ep).fault_type == ft]
        b1_ft = [ep for ep in b1_matched if extract_fault(ep) is not None and extract_fault(ep).fault_type == ft]
        per_ft[ft] = RecoveryComparison(
            b0=compute_recovery_metrics(b0_ft),
            b1=compute_recovery_metrics(b1_ft),
            matched_pairs=min(len(b0_ft), len(b1_ft)),
            b0_unmatched=max(0, len(b0_ft) - min(len(b0_ft), len(b1_ft))),
            b1_unmatched=max(0, len(b1_ft) - min(len(b0_ft), len(b1_ft))),
            privileged_access=priv,
        )

    return RecoveryComparison(
        b0=b0_metrics,
        b1=b1_metrics,
        matched_pairs=matched_pairs,
        b0_unmatched=b0_unmatched,
        b1_unmatched=b1_unmatched,
        privileged_access=priv,
        deltas=deltas,
        per_fault_type=per_ft,
    )


def label_condition(privileged_access: Optional[bool]) -> str:
    """把 privileged_access 翻译成结果标签 (oracle/观测型基线)。"""
    if privileged_access is True:
        return "oracle"
    if privileged_access is False:
        return "observational"
    return "unlabeled"
