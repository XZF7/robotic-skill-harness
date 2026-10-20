"""基线 (baseline) 加载与对比基础设施。

- Baseline: 一个基线的名称 + 一组 EpisodeResult + 聚合后的 EpisodeMetrics。
- rank_baselines: 按 success_rate 降序排名。
- compare_baselines: 相对参考基线的各项差值 (成功率/平均步数/平均奖励)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from interfaces import EpisodeResult, EpisodeMetrics

from .metrics import compute_metrics
from .serialization import episodes_to_json, episodes_from_json
from .source import source_counts


@dataclass
class Baseline:
    name: str
    episodes: List[EpisodeResult]
    metrics: EpisodeMetrics
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_episodes(cls, name: str, episodes: List[EpisodeResult], meta: Optional[dict] = None) -> "Baseline":
        return cls(name=name, episodes=list(episodes), metrics=compute_metrics(episodes), meta=meta or {})

    @classmethod
    def from_json(cls, name: str, payload: str, meta: Optional[dict] = None) -> "Baseline":
        """从 episodes JSON 字符串加载基线 (自动聚合指标)。"""
        eps = episodes_from_json(payload)
        return cls.from_episodes(name, eps, meta)

    def to_json(self, **kw) -> str:
        """导出该基线的 episodes (JSON), 可被 from_json 回读。"""
        return episodes_to_json(self.episodes, **kw)

    def source_counts(self) -> dict:
        return source_counts(self.episodes)

    def is_sim_baseline(self) -> bool:
        return all(ep.source == "sim" for ep in self.episodes)


def rank_baselines(baselines: List[Baseline]) -> List[Baseline]:
    """按 success_rate 降序, 再 mean_reward 降序, 再 mean_steps 升序, 再 name 升序。"""
    return sorted(
        baselines,
        key=lambda b: (-b.metrics.success_rate, -b.metrics.mean_reward, b.metrics.mean_steps, b.name),
    )


@dataclass
class BaselineComparison:
    reference: str
    ranking: List[str]                       # 按 success_rate 降序
    success_rate_deltas: Dict[str, float]    # 各基线相对 reference 的 success_rate 差值
    mean_steps_deltas: Dict[str, float]
    mean_reward_deltas: Dict[str, float]
    reference_success_rate: float

    def to_dict(self) -> dict:
        return {
            "reference": self.reference,
            "ranking": list(self.ranking),
            "reference_success_rate": self.reference_success_rate,
            "success_rate_deltas": dict(self.success_rate_deltas),
            "mean_steps_deltas": dict(self.mean_steps_deltas),
            "mean_reward_deltas": dict(self.mean_reward_deltas),
        }


def compare_baselines(baselines: List[Baseline], reference: Optional[str] = None) -> BaselineComparison:
    """对比一组基线, 返回排名与相对 reference (默认排名第一) 的差值。"""
    if not baselines:
        raise ValueError("至少需要一个 baseline 才能对比")
    names = [b.name for b in baselines]
    if len(set(names)) != len(names):
        raise ValueError(f"baseline 名称重复: {names}")

    ranked = rank_baselines(baselines)
    ref = reference if reference is not None else ranked[0].name
    by_name = {b.name: b for b in baselines}
    if ref not in by_name:
        raise ValueError(f"reference={ref!r} 不在 baselines 中: {names}")

    ref_metrics = by_name[ref].metrics
    return BaselineComparison(
        reference=ref,
        ranking=[b.name for b in ranked],
        success_rate_deltas={b.name: round(b.metrics.success_rate - ref_metrics.success_rate, 10) for b in baselines},
        mean_steps_deltas={b.name: round(b.metrics.mean_steps - ref_metrics.mean_steps, 10) for b in baselines},
        mean_reward_deltas={b.name: round(b.metrics.mean_reward - ref_metrics.mean_reward, 10) for b in baselines},
        reference_success_rate=ref_metrics.success_rate,
    )
