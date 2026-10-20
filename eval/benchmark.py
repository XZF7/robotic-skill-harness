"""基准 (benchmark) 配置结构与一次基准运行的结果聚合。

- BenchmarkConfig: 任务列表 + seed 列表 + 每任务 episode 数 + horizon。
- BenchmarkRun: 一次运行 = 配置 + 全部 episode + 总指标 + 每任务指标。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Tuple

from interfaces import EpisodeResult, EpisodeMetrics

from .metrics import compute_metrics


@dataclass
class BenchmarkConfig:
    name: str
    tasks: List[str]
    seeds: List[int]
    episodes_per_task: int = 10
    max_steps: int = 200
    source: str = "sim"            # 真实基准的目标来源
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("BenchmarkConfig.name 不能为空")
        if not self.tasks:
            raise ValueError("BenchmarkConfig.tasks 不能为空")
        if any(not isinstance(t, str) or not t for t in self.tasks):
            raise ValueError("BenchmarkConfig.tasks 必须为非空字符串列表")
        if not self.seeds:
            raise ValueError("BenchmarkConfig.seeds 不能为空")
        if any(not isinstance(s, int) for s in self.seeds):
            raise ValueError("BenchmarkConfig.seeds 必须为 int 列表")
        if self.episodes_per_task < 1:
            raise ValueError("BenchmarkConfig.episodes_per_task 必须 >= 1")
        if self.max_steps < 1:
            raise ValueError("BenchmarkConfig.max_steps 必须 >= 1")
        if self.source not in ("mock", "sim"):
            raise ValueError(f"BenchmarkConfig.source 非法: {self.source!r}")

    @property
    def total_episodes(self) -> int:
        return len(self.tasks) * len(self.seeds) * self.episodes_per_task

    def iter_cases(self) -> Iterator[Tuple[str, int, int]]:
        """按 (task_id, seed, episode_index) 展开所有要执行的 case。"""
        for task in self.tasks:
            for seed in self.seeds:
                for i in range(self.episodes_per_task):
                    yield (task, seed, i)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "tasks": list(self.tasks),
            "seeds": list(self.seeds),
            "episodes_per_task": self.episodes_per_task,
            "max_steps": self.max_steps,
            "source": self.source,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkConfig":
        return cls(
            name=d["name"],
            tasks=list(d["tasks"]),
            seeds=list(d["seeds"]),
            episodes_per_task=int(d.get("episodes_per_task", 10)),
            max_steps=int(d.get("max_steps", 200)),
            source=d.get("source", "sim"),
            extra=dict(d.get("extra", {})),
        )

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_json(cls, s: str) -> "BenchmarkConfig":
        return cls.from_dict(json.loads(s))


@dataclass
class BenchmarkRun:
    config: BenchmarkConfig
    episodes: List[EpisodeResult]
    metrics: EpisodeMetrics
    per_task_metrics: Dict[str, EpisodeMetrics]

    @classmethod
    def from_episodes(cls, config: BenchmarkConfig, episodes: List[EpisodeResult]) -> "BenchmarkRun":
        per_task: Dict[str, List[EpisodeResult]] = {}
        for ep in episodes:
            per_task.setdefault(ep.task_id, []).append(ep)
        per_task_metrics = {t: compute_metrics(eps) for t, eps in per_task.items()}
        return cls(
            config=config,
            episodes=episodes,
            metrics=compute_metrics(episodes),
            per_task_metrics=per_task_metrics,
        )
