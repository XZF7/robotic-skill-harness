"""EpisodeResult / EpisodeMetrics contract.

source 字段区分合成 mock 结果与真实仿真结果:
- "mock": 合成/伪造的 episode 记录 (仅用于开发期测试)。
- "sim":  真实仿真产生的 episode 结果。
严禁在集成真实仿真前用 mock 结果冒充 benchmark 表现。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

from .failure import FailureEvent
from .skill import SkillResult


@dataclass
class EpisodeResult:
    task_id: str
    success: bool
    steps: int
    reward: float
    skill_results: List[SkillResult] = field(default_factory=list)
    failures: List[FailureEvent] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    source: str = "mock"          # "mock" | "sim"


@dataclass
class EpisodeMetrics:
    num_episodes: int
    success_rate: float            # [0,1]
    mean_steps: float
    mean_reward: float = 0.0
    extra: dict = field(default_factory=dict)
