"""source 字段守卫: mock (合成) 与 sim (真实仿真) 的严格区分。

上报任何 benchmark 表现前, 必须通过 `require_sim` 校验, 防止把开发期
mock 结果冒充真实仿真结果。
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, List

from interfaces import EpisodeResult

VALID_SOURCES = ("mock", "sim")


def is_mock(ep: EpisodeResult) -> bool:
    return ep.source == "mock"


def is_sim(ep: EpisodeResult) -> bool:
    return ep.source == "sim"


def validate_source(episodes: Iterable[EpisodeResult]) -> None:
    """校验所有 episode 的 source 取值合法。"""
    for ep in episodes:
        if ep.source not in VALID_SOURCES:
            raise ValueError(
                f"EpisodeResult(task_id={ep.task_id!r}) 的 source={ep.source!r} 非法, "
                f"仅允许 {VALID_SOURCES!r}"
            )


def source_counts(episodes: Iterable[EpisodeResult]) -> dict:
    validate_source(episodes)
    return dict(Counter(ep.source for ep in episodes))


def filter_by_source(episodes: Iterable[EpisodeResult], source: str) -> List[EpisodeResult]:
    if source not in VALID_SOURCES:
        raise ValueError(f"source={source!r} 非法, 仅允许 {VALID_SOURCES!r}")
    return [ep for ep in episodes if ep.source == source]


def require_sim(episodes: Iterable[EpisodeResult], *, context: str = "benchmark report") -> List[EpisodeResult]:
    """上报基准表现前的强制守卫: 所有 episode 必须来自真实仿真 (sim)。"""
    eps = list(episodes)
    validate_source(eps)
    mocks = [ep for ep in eps if is_mock(ep)]
    if mocks:
        raise ValueError(
            f"禁止用 mock 结果生成 {context}: {len(mocks)}/{len(eps)} 个 episode "
            f"source='mock'。集成真实仿真前, mock 结果只能用于开发期测试。"
        )
    return eps


def require_all_mock(episodes: Iterable[EpisodeResult], *, context: str = "dev test") -> List[EpisodeResult]:
    """开发期校验: 确保当前全部为 mock (未混入 sim)。"""
    eps = list(episodes)
    validate_source(eps)
    sims = [ep for ep in eps if is_sim(ep)]
    if sims:
        raise ValueError(
            f"{context} 期望全部为 mock, 但发现 {len(sims)} 个 episode source='sim'。"
        )
    return eps
