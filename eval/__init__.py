"""Agent D - Evaluation: episode 序列化、指标计算、基准配置、基线对比。

开发期全部使用 mock EpisodeResult (source="mock")。
严禁在集成真实仿真 (source="sim") 前把 mock 结果当作 benchmark 表现上报，
上报前必须通过 `eval.source.require_sim` 守卫。
"""
from .serialization import (
    episode_to_dict, episode_from_dict,
    episode_to_json, episode_from_json,
    episodes_to_dicts, episodes_from_dicts,
    episodes_to_json, episodes_from_json,
)
from .source import (
    VALID_SOURCES, is_mock, is_sim,
    validate_source, source_counts, filter_by_source,
    require_sim, require_all_mock,
)
from .metrics import compute_metrics
from .mocks import make_mock_episode, make_mock_episodes
from .benchmark import BenchmarkConfig, BenchmarkRun
from .baselines import Baseline, BaselineComparison, rank_baselines, compare_baselines

__all__ = [
    "episode_to_dict", "episode_from_dict",
    "episode_to_json", "episode_from_json",
    "episodes_to_dicts", "episodes_from_dicts",
    "episodes_to_json", "episodes_from_json",
    "VALID_SOURCES", "is_mock", "is_sim",
    "validate_source", "source_counts", "filter_by_source",
    "require_sim", "require_all_mock",
    "compute_metrics",
    "make_mock_episode", "make_mock_episodes",
    "BenchmarkConfig", "BenchmarkRun",
    "Baseline", "BaselineComparison", "rank_baselines", "compare_baselines",
]
