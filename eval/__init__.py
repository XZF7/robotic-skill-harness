"""Agent D - Evaluation: episode 序列化、指标计算、基准配置、基线对比、故障注入恢复评估。

开发期全部使用 mock EpisodeResult (source="mock")。
严禁在集成真实仿真 (source="sim") 前把 mock 结果当作 benchmark 表现上报，
上报前必须通过 `eval.source.require_sim` 守卫。

M3-D 新增: 故障注入 episode 消费 + B0/B1 恢复对比 (eval.fault_injection /
eval.recovery_benchmark), 所有恢复指标分母显式定义。
"""
from .serialization import (
    episode_to_dict, episode_from_dict,
    episode_to_json, episode_from_json,
    episodes_to_dicts, episodes_from_dicts,
    episodes_to_json, episodes_from_json,
    failure_event_to_dict, failure_event_from_dict,
    skill_result_to_dict, skill_result_from_dict,
    recovery_decision_to_dict, recovery_decision_from_dict,
    recovery_result_to_dict, recovery_result_from_dict,
    fault_record_to_dict, fault_record_from_dict,
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
from .fault_injection import (
    FAULT_KEY, RECOVERY_KEY, PRIVILEGED_KEY, BASELINE_KEY, SEED_KEY, DETECTED_KEY,
    attach_fault, attach_recovery, attach_privileged, attach_baseline,
    extract_fault, extract_privileged, extract_seed, extract_baseline,
    failure_manifested, is_valid_injected_failure, failure_detected, recovery_attempted,
    DEFINITIONS, RecoveryMetrics, compute_recovery_metrics, split_by_privileged,
)
from .learning_split import LearningSplit, split_episodes, verify_split
from .learning_metrics import (
    DEFINITIONS as LEARNING_METRIC_DEFINITIONS,
    average_precision,
    PredictionMetrics,
    compute_prediction_metrics,
)
from .learning_baselines import (
    MajorityClassPredictor,
    RuleBasedPredictor,
    LearnedPredictor,
    build_default_baselines,
    evaluate_baselines,
)
from .recovery_benchmark import (
    match_key, RecoveryBenchmarkConfig, RecoveryComparison,
    compare_b0_b1, label_condition,
)

__all__ = [
    "episode_to_dict", "episode_from_dict",
    "episode_to_json", "episode_from_json",
    "episodes_to_dicts", "episodes_from_dicts",
    "episodes_to_json", "episodes_from_json",
    "failure_event_to_dict", "failure_event_from_dict",
    "skill_result_to_dict", "skill_result_from_dict",
    "recovery_decision_to_dict", "recovery_decision_from_dict",
    "recovery_result_to_dict", "recovery_result_from_dict",
    "fault_record_to_dict", "fault_record_from_dict",
    "VALID_SOURCES", "is_mock", "is_sim",
    "validate_source", "source_counts", "filter_by_source",
    "require_sim", "require_all_mock",
    "compute_metrics",
    "make_mock_episode", "make_mock_episodes",
    "BenchmarkConfig", "BenchmarkRun",
    "Baseline", "BaselineComparison", "rank_baselines", "compare_baselines",
    "FAULT_KEY", "RECOVERY_KEY", "PRIVILEGED_KEY", "BASELINE_KEY", "SEED_KEY", "DETECTED_KEY",
    "attach_fault", "attach_recovery", "attach_privileged", "attach_baseline",
    "extract_fault", "extract_privileged", "extract_seed", "extract_baseline",
    "failure_manifested", "is_valid_injected_failure", "failure_detected", "recovery_attempted",
    "DEFINITIONS", "RecoveryMetrics", "compute_recovery_metrics", "split_by_privileged",
    "match_key", "RecoveryBenchmarkConfig", "RecoveryComparison",
    "compare_b0_b1", "label_condition",
    "LearningSplit", "split_episodes", "verify_split",
    "LEARNING_METRIC_DEFINITIONS", "average_precision", "PredictionMetrics",
    "compute_prediction_metrics",
    "MajorityClassPredictor", "RuleBasedPredictor", "LearnedPredictor",
    "build_default_baselines", "evaluate_baselines",
]
