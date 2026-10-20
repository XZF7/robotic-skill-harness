"""故障注入 benchmark 的 episode 消费与恢复指标计算 (M3-D)。

消费约定 (故障注入信息挂在 EpisodeResult.metrics 里, 也可直接持有 FaultRecord/RecoveryResult 对象):
- metrics["fault"]              -> FaultRecord 或 dict(见 fault_record_to_dict): 故障注入记录。
- metrics["recovery"]           -> RecoveryResult 或 dict(见 recovery_result_to_dict): 恢复结果。
- metrics["privileged_access"]  -> bool: True=oracle/状态基线(特权真值), False=观测型策略。
- metrics["baseline"]           -> "b0"(无恢复基线) | "b1"(同基线+规则恢复)。
- metrics["seed"]               -> int: 匹配 B0/B1 用的 seed。
- metrics["detected"]           -> bool(可选): 失败是否被恢复机制检测到; 缺省时由 recovery 推断。

所有指标的分母在本模块显式定义 (DEFINITIONS), 且 compute_recovery_metrics 返回的
结果中同时携带 numerators (counts) 与 denominators (显式分母), 绝不隐式。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from interfaces import (
    EpisodeResult, FaultRecord, RecoveryResult, SkillStatus,
)

from .serialization import (
    fault_record_to_dict, fault_record_from_dict,
    recovery_result_to_dict,
)


# ---- 挂载/提取: 故障注入信息挂在 metrics dict 的规范键 --------------------

FAULT_KEY = "fault"
RECOVERY_KEY = "recovery"
PRIVILEGED_KEY = "privileged_access"
BASELINE_KEY = "baseline"
SEED_KEY = "seed"
DETECTED_KEY = "detected"

_B0 = "b0"
_B1 = "b1"
_B2 = "b2"
VALID_BASELINES = (_B0, _B1, _B2)


def attach_fault(ep: EpisodeResult, fault: FaultRecord) -> EpisodeResult:
    """把 FaultRecord 挂到 episode.metrics["fault"] (规范键), 返回原 episode。"""
    ep.metrics[FAULT_KEY] = fault_record_to_dict(fault)
    return ep


def attach_recovery(ep: EpisodeResult, recovery: RecoveryResult) -> EpisodeResult:
    """把 RecoveryResult 挂到 episode.metrics["recovery"], 返回原 episode。"""
    ep.metrics[RECOVERY_KEY] = recovery_result_to_dict(recovery)
    return ep


def attach_privileged(ep: EpisodeResult, flag: bool) -> EpisodeResult:
    ep.metrics[PRIVILEGED_KEY] = bool(flag)
    return ep


def attach_baseline(ep: EpisodeResult, name: str) -> EpisodeResult:
    if name not in VALID_BASELINES:
        raise ValueError(f"baseline={name!r} 非法, 仅允许 {VALID_BASELINES!r}")
    ep.metrics[BASELINE_KEY] = name
    return ep


def extract_fault(ep: EpisodeResult) -> Optional[FaultRecord]:
    """从 metrics["fault"] 提取 FaultRecord; 兼容 dict 或 FaultRecord 对象; 无则 None。"""
    raw = ep.metrics.get(FAULT_KEY)
    if raw is None:
        return None
    if isinstance(raw, FaultRecord):
        return raw
    if isinstance(raw, dict):
        return fault_record_from_dict(raw)
    raise TypeError(f"metrics['fault'] 必须是 FaultRecord 或 dict, 得到 {type(raw)!r}")


def _normalize_recovery(ep: EpisodeResult) -> Optional[dict]:
    """把 metrics["recovery"] (dict 或 RecoveryResult) 归一化为统一 dict。

    返回键: attempts, recovered, aborted, decisions(list), present(bool)。
    注意: 不要求 final_skill_result (episode.skill_results 已是权威)。
    """
    raw = ep.metrics.get(RECOVERY_KEY)
    if raw is None:
        return None
    if isinstance(raw, RecoveryResult):
        raw = recovery_result_to_dict(raw)
    if not isinstance(raw, dict):
        raise TypeError(f"metrics['recovery'] 必须是 RecoveryResult 或 dict, 得到 {type(raw)!r}")
    decisions = raw.get("decisions", [])
    if not isinstance(decisions, list):
        raise TypeError("recovery['decisions'] 必须是 list")
    return {
        "attempts": int(raw.get("attempts", 1)),
        "recovered": bool(raw.get("recovered", False)),
        "aborted": bool(raw.get("aborted", False)),
        "decisions": decisions,
        "present": True,
    }


def extract_privileged(ep: EpisodeResult, default: Optional[bool] = None) -> Optional[bool]:
    """从 metrics["privileged_access"] 提取特权访问标志 (bool)。"""
    v = ep.metrics.get(PRIVILEGED_KEY)
    if v is None:
        return default
    return bool(v)


def extract_seed(ep: EpisodeResult) -> Optional[int]:
    v = ep.metrics.get(SEED_KEY)
    return int(v) if v is not None else None


def extract_baseline(ep: EpisodeResult) -> Optional[str]:
    v = ep.metrics.get(BASELINE_KEY)
    if v is None:
        return None
    if v not in VALID_BASELINES:
        raise ValueError(f"metrics['baseline']={v!r} 非法, 仅允许 {VALID_BASELINES!r}")
    return v


# ---- 谓词: 失败是否真实发生 / 是否被检测到 --------------------------------

_FAILED_STATUSES = {SkillStatus.FAILURE, SkillStatus.TIMEOUT, SkillStatus.ABORTED}


def failure_manifested(ep: EpisodeResult) -> bool:
    """失败是否真实发生 (特权真值 + 观测信号 + 恢复信号): 任一满足即视为真实失败。

    - episode.success == False (env 特权真值判定任务失败);
    - 存在至少一个 FailureEvent;
    - 任一 SkillResult.status ∈ {FAILURE, TIMEOUT, ABORTED};
    - 恢复被触发 (attempts>1 或 decisions 非空 或 recovered/aborted),
      因为恢复机制只在失败发生后才会介入 —— 用于覆盖『恢复后最终 success=True、
      但失败确曾发生』的 episode。
    """
    if not ep.success:
        return True
    if ep.failures:
        return True
    if any(sr.status in _FAILED_STATUSES for sr in ep.skill_results):
        return True
    rec = _normalize_recovery(ep)
    if rec is not None and (rec["attempts"] > 1 or rec["decisions"]
                            or rec["recovered"] or rec["aborted"]):
        return True
    return False


def is_valid_injected_failure(ep: EpisodeResult) -> bool:
    """『实际注入成功的失败场景』: fault.injected == True 且失败真实发生。"""
    f = extract_fault(ep)
    return f is not None and f.injected and failure_manifested(ep)


def failure_detected(ep: EpisodeResult) -> bool:
    """失败是否被恢复机制检测到。

    优先取显式 metrics["detected"]; 否则由 recovery 推断:
    存在 recovery 且 (decisions 非空 或 attempts > 1)。B0 (无 recovery) 恒为 False。
    """
    if DETECTED_KEY in ep.metrics:
        return bool(ep.metrics[DETECTED_KEY])
    rec = _normalize_recovery(ep)
    if rec is None:
        return False
    return bool(rec["decisions"]) or rec["attempts"] > 1


def recovery_attempted(ep: EpisodeResult) -> bool:
    """是否实际发生了恢复尝试 (存在 recovery 且 attempts > 1 或 decisions 非空)。"""
    rec = _normalize_recovery(ep)
    if rec is None:
        return False
    return rec["attempts"] > 1 or bool(rec["decisions"])


# ---- 指标分母显式定义 ------------------------------------------------------

# 每个指标 -> 分母的精确文字定义。compute_recovery_metrics 会原样返回。
DEFINITIONS: Dict[str, str] = {
    "task_success_rate": "num_success / num_episodes (分母=该组总 episode 数)",
    "verified_fault_injection_rate": "num_injected / num_injection_attempted "
        "(分母=尝试注入的 episode 数, 即带 FaultRecord 的 episode; 分子=injected=True)",
    "failure_detection_rate": "num_detected / num_real_failures "
        "(分母=真实发生失败的场景数: injected=True 且 failure_manifested; 分子=其中被检测到的)",
    "recovery_success_rate": "num_recovered / num_injected_failure_scenarios "
        "(分母=实际注入成功的失败场景数: injected=True 且 failure_manifested; 分子=recovered=True)",
    "avg_retry_count": "sum(attempts-1) / num_recovery_attempted "
        "(分母=实际发生恢复尝试的 episode 数; retry = attempts-1, 首次尝试不算 retry)",
    "recovery_latency_mean_steps": "sum(episode.steps - first_failure_or_inject_step) / num_recovery_attempted "
        "(分母=实际发生恢复尝试的 episode 数; latency=从首个失败/注入步到 episode 结束的步数)",
    "mean_episode_length": "sum(steps) / num_episodes (分母=该组总 episode 数)",
    "mean_cumulative_reward": "sum(reward) / num_episodes (分母=该组总 episode 数)",
}


def _first_failure_or_inject_step(ep: EpisodeResult) -> int:
    """恢复延迟的起点: 首个 FailureEvent 步, 否则 fault.step (>=0), 否则 0。"""
    if ep.failures:
        return min(f.step for f in ep.failures)
    f = extract_fault(ep)
    if f is not None and f.step >= 0:
        return f.step
    return 0


@dataclass
class RecoveryMetrics:
    """一组 (同 baseline + 同 privileged_access) 故障注入 episode 的恢复指标。

    numerators 与 denominators 显式保留, 供外部核对每个指标的分母。
    """
    num_episodes: int
    privileged_access: Optional[bool]           # True=oracle/状态基线, False=观测型, None=未标注/混合
    task_success_rate: float
    verified_fault_injection_rate: Optional[float]
    failure_detection_rate: Optional[float]
    recovery_success_rate: Optional[float]
    avg_retry_count: Optional[float]
    recovery_latency_mean_steps: Optional[float]
    recovery_latency_recovered_mean_steps: Optional[float]
    mean_episode_length: float
    mean_cumulative_reward: float
    sum_cumulative_reward: float
    failure_category_counts: Dict[str, int] = field(default_factory=dict)
    fault_type_counts: Dict[str, int] = field(default_factory=dict)
    numerators: Dict[str, int] = field(default_factory=dict)
    denominators: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "num_episodes": self.num_episodes,
            "privileged_access": self.privileged_access,
            "metrics": {
                "task_success_rate": self.task_success_rate,
                "verified_fault_injection_rate": self.verified_fault_injection_rate,
                "failure_detection_rate": self.failure_detection_rate,
                "recovery_success_rate": self.recovery_success_rate,
                "avg_retry_count": self.avg_retry_count,
                "recovery_latency_mean_steps": self.recovery_latency_mean_steps,
                "recovery_latency_recovered_mean_steps": self.recovery_latency_recovered_mean_steps,
                "mean_episode_length": self.mean_episode_length,
                "mean_cumulative_reward": self.mean_cumulative_reward,
                "sum_cumulative_reward": self.sum_cumulative_reward,
                "failure_category_counts": self.failure_category_counts,
                "fault_type_counts": self.fault_type_counts,
            },
            "numerators": self.numerators,
            "denominators": self.denominators,
            "definitions": DEFINITIONS,
        }


def _resolve_privileged(episodes: List[EpisodeResult], explicit: Optional[bool]) -> Optional[bool]:
    """确定该组的 privileged_access: 显式优先, 否则从 episode 推断 (须一致)。"""
    if explicit is not None:
        return bool(explicit)
    flags = {extract_privileged(ep) for ep in episodes}
    flags.discard(None)
    if not flags:
        return None
    if len(flags) > 1:
        raise ValueError(
            f"episode 组内 privileged_access 不一致 {sorted(flags)}; "
            "请先用 split_by_privileged 分组或显式传入 privileged_access"
        )
    return flags.pop()


def compute_recovery_metrics(
    episodes: List[EpisodeResult],
    *,
    privileged_access: Optional[bool] = None,
) -> RecoveryMetrics:
    """计算一组故障注入 episode 的恢复指标 (所有分母显式定义, 见 DEFINITIONS)。

    返回 None 的指标 (如 verified_fault_injection_rate 在无注入尝试时) 表示分母为 0, 无定义。
    """
    eps = list(episodes)
    n = len(eps)
    priv = _resolve_privileged(eps, privileged_access)

    num_success = sum(1 for ep in eps if ep.success)

    # 故障注入
    faults = [extract_fault(ep) for ep in eps]
    num_injection_attempted = sum(1 for f in faults if f is not None)
    num_injected = sum(1 for f in faults if f is not None and f.injected)

    # 真实失败场景 (注入成功且失败真实发生)
    real_failures = [ep for ep in eps if is_valid_injected_failure(ep)]
    num_real_failures = len(real_failures)

    # 检测
    num_detected = sum(1 for ep in real_failures if failure_detected(ep))

    # 恢复 (只在实际注入成功的失败场景上度量)
    injected_failures = real_failures  # 同一定义: injected=True 且 failure_manifested
    num_recovered = sum(
        1 for ep in injected_failures
        if (_normalize_recovery(ep) or {}).get("recovered", False)
    )

    # 重试与延迟 (在『实际发生恢复尝试』的 episode 上度量)
    attempted = [ep for ep in eps if recovery_attempted(ep)]
    num_recovery_attempted = len(attempted)
    if attempted:
        retries = [max(0, (_normalize_recovery(ep)["attempts"] - 1)) for ep in attempted]
        avg_retry = sum(retries) / len(retries)
        latencies = [max(0, ep.steps - _first_failure_or_inject_step(ep)) for ep in attempted]
        latency_mean = sum(latencies) / len(latencies)
        recovered_eps = [ep for ep in attempted
                         if (_normalize_recovery(ep) or {}).get("recovered", False)]
        latency_recovered_mean = (
            sum(max(0, ep.steps - _first_failure_or_inject_step(ep)) for ep in recovered_eps)
            / len(recovered_eps)
            if recovered_eps else None
        )
    else:
        avg_retry = None
        latency_mean = None
        latency_recovered_mean = None

    steps = [ep.steps for ep in eps]
    rewards = [ep.reward for ep in eps]
    mean_steps = sum(steps) / n if n else 0.0
    mean_reward = sum(rewards) / n if n else 0.0

    failure_category_counts = dict(Counter(
        f.type.value for ep in eps for f in ep.failures
    ))
    fault_type_counts = dict(Counter(
        f.fault_type for f in faults if f is not None
    ))

    return RecoveryMetrics(
        num_episodes=n,
        privileged_access=priv,
        task_success_rate=float(num_success / n) if n else 0.0,
        verified_fault_injection_rate=(num_injected / num_injection_attempted) if num_injection_attempted else None,
        failure_detection_rate=(num_detected / num_real_failures) if num_real_failures else None,
        recovery_success_rate=(num_recovered / len(injected_failures)) if injected_failures else None,
        avg_retry_count=avg_retry,
        recovery_latency_mean_steps=latency_mean,
        recovery_latency_recovered_mean_steps=latency_recovered_mean,
        mean_episode_length=float(mean_steps),
        mean_cumulative_reward=float(mean_reward),
        sum_cumulative_reward=float(sum(rewards)),
        failure_category_counts=failure_category_counts,
        fault_type_counts=fault_type_counts,
        numerators={
            "num_success": num_success,
            "num_injected": num_injected,
            "num_injection_attempted": num_injection_attempted,
            "num_detected": num_detected,
            "num_real_failures": num_real_failures,
            "num_recovered": num_recovered,
            "num_injected_failure_scenarios": len(injected_failures),
            "num_recovery_attempted": num_recovery_attempted,
        },
        denominators={
            "task_success_rate": n,
            "verified_fault_injection_rate": num_injection_attempted,
            "failure_detection_rate": num_real_failures,
            "recovery_success_rate": len(injected_failures),
            "avg_retry_count": num_recovery_attempted,
            "recovery_latency_mean_steps": num_recovery_attempted,
            "mean_episode_length": n,
            "mean_cumulative_reward": n,
        },
    )


def split_by_privileged(episodes: List[EpisodeResult]) -> Dict[Optional[bool], List[EpisodeResult]]:
    """按 privileged_access 分组: True=oracle/状态基线, False=观测型策略, None=未标注。"""
    groups: Dict[Optional[bool], List[EpisodeResult]] = {}
    for ep in episodes:
        groups.setdefault(extract_privileged(ep), []).append(ep)
    return groups
