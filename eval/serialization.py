"""EpisodeResult / FaultRecord / RecoveryResult <-> dict/JSON 序列化与反序列化 (可回读 round-trip)。

约定:
- 枚举 (SkillStatus / FailureType / RecoveryAction) 序列化为字符串值, 反序列化时还原为枚举实例。
- numpy 标量/数组递归归一化为原生 Python 类型, 保证 json.dumps 不报错。
- source 字段原样保留 (区分 mock / sim)。
- FaultRecord / RecoveryResult / RecoveryDecision / SkillResult / FailureEvent 可递归序列化,
  因此它们可安全地嵌入 EpisodeResult.metrics (故障注入/恢复信息)。
"""
from __future__ import annotations

import json
from typing import Any, List

import numpy as np

from interfaces import (
    EpisodeResult, SkillResult, SkillStatus, FailureEvent, FailureType,
    FaultRecord, RecoveryResult, RecoveryDecision, RecoveryAction,
)


def _jsonable(value: Any) -> Any:
    """递归归一化: numpy 标量/数组/枚举/契约 dataclass -> 原生 Python 类型。"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (SkillStatus, FailureType, RecoveryAction)):
        return value.value
    if isinstance(value, FaultRecord):
        return fault_record_to_dict(value)
    if isinstance(value, RecoveryResult):
        return recovery_result_to_dict(value)
    if isinstance(value, RecoveryDecision):
        return recovery_decision_to_dict(value)
    if isinstance(value, SkillResult):
        return skill_result_to_dict(value)
    if isinstance(value, FailureEvent):
        return failure_event_to_dict(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


# ---- FailureEvent ----------------------------------------------------------

def _failure_to_dict(f: FailureEvent) -> dict:
    return {
        "type": f.type.value,
        "step": int(f.step),
        "timestamp": float(f.timestamp),
        "detail": _jsonable(f.detail),
    }


def _failure_from_dict(d: dict) -> FailureEvent:
    return FailureEvent(
        type=FailureType(d["type"]),
        step=int(d["step"]),
        timestamp=float(d["timestamp"]),
        detail=dict(d.get("detail", {})),
    )


def failure_event_to_dict(f: FailureEvent) -> dict:
    return _failure_to_dict(f)


def failure_event_from_dict(d: dict) -> FailureEvent:
    return _failure_from_dict(d)


# ---- SkillResult -----------------------------------------------------------

def _skill_to_dict(s: SkillResult) -> dict:
    return {
        "status": s.status.value,
        "skill_id": s.skill_id,
        "steps": int(s.steps),
        "message": s.message,
        "failure": _failure_to_dict(s.failure) if s.failure is not None else None,
    }


def _skill_from_dict(d: dict) -> SkillResult:
    failure = d.get("failure")
    return SkillResult(
        status=SkillStatus(d["status"]),
        skill_id=d["skill_id"],
        steps=int(d["steps"]),
        message=d.get("message", ""),
        failure=_failure_from_dict(failure) if failure is not None else None,
    )


def skill_result_to_dict(s: SkillResult) -> dict:
    return _skill_to_dict(s)


def skill_result_from_dict(d: dict) -> SkillResult:
    return _skill_from_dict(d)


# ---- RecoveryDecision ------------------------------------------------------

def recovery_decision_to_dict(d: RecoveryDecision) -> dict:
    return {
        "action": d.action.value,
        "retry_count": int(d.retry_count),
        "reason": d.reason,
        "detail": _jsonable(d.detail),
    }


def recovery_decision_from_dict(d: dict) -> RecoveryDecision:
    return RecoveryDecision(
        action=RecoveryAction(d["action"]),
        retry_count=int(d.get("retry_count", 0)),
        reason=d.get("reason", ""),
        detail=dict(d.get("detail", {})),
    )


# ---- RecoveryResult --------------------------------------------------------

def recovery_result_to_dict(r: RecoveryResult) -> dict:
    return {
        "attempts": int(r.attempts),
        "final_skill_result": _skill_to_dict(r.final_skill_result),
        "decisions": [recovery_decision_to_dict(d) for d in r.decisions],
        "recovered": bool(r.recovered),
        "aborted": bool(r.aborted),
    }


def recovery_result_from_dict(d: dict) -> RecoveryResult:
    if "final_skill_result" not in d:
        raise KeyError("RecoveryResult dict 缺少必填字段 'final_skill_result'")
    return RecoveryResult(
        attempts=int(d["attempts"]),
        final_skill_result=_skill_from_dict(d["final_skill_result"]),
        decisions=[recovery_decision_from_dict(x) for x in d.get("decisions", [])],
        recovered=bool(d.get("recovered", False)),
        aborted=bool(d.get("aborted", False)),
    )


# ---- FaultRecord -----------------------------------------------------------

def fault_record_to_dict(f: FaultRecord) -> dict:
    return {
        "fault_type": f.fault_type,
        "injected": bool(f.injected),
        "step": int(f.step),
        "params": _jsonable(f.params),
        "detail": _jsonable(f.detail),
    }


def fault_record_from_dict(d: dict) -> FaultRecord:
    return FaultRecord(
        fault_type=d["fault_type"],
        injected=bool(d.get("injected", False)),
        step=int(d.get("step", 0)),
        params=dict(d.get("params", {})),
        detail=dict(d.get("detail", {})),
    )


# ---- EpisodeResult ---------------------------------------------------------

def episode_to_dict(ep: EpisodeResult) -> dict:
    return {
        "task_id": ep.task_id,
        "success": bool(ep.success),
        "steps": int(ep.steps),
        "reward": float(ep.reward),
        "skill_results": [_skill_to_dict(s) for s in ep.skill_results],
        "failures": [_failure_to_dict(f) for f in ep.failures],
        "metrics": _jsonable(ep.metrics),
        "source": ep.source,
    }


def episode_from_dict(d: dict) -> EpisodeResult:
    return EpisodeResult(
        task_id=d["task_id"],
        success=bool(d["success"]),
        steps=int(d["steps"]),
        reward=float(d["reward"]),
        skill_results=[_skill_from_dict(s) for s in d.get("skill_results", [])],
        failures=[_failure_from_dict(f) for f in d.get("failures", [])],
        metrics=dict(d.get("metrics", {})),
        source=d.get("source", "mock"),
    )


def episode_to_json(ep: EpisodeResult, **kw) -> str:
    return json.dumps(episode_to_dict(ep), **kw)


def episode_from_json(s: str) -> EpisodeResult:
    return episode_from_dict(json.loads(s))


# ---- 批量 (list[EpisodeResult]) -------------------------------------------

def episodes_to_dicts(eps: List[EpisodeResult]) -> List[dict]:
    return [episode_to_dict(ep) for ep in eps]


def episodes_from_dicts(ds: List[dict]) -> List[EpisodeResult]:
    return [episode_from_dict(d) for d in ds]


def episodes_to_json(eps: List[EpisodeResult], **kw) -> str:
    return json.dumps(episodes_to_dicts(eps), **kw)


def episodes_from_json(s: str) -> List[EpisodeResult]:
    return episodes_from_dicts(json.loads(s))
