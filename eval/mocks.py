"""开发期 mock EpisodeResult 生成器 (合成数据, source='mock')。

⚠️ 仅用于开发期测试 eval 管线 (序列化/指标/对比)。
严禁在集成真实仿真前把 mock 结果当作 benchmark 表现上报 ——
上报前必须通过 `eval.source.require_sim` 守卫。
"""
from __future__ import annotations

import random
from typing import List, Optional

from interfaces import EpisodeResult, SkillResult, SkillStatus, FailureEvent, FailureType


def make_mock_episode(
    task_id: str,
    *,
    steps: Optional[int] = None,
    success: Optional[bool] = None,
    reward: Optional[float] = None,
    seed: Optional[int] = None,
    source: str = "mock",
    skill_id: str = "pick",
    metrics: Optional[dict] = None,
) -> EpisodeResult:
    """合成一个 episode (确定性, 给定 seed)。"""
    rng = random.Random(seed)
    if steps is None:
        steps = rng.randint(20, 200)
    if success is None:
        success = rng.random() < 0.5
    if reward is None:
        reward = 1.0 if success else round(rng.uniform(0.0, 0.3), 4)

    status = SkillStatus.SUCCESS if success else SkillStatus.FAILURE
    skill_results = [SkillResult(status=status, skill_id=skill_id, steps=steps)]
    failures: List[FailureEvent] = []
    if not success:
        failures.append(FailureEvent(type=FailureType.GRIP_FAIL, step=steps, timestamp=float(steps) * 0.05))

    return EpisodeResult(
        task_id=task_id,
        success=success,
        steps=steps,
        reward=reward,
        skill_results=skill_results,
        failures=failures,
        metrics=metrics if metrics is not None else {"seed": seed},
        source=source,
    )


def make_mock_episodes(
    task_id: str,
    n: int,
    *,
    seed: int = 0,
    success_rate: float = 0.5,
    source: str = "mock",
    skill_id: str = "pick",
) -> List[EpisodeResult]:
    """批量合成 n 个 episode, 成功比例约 success_rate (确定性, 给定 seed)。"""
    if n < 0:
        raise ValueError(f"n 必须 >= 0, 得到 {n}")
    if not 0.0 <= success_rate <= 1.0:
        raise ValueError(f"success_rate 必须在 [0,1], 得到 {success_rate}")
    rng = random.Random(seed)
    out: List[EpisodeResult] = []
    for i in range(n):
        success = rng.random() < success_rate
        steps = rng.randint(20, 200)
        reward = 1.0 if success else round(rng.uniform(0.0, 0.3), 4)
        status = SkillStatus.SUCCESS if success else SkillStatus.FAILURE
        failures = (
            [FailureEvent(type=FailureType.GRIP_FAIL, step=steps, timestamp=float(steps) * 0.05)]
            if not success else []
        )
        out.append(EpisodeResult(
            task_id=task_id,
            success=success,
            steps=steps,
            reward=reward,
            skill_results=[SkillResult(status=status, skill_id=skill_id, steps=steps)],
            failures=failures,
            metrics={"seed": seed, "index": i},
            source=source,
        ))
    return out


# ---- 故障注入 mock episode (M3-D smoke test / 种子扫描用, source='mock') ----

def make_mock_fault_episode(
    task_id: str,
    *,
    seed: int,
    fault_type: str,
    injected: bool = True,
    success: bool = False,
    detected: bool = True,
    recovered: bool = False,
    attempts: int = 1,
    steps: int = 80,
    reward: float = 0.0,
    baseline: str = "b0",
    privileged_access: bool = False,
    fault_params: Optional[dict] = None,
    source: str = "mock",
    failure_type: FailureType = FailureType.GRIP_FAIL,
    failure_step: Optional[int] = None,
) -> EpisodeResult:
    """合成一个带故障注入信息的 episode (确定性, 由调用方给出全部关键标志)。

    metrics 里挂规范键: fault / recovery / privileged_access / baseline / seed / detected。
    用于 smoke test 与可配置种子扫描; 严禁冒充真实 benchmark (source='mock', require_sim 会拦截)。
    """
    from interfaces import FaultRecord, RecoveryResult, RecoveryDecision, RecoveryAction

    fault = FaultRecord(
        fault_type=fault_type,
        injected=injected,
        step=0 if failure_step is None else max(0, failure_step - 1),
        params=dict(fault_params or {}),
    )
    final_status = SkillStatus.SUCCESS if success else SkillStatus.FAILURE
    final_sr = SkillResult(status=final_status, skill_id="pick", steps=steps)
    decisions = []
    if attempts > 1:
        for i in range(1, attempts):
            decisions.append(RecoveryDecision(
                action=RecoveryAction.RETRY, retry_count=i, reason=failure_type.value,
            ))
    recovery = RecoveryResult(
        attempts=attempts,
        final_skill_result=final_sr,
        decisions=decisions,
        recovered=recovered,
        aborted=(not recovered and attempts > 1),
    )

    failures: List[FailureEvent] = []
    if not success or attempts > 1 or failure_step is not None:
        failures.append(FailureEvent(
            type=failure_type,
            step=failure_step if failure_step is not None else steps // 2,
            timestamp=float(failure_step if failure_step is not None else steps // 2) * 0.05,
        ))

    from eval.fault_injection import (
        attach_fault, attach_recovery, attach_privileged, attach_baseline,
    )
    ep = EpisodeResult(
        task_id=task_id,
        success=success,
        steps=steps,
        reward=reward,
        skill_results=[final_sr],
        failures=failures,
        metrics={"seed": seed, "detected": detected},
        source=source,
    )
    attach_fault(ep, fault)
    attach_recovery(ep, recovery)
    attach_privileged(ep, privileged_access)
    attach_baseline(ep, baseline)
    return ep


def make_mock_fault_episodes(
    task_id: str,
    fault_type: str,
    seeds: List[int],
    *,
    baseline: str = "b0",
    injected: bool = True,
    recovered: bool = False,
    attempts: int = 1,
    success: bool = False,
    detected: bool = False,
    privileged_access: bool = False,
    fault_params: Optional[dict] = None,
    source: str = "mock",
) -> List[EpisodeResult]:
    """批量合成 matched 的故障注入 episode (给定 seed 列表)。"""
    out: List[EpisodeResult] = []
    for seed in seeds:
        out.append(make_mock_fault_episode(
            task_id=task_id,
            seed=seed,
            fault_type=fault_type,
            injected=injected,
            success=success,
            detected=detected,
            recovered=recovered,
            attempts=attempts,
            baseline=baseline,
            privileged_access=privileged_access,
            fault_params=fault_params,
            source=source,
        ))
    return out
