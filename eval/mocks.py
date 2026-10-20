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
