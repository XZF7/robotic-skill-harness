"""SkillExecutor: 执行一个 RobotSkill 直到 is_done 或 env 提前结束。"""
from __future__ import annotations

from typing import Callable, Optional

from interfaces import (
    FailureEvent,
    FailureType,
    RobotObservation,
    RobotSkill,
    SkillResult,
    SkillStatus,
    SimulatorAdapter,
)


class SkillExecutor:
    """执行技能：reset -> 循环 act(+adapter.step) -> 由技能自身 result() 定成败。

    单一权威结果 (SkillResult):
    - 技能完成 (is_done) 或 env 提前终止/截断 -> 返回 skill.result(final_obs)。
      技能自报 SUCCESS/FAILURE；executor 不看 truth.success。
    - 步数预算耗尽 -> TIMEOUT (executor 产生)。
    - abort_check 返回 True -> ABORTED (executor 产生)。
    - skill.act 抛异常 -> FAILURE (FailureType.MOTION_FAIL, executor 产生)。

    mock 模式: 不传 adapter 时观测不推进，技能自行维护内部进度 (is_done)。
    """

    def __init__(self, default_max_steps: Optional[int] = None) -> None:
        self.default_max_steps = default_max_steps

    def execute(
        self,
        skill: RobotSkill,
        obs: RobotObservation,
        *,
        max_steps: Optional[int] = None,
        adapter: Optional[SimulatorAdapter] = None,
        abort_check: Optional[Callable[[], bool]] = None,
        reset_skill: bool = True,
    ) -> SkillResult:
        budget = max_steps if max_steps is not None else self.default_max_steps
        if budget is not None and budget < 0:
            raise ValueError("max_steps 不能为负")

        if reset_skill:
            skill.reset(obs)
        current: RobotObservation = obs
        steps = 0

        while not skill.is_done:
            if abort_check is not None and abort_check():
                return SkillResult(
                    status=SkillStatus.ABORTED,
                    skill_id=skill.skill_id,
                    steps=steps,
                    message="aborted by external signal",
                )
            if budget is not None and steps >= budget:
                failure = FailureEvent(
                    type=FailureType.TIMEOUT,
                    step=steps,
                    timestamp=float(current.timestamp),
                    detail={"budget": budget},
                )
                return SkillResult(
                    status=SkillStatus.TIMEOUT,
                    skill_id=skill.skill_id,
                    steps=steps,
                    message=f"step budget {budget} exhausted",
                    failure=failure,
                )
            try:
                action = skill.act(current)
            except Exception as exc:  # noqa: BLE001 - 运行时兜底
                failure = FailureEvent(
                    type=FailureType.MOTION_FAIL,
                    step=steps,
                    timestamp=float(current.timestamp),
                    detail={"exception": type(exc).__name__, "message": str(exc)},
                )
                return SkillResult(
                    status=SkillStatus.FAILURE,
                    skill_id=skill.skill_id,
                    steps=steps,
                    message=f"act raised {type(exc).__name__}",
                    failure=failure,
                )
            steps += 1
            if adapter is not None:
                step_result = adapter.step(action)
                current = step_result.observation
                if step_result.episode.terminated or step_result.episode.truncated:
                    break

        # 技能完成 或 env 提前结束 -> 权威结果来自技能自身 (executor 不看 truth)。
        return skill.result(current)
