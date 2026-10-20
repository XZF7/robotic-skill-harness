"""SkillExecutor: 执行一个 RobotSkill 直到 is_done 或超时。"""
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


def _failure_type_from_obs(obs: RobotObservation) -> FailureType:
    """从观测的特权真值中尽量提取失败类型；缺省 UNKNOWN。"""
    try:
        contact = getattr(obs.truth, "contact", None) or {}
        raw = contact.get("failure_type")
        if isinstance(raw, FailureType):
            return raw
        if isinstance(raw, str):
            return FailureType(raw)
    except (ValueError, AttributeError):
        pass
    return FailureType.UNKNOWN


class SkillExecutor:
    """执行技能：reset → 循环 act(+adapter.step) → 判定结果。

    终止语义:
    - skill.is_done 为 True 时: 依据最终观测 truth.success 判 SUCCESS/FAILURE。
    - 步数预算耗尽 (steps >= max_steps) 仍未 done: TIMEOUT。
    - abort_check 回调返回 True: ABORTED。
    - skill.act 抛异常: FAILURE (FailureType.MOTION_FAIL)。

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
            except Exception as exc:  # noqa: BLE001 — 运行时兜底
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

        # skill.is_done 为 True：以特权真值 success 判定成功/失败。
        if current.truth.success:
            return SkillResult(
                status=SkillStatus.SUCCESS,
                skill_id=skill.skill_id,
                steps=steps,
                message="skill goal reached",
            )
        failure = FailureEvent(
            type=_failure_type_from_obs(current),
            step=steps,
            timestamp=float(current.timestamp),
            detail={"truth_success": False},
        )
        return SkillResult(
            status=SkillStatus.FAILURE,
            skill_id=skill.skill_id,
            steps=steps,
            message="skill completed without success",
            failure=failure,
        )
