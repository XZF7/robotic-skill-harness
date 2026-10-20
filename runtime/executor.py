"""SkillExecutor: 执行一个 RobotSkill 直到 is_done 或 env 提前结束。"""
from __future__ import annotations

from typing import Callable, Optional

from interfaces import (
    EpisodeInfo,
    FailureEvent,
    FailureType,
    RobotObservation,
    RobotSkill,
    SkillResult,
    SkillStatus,
    SimulatorAdapter,
)


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


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
        # 最近一次 adapter.step 产生的 EpisodeInfo (供 harness 取最终真值)。
        self.last_episode: Optional[EpisodeInfo] = None
        # 累计 episode reward (所有步奖励之和，供评估层取累计值而非末步值)。
        self.total_reward: float = 0.0

    def execute(
        self,
        skill: RobotSkill,
        obs: RobotObservation,
        *,
        max_steps: Optional[int] = None,
        adapter: Optional[SimulatorAdapter] = None,
        abort_check: Optional[Callable[[], bool]] = None,
        step_hook: Optional[Callable[[RobotObservation], Optional[object]]] = None,
        reset_skill: bool = True,
    ) -> SkillResult:
        budget = max_steps if max_steps is not None else self.default_max_steps
        if budget is not None and budget < 0:
            raise ValueError("max_steps 不能为负")

        self.last_episode = None
        self.total_reward = 0.0
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
                self.last_episode = step_result.episode
                self.total_reward += float(step_result.episode.reward)
                if step_result.episode.terminated or step_result.episode.truncated:
                    break

            # 学习预判 (M4-B): 每步预测失败风险, 高风险提前中止本 attempt。
            # step_hook 只观察、不发指令; 返回非 None 即「提前重试」信号。
            if step_hook is not None:
                trigger = step_hook(current)
                if trigger is not None:
                    return self._predictive_result(skill, steps, current, trigger)

        # 技能完成 或 env 提前结束 -> 权威结果来自技能自身 (executor 不看 truth)。
        return skill.result(current)

    def _predictive_result(
        self,
        skill: RobotSkill,
        steps: int,
        obs: RobotObservation,
        trigger: object,
    ) -> SkillResult:
        """把学习预判触发转成一次提前中止的 FAILURE (FailureType.UNKNOWN + predictive 标记)。

        交给 RecoveryRunner 走规则恢复 (RETRY/ABORT), 不绕过 max_retries。
        """
        detail = {"predictive": True}
        for key in ("prob", "confidence", "threshold", "step"):
            val = getattr(trigger, key, None)
            if val is not None:
                detail[f"predict_{key}"] = _to_float(val)
        failure = FailureEvent(
            type=FailureType.UNKNOWN,
            step=steps,
            timestamp=_to_float(getattr(obs, "timestamp", 0.0)),
            detail=detail,
        )
        return SkillResult(
            status=SkillStatus.FAILURE,
            skill_id=skill.skill_id,
            steps=steps,
            message="predictive recovery triggered",
            failure=failure,
        )
