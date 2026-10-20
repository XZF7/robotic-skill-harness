"""runtime/harness.py — TaskRunner: 任务驱动 harness。

把真实仿真任务经 executor 跑通:
    Task -> SkillExecutor -> PickCubeSkill -> ManiSkill -> SkillResult -> ExecutionLogger

单一权威结果: 只返回 executor 产出的 SkillResult，绝不二次判定。
同时捕获最终 EpisodeInfo (env 特权真值) 与 TaskState，供下游独立核对
「技能自报结果」与「观测结果」是否一致。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from interfaces import (
    EpisodeInfo,
    EpisodeResult,
    RobotSkill,
    SkillResult,
    SimulatorAdapter,
)

from .executor import SkillExecutor
from .logger import ExecutionLogger
from .registry import SkillRegistry
from .state_machine import TaskState, TaskStateMachine


@dataclass
class TaskRunResult:
    """一次任务执行的完整产出。

    - skill_result: 唯一权威 SkillResult (来自 executor，不得二次判定)。
    - episode: 最终 EpisodeInfo (env 任务成功真值 terminated/truncated/success)。
    - task_state: TaskState (由 skill_result 经状态机推导)。
    """

    skill_result: SkillResult
    episode: Optional[EpisodeInfo]
    task_state: TaskState

    def to_episode_result(self, task_id: str) -> EpisodeResult:
        """构造供 Agent D 评估框架消费的 EpisodeResult(source="sim")。

        success 字段 = env 特权真值 (episode.success)，与 skill_result.status 独立，
        供下游核对「技能自报结果」与「观测结果」是否一致。
        """
        sr = self.skill_result
        episode = self.episode
        success = bool(episode.success) if episode is not None else False
        reward = float(episode.reward) if episode is not None else 0.0
        metrics = {
            "skill_status": sr.status.value,
            "skill_steps": sr.steps,
            "task_state": self.task_state.value,
        }
        if episode is not None:
            metrics["terminated"] = bool(episode.terminated)
            metrics["truncated"] = bool(episode.truncated)
        return EpisodeResult(
            task_id=task_id,
            success=success,
            steps=sr.steps,
            reward=reward,
            skill_results=[sr],
            failures=[sr.failure] if sr.failure is not None else [],
            metrics=metrics,
            source="sim",
        )


class TaskRunner:
    """组装 adapter + skill + registry + executor + state machine + logger。

    职责边界: 只负责「装配 + 编排 + 透传」；成败判定完全交给 executor
    (其权威结果来自 skill.result()，TIMEOUT/ABORTED/异常由 executor 产生)。
    harness 不读 truth 二次判定、不重算 SkillResult。
    """

    def __init__(
        self,
        adapter: SimulatorAdapter,
        skill: RobotSkill,
        *,
        task_id: str,
        executor: Optional[SkillExecutor] = None,
        registry: Optional[SkillRegistry] = None,
        state_machine: Optional[TaskStateMachine] = None,
        logger: Optional[ExecutionLogger] = None,
        default_max_steps: Optional[int] = None,
    ) -> None:
        self.adapter = adapter
        self.skill = skill
        self.task_id = task_id
        self.registry = registry if registry is not None else SkillRegistry()
        if skill.skill_id not in self.registry:
            self.registry.register(skill)
        self.executor = (
            executor if executor is not None
            else SkillExecutor(default_max_steps=default_max_steps)
        )
        self.state_machine = (
            state_machine if state_machine is not None else TaskStateMachine()
        )
        self.logger = logger if logger is not None else ExecutionLogger()

    def run(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        max_steps: Optional[int] = None,
        abort_check: Optional[Callable[[], bool]] = None,
    ) -> TaskRunResult:
        """执行一个 episode，返回 TaskRunResult。

        执行路径: registry 查技能 -> executor (reset + 循环 act/step) ->
                  executor 产 SkillResult (唯一权威) -> 状态机推进 -> logger 落一条 trace。
        """
        self.state_machine.reset()
        self.state_machine.start()

        obs = self.adapter.reset(seed=seed, options=options)

        # 经 registry 取技能 (单一来源)，避免绕过注册表。
        skill = self.registry.lookup(self.skill.skill_id)

        result = self.executor.execute(
            skill,
            obs,
            max_steps=max_steps,
            adapter=self.adapter,
            abort_check=abort_check,
        )

        task_state = self.state_machine.apply_skill_result(result)
        episode = self.executor.last_episode

        self.logger.log_result(
            result,
            extra={
                "task_id": self.task_id,
                "task_state": task_state.value,
                "seed": seed,
                "terminated": bool(episode.terminated) if episode is not None else None,
                "truncated": bool(episode.truncated) if episode is not None else None,
                "env_success": bool(episode.success) if episode is not None else None,
            },
        )

        return TaskRunResult(
            skill_result=result,
            episode=episode,
            task_state=task_state,
        )

    def close(self) -> None:
        """释放 adapter 资源。"""
        try:
            self.adapter.close()
        except Exception:
            pass
        self.logger.close()
