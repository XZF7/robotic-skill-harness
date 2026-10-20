"""runtime/recovery.py — 规则式失败恢复 (M3-B)。

RecoveryManager    : 依据检测到的失败产出 RecoveryDecision (RETRY/REOBSERVE/ABORT)，
                     有界重试、确定性、不依赖 LLM。
RecoveryStateMachine : 恢复状态转移, 记录恢复决策序列。
RecordingAdapter    : 装饰 SimulatorAdapter, 记录每步观测 (供失败检测/恢复), 不改 env。
RecoveryRunner      : 同一 episode 内编排「执行 -> 检测 -> 恢复决策 -> 重试」全链路,
                     产出唯一最终结果 RecoveryResult。

关键约束:
- 恢复在同一 episode 内 (只 reset 一次 env); 每次重试仅 skill.reset, 不 reset env。
- terminated/truncated 后不再重试 (env 已结束/无效)。
- 每次技能尝试产恰好一个权威 SkillResult; 多次尝试汇总为唯一 RecoveryResult。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from interfaces import (
    EpisodeInfo,
    FailureEvent,
    FailurePredictor,
    FailureType,
    RecoveryAction,
    RecoveryDecision,
    RecoveryResult,
    RobotObservation,
    RobotSkill,
    SimulatorAdapter,
    SkillResult,
    SkillStatus,
    StepResult,
)

from .errors import InvalidTransitionError
from .executor import SkillExecutor
from .failure_detector import FailureDetector
from .logger import ExecutionLogger, make_record
from .predictive import PredictiveGuard, RecoveryMode
from .registry import SkillRegistry


# --------------------------------------------------------------------------
# RecoveryManager
# --------------------------------------------------------------------------
@dataclass
class RecoveryManagerConfig:
    """重试策略配置 (全部可覆盖)。"""

    # 允许的最大重试次数 (首次尝试不计入重试)。总尝试次数 = 1 + max_retries。
    max_retries: int = 2
    # 触发 REOBSERVE 的失败类型 (重新观测后重试)。
    reobserve_on: Tuple[FailureType, ...] = (FailureType.INVALID_OBS,)
    # 触发立即 ABORT 的失败类型 (无恢复价值)。
    abort_on: Tuple[FailureType, ...] = (FailureType.RECOVERY_EXHAUSTED,)
    # 其余失败类型 -> RETRY。


class RecoveryManager:
    """规则式恢复决策器。确定性: 相同输入 -> 相同决策。"""

    def __init__(
        self,
        max_retries: Optional[int] = None,
        config: Optional[RecoveryManagerConfig] = None,
    ) -> None:
        self.config = config if config is not None else RecoveryManagerConfig()
        if max_retries is not None:
            if max_retries < 0:
                raise ValueError("max_retries 不能为负")
            self.config.max_retries = max_retries
        if self.config.max_retries < 0:
            raise ValueError("max_retries 不能为负")

    @property
    def max_retries(self) -> int:
        return self.config.max_retries

    def action_for(self, failure_type: FailureType) -> RecoveryAction:
        if failure_type in self.config.abort_on:
            return RecoveryAction.ABORT
        if failure_type in self.config.reobserve_on:
            return RecoveryAction.REOBSERVE
        return RecoveryAction.RETRY

    def decide(
        self,
        failure: FailureEvent,
        retry_count: int,
        *,
        env_ended: bool = False,
    ) -> RecoveryDecision:
        """依据检测到的失败产出恢复决策。

        retry_count: 当前已重试次数 (0 表示首次失败后的首次决策)。
        env_ended:   episode 已 terminated/truncated -> 不再重试 (同 episode 内无法恢复)。
        """
        detail: Dict = {
            "failure_type": failure.type.value,
            "max_retries": self.max_retries,
        }
        if env_ended:
            return RecoveryDecision(
                action=RecoveryAction.ABORT,
                retry_count=retry_count,
                reason="episode ended (terminated/truncated); no retry within episode",
                detail=dict(detail, episode_ended=True),
            )
        if retry_count >= self.max_retries:
            detail["exhausted"] = True
            return RecoveryDecision(
                action=RecoveryAction.ABORT,
                retry_count=retry_count,
                reason="recovery exhausted (retry budget reached)",
                detail=detail,
            )
        action = self.action_for(failure.type)
        reason = {
            RecoveryAction.RETRY: "retry skill within same episode",
            RecoveryAction.REOBSERVE: "re-observe scene then retry",
            RecoveryAction.ABORT: "unrecoverable failure",
        }[action]
        return RecoveryDecision(
            action=action, retry_count=retry_count, reason=reason, detail=detail,
        )


# --------------------------------------------------------------------------
# RecoveryStateMachine
# --------------------------------------------------------------------------
class RecoveryState(str, Enum):
    IDLE = "idle"              # 未开始
    RUNNING = "running"        # 某次技能尝试执行中
    RECOVERING = "recovering"  # 检测到失败, 准备/执行恢复
    SUCCEEDED = "succeeded"    # 任务成功 (含恢复后成功)
    ABORTED = "aborted"        # 无法恢复, 安全中止


_RECOVERY_TERMINAL = frozenset({RecoveryState.SUCCEEDED, RecoveryState.ABORTED})

_RECOVERY_TRANSITIONS = {
    RecoveryState.IDLE: {RecoveryState.RUNNING},
    RecoveryState.RUNNING: {RecoveryState.SUCCEEDED, RecoveryState.RECOVERING, RecoveryState.ABORTED},
    RecoveryState.RECOVERING: {RecoveryState.RUNNING, RecoveryState.ABORTED},
    RecoveryState.SUCCEEDED: set(),
    RecoveryState.ABORTED: set(),
}


class RecoveryStateMachine:
    """恢复状态机: 记录状态转移与恢复决策序列。"""

    def __init__(self, initial: RecoveryState = RecoveryState.IDLE) -> None:
        self._state: RecoveryState = initial
        self._history: List[Tuple[RecoveryState, RecoveryState]] = []
        self._decisions: List[RecoveryDecision] = []

    @property
    def state(self) -> RecoveryState:
        return self._state

    @property
    def history(self) -> List[Tuple[RecoveryState, RecoveryState]]:
        return list(self._history)

    @property
    def decisions(self) -> List[RecoveryDecision]:
        return list(self._decisions)

    def is_terminal(self) -> bool:
        return self._state in _RECOVERY_TERMINAL

    def can_transition(self, target: RecoveryState) -> bool:
        return target in _RECOVERY_TRANSITIONS[self._state]

    def transition_to(self, target: RecoveryState) -> RecoveryState:
        if not isinstance(target, RecoveryState):
            raise TypeError(f"target 必须是 RecoveryState, 得到 {type(target).__name__}")
        if not self.can_transition(target):
            raise InvalidTransitionError(
                f"非法恢复转移: {self._state.value} -> {target.value}"
            )
        self._history.append((self._state, target))
        self._state = target
        return target

    def start(self) -> RecoveryState:
        return self.transition_to(RecoveryState.RUNNING)

    def mark_failed(self, decision: Optional[RecoveryDecision] = None) -> RecoveryState:
        """一次尝试失败 -> RECOVERING (若 decision 非空则记录)。"""
        if decision is not None:
            self._decisions.append(decision)
        return self.transition_to(RecoveryState.RECOVERING)

    def mark_retrying(self) -> RecoveryState:
        """从 RECOVERING 回到 RUNNING (发起下一次尝试)。"""
        return self.transition_to(RecoveryState.RUNNING)

    def mark_succeeded(self) -> RecoveryState:
        return self.transition_to(RecoveryState.SUCCEEDED)

    def mark_aborted(self, decision: Optional[RecoveryDecision] = None) -> RecoveryState:
        if decision is not None:
            self._decisions.append(decision)
        return self.transition_to(RecoveryState.ABORTED)

    def reset(self) -> None:
        self._state = RecoveryState.IDLE
        self._history = []
        self._decisions = []


# --------------------------------------------------------------------------
# RecordingAdapter: 记录每步观测, 不改 env。
# --------------------------------------------------------------------------
class RecordingAdapter(SimulatorAdapter):
    """装饰 SimulatorAdapter: 记录每步观测历史与最近观测。

    不改动被装饰对象的行为; 仅供失败检测 (obs.policy) 与恢复编排读取。
    """

    def __init__(self, inner: SimulatorAdapter) -> None:
        self._inner = inner
        self.history: List[RobotObservation] = []
        self.last_obs: Optional[RobotObservation] = None

    def reset(self, *, seed=None, options=None) -> RobotObservation:
        self.history = []
        obs = self._inner.reset(seed=seed, options=options)
        self.last_obs = obs
        self.history.append(obs)
        return obs

    def step(self, action: np.ndarray) -> StepResult:
        sr = self._inner.step(action)
        self.last_obs = sr.observation
        self.history.append(sr.observation)
        return sr

    def close(self) -> None:
        self._inner.close()

    @property
    def num_envs(self) -> int:
        return self._inner.num_envs

    @property
    def action_space(self) -> object:
        return self._inner.action_space

    @property
    def max_episode_steps(self) -> int:
        return self._inner.max_episode_steps


# --------------------------------------------------------------------------
# RecoveryRunner: 同一 episode 内执行 + 检测 + 恢复 全链路。
# --------------------------------------------------------------------------
@dataclass
class RecoveryRunResult:
    """恢复编排的完整产出 (含权威 RecoveryResult 与逐次尝试记录)。"""

    recovery: RecoveryResult
    attempt_results: List[SkillResult] = field(default_factory=list)
    task_state: str = ""
    episode: Optional[EpisodeInfo] = None


class RecoveryRunner:
    """在单一 episode 内编排技能执行与失败恢复。

    流程 (不 reset env 于重试之间):
        adapter.reset (一次) -> [executor.execute -> 检测 -> 决策 -> 重试]* -> RecoveryResult
    每次尝试一个权威 SkillResult (来自 executor), 逐次落 trace;
    最终汇总为唯一 RecoveryResult。
    """

    def __init__(
        self,
        adapter: SimulatorAdapter,
        skill: RobotSkill,
        *,
        task_id: str,
        executor: Optional[SkillExecutor] = None,
        detector: Optional[FailureDetector] = None,
        manager: Optional[RecoveryManager] = None,
        logger: Optional[ExecutionLogger] = None,
        registry: Optional[SkillRegistry] = None,
        recovery_sm: Optional[RecoveryStateMachine] = None,
        default_max_steps: Optional[int] = None,
        predictor: Optional[FailurePredictor] = None,
        recovery_mode: str = "b1",
        predictor_threshold: Optional[float] = None,
        predictor_min_confidence: float = 0.0,
    ) -> None:
        self.task_id = task_id
        self.recording = RecordingAdapter(adapter)
        self.skill = skill
        self.registry = registry if registry is not None else SkillRegistry()
        if skill.skill_id not in self.registry:
            self.registry.register(skill)
        self.executor = (
            executor if executor is not None
            else SkillExecutor(default_max_steps=default_max_steps)
        )
        self.detector = detector if detector is not None else FailureDetector()
        self.manager = manager if manager is not None else RecoveryManager()
        self.logger = logger if logger is not None else ExecutionLogger()
        self.recovery_sm = (
            recovery_sm if recovery_sm is not None else RecoveryStateMachine()
        )
        self.attempt_results: List[SkillResult] = []

        # 学习预判 (M4-B): b1 = 纯规则 (guard 关闭)；b2 = 规则 + 学习预判提前触发。
        # 仅当 predictor 提供且 mode=b2 时才启用 guard；其余情况回退为纯规则恢复。
        try:
            self.recovery_mode = RecoveryMode(recovery_mode)
        except ValueError:
            raise ValueError(
                f"recovery_mode 必须是 {[m.value for m in RecoveryMode]}, 得到 {recovery_mode!r}"
            )
        self.predictor = predictor
        self.guard: Optional[PredictiveGuard] = None
        if predictor is not None and self.recovery_mode is RecoveryMode.B2_PREDICTIVE:
            self.guard = PredictiveGuard(
                predictor,
                mode=RecoveryMode.B2_PREDICTIVE,
                threshold=predictor_threshold,
                min_confidence=predictor_min_confidence,
            )

    def run(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        max_steps: Optional[int] = None,
        abort_check: Optional[Callable[[], bool]] = None,
    ) -> RecoveryRunResult:
        """执行一个 episode 的完整恢复编排, 返回 RecoveryRunResult。"""
        self.recovery_sm.reset()
        self.recovery_sm.start()
        self.attempt_results = []
        if self.guard is not None:
            self.guard.reset()
        step_hook = self.guard.step_hook if self.guard is not None else None

        # 唯一一次 env reset (恢复在同一 episode 内)。
        obs = self.recording.reset(seed=seed, options=options)
        skill = self.registry.lookup(self.skill.skill_id)

        attempts = 0
        decisions: List[RecoveryDecision] = []
        final_result: Optional[SkillResult] = None
        episode: Optional[EpisodeInfo] = None
        recovered = False
        aborted = False

        while True:
            attempts += 1
            self._log_attempt_begin(attempts)
            if self.guard is not None:
                self.guard.begin_attempt()
            hist_start = len(self.recording.history)
            result = self.executor.execute(
                skill, obs, max_steps=max_steps, adapter=self.recording,
                abort_check=abort_check, step_hook=step_hook,
            )
            self.attempt_results.append(result)
            episode = self.executor.last_episode
            final_obs = self.recording.last_obs
            # 本次尝试的观测历史切片 (不跨尝试混入)。
            attempt_history = self.recording.history[hist_start:]
            preds = self.guard.drain_attempt() if self.guard is not None else []
            self._log_predictions(preds, attempts)
            self._log_attempt_end(result, attempts, episode, predictions=preds)

            # 成功 -> 任务达成 (恢复后成功 => recovered=True)。
            if result.status is SkillStatus.SUCCESS:
                final_result = result
                recovered = attempts > 1
                self.recovery_sm.mark_succeeded()
                break

            # 外部中止 -> 不重试。
            if result.status is SkillStatus.ABORTED:
                final_result = result
                aborted = True
                self.recovery_sm.mark_aborted()
                break

            # 失败检测 (只用可观测信息)。
            failure = self.detector.detect(result, final_obs, obs_history=attempt_history)
            if failure is None:
                # 兜底: 非 SUCCESS/ABORTED 却未检出失败 -> UNKNOWN。
                failure = FailureEvent(
                    type=FailureType.UNKNOWN, step=result.steps, timestamp=0.0,
                    detail={"source": "detector", "reason": "undetected non-success"},
                )

            env_ended = bool(episode is not None and (episode.terminated or episode.truncated))
            decision = self.manager.decide(
                failure, retry_count=attempts - 1, env_ended=env_ended,
            )
            decisions.append(decision)
            self._log_decision(decision, attempts, failure)

            if decision.action is RecoveryAction.ABORT:
                final_result = result
                aborted = True
                self.recovery_sm.mark_aborted(decision)
                break

            # RETRY / REOBSERVE -> 同 episode 内重试 (不 reset env)。
            self.recovery_sm.mark_failed(decision)
            self.recovery_sm.mark_retrying()
            obs = final_obs if final_obs is not None else obs

        assert final_result is not None  # 循环至少执行一次
        recovery = RecoveryResult(
            attempts=attempts,
            final_skill_result=final_result,
            decisions=decisions,
            recovered=recovered,
            aborted=aborted,
        )
        self._log_recovery_summary(recovery, episode)
        return RecoveryRunResult(
            recovery=recovery,
            attempt_results=self.attempt_results,
            task_state=self.recovery_sm.state.value,
            episode=episode,
        )

    # ------------------------------------------------------------------ #
    # 结构化 trace
    # ------------------------------------------------------------------ #
    def _log_attempt_begin(self, attempt: int) -> None:
        self.logger.log({
            "event": "skill_attempt_begin",
            "task_id": self.task_id,
            "attempt": attempt,
        })

    def _log_attempt_end(self, result: SkillResult, attempt: int,
                         episode: Optional[EpisodeInfo],
                         predictions: Optional[List] = None) -> None:
        preds = list(predictions) if predictions else []
        self.logger.log_result(
            result,
            event="skill_attempt",
            extra={
                "task_id": self.task_id,
                "attempt": attempt,
                "terminated": bool(episode.terminated) if episode is not None else None,
                "truncated": bool(episode.truncated) if episode is not None else None,
                "env_success": bool(episode.success) if episode is not None else None,
                "recovery_mode": self.recovery_mode.value,
                "num_predictions": len(preds),
                "predictive_triggered": any(p.triggered for p in preds),
            },
        )

    def _log_predictions(self, preds: List, attempt: int) -> None:
        for p in preds:
            self.logger.log({
                "event": "predictive_prediction",
                "task_id": self.task_id,
                "attempt": attempt,
                "step": p.step,
                "prob": p.prob,
                "confidence": p.confidence,
                "prediction": p.prediction,
                "triggered": p.triggered,
                "threshold": p.threshold,
                "error": p.error,
            })

    def _log_decision(self, decision: RecoveryDecision, attempt: int,
                      failure: FailureEvent) -> None:
        self.logger.log({
            "event": "recovery_decision",
            "task_id": self.task_id,
            "attempt": attempt,
            "action": decision.action.value,
            "retry_count": decision.retry_count,
            "reason": decision.reason,
            "failure_type": failure.type.value,
            "detail": decision.detail,
            "predictive_triggered": bool(failure.detail.get("predictive")),
        })

    def _log_recovery_summary(self, recovery: RecoveryResult,
                              episode: Optional[EpisodeInfo]) -> None:
        sr = recovery.final_skill_result
        if self.guard is not None:
            recs = self.guard.records
            predictive = {
                "mode": self.recovery_mode.value,
                "num_predictions": len(recs),
                "num_triggers": sum(1 for r in recs if r.triggered),
                "num_fallbacks": sum(1 for r in recs if r.error is not None),
            }
        else:
            predictive = {
                "mode": self.recovery_mode.value,
                "num_predictions": 0,
                "num_triggers": 0,
                "num_fallbacks": 0,
            }
        self.logger.log({
            "event": "recovery_result",
            "task_id": self.task_id,
            "attempts": recovery.attempts,
            "recovered": recovery.recovered,
            "aborted": recovery.aborted,
            "final_status": sr.status.value,
            "final_skill_id": sr.skill_id,
            "num_decisions": len(recovery.decisions),
            "decisions": [d.action.value for d in recovery.decisions],
            "terminated": bool(episode.terminated) if episode is not None else None,
            "truncated": bool(episode.truncated) if episode is not None else None,
            "env_success": bool(episode.success) if episode is not None else None,
            "predictive": predictive,
        })

    def close(self) -> None:
        try:
            self.recording.close()
        except Exception:
            pass
        self.logger.close()
