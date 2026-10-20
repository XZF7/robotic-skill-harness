"""TaskStateMachine: 任务状态与合法转移。"""
from __future__ import annotations

from enum import Enum
from typing import List, Tuple

from interfaces import SkillResult, SkillStatus

from .errors import InvalidTransitionError


class TaskState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    ABORTED = "aborted"


TERMINAL_STATES = frozenset({
    TaskState.SUCCESS, TaskState.FAILURE, TaskState.TIMEOUT, TaskState.ABORTED,
})

# 合法转移表：IDLE → RUNNING/ABORTED；RUNNING → 4 个终态；终态无出边。
_TRANSITIONS = {
    TaskState.IDLE: {TaskState.RUNNING, TaskState.ABORTED},
    TaskState.RUNNING: {
        TaskState.SUCCESS, TaskState.FAILURE,
        TaskState.TIMEOUT, TaskState.ABORTED,
    },
    TaskState.SUCCESS: set(),
    TaskState.FAILURE: set(),
    TaskState.TIMEOUT: set(),
    TaskState.ABORTED: set(),
}


def skill_status_to_task_state(status: SkillStatus) -> TaskState:
    return {
        SkillStatus.SUCCESS: TaskState.SUCCESS,
        SkillStatus.FAILURE: TaskState.FAILURE,
        SkillStatus.TIMEOUT: TaskState.TIMEOUT,
        SkillStatus.ABORTED: TaskState.ABORTED,
    }[status]


class TaskStateMachine:
    """任务状态机。

    - transition_to(target): 合法转移则推进并记录历史；非法转移抛 InvalidTransitionError。
    - start(): IDLE → RUNNING。
    - apply_skill_result(result): 依据 SkillResult.status 推导目标态并转移。
    - reset(): 强制回到 IDLE (复用，不计入转移历史)。
    """

    def __init__(self, initial: TaskState = TaskState.IDLE) -> None:
        self._state: TaskState = initial
        self._history: List[Tuple[TaskState, TaskState]] = []

    @property
    def state(self) -> TaskState:
        return self._state

    @property
    def history(self) -> List[Tuple[TaskState, TaskState]]:
        return list(self._history)

    def can_transition(self, target: TaskState) -> bool:
        return target in _TRANSITIONS[self._state]

    def is_terminal(self) -> bool:
        return self._state in TERMINAL_STATES

    def transition_to(self, target: TaskState) -> TaskState:
        if not isinstance(target, TaskState):
            raise TypeError(f"target 必须是 TaskState, 得到 {type(target).__name__}")
        if not self.can_transition(target):
            raise InvalidTransitionError(
                f"非法转移: {self._state.value} -> {target.value}"
            )
        self._history.append((self._state, target))
        self._state = target
        return target

    def start(self) -> TaskState:
        return self.transition_to(TaskState.RUNNING)

    def apply_skill_result(self, result: SkillResult) -> TaskState:
        return self.transition_to(skill_status_to_task_state(result.status))

    def reset(self) -> None:
        self._state = TaskState.IDLE
        self._history = []
