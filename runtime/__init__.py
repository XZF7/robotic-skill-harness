"""Robotic Skill Harness — Runtime (Agent B)。

纯 mock 运行时：SkillRegistry / SkillExecutor / TaskStateMachine / ExecutionLogger。
不依赖 sim/ 与 Agent A 的真实仿真；仅依赖冻结接口 interfaces/。
"""
from .errors import (
    DuplicateSkillError,
    InvalidTransitionError,
    SkillNotFoundError,
)
from .executor import SkillExecutor
from .failure_detector import FailureDetector, FailureDetectorConfig
from .harness import TaskRunner, TaskRunResult
from .recovery import (
    RecordingAdapter,
    RecoveryManager,
    RecoveryManagerConfig,
    RecoveryRunResult,
    RecoveryRunner,
    RecoveryState,
    RecoveryStateMachine,
)
from .logger import ExecutionLogger
from .registry import SkillRegistry
from .state_machine import (
    TaskState,
    TaskStateMachine,
    skill_status_to_task_state,
)

__all__ = [
    "SkillRegistry",
    "SkillExecutor",
    "TaskRunner",
    "TaskRunResult",
    "FailureDetector",
    "FailureDetectorConfig",
    "RecoveryManager",
    "RecoveryManagerConfig",
    "RecoveryRunner",
    "RecoveryRunResult",
    "RecoveryState",
    "RecoveryStateMachine",
    "RecordingAdapter",
    "TaskState",
    "TaskStateMachine",
    "skill_status_to_task_state",
    "ExecutionLogger",
    "SkillNotFoundError",
    "DuplicateSkillError",
    "InvalidTransitionError",
]
