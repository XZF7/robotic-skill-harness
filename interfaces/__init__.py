"""Robotic Skill Harness 共享接口契约 (frozen)。

Lead Agent 拥有本包变更权；子代理只读，修改须先申请审批。
"""
from .observation import PolicyObservation, GroundTruth, RobotObservation
from .skill import SkillStatus, SkillResult, RobotSkill
from .adapter import EpisodeInfo, StepResult, SimulatorAdapter
from .failure import FailureType, FailureEvent
from .recovery import RecoveryAction, RecoveryDecision, RecoveryResult
from .fault import FaultRecord
from .episode import EpisodeResult, EpisodeMetrics

__all__ = [
    "PolicyObservation", "GroundTruth", "RobotObservation",
    "SkillStatus", "SkillResult", "RobotSkill",
    "EpisodeInfo", "StepResult", "SimulatorAdapter",
    "FailureType", "FailureEvent",
    "RecoveryAction", "RecoveryDecision", "RecoveryResult",
    "FaultRecord",
    "EpisodeResult", "EpisodeMetrics",
]
