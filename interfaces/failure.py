"""Failure event contract.

FailureType 即规范错误码 (error code)。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class FailureType(str, Enum):
    DROP = "drop"                  # 物体掉落
    COLLISION = "collision"        # 非预期碰撞
    GRIP_FAIL = "grip_fail"        # 抓取失败/滑脱
    MOTION_FAIL = "motion_fail"    # 规划/运动失败
    TIMEOUT = "timeout"            # 步数/时间超限
    INVALID_OBS = "invalid_obs"    # 无效/异常观测
    RECOVERY_EXHAUSTED = "recovery_exhausted"  # 恢复重试耗尽
    UNKNOWN = "unknown"


@dataclass
class FailureEvent:
    type: FailureType
    step: int                      # episode 内步数
    timestamp: float               # 仿真时间 (s)
    detail: dict = field(default_factory=dict)  # 额外诊断
