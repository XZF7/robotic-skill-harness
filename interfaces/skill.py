"""RobotSkill / SkillResult contract.

超时语义: 技能在自身步数预算内未 is_done 即记 TIMEOUT；
episode 级 horizon 由 SimulatorAdapter.truncated 表达，二者不混用。
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .failure import FailureEvent
from .observation import RobotObservation


class SkillStatus(str, Enum):
    SUCCESS = "success"    # 达到技能目标
    FAILURE = "failure"    # 执行失败 (含可识别故障)
    TIMEOUT = "timeout"    # 技能步数预算耗尽
    ABORTED = "aborted"    # 被外部中止


@dataclass
class SkillResult:
    status: SkillStatus
    skill_id: str
    steps: int
    message: str = ""
    failure: Optional[FailureEvent] = None


class RobotSkill(ABC):
    """可执行技能的统一接口。act 返回与 SimulatorAdapter.action_space 匹配的动作。"""
    skill_id: str = "abstract"

    @abstractmethod
    def reset(self, obs: RobotObservation) -> None:
        """进入新 episode 时重置技能内部状态。"""

    @abstractmethod
    def act(self, obs: RobotObservation) -> np.ndarray:
        """根据观测返回一步动作。"""

    @property
    @abstractmethod
    def is_done(self) -> bool:
        """技能是否已完成 (成功或失败)。"""
