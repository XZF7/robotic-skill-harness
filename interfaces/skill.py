"""RobotSkill / SkillResult contract.

四个概念必须严格区分 (单一权威结果):
1. 技能执行完成 (skill completion):     RobotSkill.is_done == True，技能结束工作。
2. 技能执行成功 (skill success):        SkillResult.status == SUCCESS，技能自报达成目标。
3. 任务/episode 成功 (task success):    EpisodeInfo.success，环境的特权真值 (任务是否达成)。
4. 仿真终止/截断 (termination/truncation): EpisodeInfo.terminated / truncated，环境级事件。

状态语义 (SkillStatus 的单一权威来源):
- SUCCESS / FAILURE: 由技能自己通过 result() 报告。executor 不据此判定，尤其不看 truth.success。
- TIMEOUT / ABORTED: 由 executor 在提前终止技能时产生 (技能无法自我感知被超时/中止)。
- 异常: skill.act 抛异常由 executor 捕获，转 FAILURE (FailureType.MOTION_FAIL)。

特权真值 (truth) 隔离:
- RobotObservation.truth 是特权仿真真值，仅供评估层 / 脚本化技能使用；
  视觉策略 (vision-based policy) 不得访问。executor 也不得用 truth 判定技能成败。
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .failure import FailureEvent
from .observation import RobotObservation


class SkillStatus(str, Enum):
    SUCCESS = "success"    # 技能自报达成目标
    FAILURE = "failure"    # 技能自报失败 (含可识别故障)
    TIMEOUT = "timeout"    # executor: 步数预算耗尽
    ABORTED = "aborted"    # executor: 被外部中止


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

    # 特权真值访问声明 (四类信息隔离):
    #   observable robot state  = obs.policy (qpos/qvel/tcp_pose/gripper/images)
    #   policy-accessible obs   = obs.policy (策略可读的观测)
    #   privileged ground truth = obs.truth (物体位姿/success/is_grasped/contact)
    #   evaluation-only info    = 仅评估层可用 (如 truth.success 判任务成功)
    # False (默认): 观测型策略, 严禁访问 obs.truth。
    # True:         oracle/状态基线, 可用特权位姿规划 + truth 判成功 (须显式标注)。
    privileged_access: bool = False

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

    @abstractmethod
    def result(self, final_obs: Optional[RobotObservation] = None) -> SkillResult:
        """返回该技能的权威执行结果。幂等: 多次调用返回一致结果, 无副作用。

        final_obs: 若 executor 因 env 终止/截断而提前结束 (is_done 仍为 False)，
        传入最终观测，技能据此补定状态 (脚本化技能可用 final_obs.truth 判定 SUCCESS；
        视觉策略则不得访问 truth)。

        返回值 status 只能是 SUCCESS 或 FAILURE (TIMEOUT/ABORTED 由 executor 负责)。
        """
