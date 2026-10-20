"""SimulatorAdapter contract.

终止语义 (关键, 三者独立, 不得混用):
- terminated: 环境判定 episode 结束 (任务达成或失败条件触发)。
- truncated:  达到最大步数 (horizon) 被截断。
- success:    任务是否成功 (privileged truth)，与 terminated 独立。
  例: success=True 时 terminated 通常为 True；
      truncated=True 时 success 可能为 False (超时未完成)。

ManiSkill 批量张量约定 (由 adapter 负责转换):
- torch 张量 → numpy 数组。
- 单环境 (num_envs==1): 去掉 batch 维，得到非批量数组。
- 多环境: 保留 batch 维 [num_envs, ...]。
- 契约暴露的 RobotObservation 一律为上述 numpy 约定。
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .observation import RobotObservation


@dataclass
class EpisodeInfo:
    terminated: bool
    truncated: bool
    success: bool
    reward: float = 0.0
    info: dict = field(default_factory=dict)   # 原始 info 透传


@dataclass
class StepResult:
    observation: RobotObservation
    reward: float
    episode: EpisodeInfo


class SimulatorAdapter(ABC):
    """ManiSkill 环境统一封装，屏蔽具体 env 细节。"""

    @abstractmethod
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[dict] = None) -> RobotObservation:
        """重置环境并返回初始观测。"""

    @abstractmethod
    def step(self, action: np.ndarray) -> StepResult:
        """执行一步动作，返回观测 + reward + episode 状态。"""

    @abstractmethod
    def close(self) -> None:
        """释放环境资源。"""

    @property
    @abstractmethod
    def num_envs(self) -> int:
        """并行环境数。"""

    @property
    @abstractmethod
    def action_space(self) -> object:
        """动作空间描述 (维度/界)。"""

    @property
    @abstractmethod
    def max_episode_steps(self) -> int:
        """episode horizon (最大步数)。"""
