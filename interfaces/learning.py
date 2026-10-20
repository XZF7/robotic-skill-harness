"""M4 学习预测契约: 特征 schema + 失败标注 + 数据集 schema + 预测接口。

关键约束 (M4-0):
- 特征只用 policy-accessible 观测 (obs.policy: qpos/qvel/gripper/tcp_pose)，
  严禁 obs.truth (特权真值: 物体位姿/success/is_grasped/contact)。
- 状态基线 (state-based): 特征来自 proprioception，不含视觉，也不含特权物体位姿。
- 标签由完整 episode 离线生成 (用未来结果)，模型输入只用当前/历史观测 —— 不泄漏未来。
- 元数据 (seed/fault_type/outcome) 与模型输入特征严格分离。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .observation import RobotObservation

# --------------------------------------------------------------------------- #
# 特征 schema (policy-accessible proprioception)
# --------------------------------------------------------------------------- #
# [qpos(7), qvel(7), gripper(1), tcp_pose(7)] = 22 维
STATE_FEATURE_NAMES = [
    *[f"qpos_{i}" for i in range(7)],
    *[f"qvel_{i}" for i in range(7)],
    "gripper",
    "tcp_x", "tcp_y", "tcp_z", "tcp_qw", "tcp_qx", "tcp_qy", "tcp_qz",
]
STATE_FEATURE_DIM = len(STATE_FEATURE_NAMES)  # 22


def extract_state_features(obs: RobotObservation) -> np.ndarray:
    """从观测提取状态特征 (仅 obs.policy，绝不读 obs.truth)。返回 (22,) float32。"""
    p = obs.policy
    qpos = np.asarray(p.qpos, dtype=np.float32).reshape(-1)[:7]
    qvel = np.asarray(p.qvel, dtype=np.float32).reshape(-1)[:7]
    gripper = np.asarray([float(p.gripper)], dtype=np.float32)
    tcp = np.asarray(p.tcp_pose, dtype=np.float32).reshape(-1)[:7]
    feats = np.concatenate([qpos, qvel, gripper, tcp]).astype(np.float32)
    assert feats.shape == (STATE_FEATURE_DIM,), f"unexpected feature shape {feats.shape}"
    return feats


# --------------------------------------------------------------------------- #
# 失败标注 (标签由完整 episode 离线生成, 不泄漏未来到模型输入)
# --------------------------------------------------------------------------- #
DEFAULT_PREDICTION_HORIZON = 20  # 预测窗口 H (步)


def label_failure(
    steps: List[int],
    failed: bool,
    t_end: int,
    horizon: int = DEFAULT_PREDICTION_HORIZON,
) -> np.ndarray:
    """为每个帧生成失败标签 (0/1)。

    steps: 每帧的 step 数 (升序)。
    failed: episode 是否失败。
    t_end: 失败 episode 的最终步 (成功 episode 忽略)。
    horizon: 预测窗口 H —— 失败 episode 的最后 H 步内帧标 1 (即将失败)。

    返回 (len(steps),) int 标签; 成功 episode 全部为 0。
    """
    labels = np.zeros(len(steps), dtype=np.int64)
    if failed:
        labels[:] = [1 if (t_end - int(s)) <= horizon else 0 for s in steps]
    return labels


# --------------------------------------------------------------------------- #
# 数据集 schema (元数据与模型输入特征分离)
# --------------------------------------------------------------------------- #
@dataclass
class FrameRecord:
    step: int
    features: np.ndarray   # (22,) float32 模型输入特征
    label: int             # 0/1 (离线标注)


@dataclass
class EpisodeRecord:
    episode_id: str
    seed: int
    fault_type: Optional[str]   # None = 无故障 (正常执行)
    fault_params: dict = field(default_factory=dict)
    outcome: bool = False       # episode 是否成功 (离线标注用, 非模型输入)
    frames: List[FrameRecord] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 预测接口 (runtime 集成用)
# --------------------------------------------------------------------------- #
@dataclass
class FailurePrediction:
    prob: float          # P(failure within horizon) ∈ [0,1]
    prediction: bool     # prob >= threshold
    confidence: float = 0.0  # ∈ [0,1]


class FailurePredictor(ABC):
    """失败风险预测器。features: (22,) 单帧 (模型可内部维护历史窗口)。"""

    @abstractmethod
    def predict(self, features: np.ndarray) -> FailurePrediction:
        """给定当前状态特征，返回失败风险预测。"""
