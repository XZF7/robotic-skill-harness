"""Robot observation contracts.

单位: SI (位置 m, 角度 rad, 速度 m/s 或 rad/s, 时间 s)
坐标系: 世界系 SAPIEN 约定 (x 前, y 左, z 上)
姿态: 四元数 (w, x, y, z)；位姿向量 = [x, y, z, qw, qx, qy, qz] (7,)

关键区分 (约束: 策略观测与特权真值分离):
- PolicyObservation: 策略可见 (proprioception + 可选图像)。
- GroundTruth:       特权仿真真值 (物体位姿、success 标志等)，
                     仅用于训练监督/评估，严禁在推理期喂给策略。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class PolicyObservation:
    """策略可见观测。所有数组为 numpy、单环境时无 batch 维。"""
    qpos: np.ndarray          # 关节角 (rad), shape (n_joints,)
    qvel: np.ndarray          # 关节速度 (rad/s), shape (n_joints,)
    tcp_pose: np.ndarray      # 末端位姿 世界系 (7,) = [x,y,z,qw,qx,qy,qz]
    gripper: float            # 抓手开度 [0,1]
    images: Optional[Dict[str, np.ndarray]] = None  # {cam: HxWxC uint8}


@dataclass
class GroundTruth:
    """特权真值 (策略不可见)。"""
    object_poses: Dict[str, np.ndarray]   # {name: 世界系位姿 (7,)}
    success: bool                          # 任务是否达成
    is_grasped: bool                       # 是否已抓取
    contact: dict = field(default_factory=dict)  # 可选接触/特权信息


@dataclass
class RobotObservation:
    """一步之后的完整观测 = 策略观测 + 特权真值。"""
    policy: PolicyObservation
    truth: GroundTruth
    step: int                # episode 内步数 (从 0 起)
    timestamp: float         # 仿真时间 (s)
