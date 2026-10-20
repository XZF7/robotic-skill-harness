"""PickCubeAdapter: ManiSkill PickCube-v1 的 SimulatorAdapter 实现。

封装 Franka Panda + PickCube-v1 环境，遵循 interfaces 的批量张量约定：
- torch 张量 -> numpy
- 单环境 (num_envs==1) 去掉 batch 维
- terminated / truncated / success 三者独立
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

import mani_skill.envs  # noqa: F401  触发环境注册
import gymnasium as gym

from interfaces import (
    EpisodeInfo,
    GroundTruth,
    PolicyObservation,
    RobotObservation,
    SimulatorAdapter,
    StepResult,
)

# Franka Panda 抓手 "全开" 时单指关节角 (rad)。抓手关节范围 [0, 0.04]。
_PANDA_GRIPPER_OPEN = 0.04


def _to_np(x):
    """torch 张量 -> numpy；单环境去掉 batch 维 (shape[0]==1)。"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        if x.ndim == 1 and x.shape[0] == 1:
            return x[0]
        if x.ndim >= 2 and x.shape[0] == 1:
            return x[0]
    return x


def _as_bool(x) -> bool:
    x = _to_np(x)
    return bool(x)


def _as_float(x) -> float:
    x = _to_np(x)
    return float(x)


class PickCubeAdapter(SimulatorAdapter):
    """PickCube-v1 (Franka Panda, pd_joint_pos) 环境封装。

    动作空间: (8,) float32 = [7 关节角 (rad), 1 抓手 (-1 关 / 1 开)]。
    """

    def __init__(
        self,
        env_id: str = "PickCube-v1",
        control_mode: str = "pd_joint_pos",
        sim_backend: str = "cpu",
        render_backend: str = "none",
        max_episode_steps: int = 200,
        obs_mode: str = "none",
    ):
        self._max_episode_steps = max_episode_steps
        self._step_count = 0
        # render_mode="none" + render_backend="none" -> 完全关闭渲染 (headless)
        self._env = gym.make(
            env_id,
            obs_mode=obs_mode,
            control_mode=control_mode,
            render_mode="none",
            render_backend=render_backend,
            sim_backend=sim_backend,
            max_episode_steps=max_episode_steps,
        )
        self._control_timestep = float(self._env.unwrapped.control_timestep)

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    @property
    def env(self):
        """暴露底层 gym env，供脚本化技能做运动规划。"""
        return self._env

    def _gripper_openness(self) -> float:
        u = self._env.unwrapped
        finger = u.agent.robot.get_qpos().cpu().numpy()[0][7:9]
        return float(np.clip(np.mean(finger) / _PANDA_GRIPPER_OPEN, 0.0, 1.0))

    def _build_observation(self, info: dict) -> RobotObservation:
        u = self._env.unwrapped
        robot = u.agent.robot
        qpos = robot.get_qpos().cpu().numpy()[0]      # (9,) 7 臂 + 2 指
        qvel = robot.get_qvel().cpu().numpy()[0]      # (9,)
        tcp_pose = u.agent.tcp_pose.raw_pose.cpu().numpy()[0]  # (7,)

        policy = PolicyObservation(
            qpos=qpos[:7].astype(np.float32),
            qvel=qvel[:7].astype(np.float32),
            tcp_pose=tcp_pose.astype(np.float32),
            gripper=self._gripper_openness(),
            images=None,
        )
        truth = GroundTruth(
            object_poses={
                "cube": u.cube.pose.raw_pose.cpu().numpy()[0].astype(np.float32),
                "goal_site": u.goal_site.pose.raw_pose.cpu().numpy()[0].astype(np.float32),
            },
            success=_as_bool(info.get("success", False)),
            is_grasped=_as_bool(info.get("is_grasped", False)),
            contact={},
        )
        return RobotObservation(
            policy=policy,
            truth=truth,
            step=self._step_count,
            timestamp=self._step_count * self._control_timestep,
        )

    # ------------------------------------------------------------------ #
    # SimulatorAdapter 接口
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None,
              options: Optional[dict] = None) -> RobotObservation:
        self._step_count = 0
        obs, info = self._env.reset(seed=seed, options=options)
        return self._build_observation(info)

    def step(self, action: np.ndarray) -> StepResult:
        action = np.asarray(action, dtype=np.float32)
        obs, reward, terminated, truncated, info = self._env.step(action)
        self._step_count += 1
        episode = EpisodeInfo(
            terminated=_as_bool(terminated),
            truncated=_as_bool(truncated),
            success=_as_bool(info.get("success", False)),
            reward=_as_float(reward),
            info={k: _to_np(v) for k, v in info.items()},
        )
        return StepResult(
            observation=self._build_observation(info),
            reward=episode.reward,
            episode=episode,
        )

    def close(self) -> None:
        self._env.close()

    # ------------------------------------------------------------------ #
    # 属性
    # ------------------------------------------------------------------ #
    @property
    def num_envs(self) -> int:
        return 1

    @property
    def action_space(self) -> object:
        space = self._env.action_space
        return {
            "shape": tuple(space.shape),
            "low": np.asarray(space.low, dtype=np.float32),
            "high": np.asarray(space.high, dtype=np.float32),
            "dtype": str(space.dtype),
        }

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps
