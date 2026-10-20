"""脚本化 Pick 技能：基于 mplib 运动规划抓取并搬运方块。

流程 (对齐官方 solvePickCube):
  reach (从上方接近) -> grasp (下探抓取点) -> close gripper -> move to goal
内部用 mplib 的 plan_screw 预规划三段轨迹，拼成一个动作队列，
每次 act() 弹出一个动作交回 adapter 逐步执行 (不内部调 step)。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import sapien

from mani_skill.examples.motionplanning.panda.motionplanner import     PandaArmMotionPlanningSolver
from mani_skill.examples.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb,
    get_actor_obb,
)

from interfaces import (
    FailureEvent,
    FailureType,
    RobotObservation,
    RobotSkill,
    SkillResult,
    SkillStatus,
)

# 与官方 mplib 求解器一致：OPEN=1, CLOSED=-1 (pd_joint_pos 的抓手动作)
_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSED = -1.0
_CLOSE_STEPS = 6          # 抓手闭合步数 (对齐官方 close_gripper(t=6))
_FINGER_LENGTH = 0.025    # 抓取深度


class PickCubeSkill(RobotSkill):
    """PickCube 脚本化抓取技能 (mplib 运动规划)。"""

    skill_id = "pick_cube_mplib"

    def __init__(self, adapter):
        # adapter 提供底层 env (unwrapped) 用于规划
        self._adapter = adapter
        self._env = adapter.env
        self._planner: Optional[PandaArmMotionPlanningSolver] = None
        self._queue: List[np.ndarray] = []
        self._done = False
        self._status: Optional[SkillStatus] = None
        self._failure: Optional[FailureEvent] = None
        self._steps = 0
        self._last_action: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # 规划
    # ------------------------------------------------------------------ #
    def _setup_planner(self) -> PandaArmMotionPlanningSolver:
        u = self._env.unwrapped
        return PandaArmMotionPlanningSolver(
            self._env,
            debug=False,
            vis=False,
            base_pose=u.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
        )

    def _plan_to_pose(self, pose: sapien.Pose, start_qpos: np.ndarray):
        """plan_screw 到目标位姿，返回 (N,7) 关节轨迹；失败返回 None (重试一次)。"""
        u = self._env.unwrapped
        target = np.concatenate([pose.p, pose.q])
        result = None
        for _ in range(2):
            result = self._planner.planner.plan_screw(
                target,
                start_qpos,
                time_step=u.control_timestep,
                use_point_cloud=False,
            )
            if result["status"] == "Success":
                return result["position"].astype(np.float32)
        return None

    def _plan(self) -> None:
        u = self._env.unwrapped
        self._planner = self._setup_planner()

        # 1) 由方块 OBB 计算抓取位姿 (对齐官方 solvePickCube)
        obb = get_actor_obb(u.cube)
        approaching = np.array([0, 0, -1])
        target_closing = u.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        grasp_info = compute_grasp_info_by_obb(
            obb,
            approaching=approaching,
            target_closing=target_closing,
            depth=_FINGER_LENGTH,
        )
        closing = grasp_info["closing"]
        grasp_pose = u.agent.build_grasp_pose(approaching, closing, u.cube.pose.sp.p)
        reach_pose = grasp_pose * sapien.Pose([0, 0, -0.05])
        goal_pose = sapien.Pose(u.goal_site.pose.sp.p, grasp_pose.q)

        start_qpos = u.agent.robot.get_qpos().cpu().numpy()[0][:7].astype(np.float32)

        # 2) 三段预规划 (reach -> grasp -> goal)
        reach_path = self._plan_to_pose(reach_pose, start_qpos)
        if reach_path is None:
            raise RuntimeError("reach 阶段运动规划失败")
        grasp_path = self._plan_to_pose(grasp_pose, reach_path[-1])
        if grasp_path is None:
            raise RuntimeError("grasp 阶段运动规划失败")
        goal_path = self._plan_to_pose(goal_pose, grasp_path[-1])
        if goal_path is None:
            raise RuntimeError("move-to-goal 阶段运动规划失败")

        # 3) 拼接动作队列: [7 关节角, 抓手]
        self._queue = []
        for q in reach_path:
            self._queue.append(np.concatenate([q, [_GRIPPER_OPEN]]).astype(np.float32))
        for q in grasp_path:
            self._queue.append(np.concatenate([q, [_GRIPPER_OPEN]]).astype(np.float32))
        for _ in range(_CLOSE_STEPS):
            self._queue.append(np.concatenate([grasp_path[-1], [_GRIPPER_CLOSED]]).astype(np.float32))
        for q in goal_path:
            self._queue.append(np.concatenate([q, [_GRIPPER_CLOSED]]).astype(np.float32))

    # ------------------------------------------------------------------ #
    # RobotSkill 接口
    # ------------------------------------------------------------------ #
    def reset(self, obs: RobotObservation) -> None:
        self._done = False
        self._status = None
        self._failure = None
        self._steps = 0
        self._queue = []
        self._last_action = None
        try:
            self._plan()
        except Exception as e:  # noqa: BLE001  规划失败即技能失败
            self._status = SkillStatus.FAILURE
            self._failure = FailureEvent(
                type=FailureType.MOTION_FAIL,
                step=0,
                timestamp=0.0,
                detail={"error": str(e)},
            )
            self._done = True

    def act(self, obs: RobotObservation) -> np.ndarray:
        # 已经成功 (上一步动作后) -> 提前结束
        if obs.truth.success:
            self._status = SkillStatus.SUCCESS
            self._done = True
            return self._hold_action(obs)

        if self._queue:
            action = self._queue.pop(0)
            self._last_action = action
            self._steps += 1
            return action

        # 队列耗尽仍未成功 -> 失败
        self._status = SkillStatus.FAILURE
        self._failure = FailureEvent(
            type=FailureType.MOTION_FAIL,
            step=self._steps,
            timestamp=obs.timestamp,
            detail={"error": "plan exhausted without success"},
        )
        self._done = True
        return self._hold_action(obs)

    def _hold_action(self, obs: RobotObservation) -> np.ndarray:
        """技能已结束时的安全保持动作 (停在当前关节角，抓手保持)。"""
        if self._last_action is not None:
            return self._last_action
        qpos = obs.policy.qpos.astype(np.float32)
        return np.concatenate([qpos, [_GRIPPER_OPEN]]).astype(np.float32)

    @property
    def is_done(self) -> bool:
        return self._done

    def finalize(self, obs: RobotObservation) -> SkillResult:
        """episode 结束时由驱动循环调用，依据最终观测确定技能结果。

        (若 env 因 success 触发 terminated，循环会在下一次 act 前 break，
         故需在此依据最终观测补定 status。)
        """
        if self._status is None:
            if obs.truth.success:
                self._status = SkillStatus.SUCCESS
            else:
                self._status = SkillStatus.FAILURE
                self._failure = FailureEvent(
                    type=FailureType.MOTION_FAIL,
                    step=self._steps,
                    timestamp=obs.timestamp,
                    detail={"error": "episode ended without success"},
                )
            self._done = True
        return self.result()

    # ------------------------------------------------------------------ #
    # 结果
    # ------------------------------------------------------------------ #
    @property
    def status(self) -> Optional[SkillStatus]:
        return self._status

    def result(self) -> SkillResult:
        status = self._status
        if status is None:
            status = SkillStatus.TIMEOUT
        return SkillResult(
            status=status,
            skill_id=self.skill_id,
            steps=self._steps,
            message=("" if status is SkillStatus.SUCCESS
                     else (self._failure.detail.get("error", "") if self._failure else "")),
            failure=self._failure,
        )
