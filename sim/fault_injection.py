"""FaultInjector: PickCube-v1 的真实物理故障注入 (M3-A)。

在 episode 内指定步数注入故障，故障必须真实改变仿真物理状态（位姿/速度/关节角），
不能只置一个标志位。每条故障产出 interfaces.FaultRecord，其中 `injected` 通过
注入后立即读回物理状态来核实（不能只凭"我调用了 set_pose"）。

已调查并验证可用的底层 API (ManiSkill 3.0.1, CPU 后端, 单环境):
- Actor (cube):    set_pose / set_linear_velocity / set_angular_velocity / pose / get_*_velocity
                   —— 直接写 sapien.Entity 状态，物理正向积分从新状态继续，不破坏仿真。
- Articulation (robot): set_qpos / get_qpos / get_qlimits —— 9 dof (7 臂 + 2 指),
                   抓手关节为 qpos[7:9]，范围 [0, 0.04] (0 全闭, 0.04 全开)。
- agent.is_grasping(cube) —— 依据指端接触力判定，需物理步进后刷新。
- 注意: 单次 set_qpos 抓手张开会被 pd_joint_pos 控制器在下一步重新闭合，因此
  "物体掉落" 必须用持续张开 (duration>1) 才能让 cube 脱离抓取。

故障类型:
- object_displacement  一次性: 把 cube 位移 (teleport)，规划好的轨迹抓空。
- object_impulse       一次性: 给 cube 施加线速度冲量 (knock-away)。
- grasp_perturb        一次性: 抓取阶段给臂关节角加噪声，末端偏离导致抓空。
- gripper_drop         持续性: 在 cube 被抓离桌面后持续张开抓手 duration 步，cube 掉落。

用法 (harness 每步调用一次 after_step):
    injector = FaultInjector(adapter, seed=42)
    obs = adapter.reset(seed=0)
    skill = PickCubeSkill(adapter); skill.reset(obs)
    for i in range(adapter.max_episode_steps):
        action = skill.act(obs)
        res = adapter.step(action); obs = res.observation
        # 条件触发 (如 step==20 时位移, 或 airborne 时掉落):
        injector.inject("object_displacement", step=i+1, params={"delta": [0.08,0,0]})
        injector.after_step(obs)
        if res.episode.terminated or res.episode.truncated: break
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import sapien

from interfaces import FaultRecord

_GRIPPER_OPEN = 0.04   # Panda 抓手全开时单指关节角 (rad)，范围 [0, 0.04]

FAULT_OBJECT_DISPLACEMENT = "object_displacement"
FAULT_OBJECT_IMPULSE = "object_impulse"
FAULT_GRASP_PERTURB = "grasp_perturb"
FAULT_GRIPPER_DROP = "gripper_drop"

_ONESHOT_FAULTS = {
    FAULT_OBJECT_DISPLACEMENT,
    FAULT_OBJECT_IMPULSE,
    FAULT_GRASP_PERTURB,
}


@dataclass
class _ActiveFault:
    """持续中的故障 (如 gripper_drop 需要在多个步内反复张开抓手)。"""
    fault_type: str
    step: int                     # 注入起始步 (记入 FaultRecord.step)
    params: dict
    duration: int                 # 持续总步数
    remaining: int                # 剩余还需施加的步数
    state: dict = field(default_factory=dict)  # 故障自用暂存 (起始位姿等)


class FaultInjector:
    """确定性真实物理故障注入器。seed 固定时注入参数与结果可复现。"""

    def __init__(self, adapter, seed: Optional[int] = None):
        self._adapter = adapter
        self._env = adapter.env
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._records: List[FaultRecord] = []
        self._scheduled: Dict[int, List[tuple]] = {}
        self._active: List[_ActiveFault] = []

    # ------------------------------------------------------------------ #
    # 只读状态助手 (供故障函数读回状态做注入核实 / 供测试触发条件判断)
    # ------------------------------------------------------------------ #
    @property
    def u(self):
        return self._env.unwrapped

    def _cube_p(self) -> np.ndarray:
        return self.u.cube.pose.p.cpu().numpy()[0].copy()

    def _cube_q(self) -> np.ndarray:
        return self.u.cube.pose.q.cpu().numpy()[0].copy()

    def _cube_lin_vel(self) -> np.ndarray:
        return self.u.cube.get_linear_velocity().cpu().numpy()[0].copy()

    def _robot_qpos(self) -> np.ndarray:
        return self.u.agent.robot.get_qpos().cpu().numpy()[0].copy()

    def is_grasped(self) -> bool:
        t = self.u.agent.is_grasping(self.u.cube)
        return bool(t.cpu().numpy()[0])

    def cube_pose(self) -> np.ndarray:
        """cube 世界系位姿 [x,y,z] (供测试断言)。"""
        return self._cube_p()

    def cube_airborne(self, z_thresh: float = 0.05) -> bool:
        """cube 是否已被抓离桌面 (用于选择掉落注入时机)。"""
        return bool(self._cube_p()[2] > z_thresh)

    # ------------------------------------------------------------------ #
    # 写状态原语 (不调用 reset，直接改底层状态，物理正向积分继续)
    # ------------------------------------------------------------------ #
    def _set_cube_pose(self, p: np.ndarray, q: np.ndarray) -> None:
        self.u.cube.set_pose(
            sapien.Pose(p=np.asarray(p, np.float32), q=np.asarray(q, np.float32))
        )

    def _set_cube_lin_vel(self, v: np.ndarray) -> None:
        self.u.cube.set_linear_velocity(np.asarray(v, np.float32))

    def _set_cube_ang_vel(self, w: np.ndarray) -> None:
        self.u.cube.set_angular_velocity(np.asarray(w, np.float32))

    def _set_robot_qpos(self, q: np.ndarray) -> None:
        self.u.agent.robot.set_qpos(np.asarray(q, np.float32))

    def _open_gripper(self) -> None:
        q = self._robot_qpos()
        q[7] = _GRIPPER_OPEN
        q[8] = _GRIPPER_OPEN
        self._set_robot_qpos(q)

    # ------------------------------------------------------------------ #
    # 一次性故障实现
    # ------------------------------------------------------------------ #
    def _fault_object_displacement(self, step: int, params: dict) -> FaultRecord:
        before_p = self._cube_p()
        delta = params.get("delta")
        if delta is None:
            mag = float(params.get("distance", 0.07))
            ang = float(self._rng.uniform(0.0, 2.0 * np.pi))
            delta = np.array([mag * np.cos(ang), mag * np.sin(ang), 0.0], np.float32)
        else:
            delta = np.asarray(delta, np.float32).copy()
        new_p = before_p + delta
        # 不穿透桌面 (z 不低于 cube 半高)
        half = float(self.u.cube_half_size)
        new_p[2] = max(new_p[2], half)
        self._set_cube_pose(new_p, self._cube_q())
        # 清掉残余速度，避免位移后被旧速度拖回
        self._set_cube_lin_vel(np.zeros(3, np.float32))
        self._set_cube_ang_vel(np.zeros(3, np.float32))
        after_p = self._cube_p()
        moved = float(np.linalg.norm(after_p - before_p))
        return FaultRecord(
            fault_type=FAULT_OBJECT_DISPLACEMENT,
            injected=bool(moved > 1e-3),
            step=step,
            params={"delta": delta.tolist(), "distance": float(np.linalg.norm(delta))},
            detail={
                "before_pos": before_p.tolist(),
                "after_pos": after_p.tolist(),
                "moved": moved,
                "verification": "cube.pose readback after set_pose",
            },
        )

    def _fault_object_impulse(self, step: int, params: dict) -> FaultRecord:
        before_v = self._cube_lin_vel()
        impulse = params.get("velocity")
        if impulse is None:
            mag = float(params.get("speed", 0.6))
            ang = float(self._rng.uniform(0.0, 2.0 * np.pi))
            impulse = np.array([mag * np.cos(ang), mag * np.sin(ang), 0.0], np.float32)
        else:
            impulse = np.asarray(impulse, np.float32).copy()
        self._set_cube_lin_vel(impulse)
        after_v = self._cube_lin_vel()
        dv = float(np.linalg.norm(after_v - before_v))
        return FaultRecord(
            fault_type=FAULT_OBJECT_IMPULSE,
            injected=bool(dv > 1e-3),
            step=step,
            params={"velocity": impulse.tolist(), "speed": float(np.linalg.norm(impulse))},
            detail={
                "before_vel": before_v.tolist(),
                "after_vel": after_v.tolist(),
                "dv": dv,
                "verification": "cube linear velocity readback after set_linear_velocity",
            },
        )

    def _fault_grasp_perturb(self, step: int, params: dict) -> FaultRecord:
        before_q = self._robot_qpos()[:7].copy()
        scale = float(params.get("scale", 0.25))
        joints = params.get("joints")
        if joints is None:
            joints = [0, 1, 3]
        joints = [int(j) for j in joints]
        # 相干扰动: 所有指定关节沿同一方向偏移 (末端一致位移)。随机关节独立符号会相互
        # 抵消、且单步扰动会被 pd 控制器在抓取前吸收，导致抓空不可靠；相干同向偏移
        # 能可靠地把 TCP 推离抓取点。sign 由 rng 决定 (保证 seed 可复现)。
        sign = float(self._rng.choice([-1.0, 1.0]))
        noise = (sign * scale * np.ones(len(joints), dtype=np.float32))
        q = self._robot_qpos()
        for j, n in zip(joints, noise):
            q[j] += float(n)
        # 限制在关节限位内，避免越界/不稳定
        limits = self.u.agent.robot.get_qlimits().cpu().numpy()[0][:7]
        q[:7] = np.clip(q[:7], limits[:, 0], limits[:, 1])
        self._set_robot_qpos(q)
        after_q = self._robot_qpos()[:7].copy()
        d = float(np.linalg.norm(after_q - before_q))
        return FaultRecord(
            fault_type=FAULT_GRASP_PERTURB,
            injected=bool(d > 1e-4),
            step=step,
            params={"scale": scale, "joints": joints, "sign": sign, "noise": noise.tolist()},
            detail={
                "qpos_delta_norm": d,
                "before_qpos": before_q.tolist(),
                "after_qpos": after_q.tolist(),
                "verification": "robot qpos readback after set_qpos",
            },
        )

    # ------------------------------------------------------------------ #
    # 持续性故障实现 (gripper_drop)
    # ------------------------------------------------------------------ #
    def _mutate_sustained(self, af: _ActiveFault) -> None:
        if af.fault_type == FAULT_GRIPPER_DROP:
            if not af.state:
                af.state["start_z"] = float(self._cube_p()[2])
                af.state["finger_before"] = self._robot_qpos()[7:9].tolist()
            self._open_gripper()
        else:
            raise ValueError(f"unsupported sustained fault type: {af.fault_type}")

    def _finalize_sustained(self, af: _ActiveFault, step: int) -> FaultRecord:
        if af.fault_type == FAULT_GRIPPER_DROP:
            start_z = af.state.get("start_z")
            end_z = float(self._cube_p()[2])
            finger_after = self._robot_qpos()[7:9].tolist()
            grasped_now = self.is_grasped()
            fell = (start_z is not None) and (end_z < start_z - 0.01)
            injected = bool(fell)
            return FaultRecord(
                fault_type=FAULT_GRIPPER_DROP,
                injected=injected,
                step=af.step,
                params=dict(af.params, duration=af.duration),
                detail={
                    "start_z": start_z,
                    "end_z": end_z,
                    "finger_before": af.state.get("finger_before"),
                    "finger_after": finger_after,
                    "is_grasped_after": grasped_now,
                    "never_grasped": not grasped_now and not fell,
                    "fell": fell,
                    "verification": "sustained gripper open; cube fell / grasp released",
                },
            )
        raise ValueError(f"unsupported sustained fault type: {af.fault_type}")

    # ------------------------------------------------------------------ #
    # 公共 API
    # ------------------------------------------------------------------ #
    def schedule(self, fault_type: str, step: int, *, params: Optional[dict] = None,
                 duration: int = 1) -> None:
        """预定在 episode 第 `step` 步注入；duration>1 表示持续 duration 步。"""
        self._scheduled.setdefault(int(step), []).append(
            (fault_type, dict(params or {}), int(duration))
        )

    def inject(self, fault_type: str, *, step: Optional[int] = None,
               params: Optional[dict] = None, duration: int = 1):
        """立即注入。一次性故障返回 FaultRecord；持续故障返回 None (结束时产出记录)。"""
        if step is None:
            step = self._step
        step = int(step)
        params = dict(params or {})
        if duration > 1:
            af = _ActiveFault(fault_type, step, params, int(duration), int(duration))
            self._active.append(af)
            self._mutate_sustained(af)
            af.remaining -= 1
            if af.remaining == 0:
                self._records.append(self._finalize_sustained(af, step))
                self._active.remove(af)
            return None
        rec = self._apply_oneshot(fault_type, step, params)
        self._records.append(rec)
        return rec

    def _apply_oneshot(self, fault_type: str, step: int, params: dict) -> FaultRecord:
        if fault_type == FAULT_OBJECT_DISPLACEMENT:
            return self._fault_object_displacement(step, params)
        if fault_type == FAULT_OBJECT_IMPULSE:
            return self._fault_object_impulse(step, params)
        if fault_type == FAULT_GRASP_PERTURB:
            return self._fault_grasp_perturb(step, params)
        raise ValueError(f"unknown one-shot fault type: {fault_type}")

    def after_step(self, obs) -> None:
        """harness 每步调用：先触发到期的一次性/持续故障，再推进已激活的持续故障。"""
        self._step = int(obs.step)
        for fault_type, params, duration in self._scheduled.pop(self._step, []):
            if duration > 1:
                af = _ActiveFault(fault_type, self._step, params, duration, duration)
                self._active.append(af)
                self._mutate_sustained(af)
                af.remaining -= 1
                if af.remaining == 0:
                    self._records.append(self._finalize_sustained(af, self._step))
                    self._active.remove(af)
            else:
                self._records.append(self._apply_oneshot(fault_type, self._step, params))
        still_active = []
        for af in self._active:
            self._mutate_sustained(af)
            af.remaining -= 1
            if af.remaining > 0:
                still_active.append(af)
            else:
                self._records.append(self._finalize_sustained(af, self._step))
        self._active = still_active

    def reset(self) -> None:
        """清空记录/调度/持续状态（不改 rng，保证同 seed 可复现）。"""
        self._records.clear()
        self._scheduled.clear()
        self._active.clear()
        self._step = 0

    @property
    def records(self) -> List[FaultRecord]:
        return list(self._records)

    @property
    def seed(self) -> Optional[int]:
        return self._seed
