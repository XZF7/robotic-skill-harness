"""runtime/failure_detector.py — 规则式失败检测 (M3-B)。

FailureDetector 只用可观测信息 (obs.policy: qpos/qvel/tcp_pose/gripper/images)
检测失败，严禁读 obs.truth (特权真值留在评估层)。

可检测的失败类型:
- INVALID_OBS : 观测损坏 (NaN/Inf、缺失字段、gripper 越界、时间戳异常)。
- COLLISION   : 关节速度突增 (|qvel| 超阈值) 或 TCP 位姿跳变 (相邻步位移超阈值)。
- GRIP_FAIL   : 抓手全闭 (开度 <= eps)，即「夹空」——两指之间没有物体。
- DROP        : 曾处于抓取窗 (两指夹持物体 => 部分开度) 后滑脱
                 (开度跌回全闭或升至全开)。需观测历史。
- 其余 (MOTION_FAIL/TIMEOUT/UNKNOWN/…) : 由技能/executor 自报 FailureEvent 规范化。

单一入口 detect(): 返回规范化 FailureEvent 或 None (无失败)。
优先级: INVALID_OBS (观测损坏硬信号) > 技能/executor 自报 failure >
         可观测异常启发式 > FAILURE 无 failure -> UNKNOWN。
SUCCESS / ABORTED 不算可恢复失败 (返回 None)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from interfaces import FailureEvent, FailureType, RobotObservation, SkillResult, SkillStatus

# 观测字段名 -> 解释 (仅用于诊断信息)。
_POLICY_FIELDS = ("qpos", "qvel", "tcp_pose")


@dataclass
class FailureDetectorConfig:
    """失败检测阈值配置 (全部可覆盖)。"""

    # COLLISION: 单步最大关节速度 (rad/s) 超阈值 -> 碰撞。
    collision_qvel_threshold: Optional[float] = 20.0
    # COLLISION: 相邻观测 TCP 位置位移 (m) 超阈值 -> 碰撞/瞬移。
    collision_tcp_delta_threshold: Optional[float] = 0.1
    # GRIP_FAIL: 抓手开度 <= eps 视为「全闭夹空」。
    grasp_close_eps: float = 0.02
    # DROP 抓取窗: 开度落在此区间视为「两指夹持物体」。
    drop_grasp_low: float = 0.05
    drop_grasp_high: float = 0.95
    # DROP: 抓取窗后开度升至 >= drop_open_eps 视为「释放/打开后物体丢失」。
    drop_open_eps: float = 0.9
    # 是否启用可观测异常启发式 (COLLISION/GRIP_FAIL/DROP)。
    enable_policy_heuristics: bool = True


class FailureDetector:
    """规则式失败检测器。

    只用 obs.policy 与技能自报的 SkillResult/FailureEvent，绝不读 obs.truth。
    确定性: 相同输入 -> 相同输出。
    """

    def __init__(self, config: Optional[FailureDetectorConfig] = None) -> None:
        self.config = config if config is not None else FailureDetectorConfig()

    # ------------------------------------------------------------------ #
    # 单观测完整性校验 (INVALID_OBS)
    # ------------------------------------------------------------------ #
    def validate_observation(self, obs: Optional[RobotObservation]) -> Optional[FailureEvent]:
        """校验单个观测的策略字段是否有效。仅用 obs.policy。

        返回 INVALID_OBS FailureEvent 或 None。
        """
        if obs is None:
            return FailureEvent(
                type=FailureType.INVALID_OBS, step=-1, timestamp=0.0,
                detail={"reason": "observation is None"},
            )
        p = obs.policy
        if p is None:
            return FailureEvent(
                type=FailureType.INVALID_OBS, step=obs.step, timestamp=_safe_ts(obs),
                detail={"reason": "policy is None"},
            )
        for name in _POLICY_FIELDS:
            arr = getattr(p, name, None)
            if arr is None:
                return FailureEvent(
                    type=FailureType.INVALID_OBS, step=obs.step, timestamp=_safe_ts(obs),
                    detail={"reason": f"policy.{name} is None"},
                )
            arr = np.asarray(arr)
            if arr.size == 0 or not np.all(np.isfinite(arr)):
                return FailureEvent(
                    type=FailureType.INVALID_OBS, step=obs.step, timestamp=_safe_ts(obs),
                    detail={"reason": f"policy.{name} has non-finite values",
                            "field": name},
                )
        g = p.gripper
        if g is None or not np.isfinite(float(g)):
            return FailureEvent(
                type=FailureType.INVALID_OBS, step=obs.step, timestamp=_safe_ts(obs),
                detail={"reason": "policy.gripper is non-finite"},
            )
        if not (0.0 <= float(g) <= 1.0):
            return FailureEvent(
                type=FailureType.INVALID_OBS, step=obs.step, timestamp=_safe_ts(obs),
                detail={"reason": "policy.gripper out of [0,1]",
                        "gripper": float(g)},
            )
        if not np.isfinite(float(obs.timestamp)) or float(obs.timestamp) < 0.0:
            return FailureEvent(
                type=FailureType.INVALID_OBS, step=obs.step, timestamp=_safe_ts(obs),
                detail={"reason": "observation timestamp invalid"},
            )
        return None

    # ------------------------------------------------------------------ #
    # 可观测异常启发式 (COLLISION / GRIP_FAIL / DROP)
    # ------------------------------------------------------------------ #
    def detect_policy_anomaly(
        self,
        history: Sequence[RobotObservation],
    ) -> Optional[FailureEvent]:
        """从观测历史 (仅 policy) 检测可观测异常。history 至少 1 个观测。

        返回 FailureEvent (source="policy") 或 None。
        """
        if not self.config.enable_policy_heuristics:
            return None
        history = [o for o in history if o is not None]
        if not history:
            return None

        # 1) COLLISION: |qvel| 突增 或 TCP 位移跳变。
        ev = self._detect_collision(history)
        if ev is not None:
            return ev

        # 2) DROP: 曾在抓取窗后滑脱 (需历史)。
        ev = self._detect_drop(history)
        if ev is not None:
            return ev

        # 3) GRIP_FAIL: 最终抓手全闭夹空 (单观测即可)。
        final = history[-1]
        ev = self._detect_grip_fail(final)
        if ev is not None:
            return ev

        return None

    def _detect_collision(self, history: Sequence[RobotObservation]) -> Optional[FailureEvent]:
        cfg = self.config
        for obs in history:
            qvel = np.asarray(obs.policy.qvel)
            if qvel.size and cfg.collision_qvel_threshold is not None:
                if float(np.max(np.abs(qvel))) > cfg.collision_qvel_threshold:
                    return FailureEvent(
                        type=FailureType.COLLISION, step=obs.step,
                        timestamp=_safe_ts(obs),
                        detail={"source": "policy", "reason": "qvel spike",
                                "max_qvel": float(np.max(np.abs(qvel))),
                                "threshold": cfg.collision_qvel_threshold},
                    )
        if cfg.collision_tcp_delta_threshold is not None and len(history) >= 2:
            for a, b in zip(history[:-1], history[1:]):
                pa = np.asarray(a.policy.tcp_pose)[:3]
                pb = np.asarray(b.policy.tcp_pose)[:3]
                if pa.size == 3 and pb.size == 3:
                    delta = float(np.linalg.norm(pb - pa))
                    if delta > cfg.collision_tcp_delta_threshold:
                        return FailureEvent(
                            type=FailureType.COLLISION, step=b.step,
                            timestamp=_safe_ts(b),
                            detail={"source": "policy", "reason": "tcp teleport",
                                    "delta": delta,
                                    "threshold": cfg.collision_tcp_delta_threshold},
                        )
        return None

    def _detect_grip_fail(self, obs: RobotObservation) -> Optional[FailureEvent]:
        g = float(obs.policy.gripper)
        if g <= self.config.grasp_close_eps:
            return FailureEvent(
                type=FailureType.GRIP_FAIL, step=obs.step, timestamp=_safe_ts(obs),
                detail={"source": "policy", "reason": "gripper fully closed (grasped nothing)",
                        "gripper": g, "eps": self.config.grasp_close_eps},
            )
        return None

    def _detect_drop(self, history: Sequence[RobotObservation]) -> Optional[FailureEvent]:
        cfg = self.config
        holding_seen = False
        for obs in history:
            g = float(obs.policy.gripper)
            if cfg.drop_grasp_low <= g <= cfg.drop_grasp_high:
                holding_seen = True
            elif holding_seen and (g <= cfg.grasp_close_eps or g >= cfg.drop_open_eps):
                return FailureEvent(
                    type=FailureType.DROP, step=obs.step, timestamp=_safe_ts(obs),
                    detail={"source": "policy", "reason": "object lost after grasp",
                            "gripper": g},
                )
        return None

    # ------------------------------------------------------------------ #
    # 权威入口
    # ------------------------------------------------------------------ #
    def detect(
        self,
        result: SkillResult,
        final_obs: Optional[RobotObservation] = None,
        obs_history: Optional[Sequence[RobotObservation]] = None,
    ) -> Optional[FailureEvent]:
        """权威检测: 返回规范化 FailureEvent 或 None (无失败)。

        result : executor 产出的唯一权威 SkillResult (技能/executor 自报)。
        final_obs / obs_history : 仅用其 policy 字段, 绝不读 truth。
        """
        # SUCCESS / ABORTED 不算可恢复失败。
        if result.status is SkillStatus.SUCCESS:
            # 但仍校验观测完整性: 成功却观测损坏 -> INVALID_OBS。
            inv = self.validate_observation(final_obs)
            return inv
        if result.status is SkillStatus.ABORTED:
            return None

        # 观测损坏 (硬信号) 优先。
        inv = self.validate_observation(final_obs)
        if inv is not None:
            return inv

        # 技能/executor 自报 failure。
        if result.failure is not None:
            return _canonicalize(result.failure, source="skill")

        # 可观测异常启发式 (仅当技能未自报具体类型时)。
        seq: List[RobotObservation] = list(obs_history) if obs_history else []
        if final_obs is not None and (not seq or seq[-1] is not final_obs):
            seq.append(final_obs)
        anomaly = self.detect_policy_anomaly(seq)
        if anomaly is not None:
            return anomaly

        # FAILURE / TIMEOUT 但无具体类型 -> UNKNOWN。
        return FailureEvent(
            type=FailureType.UNKNOWN, step=result.steps, timestamp=0.0,
            detail={"source": "detector", "reason": "no specific failure type",
                    "status": result.status.value},
        )


def _safe_ts(obs: RobotObservation) -> float:
    ts = float(getattr(obs, "timestamp", 0.0))
    return ts if np.isfinite(ts) else 0.0


def _canonicalize(failure: FailureEvent, source: str) -> FailureEvent:
    """复制 FailureEvent 并标注来源，不改动原对象。"""
    detail = dict(failure.detail)
    detail.setdefault("source", source)
    return FailureEvent(
        type=failure.type, step=failure.step, timestamp=failure.timestamp,
        detail=detail,
    )
