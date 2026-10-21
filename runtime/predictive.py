"""runtime/predictive.py — 学习失败预测器的运行时集成 (M4-B / M5-B)。

把 M4 训练出的 FailurePredictor 接入恢复编排，实现「预判性恢复」：
在执行 attempt 期间每步调用 predictor.predict(extract_state_features(obs))，
当预测失败风险高时返回 PredictiveTrigger，由 executor 提前中止当前 attempt，
交 RecoveryRunner 走既有规则恢复 (RETRY/REOBSERVE/ABORT) —— max_retries 安全约束不变。

M5-B (时序门控): 保留原 B2 单帧触发的参考行为，新增 4 种 TemporalGate 变体 (不重训 MLP):
- debounce : 连续 K 帧正预测才触发 (抑制单帧噪声)。
- phase    : 仅在有意义操作相位评估预测 (抑制抓取闭合阶段的误报)。
- cooldown : 触发后冷却窗口内禁止重复触发 (避免重复信号消耗重试预算)。
- advisory : 只记录预测不中断执行 (分离预测质量与恢复决策影响)。
原 B2 (gate=None / mode=NONE) 行为不变。

M5-B 冻结结果 (验证种子 40-44, object_displacement + grasp_perturb, threshold=0.5):
  phase 门控 (gripper_closed<=0.5) 胜出: task_success 7/10 (b2=6/10), aborted 3 (b2=4),
  total_triggers 13 (b2=15)。见 FROZEN_B2T_GATE。held-out (100-129) 留待 D 最终评估。

关键约束:
- 学习预测只影响「是否提前重试」，绝不直接发底层指令 (不产动作、不改 env)。
- 推理异常 / 输入无效 / 置信度不足 / 非有限概率 -> 回退规则式行为 (返回 None 并记录 error)。
- b1 = 纯规则恢复 (guard 关闭)；b2 = 规则 + 学习预判提前触发 (参考, 单帧)。
- b2t = 规则 + 学习预判 + 时序门控 (M5-B 变体)。
- 学习模型不得绕过安全控制: 触发走的是同一 RecoveryManager.decide，重试预算 (max_retries)
  与 env 终止检查强制生效；本模块无任何 env/动作写权限。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple

import numpy as np

from interfaces import (
    FailurePrediction,
    FailurePredictor,
    RobotObservation,
    STATE_FEATURE_DIM,
    extract_state_features,
)


class RecoveryMode(str, Enum):
    B1_RULE = "b1"           # 纯规则恢复 (现有 RecoveryRunner 行为)
    B2_PREDICTIVE = "b2"     # 规则恢复 + 学习预判提前触发 (参考, 单帧)
    B2T_TEMPORAL = "b2t"     # 规则恢复 + 学习预判 + 时序门控 (M5-B)


class TemporalGateMode(str, Enum):
    NONE = "none"            # 无门控 (等价参考 B2: 单帧即触发)
    DEBOUNCE = "debounce"    # 连续 K 帧正预测才触发
    PHASE = "phase"          # 仅在有意义操作相位评估
    COOLDOWN = "cooldown"    # 触发后冷却窗口内禁止重复触发
    ADVISORY = "advisory"    # 只记录不中断


@dataclass
class TemporalGateConfig:
    """时序门控配置 (mode 决定参数子集生效)。"""

    mode: TemporalGateMode = TemporalGateMode.NONE
    # debounce: 连续 K 帧正预测才触发。
    debounce_k: int = 3
    # cooldown: 触发后 cooldown_steps 步内禁止再次触发 (跨 attempt 生效)。
    cooldown_steps: int = 10
    # phase: 相位门控信号 (仅 obs.policy 可得, 不读 obs.truth)。
    phase_signal: str = "gripper_closed"   # "gripper_closed" | "tcp_near_table"
    phase_gripper_threshold: float = 0.5   # gripper_openness <= 此值视为已闭合
    phase_tcp_z_threshold: float = 0.10    # tcp_z <= 此值视为接近物体/桌面

    def __post_init__(self) -> None:
        self.mode = TemporalGateMode(self.mode)
        if self.debounce_k < 1:
            raise ValueError("debounce_k 必须 >= 1")
        if self.cooldown_steps < 0:
            raise ValueError("cooldown_steps 必须 >= 0")
        if self.phase_signal not in ("gripper_closed", "tcp_near_table"):
            raise ValueError(f"phase_signal 非法: {self.phase_signal!r}")


class TemporalGate:
    """时序门控决策器 (加性扩展, 不改 B2 参考行为)。

    evaluate(step, single_frame_trigger, obs) -> (should_trigger, gate_reason)。
    single_frame_trigger = 单帧预测 (prob>=threshold 且 confidence>=min_confidence)。
    门控只决定是否把该单帧信号转成 PredictiveTrigger (提前中止), 不改变预测本身。
    """

    def __init__(self, config: Optional[TemporalGateConfig] = None) -> None:
        self.config = config if config is not None else TemporalGateConfig()
        self.reset()

    @property
    def mode(self) -> TemporalGateMode:
        return self.config.mode

    # ------------------------------------------------------------------ #
    # 状态生命周期
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """run 级重置 (新 episode 开始)。清空 cooldown 的最近触发步。"""
        self._consec_pos = 0
        self._last_trigger_step: Optional[int] = None

    def begin_attempt(self) -> None:
        """attempt 级重置。清空 debounce 计数; cooldown 最近触发步跨 attempt 保留。"""
        self._consec_pos = 0

    # ------------------------------------------------------------------ #
    # 相位判定 (仅 obs.policy 可观测信号, 绝不读 obs.truth)
    # ------------------------------------------------------------------ #
    def _in_phase(self, obs: RobotObservation) -> bool:
        p = getattr(obs, "policy", None)
        if p is None:
            return False
        if self.config.phase_signal == "gripper_closed":
            try:
                return float(p.gripper) <= self.config.phase_gripper_threshold
            except (TypeError, ValueError):
                return False
        # tcp_near_table
        try:
            tcp = np.asarray(getattr(p, "tcp_pose", None), dtype=np.float32).reshape(-1)
            return bool(tcp.size >= 3 and float(tcp[2]) <= self.config.phase_tcp_z_threshold)
        except (TypeError, ValueError):
            return False

    # ------------------------------------------------------------------ #
    # 门控决策
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        step: int,
        single_frame_trigger: bool,
        obs: RobotObservation,
    ) -> Tuple[bool, Optional[str]]:
        mode = self.mode
        if mode is TemporalGateMode.NONE:
            return (bool(single_frame_trigger), None)
        if mode is TemporalGateMode.DEBOUNCE:
            if single_frame_trigger:
                self._consec_pos += 1
            else:
                self._consec_pos = 0
            if self._consec_pos >= self.config.debounce_k:
                return (True, None)
            return (False, f"debounce({self._consec_pos}/{self.config.debounce_k})")
        if mode is TemporalGateMode.PHASE:
            if not self._in_phase(obs):
                return (False, "out_of_phase")
            return (bool(single_frame_trigger), None)
        if mode is TemporalGateMode.COOLDOWN:
            if not single_frame_trigger:
                return (False, None)
            if self._last_trigger_step is not None and (
                step - self._last_trigger_step
            ) < self.config.cooldown_steps:
                return (
                    False,
                    f"cooldown({step - self._last_trigger_step}<{self.config.cooldown_steps})",
                )
            self._last_trigger_step = step
            return (True, None)
        if mode is TemporalGateMode.ADVISORY:
            return (False, "advisory")
        return (False, f"unknown_mode({mode})")


# M5-B 冻结变体 (验证种子 40-44 选参结果): phase 门控, gripper 闭合后评估。
# 用法: RecoveryRunner(..., recovery_mode="b2t", temporal_gate=FROZEN_B2T_GATE)
FROZEN_B2T_GATE = TemporalGateConfig(
    mode=TemporalGateMode.PHASE,
    phase_signal="gripper_closed",
    phase_gripper_threshold=0.5,
)


@dataclass
class PredictiveTrigger:
    """预测到高风险时交回 executor 的提前触发信号 (只描述「是否提前重试」，无动作)。"""

    step: int
    prob: float
    confidence: float
    threshold: float


@dataclass
class PredictionRecord:
    """单次预测的结构化记录 (进 trace)。

    prediction  = 模型正预测 (prob >= threshold)。
    triggered   = guard 实际返回 PredictiveTrigger (提前中止)。
    gate_reason = 门控抑制原因 (仅当 prediction 成立但未触发时非空)。
    """

    step: int
    prob: float
    confidence: float
    prediction: bool
    triggered: bool
    threshold: Optional[float] = None
    gate_reason: Optional[str] = None
    error: Optional[str] = None


@dataclass
class PredictiveConfig:
    """学习预判配置 (全部可覆盖)。"""

    mode: RecoveryMode = RecoveryMode.B1_RULE
    threshold: Optional[float] = None   # None -> 用 predictor.threshold (默认 0.5)
    min_confidence: float = 0.0         # confidence 低于此值不触发 (回退规则)


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ValueError(f"invalid numeric value: {v!r}")


class PredictiveGuard:
    """学习预测器的运行时守卫：逐步预测 + 有界触发 + 安全回退 + 可选时序门控。

    step_hook 签名匹配 executor 的 step_hook 约定：返回 None (不触发) 或
    PredictiveTrigger (提前中止当前 attempt)。任何异常/无效输入都不向上抛出，
    而是记录 error 并返回 None —— 即回退到规则式恢复，不影响安全。

    gate=None 或 gate.mode=NONE 时与参考 B2 逐字等价 (单帧即触发)。
    """

    def __init__(
        self,
        predictor: FailurePredictor,
        *,
        mode: RecoveryMode = RecoveryMode.B2_PREDICTIVE,
        threshold: Optional[float] = None,
        min_confidence: float = 0.0,
        feature_fn: Callable[[RobotObservation], np.ndarray] = extract_state_features,
        gate: Optional[TemporalGateConfig] = None,
    ) -> None:
        self.predictor = predictor
        self.mode = RecoveryMode(mode)
        self.feature_fn = feature_fn
        self.min_confidence = float(min_confidence)
        self.threshold = (
            float(threshold) if threshold is not None
            else float(getattr(predictor, "threshold", 0.5))
        )
        self.gate = TemporalGate(gate if gate is not None else TemporalGateConfig())
        self._records: List[PredictionRecord] = []
        self._attempt: List[PredictionRecord] = []

    # ------------------------------------------------------------------ #
    # 状态 / 记录
    # ------------------------------------------------------------------ #
    @property
    def enabled(self) -> bool:
        return self.mode in (RecoveryMode.B2_PREDICTIVE, RecoveryMode.B2T_TEMPORAL)

    @property
    def records(self) -> List[PredictionRecord]:
        """本 run 内已 drain 的全部预测记录 (跨 attempt 汇总)。"""
        return list(self._records)

    def reset(self) -> None:
        self._records = []
        self._attempt = []
        self.gate.reset()

    def begin_attempt(self) -> None:
        self._attempt = []
        self.gate.begin_attempt()

    def drain_attempt(self) -> List[PredictionRecord]:
        """取回本次 attempt 的预测记录并清空 attempt 缓冲 (追加进 run 汇总)。"""
        recs = list(self._attempt)
        self._records.extend(recs)
        self._attempt = []
        return recs

    # ------------------------------------------------------------------ #
    # 逐步预测 (executor step_hook)
    # ------------------------------------------------------------------ #
    def step_hook(self, obs: RobotObservation) -> Optional[PredictiveTrigger]:
        if not self.enabled:
            return None
        step = int(getattr(obs, "step", -1))
        rec = PredictionRecord(
            step=step, prob=0.0, confidence=0.0, prediction=False,
            triggered=False, threshold=self.threshold,
        )
        try:
            features = np.asarray(self.feature_fn(obs), dtype=np.float32).reshape(-1)
            if features.shape != (STATE_FEATURE_DIM,):
                raise ValueError(
                    f"feature dim must be {STATE_FEATURE_DIM}, got {features.shape}"
                )
            if not np.all(np.isfinite(features)):
                raise ValueError("features contain non-finite values")
            pred: FailurePrediction = self.predictor.predict(features)
            prob = _to_float(getattr(pred, "prob", float("nan")))
            if not np.isfinite(prob) or not (0.0 <= prob <= 1.0):
                raise ValueError(f"invalid prob {prob}")
            confidence = _to_float(getattr(pred, "confidence", 0.0))
            if not np.isfinite(confidence) or not (0.0 <= confidence <= 1.0):
                raise ValueError(f"invalid confidence {confidence}")
        except Exception as exc:  # noqa: BLE001 - 运行时兜底: 回退规则
            rec.error = f"{type(exc).__name__}: {exc}"
            self._attempt.append(rec)
            return None

        rec.prob = prob
        rec.confidence = confidence
        rec.prediction = bool(prob >= self.threshold)
        single = bool(rec.prediction and confidence >= self.min_confidence)

        # 时序门控 (NONE 时等价参考 B2: 单帧即触发)。
        should_trigger, gate_reason = self.gate.evaluate(step, single, obs)
        rec.triggered = bool(should_trigger)
        rec.gate_reason = gate_reason if not rec.triggered else None
        self._attempt.append(rec)

        if rec.triggered:
            return PredictiveTrigger(
                step=step, prob=prob, confidence=confidence, threshold=self.threshold,
            )
        return None


def load_predictor_from_checkpoint(
    checkpoint_path: str,
    config_path: Optional[str] = None,
) -> FailurePredictor:
    """从 M4 checkpoint 重建 MLPFailurePredictor (懒加载 torch/learning，避免硬依赖)。

    checkpoint dict 约定 (见 learning/evaluate.py):
      state_dict / mean / std / threshold / hidden_dims / input_dim / dropout。
    config_path 仅作为缺失字段的回退 (mean/std/threshold 等通常已存进 checkpoint)。
    """
    import json
    import warnings

    import torch

    from learning.model import MLP, MLPFailurePredictor

    # checkpoint 内含 numpy 数组 (mean/std)，weights_only=True 无法加载；
    # 抑制 torch.load 的 FutureWarning (checkpoint 来自本仓库可信来源)。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg: dict = {}
    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    mean = np.asarray(ckpt.get("mean", cfg.get("mean")), dtype=np.float32)
    std = np.asarray(ckpt.get("std", cfg.get("std")), dtype=np.float32)
    threshold = float(ckpt.get("threshold", cfg.get("threshold", 0.5)))
    hidden_dims = list(ckpt.get("hidden_dims", cfg.get("hidden_dims", [32])))
    input_dim = int(ckpt.get("input_dim", cfg.get("input_dim", STATE_FEATURE_DIM)))
    dropout = float(ckpt.get("dropout", cfg.get("dropout", 0.0)))

    model = MLP(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return MLPFailurePredictor(model, mean, std, threshold=threshold, device="cpu")
