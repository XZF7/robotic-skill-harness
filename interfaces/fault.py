"""Fault injection contract: 故障注入记录。

由 sim 故障注入 (M3-A) 产生，评估层 (M3-D) 消费以计算『已验证故障注入率』，
并区分真正注入成功的故障场景 (恢复成功率只在有效注入场景上度量)。

约束 (M3-A):
- 故障必须真实影响仿真物理行为，不只是一个标志位。
- injected 记录注入是否真实成功 (须核实，不能只凭意图)。
- 不得通过 reset 来『伪恢复』。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FaultRecord:
    fault_type: str          # 故障类型 (如 "object_displacement", "grasp_perturb")
    injected: bool           # 是否真实注入成功
    step: int = 0            # 注入时机 (episode 步数, -1 表示 reset 前)
    params: dict = field(default_factory=dict)   # 注入参数 (幅度/方向等)
    detail: dict = field(default_factory=dict)   # 额外诊断 / 注入失败原因
