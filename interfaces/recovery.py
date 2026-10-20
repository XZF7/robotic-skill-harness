"""Recovery contract: 恢复动作/决策/结果 (M3 失败恢复的共享契约)。

由 Recovery Manager (runtime, M3-B) 产生，评估层 (eval, M3-D) 消费以计算恢复指标。

约束 (M3-B):
- 恢复在同一 episode 内进行 (env 仍有效)，不得把新 reset 当作成功恢复。
- 每次技能尝试产生恰好一个权威 SkillResult；多次尝试属同一任务，产出唯一最终结果。
- 失败检测用可观测信息 (obs.policy)，特权真值 (obs.truth) 仅留在评估层。
- 规则式决策，不依赖 LLM。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List

from .skill import SkillResult


class RecoveryAction(str, Enum):
    RETRY = "retry"          # 重新执行技能
    REOBSERVE = "reobserve"  # 重新观测场景后重试
    ABORT = "abort"          # 无法恢复，安全终止


@dataclass
class RecoveryDecision:
    action: RecoveryAction
    retry_count: int         # 当前已重试次数 (0 表示首次决策)
    reason: str = ""
    detail: dict = field(default_factory=dict)


@dataclass
class RecoveryResult:
    attempts: int                       # 总尝试次数 (1 + retries)
    final_skill_result: SkillResult     # 最后一次尝试的权威结果
    decisions: List[RecoveryDecision] = field(default_factory=list)
    recovered: bool = False             # 是否在某次重试后成功
    aborted: bool = False               # 是否因无法恢复而中止
