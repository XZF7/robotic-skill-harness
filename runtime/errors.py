"""Runtime 领域异常。"""
from __future__ import annotations


class SkillNotFoundError(KeyError):
    """按 skill_id 查找技能未命中。"""


class DuplicateSkillError(ValueError):
    """重复注册相同 skill_id。"""


class InvalidTransitionError(RuntimeError):
    """任务状态机的非法状态转移。"""
