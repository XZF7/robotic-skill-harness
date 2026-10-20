"""SkillRegistry: 按 skill_id 注册/查找/列出技能。"""
from __future__ import annotations

from typing import Dict, Iterator, List, Optional

from interfaces import RobotSkill

from .errors import DuplicateSkillError, SkillNotFoundError


class SkillRegistry:
    """技能注册表。

    键为 skill.skill_id (字符串)。重复注册报 DuplicateSkillError；
    查找缺失报 SkillNotFoundError。列表返回排序后的 skill_id。
    """

    def __init__(self) -> None:
        self._skills: Dict[str, RobotSkill] = {}

    def register(self, skill: RobotSkill) -> str:
        skill_id = getattr(skill, "skill_id", None)
        if not skill_id:
            raise ValueError("skill 必须提供非空 skill_id")
        if skill_id in self._skills:
            raise DuplicateSkillError(f"skill_id 已注册: {skill_id!r}")
        self._skills[skill_id] = skill
        return skill_id

    def unregister(self, skill_id: str) -> Optional[RobotSkill]:
        return self._skills.pop(skill_id, None)

    def lookup(self, skill_id: str) -> RobotSkill:
        try:
            return self._skills[skill_id]
        except KeyError:
            raise SkillNotFoundError(f"未注册的 skill_id: {skill_id!r}") from None

    def get(self, skill_id: str, default=None) -> Optional[RobotSkill]:
        return self._skills.get(skill_id, default)

    def list_skills(self) -> List[str]:
        return sorted(self._skills.keys())

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, skill_id: str) -> bool:
        return skill_id in self._skills

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._skills))
