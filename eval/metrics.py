"""从一批 EpisodeResult 计算 EpisodeMetrics (success_rate / mean_steps / mean_reward / extra)。"""
from __future__ import annotations

from collections import Counter
from typing import List

import numpy as np

from interfaces import EpisodeResult, EpisodeMetrics, SkillStatus

from .source import source_counts


def compute_metrics(episodes: List[EpisodeResult]) -> EpisodeMetrics:
    """聚合一批 episode -> EpisodeMetrics。

    空列表返回全 0 的 metrics (success_rate=0.0, mean_steps=0.0, mean_reward=0.0)。
    extra 包含: num_success, num_failure, min_steps, max_steps, std_steps,
                mean_reward_success, skill_success_rate,
                failure_type_counts, source_counts。
    """
    n = len(episodes)
    if n == 0:
        return EpisodeMetrics(
            num_episodes=0, success_rate=0.0, mean_steps=0.0, mean_reward=0.0,
            extra={
                "num_success": 0, "num_failure": 0,
                "min_steps": 0.0, "max_steps": 0.0, "std_steps": 0.0,
                "mean_reward_success": 0.0, "skill_success_rate": 0.0,
                "failure_type_counts": {}, "source_counts": {},
            },
        )

    success_mask = np.array([ep.success for ep in episodes], dtype=bool)
    steps = np.array([ep.steps for ep in episodes], dtype=float)
    rewards = np.array([ep.reward for ep in episodes], dtype=float)

    num_success = int(success_mask.sum())
    num_failure = n - num_success

    # skill 成功率: SUCCESS 的 SkillResult 占全部 SkillResult 的比例 (无则 0.0)
    skill_statuses = [
        sr.status
        for ep in episodes
        for sr in ep.skill_results
    ]
    skill_total = len(skill_statuses)
    skill_success_rate = (
        sum(1 for s in skill_statuses if s is SkillStatus.SUCCESS) / skill_total
        if skill_total else 0.0
    )

    # 失败类型计数 (跨所有 episode)
    failure_type_counts = dict(Counter(
        f.type.value
        for ep in episodes
        for f in ep.failures
    ))

    return EpisodeMetrics(
        num_episodes=n,
        success_rate=float(num_success / n),
        mean_steps=float(steps.mean()),
        mean_reward=float(rewards.mean()),
        extra={
            "num_success": num_success,
            "num_failure": num_failure,
            "min_steps": float(steps.min()),
            "max_steps": float(steps.max()),
            "std_steps": float(steps.std()),
            "mean_reward_success": float(rewards[success_mask].mean()) if num_success else 0.0,
            "skill_success_rate": float(skill_success_rate),
            "failure_type_counts": failure_type_counts,
            "source_counts": source_counts(episodes),
        },
    )
