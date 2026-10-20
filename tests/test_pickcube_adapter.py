"""PickCubeAdapter + PickCubeSkill 单元测试。

覆盖:
- adapter reset/step/close 语义
- 观测形状与批量维去除 (单环境)
- terminated / truncated / success 三标志独立性
- 脚本化 pick 技能端到端成功
"""
from __future__ import annotations

import numpy as np
import pytest

from interfaces import EpisodeInfo, SkillStatus, StepResult
from sim import PickCubeAdapter, PickCubeSkill


@pytest.fixture(scope="module")
def adapter():
    a = PickCubeAdapter(max_episode_steps=200)
    yield a
    a.close()


def _hold(obs) -> np.ndarray:
    """在当前关节角保持静止、抓手打开的动作。"""
    return np.concatenate([obs.policy.qpos, [1.0]]).astype(np.float32)


class TestAdapterProperties:
    def test_num_envs(self, adapter):
        assert adapter.num_envs == 1

    def test_action_space(self, adapter):
        sp = adapter.action_space
        assert sp["shape"] == (8,)
        assert sp["dtype"] == "float32"
        assert sp["low"].shape == (8,) and sp["high"].shape == (8,)

    def test_max_episode_steps(self, adapter):
        assert adapter.max_episode_steps == 200


class TestAdapterReset:
    def test_observation_shapes_no_batch_dim(self, adapter):
        obs = adapter.reset(seed=0)
        assert obs.policy.qpos.shape == (7,)          # 去掉 batch 维
        assert obs.policy.qvel.shape == (7,)
        assert obs.policy.tcp_pose.shape == (7,)
        assert isinstance(obs.policy.gripper, float)
        assert 0.0 <= obs.policy.gripper <= 1.0
        assert obs.truth.object_poses["cube"].shape == (7,)
        assert obs.step == 0
        assert obs.timestamp == 0.0

    def test_reset_initial_flags_false(self, adapter):
        obs = adapter.reset(seed=1)
        assert obs.truth.success is False
        assert obs.truth.is_grasped is False


class TestAdapterStep:
    def test_step_returns_step_result(self, adapter):
        obs = adapter.reset(seed=0)
        res = adapter.step(_hold(obs))
        assert isinstance(res, StepResult)
        assert isinstance(res.episode, EpisodeInfo)
        assert isinstance(res.episode.terminated, bool)
        assert isinstance(res.episode.truncated, bool)
        assert isinstance(res.episode.success, bool)
        assert isinstance(res.reward, float)

    def test_step_increments_observation_step(self, adapter):
        obs = adapter.reset(seed=0)
        res = adapter.step(_hold(obs))
        assert res.observation.step == 1


class TestEpisodeSemantics:
    def test_truncated_without_success(self):
        """静止保持到 horizon -> truncated=True, success=False, terminated=False。"""
        a = PickCubeAdapter(max_episode_steps=10)
        try:
            obs = a.reset(seed=0)
            last = None
            for _ in range(10):
                last = a.step(_hold(obs))
                obs = last.observation
                if last.episode.truncated:
                    break
            assert last is not None
            assert last.episode.truncated is True
            assert last.episode.terminated is False
            assert last.episode.success is False
        finally:
            a.close()


class TestSkillEndToEnd:
    def test_pick_succeeds_and_terminates(self):
        """固定种子跑完整 pick 技能 -> 成功, 且 terminated (terminate_on_success)。"""
        a = PickCubeAdapter(max_episode_steps=200)
        try:
            obs = a.reset(seed=0)
            skill = PickCubeSkill(a)
            skill.reset(obs)
            assert skill.is_done is False
            assert len(skill._queue) > 0          # 已预规划动作队列

            terminated = False
            for _ in range(a.max_episode_steps):
                if skill.is_done:
                    break
                action = skill.act(obs)
                assert action.shape == (8,)
                assert action.dtype == np.float32
                res = a.step(action)
                obs = res.observation
                if res.episode.terminated or res.episode.truncated:
                    terminated = res.episode.terminated
                    break

            result = skill.finalize(obs)
            assert obs.truth.success is True
            assert terminated is True              # 成功触发 terminated
            assert result.status is SkillStatus.SUCCESS
            assert skill.is_done is True
        finally:
            a.close()

    def test_skill_act_shape_matches_action_space(self):
        a = PickCubeAdapter(max_episode_steps=200)
        try:
            obs = a.reset(seed=0)
            skill = PickCubeSkill(a)
            skill.reset(obs)
            action = skill.act(obs)
            assert action.shape == (8,)
            assert action.dtype == np.float32
        finally:
            a.close()
