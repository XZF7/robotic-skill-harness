"""EpisodeResult 序列化 round-trip 测试。"""
import json

import numpy as np

from interfaces import (
    EpisodeResult, SkillResult, SkillStatus, FailureEvent, FailureType,
)
from eval.serialization import (
    episode_to_dict, episode_from_dict,
    episode_to_json, episode_from_json,
    episodes_to_dicts, episodes_from_dicts,
    episodes_to_json, episodes_from_json,
)


def _native_episode(source="mock"):
    """纯 Python 原生类型 episode, 用于精确相等 round-trip。"""
    f = FailureEvent(type=FailureType.COLLISION, step=7, timestamp=0.35, detail={"obj": "cube", "vel": 1.5})
    s = SkillResult(status=SkillStatus.FAILURE, skill_id="pick", steps=7, message="slipped", failure=f)
    return EpisodeResult(
        task_id="PickCube-v1", success=False, steps=12, reward=0.25,
        skill_results=[s], failures=[f],
        metrics={"max_vel": 2.0, "traj_len": [1, 2, 3]},
        source=source,
    )


def _numpy_episode(source="mock"):
    """含 numpy 值的 episode: JSON 往返会把 numpy 归一化为原生类型。"""
    f = FailureEvent(type=FailureType.COLLISION, step=7, timestamp=0.35,
                     detail={"obj": "cube", "vel": np.float32(1.5)})
    s = SkillResult(status=SkillStatus.FAILURE, skill_id="pick", steps=7, message="slipped", failure=f)
    return EpisodeResult(
        task_id="PickCube-v1", success=False, steps=12, reward=0.25,
        skill_results=[s], failures=[f],
        metrics={"max_vel": np.float64(2.0), "traj_len": np.array([1, 2, 3])},
        source=source,
    )


class TestEpisodeRoundTrip:
    def test_dict_roundtrip_full(self):
        ep = _native_episode()
        assert episode_from_dict(episode_to_dict(ep)) == ep

    def test_json_roundtrip_full(self):
        ep = _native_episode()
        assert episode_from_json(episode_to_json(ep)) == ep

    def test_json_is_valid_json(self):
        json.loads(episode_to_json(_native_episode()))  # 不得抛异常

    def test_roundtrip_normalizes_numpy(self):
        ep = _numpy_episode()
        back = episode_from_json(episode_to_json(ep))
        # JSON 无法保留 numpy: 往返后归一化为原生类型, 值不变
        assert back.metrics["max_vel"] == 2.0
        assert back.metrics["traj_len"] == [1, 2, 3]
        assert back.failures[0].detail["vel"] == 1.5
        assert episode_to_dict(back) == episode_to_dict(ep)

    def test_enums_restored_as_enum(self):
        back = episode_from_json(episode_to_json(_native_episode()))
        assert back.skill_results[0].status is SkillStatus.FAILURE
        assert back.failures[0].type is FailureType.COLLISION

    def test_numpy_coerced_to_native(self):
        d = episode_to_dict(_numpy_episode())
        assert isinstance(d["metrics"]["max_vel"], float)
        assert isinstance(d["metrics"]["traj_len"], list)

    def test_source_preserved(self):
        for src in ("mock", "sim"):
            back = episode_from_json(episode_to_json(_native_episode(source=src)))
            assert back.source == src

    def test_default_source_mock_on_missing(self):
        d = episode_to_dict(_native_episode())
        d.pop("source")
        assert episode_from_dict(d).source == "mock"


class TestBatch:
    def test_batch_roundtrip(self):
        eps = [_native_episode("mock"), _native_episode("sim")]
        assert episodes_from_dicts(episodes_to_dicts(eps)) == eps

    def test_batch_json_roundtrip(self):
        eps = [_native_episode("mock"), _native_episode("sim")]
        assert episodes_from_json(episodes_to_json(eps)) == eps

    def test_empty_batch(self):
        assert episodes_from_json(episodes_to_json([])) == []
