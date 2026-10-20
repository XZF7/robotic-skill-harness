"""source 字段 mock/sim 区分与守卫测试。"""
import pytest

from eval.source import (
    VALID_SOURCES, is_mock, is_sim, validate_source,
    source_counts, filter_by_source, require_sim, require_all_mock,
)
from eval.mocks import make_mock_episodes
from interfaces import EpisodeResult


def _ep(source):
    return EpisodeResult(task_id="t", success=True, steps=10, reward=1.0, source=source)


class TestPredicates:
    def test_is_mock_sim(self):
        assert is_mock(_ep("mock")) and not is_mock(_ep("sim"))
        assert is_sim(_ep("sim")) and not is_sim(_ep("mock"))


class TestValidate:
    def test_valid_sources(self):
        validate_source([_ep("mock"), _ep("sim")])  # 不抛

    def test_invalid_source_raises(self):
        with pytest.raises(ValueError):
            validate_source([_ep("bogus")])


class TestCountsAndFilter:
    def test_source_counts(self):
        eps = [_ep("mock"), _ep("mock"), _ep("sim")]
        assert source_counts(eps) == {"mock": 2, "sim": 1}

    def test_filter_by_source(self):
        eps = [_ep("mock"), _ep("sim"), _ep("mock")]
        assert len(filter_by_source(eps, "sim")) == 1
        assert len(filter_by_source(eps, "mock")) == 2

    def test_filter_invalid_source_raises(self):
        with pytest.raises(ValueError):
            filter_by_source([_ep("mock")], "bogus")


class TestGuard:
    def test_require_sim_passes_on_sim(self):
        eps = [_ep("sim"), _ep("sim")]
        assert require_sim(eps) == eps

    def test_require_sim_blocks_mock(self):
        with pytest.raises(ValueError, match="mock"):
            require_sim([_ep("sim"), _ep("mock")])

    def test_require_sim_blocks_all_mock(self):
        with pytest.raises(ValueError, match="mock"):
            require_sim(make_mock_episodes("t", 3))

    def test_require_all_mock_passes(self):
        eps = make_mock_episodes("t", 3)
        assert require_all_mock(eps) == eps

    def test_require_all_mock_blocks_sim(self):
        with pytest.raises(ValueError, match="sim"):
            require_all_mock([_ep("sim")])


class TestEndToEndGuard:
    def test_mock_never_reported_as_benchmark(self):
        """集成真实仿真前: 开发期 mock 数据无法通过上报守卫。"""
        dev_eps = make_mock_episodes("PickCube-v1", 20, success_rate=0.7)
        with pytest.raises(ValueError):
            require_sim(dev_eps, context="M1 benchmark report")
