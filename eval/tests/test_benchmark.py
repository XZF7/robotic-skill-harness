"""基准配置结构测试。"""
import pytest

from eval.benchmark import BenchmarkConfig, BenchmarkRun
from eval.mocks import make_mock_episodes


def _config(**kw):
    defaults = dict(name="m1", tasks=["PickCube-v1", "StackCube-v1"], seeds=[0, 1, 2], episodes_per_task=5)
    defaults.update(kw)
    return BenchmarkConfig(**defaults)


class TestConfigValidation:
    def test_valid(self):
        cfg = _config()
        assert cfg.total_episodes == 2 * 3 * 5

    @pytest.mark.parametrize("bad", [
        dict(name=""), dict(tasks=[]), dict(seeds=[]),
        dict(episodes_per_task=0), dict(max_steps=0), dict(source="bogus"),
        dict(seeds=["a"]),
    ])
    def test_invalid(self, bad):
        with pytest.raises(ValueError):
            _config(**bad)

    def test_iter_cases_count_and_order(self):
        cfg = _config(tasks=["A", "B"], seeds=[1, 2], episodes_per_task=3)
        cases = list(cfg.iter_cases())
        assert len(cases) == 12
        assert cases[0] == ("A", 1, 0)
        assert cases[-1] == ("B", 2, 2)


class TestConfigSerialization:
    def test_dict_roundtrip(self):
        cfg = _config(extra={"note": "x"})
        assert BenchmarkConfig.from_dict(cfg.to_dict()) == cfg

    def test_json_roundtrip(self):
        cfg = _config()
        assert BenchmarkConfig.from_json(cfg.to_json()) == cfg


class TestBenchmarkRun:
    def test_per_task_grouping(self):
        cfg = _config(tasks=["A", "B"], seeds=[0], episodes_per_task=3)
        eps = make_mock_episodes("A", 3) + make_mock_episodes("B", 3, seed=1)
        run = BenchmarkRun.from_episodes(cfg, eps)
        assert run.metrics.num_episodes == 6
        assert set(run.per_task_metrics.keys()) == {"A", "B"}
        assert run.per_task_metrics["A"].num_episodes == 3
        assert run.per_task_metrics["B"].num_episodes == 3
