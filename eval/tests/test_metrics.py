"""指标计算测试。"""
from interfaces import EpisodeResult, SkillResult, SkillStatus, FailureEvent, FailureType
from eval.metrics import compute_metrics


def _ep(task_id, success, steps, reward, *, source="sim", skills=None, failures=None):
    return EpisodeResult(
        task_id=task_id, success=success, steps=steps, reward=reward,
        skill_results=skills or [], failures=failures or [], source=source,
    )


class TestBasicMetrics:
    def test_success_rate_and_means(self):
        eps = [
            _ep("t", True, 10, 1.0),
            _ep("t", True, 20, 1.0),
            _ep("t", False, 40, 0.0),
            _ep("t", False, 50, 0.1),
        ]
        m = compute_metrics(eps)
        assert m.num_episodes == 4
        assert m.success_rate == 0.5
        assert m.mean_steps == 30.0
        assert m.mean_reward == 0.525

    def test_all_success(self):
        eps = [_ep("t", True, 10, 1.0) for _ in range(3)]
        m = compute_metrics(eps)
        assert m.success_rate == 1.0
        assert m.extra["num_success"] == 3
        assert m.extra["num_failure"] == 0

    def test_empty(self):
        m = compute_metrics([])
        assert m.num_episodes == 0
        assert m.success_rate == 0.0
        assert m.mean_steps == 0.0
        assert m.mean_reward == 0.0


class TestExtra:
    def test_step_stats(self):
        eps = [_ep("t", True, s, 1.0) for s in (10, 20, 30)]
        m = compute_metrics(eps)
        assert m.extra["min_steps"] == 10.0
        assert m.extra["max_steps"] == 30.0
        assert abs(m.extra["std_steps"] - (200 / 3) ** 0.5) < 1e-9

    def test_mean_reward_success(self):
        eps = [_ep("t", True, 10, 1.0), _ep("t", True, 10, 0.5), _ep("t", False, 10, 0.0)]
        m = compute_metrics(eps)
        assert m.extra["mean_reward_success"] == 0.75

    def test_skill_success_rate(self):
        eps = [
            _ep("t", True, 10, 1.0, skills=[SkillResult(SkillStatus.SUCCESS, "pick", 10)]),
            _ep("t", False, 10, 0.0, skills=[SkillResult(SkillStatus.FAILURE, "pick", 10)]),
        ]
        m = compute_metrics(eps)
        assert m.extra["skill_success_rate"] == 0.5

    def test_skill_success_rate_empty(self):
        m = compute_metrics([_ep("t", True, 10, 1.0)])
        assert m.extra["skill_success_rate"] == 0.0

    def test_failure_type_counts(self):
        eps = [
            _ep("t", False, 10, 0.0, failures=[FailureEvent(FailureType.GRIP_FAIL, 10, 0.5)]),
            _ep("t", False, 10, 0.0, failures=[FailureEvent(FailureType.GRIP_FAIL, 10, 0.5),
                                                FailureEvent(FailureType.TIMEOUT, 10, 0.5)]),
        ]
        m = compute_metrics(eps)
        assert m.extra["failure_type_counts"] == {"grip_fail": 2, "timeout": 1}

    def test_source_counts(self):
        eps = [_ep("t", True, 10, 1.0, source="sim"), _ep("t", True, 10, 1.0, source="mock")]
        m = compute_metrics(eps)
        assert m.extra["source_counts"] == {"sim": 1, "mock": 1}
