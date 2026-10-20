"""episode 级切分测试: 防帧泄漏 / held-out seeds / held-out 扰动配置。"""
import pytest

from interfaces import EpisodeRecord
from eval.learning_split import LearningSplit, split_episodes, verify_split
from eval.learning_synthetic import make_synthetic_episode, make_synthetic_episodes


def _all_ids(split: LearningSplit):
    return (
        [ep.episode_id for ep in split.train]
        + [ep.episode_id for ep in split.val]
        + [ep.episode_id for ep in split.test]
    )


def _mk(n, fault_types, **kw):
    return make_synthetic_episodes(
        seeds=list(range(n)),
        fault_types=fault_types,
        failed_frac=0.5,
        **kw,
    )


class TestAtomicityAndLeakage:
    def test_every_episode_appears_exactly_once(self):
        eps = _mk(20, ["object_displacement", "grasp_perturb"])
        split = split_episodes(eps, val_frac=0.15, test_frac=0.15, seed=42)
        ids = _all_ids(split)
        assert len(ids) == len(eps)
        assert len(set(ids)) == len(eps), "episode 跨 split 重复"
        assert set(ids) == {ep.episode_id for ep in eps}

    def test_no_frame_level_split(self):
        """同一 episode 的所有帧整体保留在同一 split (帧不跨 split)。"""
        eps = _mk(12, ["a", "b"])
        split = split_episodes(eps, val_frac=0.15, test_frac=0.15, seed=1)
        by_id = {ep.episode_id: ep for ep in eps}
        for part in (split.train, split.val, split.test):
            for ep in part:
                assert len(ep.frames) == len(by_id[ep.episode_id].frames)

    def test_held_out_seeds_disjoint(self):
        eps = _mk(30, ["a", "b", "c"])
        split = split_episodes(eps, val_frac=0.15, test_frac=0.15, seed=7)
        t, v, tr = set(split.test_seeds), set(split.val_seeds), set(split.train_seeds)
        assert not (t & tr) and not (t & v) and not (v & tr)
        assert split._seeds_disjoint()

    def test_verify_split_ok(self):
        split = split_episodes(_mk(10, ["a", "b"]), val_frac=0.15, test_frac=0.15, seed=0)
        verify_split(split)  # 不抛即通过


class TestHeldOutFaultType:
    def test_held_out_goes_to_test_only(self):
        eps = _mk(30, ["object_displacement", "grasp_perturb", "object_impulse"])
        split = split_episodes(
            eps, val_frac=0.15, test_frac=0.15, seed=3,
            held_out_fault_types=["object_impulse"],
        )
        held = [ep for ep in eps if ep.fault_type == "object_impulse"]
        assert held, "测试数据须含 held-out fault_type"
        test_ft = {ep.fault_type for ep in split.test}
        train_val_ft = {ep.fault_type for ep in split.train + split.val}
        assert "object_impulse" in test_ft
        assert "object_impulse" not in train_val_ft
        # 所有 held-out episode 都在 test。
        held_ids = {ep.episode_id for ep in held}
        assert held_ids <= {ep.episode_id for ep in split.test}

    def test_mixed_held_out_seed_raises(self):
        """同一 seed 同时含 held-out 与非 held-out fault_type -> ValueError。"""
        e1 = make_synthetic_episode("x", seed=7, fault_type="object_impulse")
        e2 = make_synthetic_episode("y", seed=7, fault_type="object_displacement")
        with pytest.raises(ValueError):
            split_episodes(
                [e1, e2], held_out_fault_types=["object_impulse"],
            )


class TestSeedGrouping:
    def test_same_seed_not_split(self):
        """相同 seed 的多个 episode 必须同 split。"""
        e1 = make_synthetic_episode("e1", seed=7, fault_type="a")
        e2 = make_synthetic_episode("e2", seed=7, fault_type="a")
        others = make_synthetic_episodes(seeds=[0, 1, 2, 3, 4, 5], fault_types=["a", "b"])
        split = split_episodes([e1, e2] + others, val_frac=0.15, test_frac=0.15, seed=0)
        def loc(eid):
            for name, part in (("train", split.train), ("val", split.val), ("test", split.test)):
                if eid in {ep.episode_id for ep in part}:
                    return name
            return None
        assert loc("e1") == loc("e2") is not None


class TestDeterminismAndBalance:
    def test_deterministic(self):
        eps = _mk(25, ["a", "b", "c"])
        a = split_episodes(eps, val_frac=0.15, test_frac=0.15, seed=5)
        b = split_episodes(eps, val_frac=0.15, test_frac=0.15, seed=5)
        assert a.to_dict() == b.to_dict()

    def test_stratified_balance(self):
        """每层 10 个 seed 时, test 集应含全部 fault_type (分层尽量均衡)。"""
        eps = _mk(30, ["a", "b", "c"])
        split = split_episodes(eps, val_frac=0.15, test_frac=0.15, seed=9)
        assert {ep.fault_type for ep in split.test} == {"a", "b", "c"}

    def test_summary_raw_counts(self):
        eps = _mk(20, ["a", "b"])
        split = split_episodes(eps, val_frac=0.15, test_frac=0.15, seed=2)
        s = split.summary()
        total_eps = s["train"]["num_episodes"] + s["val"]["num_episodes"] + s["test"]["num_episodes"]
        assert total_eps == 20
        total_frames = (
            s["train"]["num_frames"] + s["val"]["num_frames"] + s["test"]["num_frames"]
        )
        assert total_frames == sum(len(ep.frames) for ep in eps)


class TestErrors:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            split_episodes([])

    def test_duplicate_episode_id_raises(self):
        e1 = make_synthetic_episode("dup", seed=1, fault_type="a")
        e2 = make_synthetic_episode("dup", seed=2, fault_type="a")
        with pytest.raises(ValueError):
            split_episodes([e1, e2])

    def test_bad_fracs_raise(self):
        eps = _mk(10, ["a"])
        with pytest.raises(ValueError):
            split_episodes(eps, val_frac=0.6, test_frac=0.6)

    def test_verify_split_detects_leak(self):
        e1 = make_synthetic_episode("a", seed=1, fault_type="x")
        e2 = make_synthetic_episode("b", seed=2, fault_type="x")
        bad = LearningSplit(train=[e1], val=[], test=[e1, e2])
        with pytest.raises(AssertionError):
            verify_split(bad)
