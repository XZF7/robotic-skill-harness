"""M4 学习契约测试: 数据契约有效性 + 特征泄漏防护。"""
import numpy as np
import pytest

from interfaces import (
    PolicyObservation, GroundTruth, RobotObservation,
    extract_state_features, label_failure,
    STATE_FEATURE_DIM, STATE_FEATURE_NAMES,
    DEFAULT_PREDICTION_HORIZON,
    FrameRecord, EpisodeRecord, FailurePrediction, FailurePredictor,
)


def _obs(success=False, object_pose=None):
    policy = PolicyObservation(
        qpos=np.arange(7, dtype=np.float32),
        qvel=np.arange(7, dtype=np.float32) * 0.1,
        tcp_pose=np.array([0.1, 0.2, 0.3, 1, 0, 0, 0], dtype=np.float32),
        gripper=0.5,
    )
    truth = GroundTruth(
        object_poses=object_pose or {"cube": np.zeros(7)},
        success=success,
        is_grasped=False,
    )
    return RobotObservation(policy=policy, truth=truth, step=0, timestamp=0.0)


class TestFeatureSchema:
    def test_feature_dim(self):
        feats = extract_state_features(_obs())
        assert feats.shape == (STATE_FEATURE_DIM,)
        assert feats.dtype == np.float32

    def test_feature_names_count(self):
        assert len(STATE_FEATURE_NAMES) == STATE_FEATURE_DIM == 22

    def test_no_nan_inf(self):
        feats = extract_state_features(_obs())
        assert np.all(np.isfinite(feats))

    def test_feature_leakage_truth_ignored(self):
        """特征只读 obs.policy：改变 obs.truth 不得影响特征 (泄漏防护)。"""
        a = _obs(success=False, object_pose={"cube": np.zeros(7)})
        b = _obs(success=True, object_pose={"cube": np.ones(7) * 9.9})
        b.truth.is_grasped = True
        assert np.array_equal(extract_state_features(a), extract_state_features(b)), \
            "特征不得依赖特权真值 obs.truth"

    def test_feature_leakage_no_success_field(self):
        """确认特征提取不触碰 success/is_grasped (不存在于 policy)。"""
        obs = _obs()
        feats = extract_state_features(obs)
        # 特征长度固定 22，不随 truth 变化
        assert feats.shape == (22,)


class TestFailureLabeling:
    def test_failed_last_horizon_positive(self):
        steps = [0, 10, 50, 60, 70, 74]
        labels = label_failure(steps, failed=True, t_end=74, horizon=20)
        # t_end - step <= 20 => step >= 54 -> 正样本
        assert labels.tolist() == [0, 0, 0, 1, 1, 1]

    def test_success_all_zero(self):
        steps = [0, 10, 50, 71]
        labels = label_failure(steps, failed=False, t_end=71, horizon=20)
        assert labels.tolist() == [0, 0, 0, 0]

    def test_horizon_boundary(self):
        steps = [54, 55]
        labels = label_failure(steps, failed=True, t_end=74, horizon=20)
        assert labels.tolist() == [1, 1]  # 74-54=20 <= 20

    def test_default_horizon(self):
        assert DEFAULT_PREDICTION_HORIZON == 20


class TestDatasetSchema:
    def test_frame_record(self):
        f = FrameRecord(step=3, features=np.zeros(22, dtype=np.float32), label=1)
        assert f.features.shape == (22,) and f.label == 1

    def test_episode_record_metadata_separate(self):
        ep = EpisodeRecord(
            episode_id="ep0", seed=0, fault_type="object_displacement",
            outcome=False,
            frames=[FrameRecord(step=0, features=np.zeros(22, dtype=np.float32), label=0)],
        )
        assert ep.outcome is False and ep.fault_type == "object_displacement"
        assert len(ep.frames) == 1


class TestPredictorContract:
    def test_predictor_abstract(self):
        with pytest.raises(TypeError):
            FailurePredictor()

    def test_prediction_dataclass(self):
        p = FailurePrediction(prob=0.9, prediction=True, confidence=0.8)
        assert 0.0 <= p.prob <= 1.0 and p.prediction is True
