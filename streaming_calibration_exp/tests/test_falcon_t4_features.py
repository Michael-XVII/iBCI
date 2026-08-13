"""Native FALCON M1/M2 T4 calibration-only tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.falcon_datamodule import FalconDataModule
from src.data.falcon_t4_features import (
    deterministic_row_permutation,
    t4_from_trial_sums,
)


def test_native_t4_recovers_cosine_coefficients() -> None:
    angles = np.asarray([0.0, np.pi / 2, np.pi, -np.pi / 2], dtype=np.float32)
    design = np.stack([np.ones(4), np.cos(angles), np.sin(angles)], axis=1)
    # Two channels with known [baseline, a, c] rate coefficients.
    rates = design @ np.asarray([[5.0, 2.0], [3.0, -1.0], [4.0, 2.0]])
    lengths = np.asarray([10, 12, 11, 9])
    features = t4_from_trial_sums(rates * lengths[:, None], lengths, angles, source="synthetic")
    np.testing.assert_allclose(features[:, 0], [3.0, -1.0], atol=1e-5)
    np.testing.assert_allclose(features[:, 1], [4.0, 2.0], atol=1e-5)
    np.testing.assert_allclose(features[:, 2], [5.0, np.sqrt(5.0)], atol=1e-5)
    np.testing.assert_allclose(features[:, 3], [5.0, 2.0], atol=1e-5)


def test_native_t4_rejects_rank_deficient_labels() -> None:
    with pytest.raises(ValueError, match="rank"):
        t4_from_trial_sums(
            np.ones((4, 3)), np.ones(4), np.zeros(4), source="rank-deficient"
        )


def test_ts4_permutation_is_deterministic_nonidentity() -> None:
    first = deterministic_row_permutation(12, session_name="ses-a", seed=42)
    second = deterministic_row_permutation(12, session_name="ses-a", seed=42)
    np.testing.assert_array_equal(first, second)
    assert sorted(first.tolist()) == list(range(12))
    assert not np.array_equal(first, np.arange(12))


@pytest.mark.parametrize(
    ("task", "relative_dir", "window_size", "calibration_n_trials", "channels"),
    [
        ("m1", "000941", 100, 10, 64),
        ("m2", "000953", 50, 33, 96),
    ],
)
def test_native_t4_uses_heldin_calibration_only(
    task: str,
    relative_dir: str,
    window_size: int,
    calibration_n_trials: int,
    channels: int,
) -> None:
    data_dir = Path(__file__).resolve().parents[2] / "SPINT-main" / "data" / relative_dir
    if not data_dir.is_dir():
        pytest.skip(f"FALCON data unavailable: {data_dir}")
    dm = FalconDataModule(
        task=task,
        data_dir=str(data_dir),
        heldin_session_names=[""],
        batch_size=2,
        window_size=window_size,
        calibration_n_trials=calibration_n_trials,
        random_calibration=False,
        smooth_calibration=False,
        max_trial_length=1024 if task == "m1" else 100,
        use_intertrials=True,
        use_calib_intertrials=False,
        interpolate_trials=True,
        interpolate_trials_kind="cubic",
        validation_protocol="loso",
        loso_fold=0,
        include_heldout_in_fit=False,
        include_heldout_in_test=False,
        num_workers=0,
        pin_memory=False,
        side_feature_group="t4",
        side_feature_shuffle_seed=42,
    )
    dm.setup("fit")
    assert dm.val_heldout_dataset is None
    batch = next(iter(dm.val_dataloader()))
    assert len(batch) == 5
    assert tuple(batch[4].shape) == (2, channels, 4)
    assert np.asarray(dm.native_t4_normalization["mean"]).shape == (4,)


def test_native_t4_allows_heldout_validation_without_training_leakage() -> None:
    data_dir = Path(__file__).resolve().parents[1] / ".." / "dataset" / "ial-mohd" / "000941"
    if not data_dir.is_dir():
        data_dir = Path("/home/ial-mohd/dataset/ial-mohd/000941")
    if not data_dir.is_dir():
        pytest.skip(f"FALCON M1 data unavailable: {data_dir}")
    dm = FalconDataModule(
        task="m1",
        data_dir=str(data_dir),
        heldin_session_names=[""],
        batch_size=2,
        window_size=100,
        calibration_n_trials=10,
        random_calibration=False,
        smooth_calibration=False,
        max_trial_length=1024,
        interpolate_trials=True,
        interpolate_trials_kind="cubic",
        validation_protocol="minival",
        include_heldout_in_fit=True,
        include_heldout_in_test=True,
        num_workers=0,
        pin_memory=False,
        side_feature_group="t4",
    )
    dm.setup("fit")
    assert set(dm.train_session_names) == set(dm.train_calib_heldin_sessions)
    assert set(dm.val_heldout_dataset.neural_data).isdisjoint(dm.train_session_names)
    assert dm.get_split_manifest()["heldout_evaluated_in_fit"] is True
    loaders = dm.val_dataloader()
    assert isinstance(loaders, list) and len(loaders) == 2
