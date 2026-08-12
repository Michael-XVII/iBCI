"""Tests for M1 PCT4 calibration-only side features."""
from __future__ import annotations

import numpy as np
import pytest

from src.data.falcon_m1_pct4_features import (
    PCT4_DIM,
    deterministic_label_permutation,
    deterministic_pct4_row_permutation,
    fit_train_pct4_stats,
    pct4_from_phase_sums,
    phase_window_trial_sums,
)
from src.data.falcon_datamodule import FalconDataset


def test_pct4_recovers_known_signed_slopes() -> None:
    angles = np.asarray([0.0, np.pi / 2, np.pi, -np.pi / 2], dtype=np.float64)
    design = np.stack([np.ones(4), np.cos(angles), np.sin(angles)], axis=1)
    reach_rates = design @ np.asarray([[5.0, 1.0], [3.0, -2.0], [4.0, 0.5]])
    post_rates = design @ np.asarray([[7.0, 2.0], [-1.0, 3.0], [2.5, -4.0]])
    reach_lengths = np.asarray([10, 11, 12, 13])
    post_lengths = np.asarray([20, 21, 22, 23])
    features = pct4_from_phase_sums(
        reach_rates * reach_lengths[:, None],
        reach_lengths,
        post_rates * post_lengths[:, None],
        post_lengths,
        angles,
        source="synthetic",
    )
    assert features.shape == (2, PCT4_DIM)
    np.testing.assert_allclose(features[:, 0], [3.0, -2.0], atol=1e-5)
    np.testing.assert_allclose(features[:, 1], [4.0, 0.5], atol=1e-5)
    np.testing.assert_allclose(features[:, 2], [-1.0, 3.0], atol=1e-5)
    np.testing.assert_allclose(features[:, 3], [2.5, -4.0], atol=1e-5)


def test_pct4_rejects_rank_deficient_direction_design() -> None:
    with pytest.raises(ValueError, match="rank"):
        pct4_from_phase_sums(
            np.ones((4, 3)),
            np.ones(4),
            np.ones((4, 3)),
            np.ones(4),
            np.zeros(4),
            source="rank-deficient",
        )


def test_phase_window_bin_end_semantics() -> None:
    timestamps = np.asarray([0.02, 0.04, 0.06, 0.08], dtype=np.float64)
    neural = np.arange(8, dtype=np.float64).reshape(4, 2)
    sums, lengths = phase_window_trial_sums(
        neural,
        np.ones(4, dtype=bool),
        timestamps,
        np.asarray([0.02]),
        np.asarray([0.06]),
        source="bin-end",
    )
    np.testing.assert_allclose(sums, neural[0:2].sum(axis=0, keepdims=True))
    np.testing.assert_array_equal(lengths, [2])


def test_phase_window_uses_eval_mask() -> None:
    timestamps = np.asarray([0.02, 0.04, 0.06, 0.08], dtype=np.float64)
    neural = np.arange(8, dtype=np.float64).reshape(4, 2)
    sums, lengths = phase_window_trial_sums(
        neural,
        np.asarray([True, False, True, False]),
        timestamps,
        np.asarray([0.02]),
        np.asarray([0.08]),
        source="eval-mask",
    )
    np.testing.assert_allclose(sums, neural[[0, 2]].sum(axis=0, keepdims=True))
    np.testing.assert_array_equal(lengths, [2])


def test_pct4_is_linear_under_channel_pooling() -> None:
    angles = np.asarray([0.0, np.pi / 2, np.pi, -np.pi / 2], dtype=np.float64)
    design = np.stack([np.ones(4), np.cos(angles), np.sin(angles)], axis=1)
    reach_rates = design @ np.asarray([[1.0, 3.0], [2.0, -1.0], [0.5, 4.0]])
    post_rates = design @ np.asarray([[2.0, -2.0], [1.5, 0.25], [-3.0, 0.75]])
    lengths = np.asarray([8, 9, 10, 11])
    individual = pct4_from_phase_sums(
        reach_rates * lengths[:, None], lengths, post_rates * lengths[:, None], lengths, angles, source="ind"
    )
    pooled = pct4_from_phase_sums(
        (reach_rates.sum(axis=1, keepdims=True) * lengths[:, None]),
        lengths,
        (post_rates.sum(axis=1, keepdims=True) * lengths[:, None]),
        lengths,
        angles,
        source="pooled",
    )
    np.testing.assert_allclose(pooled[0], individual.sum(axis=0), atol=1e-12)


def test_label_shuffle_is_deterministic_and_nonidentity() -> None:
    first = deterministic_label_permutation(10, session_name="ses-a", seed=42)
    second = deterministic_label_permutation(10, session_name="ses-a", seed=42)
    np.testing.assert_array_equal(first, second)
    assert sorted(first.tolist()) == list(range(10))
    assert not np.array_equal(first, np.arange(10))
    row = deterministic_pct4_row_permutation(12, session_name="ses-a", seed=42)
    assert sorted(row.tolist()) == list(range(12))
    assert not np.array_equal(row, np.arange(12))


def test_pct4_normalization_source_only() -> None:
    angles = np.asarray([0.0, np.pi / 2, np.pi, -np.pi / 2], dtype=np.float64)
    lengths = np.ones(4)
    train = {
        "A": np.ones((4, 2)),
        "B": np.full((4, 2), 2.0),
        "C": np.full((4, 2), 1000.0),
    }
    post = {name: value * 3.0 for name, value in train.items()}
    lens = {name: lengths for name in train}
    angle_map = {name: angles for name in train}
    mean1, std1 = fit_train_pct4_stats(train, lens, post, lens, angle_map, ["A", "B"], 4)
    train["C"] *= -100.0
    post["C"] *= -100.0
    mean2, std2 = fit_train_pct4_stats(train, lens, post, lens, angle_map, ["A", "B"], 4)
    np.testing.assert_array_equal(mean1, mean2)
    np.testing.assert_array_equal(std1, std2)


def test_pct4_z4_exact_zero() -> None:
    ds = object.__new__(FalconDataset)
    ds.side_feature_group = "pct4_z4"
    ds.calib_pct4_reach_sums = {"s": np.ones((4, 3), dtype=np.float32)}
    ds._side_feature_cache = {}
    values = FalconDataset._native_pct4_side_features(ds, "s", 0, 4)
    np.testing.assert_array_equal(values, np.zeros((3, 4), dtype=np.float32))


def test_pct4_rejects_non_m1() -> None:
    with pytest.raises(ValueError, match="M1 only"):
        FalconDataset({}, {}, side_feature_group="pct4", smooth_calibration=False, task="m2")
