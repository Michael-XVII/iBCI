from __future__ import annotations

import numpy as np
import pytest

from mc_maze.unit_side_features import (
    SIDE_FEATURE_DIMS,
    T4R_PRIOR_VARIANCE_FLOOR,
    _t4r_prior_hash,
    posterior_mean_t4,
)


def _data():
    directions = np.tile(np.arange(8, dtype=np.int64), 3)
    theta = -3.0 * np.pi / 4.0 + directions * (np.pi / 4.0)
    rng = np.random.RandomState(42)
    rates = np.stack((3.0 + 2.0 * np.cos(theta) - 1.0 * np.sin(theta), 4.0 - .5 * np.cos(theta) + 1.5 * np.sin(theta)))
    return rates + rng.normal(scale=.2, size=rates.shape), directions


def test_t4r_weak_prior_matches_trial_level_ols_and_keeps_intercept_unpenalized():
    rates, directions = _data()
    features, _variance, rank, condition = posterior_mean_t4(rates, directions, prior_variance=1.0e14)
    design = np.stack((np.ones(directions.size), np.cos(-3.0 * np.pi / 4.0 + directions * np.pi / 4.0), np.sin(-3.0 * np.pi / 4.0 + directions * np.pi / 4.0)), axis=1)
    ols = np.linalg.lstsq(design, rates.T, rcond=None)[0].T
    assert rank == 3 and np.isfinite(condition)
    assert np.allclose(features[:, :2], ols[:, 1:3], rtol=1e-5, atol=1e-5)
    assert np.allclose(features[:, 2], np.hypot(features[:, 0], features[:, 1]))
    assert np.allclose(features[:, 3], ols[:, 0], rtol=1e-5, atol=1e-5)


def test_t4r_strong_prior_shrinks_direction_and_registry_is_four_dimensional():
    rates, directions = _data()
    weak, _, _, _ = posterior_mean_t4(rates, directions, prior_variance=1.0e8)
    strong, _, _, _ = posterior_mean_t4(rates, directions, prior_variance=1.0e-4)
    assert np.all(np.linalg.norm(strong[:, :2], axis=1) < np.linalg.norm(weak[:, :2], axis=1))
    assert SIDE_FEATURE_DIMS["t4r"] == 4


def test_t4r_rejects_degenerate_or_invalid_prior_and_receipt_hash_is_stable():
    rates, directions = _data()
    with pytest.raises(ValueError, match="positive"):
        posterior_mean_t4(rates, directions, prior_variance=T4R_PRIOR_VARIANCE_FLOOR / 2)
    with pytest.raises(ValueError, match="rank=3"):
        posterior_mean_t4(rates[:, :3], np.zeros(3, dtype=np.int64), prior_variance=1.0)
    receipt = {"formula_version": "v", "prior_variance": 1.0, "pool_size": 50, "source_sessions": ["a"], "source_fingerprints": [{"x": 1}], "source_unit_count": 2}
    assert _t4r_prior_hash(receipt) == _t4r_prior_hash(dict(receipt))
