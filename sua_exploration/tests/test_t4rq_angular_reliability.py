from __future__ import annotations

import math

import numpy as np

from mc_maze.unit_side_features import (
    SIDE_FEATURE_DIMS,
    T4RQ_ZERO_MODULATION_RELIABILITY,
    posterior_angular_reliability,
)


def _rotation(angle: float) -> np.ndarray:
    return np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])


def test_angular_reliability_is_so2_invariant():
    features = np.asarray([[2.0, 1.0, math.sqrt(5.0), 3.0]], dtype=np.float32)
    covariance = np.asarray([[[.5, .1], [.1, .25]]], dtype=np.float32)
    angle = .73
    rotation = _rotation(angle)
    rotated_mu = features[:, :2] @ rotation.T
    rotated_features = np.column_stack((rotated_mu, np.linalg.norm(rotated_mu, axis=1), features[:, 3]))
    rotated_covariance = np.asarray([rotation @ covariance[0] @ rotation.T], dtype=np.float32)
    assert np.allclose(
        posterior_angular_reliability(features, covariance),
        posterior_angular_reliability(rotated_features, rotated_covariance),
        atol=1e-6,
    )


def test_zero_modulation_is_fail_closed_and_group_is_five_dimensional():
    features = np.asarray([[0.0, 0.0, 0.0, 4.0]], dtype=np.float32)
    covariance = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    reliability = posterior_angular_reliability(features, covariance)
    assert reliability.shape == (1,)
    assert reliability[0] == T4RQ_ZERO_MODULATION_RELIABILITY
    assert SIDE_FEATURE_DIMS["t4rq"] == 5
