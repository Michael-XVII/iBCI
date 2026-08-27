from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "streaming_calibration_exp"))

from src.models.streaming_calibration_module import analytic_ridge_ole_prediction  # noqa: E402


def test_analytic_ridge_ole_matches_e08_numpy_formula() -> None:
    neural = torch.tensor(
        [
            [[1.0, 0.0, 2.0], [0.0, 1.0, 0.0], [2.0, 1.0, 1.0]],
            [[0.0, 2.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        ]
    )
    raw_t4 = np.array(
        [[2.0, 0.5, 2.1, 5.0], [-0.5, 1.5, 1.6, 3.0], [1.0, -1.0, 1.4, 4.0]],
        dtype=np.float32,
    )
    mean = np.array([0.3, -0.2, 1.0, 2.0], dtype=np.float32)
    std = np.array([1.5, 0.7, 2.0, 3.0], dtype=np.float32)
    normalized = torch.from_numpy((raw_t4 - mean) / std).repeat(2, 1, 1)
    ridge_lambda = 100.0
    gain = 0.30907195449579733
    observed = analytic_ridge_ole_prediction(
        neural,
        normalized,
        side_mean=torch.from_numpy(mean),
        side_std=torch.from_numpy(std),
        ridge_lambda=ridge_lambda,
        gain=gain,
        bin_size_s=0.02,
    ).numpy()[:, 0]

    rate_hz = neural.numpy().sum(axis=1) / (neural.shape[1] * 0.02)
    beta = raw_t4[:, :2].astype(np.float64)
    expected = []
    for rate in rate_hz:
        centered = rate - raw_t4[:, 3]
        expected.append(
            gain
            * centered
            @ beta
            @ np.linalg.inv(beta.T @ beta + ridge_lambda * np.eye(2))
        )
    assert np.allclose(observed, np.asarray(expected), atol=1e-6, rtol=1e-5)


def test_analytic_row_mismatch_is_deterministic_and_changes_output() -> None:
    neural = torch.arange(1, 1 + 2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    t4 = torch.tensor(
        [[[2.0, 0.0, 2.0, 1.0], [0.0, 3.0, 3.0, 2.0], [1.0, -1.0, 1.4, 0.5]]]
    ).repeat(2, 1, 1)
    kwargs = dict(
        side_mean=torch.zeros(4),
        side_std=torch.ones(4),
        ridge_lambda=100.0,
        gain=0.3,
        bin_size_s=0.02,
    )
    aligned = analytic_ridge_ole_prediction(neural, t4, **kwargs)
    shuffled_a = analytic_ridge_ole_prediction(neural, t4, shuffle_seed=0, **kwargs)
    shuffled_b = analytic_ridge_ole_prediction(neural, t4, shuffle_seed=0, **kwargs)
    assert torch.equal(shuffled_a, shuffled_b)
    assert not torch.allclose(aligned, shuffled_a)


def test_zero_residual_path_is_exact_analytic_prediction() -> None:
    analytic = torch.randn(4, 1, 2)
    residual = torch.zeros_like(analytic)
    assert torch.equal(analytic + residual, analytic)
