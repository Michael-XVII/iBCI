from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from eval_e08_analytic_only_dandi688 import (  # noqa: E402
    add_stats,
    analytic_predictions,
    fit_isotropic_gain,
    gain_from_stats,
    prediction_stats,
    r2_from_stats,
    select_lambda_source_loso,
    subtract_stats,
)


def test_analytic_formulas_match_direct_linear_algebra() -> None:
    rates = np.array([[3.0, 6.0, 2.0], [5.0, 1.0, 4.0]])
    t4 = np.array([[2.0, 0.0, 2.0, 1.0], [0.0, 3.0, 3.0, 2.0], [1.0, -1.0, np.sqrt(2.0), 0.5]])
    beta = t4[:, :2]
    centered = rates - t4[:, 3]
    assert np.allclose(analytic_predictions(rates, t4, "population_vector_like"), centered @ beta)
    ridge = 0.7
    expected = centered @ beta @ np.linalg.inv(beta.T @ beta + ridge * np.eye(2))
    assert np.allclose(analytic_predictions(rates, t4, "ridge_ole", ridge), expected)


def test_stats_gain_and_r2_match_materialized_values() -> None:
    z = np.array([[1.0, 2.0], [-2.0, 1.0], [0.5, -1.0], [3.0, 0.2]])
    y = 1.7 * z + np.array([[0.1, -0.2], [0.0, 0.1], [-0.1, 0.05], [0.2, 0.0]])
    stats_a = prediction_stats(z[:2], y[:2])
    stats_b = prediction_stats(z[2:], y[2:])
    total = add_stats([stats_a, stats_b])
    gain = fit_isotropic_gain(z, y)
    assert np.isclose(gain_from_stats(total), gain)
    residual = y - gain * z
    centered = y - y.mean(axis=0)
    expected_r2 = 1.0 - np.sum(residual**2) / np.sum(centered**2)
    assert np.isclose(r2_from_stats(total, gain), expected_r2)
    recovered_b = subtract_stats(total, stats_a)
    for key in ("n", "sum_z2", "sum_zy"):
        assert np.allclose(recovered_b[key], stats_b[key])


def test_source_loso_selects_exact_decoder() -> None:
    stats_by_lambda = {0.0: {}, 10.0: {}}
    for index in range(3):
        y = np.array([[1.0 + index, -0.5], [2.0 + index, 1.0]])
        stats_by_lambda[0.0][f"s{index}"] = prediction_stats(y.copy(), y)
        stats_by_lambda[10.0][f"s{index}"] = prediction_stats(np.zeros_like(y), y)
    selected, audit = select_lambda_source_loso(stats_by_lambda)
    assert selected == 0.0
    assert audit["0.0"]["mean_loso_r2"] > audit["10.0"]["mean_loso_r2"]
