"""Pure protocol primitives for the H1 M3 secondary transfer diagnostic.

This review-stage module has no NWB, checkpoint, CUDA, training, or evaluation
entry point.  It fixes the M3 support, causal boundary, scoring, and aggregation
semantics for CPU tests before target evaluation is authorized.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from src.data.h1_m4_cce_date_lodo import target_sessions_for_date
from src.data.h1_m4_eb_pilot import fit_deployment_carrier, interpolate_trial_identity
from src.h1_hc_date_lodo_regen_v1 import variance_weighted_r2
from src.h1_m4_cce_contract import CONFIRMATORY_DATES, NORMALIZER_FLOOR


SCHEMA = "h1_cal_aug_prefix_cycle_m3_transfer_v1"
ARMS = ("t0", "c1")
M3_BUDGET = 3
WINDOW_SIZE = 700
OUTPUT_SCALE = 20.0
M4_AUDIT_COMMIT = "0d0ab2f"
M4_AUDIT_TERMINAL_SHA256 = "3ff971dc576958b13ace990bcca8aea2e8b999e2af2ed50f418296d05f8d5cfc"
M4_METADATA_TERMINAL_SHA256 = "e692db3b744a7831610c2338ce34504ccedc4c8696e4d73333de899ed295b563"
EXPERIMENT4_A1_COMMIT = "c60052c9d8ccb8391d6ce53bde9ccfb4f2319884"
EXPERIMENT4_A1_TERMINAL_SHA256 = "dc9e7ab44954d3d193f67f9bf8936aafdaf2b05be9968d5e0091c0b0ecf092fd"


class M3TransferProtocolError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise M3TransferProtocolError(message)


def dry_plan() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "DRY_REVIEW_ONLY_NO_WRITE_NO_DATA_NO_CUDA",
        "estimand": "secondary_m3_transfer_diagnostic",
        "outer_dates": list(CONFIRMATORY_DATES),
        "arms": list(ARMS),
        "calibration_trials": M3_BUDGET,
        "first_scoring_trial_ordinal": 4,
        "window_bins": WINDOW_SIZE,
        "last_bin_only": True,
        "prediction_divisor": OUTPUT_SCALE,
        "normalizer_floor": NORMALIZER_FLOOR,
        "score_dtype": "float64",
        "outputs": 7,
        "checkpoint_epoch_zero_based": 49,
        "retraining": False,
        "optimizer_steps": 0,
        "backward_steps": 0,
        "model_updates": 0,
        "heldout_calib_files_opened": 0,
        "target_files_opened": 0,
        "cuda_initialized": False,
        "execution_entrypoints_enabled": False,
        "m4_governing_result_unchanged": True,
    }


def ordered_legal_trials(trial_num: np.ndarray, eval_mask: np.ndarray) -> tuple[float, ...]:
    labels = np.asarray(trial_num, dtype=np.float64).reshape(-1)
    mask = np.asarray(eval_mask, dtype=bool).reshape(-1)
    _need(labels.shape == mask.shape, "TrialNum/eval mask length mismatch")
    ordered = labels[mask & np.isfinite(labels)]
    _need(ordered.size > 0 and not np.any(np.diff(ordered) < 0.0),
          "TrialNum must be nonempty and chronological on eval-valid bins")
    values: list[float] = []
    for value in ordered.tolist():
        if not values or float(value) != values[-1]:
            values.append(float(value))
    return tuple(values)


def m3_support_and_query_trial(trial_values: Sequence[float]) -> tuple[tuple[float, ...], float]:
    values = tuple(float(value) for value in trial_values)
    _need(len(values) >= 4, "M3 post-calibration evaluation requires a fourth legal trial")
    _need(len(set(values)) == len(values), "legal TrialNum values must be unique")
    return values[:M3_BUDGET], values[M3_BUDGET]


def m3_causal_surface(
    trial_num: np.ndarray,
    eval_mask: np.ndarray,
    recording_bins: int,
    *,
    window: int = WINDOW_SIZE,
) -> dict[str, Any]:
    labels = np.asarray(trial_num, dtype=np.float64).reshape(-1)
    mask = np.asarray(eval_mask, dtype=bool).reshape(-1)
    _need(labels.shape == mask.shape == (int(recording_bins),), "recording array length mismatch")
    _need(int(window) == WINDOW_SIZE, "M3 protocol fixes W=700")
    support, query_trial = m3_support_and_query_trial(ordered_legal_trials(labels, mask))
    boundary_rows = np.flatnonzero(mask & np.isfinite(labels) & (labels == query_trial))
    _need(boundary_rows.size > 0, "fourth legal trial has no eval-valid causal boundary")
    boundary = int(boundary_rows[0])
    last_start = int(recording_bins) - WINDOW_SIZE
    _need(last_start >= boundary, "no complete W700 window beginning in trial 4+")
    starts = np.arange(boundary, last_start + 1, dtype=np.int64)
    output_bins = starts + WINDOW_SIZE - 1
    score_mask = np.asarray(mask[output_bins], dtype=bool)
    _need(int(score_mask.sum()) > 1, "insufficient eval-valid output bins for R2")
    return {
        "support": list(support),
        "query_trial": query_trial,
        "boundary_bin": boundary,
        "starts": starts,
        "output_bins": output_bins,
        "score_mask": score_mask,
    }


def materialize_m3_calibration(record: Any, plan: Any, normalizer: float) -> dict[str, Any]:
    _need(np.isfinite(normalizer) and float(normalizer) >= 0.0, "normalizer must be nonnegative and finite")
    support, query_trial = m3_support_and_query_trial(record.trial_values)
    identity = np.ascontiguousarray(
        np.stack([interpolate_trial_identity(record, value) for value in support]), dtype=np.float32
    )
    fitted = fit_deployment_carrier(record, plan, support)
    denominator = max(float(normalizer), NORMALIZER_FLOOR)
    carrier = np.ascontiguousarray(np.asarray(fitted["carrier"], np.float64) / denominator, dtype=np.float32)
    _need(
        identity.ndim == 3
        and carrier.ndim == 2
        and identity.shape[0] == M3_BUDGET
        and carrier.shape[1] == 4
        and identity.shape[2] == carrier.shape[0],
        "M3 identity/carrier shape alignment drift",
    )
    _need(np.isfinite(identity).all() and np.isfinite(carrier).all(), "nonfinite M3 calibration")
    return {"support": list(support), "query_trial": query_trial, "identity": identity, "carrier": carrier}


def score_m3(target: np.ndarray, prediction: np.ndarray, score_mask: np.ndarray) -> float:
    truth = np.asarray(target, dtype=np.float64)
    estimate = np.asarray(prediction, dtype=np.float64)
    mask = np.asarray(score_mask, dtype=bool).reshape(-1)
    _need(truth.ndim == estimate.ndim == 2 and truth.shape == estimate.shape, "target/prediction shape drift")
    _need(truth.shape[1] == 7 and truth.shape[0] == mask.shape[0], "M3 score requires [N,7] and aligned mask")
    _need(int(mask.sum()) > 1, "insufficient M3 score rows")
    _need(np.isfinite(truth[mask]).all() and np.isfinite(estimate[mask]).all(), "nonfinite M3 score arrays")
    return float(variance_weighted_r2(truth[mask], estimate[mask]))


def scale_last_bin_prediction(model_output: np.ndarray) -> np.ndarray:
    output = np.asarray(model_output)
    _need(output.ndim == 3 and output.shape[2] == 7, "model output must be [batch,time,7]")
    values = np.asarray(output[:, -1, :], dtype=np.float32) / np.float32(OUTPUT_SCALE)
    _need(np.isfinite(values).all(), "nonfinite scaled M3 prediction")
    return np.ascontiguousarray(values, dtype=np.float32)


def aggregate_m3_results(per_recording: Mapping[str, Mapping[str, Mapping[str, float]]]) -> dict[str, Any]:
    _need(tuple(per_recording) == CONFIRMATORY_DATES, "M3 date roster/order drift")
    dates: dict[str, Any] = {}
    deltas: list[float] = []
    for date in CONFIRMATORY_DATES:
        rows = per_recording[date]
        expected = target_sessions_for_date(date)
        _need(tuple(rows) == expected, f"M3 recording roster/order drift: {date}")
        _need(all(set(rows[session]) == set(ARMS) for session in expected), f"M3 arm roster drift: {date}")
        arm_scores = {arm: [float(rows[session][arm]) for session in expected] for arm in ARMS}
        _need(all(np.isfinite(value) for values in arm_scores.values() for value in values), f"nonfinite M3 R2: {date}")
        means = {arm: float(np.mean(values, dtype=np.float64)) for arm, values in arm_scores.items()}
        delta = means["c1"] - means["t0"]
        deltas.append(delta)
        dates[date] = {
            "equal_recording_mean_r2": means,
            "delta_c1_minus_t0": delta,
            "per_recording_r2": {
                session: {
                    "t0": float(rows[session]["t0"]),
                    "c1": float(rows[session]["c1"]),
                    "delta_c1_minus_t0": float(rows[session]["c1"] - rows[session]["t0"]),
                }
                for session in expected
            },
        }
    return {
        "status": "COMPLETE_SECONDARY_M3_TRANSFER_DIAGNOSTIC",
        "date_order": list(CONFIRMATORY_DATES),
        "dates": dates,
        "equal_date_mean_delta_r2": float(np.mean(deltas, dtype=np.float64)),
        "positive_date_count": int(sum(value > 0.0 for value in deltas)),
        "governing_m4_result_modified": False,
        "selection_performed": False,
    }


__all__ = (
    "ARMS", "M3_BUDGET", "M3TransferProtocolError", "OUTPUT_SCALE", "SCHEMA", "WINDOW_SIZE",
    "aggregate_m3_results", "dry_plan", "m3_causal_surface", "m3_support_and_query_trial",
    "materialize_m3_calibration", "ordered_legal_trials", "scale_last_bin_prediction", "score_m3",
)
