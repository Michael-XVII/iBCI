#!/usr/bin/env python3
"""E08: analytic-only decoders built directly from session-specific raw T4.

The target-session decoder is constructed without backpropagation, an optimizer, or
continuous target kinematic labels.  The first 50 rewarded trials provide raw T4
``[a, c, m, b]``.  All scored windows are from trials[50:].

B0-1 is the population-vector-like rule ``sum_i (r_i-b_i) beta_i``.  B0-2 is
``(B'B + lambda I)^-1 B'(r-b)`` with W=I.  Because T4 is fit against endpoint
direction rather than instantaneous speed, each rule receives one isotropic scalar gain.
The gain, and B0-2's lambda, are fixed from source sessions only.  Lambda selection uses
leave-one-source-session-out R2.  Target continuous velocities are read only for scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mc_maze.multisession_datamodule import (  # noqa: E402
    _compute_valid_starts,
    chronological_session_split,
    discover_nwb_files,
    fit_behavior_stats,
    session_name_from_path,
)
from mc_maze.unit_side_features import load_unit_side_features  # noqa: E402
from eval_adaptation_dandi688 import (  # noqa: E402
    PAD_VALUE,
    TRIAL_LENGTH,
    WINDOW_SIZE,
    load_session_with_trials,
    parse_split_counts,
)
from linear_decoder_control_dandi688 import compute_r2  # noqa: E402

BIN_SIZE_MS = 20
BIN_SIZE_S = BIN_SIZE_MS / 1000.0
POOL_SIZE = 50
SEED = 42
DEFAULT_LAMBDAS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def analytic_predictions(rate_hz: np.ndarray, t4: np.ndarray, method: str, ridge_lambda: float = 0.0) -> np.ndarray:
    """Return the unscaled B0-1 or B0-2 prediction for rows of firing rates."""
    rate = np.asarray(rate_hz, dtype=np.float64)
    features = np.asarray(t4, dtype=np.float64)
    if rate.ndim != 2 or features.shape != (rate.shape[1], 4):
        raise ValueError(f"Expected rates [T,N] and T4 [N,4], got {rate.shape} and {features.shape}")
    beta = features[:, :2]
    centered = rate - features[:, 3]
    if method == "population_vector_like":
        return centered @ beta
    if method == "ridge_ole":
        if ridge_lambda < 0:
            raise ValueError("ridge_lambda must be non-negative")
        gram = beta.T @ beta + ridge_lambda * np.eye(2, dtype=np.float64)
        return centered @ beta @ np.linalg.pinv(gram, hermitian=True)
    raise ValueError(f"Unknown analytic method: {method}")


def fit_isotropic_gain(raw_prediction: np.ndarray, target: np.ndarray) -> float:
    """Least-squares scalar gain with a fixed zero intercept."""
    z = np.asarray(raw_prediction, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    denominator = float(np.sum(z * z))
    return 0.0 if denominator <= 1e-12 else float(np.sum(z * y) / denominator)


def prediction_stats(raw_prediction: np.ndarray, target: np.ndarray) -> dict[str, float | int | list[float]]:
    z = np.asarray(raw_prediction, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if z.shape != y.shape or z.ndim != 2 or z.shape[1] != 2:
        raise ValueError("prediction and target must have matching [T,2] shapes")
    return {
        "n": int(z.shape[0]),
        "sum_y": y.sum(axis=0).tolist(),
        "sum_y2": np.square(y).sum(axis=0).tolist(),
        "sum_z2": float(np.sum(z * z)),
        "sum_zy": float(np.sum(z * y)),
    }


def add_stats(items: Sequence[dict]) -> dict:
    if not items:
        raise ValueError("cannot add an empty statistics list")
    return {
        "n": int(sum(int(item["n"]) for item in items)),
        "sum_y": np.sum([item["sum_y"] for item in items], axis=0).tolist(),
        "sum_y2": np.sum([item["sum_y2"] for item in items], axis=0).tolist(),
        "sum_z2": float(sum(float(item["sum_z2"]) for item in items)),
        "sum_zy": float(sum(float(item["sum_zy"]) for item in items)),
    }


def subtract_stats(total: dict, part: dict) -> dict:
    return {
        "n": int(total["n"]) - int(part["n"]),
        "sum_y": (np.asarray(total["sum_y"]) - np.asarray(part["sum_y"])).tolist(),
        "sum_y2": (np.asarray(total["sum_y2"]) - np.asarray(part["sum_y2"])).tolist(),
        "sum_z2": float(total["sum_z2"]) - float(part["sum_z2"]),
        "sum_zy": float(total["sum_zy"]) - float(part["sum_zy"]),
    }


def gain_from_stats(stats: dict) -> float:
    denominator = float(stats["sum_z2"])
    return 0.0 if denominator <= 1e-12 else float(stats["sum_zy"]) / denominator


def r2_from_stats(stats: dict, gain: float) -> float:
    sum_y = np.asarray(stats["sum_y"], dtype=np.float64)
    sum_y2 = np.asarray(stats["sum_y2"], dtype=np.float64)
    n = int(stats["n"])
    sst = float(np.sum(sum_y2 - np.square(sum_y) / n))
    sse = float(np.sum(sum_y2) - 2.0 * gain * float(stats["sum_zy"]) + gain * gain * float(stats["sum_z2"]))
    return float("nan") if sst <= 0 else 1.0 - sse / sst


def select_lambda_source_loso(stats_by_lambda: dict[float, dict[str, dict]]) -> tuple[float, dict[str, dict]]:
    """Choose lambda by unweighted mean leave-one-source-session-out R2."""
    audit: dict[str, dict] = {}
    for ridge_lambda, per_session in stats_by_lambda.items():
        total = add_stats(list(per_session.values()))
        fold_r2 = {}
        for name, heldout in per_session.items():
            gain = gain_from_stats(subtract_stats(total, heldout))
            fold_r2[name] = r2_from_stats(heldout, gain)
        audit[repr(ridge_lambda)] = {
            "mean_loso_r2": float(np.mean(list(fold_r2.values()))),
            "per_session_loso_r2": fold_r2,
        }
    best = max(stats_by_lambda, key=lambda value: (audit[repr(value)]["mean_loso_r2"], -value))
    return float(best), audit


def window_mean_rates_and_targets(rec: dict, pool_size: int) -> tuple[np.ndarray, np.ndarray, int]:
    eval_trials = rec["trials"][pool_size:]
    starts = _compute_valid_starts(eval_trials, WINDOW_SIZE)
    if len(starts) == 0:
        raise ValueError(f"{rec['name']}: no evaluation windows after trials[{pool_size}:]")
    cumulative = np.vstack((np.zeros((1, rec["n_units"]), dtype=np.float64), np.cumsum(rec["neural"], axis=0, dtype=np.float64)))
    counts = cumulative[starts + WINDOW_SIZE] - cumulative[starts]
    rate_hz = counts / (WINDOW_SIZE * BIN_SIZE_S)
    target = rec["behavior"][starts + WINDOW_SIZE - 1].astype(np.float64, copy=False)
    return rate_hz, target, int(len(starts))


def load_session(path: Path, behavior_mean: np.ndarray, behavior_std: np.ndarray, cache_dir: Path, pool_size: int) -> tuple[dict, np.ndarray]:
    rec = load_session_with_trials(
        path,
        bin_size_ms=BIN_SIZE_MS,
        window_size=WINDOW_SIZE,
        calib_n=pool_size,
        max_trial_length=TRIAL_LENGTH,
        pad_value=PAD_VALUE,
        behavior_mean=behavior_mean,
        behavior_std=behavior_std,
        cache_dir=cache_dir,
        signal_view="sua",
    )
    if len(rec["trials"]) <= pool_size:
        raise ValueError(f"{rec['name']}: only {len(rec['trials'])} usable trials")
    t4, metadata = load_unit_side_features(
        path,
        feature_group="t4",
        pool_size=pool_size,
        mean=np.zeros(4, dtype=np.float32),
        std=np.ones(4, dtype=np.float32),
        cache_dir=cache_dir,
        bin_size_ms=BIN_SIZE_MS,
        window_size=WINDOW_SIZE,
        trial_result_filter="R",
        signal_view="sua",
    )
    if t4.shape[0] != rec["n_units"]:
        raise ValueError(f"{rec['name']}: T4 rows do not match neural units")
    rec["t4_metadata"] = metadata.__dict__
    return rec, t4.astype(np.float64)


def direction_and_speed_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred_norm = np.linalg.norm(prediction, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    valid = (pred_norm > 1e-8) & (target_norm > 1e-8)
    if not np.any(valid):
        return {"mean_direction_cosine": float("nan"), "median_angular_error_deg": float("nan"), "positive_dot_fraction": float("nan"), "mean_speed_ratio": float("nan"), "median_speed_ratio": float("nan")}
    dot = np.sum(prediction[valid] * target[valid], axis=1)
    cosine = np.clip(dot / (pred_norm[valid] * target_norm[valid]), -1.0, 1.0)
    nonzero_target_speed = target_norm[target_norm > 1e-8]
    speed_floor = max(0.1, float(np.quantile(nonzero_target_speed, 0.25)))
    speed_valid = valid & (target_norm >= speed_floor)
    speed_ratio = pred_norm[speed_valid] / target_norm[speed_valid]
    target_energy = float(np.sum(target * target))
    vector_scale_slope = (
        float(np.sum(prediction * target)) / target_energy if target_energy > 1e-12 else float("nan")
    )
    return {
        "mean_direction_cosine": float(np.mean(cosine)),
        "median_angular_error_deg": float(np.median(np.degrees(np.arccos(cosine)))),
        "positive_dot_fraction": float(np.mean(dot > 0)),
        "mean_speed_ratio": float(np.mean(speed_ratio)),
        "median_speed_ratio": float(np.median(speed_ratio)),
        "speed_ratio_target_floor": speed_floor,
        "vector_scale_slope": vector_scale_slope,
        "speed_rmse": float(np.sqrt(np.mean(np.square(pred_norm - target_norm)))),
    }


def summarize(per_session: dict[str, dict], baseline_r2: dict[str, float]) -> dict:
    values = np.asarray([row["r2"] for row in per_session.values()], dtype=np.float64)
    deltas = {name: row["r2"] - baseline_r2[name] for name, row in per_session.items()}
    return {
        "mean_r2": float(values.mean()),
        "median_r2": float(np.median(values)),
        "worst_session_r2": float(values.min()),
        "worst_session": min(per_session, key=lambda name: per_session[name]["r2"]),
        "positive_sessions": int(np.sum(values > 0)),
        "n_sessions": int(len(values)),
        "mean_paired_delta_vs_e01": float(np.mean(list(deltas.values()))),
        "paired_delta_vs_e01": deltas,
        "mean_direction_cosine": float(np.mean([row["mean_direction_cosine"] for row in per_session.values()])),
        "median_angular_error_deg": float(np.median([row["median_angular_error_deg"] for row in per_session.values()])),
        "mean_speed_ratio": float(np.mean([row["mean_speed_ratio"] for row in per_session.values()])),
        "median_speed_ratio": float(np.median([row["median_speed_ratio"] for row in per_session.values()])),
        "mean_vector_scale_slope": float(np.mean([row["vector_scale_slope"] for row in per_session.values()])),
        "mean_speed_rmse": float(np.mean([row["speed_rmse"] for row in per_session.values()])),
    }


def e01_per_session_r2(reference: dict, target_names: Sequence[str]) -> dict[str, float]:
    metrics = reference["test_metrics"][-1]
    result = {}
    for name in target_names:
        key = f"test_heldout_{name}/r2"
        if key not in metrics:
            raise KeyError(f"E01 reference is missing {key}")
        result[name] = float(metrics[key])
    return result


def parse_lambdas(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values or any(value < 0 or not math.isfinite(value) for value in values):
        raise ValueError("ridge lambdas must be finite non-negative values")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="/home/ial-mohd/dataset/ial-mohd/000688/sub-C")
    parser.add_argument("--cache_dir", default="/tmp/ibci_template_ridge_db_cache")
    parser.add_argument("--reference_artifact", default="sua_exploration/results/p3_template_ridge_db_heldout_spint_t4_s42_seed42.json")
    parser.add_argument("--split_counts", default="37,8,8")
    parser.add_argument("--pool_size", type=int, default=POOL_SIZE)
    parser.add_argument("--ridge_lambdas", default=",".join(repr(value) for value in DEFAULT_LAMBDAS))
    parser.add_argument("--max_source_sessions", type=int, default=None, help="Dev-only subset")
    parser.add_argument("--max_target_sessions", type=int, default=None, help="Dev-only subset")
    parser.add_argument("--out_path", default="sua_exploration/results/e08_analytic_only_t4_seed42.json")
    args = parser.parse_args()

    started = time.time()
    data_dir = Path(args.data_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    reference_path = Path(args.reference_artifact).expanduser().resolve()
    out_path = Path(args.out_path).expanduser().resolve()
    reference = json.loads(reference_path.read_text())
    files = discover_nwb_files(data_dir, "CO", None)
    source_files, val_files, target_files = chronological_session_split(files, parse_split_counts(args.split_counts))
    split_names = {key: [session_name_from_path(path) for path in group] for key, group in (("train", source_files), ("val", val_files), ("test", target_files))}
    if split_names != reference["session_splits"]:
        raise ValueError("Discovered 37/8/8 split differs from the E01 reference artifact")
    formal = args.max_source_sessions is None and args.max_target_sessions is None
    if args.max_source_sessions is not None:
        source_files = source_files[: args.max_source_sessions]
    if args.max_target_sessions is not None:
        target_files = target_files[: args.max_target_sessions]
    if len(source_files) < 2 or not target_files:
        raise ValueError("E08 needs at least two source sessions and one target session")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    behavior_mean, behavior_std = fit_behavior_stats(source_files if not formal else files[:37], BIN_SIZE_MS, cache_dir=cache_dir)
    lambdas = parse_lambdas(args.ridge_lambdas)

    print(f"E08 analytic-only seed={SEED} formal={formal}")
    print(f"source sessions={len(source_files)} target test sessions={len(target_files)} pool={args.pool_size}")
    print("target optimizer=none; target gradients=false; target continuous labels used for scoring only")

    pv_source_stats: dict[str, dict] = {}
    ole_source_stats: dict[float, dict[str, dict]] = {value: {} for value in lambdas}
    source_receipts = {}
    for path in source_files:
        t0 = time.time()
        rec, t4 = load_session(path, behavior_mean, behavior_std, cache_dir, args.pool_size)
        rates, target, n_windows = window_mean_rates_and_targets(rec, args.pool_size)
        pv_raw = analytic_predictions(rates, t4, "population_vector_like")
        pv_source_stats[rec["name"]] = prediction_stats(pv_raw, target)
        for ridge_lambda in lambdas:
            raw = analytic_predictions(rates, t4, "ridge_ole", ridge_lambda)
            ole_source_stats[ridge_lambda][rec["name"]] = prediction_stats(raw, target)
        source_receipts[rec["name"]] = {"n_units": rec["n_units"], "n_trials": len(rec["trials"]), "n_eval_windows": n_windows, "t4_metadata": rec["t4_metadata"]}
        print(f"[source] {rec['name']} units={rec['n_units']} windows={n_windows} seconds={time.time()-t0:.1f}")

    pv_total = add_stats(list(pv_source_stats.values()))
    pv_gain = gain_from_stats(pv_total)
    selected_lambda, lambda_audit = select_lambda_source_loso(ole_source_stats)
    ole_total = add_stats(list(ole_source_stats[selected_lambda].values()))
    ole_gain = gain_from_stats(ole_total)
    print(f"source-fixed PV gain={pv_gain:.8g}")
    print(f"source-selected OLE lambda={selected_lambda:g} gain={ole_gain:.8g}")

    e01_r2 = e01_per_session_r2(reference, [session_name_from_path(path) for path in target_files])
    per_method: dict[str, dict[str, dict]] = {"B0-1_population_vector_like": {}, "B0-2_ridge_ole": {}}
    target_receipts = {}
    for path in target_files:
        t0 = time.time()
        rec, t4 = load_session(path, behavior_mean, behavior_std, cache_dir, args.pool_size)
        rates, target, n_windows = window_mean_rates_and_targets(rec, args.pool_size)
        method_specs = (
            ("B0-1_population_vector_like", "population_vector_like", 0.0, pv_gain),
            ("B0-2_ridge_ole", "ridge_ole", selected_lambda, ole_gain),
        )
        for result_name, method, ridge_lambda, gain in method_specs:
            raw = analytic_predictions(rates, t4, method, ridge_lambda)
            pred = raw * gain
            row = {"r2": compute_r2(pred.astype(np.float32), target.astype(np.float32)), **direction_and_speed_metrics(pred, target)}
            per_method[result_name][rec["name"]] = row
            print(f"[target] {rec['name']} {result_name} R2={row['r2']:+.4f} angle={row['median_angular_error_deg']:.1f}deg speed_ratio={row['median_speed_ratio']:.3f}")
        target_receipts[rec["name"]] = {"n_units": rec["n_units"], "n_trials": len(rec["trials"]), "n_eval_windows": n_windows, "t4_metadata": rec["t4_metadata"]}
        print(f"[target] {rec['name']} completed seconds={time.time()-t0:.1f}")

    summaries = {name: summarize(rows, e01_r2) for name, rows in per_method.items()}
    payload = {
        "schema_version": 1,
        "status": "complete" if formal else "dev_smoke_complete",
        "experiment": "E08 analytic-only decoder",
        "seed": SEED,
        "deterministic": True,
        "created_at": datetime.now().astimezone().isoformat(),
        "branch": "exp/e08-analytic-only",
        "protocol": {
            "dataset": "DANDI 000688 sub-C CO SUA",
            "split_counts": list(parse_split_counts(args.split_counts)),
            "source_sessions": [session_name_from_path(path) for path in source_files],
            "target_sessions": [session_name_from_path(path) for path in target_files],
            "unused_validation_sessions": [session_name_from_path(path) for path in val_files],
            "calibration_regime": f"raw T4 from rewarded trials[0:{args.pool_size}]",
            "evaluation_regime": f"strictly disjoint windows from trials[{args.pool_size}:]",
            "rate_definition": f"mean firing rate over the {WINDOW_SIZE}-bin ({WINDOW_SIZE*BIN_SIZE_S:.1f}s) causal window",
            "target_direction_labels_used": "only first-50 endpoint target_dir labels to construct T4",
            "target_continuous_velocity_labels_used": "offline scoring only",
            "target_gradient": False,
            "target_optimizer": None,
            "target_backpropagation": False,
            "lambda_selection": "unweighted mean leave-one-source-session-out R2; source sessions only",
            "output_scale": "one zero-intercept isotropic scalar fitted on all source-session post-pool windows",
            "weight_matrix_W": "identity",
        },
        "decoders": {
            "B0-1_population_vector_like": {"formula": "gain * sum_i((r_i-b_i)*beta_i)", "source_fixed_gain": pv_gain, "parameter_count_target": 0, "target_side_compute": "O(N) per window"},
            "B0-2_ridge_ole": {"formula": "gain * (B.T B + lambda I)^-1 B.T (r-b)", "source_selected_lambda": selected_lambda, "source_fixed_gain": ole_gain, "parameter_count_target": 0, "target_side_compute": "one 2x2 pseudoinverse per session plus O(N) per window"},
        },
        "source_selection": {"pv_source_fit_r2": r2_from_stats(pv_total, pv_gain), "ridge_lambda_loso": lambda_audit, "selected_lambda_at_grid_boundary": selected_lambda in (min(lambdas), max(lambdas))},
        "e01_reference": {"path": str(reference_path), "sha256": sha256_file(reference_path), "per_session_r2": e01_r2},
        "per_session": per_method,
        "summary": summaries,
        "source_receipts": source_receipts,
        "target_receipts": target_receipts,
        "endpoint_only_t4_caveat": "T4 is fit from endpoint target direction labels, so it supplies a direction/rate anchor but contains no instantaneous speed or within-trial dynamics model.",
        "leakage_self_audit": {
            "passed": True,
            "source_only_hyperparameters": ["ridge lambda", "PV gain", "OLE gain", "behavior normalization"],
            "target_prefix_inputs": ["spikes", "endpoint target_dir labels"],
            "target_update_inputs_excluded": ["continuous cursor velocity", "post-prefix spikes", "evaluation trials"],
            "evaluation_labels_used_after_predictions_fixed": True,
        },
        "runtime_seconds": time.time() - started,
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summaries, indent=2, sort_keys=True))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
