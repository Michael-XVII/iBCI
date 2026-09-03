"""Strict evaluation-only test_heldout compatibility loop for frozen H1 M3 packages."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.h1_cal_aug_all_source_m3_deployment_v1_contract import ARMS, HELDOUT_SESSION_TO_FALCON_KEY
from src.h1_cal_aug_all_source_m3_spint_style_heldout_calib_r2_v1 import (
    CHECKPOINT_SHA256,
    heldout_paths,
    legacy_spint_score_mask,
)
from src.h1_hc_date_lodo_regen_v1 import (
    publish_json,
    publish_npz,
    publish_text,
    variance_weighted_r2,
    verify_sidecar,
)
from src.h1_m4_cce_contract import sha256_file, state_hash


SCHEMA = "h1_cal_aug_all_source_m3_test_heldout_compat_v1"
LABEL = "local SPINT-style held-out-calib test-loop R²"
STATUS_PREDECESSOR = "PASS_H1_M3_TEST_HELDOUT_COMPAT_PREDECESSOR"
STATUS_METRICS = "PASS_H1_M3_TEST_HELDOUT_COMPAT_NUMERICAL_EQUIVALENCE"
STATUS_TERMINAL = "COMPLETE_H1_M3_TEST_HELDOUT_COMPAT"
VAL_SCHEMA = "h1_cal_aug_all_source_m3_spint_style_heldout_calib_r2_v1"
VAL_TERMINAL_SHA256 = "a508f83bb9c22fe4d21329e7b02debae337209d3a6822f690a47b2385b20f5b4"
VAL_METRICS_SHA256 = "335b95ee1465f7687347a86de5161d506a3e9b6426a3606ab976622a23fcb03b"
PACKAGE_SCHEMA = "h1_cal_aug_all_source_m3_deployment_v1_package"
WINDOW_SIZE = 700
PREDICTION_DIVISOR = 20.0
PRED_RTOL = 2e-3
PRED_ATOL = 2e-4
R2_ABS_TOL = 2e-4


class TestHeldoutCompatError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise TestHeldoutCompatError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path, schema: str | None = None) -> tuple[dict[str, Any], str]:
    digest = verify_sidecar(path)
    body = json.loads(path.read_text(encoding="utf-8"))
    if schema is not None:
        _need(body.get("schema") == schema, f"schema drift: {path}")
    return body, digest


def create_attempt(result_root: Path, package_root: Path, val_root: Path, closure: Mapping[str, str], head: str) -> dict[str, Any]:
    root = result_root.resolve()
    _need(not root.exists(), f"compatibility result root is not fresh: {root}")
    body = {
        "schema": SCHEMA,
        "artifact": "attempt",
        "status": "ATTEMPT_BEFORE_TEST_HELDOUT_DATA_AND_CUDA",
        "created_at_utc": utc_now(),
        "git_head": head,
        "closure": dict(closure),
        "package_root": str(package_root.resolve()),
        "val_result_root": str(val_root.resolve()),
        "training": False,
        "checkpoint_selection": False,
        "optimizer_steps": 0,
        "backward_steps": 0,
        "model_updates": 0,
        "heldout_recordings_opened": 0,
        "cuda_initialized": False,
        "evalai_submissions": 0,
    }
    publish_json(root / "attempt.json", body)
    return body


def load_attempt(result_root: Path) -> dict[str, Any]:
    body, _ = _load(result_root.resolve() / "attempt.json", SCHEMA)
    _need(body.get("status") == "ATTEMPT_BEFORE_TEST_HELDOUT_DATA_AND_CUDA", "attempt status drift")
    _need(body.get("training") is False and body.get("checkpoint_selection") is False, "attempt training/selection drift")
    _need(body.get("optimizer_steps") == body.get("backward_steps") == body.get("model_updates") == 0, "attempt update drift")
    return body


def validate_predecessor(package_root: Path, val_root: Path, result_root: Path) -> dict[str, Any]:
    import torch

    val_terminal, val_terminal_sha = _load(val_root.resolve() / "terminal.json", VAL_SCHEMA)
    _need(val_terminal_sha == VAL_TERMINAL_SHA256, "sealed val-style terminal SHA drift")
    val_metrics, val_metrics_sha = _load(val_root.resolve() / "evaluation/metrics.json", f"{VAL_SCHEMA}_metrics")
    _need(val_metrics_sha == VAL_METRICS_SHA256 == val_terminal["metrics_sha256"], "sealed val-style metrics SHA drift")
    val_cache_manifest, val_cache_manifest_sha = _load(val_root.resolve() / "evaluation/prediction_cache.json", f"{VAL_SCHEMA}_prediction_cache")
    val_cache_path = val_root.resolve() / "evaluation/prediction_cache.npz"
    _need(verify_sidecar(val_cache_path) == val_metrics["prediction_cache_sha256"] == val_cache_manifest["arrays_file_sha256"], "sealed val-style cache drift")
    package_authority, package_authority_sha = _load(package_root.resolve() / "packages/packages.json", "h1_cal_aug_all_source_m3_deployment_v1_packages")
    rows = []
    for row in package_authority["packages"]:
        arm = row["arm"]
        _need(arm in ARMS and row["checkpoint_sha256"] == CHECKPOINT_SHA256[arm], f"{arm} checkpoint drift")
        path = package_root.resolve() / row["relative"]
        _need(verify_sidecar(path) == row["sha256"], f"{arm} package SHA drift")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _need(payload.get("schema") == PACKAGE_SCHEMA and payload.get("checkpoint_sha256") == CHECKPOINT_SHA256[arm], f"{arm} package provenance drift")
        _need(state_hash(payload["state_dict"]) == payload.get("model_state_sha256") == row["model_state_sha256"], f"{arm} state drift")
        rows.append({
            "arm": arm,
            "package_relative": row["relative"],
            "package_sha256": row["sha256"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "model_state_sha256": row["model_state_sha256"],
        })
    body = {
        "schema": f"{SCHEMA}_predecessor_authority",
        "status": STATUS_PREDECESSOR,
        "val_terminal_sha256": val_terminal_sha,
        "val_metrics_sha256": val_metrics_sha,
        "val_prediction_cache_manifest_sha256": val_cache_manifest_sha,
        "val_prediction_cache_sha256": val_metrics["prediction_cache_sha256"],
        "package_authority_sha256": package_authority_sha,
        "packages": rows,
        "training": False,
        "checkpoint_selection": False,
        "optimizer_steps": 0,
        "backward_steps": 0,
        "model_updates": 0,
        "evalai_submissions": 0,
    }
    publish_json(result_root.resolve() / "predecessor_authority.json", body)
    return body


def test_loop_inference(path: Path, payload: Mapping[str, Any], key: str, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str]:
    import torch
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb
    from src.models.components.h1_carrierid_spint import H1CarrierIdSpint

    neural, targets, _trial_change, eval_mask = load_nwb(path, FalconTask.h1)
    neural = np.asarray(neural, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    eval_mask = np.asarray(eval_mask, dtype=bool).reshape(-1)
    _need(neural.ndim == 2 and neural.shape[1] == 176 and targets.shape == (len(neural), 7), f"NWB shape drift: {key}")
    _need(len(eval_mask) == len(neural), f"eval-mask drift: {key}")
    score_mask = legacy_spint_score_mask(eval_mask, batch_size=32)
    indices = np.flatnonzero(score_mask)
    _need(len(indices) % 32 == 0, f"legacy batch population drift: {key}")
    session = payload["sessions"][key]
    identity = torch.as_tensor(np.asarray(session["identity"], dtype=np.float32), device=device).unsqueeze(0)
    carrier = torch.as_tensor(np.asarray(session["carrier"], dtype=np.float32), device=device).unsqueeze(0)
    model = H1CarrierIdSpint(**payload["model_kwargs"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval().to(device)
    before = state_hash(model.state_dict())
    padded = np.pad(neural, ((WINDOW_SIZE - 1, 0), (0, 0)), constant_values=0.0)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(indices), 32):
            batch_indices = indices[start:start + 32]
            windows = np.stack([padded[index:index + WINDOW_SIZE] for index in batch_indices]).astype(np.float32, copy=False)
            x = torch.as_tensor(windows, dtype=torch.float32, device=device)
            output = model(x, calib_trialized_neural_features=identity.expand(len(x), -1, -1, -1), carrier=carrier.expand(len(x), -1, -1))
            predictions.append((output[:, -1, :] / PREDICTION_DIVISOR).detach().cpu().numpy().astype(np.float32))
    prediction = np.concatenate(predictions, axis=0)
    target = targets[indices]
    after = state_hash(model.state_dict())
    _need(before == after == payload["model_state_sha256"], f"model state changed: {key}")
    return prediction, target, indices.astype(np.int64), before, after


def run_compatibility(package_root: Path, val_root: Path, data_root: Path, result_root: Path, *, device: str = "cuda:0") -> dict[str, Any]:
    import torch

    authority, authority_sha = _load(result_root.resolve() / "predecessor_authority.json", f"{SCHEMA}_predecessor_authority")
    _need(authority["status"] == STATUS_PREDECESSOR, "predecessor gate missing")
    paths = heldout_paths(data_root)
    packages = {row["arm"]: row for row in authority["packages"]}
    val_metrics, _ = _load(val_root.resolve() / "evaluation/metrics.json", f"{VAL_SCHEMA}_metrics")
    with np.load(val_root.resolve() / "evaluation/prediction_cache.npz", allow_pickle=False) as cache:
        val_cache = {name: np.asarray(cache[name]) for name in cache.files}
    arm_results = {}
    result_arrays = {}
    equivalence = {}
    access_rows = []
    for arm in ARMS:
        row = packages[arm]
        package_path = package_root.resolve() / row["package_relative"]
        payload = torch.load(package_path, map_location="cpu", weights_only=False)
        scores = {}
        session_equivalence = {}
        state_before = set()
        state_after = set()
        for index, ((session, key), path) in enumerate(zip(HELDOUT_SESSION_TO_FALCON_KEY, paths, strict=True)):
            prediction, target, indices, before, after = test_loop_inference(path, payload, key, device)
            state_before.add(before); state_after.add(after)
            old_mask = np.asarray(val_cache[f"{arm}_{index}_legacy_score_mask"], dtype=bool)
            old_indices = np.flatnonzero(old_mask)
            old_prediction = np.asarray(val_cache[f"{arm}_{index}_prediction"], dtype=np.float32)[old_mask]
            old_target = np.asarray(val_cache[f"{arm}_{index}_target"], dtype=np.float32)[old_mask]
            _need(np.array_equal(indices, old_indices), f"{arm}/{key}: scoring population differs")
            _need(np.array_equal(target, old_target), f"{arm}/{key}: target bytes differ")
            pred_close = bool(np.allclose(prediction, old_prediction, rtol=PRED_RTOL, atol=PRED_ATOL))
            _need(pred_close, f"{arm}/{key}: predictions exceed floating tolerance")
            score = variance_weighted_r2(target, prediction)
            old_score = float(val_metrics["per_session"][key][arm])
            r2_diff = abs(score - old_score)
            _need(r2_diff <= R2_ABS_TOL, f"{arm}/{key}: R² differs by {r2_diff}")
            scores[key] = score
            session_equivalence[key] = {
                "score_population_exact": True,
                "target_exact": True,
                "prediction_allclose": pred_close,
                "prediction_max_abs_diff": float(np.max(np.abs(prediction.astype(np.float64) - old_prediction.astype(np.float64)))),
                "r2_abs_diff": r2_diff,
                "scored_bins": int(len(indices)),
            }
            prefix = f"{arm}_{index}"
            result_arrays[f"{prefix}_prediction"] = prediction
            result_arrays[f"{prefix}_target"] = target
            result_arrays[f"{prefix}_source_indices"] = indices
        values = np.asarray([scores[key] for _, key in HELDOUT_SESSION_TO_FALCON_KEY], dtype=np.float64)
        mean = float(np.mean(values, dtype=np.float64))
        sample_std = float(np.std(values, ddof=1, dtype=np.float64))
        population_std = float(np.std(values, ddof=0, dtype=np.float64))
        mean_diff = abs(mean - float(val_metrics["arms"][arm]["equal_session_mean_r2"]))
        _need(mean_diff <= R2_ABS_TOL, f"{arm}: mean R² equivalence failed")
        arm_results[arm] = {
            "test_heldout/r2_mean": mean,
            "test_heldout/r2_std": sample_std,
            "population_std_auxiliary": population_std,
            "per_session_r2": scores,
            "model_state_before_sha256": next(iter(state_before)),
            "model_state_after_sha256": next(iter(state_after)),
        }
        equivalence[arm] = {"per_session": session_equivalence, "mean_r2_abs_diff": mean_diff}
        access_rows.append({
            "arm": arm,
            "recordings_opened": 14,
            "files": [{"session": session, "falcon_key": key, "filename": path.name, "bytes": int(path.stat().st_size), "sha256": sha256_file(path)} for (session, key), path in zip(HELDOUT_SESSION_TO_FALCON_KEY, paths, strict=True)],
        })
    per_session = {}
    for _session, key in HELDOUT_SESSION_TO_FALCON_KEY:
        t0 = arm_results["t0"]["per_session_r2"][key]
        c1 = arm_results["c1"]["per_session_r2"][key]
        per_session[key] = {"t0": t0, "c1": c1, "delta_c1_minus_t0": c1 - t0}
    delta = arm_results["c1"]["test_heldout/r2_mean"] - arm_results["t0"]["test_heldout/r2_mean"]
    old_delta = float(val_metrics["equal_session_mean_delta_c1_minus_t0"])
    _need(abs(delta - old_delta) <= R2_ABS_TOL, "C1-T0 aggregate equivalence failed")
    cache_sha = publish_npz(result_root.resolve() / "evaluation/test_loop_cache.npz", **result_arrays)
    access_sha = publish_json(result_root.resolve() / "evaluation/target_access.json", {
        "schema": f"{SCHEMA}_target_access",
        "scope": "14 public held-out-calib recordings; same M3 calibration trials reused for local test-loop scoring",
        "arms": access_rows,
        "unique_recordings": 14,
        "total_recording_passes": 28,
        "independent_postcalibration_stream": False,
        "evalai_submissions": 0,
        "optimizer_steps": 0,
        "backward_steps": 0,
        "model_updates": 0,
    })
    body = {
        "schema": f"{SCHEMA}_metrics",
        "status": STATUS_METRICS,
        "label": LABEL,
        "predecessor_authority_sha256": authority_sha,
        "arms": arm_results,
        "per_session": per_session,
        "mean_delta_c1_minus_t0": delta,
        "positive_session_count": sum(row["delta_c1_minus_t0"] > 0 for row in per_session.values()),
        "val_style_reference": {"t0": val_metrics["arms"]["t0"]["equal_session_mean_r2"], "c1": val_metrics["arms"]["c1"]["equal_session_mean_r2"], "delta": old_delta},
        "numerical_equivalence": equivalence,
        "prediction_tolerance": {"rtol": PRED_RTOL, "atol": PRED_ATOL},
        "r2_absolute_tolerance": R2_ABS_TOL,
        "scoring_population_identical": True,
        "test_loop_cache_sha256": cache_sha,
        "target_access_sha256": access_sha,
        "training": False,
        "checkpoint_selection": False,
        "optimizer_steps": 0,
        "backward_steps": 0,
        "model_updates": 0,
        "evalai_submissions": 0,
        "official_hidden_test_r2": False,
    }
    publish_json(result_root.resolve() / "evaluation/metrics.json", body)
    return body


def verify_terminal(result_root: Path) -> dict[str, Any]:
    root = result_root.resolve()
    attempt = load_attempt(root)
    attempt_sha = verify_sidecar(root / "attempt.json")
    authority, authority_sha = _load(root / "predecessor_authority.json", f"{SCHEMA}_predecessor_authority")
    metrics, metrics_sha = _load(root / "evaluation/metrics.json", f"{SCHEMA}_metrics")
    _need(authority["status"] == STATUS_PREDECESSOR and metrics["status"] == STATUS_METRICS, "terminal status drift")
    _need(metrics["scoring_population_identical"] is True, "scoring population mismatch")
    _need(metrics["training"] is False and metrics["checkpoint_selection"] is False, "forbidden training/selection")
    _need(metrics["optimizer_steps"] == metrics["backward_steps"] == metrics["model_updates"] == metrics["evalai_submissions"] == 0, "forbidden execution count")
    body = {
        "schema": SCHEMA,
        "status": STATUS_TERMINAL,
        "finished_at_utc": utc_now(),
        "attempt_sha256": attempt_sha,
        "git_head": attempt["git_head"],
        "predecessor_authority_sha256": authority_sha,
        "metrics_sha256": metrics_sha,
        "label": LABEL,
        "test_heldout/r2_mean": {arm: metrics["arms"][arm]["test_heldout/r2_mean"] for arm in ARMS},
        "test_heldout/r2_std": {arm: metrics["arms"][arm]["test_heldout/r2_std"] for arm in ARMS},
        "per_session": metrics["per_session"],
        "mean_delta_c1_minus_t0": metrics["mean_delta_c1_minus_t0"],
        "positive_session_count": metrics["positive_session_count"],
        "val_style_reference": metrics["val_style_reference"],
        "scoring_population_identical": True,
        "numerical_equivalence_passed": True,
        "training": False,
        "checkpoint_selection": False,
        "optimizer_steps": 0,
        "backward_steps": 0,
        "model_updates": 0,
        "evalai_submissions": 0,
        "official_hidden_test_r2": False,
    }
    terminal_sha = publish_json(root / "terminal.json", body)
    lines = [
        "# H1 M3 Test-Heldout Compatibility V1", "",
        f"- Status: `{STATUS_TERMINAL}`",
        f"- Label: `{LABEL}`",
        f"- T0 test_heldout/r2_mean: `{body['test_heldout/r2_mean']['t0']:.9f}`",
        f"- T0 test_heldout/r2_std: `{body['test_heldout/r2_std']['t0']:.9f}`",
        f"- C1 test_heldout/r2_mean: `{body['test_heldout/r2_mean']['c1']:.9f}`",
        f"- C1 test_heldout/r2_std: `{body['test_heldout/r2_std']['c1']:.9f}`",
        f"- Mean delta C1-T0: `{body['mean_delta_c1_minus_t0']:+.9f}`",
        f"- Positive sessions: `{body['positive_session_count']}/14`",
        "- Val-style scoring population and metrics numerically equivalent: `true`.",
        "- Official hidden-test R² / training / EvalAI submissions: `false` / `false` / `0`.", "",
        "| Session | T0 R² | C1 R² | Delta |", "|---|---:|---:|---:|",
    ]
    for key, row in body["per_session"].items():
        lines.append(f"| {key} | {row['t0']:.9f} | {row['c1']:.9f} | {row['delta_c1_minus_t0']:+.9f} |")
    lines.extend(["", f"Terminal SHA-256: `{terminal_sha}`"])
    publish_text(root / "EXPERIMENT_RECORD.md", "\n".join(lines) + "\n")
    return body


__all__ = (
    "LABEL", "R2_ABS_TOL", "SCHEMA", "STATUS_METRICS", "STATUS_TERMINAL",
    "TestHeldoutCompatError", "create_attempt", "load_attempt", "run_compatibility",
    "test_loop_inference", "validate_predecessor", "verify_terminal",
)
