"""Additive original-SPINT-style held-out-calib diagnostic for frozen H1 M3 packages."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.data.h1_cal_aug_all_source_heldout_v1 import H1_HELDOUT_SESSIONS, index_heldout_calib
from src.data.h1_m4_eb_pilot import array_sha256, session_from_path
from src.h1_cal_aug_all_source_m3_deployment_v1_contract import (
    ARMS,
    HELDOUT_SESSION_TO_FALCON_KEY,
)
from src.h1_cal_aug_all_source_m3_deployment_v1_exec import _load_json
from src.h1_hc_date_lodo_regen_v1 import (
    publish_json,
    publish_npz,
    publish_text,
    variance_weighted_r2,
    verify_sidecar,
)
from src.h1_m4_cce_contract import sha256_file, state_hash


SCHEMA = "h1_cal_aug_all_source_m3_spint_style_heldout_calib_r2_v1"
LABEL = "local original-SPINT-style held-out-calib validation R²"
STATUS_PREDECESSOR = "PASS_H1_M3_SPINT_STYLE_HELDOUT_CALIB_PREDECESSOR"
STATUS_METRICS = "PASS_H1_M3_SPINT_STYLE_HELDOUT_CALIB_R2"
STATUS_TERMINAL = "COMPLETE_H1_M3_SPINT_STYLE_HELDOUT_CALIB_DIAGNOSTIC"
PACKAGE_A1_SCHEMA = "h1_cal_aug_all_source_m3_deployment_v1_package_a1"
PACKAGE_SCHEMA = "h1_cal_aug_all_source_m3_deployment_v1_packages"
PACKAGE_A1_TERMINAL_SHA256 = "4137495462a299e948beb58be578c739cc211330de4769992c03e743d7c7bf26"
CHECKPOINT_SHA256 = {
    "t0": "6d4d14226b706951274982438b588527beb442200aad2f50f9d18b68e54a9648",
    "c1": "0f406a8e69fdb57cf6a5480149f04ab3500e7fad849d36db38042edbadb2cd06",
}
LEGACY_BATCH_SIZE = 32


class SpintStyleHeldoutError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise SpintStyleHeldoutError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_attempt(result_root: Path, predecessor_root: Path, closure: Mapping[str, str], head: str) -> dict[str, Any]:
    root = result_root.resolve()
    _need(not root.exists(), f"diagnostic result root is not fresh: {root}")
    body = {
        "schema": SCHEMA,
        "artifact": "attempt",
        "status": "ATTEMPT_BEFORE_HELDOUT_CALIB_SCORING_AND_CUDA",
        "created_at_utc": utc_now(),
        "git_head": head,
        "closure": dict(closure),
        "predecessor_root": str(predecessor_root.resolve()),
        "training": False,
        "optimizer_steps": 0,
        "backward_steps": 0,
        "model_updates": 0,
        "heldout_calib_scoring_recordings_opened": 0,
        "evalai_submissions": 0,
        "official_heldout_score_accessed": False,
        "cuda_initialized": False,
    }
    publish_json(root / "attempt.json", body)
    return body


def load_attempt(result_root: Path) -> dict[str, Any]:
    body, _ = _load_json(result_root.resolve() / "attempt.json", SCHEMA)
    _need(body.get("status") == "ATTEMPT_BEFORE_HELDOUT_CALIB_SCORING_AND_CUDA", "attempt status drift")
    _need(body.get("training") is False, "training authorized by attempt")
    _need(body.get("optimizer_steps") == body.get("backward_steps") == body.get("model_updates") == 0, "attempt update count drift")
    return body


def legacy_spint_score_mask(eval_mask: np.ndarray, batch_size: int = LEGACY_BATCH_SIZE) -> np.ndarray:
    """Reproduce SessionBatchSampler's chronological, per-session drop-last population."""
    mask = np.asarray(eval_mask, dtype=bool).reshape(-1)
    _need(int(batch_size) > 0, "legacy batch size must be positive")
    indices = np.flatnonzero(mask)
    kept = (len(indices) // int(batch_size)) * int(batch_size)
    _need(kept > 1, "insufficient eval-valid bins for legacy SPINT R²")
    result = np.zeros(mask.shape, dtype=bool)
    result[indices[:kept]] = True
    return result


def heldout_paths(data_root: Path) -> tuple[Path, ...]:
    observed = index_heldout_calib(data_root)
    _need(tuple(observed) == H1_HELDOUT_SESSIONS, "held-out-calib roster/order drift")
    paths = tuple(observed[session].resolve() for session in H1_HELDOUT_SESSIONS)
    _need(tuple(session_from_path(path) for path in paths) == H1_HELDOUT_SESSIONS, "held-out filename/session drift")
    return paths


def validate_predecessor(predecessor_root: Path, result_root: Path) -> dict[str, Any]:
    import torch

    root = predecessor_root.resolve()
    terminal, terminal_sha = _load_json(root / "terminal.json", PACKAGE_A1_SCHEMA)
    _need(terminal_sha == PACKAGE_A1_TERMINAL_SHA256, "Package-A1 terminal SHA drift")
    _need(terminal.get("status") == "COMPLETE_LOCAL_H1_ALL_SOURCE_M3_DEPLOYMENT_READY_NO_EVALAI_SUBMISSION", "Package-A1 terminal status drift")
    packages, packages_sha = _load_json(root / "packages/packages.json", PACKAGE_SCHEMA)
    _need(packages_sha == terminal["packages_sha256"] and len(packages.get("packages", ())) == 2, "package authority drift")
    rows = []
    for row in packages["packages"]:
        arm = row["arm"]
        _need(arm in ARMS and row["checkpoint_sha256"] == CHECKPOINT_SHA256[arm], f"{arm} checkpoint binding drift")
        path = root / row["relative"]
        _need(verify_sidecar(path) == row["sha256"], f"{arm} package SHA drift")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _need(payload.get("checkpoint_sha256") == CHECKPOINT_SHA256[arm], f"{arm} embedded checkpoint SHA drift")
        _need(state_hash(payload["state_dict"]) == row["model_state_sha256"] == payload.get("model_state_sha256"), f"{arm} model state drift")
        _need(payload.get("optimizer_steps") == payload.get("backward_steps") == payload.get("model_updates") == 0, f"{arm} packaged updates drift")
        rows.append({
            "arm": arm,
            "package_relative": row["relative"],
            "package_sha256": row["sha256"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "model_state_sha256": row["model_state_sha256"],
        })
    calibration, calibration_sha = _load_json(root / "packages/calibration_authority.json", f"{PACKAGE_A1_SCHEMA}_calibration_authority")
    heldout = [row for row in calibration["sessions"] if row["scope"] == "held-out-calib"]
    _need(len(heldout) == 14 and all(row["legal_trial_count"] == 3 and len(row["calibration_trials"]) == 3 for row in heldout), "frozen held-out M3 calibration drift")
    body = {
        "schema": f"{SCHEMA}_predecessor_authority",
        "status": STATUS_PREDECESSOR,
        "package_a1_terminal_sha256": terminal_sha,
        "packages_sha256": packages_sha,
        "calibration_authority_sha256": calibration_sha,
        "source_authority_sha256": terminal["source_authority_sha256"],
        "pair_integrity_sha256": terminal["pair_integrity_sha256"],
        "packages": rows,
        "heldout_m3_payloads": 14,
        "training": False,
        "optimizer_steps": 0,
        "backward_steps": 0,
        "model_updates": 0,
        "evalai_submissions": 0,
    }
    publish_json(result_root.resolve() / "predecessor_authority.json", body)
    return body


def evaluate(data_root: Path, predecessor_root: Path, result_root: Path, *, device: str = "cuda:0") -> dict[str, Any]:
    import falcon_challenge.evaluator as evaluator_module
    from falcon_challenge.config import FalconConfig, FalconTask
    from falcon_challenge.evaluator import FalconEvaluator
    from third_party.falcon_challenge.h1_carrier_id_spint_decoder import H1CarrierIdSpintDecoder

    load_attempt(result_root)
    authority, authority_sha = _load_json(result_root.resolve() / "predecessor_authority.json", f"{SCHEMA}_predecessor_authority")
    _need(authority["status"] == STATUS_PREDECESSOR, "predecessor gate must pass before evaluation")
    paths = heldout_paths(data_root)
    session_to_key = dict(HELDOUT_SESSION_TO_FALCON_KEY)
    packages = {row["arm"]: row for row in authority["packages"]}
    arm_results: dict[str, Any] = {}
    cache_arrays: dict[str, np.ndarray] = {}
    access_rows = []
    original_tqdm = evaluator_module.tqdm
    evaluator_module.tqdm = lambda iterable, *args, **kwargs: iterable
    try:
        for arm in ARMS:
            row = packages[arm]
            package_path = predecessor_root.resolve() / row["package_relative"]
            _need(verify_sidecar(package_path) == row["package_sha256"], f"{arm} package changed before evaluation")
            decoder = H1CarrierIdSpintDecoder(FalconConfig(task=FalconTask.h1), package_path, batch_size=14, device=device)
            before = decoder.model_state_sha256()
            evaluator = FalconEvaluator(eval_remote=False, split="h1", verbose=False, dataloader_workers=0)
            predictions, targets, masks, _compute_times, _neural_times = evaluator.predict_files(decoder, list(paths))
            after = decoder.model_state_sha256()
            _need(before == after == row["model_state_sha256"], f"{arm} model state changed during held-out-calib diagnostic")
            scores = {}
            counts = {}
            dropped = {}
            for index, (session, key) in enumerate(HELDOUT_SESSION_TO_FALCON_KEY):
                _need(key == session_to_key[session] and key in predictions and key in targets and key in masks, f"missing held-out prediction key: {key}")
                prediction = np.asarray(predictions[key], dtype=np.float32)
                target = np.asarray(targets[key], dtype=np.float32)
                eval_mask = np.asarray(masks[key], dtype=bool).reshape(-1)
                _need(prediction.shape == target.shape and prediction.ndim == 2 and prediction.shape[1] == 7 and len(eval_mask) == len(prediction), f"held-out array shape drift: {key}")
                score_mask = legacy_spint_score_mask(eval_mask)
                scores[key] = variance_weighted_r2(target[score_mask], prediction[score_mask])
                counts[key] = int(score_mask.sum())
                dropped[key] = int(eval_mask.sum() - score_mask.sum())
                prefix = f"{arm}_{index}"
                cache_arrays[f"{prefix}_prediction"] = prediction
                cache_arrays[f"{prefix}_target"] = target
                cache_arrays[f"{prefix}_eval_mask"] = eval_mask
                cache_arrays[f"{prefix}_legacy_score_mask"] = score_mask
            values = np.asarray([scores[key] for _, key in HELDOUT_SESSION_TO_FALCON_KEY], dtype=np.float64)
            arm_results[arm] = {
                "per_session_r2": scores,
                "equal_session_mean_r2": float(np.mean(values, dtype=np.float64)),
                "population_std_r2": float(np.std(values, ddof=0, dtype=np.float64)),
                "legacy_scored_bins": counts,
                "legacy_drop_last_bins": dropped,
                "model_state_before_sha256": before,
                "model_state_after_sha256": after,
            }
            access_rows.append({
                "arm": arm,
                "recordings_opened": 14,
                "bytes_read": sum(int(path.stat().st_size) for path in paths),
                "files": [
                    {"session": session, "falcon_key": key, "filename": path.name, "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}
                    for (session, key), path in zip(HELDOUT_SESSION_TO_FALCON_KEY, paths, strict=True)
                ],
            })
    finally:
        evaluator_module.tqdm = original_tqdm

    for index, (_session, _key) in enumerate(HELDOUT_SESSION_TO_FALCON_KEY):
        _need(np.array_equal(cache_arrays[f"t0_{index}_target"], cache_arrays[f"c1_{index}_target"]), "T0/C1 held-out targets differ")
        _need(np.array_equal(cache_arrays[f"t0_{index}_legacy_score_mask"], cache_arrays[f"c1_{index}_legacy_score_mask"]), "T0/C1 held-out score masks differ")
    per_session = {}
    for _session, key in HELDOUT_SESSION_TO_FALCON_KEY:
        t0 = float(arm_results["t0"]["per_session_r2"][key])
        c1 = float(arm_results["c1"]["per_session_r2"][key])
        per_session[key] = {"t0": t0, "c1": c1, "delta_c1_minus_t0": c1 - t0}
    cache_sha = publish_npz(result_root.resolve() / "evaluation/prediction_cache.npz", **cache_arrays)
    cache_manifest_sha = publish_json(result_root.resolve() / "evaluation/prediction_cache.json", {
        "schema": f"{SCHEMA}_prediction_cache",
        "arrays_file_sha256": cache_sha,
        "array_sha256": {name: array_sha256(value) for name, value in cache_arrays.items()},
        "array_shape": {name: list(value.shape) for name, value in cache_arrays.items()},
    })
    access_sha = publish_json(result_root.resolve() / "evaluation/target_access.json", {
        "schema": f"{SCHEMA}_target_access",
        "scope": "14 public held-out-calib NWBs reused for frozen M3 calibration and local validation scoring",
        "arms": access_rows,
        "total_recordings_opened": 28,
        "unique_recordings_opened": 14,
        "calibration_trials_reused_for_scoring": True,
        "independent_postcalibration_stream": False,
        "evalai_test_recordings_opened": 0,
        "official_heldout_score_accessed": False,
        "optimizer_steps": 0,
        "backward_steps": 0,
        "model_updates": 0,
        "evalai_submissions": 0,
    })
    means = {arm: arm_results[arm]["equal_session_mean_r2"] for arm in ARMS}
    body = {
        "schema": f"{SCHEMA}_metrics",
        "status": STATUS_METRICS,
        "label": LABEL,
        "semantic_match": "original SPINT held-out-calib eval-mask, W700 zero-padded last-bin population, per-session batch32 drop-last, equal-session aggregation",
        "successor_calibration_difference": "frozen earliest-M3 identity plus H-C carrier; original generic SPINT used calibration_n_trials=2 and no H-C carrier",
        "calibration_trials_reused_for_scoring": True,
        "official_heldout_r2": False,
        "predecessor_authority_sha256": authority_sha,
        "arms": arm_results,
        "per_session": per_session,
        "equal_session_mean_delta_c1_minus_t0": means["c1"] - means["t0"],
        "positive_session_count": sum(row["delta_c1_minus_t0"] > 0.0 for row in per_session.values()),
        "prediction_cache_sha256": cache_sha,
        "prediction_cache_manifest_sha256": cache_manifest_sha,
        "target_access_sha256": access_sha,
        "training": False,
        "checkpoint_or_hyperparameter_selection": False,
        "optimizer_steps": 0,
        "backward_steps": 0,
        "model_updates": 0,
        "evalai_submissions": 0,
    }
    publish_json(result_root.resolve() / "evaluation/metrics.json", body)
    return body


def verify_terminal(result_root: Path) -> dict[str, Any]:
    root = result_root.resolve()
    attempt = load_attempt(root)
    attempt_sha = verify_sidecar(root / "attempt.json")
    authority, authority_sha = _load_json(root / "predecessor_authority.json", f"{SCHEMA}_predecessor_authority")
    metrics, metrics_sha = _load_json(root / "evaluation/metrics.json", f"{SCHEMA}_metrics")
    _need(authority["status"] == STATUS_PREDECESSOR and metrics["status"] == STATUS_METRICS, "terminal input status drift")
    manifest, manifest_sha = _load_json(root / "evaluation/prediction_cache.json", f"{SCHEMA}_prediction_cache")
    _need(manifest_sha == metrics["prediction_cache_manifest_sha256"], "prediction manifest SHA drift")
    cache_path = root / "evaluation/prediction_cache.npz"
    _need(verify_sidecar(cache_path) == metrics["prediction_cache_sha256"] == manifest["arrays_file_sha256"], "prediction cache SHA drift")
    with np.load(cache_path, allow_pickle=False) as values:
        arrays = {name: np.asarray(values[name]) for name in values.files}
    recomputed = {arm: {} for arm in ARMS}
    for index, (_session, key) in enumerate(HELDOUT_SESSION_TO_FALCON_KEY):
        for arm in ARMS:
            prefix = f"{arm}_{index}"
            prediction = arrays[f"{prefix}_prediction"]
            target = arrays[f"{prefix}_target"]
            score_mask = np.asarray(arrays[f"{prefix}_legacy_score_mask"], dtype=bool)
            _need(array_sha256(prediction) == manifest["array_sha256"][f"{prefix}_prediction"], "cached prediction SHA drift")
            _need(np.array_equal(target, arrays[f"t0_{index}_target"]), "cached target pairing drift")
            _need(np.array_equal(score_mask, arrays[f"t0_{index}_legacy_score_mask"]), "cached score-mask pairing drift")
            recomputed[arm][key] = variance_weighted_r2(target[score_mask], prediction[score_mask])
            _need(math.isclose(recomputed[arm][key], metrics["per_session"][key][arm], rel_tol=0.0, abs_tol=1e-12), "metric recomputation drift")
    means = {arm: float(np.mean(list(recomputed[arm].values()), dtype=np.float64)) for arm in ARMS}
    stds = {arm: float(np.std(list(recomputed[arm].values()), ddof=0, dtype=np.float64)) for arm in ARMS}
    delta = means["c1"] - means["t0"]
    _need(math.isclose(delta, metrics["equal_session_mean_delta_c1_minus_t0"], rel_tol=0.0, abs_tol=1e-12), "aggregate delta drift")
    body = {
        "schema": SCHEMA,
        "status": STATUS_TERMINAL,
        "finished_at_utc": utc_now(),
        "attempt_sha256": attempt_sha,
        "git_head": attempt["git_head"],
        "predecessor_authority_sha256": authority_sha,
        "metrics_sha256": metrics_sha,
        "label": LABEL,
        "per_session": metrics["per_session"],
        "equal_session_mean_r2": means,
        "population_std_r2": stds,
        "equal_session_mean_delta_c1_minus_t0": delta,
        "positive_session_count": metrics["positive_session_count"],
        "calibration_trials_reused_for_scoring": True,
        "independent_postcalibration_stream": False,
        "official_heldout_r2": False,
        "training": False,
        "optimizer_steps": 0,
        "backward_steps": 0,
        "model_updates": 0,
        "evalai_submissions": 0,
    }
    terminal_sha = publish_json(root / "terminal.json", body)
    lines = [
        "# H1 M3 Original-SPINT-Style Held-Out-Calib Diagnostic", "",
        f"- Status: `{STATUS_TERMINAL}`",
        f"- Label: `{LABEL}`",
        f"- T0 mean/std: `{means['t0']:.9f}` / `{stds['t0']:.9f}`",
        f"- C1 mean/std: `{means['c1']:.9f}` / `{stds['c1']:.9f}`",
        f"- Mean delta C1-T0: `{delta:+.9f}`",
        f"- Positive sessions: `{metrics['positive_session_count']}/14`",
        "- Calibration trials are reused for scoring: `true`.",
        "- Official held-out R² / EvalAI submissions / training: `false` / `0` / `false`.", "",
        "| FALCON session | T0 R² | C1 R² | Delta |", "|---|---:|---:|---:|",
    ]
    for key, row in metrics["per_session"].items():
        lines.append(f"| {key} | {row['t0']:.9f} | {row['c1']:.9f} | {row['delta_c1_minus_t0']:+.9f} |")
    lines.extend(["", f"Terminal SHA-256: `{terminal_sha}`"])
    publish_text(root / "EXPERIMENT_RECORD.md", "\n".join(lines) + "\n")
    return body


__all__ = (
    "LABEL", "SCHEMA", "STATUS_METRICS", "STATUS_PREDECESSOR", "STATUS_TERMINAL",
    "SpintStyleHeldoutError", "create_attempt", "evaluate", "heldout_paths",
    "legacy_spint_score_mask", "load_attempt", "validate_predecessor", "verify_terminal",
)
