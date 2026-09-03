"""Additive packaging successor for H1 all-source M3 deployment V1."""
from __future__ import annotations

from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from src.data.h1_cal_aug_all_source_heldout_v1 import H1_HELDOUT_SESSIONS, index_heldout_calib
from src.data.h1_m4_eb_pilot import (
    EXPECTED_NEURONS,
    H1_HELDIN_SESSIONS,
    H1PilotRecord,
    _trial_blocks,
    array_sha256,
    fit_deployment_carrier,
    index_heldin_calib,
    interpolate_trial_identity,
    load_record,
    session_date,
    session_from_path,
)
from src.h1_cal_aug_all_source_m3_deployment_v1_contract import (
    ARMS,
    HELDIN_SESSION_TO_FALCON_KEY,
    HELDOUT_SESSION_TO_FALCON_KEY,
    PREDICTION_DIVISOR,
    WINDOW_SIZE,
)
from src.h1_cal_aug_all_source_m3_deployment_v1_exec import (
    STATUS_MINIVAL,
    STATUS_REHEARSAL,
    _load_json,
    run_local_minival,
    run_package_rehearsal,
)
from src.h1_hc_date_lodo_regen_v1 import RegenPlan, _publish_bytes, model_config, publish_json, publish_text, verify_sidecar
from src.h1_m4_cce_contract import NORMALIZER_FLOOR, canonical_sha256, sha256_file, state_hash


SCHEMA = "h1_cal_aug_all_source_m3_deployment_v1_package_a1"
BASE_SCHEMA = "h1_cal_aug_all_source_m3_deployment_v1"
STATUS_PACKAGES = "PASS_H1_ALL_SOURCE_M3_DEPLOYMENT_PACKAGES"
STATUS_TERMINAL = "COMPLETE_LOCAL_H1_ALL_SOURCE_M3_DEPLOYMENT_READY_NO_EVALAI_SUBMISSION"
PREDECESSOR_FAILURE_SHA256 = "71a6915f7f1273d2dd78b71b47c5456957acaf020f26de6fc2c33c0c2511576a"
PREDECESSOR_SOURCE_SHA256 = "8ea4bb1174c00ab713843cd7561562d43f81509eaaea6ea12ee80cd4eba95de7"
PREDECESSOR_PAIR_SHA256 = "b2a4fd570b152028e3e3ab99bbf1bbb11b6ed49d6907cc26772974d0ef4a7e9d"
TRAINING_TERMINAL_SHA256 = {
    "t0": "aa394599084ff789aa85613d792b5ad438188326694c73c33c9d7864d666ae1b",
    "c1": "873c431fdb3194787c9f46a24c3389db3a40365e7e13926300265dfbd9583082",
}
CHECKPOINT_SHA256 = {
    "t0": "6d4d14226b706951274982438b588527beb442200aad2f50f9d18b68e54a9648",
    "c1": "0f406a8e69fdb57cf6a5480149f04ab3500e7fad849d36db38042edbadb2cd06",
}


class PackageA1Error(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise PackageA1Error(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_attempt(result_root: Path, predecessor_root: Path, closure: Mapping[str, str], head: str) -> dict[str, Any]:
    root = result_root.resolve()
    _need(not root.exists(), f"successor result root is not fresh: {root}")
    body = {
        "schema": SCHEMA, "artifact": "attempt", "status": "ATTEMPT_BEFORE_CALIBRATION_DATA_AND_CUDA",
        "created_at_utc": utc_now(), "head": head, "closure": dict(closure),
        "code_closure_sha256": canonical_sha256(dict(closure)),
        "predecessor_root": str(predecessor_root.resolve()),
        "training": False, "optimizer_steps": 0, "backward_steps": 0, "model_updates": 0,
        "heldin_calib_recordings_opened": 0, "heldout_calib_recordings_opened": 0,
        "heldin_minival_recordings_opened": 0, "evalai_test_recordings_opened": 0,
        "cuda_initialized": False, "evalai_submissions": 0,
    }
    publish_json(root / "attempt.json", body)
    return body


def load_attempt(result_root: Path) -> dict[str, Any]:
    body, _ = _load_json(result_root.resolve() / "attempt.json", SCHEMA)
    _need(body.get("status") == "ATTEMPT_BEFORE_CALIBRATION_DATA_AND_CUDA", "A1 attempt drift")
    _need(body.get("training") is False and body.get("optimizer_steps") == body.get("backward_steps") == body.get("model_updates") == 0, "A1 attempt training drift")
    return body


def validate_predecessor(predecessor_root: Path, result_root: Path) -> dict[str, Any]:
    import torch
    root = predecessor_root.resolve()
    checks = (
        (root / "packages_failure.json", PREDECESSOR_FAILURE_SHA256),
        (root / "source_authority/authority.json", PREDECESSOR_SOURCE_SHA256),
        (root / "pair_integrity/paired_integrity.json", PREDECESSOR_PAIR_SHA256),
    )
    for path, digest in checks:
        _need(verify_sidecar(path) == digest, f"predecessor SHA drift: {path.name}")
    failure = json.loads((root / "packages_failure.json").read_text(encoding="utf-8"))
    _need(failure.get("phase") == "packages" and failure.get("status") == "FAIL_IMMUTABLE_NO_AUTOMATIC_RETRY", "predecessor is not the registered packaging failure")
    training = []
    for arm in ARMS:
        terminal_path = root / f"training/{arm}/terminal.json"
        _need(verify_sidecar(terminal_path) == TRAINING_TERMINAL_SHA256[arm], f"{arm} terminal SHA drift")
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        _need(terminal.get("epoch_zero_based") == 49 and terminal.get("global_step") == 206650, f"{arm} epoch/steps drift")
        checkpoint_path = root / terminal["checkpoint"]["relative"]
        _need(verify_sidecar(checkpoint_path) == terminal["checkpoint"]["sha256"] == CHECKPOINT_SHA256[arm], f"{arm} checkpoint SHA drift")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = checkpoint["metadata"]
        _need(metadata.get("terminal_state_sha256") == terminal["terminal_state_sha256"], f"{arm} state provenance drift")
        _need(state_hash(checkpoint["state_dict"]) == terminal["terminal_state_sha256"], f"{arm} state hash drift")
        _need(metadata.get("checkpoint_selection") is False and metadata.get("warm_start") is False, f"{arm} checkpoint selection drift")
        _need(metadata.get("target_optimizer_steps") == metadata.get("target_backward_steps") == metadata.get("target_model_updates") == 0, f"{arm} target update drift")
        training.append({"arm": arm, "terminal_sha256": TRAINING_TERMINAL_SHA256[arm], "checkpoint_sha256": CHECKPOINT_SHA256[arm], "terminal_state_sha256": terminal["terminal_state_sha256"], "global_step": terminal["global_step"], "training_elapsed_seconds": terminal["training_elapsed_seconds"]})
    body = {
        "schema": f"{SCHEMA}_predecessor_authority", "status": "PASS_PACKAGE_A1_PREDECESSOR",
        "failure_sha256": PREDECESSOR_FAILURE_SHA256, "source_authority_sha256": PREDECESSOR_SOURCE_SHA256,
        "pair_integrity_sha256": PREDECESSOR_PAIR_SHA256, "training": training,
        "retraining": False, "optimizer_steps": 0, "backward_steps": 0, "model_updates": 0,
        "heldin_calib_recordings_opened": 0, "heldout_calib_recordings_opened": 0,
        "heldin_minival_recordings_opened": 0, "evalai_test_recordings_opened": 0,
        "evalai_submissions": 0,
    }
    publish_json(result_root.resolve() / "predecessor_authority.json", body)
    return body


def ordered_m3_eval_trials(trial_num: np.ndarray, eval_mask: np.ndarray) -> tuple[float, ...]:
    labels = np.asarray(trial_num, dtype=np.float64).reshape(-1)
    mask = np.asarray(eval_mask, dtype=bool).reshape(-1)
    _need(labels.shape == mask.shape, "TrialNum/eval-mask length drift")
    ordered = labels[mask & np.isfinite(labels)]
    _need(ordered.size > 0 and not np.any(np.diff(ordered) < 0.0), "M3 TrialNum must be chronological")
    values = []
    for value in ordered.tolist():
        if not values or float(value) != values[-1]:
            values.append(float(value))
    _need(len(values) >= 3, "M3 calibration requires at least three legal trials")
    return tuple(values)


def load_heldout_m3_record(path: Path) -> H1PilotRecord:
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb
    from pynwb import NWBHDF5IO
    resolved = path.resolve()
    _need("sub-HumanPitt-held-out-calib" in str(resolved), "M3 held-out loader scope drift")
    neural, velocity, trial_change, eval_mask = load_nwb(resolved, FalconTask.h1)
    with NWBHDF5IO(str(resolved), "r", load_namespaces=True) as handle:
        nwb = handle.read()
        _need("TrialNum" in nwb.acquisition, "held-out TrialNum missing")
        trial_num = np.asarray(nwb.acquisition["TrialNum"].data[:], dtype=np.float64)
    spikes64 = np.asarray(neural, np.float64); targets64 = np.asarray(velocity, np.float64)
    spikes = spikes64.astype(np.float32); targets = targets64.astype(np.float32)
    changes = np.asarray(trial_change, bool).reshape(-1); mask = np.asarray(eval_mask, bool).reshape(-1)
    _need(spikes.ndim == 2 and spikes.shape[1] == EXPECTED_NEURONS and targets.shape == (len(spikes), 7), "held-out M3 record shape drift")
    _need(len(changes) == len(mask) == len(trial_num) == len(spikes), "held-out M3 record alignment drift")
    name = session_from_path(resolved)
    values = ordered_m3_eval_trials(trial_num, mask)
    _need(len(values) == 3, f"{name}: registered held-out calibration must contain exactly three legal trials")
    trials = tuple(_trial_blocks(value, spikes64, targets64, mask, trial_num) for value in values)
    return H1PilotRecord(name, session_date(name), resolved, sha256_file(resolved), spikes, targets, changes, mask, trial_num, values, trials)


def _load_plan_normalizer(predecessor_root: Path):
    root = predecessor_root.resolve() / "source_authority"
    authority, authority_sha = _load_json(root / "authority.json", f"{BASE_SCHEMA}_source_authority")
    _need(authority_sha == PREDECESSOR_SOURCE_SHA256, "source authority binding drift")
    plan_body, plan_sha = _load_json(root / "plan.json", f"{BASE_SCHEMA}_plan")
    _need(plan_sha == authority["plan_sha256"] and verify_sidecar(root / "plan.npz") == plan_body["arrays_file_sha256"], "frozen plan drift")
    with np.load(root / "plan.npz", allow_pickle=False) as arrays:
        plan = RegenPlan(
            "h1_all_source_13", tuple(plan_body["source_sessions"]), tuple(plan_body["source_input_sha256"]),
            np.asarray(arrays["mean"], np.float64), np.asarray(arrays["scale"], np.float64),
            np.asarray(arrays["pcs"], np.float64), int(arrays["q"].item()), float(arrays["lambda"].item()),
            np.asarray(arrays["U"], np.float64), np.asarray(arrays["mu"], np.float64), float(arrays["tau2"].item()),
            str(plan_body["selection_sha256"]), str(plan_body["transform_sha256"]),
        )
    normalizer, normalizer_sha = _load_json(root / "normalizer.json", f"{BASE_SCHEMA}_normalizer")
    _need(normalizer_sha == authority["normalizer_sha256"], "frozen normalizer drift")
    _need(plan.q == authority["selected_q"] and plan.ridge_lambda == authority["selected_lambda"], "frozen q/lambda drift")
    return plan, float(normalizer["s_src"]), authority, authority_sha


def build_packages(predecessor_root: Path, data_root: Path, result_root: Path) -> dict[str, Any]:
    import torch
    from third_party.falcon_challenge.h1_carrier_id_spint_decoder import PACKAGE_SCHEMA
    load_attempt(result_root)
    predecessor = validate_predecessor(predecessor_root, result_root)
    predecessor_sha = verify_sidecar(result_root.resolve() / "predecessor_authority.json")
    plan, normalizer, authority, authority_sha = _load_plan_normalizer(predecessor_root)
    heldin_paths = index_heldin_calib(data_root); heldout_paths = index_heldout_calib(data_root)
    denominator = max(normalizer, NORMALIZER_FLOOR)
    sessions = {}; manifest = []; heldin_bytes = 0; heldout_bytes = 0
    for session, key in HELDIN_SESSION_TO_FALCON_KEY + HELDOUT_SESSION_TO_FALCON_KEY:
        heldout = session in H1_HELDOUT_SESSIONS
        path = heldout_paths[session] if heldout else heldin_paths[session]
        record = load_heldout_m3_record(path) if heldout else load_record(path)
        support = tuple(float(value) for value in record.trial_values[:3])
        _need(len(support) == 3, f"{session}: incomplete earliest M3")
        identity = np.ascontiguousarray(np.stack([interpolate_trial_identity(record, value) for value in support]), np.float32)
        fitted = fit_deployment_carrier(record, plan, support)
        carrier = np.ascontiguousarray(np.asarray(fitted["carrier"], np.float64) / denominator, np.float32)
        _need(identity.shape == (3, 1024, 176) and carrier.shape == (176, 4), f"{session}: M3 payload shape drift")
        sessions[key] = {"identity": identity, "carrier": carrier, "calibration_trials": list(support), "session": session}
        size = int(path.stat().st_size); heldout_bytes += size if heldout else 0; heldin_bytes += 0 if heldout else size
        manifest.append({
            "session": session, "falcon_key": key, "scope": "held-out-calib" if heldout else "held-in-calib",
            "filename": path.name, "bytes": size, "nwb_sha256": record.input_sha256,
            "legal_trial_count": len(record.trial_values), "calibration_trials": list(support),
            "identity_sha256": array_sha256(identity), "carrier_sha256": array_sha256(carrier),
            "carrier_support_m": fitted["support_m"],
        })
    calibration = {
        "schema": f"{SCHEMA}_calibration_authority", "status": "PASS_PACKAGE_A1_M3_CALIBRATION",
        "predecessor_authority_sha256": predecessor_sha, "source_authority_sha256": authority_sha,
        "selected_q": plan.q, "selected_lambda": plan.ridge_lambda,
        "source_normalizer_sha256": authority["normalizer_sha256"], "sessions": manifest,
        "heldin_calib_recordings_opened": 13, "heldin_calib_bytes_read": heldin_bytes,
        "heldout_calib_recordings_opened": 14, "heldout_calib_bytes_read": heldout_bytes,
        "heldin_minival_recordings_opened": 0, "heldout_calibration_rows_scored": 0,
        "evalai_test_recordings_opened": 0, "optimizer_steps": 0, "backward_steps": 0,
        "model_updates": 0, "evalai_submissions": 0,
    }
    calibration_sha = publish_json(result_root.resolve() / "packages/calibration_authority.json", calibration)
    package_rows = []; predecessor_training = {row["arm"]: row for row in predecessor["training"]}
    for arm in ARMS:
        terminal = json.loads((predecessor_root.resolve() / f"training/{arm}/terminal.json").read_text(encoding="utf-8"))
        checkpoint_path = predecessor_root.resolve() / terminal["checkpoint"]["relative"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        _need(state_hash(checkpoint["state_dict"]) == predecessor_training[arm]["terminal_state_sha256"], f"{arm} package state drift")
        package = {
            "schema": PACKAGE_SCHEMA, "task": "h1", "arm": arm,
            "state_dict": checkpoint["state_dict"], "model_kwargs": model_config()["model_kwargs"],
            "model_state_sha256": predecessor_training[arm]["terminal_state_sha256"],
            "checkpoint_sha256": CHECKPOINT_SHA256[arm], "source_authority_sha256": authority_sha,
            "calibration_authority_sha256": calibration_sha, "window_size": WINDOW_SIZE,
            "prediction_divisor": PREDICTION_DIVISOR, "sessions": sessions,
            "optimizer_steps": 0, "backward_steps": 0, "model_updates": 0, "evalai_submissions": 0,
        }
        buffer = io.BytesIO(); torch.save(package, buffer)
        package_path = result_root.resolve() / f"packages/{arm}.pt"
        package_sha = _publish_bytes(package_path, buffer.getvalue())
        package_rows.append({"arm": arm, "relative": str(package_path.relative_to(result_root.resolve())), "sha256": package_sha, "checkpoint_sha256": CHECKPOINT_SHA256[arm], "model_state_sha256": predecessor_training[arm]["terminal_state_sha256"]})
    body = {
        "schema": f"{BASE_SCHEMA}_packages", "status": STATUS_PACKAGES,
        "package_successor_schema": SCHEMA, "predecessor_authority_sha256": predecessor_sha,
        "calibration_authority_sha256": calibration_sha, "source_authority_sha256": authority_sha,
        "packages": package_rows, "session_payloads": 27, "retraining": False,
        "optimizer_steps": 0, "backward_steps": 0, "model_updates": 0, "evalai_submissions": 0,
    }
    publish_json(result_root.resolve() / "packages/packages.json", body)
    return body


def verify_terminal(predecessor_root: Path, result_root: Path) -> dict[str, Any]:
    root = result_root.resolve(); attempt = load_attempt(root); attempt_sha = verify_sidecar(root / "attempt.json")
    predecessor, predecessor_sha = _load_json(root / "predecessor_authority.json", f"{SCHEMA}_predecessor_authority")
    _need(predecessor["status"] == "PASS_PACKAGE_A1_PREDECESSOR", "A1 predecessor authority drift")
    packages, packages_sha = _load_json(root / "packages/packages.json", f"{BASE_SCHEMA}_packages")
    _need(packages["status"] == STATUS_PACKAGES and len(packages["packages"]) == 2, "A1 package authority drift")
    for row in packages["packages"]:
        _need(verify_sidecar(root / row["relative"]) == row["sha256"], f"{row['arm']} package SHA drift")
    calibration, calibration_sha = _load_json(root / "packages/calibration_authority.json", f"{SCHEMA}_calibration_authority")
    _need(calibration["status"] == "PASS_PACKAGE_A1_M3_CALIBRATION" and len(calibration["sessions"]) == 27, "A1 calibration authority drift")
    _need(sum(row["scope"] == "held-out-calib" and row["legal_trial_count"] == 3 for row in calibration["sessions"]) == 14, "A1 held-out M3 roster drift")
    _need(calibration["heldout_calibration_rows_scored"] == 0, "held-out calibration rows scored")
    metrics, metrics_sha = _load_json(root / "minival/metrics.json", f"{BASE_SCHEMA}_minival_metrics")
    _need(metrics["status"] == STATUS_MINIVAL and metrics["label"] == "local held-in-minival deployment sanity R²", "A1 minival status/label drift")
    rehearsal, rehearsal_sha = _load_json(root / "packages/rehearsal.json", f"{BASE_SCHEMA}_package_rehearsal")
    _need(rehearsal["status"] == STATUS_REHEARSAL and rehearsal["package_reload_numerical_equivalence"] is True, "A1 package rehearsal drift")
    cache_manifest, manifest_sha = _load_json(root / "minival/prediction_cache.json", f"{BASE_SCHEMA}_minival_prediction_cache")
    _need(manifest_sha == metrics["prediction_cache_manifest_sha256"], "A1 prediction manifest drift")
    cache_path = root / "minival/prediction_cache.npz"
    _need(verify_sidecar(cache_path) == metrics["prediction_cache_sha256"] == cache_manifest["arrays_file_sha256"], "A1 prediction cache drift")
    with np.load(cache_path, allow_pickle=False) as values:
        arrays = {name: np.asarray(values[name]) for name in values.files}
    recomputed = {arm: {} for arm in ARMS}
    for index, (_session, key) in enumerate(HELDIN_SESSION_TO_FALCON_KEY):
        for arm in ARMS:
            prefix = f"{arm}_{index}"; prediction = arrays[f"{prefix}_prediction"]
            target = arrays[f"{prefix}_target"]; mask = np.asarray(arrays[f"{prefix}_score_mask"], bool)
            _need(np.array_equal(target, arrays[f"t0_{index}_target"]) and np.array_equal(mask, arrays[f"t0_{index}_score_mask"]), "A1 paired minival surface drift")
            recomputed[arm][key] = float(1.0 - np.square(target[mask].astype(np.float64) - prediction[mask].astype(np.float64)).sum(dtype=np.float64) / np.square(target[mask].astype(np.float64) - target[mask].astype(np.float64).mean(axis=0, keepdims=True), dtype=np.float64).sum(dtype=np.float64))
            _need(math.isclose(recomputed[arm][key], metrics["per_session"][key][arm], rel_tol=0.0, abs_tol=1e-12), "A1 minival metric recomputation drift")
    means = {arm: float(np.mean(list(recomputed[arm].values()), dtype=np.float64)) for arm in ARMS}
    stds = {arm: float(np.std(list(recomputed[arm].values()), ddof=0, dtype=np.float64)) for arm in ARMS}
    delta = means["c1"] - means["t0"]
    _need(math.isclose(delta, metrics["mean_delta_c1_minus_t0"], rel_tol=0.0, abs_tol=1e-12), "A1 minival aggregate drift")
    source = json.loads((predecessor_root.resolve() / "source_authority/authority.json").read_text(encoding="utf-8"))
    body = {
        "schema": SCHEMA, "status": STATUS_TERMINAL, "finished_at_utc": utc_now(),
        "experiment_attempt_sha256": attempt_sha, "code_closure_sha256": attempt["code_closure_sha256"],
        "predecessor_authority_sha256": predecessor_sha, "predecessor_failure_sha256": PREDECESSOR_FAILURE_SHA256,
        "source_authority_sha256": PREDECESSOR_SOURCE_SHA256, "selected_q": source["selected_q"],
        "selected_lambda": source["selected_lambda"], "normalizer_sha256": source["normalizer_sha256"],
        "pair_integrity_sha256": PREDECESSOR_PAIR_SHA256, "training": predecessor["training"],
        "calibration_authority_sha256": calibration_sha, "packages_sha256": packages_sha,
        "package_rehearsal_sha256": rehearsal_sha, "minival_metrics_sha256": metrics_sha,
        "minival_label": metrics["label"], "minival_per_session": metrics["per_session"],
        "minival_mean_r2": means, "minival_std_r2_population": stds,
        "minival_delta_c1_minus_t0": delta, "retraining": False,
        "ready_for_evalai_hidden_test_without_model_change": True,
        "evalai_submission_authorized": False, "evalai_submissions": 0,
        "evalai_test_recordings_opened": 0, "official_heldout_score_accessed": False,
        "target_optimizer_steps": 0, "target_backward_steps": 0, "target_model_updates": 0,
        "post_minival_selection": False, "post_minival_retraining": False,
        "claim": "local held-in-minival deployment sanity R² only; not held-out R²",
    }
    terminal_sha = publish_json(root / "terminal.json", body)
    lines = [
        "# H1 CAL-AUG All-Source M3 Deployment V1 — Package A1", "",
        f"- Status: `{STATUS_TERMINAL}`", f"- Selected q/lambda: `{source['selected_q']}` / `{source['selected_lambda']}`",
        f"- T0 mean/std: `{means['t0']:.9f}` / `{stds['t0']:.9f}`",
        f"- C1 mean/std: `{means['c1']:.9f}` / `{stds['c1']:.9f}`",
        f"- Mean delta C1-T0: `{delta:+.9f}`",
        "- Metric label: `local held-in-minival deployment sanity R²`.",
        "- Retraining, target updates, EvalAI submissions and official held-out accesses: `0`.", "",
        "| FALCON session | T0 R² | C1 R² | Delta |", "|---|---:|---:|---:|",
    ]
    for key, row in metrics["per_session"].items():
        lines.append(f"| {key} | {row['t0']:.9f} | {row['c1']:.9f} | {row['delta_c1_minus_t0']:+.9f} |")
    lines.extend(["", f"Terminal SHA-256: `{terminal_sha}`"])
    publish_text(root / "EXPERIMENT_RECORD.md", "\n".join(lines) + "\n")
    return body


__all__ = (
    "SCHEMA", "STATUS_TERMINAL", "PackageA1Error", "build_packages", "create_attempt",
    "load_attempt", "load_heldout_m3_record", "ordered_m3_eval_trials",
    "run_local_minival", "run_package_rehearsal", "validate_predecessor", "verify_terminal",
)
