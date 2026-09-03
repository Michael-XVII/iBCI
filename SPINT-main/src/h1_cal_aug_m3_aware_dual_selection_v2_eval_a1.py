"""Evaluation-only repair for H1 M3-aware dual-selection V2 A1."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.h1_cal_aug_all_source_heldout_v1 import index_heldout_calib
from src.data.h1_m4_eb_pilot import (
    EXPECTED_NEURONS, H1PilotRecord, _trial_blocks, array_sha256,
    fit_deployment_carrier, interpolate_trial_identity, session_date,
    session_from_path,
)
from src.h1_cal_aug_all_source_m3_deployment_v1_contract import (
    HELDIN_SESSION_TO_FALCON_KEY, HELDOUT_SESSION_TO_FALCON_KEY,
)
from src.h1_cal_aug_all_source_m3_deployment_v1_exec import (
    _load_source_materialization, _minival_paths,
)
from src.h1_cal_aug_m3_aware_dual_selection_v2_contract import (
    EPOCHS, GLOBAL_STEPS, V1_C1_CHECKPOINT_SHA256, V1_DROPOUT_SHA256,
    V1_INITIAL_STATE_SHA256, select_epoch,
)
from src.h1_cal_aug_m3_aware_dual_selection_v2_exec import (
    STATUS_INTEGRITY, _evaluate_one, _load_json as _v2_load_json,
    _package_for_checkpoint,
)
from src.h1_hc_date_lodo_regen_v1 import _publish_bytes, publish_json, publish_text, verify_sidecar
from src.h1_m4_cce_contract import NORMALIZER_FLOOR, canonical_sha256, sha256_file


SCHEMA = "h1_cal_aug_m3_aware_dual_selection_v2_eval_a1"
TRAINING_SOURCE_SHA256 = "111a0bf42fd266cbf2604fbc45e9dfd217e65ea62d77f594fbdeea502c3ecac7"
TRAINING_TERMINAL_SHA256 = "4441daeeeecb3bb0e3b27f7611e54a8cfd8400d9de52da9c80103b76fe2cd713"
TRAINING_INTEGRITY_SHA256 = "b893c5c4013e81932d5c537a6aca7d39ab40a96241576390ec4403470fab285d"
TRAINING_FAILURE_SHA256 = "ac98c7c221de2ad1858009ce85b6432de49cb0245534502e8576ecbc41415761"
STATUS_PREDECESSOR = "PASS_V2_A1_FROZEN_TRAINING_PREDECESSOR"
STATUS_CALIBRATION = "PASS_V2_EVAL_A1_M3_CALIBRATION_AUTHORITY"
STATUS_SURFACE = "PASS_V2_EVAL_A1_SURFACE_COMPLETE"
STATUS_SELECTION = "PASS_V2_EVAL_A1_DUAL_SELECTION"
STATUS_TERMINAL = "COMPLETE_H1_CAL_AUG_M3_AWARE_DUAL_SELECTION_V2_EVAL_A1_NO_SUBMISSION"


class EvaluationRepairError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationRepairError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path, schema: str | None = None) -> tuple[dict[str, Any], str]:
    digest = verify_sidecar(path)
    body = json.loads(path.read_text(encoding="utf-8"))
    if schema is not None:
        _need(body.get("schema") == schema, f"schema drift: {path}")
    return body, digest


def ordered_m3_trials(trial_num: np.ndarray, eval_mask: np.ndarray) -> tuple[float, ...]:
    """Return chronological legal trials while requiring only the frozen M3 budget."""
    labels = np.asarray(trial_num, np.float64).reshape(-1)
    mask = np.asarray(eval_mask, bool).reshape(-1)
    _need(labels.shape == mask.shape, "TrialNum/eval-mask length mismatch")
    ordered = labels[mask & np.isfinite(labels)]
    _need(ordered.size > 0 and not np.any(np.diff(ordered) < 0.0), "TrialNum is empty or nonchronological")
    values: list[float] = []
    for value in ordered.tolist():
        if not values or float(value) != values[-1]:
            values.append(float(value))
    _need(len(values) >= 3, "M3 calibration requires at least three legal TrialNum trials")
    return tuple(values)


def load_heldout_m3_record(path: Path) -> H1PilotRecord:
    """Load a held-out-calib record without applying the incompatible M4 query gate."""
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb
    from pynwb import NWBHDF5IO

    resolved = path.resolve()
    _need("sub-HumanPitt-held-out-calib" in str(resolved), "M3 held-out loader scope drift")
    neural, velocity, trial_change, eval_mask = load_nwb(resolved, FalconTask.h1)
    with NWBHDF5IO(str(resolved), "r", load_namespaces=True) as handle:
        nwb = handle.read()
        _need("TrialNum" in nwb.acquisition, "held-out TrialNum missing")
        trial_num = np.asarray(nwb.acquisition["TrialNum"].data[:], np.float64)
    neural64 = np.asarray(neural, np.float64)
    velocity64 = np.asarray(velocity, np.float64)
    spikes = neural64.astype(np.float32)
    targets = velocity64.astype(np.float32)
    changes = np.asarray(trial_change, bool).reshape(-1)
    mask = np.asarray(eval_mask, bool).reshape(-1)
    _need(spikes.ndim == 2 and spikes.shape[1] == EXPECTED_NEURONS, "held-out neural shape drift")
    _need(targets.shape == (len(spikes), 7), "held-out target shape drift")
    _need(len(changes) == len(mask) == len(trial_num) == len(spikes), "held-out alignment drift")
    values = ordered_m3_trials(trial_num, mask)
    trials = tuple(_trial_blocks(value, neural64, velocity64, mask, trial_num) for value in values)
    name = session_from_path(resolved)
    return H1PilotRecord(name, session_date(name), resolved, sha256_file(resolved), spikes, targets, changes, mask, trial_num, values, trials)


def create_attempt(result_root: Path, training_root: Path, closure: Mapping[str, str], head: str) -> dict[str, Any]:
    root = result_root.resolve()
    _need(not root.exists(), f"evaluation repair result root already exists: {root}")
    body = {
        "schema": SCHEMA, "status": "ATTEMPT_EVALUATION_ONLY_BEFORE_DATA_AND_CUDA",
        "created_at_utc": utc_now(), "git_head": head, "closure": dict(closure),
        "code_closure_sha256": canonical_sha256(dict(closure)), "training_root": str(training_root.resolve()),
        "training": False, "checkpoint_selection_during_training": False,
        "optimizer_steps": 0, "backward_steps": 0, "model_updates": 0,
        "docker_builds": 0, "evalai_submissions": 0,
    }
    publish_json(root / "attempt.json", body)
    return body


def load_attempt(result_root: Path) -> dict[str, Any]:
    body, _ = _load_json(result_root.resolve() / "attempt.json", SCHEMA)
    _need(body["status"] == "ATTEMPT_EVALUATION_ONLY_BEFORE_DATA_AND_CUDA", "evaluation attempt drift")
    _need(body["training"] is False and body["optimizer_steps"] == body["backward_steps"] == body["model_updates"] == 0, "evaluation-only boundary drift")
    return body


def validate_training_predecessor(training_root: Path, v1_root: Path, result_root: Path) -> dict[str, Any]:
    load_attempt(result_root)
    root = training_root.resolve()
    source, source_sha = _load_json(root / "source_authority/authority.json")
    terminal, terminal_sha = _load_json(root / "training/c2/terminal.json")
    integrity, integrity_sha = _load_json(root / "training/integrity.json")
    failure, failure_sha = _load_json(root / "offline_dual_selection_failure.json")
    _need(source_sha == TRAINING_SOURCE_SHA256 and terminal_sha == TRAINING_TERMINAL_SHA256, "A1 source/training terminal SHA drift")
    _need(integrity_sha == TRAINING_INTEGRITY_SHA256 and failure_sha == TRAINING_FAILURE_SHA256, "A1 integrity/failure SHA drift")
    _need(integrity["status"] == STATUS_INTEGRITY and integrity["all_epoch_checkpoints_verified"], "A1 training integrity not passed")
    _need(failure["phase"] == "offline_dual_selection" and failure["error_type"] == "PilotDataError", "A1 failure lineage drift")
    _need(terminal["global_step"] == terminal["dropout_probability_count"] == GLOBAL_STEPS, "A1 step count drift")
    _need(terminal["initial_state_sha256"] == V1_INITIAL_STATE_SHA256 and terminal["dropout_probability_sha256"] == V1_DROPOUT_SHA256, "A1 initialization/dropout drift")
    _need(len(terminal["checkpoints"]) == EPOCHS, "A1 checkpoint roster incomplete")
    checkpoint_rows = []
    for epoch, row in enumerate(terminal["checkpoints"]):
        _need(row["epoch_zero_based"] == epoch, "A1 checkpoint epoch order drift")
        path = root / row["relative"]
        _need(verify_sidecar(path) == row["sha256"], f"A1 epoch {epoch} checkpoint SHA drift")
        checkpoint_rows.append({"epoch_zero_based": epoch, "relative": row["relative"], "sha256": row["sha256"], "state_sha256": row["state_sha256"]})
    v1_checkpoint = v1_root.resolve() / "training/c1/epoch_049.ckpt"
    _need(verify_sidecar(v1_checkpoint) == V1_C1_CHECKPOINT_SHA256, "V1 C1 baseline checkpoint drift")
    body = {
        "schema": f"{SCHEMA}_predecessor_authority", "status": STATUS_PREDECESSOR,
        "training_source_sha256": source_sha, "training_terminal_sha256": terminal_sha,
        "training_integrity_sha256": integrity_sha, "prior_failure_sha256": failure_sha,
        "v1_c1_checkpoint_sha256": V1_C1_CHECKPOINT_SHA256, "c2_checkpoints": checkpoint_rows,
        "checkpoint_count": EPOCHS, "training_reused": True, "new_training": False,
        "optimizer_steps": 0, "backward_steps": 0, "model_updates": 0,
    }
    publish_json(result_root.resolve() / "predecessor_authority.json", body)
    return body


def prepare_calibration_payloads(data_root: Path, v1_root: Path, result_root: Path) -> dict[str, Any]:
    predecessor, predecessor_sha = _load_json(result_root.resolve() / "predecessor_authority.json", f"{SCHEMA}_predecessor_authority")
    _need(predecessor["status"] == STATUS_PREDECESSOR, "predecessor authority must precede calibration")
    _authority, source_sha, _schedule, plan, dataset, _order, _m7, _prefixes = _load_source_materialization(data_root, v1_root)
    normalizer = _v2_load_json(v1_root.resolve() / "source_authority/normalizer.json")[0]
    denominator = max(float(normalizer["s_src"]), NORMALIZER_FLOOR)
    heldout_paths = index_heldout_calib(data_root)
    payloads: dict[str, dict[str, Any]] = {}
    rows = []
    for session, key in HELDIN_SESSION_TO_FALCON_KEY + HELDOUT_SESSION_TO_FALCON_KEY:
        if session in dataset.records:
            record = dataset.records[session]
            scope = "held-in-calib"
        else:
            record = load_heldout_m3_record(heldout_paths[session])
            scope = "held-out-calib development/model-selection"
        support = tuple(float(value) for value in record.trial_values[:3])
        _need(len(support) == 3, f"{session}: earliest M3 unavailable")
        identity = np.ascontiguousarray(np.stack([interpolate_trial_identity(record, value) for value in support]), np.float32)
        fitted = fit_deployment_carrier(record, plan, support)
        carrier = np.ascontiguousarray(np.asarray(fitted["carrier"], np.float64) / denominator, np.float32)
        _need(identity.shape == (3, 1024, 176) and carrier.shape == (176, 4), f"{session}: M3 payload shape drift")
        payloads[key] = {"identity": identity, "carrier": carrier, "calibration_trials": list(support), "session": session}
        rows.append({"session": session, "falcon_key": key, "scope": scope, "legal_trial_count": len(record.trial_values),
                     "calibration_trials": list(support), "identity_sha256": array_sha256(identity), "carrier_sha256": array_sha256(carrier),
                     "nwb_sha256": record.input_sha256})
    buffer = io.BytesIO()
    import torch
    torch.save(payloads, buffer)
    path = result_root.resolve() / "calibration/payloads.pt"
    payload_sha = _publish_bytes(path, buffer.getvalue())
    body = {
        "schema": f"{SCHEMA}_calibration_authority", "status": STATUS_CALIBRATION,
        "predecessor_authority_sha256": predecessor_sha, "v1_source_authority_sha256": source_sha,
        "v1_s_src": float(normalizer["s_src"]), "sessions": rows, "payload_relative": "calibration/payloads.pt",
        "payload_sha256": payload_sha, "heldin_calib_opened": 13, "heldout_calib_opened": 14,
        "m4_support_query_gate_used": False, "m3_minimum_legal_trials": 3,
        "optimizer_steps": 0, "backward_steps": 0, "model_updates": 0,
    }
    publish_json(result_root.resolve() / "calibration/authority.json", body)
    return body


def _load_payloads(result_root: Path) -> dict[str, Any]:
    import torch
    authority, _ = _load_json(result_root.resolve() / "calibration/authority.json", f"{SCHEMA}_calibration_authority")
    path = result_root.resolve() / authority["payload_relative"]
    _need(verify_sidecar(path) == authority["payload_sha256"], "M3 calibration payload SHA drift")
    payloads = torch.load(path, map_location="cpu", weights_only=False)
    _need(tuple(payloads) == tuple(key for _session, key in HELDIN_SESSION_TO_FALCON_KEY + HELDOUT_SESSION_TO_FALCON_KEY), "M3 payload roster/order drift")
    return payloads


def run_surface(surface: str, data_root: Path, training_root: Path, v1_root: Path, result_root: Path, physical_gpu: int) -> dict[str, Any]:
    _need(surface in {"hi", "ho"}, "unknown selection surface")
    directory = result_root.resolve() / "evaluation" / surface
    _need(not directory.exists(), f"surface already exists: {surface}")
    publish_json(directory / "attempt.json", {"schema": SCHEMA, "status": "SURFACE_ATTEMPT_BEFORE_CUDA_AND_TARGET", "surface": surface,
                                              "physical_gpu": physical_gpu, "training": False, "optimizer_steps": 0, "backward_steps": 0, "model_updates": 0})
    payloads = _load_payloads(result_root)
    predecessor = _load_json(result_root.resolve() / "predecessor_authority.json", f"{SCHEMA}_predecessor_authority")[0]
    if surface == "hi":
        paths = _minival_paths(data_root)
        mapping = HELDIN_SESSION_TO_FALCON_KEY
        metric_key = "val_hi_m3_official/r2_mean"
        role = "held-in independent minival selection"
    else:
        indexed = index_heldout_calib(data_root)
        paths = tuple(indexed[session] for session, _key in HELDOUT_SESSION_TO_FALCON_KEY)
        mapping = HELDOUT_SESSION_TO_FALCON_KEY
        metric_key = "val_ho_m3_grouped/r2_mean"
        role = "development/model-selection; not untouched held-out generalization"
    candidates = [("v1_c1_e49", 49, v1_root.resolve() / "training/c1/epoch_049.ckpt", V1_C1_CHECKPOINT_SHA256)]
    candidates.extend(("c2", row["epoch_zero_based"], training_root.resolve() / row["relative"], row["sha256"]) for row in predecessor["c2_checkpoints"])
    baseline = None
    curve = []
    source_sha = predecessor["training_source_sha256"]
    for kind, epoch, checkpoint_path, checkpoint_sha in candidates:
        package = _package_for_checkpoint(checkpoint_path, checkpoint_sha, source_sha, payloads, kind)
        metrics, before, after = _evaluate_one(package, paths, mapping, "cuda:0")
        row = {"schema": f"{SCHEMA}_candidate", "surface": surface, "surface_role": role, "candidate": kind,
               "epoch_zero_based": epoch, "checkpoint_sha256": checkpoint_sha, metric_key: metrics["r2_mean"],
               "r2_std_population": metrics["r2_std_population"], "worst_session_r2": metrics["worst_session_r2"],
               "per_session_r2": metrics["per_session_r2"], "per_recording_r2": metrics["per_recording_r2"],
               "val_ho_m3_spint14/r2_mean": metrics["spint_recording_mean_r2"] if surface == "ho" else None,
               "model_state_before_sha256": before, "model_state_after_sha256": after,
               "optimizer_steps": 0, "backward_steps": 0, "model_updates": 0}
        if kind == "c2":
            curve.append({"epoch_zero_based": epoch, metric_key: metrics["r2_mean"], "worst_session_r2": metrics["worst_session_r2"],
                          "session_std_population": metrics["r2_std_population"], "checkpoint_sha256": checkpoint_sha,
                          "per_session_r2": metrics["per_session_r2"], "per_recording_r2": metrics["per_recording_r2"],
                          "val_ho_m3_spint14/r2_mean": metrics["spint_recording_mean_r2"] if surface == "ho" else None})
            publish_json(directory / f"epoch_{epoch:03d}.json", row)
        else:
            baseline = row
            publish_json(directory / "v1_c1_e49.json", row)
        print(f"EVAL_END surface={surface} candidate={kind} epoch={epoch:03d} r2={metrics['r2_mean']:.9g}", flush=True)
    _need(baseline is not None and len(curve) == EPOCHS, f"{surface}: incomplete curve")
    body = {"schema": f"{SCHEMA}_surface", "status": STATUS_SURFACE, "surface": surface, "surface_role": role,
            "metric_key": metric_key, "physical_gpu": physical_gpu, "baseline": baseline, "curve": curve,
            "candidate_count": 51, "training": False, "optimizer_steps": 0, "backward_steps": 0, "model_updates": 0}
    publish_json(directory / "curve.json", body)
    return body


def combine_and_verify(result_root: Path) -> dict[str, Any]:
    root = result_root.resolve()
    hi, hi_sha = _load_json(root / "evaluation/hi/curve.json", f"{SCHEMA}_surface")
    ho, ho_sha = _load_json(root / "evaluation/ho/curve.json", f"{SCHEMA}_surface")
    _need(hi["status"] == ho["status"] == STATUS_SURFACE, "surface terminal incomplete")
    selected_hi = dict(select_epoch(hi["curve"], "val_hi_m3_official/r2_mean"))
    selected_ho = dict(select_epoch(ho["curve"], "val_ho_m3_grouped/r2_mean"))
    hi_receipt = {"schema": f"{SCHEMA}_selection", "status": "SEALED_C2_HI_SELECTION", "surface": "HI-M3", "selected": selected_hi,
                  "tie_break": ["higher mean", "higher worst-session", "lower population std", "earlier epoch"], "retraining": False}
    ho_receipt = {"schema": f"{SCHEMA}_selection", "status": "SEALED_C2_HO_SELECTION", "surface": "HO-M3 development/model-selection",
                  "selected": selected_ho, "selection_metric": "val_ho_m3_grouped/r2_mean", "secondary_not_used": "val_ho_m3_spint14/r2_mean",
                  "tie_break": ["higher mean", "higher worst-session", "lower population std", "earlier epoch"], "retraining": False}
    hi_selection_sha = publish_json(root / "selection/c2_hi.json", hi_receipt)
    ho_selection_sha = publish_json(root / "selection/c2_ho.json", ho_receipt)
    x = np.asarray([row["val_hi_m3_official/r2_mean"] for row in hi["curve"]], np.float64)
    y = np.asarray([row["val_ho_m3_grouped/r2_mean"] for row in ho["curve"]], np.float64)
    from scipy.stats import pearsonr, spearmanr
    c2_hi_e49, c2_ho_e49 = hi["curve"][49], ho["curve"][49]
    body = {"schema": f"{SCHEMA}_dual_selection", "status": STATUS_SELECTION,
            "hi_curve_sha256": hi_sha, "ho_curve_sha256": ho_sha,
            "v1_c1_e49": {"hi": hi["baseline"], "ho": ho["baseline"]},
            "c2_e49": {"hi": c2_hi_e49, "ho": c2_ho_e49}, "c2_hi_selected": selected_hi, "c2_ho_selected": selected_ho,
            "curve_correlation": {"pearson": float(pearsonr(x, y).statistic), "spearman": float(spearmanr(x, y).statistic)},
            "deltas": {
                "c2_e49_minus_v1_c1_e49_hi": c2_hi_e49["val_hi_m3_official/r2_mean"] - hi["baseline"]["val_hi_m3_official/r2_mean"],
                "c2_e49_minus_v1_c1_e49_ho": c2_ho_e49["val_ho_m3_grouped/r2_mean"] - ho["baseline"]["val_ho_m3_grouped/r2_mean"],
                "c2_hi_minus_c2_e49": selected_hi["val_hi_m3_official/r2_mean"] - c2_hi_e49["val_hi_m3_official/r2_mean"],
                "c2_ho_minus_c2_e49": selected_ho["val_ho_m3_grouped/r2_mean"] - c2_ho_e49["val_ho_m3_grouped/r2_mean"],
                "c2_hi_minus_v1_c1_e49": selected_hi["val_hi_m3_official/r2_mean"] - hi["baseline"]["val_hi_m3_official/r2_mean"],
                "c2_ho_minus_v1_c1_e49": selected_ho["val_ho_m3_grouped/r2_mean"] - ho["baseline"]["val_ho_m3_grouped/r2_mean"],
            }, "selection_c2_hi_sha256": hi_selection_sha, "selection_c2_ho_sha256": ho_selection_sha,
            "optimizer_steps": 0, "backward_steps": 0, "model_updates": 0, "docker_builds": 0, "evalai_submissions": 0}
    dual_sha = publish_json(root / "dual_selection.json", body)
    publish_json(root / "target_access.json", {"schema": f"{SCHEMA}_target_access", "heldin_calib_opened_for_m3": 13,
                 "heldout_calib_opened_for_m3": 14, "heldin_minival_recordings_per_candidate": 13,
                 "heldout_calib_recordings_per_candidate": 14, "candidate_count_per_surface": 51,
                 "ho_role": "development/model-selection; not untouched held-out generalization",
                 "optimizer_steps": 0, "backward_steps": 0, "model_updates": 0})
    terminal = {"schema": SCHEMA, "status": STATUS_TERMINAL, "finished_at_utc": utc_now(), "dual_selection_sha256": dual_sha,
                "c2_hi_selection_sha256": hi_selection_sha, "c2_ho_selection_sha256": ho_selection_sha,
                "c2_hi_epoch": selected_hi["epoch_zero_based"], "c2_ho_epoch": selected_ho["epoch_zero_based"],
                "optimizer_steps": 0, "backward_steps": 0, "model_updates": 0, "docker_builds": 0,
                "evalai_submissions": 0, "automatic_successor": False,
                "stop_boundary": "local dual selection complete; no Docker or EvalAI"}
    terminal_sha = publish_json(root / "terminal.json", terminal)
    publish_text(root / "EXPERIMENT_RECORD.md", f"# H1 M3-Aware Dual-Selection V2 Evaluation A1\n\n- Status: `{STATUS_TERMINAL}`\n- C2-HI epoch: `{selected_hi['epoch_zero_based']}`\n- C2-HO epoch: `{selected_ho['epoch_zero_based']}`\n- HO is development/model-selection, not untouched held-out generalization.\n- New training/target updates/EvalAI: `0/0/0`.\n- Terminal SHA-256: `{terminal_sha}`\n")
    return terminal


def dry_plan() -> dict[str, Any]:
    return {"schema": SCHEMA, "status": "DRY_RUN_NO_WRITE_NO_DATA_NO_CUDA", "new_training": False,
            "frozen_c2_checkpoints": 50, "surfaces": {"gpu0": "HI-M3", "gpu1": "HO-M3"},
            "m3_loader_minimum_trials": 3, "m4_loader_used": False, "optimizer_steps": 0,
            "backward_steps": 0, "model_updates": 0, "docker_builds": 0, "evalai_submissions": 0}


__all__ = ("SCHEMA", "combine_and_verify", "create_attempt", "dry_plan", "load_attempt",
           "load_heldout_m3_record", "ordered_m3_trials", "prepare_calibration_payloads",
           "run_surface", "validate_training_predecessor")
