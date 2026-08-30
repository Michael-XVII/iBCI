"""Matched fold0 evaluation for the H1 variable-activity successor."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tfpd_exploration.src.m1_h1_activity_headroom_v1.core import ARM_ORDER, ActivityHeadroomError
from tfpd_exploration.src.m1_h1_activity_headroom_v1.h1 import (
    ACCEPTED_QUERY_SHA256,
    BATCH_SIZE,
    CONFIG_RELATIVE,
    CONFIG_SHA256,
    DATA_RELATIVE,
    EB_RECEIPT_RELATIVE,
    OUTPUTS,
    RAW_RECEIPT_RELATIVE,
    SOURCE_AUTHORITY_RECEIPT_SHA256,
    SOURCE_AUTHORITY_RELATIVE,
    SUPPORT_TRIALS,
    TERMINAL_RELATIVE,
    TERMINAL_SHA256,
    _bootstrap,
    _evaluate_arm,
    _evaluate_direct_static,
    _output_trial_positions,
)


BASELINE_RESULT_RELATIVE = "tfpd_exploration/results/m1_h1_activity_headroom_v1/h1_fold0_hc.json"
BASELINE_RESULT_SHA256 = "5ec15848efffd3d0d7d1f6d0cbc077c99dcbdc32787965de3ba36472b50990a9"
TRAINING_RECEIPT_RELATIVE = "tfpd_exploration/results/h1_variable_activity_exposure_v1/full.json"
TRAINING_RECEIPT_SHA256 = "5cd23406dff6781e0105b60ccc778f349d0e494ce7f3a9a7f791828df381059f"
SUCCESSOR_CHECKPOINT_RELATIVE = "tfpd_exploration/results/h1_variable_activity_exposure_v1/checkpoint.pt"
SUCCESSOR_CHECKPOINT_SHA256 = "69f642f6aeba78a2c136316b78338dec4d8f31968a6071d860c1dc5935c6e77a"
SUCCESSOR_STATE_SHA256 = "fe6746208b54fd7dc03477a4154709f1f16930c9942141ab2d29a3110be6d60d"


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ActivityHeadroomError(message)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_successor_model(root: Path, device: str) -> tuple[Any, str]:
    import torch
    from scripts.h1_carrierid_evaluate import (
        _instantiate,
        _load_carrierid_checkpoint,
        _validate_carrierid_config,
    )
    from src.h1_m4_eb_normalized_v2_contract import state_hash
    from .plan import SEALED_CHECKPOINT_RELATIVE

    config_path = root / CONFIG_RELATIVE
    sealed = root / SEALED_CHECKPOINT_RELATIVE
    successor = root / SUCCESSOR_CHECKPOINT_RELATIVE
    _need(
        _sha_file(config_path) == CONFIG_SHA256
        and successor.is_file() and not successor.is_symlink()
        and _sha_file(successor) == SUCCESSOR_CHECKPOINT_SHA256,
        "H1 successor checkpoint/config binding drift",
    )
    config = _validate_carrierid_config(config_path, "full")
    sealed_payload, _metadata = _load_carrierid_checkpoint(sealed, config_path, "full")
    model = _instantiate(config, sealed_payload, torch.device(device))
    payload = torch.load(successor, map_location="cpu", weights_only=True)
    _need(
        payload.get("schema") == "h1_variable_activity_exposure_checkpoint_v1"
        and payload.get("trained_state_sha256") == SUCCESSOR_STATE_SHA256
        and payload.get("steps") == 4555
        and payload.get("epochs_completed") == 5
        and isinstance(payload.get("state_dict"), dict),
        "H1 successor checkpoint schema/training binding drift",
    )
    observed = model.load_state_dict(payload["state_dict"], strict=True)
    _need(not observed.missing_keys and not observed.unexpected_keys, "H1 successor strict state load drift")
    model = model.to(device).eval()
    state = state_hash(model.state_dict())
    _need(state == SUCCESSOR_STATE_SHA256, "H1 successor state digest drift")
    return model, state


def run_evaluation(root: Path, *, device: str) -> dict[str, Any]:
    import torch

    root = Path(root).resolve()
    _bootstrap(root)
    from src.data.h1_m4_eb_normalized_v2 import H1M4EBNormalizedV2StrictTargetDataset, fit_source_normalizer_from_cache
    from src.data.h1_m4_eb_pilot import (
        H1_M4_FOLD0_TARGET,
        interpolate_trial_identity,
        load_immutable_source_authority,
        load_source_records,
        load_target_records,
        validate_target_receipt_binding,
    )
    from src.h1_m4_eb_normalized_v2_contract import state_hash

    _need(device.startswith("cuda") and torch.cuda.is_available(), "H1 successor evaluation requires CUDA")
    terminal = root / TERMINAL_RELATIVE
    baseline_path = root / BASELINE_RESULT_RELATIVE
    training_path = root / TRAINING_RECEIPT_RELATIVE
    _need(
        _sha_file(terminal) == TERMINAL_SHA256
        and _sha_file(baseline_path) == BASELINE_RESULT_SHA256
        and _sha_file(training_path) == TRAINING_RECEIPT_SHA256,
        "H1 successor predecessor result/training binding drift",
    )
    training = json.loads(training_path.read_text(encoding="utf-8"))
    _need(training.get("status") == "FULL_TRAINING_COMPLETE"
          and training.get("source", {}).get("target_sessions_opened") == [],
          "H1 successor training receipt is not accepted source-only evidence")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    _need(baseline.get("status") == "COMPLETE_FROZEN_WEIGHT_ACTIVITY_HEADROOM",
          "H1 sealed activity-headroom predecessor is not accepted")
    source_records = load_source_records(root / DATA_RELATIVE)
    plan, _plan_manifest, carrier_cache, authority = load_immutable_source_authority(
        source_records,
        root / SOURCE_AUTHORITY_RELATIVE,
        SOURCE_AUTHORITY_RECEIPT_SHA256,
    )
    normalizer, _raw = fit_source_normalizer_from_cache(carrier_cache)
    records = load_target_records(root / DATA_RELATIVE)
    validate_target_receipt_binding(records, plan, root / RAW_RECEIPT_RELATIVE, root / EB_RECEIPT_RELATIVE)
    dataset = H1M4EBNormalizedV2StrictTargetDataset(records, plan, normalizer, "full")
    _need(dataset.window_indices_sha256 == ACCEPTED_QUERY_SHA256 and len(dataset) == 8965,
          "H1 successor target surface drift")
    model, state_before = _load_successor_model(root, device)
    trial_activity_by_session = {
        session: np.ascontiguousarray(np.stack([
            interpolate_trial_identity(records[session], value)
            for value in records[session].trial_values
        ]), dtype=np.float32)
        for session in H1_M4_FOLD0_TARGET
    }
    for session in H1_M4_FOLD0_TARGET:
        _need(np.array_equal(
            trial_activity_by_session[session][:SUPPORT_TRIALS],
            np.asarray(dataset.support[session].identity, dtype=np.float32),
        ), "H1 successor support identity reconstruction drift")
    output_trials = _output_trial_positions(dataset)
    cached = [
        _evaluate_arm(
            model=model,
            dataset=dataset,
            trial_activity_by_session=trial_activity_by_session,
            output_trials=output_trials,
            arm=arm,
            device=device,
        )
        for arm in ARM_ORDER
    ]
    direct = _evaluate_direct_static(model, dataset, device=device)
    max_abs = float(np.max(np.abs(
        direct["_prediction"].astype(np.float64) - cached[0]["_prediction"].astype(np.float64)
    )))
    r2_abs = abs(float(direct["pooled_r2"]) - float(cached[0]["pooled_r2"]))
    _need(max_abs <= 2.0e-6 and r2_abs <= 2.0e-7, "H1 successor cached-identity parity failed")
    direct["cached_identity_parity"] = {
        "max_abs_prediction": max_abs,
        "abs_r2": r2_abs,
        "tolerance_prediction": 2.0e-6,
        "tolerance_r2": 2.0e-7,
        "pass": True,
    }
    results = [direct, *cached[1:]]
    baseline_by_arm = {row["arm"]: row for row in baseline["results"]}
    deltas = []
    for row in results:
        predecessor = baseline_by_arm[row["arm"]]
        _need(row["target_sha256"] == predecessor["target_sha256"], "H1 successor target digest mismatch")
        deltas.append({
            "arm": row["arm"],
            "equal_recording_delta": float(row["equal_recording_mean_r2"] - predecessor["equal_recording_mean_r2"]),
            "pooled_delta": float(row["pooled_r2"] - predecessor["pooled_r2"]),
            "per_recording_delta": {
                session: float(value["r2"] - predecessor["per_recording"][session]["r2"])
                for session, value in row["per_recording"].items()
            },
        })
        row.pop("_prediction", None)
        row.pop("_target", None)
    state_after = state_hash(model.state_dict())
    _need(state_before == state_after, "H1 successor state changed during evaluation")
    delta_by_arm = {row["arm"]: row for row in deltas}
    static_safe = delta_by_arm["STATIC_SUPPORT"]["equal_recording_delta"] >= -0.01
    growing_better = delta_by_arm["CAUSAL_GROWING_CAP30"]["equal_recording_delta"] > 0.0
    verdict = "PASS_VARIABLE_ACTIVITY_SUCCESSOR" if static_safe and growing_better else "STOP_VARIABLE_ACTIVITY_SUCCESSOR"
    return {
        "schema": "h1_variable_activity_exposure_matched_score_v1",
        "status": "COMPLETE_MATCHED_ACTIVITY_HEADROOM_SCORE",
        "verdict": verdict,
        "device": device,
        "surface": "fold0_strict_post_m4_support_two_recordings",
        "authority": {
            "baseline_result_relative": BASELINE_RESULT_RELATIVE,
            "baseline_result_sha256": BASELINE_RESULT_SHA256,
            "training_receipt_relative": TRAINING_RECEIPT_RELATIVE,
            "training_receipt_sha256": TRAINING_RECEIPT_SHA256,
            "successor_checkpoint_relative": SUCCESSOR_CHECKPOINT_RELATIVE,
            "successor_checkpoint_sha256": SUCCESSOR_CHECKPOINT_SHA256,
            "successor_state_sha256": SUCCESSOR_STATE_SHA256,
            "query_window_indices_sha256": ACCEPTED_QUERY_SHA256,
            "source_authority": authority,
        },
        "results": results,
        "paired_deltas_vs_sealed_hc": deltas,
        "decision_gates": {
            "static_equal_recording_delta_min": -0.01,
            "static_safe": static_safe,
            "growing_equal_recording_delta_must_be_positive": True,
            "growing_better": growing_better,
        },
        "model_state_before_sha256": state_before,
        "model_state_after_sha256": state_after,
        "model_state_immutable": True,
        "target_optimizer_backward_update": 0,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }


__all__ = ("run_evaluation",)
