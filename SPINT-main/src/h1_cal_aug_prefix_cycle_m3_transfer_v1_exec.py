"""Formal execution layer for the pre-registered H1 M3 transfer diagnostic."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from src.data.h1_m4_cce_date_lodo import target_sessions_for_date
from src.data.h1_m4_eb_pilot import array_sha256
from src.h1_cal_aug_prefix_cycle_eval_a1 import (
    collect_training_authority,
    _gpu_profile,
    _load_arm_model,
    _load_plan_normalizer,
    _target_records,
)
from src.h1_cal_aug_prefix_cycle_m3_transfer_v1 import (
    ARMS,
    EXPERIMENT4_A1_COMMIT,
    EXPERIMENT4_A1_TERMINAL_SHA256,
    M4_AUDIT_COMMIT,
    M4_AUDIT_TERMINAL_SHA256,
    M4_METADATA_TERMINAL_SHA256,
    SCHEMA,
    WINDOW_SIZE,
    aggregate_m3_results,
    m3_causal_surface,
    materialize_m3_calibration,
    scale_last_bin_prediction,
    score_m3,
)
from src.h1_cal_aug_prefix_cycle_v1 import BATCH_SIZE, validate_predecessors
from src.h1_hc_date_lodo_regen_v1 import publish_json, publish_npz, publish_text, verify_sidecar
from src.h1_m4_cce_contract import CONFIRMATORY_DATES, canonical_sha256, state_hash


STATUS_AUTHORITY = "PASS_H1_M3_SEALED_TEN_CHECKPOINT_AUTHORITY"
STATUS_CELL = "PASS_H1_CAL_AUG_PREFIX_CYCLE_M3_DATE_EVALUATED"
STATUS_COMPLETE = "COMPLETE_H1_CAL_AUG_PREFIX_CYCLE_M3_TRANSFER_V1"
VERDICT_SUPPORT = "SUPPORT_M3_PREFIX_EXTRAPOLATION"
VERDICT_STRONG = "STRONG_M3_PREFIX_EXTRAPOLATION"
VERDICT_NO_CLEAR = "NO_CLEAR_M3_PREFIX_EXTRAPOLATION"


class M3ExecutionError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise M3ExecutionError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path, schema: str | None = None) -> tuple[dict[str, Any], str]:
    digest = verify_sidecar(path)
    body = json.loads(path.read_text(encoding="utf-8"))
    if schema is not None:
        _need(body.get("schema") == schema, f"schema drift: {path}")
    return body, digest


def create_attempt(result_root: Path, closure: Mapping[str, str], head: str, cpu_gate_digest: str) -> dict[str, Any]:
    root = result_root.resolve()
    _need(not root.exists(), f"canonical M3 result root is not fresh: {root}")
    body = {
        "schema": SCHEMA,
        "artifact": "attempt",
        "status": "ATTEMPT_AFTER_CPU_GATE_BEFORE_AUTHORITY_CUDA_AND_TARGET",
        "created_at_utc": utc_now(),
        "head": head,
        "closure": dict(closure),
        "code_closure_sha256": canonical_sha256(dict(closure)),
        "cpu_gate_stdout_sha256": cpu_gate_digest,
        "experiment4_a1_commit": EXPERIMENT4_A1_COMMIT,
        "experiment4_a1_terminal_sha256": EXPERIMENT4_A1_TERMINAL_SHA256,
        "m4_audit_commit": M4_AUDIT_COMMIT,
        "m4_audit_terminal_sha256": M4_AUDIT_TERMINAL_SHA256,
        "m4_metadata_terminal_sha256": M4_METADATA_TERMINAL_SHA256,
        "outer_dates": list(CONFIRMATORY_DATES),
        "arms": list(ARMS),
        "cuda_initialized": False,
        "target_recordings_opened": 0,
        "target_bytes_read": 0,
        "heldout_calib_recordings_opened": 0,
        "optimizer_steps": 0,
        "backward_steps": 0,
        "parameter_updates": 0,
        "warm_starts": 0,
        "checkpoint_selections": 0,
    }
    publish_json(root / "attempt.json", body)
    return body


def load_attempt(result_root: Path) -> dict[str, Any]:
    body, _ = _load_json(result_root.resolve() / "attempt.json", SCHEMA)
    _need(body.get("status") == "ATTEMPT_AFTER_CPU_GATE_BEFORE_AUTHORITY_CUDA_AND_TARGET", "attempt status drift")
    _need(body.get("outer_dates") == list(CONFIRMATORY_DATES) and body.get("arms") == list(ARMS), "attempt roster drift")
    for field in (
        "target_recordings_opened", "target_bytes_read", "heldout_calib_recordings_opened",
        "optimizer_steps", "backward_steps", "parameter_updates", "warm_starts", "checkpoint_selections",
    ):
        _need(body.get(field) == 0, f"attempt nonzero forbidden field: {field}")
    _need(body.get("cuda_initialized") is False, "attempt after CUDA")
    return body


def collect_predecessor_authority(
    training_root: Path,
    eval_a1_root: Path,
    m4_audit_root: Path,
    regen_root: Path,
    experiment3_root: Path,
) -> dict[str, Any]:
    """Validate every predecessor checkpoint before any target recording opens."""
    eval_terminal, eval_sha = _load_json(eval_a1_root.resolve() / "terminal.json", "h1_cal_aug_prefix_cycle_v1_eval_a1")
    _need(eval_sha == EXPERIMENT4_A1_TERMINAL_SHA256, "Experiment-4 A1 terminal SHA drift")
    _need(eval_terminal.get("status") == "PASS_H1_CAL_AUG_PREFIX_CYCLE_TRANSFER", "Experiment-4 A1 terminal status drift")
    m4_terminal, m4_sha = _load_json(m4_audit_root.resolve() / "terminal.json", SCHEMA.replace("prefix_cycle_m3_transfer", "all_source_heldout"))
    _need(m4_sha == M4_AUDIT_TERMINAL_SHA256, "M4 audit terminal SHA drift")
    _need(m4_terminal.get("status") == "STOP_H1_ALL_SOURCE_HELDOUT_M4_PROTOCOL_INFEASIBLE", "M4 STOP status drift")
    metadata_terminal, metadata_sha = _load_json(
        m4_audit_root.resolve() / "metadata_feasibility_terminal.json",
        "h1_cal_aug_all_source_heldout_v1_metadata_feasibility_terminal",
    )
    _need(metadata_sha == M4_METADATA_TERMINAL_SHA256, "M4 metadata terminal SHA drift")
    _need(metadata_terminal.get("m4_evaluable_recordings") == 0, "M4 metadata feasibility drift")
    _need(metadata_terminal.get("gpu_training_started") is False, "M4 audit unexpectedly trained")

    training = collect_training_authority(training_root)
    _need(training.get("status") == "PASS_H1_CAL_AUG_PREFIX_CYCLE_V1_EVAL_A1_TRAINING_AUTHORITY", "training authority drift")
    _need(training.get("model_count") == 10 and training.get("date_order") == list(CONFIRMATORY_DATES), "ten-model roster drift")
    _need(all(row["outer_date"] in CONFIRMATORY_DATES and row["arm"] in ARMS for row in training["models"]),
          "checkpoint/date pairing drift")
    _need(len({(row["outer_date"], row["arm"]) for row in training["models"]}) == 10, "checkpoint pair duplication")
    source_predecessors = validate_predecessors(experiment3_root, regen_root)
    return {
        "schema": f"{SCHEMA}_checkpoint_authority",
        "status": STATUS_AUTHORITY,
        "validated_at_utc": utc_now(),
        "experiment4_a1_commit": EXPERIMENT4_A1_COMMIT,
        "experiment4_a1_terminal_sha256": eval_sha,
        "m4_audit_commit": M4_AUDIT_COMMIT,
        "m4_audit_terminal_sha256": m4_sha,
        "m4_metadata_terminal_sha256": metadata_sha,
        "outer_dates": list(CONFIRMATORY_DATES),
        "pair_mapping": {date: {"t0": date, "c1": date} for date in CONFIRMATORY_DATES},
        "training_authority": training,
        "source_predecessor_authority": source_predecessors,
        "checkpoint_selection_performed": False,
        "warm_start_performed": False,
        "optimizer_steps": 0,
        "backward_steps": 0,
        "parameter_updates": 0,
        "target_recordings_opened": 0,
        "target_bytes_read": 0,
        "heldout_calib_recordings_opened": 0,
    }


def prepare_predecessor_authority(
    training_root: Path,
    eval_a1_root: Path,
    m4_audit_root: Path,
    regen_root: Path,
    experiment3_root: Path,
    result_root: Path,
) -> dict[str, Any]:
    load_attempt(result_root)
    body = collect_predecessor_authority(training_root, eval_a1_root, m4_audit_root, regen_root, experiment3_root)
    publish_json(result_root.resolve() / "training" / "checkpoint_authority.json", body)
    return body


def _load_checkpoint_authority(result_root: Path) -> tuple[dict[str, Any], str]:
    body, digest = _load_json(
        result_root.resolve() / "training" / "checkpoint_authority.json",
        f"{SCHEMA}_checkpoint_authority",
    )
    _need(body.get("status") == STATUS_AUTHORITY, "checkpoint authority status drift")
    _need(body.get("outer_dates") == list(CONFIRMATORY_DATES), "checkpoint authority date drift")
    _need(body.get("heldout_calib_recordings_opened") == 0, "held-out access in checkpoint authority")
    return body, digest


def _infer_arm(model: Any, record: Any, identity: np.ndarray, carrier: np.ndarray, starts: np.ndarray, device: str) -> np.ndarray:
    import torch

    prediction = np.empty((len(starts), 7), dtype=np.float32)
    identity_one = torch.as_tensor(identity, dtype=torch.float32, device=device).unsqueeze(0)
    carrier_one = torch.as_tensor(carrier, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        for offset in range(0, len(starts), BATCH_SIZE):
            selected = starts[offset:offset + BATCH_SIZE]
            neural = np.ascontiguousarray(
                np.stack([record.neural[int(start):int(start) + WINDOW_SIZE] for start in selected]),
                dtype=np.float32,
            )
            count = len(selected)
            output = model(
                torch.as_tensor(neural, dtype=torch.float32, device=device),
                calib_trialized_neural_features=identity_one.expand(count, -1, -1, -1),
                carrier=carrier_one.expand(count, -1, -1),
            )
            values = scale_last_bin_prediction(output.detach().cpu().numpy())
            prediction[offset:offset + count] = values
    _need(np.isfinite(prediction).all(), "nonfinite M3 prediction")
    return prediction


def run_evaluation_cell(
    data_root: Path,
    regen_root: Path,
    training_root: Path,
    result_root: Path,
    outer_date: str,
    physical_gpu: int,
) -> dict[str, Any]:
    _need(outer_date in CONFIRMATORY_DATES, "unregistered M3 outer date")
    directory = result_root.resolve() / "evaluation" / outer_date
    _need(not directory.exists(), f"M3 evaluation cell exists: {outer_date}")
    publish_json(directory / "attempt.json", {
        "schema": SCHEMA,
        "artifact": "evaluation_attempt",
        "status": "ATTEMPT_BEFORE_CUDA_CHECKPOINT_AND_TARGET",
        "outer_date": outer_date,
        "physical_gpu": int(physical_gpu),
        "created_at_utc": utc_now(),
        "target_recordings_opened": 0,
        "target_bytes_read": 0,
        "heldout_calib_recordings_opened": 0,
        "optimizer_steps": 0,
        "backward_steps": 0,
        "parameter_updates": 0,
    })
    access: dict[str, Any] = {
        "outer_date": outer_date,
        "target_recordings_opened": 0,
        "target_bytes_read": 0,
        "target_sessions_opened": [],
        "files": [],
        "heldout_calib_recordings_opened": 0,
    }
    started = time.monotonic()
    try:
        import torch

        _need(torch.cuda.is_available(), "M3 evaluation requires CUDA")
        authority, authority_sha = _load_checkpoint_authority(result_root)
        _need(authority["pair_mapping"][outer_date] == {"t0": outer_date, "c1": outer_date}, "date/pair mapping drift")
        plan, normalizer = _load_plan_normalizer(regen_root, outer_date)
        models: dict[str, Any] = {}
        state_before: dict[str, str] = {}
        training_terminals: dict[str, dict[str, Any]] = {}
        training_terminal_sha: dict[str, str] = {}
        for arm in ARMS:
            models[arm], state_before[arm], training_terminals[arm], training_terminal_sha[arm] = _load_arm_model(
                training_root, outer_date, arm, "cuda:0"
            )
            _need(training_terminals[arm].get("epoch_zero_based") == 49, f"non-epoch49 model: {outer_date}/{arm}")

        records = _target_records(data_root, outer_date, access)
        _need(tuple(records) == target_sessions_for_date(outer_date), "outer-date target roster drift")
        arrays: dict[str, np.ndarray] = {}
        manifest_rows: list[dict[str, Any]] = []
        per_recording: dict[str, dict[str, float]] = {}
        for index, (session, record) in enumerate(records.items()):
            calibration = materialize_m3_calibration(record, plan, normalizer)
            surface = m3_causal_surface(record.trial_num, record.eval_mask, record.neural.shape[0])
            _need(calibration["support"] == surface["support"], f"M3 support drift: {session}")
            _need(calibration["query_trial"] == surface["query_trial"], f"M3 query drift: {session}")
            starts = np.asarray(surface["starts"], dtype=np.int64)
            output_bins = np.asarray(surface["output_bins"], dtype=np.int64)
            score_mask = np.asarray(surface["score_mask"], dtype=bool)
            target = np.ascontiguousarray(record.velocity[output_bins], dtype=np.float32)
            prefix = f"recording_{index}"
            arrays[f"{prefix}_target"] = target
            arrays[f"{prefix}_score_mask"] = score_mask
            arrays[f"{prefix}_starts"] = starts
            arrays[f"{prefix}_output_bins"] = output_bins
            predictions: dict[str, np.ndarray] = {}
            scores: dict[str, float] = {}
            for arm in ARMS:
                predictions[arm] = _infer_arm(
                    models[arm], record, calibration["identity"], calibration["carrier"], starts, "cuda:0"
                )
                arrays[f"{prefix}_{arm}_prediction"] = predictions[arm]
                scores[arm] = score_m3(target, predictions[arm], score_mask)
            per_recording[session] = {
                "t0": scores["t0"],
                "c1": scores["c1"],
                "delta_c1_minus_t0": scores["c1"] - scores["t0"],
            }
            manifest_rows.append({
                "session": session,
                "support": calibration["support"],
                "first_query_trial": calibration["query_trial"],
                "boundary_bin": int(surface["boundary_bin"]),
                "first_start_bin": int(starts[0]),
                "first_output_bin": int(output_bins[0]),
                "identity_sha256": array_sha256(calibration["identity"]),
                "carrier_sha256": array_sha256(calibration["carrier"]),
                "score_rows": int(score_mask.sum()),
            })

        means = {arm: float(np.mean([rows[arm] for rows in per_recording.values()], dtype=np.float64)) for arm in ARMS}
        metrics = {
            "schema": f"{SCHEMA}_metrics",
            "outer_date": outer_date,
            "budget": 3,
            "equal_recording_mean": {
                "R2_T0": means["t0"],
                "R2_C1": means["c1"],
                "delta_c1_minus_t0": means["c1"] - means["t0"],
            },
            "per_recording": {
                session: {
                    "R2_T0": rows["t0"],
                    "R2_C1": rows["c1"],
                    "delta_c1_minus_t0": rows["delta_c1_minus_t0"],
                }
                for session, rows in per_recording.items()
            },
            "pooled_bin_r2_calculated": False,
        }
        cache_path = directory / "prediction_cache.npz"
        cache_sha = publish_npz(cache_path, **arrays)
        manifest_sha = publish_json(directory / "prediction_cache.json", {
            "schema": f"{SCHEMA}_prediction_cache",
            "outer_date": outer_date,
            "sessions": list(records),
            "arrays_file_sha256": cache_sha,
            "array_sha256": {name: array_sha256(value) for name, value in arrays.items()},
            "array_shape": {name: list(value.shape) for name, value in arrays.items()},
            "recordings": manifest_rows,
            "prediction_divisor": 20.0,
            "window_bins": WINDOW_SIZE,
            "last_bin_only": True,
        })
        metrics_sha = publish_json(directory / "metrics.json", metrics)
        state_after = {arm: state_hash(models[arm].state_dict()) for arm in ARMS}
        _need(state_after == state_before, f"model state changed during M3 evaluation: {outer_date}")
        audit = {
            **access,
            "schema": f"{SCHEMA}_target_access",
            "expected_sessions": list(target_sessions_for_date(outer_date)),
            "authorized_outer_date_only": True,
            "heldout_calib_recordings_opened": 0,
            "optimizer_steps": 0,
            "backward_steps": 0,
            "parameter_updates": 0,
            "warm_starts": 0,
            "checkpoint_selections": 0,
            "ema_applied": False,
            "tta_applied": False,
            "postprocessing_applied": False,
            "target_plan_fits": 0,
            "target_prior_fits": 0,
            "target_normalizer_fits": 0,
            "pooled_bin_r2_calculations": 0,
        }
        audit_sha = publish_json(directory / "target_access.json", audit)
        body = {
            "schema": SCHEMA,
            "status": STATUS_CELL,
            "outer_date": outer_date,
            "gpu": {**_gpu_profile(physical_gpu), "physical_index": int(physical_gpu)},
            "checkpoint_authority_sha256": authority_sha,
            "training_terminal_sha256": training_terminal_sha,
            "model_state_before_sha256": state_before,
            "model_state_after_sha256": state_after,
            "model_state_immutable": True,
            "metrics": metrics,
            "metrics_sha256": metrics_sha,
            "prediction_cache_sha256": cache_sha,
            "prediction_cache_manifest_sha256": manifest_sha,
            "target_access_sha256": audit_sha,
            "optimizer_steps": 0,
            "backward_steps": 0,
            "parameter_updates": 0,
            "warm_starts": 0,
            "checkpoint_selections": 0,
            "elapsed_seconds": time.monotonic() - started,
            "finished_at_utc": utc_now(),
        }
        publish_json(directory / "terminal.json", body)
        return body
    except BaseException as error:
        try:
            publish_json(directory / "failure.json", {
                "schema": SCHEMA,
                "status": "FAIL_M3_EVALUATION_NO_AUTOMATIC_RETRY",
                "outer_date": outer_date,
                "error_type": type(error).__name__,
                "error": str(error),
                "target_access": access,
                "optimizer_steps": 0,
                "backward_steps": 0,
                "parameter_updates": 0,
                "elapsed_seconds": time.monotonic() - started,
            })
        except BaseException:
            pass
        raise


def _verify_cache(directory: Path, terminal: Mapping[str, Any], outer_date: str) -> dict[str, Any]:
    manifest, manifest_sha = _load_json(directory / "prediction_cache.json", f"{SCHEMA}_prediction_cache")
    _need(manifest_sha == terminal.get("prediction_cache_manifest_sha256"), "prediction manifest SHA drift")
    cache_path = directory / "prediction_cache.npz"
    _need(verify_sidecar(cache_path) == terminal.get("prediction_cache_sha256") == manifest.get("arrays_file_sha256"),
          "prediction cache SHA drift")
    with np.load(cache_path, allow_pickle=False) as loaded:
        arrays = {name: np.asarray(loaded[name]) for name in loaded.files}
    _need(set(arrays) == set(manifest["array_sha256"]), "prediction cache array set drift")
    for name, value in arrays.items():
        _need(array_sha256(value) == manifest["array_sha256"][name], f"prediction array SHA drift: {name}")
        _need(list(value.shape) == manifest["array_shape"][name], f"prediction array shape drift: {name}")
    per_recording: dict[str, dict[str, float]] = {}
    for index, session in enumerate(target_sessions_for_date(outer_date)):
        prefix = f"recording_{index}"
        target = arrays[f"{prefix}_target"]
        mask = np.asarray(arrays[f"{prefix}_score_mask"], dtype=bool)
        starts = arrays[f"{prefix}_starts"]
        outputs = arrays[f"{prefix}_output_bins"]
        _need(np.array_equal(outputs, starts + WINDOW_SIZE - 1), f"W700 output alignment drift: {session}")
        scores = {arm: score_m3(target, arrays[f"{prefix}_{arm}_prediction"], mask) for arm in ARMS}
        per_recording[session] = {
            "R2_T0": scores["t0"],
            "R2_C1": scores["c1"],
            "delta_c1_minus_t0": scores["c1"] - scores["t0"],
        }
    means = {
        "R2_T0": float(np.mean([row["R2_T0"] for row in per_recording.values()], dtype=np.float64)),
        "R2_C1": float(np.mean([row["R2_C1"] for row in per_recording.values()], dtype=np.float64)),
    }
    means["delta_c1_minus_t0"] = means["R2_C1"] - means["R2_T0"]
    recomputed = {
        "schema": f"{SCHEMA}_metrics",
        "outer_date": outer_date,
        "budget": 3,
        "equal_recording_mean": means,
        "per_recording": per_recording,
        "pooled_bin_r2_calculated": False,
    }
    metrics, metrics_sha = _load_json(directory / "metrics.json", f"{SCHEMA}_metrics")
    _need(metrics_sha == terminal.get("metrics_sha256") and metrics == recomputed == terminal.get("metrics"),
          f"M3 metric recomputation drift: {outer_date}")
    return metrics


def interpretation(equal_date_delta: float, positive_dates: int) -> str:
    if equal_date_delta >= 0.01 and positive_dates >= 4:
        return VERDICT_STRONG
    if equal_date_delta > 0.0 and positive_dates >= 4:
        return VERDICT_SUPPORT
    return VERDICT_NO_CLEAR


def verify_terminal(
    training_root: Path,
    eval_a1_root: Path,
    m4_audit_root: Path,
    regen_root: Path,
    experiment3_root: Path,
    result_root: Path,
) -> dict[str, Any]:
    root = result_root.resolve()
    attempt = load_attempt(root)
    attempt_sha = verify_sidecar(root / "attempt.json")
    live_authority = collect_predecessor_authority(
        training_root, eval_a1_root, m4_audit_root, regen_root, experiment3_root
    )
    authority, authority_sha = _load_checkpoint_authority(root)
    for transient in ("validated_at_utc",):
        live_authority.pop(transient, None)
        authority_copy = dict(authority)
        authority_copy.pop(transient, None)
        _need(authority_copy == live_authority, "live checkpoint authority drift")

    aggregate_input: dict[str, dict[str, dict[str, float]]] = {}
    cells = []
    total_recordings = 0
    total_bytes = 0
    for date in CONFIRMATORY_DATES:
        directory = root / "evaluation" / date
        terminal, terminal_sha = _load_json(directory / "terminal.json", SCHEMA)
        _need(terminal.get("status") == STATUS_CELL and terminal.get("outer_date") == date, f"cell terminal drift: {date}")
        _need(terminal.get("checkpoint_authority_sha256") == authority_sha, f"authority binding drift: {date}")
        _need(terminal.get("model_state_immutable") is True, f"model state mutation: {date}")
        _need(terminal.get("model_state_before_sha256") == terminal.get("model_state_after_sha256"), f"state hash drift: {date}")
        for field in ("optimizer_steps", "backward_steps", "parameter_updates", "warm_starts", "checkpoint_selections"):
            _need(terminal.get(field) == 0, f"forbidden cell operation: {date}/{field}")
        audit, audit_sha = _load_json(directory / "target_access.json", f"{SCHEMA}_target_access")
        _need(audit_sha == terminal.get("target_access_sha256"), f"target audit binding drift: {date}")
        _need(tuple(audit["target_sessions_opened"]) == target_sessions_for_date(date), f"target date isolation drift: {date}")
        _need(audit.get("heldout_calib_recordings_opened") == 0, f"held-out access: {date}")
        for field in (
            "optimizer_steps", "backward_steps", "parameter_updates", "warm_starts", "checkpoint_selections",
            "target_plan_fits", "target_prior_fits", "target_normalizer_fits", "pooled_bin_r2_calculations",
        ):
            _need(audit.get(field) == 0, f"forbidden target activity: {date}/{field}")
        _need(not any(audit.get(field) for field in ("ema_applied", "tta_applied", "postprocessing_applied")),
              f"forbidden postprocessing: {date}")
        metrics = _verify_cache(directory, terminal, date)
        aggregate_input[date] = {
            session: {"t0": row["R2_T0"], "c1": row["R2_C1"]}
            for session, row in metrics["per_recording"].items()
        }
        total_recordings += int(audit["target_recordings_opened"])
        total_bytes += int(audit["target_bytes_read"])
        cells.append({
            "outer_date": date,
            "terminal_sha256": terminal_sha,
            "metrics_sha256": terminal["metrics_sha256"],
            "target_access_sha256": audit_sha,
            "gpu": terminal["gpu"],
        })
    aggregate = aggregate_m3_results(aggregate_input)
    verdict = interpretation(aggregate["equal_date_mean_delta_r2"], aggregate["positive_date_count"])
    body = {
        "schema": SCHEMA,
        "status": STATUS_COMPLETE,
        "verdict": verdict,
        "finished_at_utc": utc_now(),
        "interpretation_thresholds": {
            "support": {"equal_date_mean_delta_gt": 0.0, "positive_dates_min": 4},
            "strong": {"equal_date_mean_delta_min": 0.01, "positive_dates_min": 4},
        },
        "secondary_diagnostic_only": True,
        "sealed_m4_governing_result_modified": False,
        "official_heldout_r2_claimed": False,
        "date_order": list(CONFIRMATORY_DATES),
        "aggregate": aggregate,
        "cells": cells,
        "attempt_sha256": attempt_sha,
        "code_closure_sha256": attempt["code_closure_sha256"],
        "checkpoint_authority_sha256": authority_sha,
        "target_recordings_opened": total_recordings,
        "target_bytes_read": total_bytes,
        "heldout_calib_recordings_opened": 0,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "target_parameter_updates": 0,
        "warm_starts": 0,
        "checkpoint_selections": 0,
        "automatic_successor_created": False,
        "claim": "secondary M3 diagnostic on five sealed matched held-in date-LODO checkpoint pairs only",
    }
    terminal_sha = publish_json(root / "terminal.json", body)
    lines = [
        "# H1 CAL-AUG Prefix-Cycle M3 Transfer V1",
        "",
        f"- Status: `{STATUS_COMPLETE}`",
        f"- Verdict: `{verdict}`",
        f"- Equal-date mean delta C1−T0: `{aggregate['equal_date_mean_delta_r2']:+.9f}`",
        f"- Positive dates: `{aggregate['positive_date_count']}/5`",
        "- Target optimizer/backward/parameter updates, warm starts and checkpoint selections: `0`.",
        "- H1 held-out-calib recordings opened: `0`.",
        "- This is a secondary five-date held-in date-LODO diagnostic, not official held-out R².",
        "",
        "| Date | R2 T0 | R2 C1 | Delta C1−T0 |",
        "|---|---:|---:|---:|",
    ]
    for date in CONFIRMATORY_DATES:
        row = aggregate["dates"][date]["equal_recording_mean_r2"]
        delta = aggregate["dates"][date]["delta_c1_minus_t0"]
        lines.append(f"| {date} | {row['t0']:+.9f} | {row['c1']:+.9f} | {delta:+.9f} |")
    lines.extend(["", "## Per-recording paired results", "", "| Date | Session | R2 T0 | R2 C1 | Delta C1−T0 |", "|---|---|---:|---:|---:|"])
    for date in CONFIRMATORY_DATES:
        for session, row in aggregate["dates"][date]["per_recording_r2"].items():
            lines.append(
                f"| {date} | {session} | {row['t0']:+.9f} | {row['c1']:+.9f} | {row['delta_c1_minus_t0']:+.9f} |"
            )
    lines.extend(["", f"Terminal SHA-256: `{terminal_sha}`", ""])
    publish_text(root / "EXPERIMENT_RECORD.md", "\n".join(lines))
    return body


__all__ = (
    "STATUS_AUTHORITY", "STATUS_CELL", "STATUS_COMPLETE", "VERDICT_NO_CLEAR", "VERDICT_STRONG",
    "VERDICT_SUPPORT", "collect_predecessor_authority", "create_attempt", "interpretation", "load_attempt",
    "prepare_predecessor_authority", "run_evaluation_cell", "verify_terminal",
)
