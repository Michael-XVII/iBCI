"""Additive evaluation amendment for the sealed H1 CAL-AUG Prefix-Cycle V1 training run."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from src.data.h1_m4_cce_date_lodo import target_sessions_for_date
from src.data.h1_m4_eb_pilot import array_sha256
from src.h1_cal_aug_prefix_cycle_v1 import (
    ARMS,
    BUDGETS,
    SCHEMA as TRAINING_SCHEMA,
    STATUS_ARM,
    STATUS_NO_TRANSFER,
    STATUS_PASS,
    _gpu_profile,
    _infer_budget,
    _load_arm_model,
    _load_plan_normalizer,
    _target_records,
    transfer_decision,
    validate_predecessors,
)
from src.h1_hc_date_lodo_regen_v1 import (
    publish_json,
    publish_npz,
    publish_text,
    variance_weighted_r2,
    verify_sidecar,
)
from src.h1_m4_cce_contract import CONFIRMATORY_DATES, canonical_sha256, state_hash


SCHEMA = "h1_cal_aug_prefix_cycle_v1_eval_a1"
STATUS_AUTHORITY = "PASS_H1_CAL_AUG_PREFIX_CYCLE_V1_EVAL_A1_TRAINING_AUTHORITY"
STATUS_EVAL = "PASS_H1_CAL_AUG_PREFIX_CYCLE_V1_EVAL_A1_DATE_EVALUATED"
TRAINING_SEAL_COMMIT = "a162018f5195fc30578a6d3e8de43a216b216937"
TRAINING_ATTEMPT_SHA256 = "bf5bf46e0a56780a17a038468c8526553a882048e744bc28dad86338e3476b02"
TRAINING_FAILURE_SHA256 = "b2b83a6c4b85023d941833fa4f2fc531fe8f61b0e1ed919f82b26af8f663a4f1"
TRAINING_SOURCE_SHA256 = "4763533232a8676dd5bab2c991ebaf25982c34feee78b4e1730097cae9416c60"


class EvalAmendmentError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise EvalAmendmentError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path, schema: str | None = None) -> tuple[dict[str, Any], str]:
    digest = verify_sidecar(path)
    body = json.loads(path.read_text(encoding="utf-8"))
    if schema is not None:
        _need(body.get("schema") == schema, f"schema drift: {path}")
    return body, digest


def dry_plan() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "DRY_NO_WRITE_NO_DATA_NO_CUDA",
        "amendment": "evaluation-only; no retraining",
        "outer_dates": list(CONFIRMATORY_DATES),
        "arms": list(ARMS),
        "budgets": list(BUDGETS),
        "training_seal_commit": TRAINING_SEAL_COMMIT,
        "target_access": 0,
    }


def create_attempt(result_root: Path, closure: Mapping[str, str], head: str) -> dict[str, Any]:
    root = result_root.resolve()
    _need(not root.exists(), f"canonical amendment root is not fresh: {root}")
    body = {
        "schema": SCHEMA,
        "artifact": "attempt",
        "status": "ATTEMPT_BEFORE_TRAINING_AUTHORITY_CUDA_AND_TARGET",
        "created_at_utc": utc_now(),
        "head": head,
        "closure": dict(closure),
        "code_closure_sha256": canonical_sha256(dict(closure)),
        "training_seal_commit": TRAINING_SEAL_COMMIT,
        "outer_dates": list(CONFIRMATORY_DATES),
        "target_recordings_opened": 0,
        "target_bytes_read": 0,
        "cuda_initialized": False,
    }
    publish_json(root / "attempt.json", body)
    return body


def load_attempt(result_root: Path) -> dict[str, Any]:
    body, _ = _load_json(result_root.resolve() / "attempt.json", SCHEMA)
    _need(body.get("status") == "ATTEMPT_BEFORE_TRAINING_AUTHORITY_CUDA_AND_TARGET", "attempt status drift")
    _need(body.get("target_recordings_opened") == body.get("target_bytes_read") == 0, "attempt target access")
    return body


def collect_training_authority(training_root: Path) -> dict[str, Any]:
    """Revalidate the sealed ten-model run without publishing into its immutable root."""
    import torch

    root = training_root.resolve()
    _need(verify_sidecar(root / "attempt.json") == TRAINING_ATTEMPT_SHA256, "training attempt drift")
    _need(verify_sidecar(root / "training_failure.json") == TRAINING_FAILURE_SHA256, "training failure receipt drift")
    _need(verify_sidecar(root / "source_authority.json") == TRAINING_SOURCE_SHA256, "training source authority drift")
    failure, _ = _load_json(root / "training_failure.json", TRAINING_SCHEMA)
    _need(failure.get("status") == "FAIL_IMMUTABLE_NO_AUTOMATIC_RETRY", "training failure status drift")
    _need(failure.get("phase") == "training" and failure.get("error_type") == "FileExistsError", "unexpected training failure")
    _need(failure.get("target_recordings_opened") == failure.get("target_bytes_read") == 0, "training failure target access")
    _need(not (root / "terminal.json").exists(), "failed training root unexpectedly has terminal")
    _need(not (root / "evaluation").exists(), "failed training root unexpectedly has evaluation")

    pair_rows = []
    model_rows = []
    for date in CONFIRMATORY_DATES:
        pair_path = root / "pairs" / date / "paired_integrity.json"
        pair, pair_sha = _load_json(pair_path, f"{TRAINING_SCHEMA}_paired_integrity")
        _need(pair.get("status") == "PASS_PAIRED_EPOCH49_INTEGRITY", f"pair status drift: {date}")
        _need(pair.get("outer_date") == date and pair.get("smoke") is False, f"pair identity drift: {date}")
        _need(pair.get("target_recordings_opened") == pair.get("target_bytes_read") == 0, f"pair target access: {date}")
        terminals: dict[str, dict[str, Any]] = {}
        terminal_shas: dict[str, str] = {}
        for arm in ARMS:
            cell = root / "pairs" / date / arm
            terminal, terminal_sha = _load_json(cell / "terminal.json", TRAINING_SCHEMA)
            _need(terminal.get("status") == STATUS_ARM, f"arm status drift: {date}/{arm}")
            _need(terminal.get("outer_date") == date and terminal.get("arm") == arm, f"arm identity drift: {date}/{arm}")
            _need(terminal.get("epoch_zero_based") == 49, f"non-epoch49 checkpoint: {date}/{arm}")
            _need(all(terminal.get(field) == 0 for field in (
                "target_recordings_opened", "target_bytes_read", "target_optimizer_steps", "target_backward_steps"
            )), f"target activity in training cell: {date}/{arm}")
            checkpoint = root / terminal["checkpoint"]["relative"]
            checkpoint_sha = verify_sidecar(checkpoint)
            _need(checkpoint_sha == terminal["checkpoint"]["sha256"], f"checkpoint SHA drift: {date}/{arm}")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
            _need(isinstance(metadata, Mapping), f"checkpoint metadata missing: {date}/{arm}")
            _need(state_hash(payload["state_dict"]) == terminal["terminal_state_sha256"], f"checkpoint state drift: {date}/{arm}")
            for field in (
                "outer_date", "arm", "global_step", "initial_state_sha256", "terminal_state_sha256",
                "dropout_probability_sha256", "dropout_probability_count", "source_authority_sha256", "config_sha256",
            ):
                _need(metadata.get(field) == terminal.get(field), f"checkpoint provenance drift: {date}/{arm}/{field}")
            _need(metadata.get("epoch_zero_based") == 49, f"checkpoint epoch drift: {date}/{arm}")
            _need(metadata.get("warm_start") is False and metadata.get("checkpoint_selection") is False,
                  f"checkpoint selection drift: {date}/{arm}")
            config, config_sha = _load_json(cell / "config.json", f"{TRAINING_SCHEMA}_common_config")
            _need(config_sha == terminal["config_sha256"] and config.get("arm") == arm, f"config drift: {date}/{arm}")
            terminals[arm] = terminal
            terminal_shas[arm] = terminal_sha
            model_rows.append({
                "outer_date": date,
                "arm": arm,
                "checkpoint_sha256": checkpoint_sha,
                "config_sha256": config_sha,
                "terminal_sha256": terminal_sha,
                "initial_state_sha256": terminal["initial_state_sha256"],
                "terminal_state_sha256": terminal["terminal_state_sha256"],
                "global_step": terminal["global_step"],
                "dropout_probability_sha256": terminal["dropout_probability_sha256"],
            })
            del payload
        t0, c1 = terminals["t0"], terminals["c1"]
        for field in ("gpu", "initial_state_sha256", "global_step", "dropout_probability_sha256", "dropout_probability_count"):
            _need(t0[field] == c1[field], f"live paired drift: {date}/{field}")
        _need(pair.get("t0_terminal_sha256") == terminal_shas["t0"], f"T0 terminal binding drift: {date}")
        _need(pair.get("c1_terminal_sha256") == terminal_shas["c1"], f"C1 terminal binding drift: {date}")
        _need(pair.get("initial_state_sha256") == t0["initial_state_sha256"], f"pair initial state drift: {date}")
        _need(pair.get("dropout_probability_sha256") == t0["dropout_probability_sha256"], f"pair dropout drift: {date}")
        _need(pair.get("global_step_per_arm") == t0["global_step"], f"pair step drift: {date}")
        pair_rows.append({"outer_date": date, "paired_integrity_sha256": pair_sha})
    return {
        "schema": f"{SCHEMA}_training_authority",
        "status": STATUS_AUTHORITY,
        "training_schema": TRAINING_SCHEMA,
        "training_seal_commit": TRAINING_SEAL_COMMIT,
        "training_attempt_sha256": TRAINING_ATTEMPT_SHA256,
        "training_failure_sha256": TRAINING_FAILURE_SHA256,
        "training_source_authority_sha256": TRAINING_SOURCE_SHA256,
        "date_order": list(CONFIRMATORY_DATES),
        "pairs": pair_rows,
        "models": model_rows,
        "model_count": len(model_rows),
        "target_recordings_opened": 0,
        "target_bytes_read": 0,
    }


def prepare_authority(training_root: Path, predecessor_root: Path, experiment3_root: Path, result_root: Path) -> dict[str, Any]:
    predecessor = validate_predecessors(experiment3_root, predecessor_root)
    training = collect_training_authority(training_root)
    body = {**training, "predecessor_authority": predecessor}
    publish_json(result_root.resolve() / "training_authority.json", body)
    return body


def run_evaluation_cell(
    data_root: Path,
    predecessor_root: Path,
    training_root: Path,
    result_root: Path,
    date: str,
    physical_gpu: int,
) -> dict[str, Any]:
    directory = result_root.resolve() / "evaluation" / date
    _need(not directory.exists(), f"evaluation exists: {date}")
    publish_json(directory / "attempt.json", {
        "schema": SCHEMA,
        "artifact": "evaluation_attempt",
        "outer_date": date,
        "physical_gpu": int(physical_gpu),
        "created_at_utc": utc_now(),
        "target_recordings_opened": 0,
        "target_bytes_read": 0,
        "optimizer_steps": 0,
        "backward_steps": 0,
    })
    access = {
        "outer_date": date,
        "target_recordings_opened": 0,
        "target_bytes_read": 0,
        "target_sessions_opened": [],
        "files": [],
    }
    started = time.monotonic()
    try:
        import torch

        _need(torch.cuda.is_available(), "evaluation requires CUDA")
        authority, authority_sha = _load_json(result_root.resolve() / "training_authority.json", f"{SCHEMA}_training_authority")
        _need(authority.get("status") == STATUS_AUTHORITY and authority.get("model_count") == 10, "training authority drift")
        plan, normalizer = _load_plan_normalizer(predecessor_root, date)
        models: dict[str, Any] = {}
        states: dict[str, str] = {}
        training_terminals: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            models[arm], states[arm], training_terminals[arm], _ = _load_arm_model(training_root, date, arm, "cuda:0")
        records = _target_records(data_root, date, access)
        arrays: dict[str, np.ndarray] = {}
        supports: dict[str, Any] = {}
        metrics: dict[str, Any] = {"schema": f"{SCHEMA}_metrics", "outer_date": date, "budgets": {}}
        for budget in BUDGETS:
            by_arm: dict[str, float] = {}
            by_recording: dict[str, dict[str, float]] = {}
            for arm in ARMS:
                scores: dict[str, float] = {}
                for index, (session, record) in enumerate(records.items()):
                    cache = _infer_budget(models[arm], record, plan, normalizer, budget, "cuda:0")
                    prefix = f"m{budget}_{arm}_{index}"
                    arrays[f"{prefix}_prediction"] = cache["prediction"]
                    arrays[f"{prefix}_target"] = cache["target"]
                    arrays[f"{prefix}_score_mask"] = cache["score_mask"]
                    arrays[f"{prefix}_output_bins"] = cache["output_bins"]
                    mask = cache["score_mask"]
                    scores[session] = variance_weighted_r2(cache["target"][mask], cache["prediction"][mask])
                    supports[f"m{budget}_{arm}_{session}"] = {
                        key: cache[key] for key in ("support", "next_trial", "identity_sha256", "carrier_sha256")
                    }
                by_recording[arm] = scores
                by_arm[arm] = float(np.mean(list(scores.values()), dtype=np.float64))
            metrics["budgets"][str(budget)] = {
                "equal_recording_mean_r2": by_arm,
                "delta_c1_minus_t0": by_arm["c1"] - by_arm["t0"],
                "per_recording_r2": by_recording,
            }
        cache_path = directory / "prediction_cache.npz"
        cache_sha = publish_npz(cache_path, **arrays)
        manifest_sha = publish_json(directory / "prediction_cache.json", {
            "schema": f"{SCHEMA}_prediction_cache",
            "outer_date": date,
            "sessions": list(records),
            "arrays_file_sha256": cache_sha,
            "array_sha256": {key: array_sha256(value) for key, value in arrays.items()},
            "array_shape": {key: list(value.shape) for key, value in arrays.items()},
            "support": supports,
        })
        metrics_sha = publish_json(directory / "metrics.json", metrics)
        for arm in ARMS:
            _need(state_hash(models[arm].state_dict()) == states[arm], f"{arm} state changed during evaluation")
        audit = {
            **access,
            "expected_sessions": list(target_sessions_for_date(date)),
            "authorized_outer_date_only": True,
            "optimizer_steps": 0,
            "backward_steps": 0,
            "model_updates": 0,
            "target_driven_selection": False,
        }
        audit_sha = publish_json(directory / "target_access.json", audit)
        body = {
            "schema": SCHEMA,
            "status": STATUS_EVAL,
            "outer_date": date,
            "gpu": {**_gpu_profile(physical_gpu), "physical_index": int(physical_gpu)},
            "training_authority_sha256": authority_sha,
            "training_terminal_sha256": {
                arm: verify_sidecar(training_root.resolve() / "pairs" / date / arm / "terminal.json") for arm in ARMS
            },
            "metrics": metrics,
            "metrics_sha256": metrics_sha,
            "prediction_cache_sha256": cache_sha,
            "prediction_cache_manifest_sha256": manifest_sha,
            "target_access": audit,
            "target_access_sha256": audit_sha,
            "model_state_immutable": True,
            "optimizer_steps": 0,
            "backward_steps": 0,
            "model_updates": 0,
            "selection_performed": False,
            "elapsed_seconds": time.monotonic() - started,
            "finished_at_utc": utc_now(),
        }
        publish_json(directory / "terminal.json", body)
        return body
    except BaseException as error:
        try:
            publish_json(directory / "failure.json", {
                "schema": SCHEMA,
                "status": "FAIL_EVALUATION_A1_NO_AUTOMATIC_RETRY",
                "outer_date": date,
                "error_type": type(error).__name__,
                "error": str(error),
                "target_access": access,
                "optimizer_steps": 0,
                "backward_steps": 0,
                "elapsed_seconds": time.monotonic() - started,
            })
        except BaseException:
            pass
        raise


def _verify_evaluation_cache(directory: Path, terminal: Mapping[str, Any], date: str) -> dict[str, Any]:
    manifest, manifest_sha = _load_json(directory / "prediction_cache.json", f"{SCHEMA}_prediction_cache")
    _need(manifest_sha == terminal.get("prediction_cache_manifest_sha256"), "prediction manifest SHA drift")
    cache_path = directory / "prediction_cache.npz"
    _need(verify_sidecar(cache_path) == terminal.get("prediction_cache_sha256") == manifest.get("arrays_file_sha256"),
          "prediction cache SHA drift")
    with np.load(cache_path, allow_pickle=False) as values:
        arrays = {name: np.asarray(values[name]) for name in values.files}
    _need(set(arrays) == set(manifest.get("array_sha256", {})), "prediction cache array set drift")
    for name, value in arrays.items():
        _need(array_sha256(value) == manifest["array_sha256"][name], f"prediction array digest drift: {name}")
        _need(list(value.shape) == manifest["array_shape"][name], f"prediction array shape drift: {name}")
    sessions = target_sessions_for_date(date)
    recomputed: dict[str, Any] = {"schema": f"{SCHEMA}_metrics", "outer_date": date, "budgets": {}}
    for budget in BUDGETS:
        by_arm: dict[str, float] = {}
        by_recording: dict[str, dict[str, float]] = {}
        for arm in ARMS:
            scores: dict[str, float] = {}
            for index, session in enumerate(sessions):
                prefix = f"m{budget}_{arm}_{index}"
                prediction = arrays[f"{prefix}_prediction"]
                target = arrays[f"{prefix}_target"]
                mask = np.asarray(arrays[f"{prefix}_score_mask"], bool)
                reference = f"m{budget}_{ARMS[0]}_{index}"
                _need(np.array_equal(target, arrays[f"{reference}_target"]), "paired target bytes drift")
                _need(np.array_equal(mask, arrays[f"{reference}_score_mask"]), "paired mask drift")
                _need(np.array_equal(arrays[f"{prefix}_output_bins"], arrays[f"{reference}_output_bins"]), "paired bins drift")
                scores[session] = variance_weighted_r2(target[mask], prediction[mask])
            by_recording[arm] = scores
            by_arm[arm] = float(np.mean(list(scores.values()), dtype=np.float64))
        recomputed["budgets"][str(budget)] = {
            "equal_recording_mean_r2": by_arm,
            "delta_c1_minus_t0": by_arm["c1"] - by_arm["t0"],
            "per_recording_r2": by_recording,
        }
    metrics, metrics_sha = _load_json(directory / "metrics.json", f"{SCHEMA}_metrics")
    _need(metrics_sha == terminal.get("metrics_sha256") and metrics == recomputed == terminal.get("metrics"),
          "evaluation metric recomputation drift")
    return metrics


def verify_terminal(
    training_root: Path,
    predecessor_root: Path,
    experiment3_root: Path,
    result_root: Path,
) -> dict[str, Any]:
    root = result_root.resolve()
    attempt = load_attempt(root)
    attempt_sha = verify_sidecar(root / "attempt.json")
    live = {**collect_training_authority(training_root), "predecessor_authority": validate_predecessors(experiment3_root, predecessor_root)}
    authority, authority_sha = _load_json(root / "training_authority.json", f"{SCHEMA}_training_authority")
    _need(authority == live and authority.get("status") == STATUS_AUTHORITY, "live training authority drift")
    metrics: dict[str, Any] = {}
    rows = []
    total_recordings = 0
    total_bytes = 0
    for date in CONFIRMATORY_DATES:
        directory = root / "evaluation" / date
        terminal, terminal_sha = _load_json(directory / "terminal.json", SCHEMA)
        _need(terminal.get("status") == STATUS_EVAL and terminal.get("model_state_immutable") is True,
              f"evaluation terminal drift: {date}")
        _need(terminal.get("training_authority_sha256") == authority_sha, f"training authority binding drift: {date}")
        _need(terminal.get("optimizer_steps") == terminal.get("backward_steps") == terminal.get("model_updates") == 0,
              f"target model update: {date}")
        audit = terminal["target_access"]
        _need(tuple(audit["target_sessions_opened"]) == target_sessions_for_date(date), f"target isolation drift: {date}")
        _need(audit.get("target_driven_selection") is False, f"target selection recorded: {date}")
        metrics[date] = _verify_evaluation_cache(directory, terminal, date)
        total_recordings += int(audit["target_recordings_opened"])
        total_bytes += int(audit["target_bytes_read"])
        rows.append({"outer_date": date, "terminal_sha256": terminal_sha, "gpu": terminal["gpu"], "metrics": metrics[date]})
    decision = transfer_decision(metrics)
    status = STATUS_PASS if decision["verdict"] == "PASS_TRANSFER" else STATUS_NO_TRANSFER
    body = {
        "schema": SCHEMA,
        "status": status,
        "finished_at_utc": utc_now(),
        "amendment": "evaluation-only continuation of sealed ten-model V1 training",
        "training_seal_commit": TRAINING_SEAL_COMMIT,
        "date_order": list(CONFIRMATORY_DATES),
        "arms": list(ARMS),
        "budgets": list(BUDGETS),
        "decision": decision,
        "experiment_attempt_sha256": attempt_sha,
        "code_closure_sha256": attempt["code_closure_sha256"],
        "training_authority_sha256": authority_sha,
        "cells": rows,
        "target_recordings_opened": total_recordings,
        "target_bytes_read": total_bytes,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "target_model_updates": 0,
        "target_driven_selection": False,
        "claim": "fixed H1 M7/M5/M4 CAL-AUG prefix-cycle on the sealed five matched date-LODO pairs only",
    }
    terminal_sha = publish_json(root / "terminal.json", body)
    lines = [
        "# H1 CAL-AUG Prefix-Cycle V1 Evaluation Amendment A1",
        "",
        f"- Status: `{status}`",
        f"- Training seal commit: `{TRAINING_SEAL_COMMIT}`",
        f"- M4 equal-date delta: `{decision['equal_date_delta_r2']['4']:+.9f}`",
        f"- M4 positive dates: `{decision['m4_positive_dates']}/5`",
        f"- M5 safety delta: `{decision['equal_date_delta_r2']['5']:+.9f}`",
        f"- M7 safety delta: `{decision['equal_date_delta_r2']['7']:+.9f}`",
        "- Target optimizer/backward/model updates and target-driven selections: `0`.",
        "",
        "| Date | M4 delta C1-T0 | M5 delta | M7 delta | GPU UUID |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        date = row["outer_date"]
        values = row["metrics"]["budgets"]
        lines.append(
            f"| {date} | {values['4']['delta_c1_minus_t0']:+.9f} | "
            f"{values['5']['delta_c1_minus_t0']:+.9f} | {values['7']['delta_c1_minus_t0']:+.9f} | "
            f"`{row['gpu']['uuid']}` |"
        )
    lines.extend(["", f"Terminal SHA-256: `{terminal_sha}`", ""])
    publish_text(root / "EXPERIMENT_RECORD.md", "\n".join(lines))
    return body


__all__ = (
    "SCHEMA",
    "STATUS_AUTHORITY",
    "STATUS_EVAL",
    "TRAINING_SEAL_COMMIT",
    "collect_training_authority",
    "create_attempt",
    "dry_plan",
    "load_attempt",
    "prepare_authority",
    "run_evaluation_cell",
    "verify_terminal",
)
