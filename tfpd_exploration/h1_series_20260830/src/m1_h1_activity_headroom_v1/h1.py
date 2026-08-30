"""Physical frozen-weight H1 H-C activity-headroom evaluation."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from .core import (
    ARM_ORDER,
    ActivityArm,
    ActivityHeadroomError,
    array_digest,
    encode_trial_activity,
    forward_with_cached_identity,
    grouped_indices,
    identity_from_encoded_trials,
    identity_from_raw_trials,
    selection_for_output_trial,
    variance_weighted_r2,
)
from .m1 import write_once


SUPPORT_TRIALS = 4
GROWING_CAP = 30
WINDOW = 700
UNITS = 176
OUTPUTS = 7
BATCH_SIZE = 32
ACCEPTED_POOLED_R2 = 0.5255107931417206
ACCEPTED_QUERY_SHA256 = "665fe535e90a221123b778171685577f67c2cc1902648cde2c8b2523e990e4da"
CHECKPOINT_RELATIVE = "SPINT-main/pilot_artifacts/h1_carrierid/gpu_runs/h32_fold0_v1/full/checkpoints/fixed_epoch50/epoch_049.ckpt"
CHECKPOINT_SHA256 = "f23e83c9ee8ca6c11d3c6b86410e856d906ccc8c37486aa13ae2e3a2af008fff"
CONFIG_RELATIVE = "SPINT-main/pilot_artifacts/h1_carrierid/gpu_runs/h32_fold0_v1/full/.hydra/config.yaml"
CONFIG_SHA256 = "049751a0135ab707968fe9a91582a88bbf726834279893a758204b62fe3874df"
TERMINAL_RELATIVE = "SPINT-main/pilot_artifacts/h1_carrierid/gpu_runs/h32_fold0_v1/H1_CARRIERID_H32_FOLD0_TERMINAL_GATE.json"
TERMINAL_SHA256 = "6783513e868e77dc222b43737a239f9b14330544eb03ce3cfe9d976de992a9a4"
RAW_RECEIPT_RELATIVE = "sua_exploration/results/h1_m4_population_decoder_carrier_date_lodo_v1/H1_M4_POPULATION_DECODER_CARRIER_CPU_RECEIPT.json"
EB_RECEIPT_RELATIVE = "sua_exploration/results/h1_m4_empirical_bayes_confidence_carrier_date_lodo_v1/H1_M4_EMPIRICAL_BAYES_CONFIDENCE_CARRIER_CPU_RECEIPT.json"
DATA_RELATIVE = "SPINT-main/data/000954"
CACHE_RELATIVE = "SPINT-main/pilot_artifacts/h1_carrierid/shared_source_cache"
SOURCE_AUTHORITY_RELATIVE = "SPINT-main/pilot_artifacts/h1_carrierid_hu/source_authority_v1"
SOURCE_AUTHORITY_RECEIPT_SHA256 = "fcb0cf843f351715677f14e3cf80acdb613fc5a59fb96bfcf7efc23836b16fb8"


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ActivityHeadroomError(message)


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap(root: Path) -> None:
    project = root / "SPINT-main"
    _need(project.is_dir(), "H1 project root is absent")
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))


def _output_trial_positions(dataset: Any) -> tuple[int, ...]:
    result = []
    for session, start in dataset.window_indices:
        record = dataset.records[session]
        output_trial = float(record.trial_num[int(start) + WINDOW - 1])
        matches = np.flatnonzero(np.asarray(record.trial_values, dtype=np.float64) == output_trial)
        _need(matches.size == 1, "H1 query output trial identity drift")
        result.append(int(matches[0]))
    return tuple(result)


def _evaluate_arm(
    *, model: Any, dataset: Any, trial_activity_by_session: dict[str, np.ndarray], output_trials: tuple[int, ...],
    arm: ActivityArm, device: str,
) -> dict[str, Any]:
    import torch

    predictions = np.empty((len(dataset), OUTPUTS), dtype=np.float32)
    targets = np.empty((len(dataset), OUTPUTS), dtype=np.float32)
    session_names = tuple(session for session, _start in dataset.window_indices)
    selections = tuple(
        selection_for_output_trial(
            arm,
            output_trial_index=output_trials[index],
            total_trials=len(dataset.records[session_names[index]].trial_values),
            support_trials=SUPPORT_TRIALS,
            growing_cap=GROWING_CAP,
        )
        for index in range(len(dataset))
    )
    composite = tuple((session_names[index], selections[index]) for index in range(len(dataset)))
    groups: dict[tuple[str, tuple[int, ...]], list[int]] = {}
    for row, key in enumerate(composite):
        groups.setdefault(key, []).append(row)
    forwards = 0
    started = time.monotonic()
    with torch.no_grad():
        for (session, selection), rows_list in groups.items():
            rows = tuple(rows_list)
            carrier = np.asarray(dataset.support[session].carriers["full"], dtype=np.float32)
            identity = identity_from_raw_trials(
                model.net, trial_activity_by_session[session], selection,
                family="h1", device=device, carrier=carrier,
            )
            for offset in range(0, len(rows), BATCH_SIZE):
                batch_rows = rows[offset:offset + BATCH_SIZE]
                xs = []
                ys = []
                for row in batch_rows:
                    current_session, start = dataset.window_indices[row]
                    _need(current_session == session, "H1 grouped session drift")
                    record = dataset.records[session]
                    end = int(start) + WINDOW
                    xs.append(record.neural[int(start):end])
                    ys.append(record.velocity[end - 1])
                output = forward_with_cached_identity(
                    model.net, np.ascontiguousarray(np.stack(xs), dtype=np.float32), identity,
                )
                pred = np.ascontiguousarray(
                    (output[:, -1, :] / float(model.hparams.behavior_scaling_factor)).detach().cpu().numpy(),
                    dtype=np.float32,
                )
                predictions[np.asarray(batch_rows, dtype=np.int64)] = pred
                targets[np.asarray(batch_rows, dtype=np.int64)] = np.ascontiguousarray(np.stack(ys), dtype=np.float32)
                forwards += 1
    per_session = {}
    for session in dataset.records:
        mask = np.asarray([name == session for name in session_names], dtype=bool)
        per_session[session] = {
            "n_windows": int(mask.sum()),
            "r2": variance_weighted_r2(predictions[mask], targets[mask]),
        }
    return {
        "arm": arm.value,
        "causal": arm is not ActivityArm.FULL_SESSION_ORACLE,
        "label_free": True,
        "h1_carrier_unchanged": True,
        "deployment_status": (
            "LABEL_FREE_BUT_NONCAUSAL" if arm is ActivityArm.FULL_SESSION_ORACLE
            else "CAUSAL_CARDINALITY_MATCHED" if arm in {ActivityArm.STATIC_SUPPORT, ActivityArm.ROLLING_FIXED_M}
            else "CAUSAL_CARDINALITY_OOD"
        ),
        "pooled_r2": variance_weighted_r2(predictions, targets),
        "equal_recording_mean_r2": float(np.mean([row["r2"] for row in per_session.values()], dtype=np.float64)),
        "per_recording": per_session,
        "n_windows": len(dataset),
        "prediction_sha256": array_digest(predictions),
        "target_sha256": array_digest(targets),
        "unique_activity_states": len(set(composite)),
        "activity_cardinality_min": min(map(len, selections)),
        "activity_cardinality_max": max(map(len, selections)),
        "forward_batches": forwards,
        "elapsed_seconds": time.monotonic() - started,
        "_prediction": predictions,
        "_target": targets,
    }


def _evaluate_direct_static(model: Any, dataset: Any, *, device: str) -> dict[str, Any]:
    """Replay the accepted H-C evaluator with its ordinary per-row M4 identity."""
    import torch

    predictions = np.empty((len(dataset), OUTPUTS), dtype=np.float32)
    targets = np.empty((len(dataset), OUTPUTS), dtype=np.float32)
    session_names: list[str] = []
    forwards = 0
    started = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(dataset), BATCH_SIZE):
            rows = tuple(range(start, min(start + BATCH_SIZE, len(dataset))))
            items = [dataset[row] for row in rows]
            neural = np.ascontiguousarray(np.stack([item[0] for item in items]), dtype=np.float32)
            target = np.ascontiguousarray(np.stack([item[1] for item in items]), dtype=np.float32)
            identity = np.ascontiguousarray(np.stack([item[2] for item in items]), dtype=np.float32)
            carrier = np.ascontiguousarray(np.stack([item[4] for item in items]), dtype=np.float32)
            output = model(
                torch.as_tensor(neural, dtype=torch.float32, device=device),
                calib_trialized_neural_features=torch.as_tensor(identity, dtype=torch.float32, device=device),
                carrier=torch.as_tensor(carrier, dtype=torch.float32, device=device),
            )
            predictions[np.asarray(rows, dtype=np.int64)] = np.ascontiguousarray(
                (output[:, -1, :] / float(model.hparams.behavior_scaling_factor)).detach().cpu().numpy(), dtype=np.float32,
            )
            targets[np.asarray(rows, dtype=np.int64)] = target[:, -1, :]
            session_names.extend(str(item[3]) for item in items)
            forwards += 1
    per_session = {}
    for session in dataset.records:
        mask = np.asarray([name == session for name in session_names], dtype=bool)
        per_session[session] = {
            "n_windows": int(mask.sum()),
            "r2": variance_weighted_r2(predictions[mask], targets[mask]),
        }
    return {
        "arm": ActivityArm.STATIC_SUPPORT.value,
        "causal": True,
        "label_free": True,
        "h1_carrier_unchanged": True,
        "deployment_status": "CAUSAL_CARDINALITY_MATCHED",
        "pooled_r2": variance_weighted_r2(predictions, targets),
        "equal_recording_mean_r2": float(np.mean([row["r2"] for row in per_session.values()], dtype=np.float64)),
        "per_recording": per_session,
        "n_windows": len(dataset),
        "prediction_sha256": array_digest(predictions),
        "target_sha256": array_digest(targets),
        "unique_activity_states": len(dataset.records),
        "activity_cardinality_min": SUPPORT_TRIALS,
        "activity_cardinality_max": SUPPORT_TRIALS,
        "forward_batches": forwards,
        "elapsed_seconds": time.monotonic() - started,
        "forward_path": "accepted_eager_repeated_B32_identity_and_fixed_carrier",
        "_prediction": predictions,
        "_target": targets,
    }


def run(root: Path, *, device: str) -> dict[str, Any]:
    import torch

    root = Path(root).resolve()
    _bootstrap(root)
    from src.data.h1_m4_eb_normalized_v2 import H1M4EBNormalizedV2StrictTargetDataset, fit_source_normalizer_from_cache
    from src.data.h1_m4_eb_pilot import (
        H1_M4_FOLD0_TARGET, interpolate_trial_identity, load_immutable_source_authority,
        load_source_records, load_target_records, validate_target_receipt_binding,
    )
    from src.h1_m4_eb_normalized_v2_contract import state_hash
    from scripts.h1_carrierid_evaluate import _instantiate, _load_carrierid_checkpoint, _validate_carrierid_config

    _need(device.startswith("cuda") and torch.cuda.is_available(), "H1 experiment requires an available CUDA device")
    checkpoint = root / CHECKPOINT_RELATIVE
    config_path = root / CONFIG_RELATIVE
    terminal = root / TERMINAL_RELATIVE
    _need(_file_sha(checkpoint) == CHECKPOINT_SHA256 and _file_sha(config_path) == CONFIG_SHA256
          and _file_sha(terminal) == TERMINAL_SHA256,
          "H1 sealed checkpoint/config/terminal binding drift")
    config = _validate_carrierid_config(config_path, "full")
    payload, metadata = _load_carrierid_checkpoint(checkpoint, config_path, "full")
    source_records = load_source_records(root / DATA_RELATIVE)
    plan, _plan_manifest, carrier_cache, source_authority = load_immutable_source_authority(
        source_records,
        root / SOURCE_AUTHORITY_RELATIVE,
        SOURCE_AUTHORITY_RECEIPT_SHA256,
    )
    normalizer, _raw_carriers = fit_source_normalizer_from_cache(carrier_cache)
    _need(metadata.get("source_manifest_sha256") == "ebdf6b92321826f6433449fa9e89f37d0e4963f9313eac70a7fc8cccff03c01b"
          and metadata.get("normalizer_sha256") == normalizer.normalizer_sha256,
          "H1 checkpoint/source authority binding drift")
    records = load_target_records(root / DATA_RELATIVE)
    validate_target_receipt_binding(records, plan, root / RAW_RECEIPT_RELATIVE, root / EB_RECEIPT_RELATIVE)
    dataset = H1M4EBNormalizedV2StrictTargetDataset(records, plan, normalizer, "full")
    _need(dataset.window_indices_sha256 == ACCEPTED_QUERY_SHA256 and len(dataset) == 8965,
          "H1 strict query surface drift")
    model = _instantiate(config, payload, torch.device(device))
    del payload
    _need(bool(model.hparams.decode_last_timestep_only)
          and bool(model.hparams.predict_scaled_behavior)
          and float(model.hparams.behavior_scaling_factor) == 20.0,
          "H1 output/scaling contract drift")
    state_before = state_hash(model.state_dict())
    trial_activity_by_session = {}
    trial_counts = {}
    for session in H1_M4_FOLD0_TARGET:
        record = records[session]
        trial_activity = np.ascontiguousarray(np.stack([
            interpolate_trial_identity(record, value) for value in record.trial_values
        ]), dtype=np.float32)
        expected_support = np.asarray(dataset.support[session].identity, dtype=np.float32)
        _need(np.array_equal(trial_activity[:SUPPORT_TRIALS], expected_support),
              "H1 support identity reconstruction drift")
        trial_activity_by_session[session] = trial_activity
        trial_counts[session] = int(trial_activity.shape[0])
    output_trials = _output_trial_positions(dataset)
    cached_results = [_evaluate_arm(
        model=model, dataset=dataset, trial_activity_by_session=trial_activity_by_session,
        output_trials=output_trials, arm=arm, device=device,
    ) for arm in ARM_ORDER]
    direct = _evaluate_direct_static(model, dataset, device=device)
    _need(abs(float(direct["pooled_r2"]) - ACCEPTED_POOLED_R2) <= 2.0e-7,
          "H1 static-support replay does not match the accepted score")
    cached = cached_results[0]
    max_abs = float(np.max(np.abs(direct["_prediction"].astype(np.float64) - cached["_prediction"].astype(np.float64))))
    r2_abs = abs(float(direct["pooled_r2"]) - float(cached["pooled_r2"]))
    _need(max_abs <= 2.0e-6 and r2_abs <= 2.0e-7,
          "H1 cached-identity path exceeds the frozen numeric parity bound")
    direct["cached_identity_parity"] = {
        "max_abs_prediction": max_abs,
        "abs_r2": r2_abs,
        "tolerance_prediction": 2.0e-6,
        "tolerance_r2": 2.0e-7,
        "pass": True,
    }
    results = [direct, *cached_results[1:]]
    for row in results:
        row.pop("_prediction", None)
        row.pop("_target", None)
    state_after = state_hash(model.state_dict())
    _need(state_before == state_after, "H1 model state changed during evaluation")
    return {
        "schema": "h1_activity_headroom_fold0_hc_v1",
        "status": "COMPLETE_FROZEN_WEIGHT_ACTIVITY_HEADROOM",
        "dataset": "H1",
        "surface": "fold0_strict_post_m4_support_two_recordings",
        "device": device,
        "authority": {
            "checkpoint_relative": CHECKPOINT_RELATIVE,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "config_relative": CONFIG_RELATIVE,
            "config_sha256": CONFIG_SHA256,
            "terminal_relative": TERMINAL_RELATIVE,
            "terminal_sha256": TERMINAL_SHA256,
            "strict_query_window_indices_sha256": ACCEPTED_QUERY_SHA256,
            "accepted_static_pooled_r2": ACCEPTED_POOLED_R2,
            "source_manifest_sha256": metadata["source_manifest_sha256"],
            "normalizer_sha256": normalizer.normalizer_sha256,
            "immutable_source_authority": source_authority,
        },
        "contract": {
            "support_trials": SUPPORT_TRIALS,
            "growing_cap": GROWING_CAP,
            "trial_counts": trial_counts,
            "fixed_carrier": True,
            "target_optimizer_backward_update": 0,
            "full_session_oracle_is_noncausal": True,
        },
        "results": results,
        "model_state_before_sha256": state_before,
        "model_state_after_sha256": state_after,
        "model_state_immutable": True,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }


__all__ = ("run", "write_once")
