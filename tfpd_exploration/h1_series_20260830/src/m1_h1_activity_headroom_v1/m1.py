"""Physical frozen-weight M1 activity-headroom evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .core import (
    ARM_ORDER,
    ActivityArm,
    ActivityHeadroomError,
    array_digest,
    canonical_json_bytes,
    encode_trial_activity,
    forward_with_cached_identity,
    grouped_indices,
    identity_from_encoded_trials,
    identity_from_raw_trials,
    selection_for_output_trial,
    variance_weighted_r2,
)


TARGET_SESSION = "20120924"
SUPPORT_TRIALS = 10
GROWING_CAP = 30
WINDOW = 100
UNITS = 64
OUTPUTS = 16
BATCH_SIZE = 128
ACCEPTED_R2 = 0.5707439184188843
ACCEPTED_PREDICTION_SHA256 = "e2b2e2f8976b07f3a1d60f659ea32aae285788c6887b02ee2fd05320ec298842"
ACCEPTED_TARGET_SHA256 = "e913d03a972154a4a7ad3eae9963174fb54dbeefd3bc32b542b5cb76bc0ff8aa"
CHECKPOINT_RELATIVE = (
    "tfpd_exploration/results/cross_session_worst_group_m1_matched_erm_full_v1_no_swa/"
    "fold_20120924_matched_erm/checkpoint_best_source_train_loss.pt"
)
CHECKPOINT_SHA256 = "91bd46c0261df141b4fd601eeaa0f105fb637e75aed572d734b94ae0e8b9486e"
CHECKPOINT_STATE_SHA256 = "2d1bdb18e99100ea3c798513f353c5686db81b4687f8ec623e14ac32fc535152"
TARGET_RELATIVE = (
    "SPINT-main/data/000941/sub-MonkeyL-held-in-calib/"
    "sub-MonkeyL-held-in-calib_ses-20120924_behavior+ecephys.nwb"
)
TARGET_SHA256 = "63ee25782c62ff2275dcfbdcaa56552ec4c26fcde00f5a74e5be54785b5c25eb"


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


def _m1_r2(prediction: np.ndarray, target: np.ndarray) -> float:
    from tfpd_exploration.src.cross_session_worst_group_fold20120924_score_v1.physical import (
        variance_weighted_last_bin_r2,
    )
    return variance_weighted_last_bin_r2(prediction, target)


def _load_dataset(root: Path) -> tuple[Any, dict[str, Any]]:
    from tfpd_exploration.src.cross_session_worst_group_v1 import source_reader

    target = root / TARGET_RELATIVE
    _need(target.is_file() and not target.is_symlink() and _file_sha(target) == TARGET_SHA256,
          "M1 target descriptor/body drift")
    runtime = source_reader.load_native_m1_runtime(root)
    recipe = source_reader.NATIVE_M1_READER_RECIPE
    datamodule = runtime.falcon_datamodule_type(**recipe.datamodule_kwargs(data_dir=root))
    record = datamodule.prepare_session_data(
        target,
        runtime.task,
        standardize_covariates=recipe.standardize_covariates,
        covariates_mean=None,
        covariates_std=None,
        use_intertrials=recipe.use_intertrials,
        include_trial_targets=False,
        include_trial_obj_ids=False,
    )
    dataset = runtime.falcon_dataset_type(
        sessions_dict={TARGET_SESSION: record},
        calib_sessions_dict={TARGET_SESSION: record},
        **recipe.dataset_kwargs(),
    )
    trials = np.ascontiguousarray(dataset.calib_trialized_neural_features[TARGET_SESSION], dtype=np.float32)
    _need(trials.shape == (414, 1024, UNITS) and int(dataset.calib_n_trials[TARGET_SESSION]) == SUPPORT_TRIALS,
          "M1 trialized activity/support topology drift")
    _need(len(dataset) == 54849, "M1 accepted query row count drift")
    _need(_file_sha(target) == TARGET_SHA256, "M1 target body changed during parse")
    return dataset, {"target_path": str(target), "target_sha256": TARGET_SHA256, "trials": trials}


def _load_model(root: Path, device: str) -> tuple[Any, str]:
    from tfpd_exploration.src.cross_session_worst_group_v1 import source_physical

    checkpoint = root / CHECKPOINT_RELATIVE
    _need(checkpoint.is_file() and not checkpoint.is_symlink() and _file_sha(checkpoint) == CHECKPOINT_SHA256,
          "M1 selected checkpoint body drift")
    captured: list[Any] = []

    def factory() -> Any:
        model = source_physical.load_exact_m1_spint_model(root).to(device)
        source_physical.materialize_exact_m1_model(model, device=device)
        captured.append(model)
        return model

    observed = source_physical.strict_reload_checkpoint_bytes(
        checkpoint.read_bytes(),
        expected_state_sha256=CHECKPOINT_STATE_SHA256,
        model_factory=factory,
        device=device,
    )
    _need(len(captured) == 1 and observed == CHECKPOINT_STATE_SHA256,
          "M1 strict checkpoint reload drift")
    model = captured[0]
    model.eval()
    return model, observed


def _output_trial_indices(dataset: Any) -> tuple[int, ...]:
    starts = np.asarray(dataset.trial_start_indices[TARGET_SESSION], dtype=np.int64)
    result = []
    for session, start in dataset.window_indices:
        _need(session == TARGET_SESSION, "M1 query session drift")
        output = int(start) + WINDOW - 1
        result.append(int(np.searchsorted(starts, output, side="right") - 1))
    _need(len(result) == len(dataset), "M1 output-trial mapping drift")
    return tuple(result)


def _evaluate_arm(
    *,
    model: Any,
    dataset: Any,
    trial_activity: np.ndarray,
    output_trials: tuple[int, ...],
    arm: ActivityArm,
    device: str,
) -> dict[str, Any]:
    import torch

    total_trials = int(trial_activity.shape[0])
    selections = tuple(
        selection_for_output_trial(
            arm,
            output_trial_index=trial,
            total_trials=total_trials,
            support_trials=SUPPORT_TRIALS,
            growing_cap=GROWING_CAP,
        )
        for trial in output_trials
    )
    prediction = np.empty((len(dataset), OUTPUTS), dtype=np.float32)
    target = np.empty((len(dataset), OUTPUTS), dtype=np.float32)
    forwards = 0
    started = time.monotonic()
    with torch.no_grad():
        for selection, rows in grouped_indices(selections):
            identity = identity_from_raw_trials(
                model, trial_activity, selection, family="m1", device=device,
            )
            for offset in range(0, len(rows), BATCH_SIZE):
                batch_rows = rows[offset:offset + BATCH_SIZE]
                xs = []
                ys = []
                for row in batch_rows:
                    session, start = dataset.window_indices[row]
                    end = int(start) + WINDOW
                    xs.append(dataset.neural_data[session][int(start):end])
                    ys.append(dataset.covariate_data[session][end - 1])
                x = np.ascontiguousarray(np.stack(xs), dtype=np.float32)
                y = np.ascontiguousarray(np.stack(ys), dtype=np.float32)
                output = forward_with_cached_identity(model, x, identity)
                pred = np.ascontiguousarray(output[:, -1, :].detach().cpu().numpy(), dtype=np.float32)
                prediction[np.asarray(batch_rows, dtype=np.int64)] = pred
                target[np.asarray(batch_rows, dtype=np.int64)] = y
                forwards += 1
    elapsed = time.monotonic() - started
    return {
        "arm": arm.value,
        "causal": arm is not ActivityArm.FULL_SESSION_ORACLE,
        "label_free": True,
        "deployment_status": (
            "LABEL_FREE_BUT_NONCAUSAL" if arm is ActivityArm.FULL_SESSION_ORACLE
            else "CAUSAL_CARDINALITY_MATCHED" if arm in {ActivityArm.STATIC_SUPPORT, ActivityArm.ROLLING_FIXED_M}
            else "CAUSAL_CARDINALITY_OOD"
        ),
        "r2": _m1_r2(prediction, target),
        "n_windows": len(dataset),
        "prediction_sha256": array_digest(prediction),
        "target_sha256": array_digest(target),
        "unique_activity_states": len(set(selections)),
        "activity_cardinality_min": min(map(len, selections)),
        "activity_cardinality_max": max(map(len, selections)),
        "forward_batches": forwards,
        "elapsed_seconds": elapsed,
        "_prediction": prediction,
        "_target": target,
    }


def _evaluate_direct_static(model: Any, dataset: Any, trials: np.ndarray, *, device: str) -> dict[str, Any]:
    """Replay the accepted production forward, including repeated B128 calibration."""
    import torch

    prediction = np.empty((len(dataset), OUTPUTS), dtype=np.float32)
    target = np.empty((len(dataset), OUTPUTS), dtype=np.float32)
    support = np.ascontiguousarray(trials[:SUPPORT_TRIALS], dtype=np.float32)
    forwards = 0
    started = time.monotonic()
    with torch.no_grad():
        for start_row in range(0, len(dataset), BATCH_SIZE):
            rows = tuple(range(start_row, min(start_row + BATCH_SIZE, len(dataset))))
            xs, ys = [], []
            for row in rows:
                session, start = dataset.window_indices[row]
                end = int(start) + WINDOW
                xs.append(dataset.neural_data[session][int(start):end])
                ys.append(dataset.covariate_data[session][end - 1])
            x = np.ascontiguousarray(np.stack(xs), dtype=np.float32)
            calibration = np.ascontiguousarray(np.stack([support] * len(rows)), dtype=np.float32)
            output = model(
                torch.as_tensor(x, dtype=torch.float32, device=device),
                calib_trialized_neural_features=torch.as_tensor(calibration, dtype=torch.float32, device=device),
            )
            prediction[np.asarray(rows, dtype=np.int64)] = np.ascontiguousarray(
                output[:, -1, :].detach().cpu().numpy(), dtype=np.float32,
            )
            target[np.asarray(rows, dtype=np.int64)] = np.ascontiguousarray(np.stack(ys), dtype=np.float32)
            forwards += 1
    return {
        "arm": ActivityArm.STATIC_SUPPORT.value,
        "causal": True,
        "label_free": True,
        "deployment_status": "CAUSAL_CARDINALITY_MATCHED",
        "r2": _m1_r2(prediction, target),
        "n_windows": len(dataset),
        "prediction_sha256": array_digest(prediction),
        "target_sha256": array_digest(target),
        "unique_activity_states": 1,
        "activity_cardinality_min": SUPPORT_TRIALS,
        "activity_cardinality_max": SUPPORT_TRIALS,
        "forward_batches": forwards,
        "elapsed_seconds": time.monotonic() - started,
        "forward_path": "accepted_eager_repeated_B128_calibration",
        "_prediction": prediction,
        "_target": target,
    }


def run(root: Path, *, device: str) -> dict[str, Any]:
    from tfpd_exploration.src.cross_session_worst_group_v1 import source_physical
    import torch

    root = Path(root).resolve()
    _need(device.startswith("cuda") and torch.cuda.is_available(), "M1 experiment requires an available CUDA device")
    dataset, authority = _load_dataset(root)
    model, state_before = _load_model(root, device)
    trials = authority.pop("trials")
    output_trials = _output_trial_indices(dataset)
    cached_results = [_evaluate_arm(
        model=model, dataset=dataset, trial_activity=trials, output_trials=output_trials,
        arm=arm, device=device,
    ) for arm in ARM_ORDER]
    direct = _evaluate_direct_static(model, dataset, trials, device=device)
    _need(direct["target_sha256"] == ACCEPTED_TARGET_SHA256
          and direct["prediction_sha256"] == ACCEPTED_PREDICTION_SHA256
          and direct["r2"] == ACCEPTED_R2,
          "M1 static-support replay does not exactly match the accepted score")
    cached = cached_results[0]
    max_abs = float(np.max(np.abs(direct["_prediction"].astype(np.float64) - cached["_prediction"].astype(np.float64))))
    r2_abs = abs(float(direct["r2"]) - float(cached["r2"]))
    _need(max_abs <= 2.0e-6 and r2_abs <= 2.0e-7,
          "M1 cached-identity path exceeds the frozen numeric parity bound")
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
    state_after = source_physical._state_digest(model)
    _need(state_before == state_after, "M1 model state changed during evaluation")
    return {
        "schema": "m1_activity_headroom_fold20120924_v1",
        "status": "COMPLETE_FROZEN_WEIGHT_ACTIVITY_HEADROOM",
        "dataset": "M1",
        "surface": "held_in_calibration_fold_20120924",
        "device": device,
        "authority": {
            **authority,
            "checkpoint_relative": CHECKPOINT_RELATIVE,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "checkpoint_state_sha256": CHECKPOINT_STATE_SHA256,
            "accepted_static_r2": ACCEPTED_R2,
            "accepted_prediction_sha256": ACCEPTED_PREDICTION_SHA256,
            "accepted_target_sha256": ACCEPTED_TARGET_SHA256,
        },
        "contract": {
            "support_trials": SUPPORT_TRIALS,
            "growing_cap": GROWING_CAP,
            "trial_count": int(trials.shape[0]),
            "target_optimizer_backward_update": 0,
            "full_session_oracle_is_noncausal": True,
        },
        "results": results,
        "model_state_before_sha256": state_before,
        "model_state_after_sha256": state_after,
        "model_state_immutable": True,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }


def write_once(path: Path, payload: dict[str, Any]) -> tuple[Path, str]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = canonical_json_bytes(payload)
    digest = hashlib.sha256(body).hexdigest()
    with path.open("xb") as handle:
        handle.write(body)
    sidecar = path.with_name(path.name + ".sha256")
    with sidecar.open("x", encoding="utf-8") as handle:
        handle.write(f"{digest}  {path.name}\n")
    path.chmod(0o444)
    sidecar.chmod(0o444)
    return path, digest
