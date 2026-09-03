"""Fail-closed training and offline dual-selection for H1 CAL-AUG V2."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import r2_score

from src.data.h1_cal_aug_all_source_heldout_v1 import index_heldout_calib
from src.data.h1_m4_eb_pilot import (
    H1_HELDIN_SESSIONS, array_sha256, fit_deployment_carrier,
    index_heldin_calib, interpolate_trial_identity,
)
from src.h1_cal_aug_all_source_m3_deployment_v1_contract import (
    HELDIN_SESSION_TO_FALCON_KEY, HELDOUT_SESSION_TO_FALCON_KEY,
)
from src.h1_cal_aug_all_source_m3_deployment_v1_exec import (
    _load_heldout_record, _load_source_materialization, _minival_paths,
)
from src.h1_cal_aug_m3_aware_dual_selection_v2_contract import (
    BATCHES_PER_EPOCH, BATCH_SIZE, C2_CYCLE, EPOCHS, GLOBAL_STEPS,
    MODEL_PARAMETERS, PREDECESSOR_COMMIT, SCHEMA, V1_BATCH_ORDER_SHA256,
    V1_C1_CHECKPOINT_SHA256, V1_CARRIER_CACHE_SHA256, V1_DROPOUT_SHA256,
    V1_INITIAL_STATE_SHA256, V1_M7_SCHEDULE_SHA256, V1_NORMALIZER_SHA256,
    V1_PLAN_SHA256, V1_SCHEDULE_SHA256, V1_SOURCE_AUTHORITY_SHA256,
    V1_SOURCE_TENSOR_SHA256, prefix_schedule, select_epoch,
)
from src.h1_hc_date_lodo_regen_v1 import (
    _finite_optimizer, _gpu_profile, _new_model, _publish_bytes, model_config,
    publish_json, publish_npz, publish_text, verify_sidecar,
)
from src.h1_m4_cce_contract import NORMALIZER_FLOOR, canonical_sha256, sha256_file, state_hash


STATUS_SOURCE = "PASS_H1_M3_AWARE_V2_FROZEN_SOURCE_AUTHORITY"
STATUS_TRAIN = "PASS_H1_M3_AWARE_V2_C2_ALL_EPOCHS"
STATUS_INTEGRITY = "PASS_H1_M3_AWARE_V2_TRAINING_INTEGRITY"
STATUS_EVALUATION = "PASS_H1_M3_AWARE_V2_DUAL_SELECTION"
STATUS_TERMINAL = "COMPLETE_H1_CAL_AUG_M3_AWARE_DUAL_SELECTION_V2_NO_SUBMISSION"


class V2ExecutionError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise V2ExecutionError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path, schema: str | None = None) -> tuple[dict[str, Any], str]:
    digest = verify_sidecar(path)
    body = json.loads(path.read_text(encoding="utf-8"))
    if schema is not None:
        _need(body.get("schema") == schema, f"schema drift: {path}")
    return body, digest


def create_attempt(result_root: Path, predecessor_root: Path, closure: Mapping[str, str], head: str) -> dict[str, Any]:
    root = result_root.resolve()
    _need(not root.exists(), f"canonical result root is not fresh: {root}")
    _need(head != PREDECESSOR_COMMIT, "successor implementation must be committed before attempt")
    body = {
        "schema": SCHEMA, "artifact": "attempt", "status": "ATTEMPT_BEFORE_DATA_AND_CUDA",
        "created_at_utc": utc_now(), "git_head": head, "predecessor_commit": PREDECESSOR_COMMIT,
        "predecessor_root": str(predecessor_root.resolve()), "closure": dict(closure),
        "code_closure_sha256": canonical_sha256(dict(closure)), "training_arms": ["c2"],
        "interleaved_validation": False, "checkpoint_selection_during_training": False,
        "heldin_calib_opened": 0, "heldin_minival_opened": 0, "heldout_calib_opened": 0,
        "cuda_initialized": False, "docker_builds": 0, "evalai_submissions": 0,
    }
    publish_json(root / "attempt.json", body)
    return body


def load_attempt(result_root: Path) -> dict[str, Any]:
    body, _ = _load_json(result_root.resolve() / "attempt.json", SCHEMA)
    _need(body["status"] == "ATTEMPT_BEFORE_DATA_AND_CUDA", "attempt status drift")
    _need(body["training_arms"] == ["c2"] and not body["interleaved_validation"], "attempt training scope drift")
    _need(body["docker_builds"] == body["evalai_submissions"] == 0, "attempt remote action drift")
    return body


def _validate_v1_receipts(predecessor_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = predecessor_root.resolve()
    authority, authority_sha = _load_json(root / "source_authority/authority.json")
    _need(authority_sha == V1_SOURCE_AUTHORITY_SHA256, "V1 source authority SHA drift")
    expected = {
        "plan_sha256": V1_PLAN_SHA256, "normalizer_sha256": V1_NORMALIZER_SHA256,
        "carrier_cache_sha256": V1_CARRIER_CACHE_SHA256, "schedule_sha256": V1_SCHEDULE_SHA256,
        "source_tensor_surface_sha256": V1_SOURCE_TENSOR_SHA256,
    }
    for key, value in expected.items():
        _need(authority.get(key) == value, f"V1 authority field drift: {key}")
    schedule, schedule_sha = _load_json(root / "source_authority/schedule.json")
    _need(schedule_sha == V1_SCHEDULE_SHA256, "V1 schedule receipt drift")
    _need(schedule["batch_order_sha256"] == V1_BATCH_ORDER_SHA256, "V1 batch order drift")
    _need(schedule["m7_schedule_sha256"] == V1_M7_SCHEDULE_SHA256, "V1 M7 schedule drift")
    terminal, terminal_sha = _load_json(root / "training/c1/terminal.json")
    _need(terminal["checkpoint"]["sha256"] == V1_C1_CHECKPOINT_SHA256, "V1 C1 terminal checkpoint drift")
    _need(terminal["initial_state_sha256"] == V1_INITIAL_STATE_SHA256, "V1 initial state drift")
    _need(terminal["global_step"] == terminal["dropout_probability_count"] == GLOBAL_STEPS, "V1 step drift")
    _need(terminal["dropout_probability_sha256"] == V1_DROPOUT_SHA256, "V1 dropout digest drift")
    checkpoint = root / "training/c1/epoch_049.ckpt"
    _need(verify_sidecar(checkpoint) == V1_C1_CHECKPOINT_SHA256, "V1 C1 checkpoint body drift")
    return authority, {"terminal": terminal, "terminal_sha256": terminal_sha}


def _m3_carrier_cache(dataset: Any, plan: Any, order: np.ndarray, m7: np.ndarray) -> tuple[dict[tuple[str, int], np.ndarray], list[dict[str, Any]]]:
    keys: set[tuple[str, int]] = set()
    for epoch in range(EPOCHS):
        for row, start in zip(order, m7[epoch], strict=True):
            name = dataset.windows[int(row)][0]
            keys.add((name, int(start)))
    result: dict[tuple[str, int], np.ndarray] = {}
    rows = []
    for name, start in sorted(keys):
        record = dataset.records[name]
        trials = tuple(float(value) for value in record.trial_values[start:start + 3])
        _need(len(trials) == 3, f"{name}/{start}: incomplete scheduled M3")
        fitted = fit_deployment_carrier(record, plan, trials)
        raw = np.asarray(fitted["carrier"], np.float64)
        result[(name, start)] = raw
        rows.append({"session": name, "m7_start_index": start, "trial_values": list(trials), "raw_sha256": array_sha256(raw)})
    return result, rows


def prepare_source_authority(data_root: Path, predecessor_root: Path, result_root: Path) -> dict[str, Any]:
    load_attempt(result_root)
    directory = result_root.resolve() / "source_authority"
    _need(not directory.exists(), "V2 source authority already exists")
    v1, v1_train = _validate_v1_receipts(predecessor_root)
    authority, authority_sha, schedule, plan, dataset, order, m7, _v1_prefixes = _load_source_materialization(data_root, predecessor_root)
    _need(authority_sha == V1_SOURCE_AUTHORITY_SHA256 and len(order) // BATCH_SIZE == BATCHES_PER_EPOCH, "V1 materialization drift")
    _need(array_sha256(order) == V1_BATCH_ORDER_SHA256 and array_sha256(m7) == V1_M7_SCHEDULE_SHA256, "V1 schedule tensor drift")
    prefixes = np.asarray(prefix_schedule(), np.int8)
    _need(prefixes.shape == (EPOCHS, BATCHES_PER_EPOCH), "C2 prefix shape drift")
    raw_m3, rows = _m3_carrier_cache(dataset, plan, order, m7)
    normalizer = _load_json(predecessor_root.resolve() / "source_authority/normalizer.json")[0]
    denominator = max(float(normalizer["s_src"]), NORMALIZER_FLOOR)
    sessions = sorted({name for name, _ in raw_m3})
    session_index = {name: index for index, name in enumerate(sessions)}
    keys = sorted(raw_m3)
    normalized = np.stack([np.asarray(raw_m3[key] / denominator, np.float32) for key in keys])
    arrays_sha = publish_npz(
        directory / "c2_schedule_and_m3_carriers.npz", c2_prefixes=prefixes,
        m3_session_index=np.asarray([session_index[key[0]] for key in keys], np.int16),
        m3_start_index=np.asarray([key[1] for key in keys], np.int16), m3_carriers=normalized,
    )
    body = {
        "schema": f"{SCHEMA}_source_authority", "status": STATUS_SOURCE, "created_at_utc": utc_now(),
        "predecessor_commit": PREDECESSOR_COMMIT, "v1_source_authority_sha256": authority_sha,
        "v1_c1_terminal_sha256": v1_train["terminal_sha256"], "v1_c1_checkpoint_sha256": V1_C1_CHECKPOINT_SHA256,
        "v1_plan_sha256": V1_PLAN_SHA256, "v1_normalizer_sha256": V1_NORMALIZER_SHA256,
        "v1_carrier_cache_sha256": V1_CARRIER_CACHE_SHA256, "v1_schedule_sha256": V1_SCHEDULE_SHA256,
        "v1_batch_order_sha256": V1_BATCH_ORDER_SHA256, "v1_m7_schedule_sha256": V1_M7_SCHEDULE_SHA256,
        "v1_source_tensor_surface_sha256": V1_SOURCE_TENSOR_SHA256, "selected_q": v1["selected_q"],
        "selected_lambda": v1["selected_lambda"], "v1_s_src": float(normalizer["s_src"]),
        "c2_cycle": list(C2_CYCLE), "c2_prefix_schedule_sha256": array_sha256(prefixes),
        "m3_carrier_entries": rows, "m3_normalized_carriers_sha256": array_sha256(normalized),
        "m3_carrier_sessions": sessions, "arrays_file_sha256": arrays_sha,
        "steps": GLOBAL_STEPS, "interleaved_validation": False,
        "heldin_calib_recordings_opened": 13, "heldin_minival_recordings_opened": 0,
        "heldout_calib_recordings_opened": 0, "optimizer_steps": 0, "backward_steps": 0,
        "model_updates": 0, "docker_builds": 0, "evalai_submissions": 0,
    }
    publish_json(directory / "authority.json", body)
    return body


def _load_v2_materialization(data_root: Path, predecessor_root: Path, result_root: Path):
    body, body_sha = _load_json(result_root.resolve() / "source_authority/authority.json", f"{SCHEMA}_source_authority")
    _need(body["status"] == STATUS_SOURCE, "V2 source authority status drift")
    _authority, authority_sha, schedule, plan, dataset, order, m7, _ = _load_source_materialization(data_root, predecessor_root)
    _need(authority_sha == body["v1_source_authority_sha256"], "V1 authority reload drift")
    path = result_root.resolve() / "source_authority/c2_schedule_and_m3_carriers.npz"
    _need(verify_sidecar(path) == body["arrays_file_sha256"], "V2 source arrays drift")
    with np.load(path, allow_pickle=False) as arrays:
        prefixes = np.asarray(arrays["c2_prefixes"], np.int8)
        session_idx = np.asarray(arrays["m3_session_index"], np.int16)
        starts = np.asarray(arrays["m3_start_index"], np.int16)
        carriers = np.asarray(arrays["m3_carriers"], np.float32)
    _need(array_sha256(prefixes) == body["c2_prefix_schedule_sha256"], "C2 prefix tensor drift")
    _need(array_sha256(carriers) == body["m3_normalized_carriers_sha256"], "M3 carrier tensor drift")
    sessions = body["m3_carrier_sessions"]
    m3 = {(sessions[int(si)], int(start)): carrier for si, start, carrier in zip(session_idx, starts, carriers, strict=True)}
    return body, body_sha, schedule, plan, dataset, order, m7, prefixes, m3


def _c2_batch(dataset: Any, rows: Sequence[int], starts: Sequence[int], prefix: int, m3: Mapping[tuple[str, int], np.ndarray]):
    _need(prefix in C2_CYCLE and len(rows) == len(starts), "C2 batch schema drift")
    names = {dataset.windows[int(row)][0] for row in rows}
    _need(len(names) == 1, "C2 batch mixes sessions")
    xs, ys, identities, carriers = [], [], [], []
    for row, support_start in zip(rows, starts, strict=True):
        name, query_start = dataset.windows[int(row)]
        support_start = int(support_start)
        values = dataset.records[name].trial_values[support_start:support_start + 7]
        _need(len(values) == 7, "scheduled M7 block incomplete")
        xs.append(dataset.neural[name][query_start:query_start + 700])
        ys.append(dataset.target[name][query_start:query_start + 700])
        identities.append(np.stack([dataset.identity[(name, float(value))] for value in values[:prefix]]))
        carriers.append(m3[(name, support_start)] if prefix == 3 else dataset.carriers[(name, support_start)])
    return tuple(np.ascontiguousarray(np.stack(value), np.float32) for value in (xs, ys, identities, carriers))


def run_c2_training(data_root: Path, predecessor_root: Path, result_root: Path, physical_gpu: int) -> dict[str, Any]:
    cell = result_root.resolve() / "training/c2"
    _need(not cell.exists(), "C2 training cell already exists")
    publish_json(cell / "attempt.json", {
        "schema": SCHEMA, "artifact": "c2_training_attempt", "created_at_utc": utc_now(),
        "physical_gpu": int(physical_gpu), "interleaved_validation": False,
        "checkpoint_selection": False, "heldin_minival_opened": 0, "heldout_calib_opened": 0,
    })
    started = time.monotonic()
    try:
        import torch
        import src.models.components.h1_carrierid_spint as carrier_module
        _need(torch.cuda.is_available(), "C2 training requires CUDA")
        source, source_sha, schedule, _plan, dataset, order, m7, prefixes, m3 = _load_v2_materialization(data_root, predecessor_root, result_root)
        config = {
            "schema": f"{SCHEMA}_c2_config", "arm": "c2", "model": model_config(),
            "cycle": list(C2_CYCLE), "epochs": EPOCHS, "batch_size": BATCH_SIZE,
            "validation": False, "early_stop": False, "warm_start": False,
            "checkpoint_selection": False, "save_all_epoch_checkpoints": True,
        }
        config_sha = publish_json(cell / "config.json", config)
        model = _new_model("cuda:0")
        _need(sum(parameter.numel() for parameter in model.parameters()) == MODEL_PARAMETERS, "model parameter count drift")
        model.train()
        initial_sha = state_hash(model.state_dict())
        _need(initial_sha == V1_INITIAL_STATE_SHA256, "C2 initial-state SHA differs from V1")
        optimizer = torch.optim.Adam(model.parameters(), lr=5.0e-5, weight_decay=0.0)
        original_uniform = carrier_module.random.uniform
        probability_digest = hashlib.sha256()
        probability_count = 0
        def tracked_uniform(low: float, high: float) -> float:
            nonlocal probability_count
            value = original_uniform(low, high)
            probability_digest.update(struct.pack("!d", float(value)))
            probability_count += 1
            return value
        carrier_module.random.uniform = tracked_uniform
        global_step = 0
        prefix_counts = {value: 0 for value in C2_CYCLE}
        epochs = []
        checkpoints = []
        peak_reserved = 0
        try:
            for epoch in range(EPOCHS):
                losses = []
                for batch_index, offset in enumerate(range(0, len(order), BATCH_SIZE)):
                    rows = order[offset:offset + BATCH_SIZE]
                    starts = m7[epoch, offset:offset + BATCH_SIZE]
                    prefix = int(prefixes[epoch, batch_index])
                    neural, target, identity, carrier = _c2_batch(dataset, rows, starts, prefix, m3)
                    optimizer.zero_grad(set_to_none=True)
                    output = model(
                        torch.as_tensor(neural, dtype=torch.float32, device="cuda:0"),
                        calib_trialized_neural_features=torch.as_tensor(identity, dtype=torch.float32, device="cuda:0"),
                        carrier=torch.as_tensor(carrier, dtype=torch.float32, device="cuda:0"),
                    )
                    loss = torch.nn.functional.mse_loss(output[:, -1:, :] / 20.0, torch.as_tensor(target, dtype=torch.float32, device="cuda:0")[:, -1:, :])
                    _need(bool(torch.isfinite(loss)), "nonfinite C2 loss")
                    loss.backward()
                    _need(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()), "nonfinite C2 gradient")
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                    prefix_counts[prefix] += 1
                    global_step += 1
                    peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved()))
                epoch_state_sha = state_hash(model.state_dict())
                payload = {
                    "schema": f"{SCHEMA}_checkpoint", "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "metadata": {"arm": "c2", "epoch_zero_based": epoch, "global_step": global_step,
                        "initial_state_sha256": initial_sha, "state_sha256": epoch_state_sha,
                        "source_authority_sha256": source_sha, "config_sha256": config_sha,
                        "interleaved_validation": False, "checkpoint_selection": False, "warm_start": False,
                        "target_optimizer_steps": 0, "target_backward_steps": 0, "target_model_updates": 0},
                }
                checkpoint_path = cell / f"epoch_{epoch:03d}.ckpt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(payload, checkpoint_path)
                checkpoint_path.chmod(0o444)
                digest = sha256_file(checkpoint_path)
                _publish_bytes(checkpoint_path.with_name(checkpoint_path.name + ".sha256"), f"{digest}  {checkpoint_path.name}\n".encode("ascii"), sidecar=False)
                checkpoints.append({"epoch_zero_based": epoch, "relative": str(checkpoint_path.relative_to(result_root.resolve())), "sha256": digest, "state_sha256": epoch_state_sha, "global_step": global_step})
                row = {"epoch_zero_based": epoch, "steps": BATCHES_PER_EPOCH, "mean_loss": float(np.mean(losses, dtype=np.float64)), "checkpoint_sha256": digest}
                epochs.append(row)
                print(f"EPOCH_END arm=c2 epoch={epoch:03d} steps={BATCHES_PER_EPOCH} mean_loss={row['mean_loss']:.9g}", flush=True)
        finally:
            carrier_module.random.uniform = original_uniform
        _need(global_step == probability_count == GLOBAL_STEPS, "C2 global/dropout count mismatch")
        _need(probability_digest.hexdigest() == V1_DROPOUT_SHA256, "C2 dropout digest differs from V1 C1")
        _need(_finite_optimizer(optimizer), "C2 Adam state nonfinite")
        _need(all(prefix_counts[value] > 0 for value in C2_CYCLE), "C2 cycle coverage incomplete")
        body = {
            "schema": SCHEMA, "status": STATUS_TRAIN, "arm": "c2", "gpu": {**_gpu_profile(physical_gpu), "physical_index": physical_gpu},
            "source_authority_sha256": source_sha, "config_sha256": config_sha,
            "initial_state_sha256": initial_sha, "terminal_state_sha256": checkpoints[-1]["state_sha256"],
            "global_step": global_step, "dropout_probability_count": probability_count,
            "dropout_probability_sha256": probability_digest.hexdigest(), "prefix_step_counts": {str(k): v for k, v in prefix_counts.items()},
            "epochs": epochs, "checkpoints": checkpoints, "peak_memory_reserved_bytes": peak_reserved,
            "training_elapsed_seconds": time.monotonic() - started, "interleaved_validation_runs": 0,
            "checkpoint_rankings_during_training": 0, "checkpoint_selections_during_training": 0,
            "target_optimizer_steps": 0, "target_backward_steps": 0, "target_model_updates": 0,
            "heldin_minival_opened": 0, "heldout_calib_opened": 0, "docker_builds": 0, "evalai_submissions": 0,
            "finished_at_utc": utc_now(),
        }
        publish_json(cell / "terminal.json", body)
        return body
    except BaseException as error:
        try:
            publish_json(cell / "failure.json", {"schema": SCHEMA, "status": "FAIL_C2_NO_AUTOMATIC_RETRY", "error_type": type(error).__name__, "error": str(error), "finished_at_utc": utc_now()})
        except BaseException:
            pass
        raise


def verify_training_integrity(predecessor_root: Path, result_root: Path) -> dict[str, Any]:
    _v1, v1_train = _validate_v1_receipts(predecessor_root)
    terminal, terminal_sha = _load_json(result_root.resolve() / "training/c2/terminal.json", SCHEMA)
    _need(terminal["status"] == STATUS_TRAIN, "C2 terminal incomplete")
    _need(terminal["initial_state_sha256"] == v1_train["terminal"]["initial_state_sha256"] == V1_INITIAL_STATE_SHA256, "initial state integrity failed")
    _need(terminal["global_step"] == terminal["dropout_probability_count"] == GLOBAL_STEPS, "C2 terminal count mismatch")
    _need(terminal["dropout_probability_sha256"] == V1_DROPOUT_SHA256, "C2 terminal dropout mismatch")
    _need(len(terminal["checkpoints"]) == EPOCHS, "C2 checkpoint roster incomplete")
    import torch
    for epoch, row in enumerate(terminal["checkpoints"]):
        _need(row["epoch_zero_based"] == epoch and row["global_step"] == (epoch + 1) * BATCHES_PER_EPOCH, "checkpoint epoch/step drift")
        path = result_root.resolve() / row["relative"]
        _need(verify_sidecar(path) == row["sha256"], f"epoch {epoch}: checkpoint SHA drift")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        meta = payload["metadata"]
        _need(meta["epoch_zero_based"] == epoch and meta["state_sha256"] == row["state_sha256"], f"epoch {epoch}: metadata drift")
        _need(state_hash(payload["state_dict"]) == row["state_sha256"], f"epoch {epoch}: model state drift")
        _need(meta["target_optimizer_steps"] == meta["target_backward_steps"] == meta["target_model_updates"] == 0, "target update drift")
    body = {
        "schema": f"{SCHEMA}_training_integrity", "status": STATUS_INTEGRITY,
        "c2_terminal_sha256": terminal_sha, "v1_c1_terminal_sha256": v1_train["terminal_sha256"],
        "v1_c1_checkpoint_sha256": V1_C1_CHECKPOINT_SHA256, "initial_state_identical": True,
        "source_authority_bound": True, "batch_order_bound": True, "m7_schedule_bound": True,
        "m4_carrier_bound": True, "source_tensor_surface_bound": True,
        "dropout_probability_sequence_matched": True, "global_step": GLOBAL_STEPS,
        "all_epoch_checkpoints_verified": True, "checkpoint_count": EPOCHS,
        "interleaved_validation_runs": 0, "optimizer_steps_on_target": 0,
        "backward_steps_on_target": 0, "model_updates_on_target": 0,
    }
    publish_json(result_root.resolve() / "training/integrity.json", body)
    return body


def _calibration_payloads(data_root: Path, predecessor_root: Path, result_root: Path):
    integrity, integrity_sha = _load_json(result_root.resolve() / "training/integrity.json", f"{SCHEMA}_training_integrity")
    _need(integrity["status"] == STATUS_INTEGRITY, "training integrity must precede target access")
    _authority, authority_sha, _schedule, plan, dataset, _order, _m7, _prefixes = _load_source_materialization(data_root, predecessor_root)
    normalizer = _load_json(predecessor_root.resolve() / "source_authority/normalizer.json")[0]
    denominator = max(float(normalizer["s_src"]), NORMALIZER_FLOOR)
    heldout_paths = index_heldout_calib(data_root)
    payloads: dict[str, dict[str, Any]] = {}
    rows = []
    for session, key in HELDIN_SESSION_TO_FALCON_KEY + HELDOUT_SESSION_TO_FALCON_KEY:
        if session in dataset.records:
            record = dataset.records[session]
            scope = "held-in-calib"
        else:
            record = _load_heldout_record(heldout_paths[session])
            scope = "held-out-calib development/model-selection"
        support = tuple(float(value) for value in record.trial_values[:3])
        _need(len(support) == 3, f"{session}: earliest M3 unavailable")
        identity = np.ascontiguousarray(np.stack([interpolate_trial_identity(record, value) for value in support]), np.float32)
        fitted = fit_deployment_carrier(record, plan, support)
        carrier = np.ascontiguousarray(np.asarray(fitted["carrier"], np.float64) / denominator, np.float32)
        payloads[key] = {"identity": identity, "carrier": carrier, "calibration_trials": list(support), "session": session}
        rows.append({"session": session, "falcon_key": key, "scope": scope, "calibration_trials": list(support), "identity_sha256": array_sha256(identity), "carrier_sha256": array_sha256(carrier)})
    body = {
        "schema": f"{SCHEMA}_m3_calibration_authority", "status": "PASS_FROZEN_M3_CALIBRATION_PAYLOADS",
        "training_integrity_sha256": integrity_sha, "v1_source_authority_sha256": authority_sha,
        "sessions": rows, "heldin_calib_opened": 13, "heldout_calib_opened": 14,
        "optimizer_steps": 0, "backward_steps": 0, "model_updates": 0,
    }
    publish_json(result_root.resolve() / "evaluation/calibration_authority.json", body)
    return payloads


def _surface_metrics(predictions: Mapping[str, Any], targets: Mapping[str, Any], masks: Mapping[str, Any], mapping: Sequence[tuple[str, str]]) -> dict[str, Any]:
    from falcon_challenge.evaluator import FalconEvaluator
    grouped: dict[str, list[str]] = defaultdict(list)
    per_recording = {}
    for _session, key in mapping:
        group = key.split("_set_", 1)[0]
        grouped[group].append(key)
        pred = np.asarray(predictions[key], np.float64)
        target = np.asarray(targets[key], np.float64)
        mask = np.asarray(masks[key], bool).reshape(-1)
        _need(pred.shape == target.shape and pred.ndim == 2 and pred.shape[1] == 7 and len(mask) == len(pred), f"surface shape drift: {key}")
        per_recording[key] = float(r2_score(target[mask], pred[mask], multioutput="variance_weighted"))
    per_session = {}
    for group in sorted(grouped):
        pred = np.concatenate([np.asarray(predictions[key], np.float64)[np.asarray(masks[key], bool).reshape(-1)] for key in grouped[group]])
        target = np.concatenate([np.asarray(targets[key], np.float64)[np.asarray(masks[key], bool).reshape(-1)] for key in grouped[group]])
        per_session[group] = float(r2_score(target, pred, multioutput="variance_weighted"))
    values = np.asarray(list(per_session.values()), np.float64)
    recording_values = np.asarray(list(per_recording.values()), np.float64)
    predictions_all, targets_masked, masks_all, lengths = [], [], [], {}
    for group in sorted(grouped):
        lengths[group] = []
        for key in grouped[group]:
            pred = np.asarray(predictions[key], np.float64)
            target = np.asarray(targets[key], np.float64)
            mask = np.asarray(masks[key], bool).reshape(-1)
            predictions_all.append(pred)
            targets_masked.append(target[mask])
            masks_all.append(mask)
            lengths[group].append(len(pred))
    official = FalconEvaluator.compute_metrics_regression(
        np.concatenate(predictions_all), np.concatenate(targets_masked),
        np.concatenate(masks_all), lengths, verbose=False,
    )
    _need(math.isclose(float(official["R2 Mean"]), float(np.mean(values)), abs_tol=1e-12), "official grouped mean mismatch")
    _need(math.isclose(float(official["R2 Std."]), float(np.std(values, ddof=0)), abs_tol=1e-12), "official grouped std mismatch")
    return {
        "r2_mean": float(official["R2 Mean"]), "r2_std_population": float(official["R2 Std."]),
        "worst_session_r2": float(np.min(values)), "per_session_r2": per_session,
        "per_recording_r2": per_recording, "spint_recording_mean_r2": float(np.mean(recording_values, dtype=np.float64)),
    }


def _evaluate_one(package: Mapping[str, Any], paths: Sequence[Path], mapping: Sequence[tuple[str, str]], device: str) -> tuple[dict[str, Any], str, str]:
    import torch
    import falcon_challenge.evaluator as evaluator_module
    from falcon_challenge.config import FalconConfig, FalconTask
    from falcon_challenge.evaluator import FalconEvaluator
    from third_party.falcon_challenge.h1_carrier_id_spint_decoder import H1CarrierIdSpintDecoder
    old_tqdm = evaluator_module.tqdm
    evaluator_module.tqdm = lambda iterable, *args, **kwargs: iterable
    temp_path = None
    try:
        handle = tempfile.NamedTemporaryFile(prefix="h1_m3aware_v2_", suffix=".pt", delete=False)
        temp_path = Path(handle.name)
        handle.close()
        torch.save(dict(package), temp_path)
        decoder = H1CarrierIdSpintDecoder(FalconConfig(task=FalconTask.h1), temp_path, batch_size=len(paths), device=device)
        before = decoder.model_state_sha256()
        evaluator = FalconEvaluator(eval_remote=False, split="h1", verbose=False, dataloader_workers=0)
        predictions, targets, masks, _compute, _neural = evaluator.predict_files(decoder, list(paths))
        after = decoder.model_state_sha256()
        _need(before == after == package["model_state_sha256"], "validator changed model state")
        return _surface_metrics(predictions, targets, masks, mapping), before, after
    finally:
        evaluator_module.tqdm = old_tqdm
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _package_for_checkpoint(checkpoint_path: Path, checkpoint_sha: str, source_sha: str, payloads: Mapping[str, Any], arm: str) -> dict[str, Any]:
    import torch
    from third_party.falcon_challenge.h1_carrier_id_spint_decoder import PACKAGE_SCHEMA
    _need(verify_sidecar(checkpoint_path) == checkpoint_sha, "validator checkpoint SHA drift")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_sha = state_hash(checkpoint["state_dict"])
    return {"schema": PACKAGE_SCHEMA, "task": "h1", "arm": arm, "state_dict": checkpoint["state_dict"],
            "model_kwargs": model_config()["model_kwargs"], "model_state_sha256": model_sha,
            "checkpoint_sha256": checkpoint_sha, "source_authority_sha256": source_sha,
            "window_size": 700, "prediction_divisor": 20.0, "sessions": dict(payloads),
            "optimizer_steps": 0, "backward_steps": 0, "model_updates": 0}


def run_offline_validation(data_root: Path, predecessor_root: Path, result_root: Path, device: str = "cuda:0") -> dict[str, Any]:
    integrity, integrity_sha = _load_json(result_root.resolve() / "training/integrity.json", f"{SCHEMA}_training_integrity")
    _need(integrity["status"] == STATUS_INTEGRITY, "training integrity not passed")
    payloads = _calibration_payloads(data_root, predecessor_root, result_root)
    hi_paths = _minival_paths(data_root)
    heldout = index_heldout_calib(data_root)
    ho_paths = tuple(heldout[session] for session, _key in HELDOUT_SESSION_TO_FALCON_KEY)
    source_sha = verify_sidecar(result_root.resolve() / "source_authority/authority.json")
    training = _load_json(result_root.resolve() / "training/c2/terminal.json", SCHEMA)[0]
    candidates = [("v1_c1_e49", 49, predecessor_root.resolve() / "training/c1/epoch_049.ckpt", V1_C1_CHECKPOINT_SHA256)]
    candidates.extend(("c2", row["epoch_zero_based"], result_root.resolve() / row["relative"], row["sha256"]) for row in training["checkpoints"])
    curve_hi, curve_ho, baseline = [], [], None
    for kind, epoch, checkpoint_path, checkpoint_sha in candidates:
        package = _package_for_checkpoint(checkpoint_path, checkpoint_sha, source_sha, payloads, kind)
        hi, hi_before, hi_after = _evaluate_one(package, hi_paths, HELDIN_SESSION_TO_FALCON_KEY, device)
        ho, ho_before, ho_after = _evaluate_one(package, ho_paths, HELDOUT_SESSION_TO_FALCON_KEY, device)
        row = {
            "schema": f"{SCHEMA}_candidate_metrics", "candidate": kind, "epoch_zero_based": epoch,
            "checkpoint_sha256": checkpoint_sha,
            "val_hi_m3_official/r2_mean": hi["r2_mean"], "val_hi_m3_official/r2_std_population": hi["r2_std_population"],
            "hi": hi, "val_ho_m3_grouped/r2_mean": ho["r2_mean"],
            "val_ho_m3_grouped/r2_std_population": ho["r2_std_population"],
            "val_ho_m3_spint14/r2_mean": ho["spint_recording_mean_r2"], "ho": ho,
            "model_state_before_after_identical": hi_before == hi_after == ho_before == ho_after,
            "optimizer_steps": 0, "backward_steps": 0, "model_updates": 0,
        }
        if kind == "c2":
            hi_row = {"epoch_zero_based": epoch, "val_hi_m3_official/r2_mean": hi["r2_mean"], "worst_session_r2": hi["worst_session_r2"], "session_std_population": hi["r2_std_population"], "checkpoint_sha256": checkpoint_sha, "per_session_r2": hi["per_session_r2"]}
            ho_row = {"epoch_zero_based": epoch, "val_ho_m3_grouped/r2_mean": ho["r2_mean"], "worst_session_r2": ho["worst_session_r2"], "session_std_population": ho["r2_std_population"], "checkpoint_sha256": checkpoint_sha, "per_session_r2": ho["per_session_r2"], "val_ho_m3_spint14/r2_mean": ho["spint_recording_mean_r2"], "per_recording_r2": ho["per_recording_r2"]}
            curve_hi.append(hi_row); curve_ho.append(ho_row)
            publish_json(result_root.resolve() / f"evaluation/epochs/epoch_{epoch:03d}.json", row)
        else:
            baseline = row
            publish_json(result_root.resolve() / "evaluation/v1_c1_e49.json", row)
        print(f"EVAL_END candidate={kind} epoch={epoch:03d} hi={hi['r2_mean']:.9g} ho={ho['r2_mean']:.9g}", flush=True)
    _need(baseline is not None and len(curve_hi) == len(curve_ho) == EPOCHS, "offline curve incomplete")
    selected_hi = dict(select_epoch(curve_hi, "val_hi_m3_official/r2_mean"))
    selected_ho = dict(select_epoch(curve_ho, "val_ho_m3_grouped/r2_mean"))
    hi_selection = {"schema": f"{SCHEMA}_selection", "status": "SEALED_C2_HI_SELECTION", "surface": "HI-M3", "selected": selected_hi, "tie_break": ["higher mean", "higher worst-session", "lower population std", "earlier epoch"], "retraining": False, "finetuning": False}
    ho_selection = {"schema": f"{SCHEMA}_selection", "status": "SEALED_C2_HO_SELECTION", "surface": "HO-M3 development/model-selection; not untouched held-out generalization", "selected": selected_ho, "selection_metric": "val_ho_m3_grouped/r2_mean", "secondary_not_used": "val_ho_m3_spint14/r2_mean", "tie_break": ["higher mean", "higher worst-session", "lower population std", "earlier epoch"], "retraining": False, "finetuning": False}
    hi_sha = publish_json(result_root.resolve() / "selection/c2_hi.json", hi_selection)
    ho_sha = publish_json(result_root.resolve() / "selection/c2_ho.json", ho_selection)
    x = np.asarray([row["val_hi_m3_official/r2_mean"] for row in curve_hi], np.float64)
    y = np.asarray([row["val_ho_m3_grouped/r2_mean"] for row in curve_ho], np.float64)
    from scipy.stats import pearsonr, spearmanr
    c2_e49_hi, c2_e49_ho = curve_hi[49], curve_ho[49]
    body = {
        "schema": f"{SCHEMA}_dual_selection", "status": STATUS_EVALUATION,
        "training_integrity_sha256": integrity_sha, "v1_c1_e49": baseline,
        "c2_e49": {"hi": c2_e49_hi, "ho": c2_e49_ho}, "c2_hi_selected": selected_hi,
        "c2_ho_selected": selected_ho, "hi_curve": curve_hi, "ho_curve": curve_ho,
        "curve_correlation": {"pearson": float(pearsonr(x, y).statistic), "spearman": float(spearmanr(x, y).statistic)},
        "deltas": {
            "c2_e49_minus_v1_c1_e49_hi": c2_e49_hi["val_hi_m3_official/r2_mean"] - baseline["val_hi_m3_official/r2_mean"],
            "c2_e49_minus_v1_c1_e49_ho": c2_e49_ho["val_ho_m3_grouped/r2_mean"] - baseline["val_ho_m3_grouped/r2_mean"],
            "c2_hi_minus_c2_e49": selected_hi["val_hi_m3_official/r2_mean"] - c2_e49_hi["val_hi_m3_official/r2_mean"],
            "c2_ho_minus_c2_e49": selected_ho["val_ho_m3_grouped/r2_mean"] - c2_e49_ho["val_ho_m3_grouped/r2_mean"],
            "c2_hi_minus_v1_c1_e49": selected_hi["val_hi_m3_official/r2_mean"] - baseline["val_hi_m3_official/r2_mean"],
            "c2_ho_minus_v1_c1_e49": selected_ho["val_ho_m3_grouped/r2_mean"] - baseline["val_ho_m3_grouped/r2_mean"],
        },
        "selection_c2_hi_sha256": hi_sha, "selection_c2_ho_sha256": ho_sha,
        "target_access": {"heldin_calib_recordings": 13, "heldin_minival_recordings_per_candidate": 13,
            "heldout_calib_recordings_for_calibration": 14, "heldout_calib_recordings_per_candidate": 14,
            "candidate_count": 51, "ho_role": "development/model-selection; not untouched held-out generalization"},
        "optimizer_steps": 0, "backward_steps": 0, "model_updates": 0,
        "docker_builds": 0, "evalai_submissions": 0,
    }
    publish_json(result_root.resolve() / "evaluation/dual_selection.json", body)
    publish_json(result_root.resolve() / "evaluation/target_access.json", {"schema": f"{SCHEMA}_target_access", **body["target_access"], "optimizer_steps": 0, "backward_steps": 0, "model_updates": 0, "evalai_submissions": 0})
    return body


def verify_terminal(result_root: Path) -> dict[str, Any]:
    root = result_root.resolve()
    integrity, integrity_sha = _load_json(root / "training/integrity.json", f"{SCHEMA}_training_integrity")
    evaluation, evaluation_sha = _load_json(root / "evaluation/dual_selection.json", f"{SCHEMA}_dual_selection")
    hi, hi_sha = _load_json(root / "selection/c2_hi.json", f"{SCHEMA}_selection")
    ho, ho_sha = _load_json(root / "selection/c2_ho.json", f"{SCHEMA}_selection")
    _need(integrity["status"] == STATUS_INTEGRITY and evaluation["status"] == STATUS_EVALUATION, "terminal predecessor incomplete")
    _need(evaluation["selection_c2_hi_sha256"] == hi_sha and evaluation["selection_c2_ho_sha256"] == ho_sha, "selection receipt drift")
    _need(dict(select_epoch(evaluation["hi_curve"], "val_hi_m3_official/r2_mean")) == hi["selected"], "HI selection recomputation drift")
    _need(dict(select_epoch(evaluation["ho_curve"], "val_ho_m3_grouped/r2_mean")) == ho["selected"], "HO selection recomputation drift")
    body = {
        "schema": SCHEMA, "status": STATUS_TERMINAL, "finished_at_utc": utc_now(),
        "training_integrity_sha256": integrity_sha, "dual_selection_sha256": evaluation_sha,
        "c2_hi_selection_sha256": hi_sha, "c2_ho_selection_sha256": ho_sha,
        "c2_hi_epoch": hi["selected"]["epoch_zero_based"], "c2_ho_epoch": ho["selected"]["epoch_zero_based"],
        "v1_c1_e49_checkpoint_sha256": V1_C1_CHECKPOINT_SHA256,
        "c2_e49_checkpoint_sha256": evaluation["c2_e49"]["hi"]["checkpoint_sha256"],
        "optimizer_steps_on_target": 0, "backward_steps_on_target": 0, "model_updates_on_target": 0,
        "docker_builds": 0, "evalai_submissions": 0, "automatic_successor": False,
        "stop_boundary": "local dual-selection sealed; no Docker or EvalAI",
    }
    terminal_sha = publish_json(root / "terminal.json", body)
    record = (
        "# H1 CAL-AUG M3-Aware Dual-Selection V2\n\n"
        f"- Status: `{STATUS_TERMINAL}`\n"
        f"- C2-HI epoch: `{body['c2_hi_epoch']}`\n"
        f"- C2-HO epoch: `{body['c2_ho_epoch']}`\n"
        "- HO-M3 is a development/model-selection surface, not untouched held-out generalization.\n"
        "- Target optimizer/backward/model updates: `0/0/0`.\n"
        "- Docker builds / EvalAI submissions: `0/0`.\n"
        f"- Terminal SHA-256: `{terminal_sha}`\n"
    )
    publish_text(root / "EXPERIMENT_RECORD.md", record)
    return body


__all__ = (
    "create_attempt", "load_attempt", "prepare_source_authority", "run_c2_training",
    "run_offline_validation", "verify_terminal", "verify_training_integrity",
)
