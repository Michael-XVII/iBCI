"""Formal local execution for H1 all-source M3 deployment finalization."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import random
import struct
import subprocess
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.h1_cal_aug_all_source_heldout_v1 import H1_HELDOUT_SESSIONS, index_heldout_calib
from src.data.h1_m4_eb_pilot import (
    EXPECTED_NEURONS,
    H1_HELDIN_SESSIONS,
    H1PilotRecord,
    _ordered_eval_trials,
    _trial_blocks,
    array_sha256,
    carrier_sha256,
    fit_deployment_carrier,
    fit_frozen_carrier,
    index_heldin_calib,
    interpolate_trial_identity,
    load_record,
    session_date,
    session_from_path,
)
from src.h1_cal_aug_all_source_m3_deployment_v1_contract import (
    ARMS,
    BATCH_SIZE,
    C1_CYCLE,
    CHECKPOINT_EPOCH_ZERO_BASED,
    EPOCHS,
    HELDIN_SESSION_TO_FALCON_KEY,
    HELDOUT_SESSION_TO_FALCON_KEY,
    PREDICTION_DIVISOR,
    SCHEMA,
    WINDOW_SIZE,
)
from src.h1_cal_aug_prefix_cycle_v1 import _batch
from src.h1_hc_date_lodo_regen_v1 import (
    BATCH_SIZE as REGEN_BATCH_SIZE,
    MODEL_PARAMETERS,
    Q_GRID,
    LAMBDA_GRID,
    RegenPlan,
    SourceDataset,
    _build_schedule,
    _date_selection,
    _finite_optimizer,
    _gpu_profile,
    _legal_starts,
    _make_final_plan,
    _new_model,
    _publish_bytes,
    _window_hash,
    model_config,
    publish_json,
    publish_npy,
    publish_npz,
    publish_text,
    variance_weighted_r2,
    verify_sidecar,
)
from src.h1_m4_cce_contract import NORMALIZER_FLOOR, NORMALIZER_FORMULA, canonical_sha256, sha256_file, state_hash


STATUS_SOURCE = "PASS_H1_ALL_SOURCE_M3_DEPLOYMENT_SOURCE_AUTHORITY"
STATUS_ARM = "PASS_H1_ALL_SOURCE_M3_DEPLOYMENT_ARM_EPOCH49"
STATUS_SMOKE = "PASS_H1_ALL_SOURCE_M3_DEPLOYMENT_PAIRED_SMOKE"
STATUS_PAIR = "PASS_H1_ALL_SOURCE_M3_DEPLOYMENT_PAIR_INTEGRITY"
STATUS_PACKAGES = "PASS_H1_ALL_SOURCE_M3_DEPLOYMENT_PACKAGES"
STATUS_MINIVAL = "PASS_LOCAL_HELDIN_MINIVAL_DEPLOYMENT_SANITY"
STATUS_REHEARSAL = "PASS_H1_ALL_SOURCE_M3_DEPLOYMENT_PACKAGE_REHEARSAL"
STATUS_TERMINAL = "COMPLETE_LOCAL_H1_ALL_SOURCE_M3_DEPLOYMENT_READY_NO_EVALAI_SUBMISSION"
ALL_SOURCE_DOMAIN = "h1_all_source_13"


class AllSourceExecutionError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise AllSourceExecutionError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path, schema: str | None = None) -> tuple[dict[str, Any], str]:
    digest = verify_sidecar(path)
    body = json.loads(path.read_text(encoding="utf-8"))
    if schema is not None:
        _need(body.get("schema") == schema, f"schema drift: {path}")
    return body, digest


def create_attempt(result_root: Path, closure: Mapping[str, str], head: str) -> dict[str, Any]:
    root = result_root.resolve()
    _need(not root.exists(), f"canonical result root is not fresh: {root}")
    body = {
        "schema": SCHEMA,
        "artifact": "attempt",
        "status": "ATTEMPT_BEFORE_NWB_DATA_AND_CUDA",
        "created_at_utc": utc_now(),
        "head": str(head),
        "closure": dict(closure),
        "code_closure_sha256": canonical_sha256(dict(closure)),
        "heldin_calib_recordings_opened": 0,
        "heldin_minival_recordings_opened": 0,
        "heldout_calib_recordings_opened": 0,
        "evalai_test_recordings_opened": 0,
        "cuda_initialized": False,
        "evalai_submissions": 0,
    }
    publish_json(root / "attempt.json", body)
    return body


def load_attempt(result_root: Path) -> dict[str, Any]:
    body, _ = _load_json(result_root.resolve() / "attempt.json", SCHEMA)
    _need(body.get("status") == "ATTEMPT_BEFORE_NWB_DATA_AND_CUDA", "attempt status drift")
    for field in ("heldin_calib_recordings_opened", "heldin_minival_recordings_opened", "heldout_calib_recordings_opened", "evalai_test_recordings_opened", "evalai_submissions"):
        _need(body.get(field) == 0, f"attempt access drift: {field}")
    return body


def _load_all_source_records(data_root: Path) -> dict[str, H1PilotRecord]:
    paths = index_heldin_calib(data_root)
    records = {name: load_record(paths[name]) for name in H1_HELDIN_SESSIONS}
    _need(tuple(records) == H1_HELDIN_SESSIONS, "all-source held-in roster drift")
    return records


def _m7_schedule(dataset: SourceDataset, order: np.ndarray) -> np.ndarray:
    _need(len(order) % BATCH_SIZE == 0, "all-source batch order is incomplete")
    names = np.asarray([dataset.windows[int(row)][0] for row in order], dtype=object)
    positions = np.arange(len(order), dtype=np.int64)
    schedule = np.empty((EPOCHS, len(order)), dtype=np.int16)
    for name, record in dataset.records.items():
        mask = names == name
        legal = len(record.trial_values) - 7 + 1
        _need(legal >= 2, f"{name}: M7 must leave a later trial")
        local = positions[mask]
        session_token = int.from_bytes(hashlib.sha256(f"{SCHEMA}|m7|{name}".encode()).digest()[:8], "big")
        for epoch in range(EPOCHS):
            values = (session_token + (epoch + 1) * 1_000_003 + (local + 1) * 97_409 + order[mask] * 65_537) % legal
            schedule[epoch, mask] = values.astype(np.int16)
    return schedule


def _prefix_schedule(batches_per_epoch: int) -> np.ndarray:
    result = np.empty((EPOCHS, batches_per_epoch), dtype=np.int8)
    for epoch in range(EPOCHS):
        token = hashlib.sha256(f"{SCHEMA}|prefix|{ALL_SOURCE_DOMAIN}|{epoch}".encode()).digest()
        offset = int.from_bytes(token[:8], "big") % len(C1_CYCLE)
        row = np.asarray([C1_CYCLE[(index + offset) % 3] for index in range(batches_per_epoch)], dtype=np.int8)
        counts = [int(np.sum(row == value)) for value in C1_CYCLE]
        _need(max(counts) - min(counts) <= 1 and 3 not in row, "C1 prefix cycle balance drift")
        result[epoch] = row
    return result


def prepare_source_authority(data_root: Path, result_root: Path) -> dict[str, Any]:
    load_attempt(result_root)
    root = result_root.resolve()
    directory = root / "source_authority"
    _need(not directory.exists(), "all-source authority already exists")
    started = time.monotonic()
    predecessor = validate_predecessors(result_root)
    predecessor_sha = verify_sidecar(root / "predecessor_authority.json")
    records = _load_all_source_records(data_root)
    selected, candidates = _date_selection(records, ALL_SOURCE_DOMAIN)
    selection = {
        "schema": f"{SCHEMA}_selection",
        "status": "PASS_SOURCE_INNER_DATE_LODO_SELECTION",
        "candidate_grid": {"q": list(Q_GRID), "lambda": list(LAMBDA_GRID)},
        "metric": "float64 variance-weighted R2; equal-recording within source date; equal-date governing",
        "tie_break": ["higher equal_date_mean_r2", "higher worst_date_r2", "smaller q", "larger lambda"],
        "candidates": candidates,
        "selected": selected,
        "heldin_minival_recordings_opened": 0,
        "heldout_calib_recordings_opened": 0,
        "evalai_test_recordings_opened": 0,
    }
    selection_sha = publish_json(directory / "selection.json", selection)
    plan = _make_final_plan(records, ALL_SOURCE_DOMAIN, selected, selection_sha)
    plan_arrays_sha = publish_npz(
        directory / "plan.npz", mean=plan.mean, scale=plan.scale, pcs=plan.pcs,
        q=np.asarray(plan.q, np.int64), **{"lambda": np.asarray(plan.ridge_lambda, np.float64)},
        U=plan.U, mu=plan.mu, tau2=np.asarray(plan.tau2, np.float64),
    )
    plan_body = {
        "schema": f"{SCHEMA}_plan",
        "source_sessions": list(plan.source_sessions),
        "source_input_sha256": list(plan.source_input_sha256),
        "q": plan.q, "lambda": plan.ridge_lambda, "tau2": plan.tau2,
        "selection_sha256": selection_sha, "transform_sha256": plan.transform_sha256,
        "arrays_file_sha256": plan_arrays_sha,
        "array_sha256": {name: array_sha256(value) for name, value in {"mean": plan.mean, "scale": plan.scale, "pcs": plan.pcs, "U": plan.U, "mu": plan.mu}.items()},
    }
    plan_sha = publish_json(directory / "plan.json", plan_body)
    entries: list[dict[str, Any]] = []
    carriers: list[np.ndarray] = []
    starts_by_session: dict[str, tuple[int, ...]] = {}
    for name, record in records.items():
        starts = _legal_starts(record)
        starts_by_session[name] = starts
        for start in starts:
            values = tuple(float(value) for value in record.trial_values[start:start + 4])
            carrier = np.asarray(fit_frozen_carrier(record, plan, values)["carrier"], dtype=np.float64)
            entries.append({"session": name, "start_index": start, "trial_values": list(values), "carrier_sha256": carrier_sha256(carrier)})
            carriers.append(carrier)
    carrier_array = np.stack(carriers)
    cache_file_sha = publish_npz(directory / "carrier_cache.npz", carriers=carrier_array)
    cache_body = {
        "schema": f"{SCHEMA}_carrier_cache", "transform_sha256": plan.transform_sha256,
        "shape": list(carrier_array.shape), "entries": entries,
        "tensor_sha256": array_sha256(carrier_array), "arrays_file_sha256": cache_file_sha,
    }
    cache_body["cache_sha256"] = canonical_sha256(cache_body)
    cache_sha = publish_json(directory / "carrier_cache.json", cache_body)
    scalar = float(np.sqrt(np.mean(np.square(carrier_array, dtype=np.float64), dtype=np.float64)))
    _need(math.isfinite(scalar) and scalar >= 0.0, "all-source carrier normalizer invalid")
    normalized = carrier_array / max(scalar, NORMALIZER_FLOOR)
    normalizer_body = {
        "schema": f"{SCHEMA}_normalizer", "formula": NORMALIZER_FORMULA,
        "floor": NORMALIZER_FLOOR, "s_src": scalar, "cache_sha256": cache_sha,
        "normalized_tensor_sha256": array_sha256(normalized),
    }
    normalizer_sha = publish_json(directory / "normalizer.json", normalizer_body)
    order, _m4_schedule, windows = _build_schedule(records, starts_by_session, ALL_SOURCE_DOMAIN)
    _need(REGEN_BATCH_SIZE == BATCH_SIZE, "batch-size constant drift")
    dataset = SourceDataset(records, cache_body, carrier_array, scalar)
    m7 = _m7_schedule(dataset, order)
    prefixes = _prefix_schedule(len(order) // BATCH_SIZE)
    schedule_file_sha = publish_npz(directory / "schedule.npz", batch_order=order, m7_starts=m7, c1_prefixes=prefixes)
    tensor_surface = {
        name: {"neural_sha256": array_sha256(record.neural), "target_sha256": array_sha256(record.velocity)}
        for name, record in records.items()
    }
    tensor_surface_sha = canonical_sha256(tensor_surface)
    schedule_body = {
        "schema": f"{SCHEMA}_schedule", "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "batches_per_epoch": int(len(order) // BATCH_SIZE), "steps_per_arm": int(EPOCHS * len(order) // BATCH_SIZE),
        "window_indices_sha256": _window_hash(windows), "source_tensor_surface": tensor_surface,
        "source_tensor_surface_sha256": tensor_surface_sha,
        "batch_order_sha256": array_sha256(order), "m7_schedule_sha256": array_sha256(m7),
        "c1_prefix_schedule_sha256": array_sha256(prefixes), "schedule_file_sha256": schedule_file_sha,
    }
    schedule_sha = publish_json(directory / "schedule.json", schedule_body)
    authority = {
        "schema": f"{SCHEMA}_source_authority", "status": STATUS_SOURCE,
        "created_at_utc": utc_now(), "source_sessions": list(records),
        "predecessor_authority_sha256": predecessor_sha, "m3_interpretation": predecessor["m3_interpretation"],
        "source_files": [{"session": name, "filename": records[name].path.name, "bytes": records[name].path.stat().st_size, "sha256": records[name].input_sha256} for name in records],
        "selected_q": plan.q, "selected_lambda": plan.ridge_lambda,
        "selection_sha256": selection_sha, "plan_sha256": plan_sha,
        "carrier_cache_sha256": cache_sha, "normalizer_sha256": normalizer_sha,
        "schedule_sha256": schedule_sha, "source_tensor_surface_sha256": tensor_surface_sha,
        "steps_per_arm": schedule_body["steps_per_arm"], "batches_per_epoch": schedule_body["batches_per_epoch"],
        "heldin_calib_recordings_opened": 13, "heldin_calib_bytes_read": sum(row["bytes"] for row in [{"bytes": records[name].path.stat().st_size} for name in records]),
        "heldin_minival_recordings_opened": 0, "heldout_calib_recordings_opened": 0,
        "evalai_test_recordings_opened": 0, "evalai_submissions": 0,
        "elapsed_seconds": time.monotonic() - started,
    }
    publish_json(directory / "authority.json", authority)
    return authority


def _load_source_materialization(data_root: Path, result_root: Path):
    directory = result_root.resolve() / "source_authority"
    authority, authority_sha = _load_json(directory / "authority.json", f"{SCHEMA}_source_authority")
    _need(authority.get("status") == STATUS_SOURCE, "source authority status drift")
    _need(authority.get("heldin_minival_recordings_opened") == authority.get("heldout_calib_recordings_opened") == authority.get("evalai_test_recordings_opened") == 0, "source authority leaked target/minival")
    records = _load_all_source_records(data_root)
    _need(tuple(records) == tuple(authority["source_sessions"]), "source reload roster drift")
    expected = {row["session"]: row["sha256"] for row in authority["source_files"]}
    _need(all(record.input_sha256 == expected[name] for name, record in records.items()), "source input SHA drift")
    plan_body, plan_sha = _load_json(directory / "plan.json", f"{SCHEMA}_plan")
    _need(plan_sha == authority["plan_sha256"] and verify_sidecar(directory / "plan.npz") == plan_body["arrays_file_sha256"], "plan authority drift")
    with np.load(directory / "plan.npz", allow_pickle=False) as arrays:
        plan = RegenPlan(
            ALL_SOURCE_DOMAIN, tuple(records), tuple(records[name].input_sha256 for name in records),
            np.asarray(arrays["mean"], np.float64), np.asarray(arrays["scale"], np.float64),
            np.asarray(arrays["pcs"], np.float64), int(arrays["q"].item()), float(arrays["lambda"].item()),
            np.asarray(arrays["U"], np.float64), np.asarray(arrays["mu"], np.float64), float(arrays["tau2"].item()),
            str(plan_body["selection_sha256"]), str(plan_body["transform_sha256"]),
        )
    _need(plan.q == authority["selected_q"] and plan.ridge_lambda == authority["selected_lambda"], "selected plan scalar drift")
    cache, cache_sha = _load_json(directory / "carrier_cache.json", f"{SCHEMA}_carrier_cache")
    _need(cache_sha == authority["carrier_cache_sha256"] and verify_sidecar(directory / "carrier_cache.npz") == cache["arrays_file_sha256"], "carrier cache drift")
    with np.load(directory / "carrier_cache.npz", allow_pickle=False) as arrays:
        carriers = np.asarray(arrays["carriers"], np.float64)
    _need(array_sha256(carriers) == cache["tensor_sha256"], "carrier tensor drift")
    normalizer, normalizer_sha = _load_json(directory / "normalizer.json", f"{SCHEMA}_normalizer")
    _need(normalizer_sha == authority["normalizer_sha256"], "normalizer authority drift")
    schedule, schedule_sha = _load_json(directory / "schedule.json", f"{SCHEMA}_schedule")
    _need(schedule_sha == authority["schedule_sha256"] and verify_sidecar(directory / "schedule.npz") == schedule["schedule_file_sha256"], "schedule authority drift")
    with np.load(directory / "schedule.npz", allow_pickle=False) as arrays:
        order = np.asarray(arrays["batch_order"], np.int64)
        m7 = np.asarray(arrays["m7_starts"], np.int16)
        prefixes = np.asarray(arrays["c1_prefixes"], np.int8)
    _need(array_sha256(order) == schedule["batch_order_sha256"] and array_sha256(m7) == schedule["m7_schedule_sha256"] and array_sha256(prefixes) == schedule["c1_prefix_schedule_sha256"], "schedule tensor drift")
    return authority, authority_sha, schedule, plan, SourceDataset(records, cache, carriers, float(normalizer["s_src"])), order, m7, prefixes


def common_config() -> dict[str, Any]:
    return {"schema": f"{SCHEMA}_common_config", "model": model_config(), "model_parameters": MODEL_PARAMETERS, "t0_identity_prefix": 7, "c1_identity_cycle": list(C1_CYCLE), "training_carrier_trials": 4, "deployment_trials": 3, "epochs": EPOCHS, "checkpoint_epoch_zero_based": CHECKPOINT_EPOCH_ZERO_BASED, "batch_size": BATCH_SIZE, "prediction_divisor": PREDICTION_DIVISOR, "validation_selection": False, "early_stopping": False, "warm_start": False, "target_fitting": False}


def run_arm(data_root: Path, result_root: Path, arm: str, physical_gpu: int, *, smoke: bool = False, max_steps: int = 12) -> dict[str, Any]:
    _need(arm in ARMS, "unknown all-source arm")
    cell = result_root.resolve() / ("smoke" if smoke else "training") / arm
    _need(not cell.exists(), f"arm cell already exists: {cell}")
    publish_json(cell / "attempt.json", {
        "schema": SCHEMA, "artifact": "arm_attempt", "arm": arm, "smoke": smoke,
        "physical_gpu": int(physical_gpu), "created_at_utc": utc_now(),
        "heldin_minival_recordings_opened": 0, "heldout_calib_recordings_opened": 0,
        "evalai_test_recordings_opened": 0, "warm_start": False,
    })
    started = time.monotonic()
    try:
        import torch
        import src.models.components.h1_carrierid_spint as carrier_module
        _need(torch.cuda.is_available(), "all-source arm requires CUDA")
        attempt = load_attempt(result_root)
        authority, authority_sha, schedule, _plan, dataset, order, m7, prefixes = _load_source_materialization(data_root, result_root)
        config = common_config()
        common_sha = canonical_sha256(config)
        config_sha = publish_json(cell / "config.json", {**config, "arm": arm, "common_config_sha256": common_sha})
        model = _new_model("cuda:0")
        _need(sum(parameter.numel() for parameter in model.parameters()) == MODEL_PARAMETERS == 10_947_836, "model parameter count drift")
        model.train()
        initial_sha = state_hash(model.state_dict())
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
        prefix_counts = {4: 0, 5: 0, 7: 0}
        epoch_rows = []
        loss_first = None
        loss_last = None
        peak_reserved = 0
        training_started = time.monotonic()
        try:
            epochs = 1 if smoke else EPOCHS
            for epoch in range(epochs):
                losses = []
                for batch_index, offset in enumerate(range(0, len(order), BATCH_SIZE)):
                    if smoke and global_step >= max_steps:
                        break
                    rows = order[offset:offset + BATCH_SIZE]
                    starts = m7[epoch, offset:offset + BATCH_SIZE]
                    prefix = 7 if arm == "t0" else int(prefixes[epoch, batch_index])
                    neural, target, identity, carrier, _session = _batch(dataset, rows, starts, prefix)
                    optimizer.zero_grad(set_to_none=True)
                    output = model(
                        torch.as_tensor(neural, dtype=torch.float32, device="cuda:0"),
                        calib_trialized_neural_features=torch.as_tensor(identity, dtype=torch.float32, device="cuda:0"),
                        carrier=torch.as_tensor(carrier, dtype=torch.float32, device="cuda:0"),
                    )
                    loss = torch.nn.functional.mse_loss(
                        output[:, -1:, :] / PREDICTION_DIVISOR,
                        torch.as_tensor(target, dtype=torch.float32, device="cuda:0")[:, -1:, :],
                    )
                    _need(bool(torch.isfinite(loss)), "nonfinite all-source loss")
                    loss.backward()
                    _need(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()), "nonfinite all-source gradient")
                    optimizer.step()
                    value = float(loss.detach().cpu())
                    loss_first = value if loss_first is None else loss_first
                    loss_last = value
                    losses.append(value)
                    prefix_counts[prefix] += 1
                    global_step += 1
                    peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved()))
                if losses:
                    row = {"epoch_zero_based": epoch, "steps": len(losses), "mean_loss": float(np.mean(losses, dtype=np.float64))}
                    epoch_rows.append(row)
                    print(f"EPOCH_END arm={arm} epoch={epoch} steps={len(losses)} mean_loss={row['mean_loss']:.9g}", flush=True)
                if smoke and global_step >= max_steps:
                    break
        finally:
            carrier_module.random.uniform = original_uniform
        training_elapsed = time.monotonic() - training_started
        expected_steps = max_steps if smoke else int(authority["steps_per_arm"])
        _need(global_step == expected_steps == probability_count, "global/dropout step drift")
        _need(_finite_optimizer(optimizer), "Adam state is nonfinite")
        if smoke and arm == "c1":
            _need(all(prefix_counts[value] > 0 for value in C1_CYCLE), "paired smoke lacks full C1 cycle")
        terminal_state_sha = state_hash(model.state_dict())
        checkpoint = None
        if not smoke:
            checkpoint_path = cell / "epoch_049.ckpt"
            payload = {
                "schema": f"{SCHEMA}_checkpoint",
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "metadata": {
                    "arm": arm, "epoch_zero_based": 49, "global_step": global_step,
                    "initial_state_sha256": initial_sha, "terminal_state_sha256": terminal_state_sha,
                    "source_authority_sha256": authority_sha, "schedule_sha256": authority["schedule_sha256"],
                    "source_tensor_surface_sha256": authority["source_tensor_surface_sha256"],
                    "carrier_cache_sha256": authority["carrier_cache_sha256"],
                    "common_config_sha256": common_sha, "config_sha256": config_sha,
                    "dropout_probability_sha256": probability_digest.hexdigest(),
                    "dropout_probability_count": probability_count,
                    "experiment_attempt_sha256": verify_sidecar(result_root.resolve() / "attempt.json"),
                    "code_closure_sha256": attempt["code_closure_sha256"],
                    "warm_start": False, "checkpoint_selection": False,
                    "target_optimizer_steps": 0, "target_backward_steps": 0, "target_model_updates": 0,
                    "heldin_minival_recordings_opened": 0, "heldout_calib_recordings_opened": 0,
                    "evalai_test_recordings_opened": 0,
                },
            }
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, checkpoint_path)
            checkpoint_path.chmod(0o444)
            checkpoint_sha = sha256_file(checkpoint_path)
            _publish_bytes(checkpoint_path.with_name(checkpoint_path.name + ".sha256"), f"{checkpoint_sha}  {checkpoint_path.name}\n".encode("ascii"), sidecar=False)
            checkpoint = {"relative": str(checkpoint_path.relative_to(result_root.resolve())), "sha256": checkpoint_sha}
        normalizer = _load_json(result_root.resolve() / "source_authority/normalizer.json")[0]
        body = {
            "schema": SCHEMA, "status": STATUS_SMOKE if smoke else STATUS_ARM,
            "arm": arm, "smoke": smoke, "gpu": {**_gpu_profile(physical_gpu), "physical_index": int(physical_gpu)},
            "source_authority_sha256": authority_sha, "schedule_sha256": authority["schedule_sha256"],
            "batch_order_sha256": schedule["batch_order_sha256"], "m7_schedule_sha256": schedule["m7_schedule_sha256"],
            "source_tensor_surface_sha256": authority["source_tensor_surface_sha256"],
            "carrier_cache_sha256": authority["carrier_cache_sha256"],
            "normalized_carrier_sha256": normalizer["normalized_tensor_sha256"],
            "common_config_sha256": common_sha, "config_sha256": config_sha,
            "initial_state_sha256": initial_sha, "terminal_state_sha256": terminal_state_sha,
            "global_step": global_step, "epoch_zero_based": 0 if smoke else 49,
            "dropout_probability_sha256": probability_digest.hexdigest(),
            "dropout_probability_count": probability_count,
            "prefix_step_counts": {str(key): value for key, value in prefix_counts.items()},
            "loss_first": loss_first, "loss_last": loss_last, "epochs": epoch_rows,
            "training_elapsed_seconds": training_elapsed, "peak_memory_reserved_bytes": peak_reserved,
            "checkpoint": checkpoint, "target_optimizer_steps": 0, "target_backward_steps": 0,
            "target_model_updates": 0, "heldin_minival_recordings_opened": 0,
            "heldout_calib_recordings_opened": 0, "evalai_test_recordings_opened": 0,
            "evalai_submissions": 0, "elapsed_seconds": time.monotonic() - started,
            "finished_at_utc": utc_now(),
        }
        publish_json(cell / "terminal.json", body)
        return body
    except BaseException as error:
        try:
            publish_json(cell / "failure.json", {
                "schema": SCHEMA, "status": "FAIL_ARM_NO_AUTOMATIC_RETRY", "arm": arm,
                "smoke": smoke, "error_type": type(error).__name__, "error": str(error),
                "physical_gpu": int(physical_gpu), "heldin_minival_recordings_opened": 0,
                "heldout_calib_recordings_opened": 0, "evalai_test_recordings_opened": 0,
                "elapsed_seconds": time.monotonic() - started,
            })
        except BaseException:
            pass
        raise


def verify_pair(result_root: Path, *, smoke: bool = False) -> dict[str, Any]:
    root = result_root.resolve() / ("smoke" if smoke else "training")
    rows = {}
    digests = {}
    for arm in ARMS:
        rows[arm], digests[arm] = _load_json(root / arm / "terminal.json", SCHEMA)
        _need(rows[arm]["status"] == (STATUS_SMOKE if smoke else STATUS_ARM), f"{arm} terminal status drift")
    t0, c1 = rows["t0"], rows["c1"]
    matched_fields = (
        "source_authority_sha256", "schedule_sha256", "batch_order_sha256", "m7_schedule_sha256",
        "source_tensor_surface_sha256", "carrier_cache_sha256", "normalized_carrier_sha256",
        "common_config_sha256", "initial_state_sha256", "global_step",
        "dropout_probability_sha256", "dropout_probability_count",
    )
    for field in matched_fields:
        _need(t0[field] == c1[field], f"paired field drift: {field}")
    for row in rows.values():
        _need(row["target_optimizer_steps"] == row["target_backward_steps"] == row["target_model_updates"] == 0, "target update recorded")
        _need(row["heldin_minival_recordings_opened"] == row["heldout_calib_recordings_opened"] == row["evalai_test_recordings_opened"] == 0, "pre-pair target access")
        if not smoke:
            _need(row["epoch_zero_based"] == 49 and row["checkpoint"], "non-epoch49 checkpoint")
            path = result_root.resolve() / row["checkpoint"]["relative"]
            _need(verify_sidecar(path) == row["checkpoint"]["sha256"], "checkpoint SHA drift")
            import torch
            payload = torch.load(path, map_location="cpu", weights_only=False)
            metadata = payload["metadata"]
            _need(metadata["terminal_state_sha256"] == row["terminal_state_sha256"], "checkpoint state provenance drift")
            _need(metadata["epoch_zero_based"] == 49 and metadata["warm_start"] is False and metadata["checkpoint_selection"] is False, "checkpoint contract drift")
    directory = result_root.resolve() / ("smoke" if smoke else "pair_integrity")
    body = {
        "schema": f"{SCHEMA}_pair_integrity", "status": STATUS_SMOKE if smoke else STATUS_PAIR,
        "smoke": smoke, "initial_state_identical": True, "source_authority_identical": True,
        "batch_order_identical": True, "m7_schedule_identical": True,
        "carrier_bytes_matched": True, "query_target_bytes_matched": True,
        "dropout_probability_sequence_matched": True, "optimizer_global_steps_matched": True,
        "global_step_per_arm": t0["global_step"],
        "t0_terminal_sha256": digests["t0"], "c1_terminal_sha256": digests["c1"],
        "t0_checkpoint_sha256": None if smoke else t0["checkpoint"]["sha256"],
        "c1_checkpoint_sha256": None if smoke else c1["checkpoint"]["sha256"],
        "target_optimizer_steps": 0, "target_backward_steps": 0, "target_model_updates": 0,
        "heldin_minival_recordings_opened": 0, "heldout_calib_recordings_opened": 0,
        "evalai_test_recordings_opened": 0, "evalai_submissions": 0,
    }
    publish_json(directory / "paired_integrity.json", body)
    return body


def _load_heldout_record(path: Path) -> H1PilotRecord:
    from falcon_challenge.config import FalconTask
    from falcon_challenge.dataloaders import load_nwb
    from pynwb import NWBHDF5IO
    resolved = path.resolve()
    _need("sub-HumanPitt-held-out-calib" in str(resolved), "held-out loader scope drift")
    neural, velocity, trial_change, eval_mask = load_nwb(resolved, FalconTask.h1)
    with NWBHDF5IO(str(resolved), "r", load_namespaces=True) as handle:
        nwb = handle.read()
        _need("TrialNum" in nwb.acquisition, "held-out TrialNum missing")
        trial_num = np.asarray(nwb.acquisition["TrialNum"].data[:], dtype=np.float64)
    spikes64 = np.asarray(neural, np.float64)
    targets64 = np.asarray(velocity, np.float64)
    spikes = spikes64.astype(np.float32)
    targets = targets64.astype(np.float32)
    changes = np.asarray(trial_change, bool).reshape(-1)
    mask = np.asarray(eval_mask, bool).reshape(-1)
    _need(spikes.ndim == 2 and spikes.shape[1] == EXPECTED_NEURONS and targets.shape == (len(spikes), 7), "held-out record shape drift")
    _need(len(changes) == len(mask) == len(trial_num) == len(spikes), "held-out record alignment drift")
    name = session_from_path(resolved)
    values = _ordered_eval_trials(trial_num, mask)
    trials = tuple(_trial_blocks(value, spikes64, targets64, mask, trial_num) for value in values)
    return H1PilotRecord(name, session_date(name), resolved, sha256_file(resolved), spikes, targets, changes, mask, trial_num, values, trials)


def _calibration_payloads(data_root: Path, result_root: Path):
    pair, pair_sha = _load_json(result_root.resolve() / "pair_integrity/paired_integrity.json", f"{SCHEMA}_pair_integrity")
    _need(pair["status"] == STATUS_PAIR, "pair integrity must precede calibration")
    _authority, authority_sha, _schedule, plan, _dataset, _order, _m7, _prefixes = _load_source_materialization(data_root, result_root)
    normalizer = _load_json(result_root.resolve() / "source_authority/normalizer.json")[0]
    denominator = max(float(normalizer["s_src"]), NORMALIZER_FLOOR)
    heldin_paths = index_heldin_calib(data_root)
    heldout_paths = index_heldout_calib(data_root)
    mapping_rows = HELDIN_SESSION_TO_FALCON_KEY + HELDOUT_SESSION_TO_FALCON_KEY
    payloads = {}
    manifest_rows = []
    heldin_bytes = 0
    heldout_bytes = 0
    for session, key in mapping_rows:
        heldout = session in H1_HELDOUT_SESSIONS
        path = heldout_paths[session] if heldout else heldin_paths[session]
        record = _load_heldout_record(path) if heldout else load_record(path)
        _need(len(record.trial_values) >= 3, f"{session}: M3 calibration unavailable")
        support = tuple(float(value) for value in record.trial_values[:3])
        identity = np.ascontiguousarray(np.stack([interpolate_trial_identity(record, value) for value in support]), np.float32)
        fitted = fit_deployment_carrier(record, plan, support)
        carrier = np.ascontiguousarray(np.asarray(fitted["carrier"], np.float64) / denominator, np.float32)
        _need(identity.shape == (3, 1024, 176) and carrier.shape == (176, 4), "M3 package shape drift")
        payloads[key] = {"identity": identity, "carrier": carrier, "calibration_trials": list(support), "session": session}
        size = int(path.stat().st_size)
        heldout_bytes += size if heldout else 0
        heldin_bytes += 0 if heldout else size
        manifest_rows.append({
            "session": session, "falcon_key": key, "scope": "held-out-calib" if heldout else "held-in-calib",
            "filename": path.name, "bytes": size, "nwb_sha256": record.input_sha256,
            "calibration_trials": list(support), "identity_sha256": array_sha256(identity),
            "carrier_sha256": array_sha256(carrier), "support_m": fitted["support_m"],
        })
    body = {
        "schema": f"{SCHEMA}_calibration_authority", "status": "PASS_M3_CALIBRATION_PAYLOADS",
        "source_authority_sha256": authority_sha, "pair_integrity_sha256": pair_sha,
        "sessions": manifest_rows, "heldin_calib_recordings_opened": 13,
        "heldin_calib_bytes_read": heldin_bytes, "heldout_calib_recordings_opened": 14,
        "heldout_calib_bytes_read": heldout_bytes, "heldin_minival_recordings_opened": 0,
        "evalai_test_recordings_opened": 0, "optimizer_steps": 0, "backward_steps": 0,
        "model_updates": 0, "evalai_submissions": 0,
    }
    calibration_sha = publish_json(result_root.resolve() / "packages/calibration_authority.json", body)
    return payloads, calibration_sha, authority_sha


def build_packages(data_root: Path, result_root: Path) -> dict[str, Any]:
    import torch
    from third_party.falcon_challenge.h1_carrier_id_spint_decoder import PACKAGE_SCHEMA
    payloads, calibration_sha, authority_sha = _calibration_payloads(data_root, result_root)
    rows = []
    for arm in ARMS:
        terminal = _load_json(result_root.resolve() / "training" / arm / "terminal.json", SCHEMA)[0]
        checkpoint_path = result_root.resolve() / terminal["checkpoint"]["relative"]
        _need(verify_sidecar(checkpoint_path) == terminal["checkpoint"]["sha256"], "package checkpoint drift")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        package = {
            "schema": PACKAGE_SCHEMA, "task": "h1", "arm": arm,
            "state_dict": checkpoint["state_dict"], "model_kwargs": model_config()["model_kwargs"],
            "model_state_sha256": terminal["terminal_state_sha256"],
            "checkpoint_sha256": terminal["checkpoint"]["sha256"],
            "source_authority_sha256": authority_sha, "calibration_authority_sha256": calibration_sha,
            "window_size": WINDOW_SIZE, "prediction_divisor": PREDICTION_DIVISOR,
            "sessions": payloads, "optimizer_steps": 0, "backward_steps": 0,
            "model_updates": 0, "evalai_submissions": 0,
        }
        buffer = io.BytesIO()
        torch.save(package, buffer)
        path = result_root.resolve() / "packages" / f"{arm}.pt"
        digest = _publish_bytes(path, buffer.getvalue())
        rows.append({"arm": arm, "relative": str(path.relative_to(result_root.resolve())), "sha256": digest, "checkpoint_sha256": terminal["checkpoint"]["sha256"], "model_state_sha256": terminal["terminal_state_sha256"]})
    body = {
        "schema": f"{SCHEMA}_packages", "status": STATUS_PACKAGES,
        "calibration_authority_sha256": calibration_sha, "source_authority_sha256": authority_sha,
        "packages": rows, "session_payloads": 27, "evalai_submissions": 0,
    }
    publish_json(result_root.resolve() / "packages/packages.json", body)
    return body


def _minival_paths(data_root: Path) -> tuple[Path, ...]:
    directory = data_root.resolve() / "sub-HumanPitt-held-in-minival"
    _need(directory.is_dir(), "held-in-minival directory missing")
    observed = {session_from_path(path.resolve()): path.resolve() for path in sorted(directory.glob("*.nwb"))}
    _need(tuple(observed) == H1_HELDIN_SESSIONS, "held-in-minival roster/order drift")
    return tuple(observed[name] for name in H1_HELDIN_SESSIONS)


def run_local_minival(data_root: Path, result_root: Path, *, device: str = "cuda:0") -> dict[str, Any]:
    import falcon_challenge.evaluator as evaluator_module
    from falcon_challenge.config import FalconConfig, FalconTask
    from falcon_challenge.evaluator import FalconEvaluator
    from third_party.falcon_challenge.h1_carrier_id_spint_decoder import H1CarrierIdSpintDecoder
    packages, packages_sha = _load_json(result_root.resolve() / "packages/packages.json", f"{SCHEMA}_packages")
    _need(packages["status"] == STATUS_PACKAGES, "packages must precede minival")
    paths = _minival_paths(data_root)
    session_to_key = dict(HELDIN_SESSION_TO_FALCON_KEY)
    arm_results = {}
    cache_arrays = {}
    access_rows = []
    original_tqdm = evaluator_module.tqdm
    evaluator_module.tqdm = lambda iterable, *args, **kwargs: iterable
    try:
        for row in packages["packages"]:
            arm = row["arm"]
            package_path = result_root.resolve() / row["relative"]
            _need(verify_sidecar(package_path) == row["sha256"], "minival package SHA drift")
            decoder = H1CarrierIdSpintDecoder(FalconConfig(task=FalconTask.h1), package_path, batch_size=13, device=device)
            before = decoder.model_state_sha256()
            evaluator = FalconEvaluator(eval_remote=False, split="h1", verbose=False, dataloader_workers=0)
            predictions, targets, masks, _compute_times, _neural_times = evaluator.predict_files(decoder, list(paths))
            after = decoder.model_state_sha256()
            _need(before == after == row["model_state_sha256"], f"{arm} model state changed during minival")
            scores = {}
            warmup_excluded = {}
            for index, (session, key) in enumerate(HELDIN_SESSION_TO_FALCON_KEY):
                _need(key == session_to_key[session] and key in predictions and key in targets and key in masks, f"minival key missing: {key}")
                prediction = np.asarray(predictions[key], np.float32)
                target = np.asarray(targets[key], np.float32)
                mask = np.asarray(masks[key], bool).reshape(-1)
                _need(prediction.shape == target.shape and prediction.ndim == 2 and prediction.shape[1] == 7 and len(mask) == len(prediction), f"minival array drift: {key}")
                complete_history = np.arange(len(mask), dtype=np.int64) >= WINDOW_SIZE - 1
                effective = mask & complete_history
                _need(int(effective.sum()) > 1, f"minival full-history score unavailable: {key}")
                scores[key] = variance_weighted_r2(target[effective], prediction[effective])
                warmup_excluded[key] = int(np.sum(mask & ~complete_history))
                prefix = f"{arm}_{index}"
                cache_arrays[f"{prefix}_prediction"] = prediction
                cache_arrays[f"{prefix}_target"] = target
                cache_arrays[f"{prefix}_eval_mask"] = mask
                cache_arrays[f"{prefix}_score_mask"] = effective
            values = np.asarray([scores[key] for _, key in HELDIN_SESSION_TO_FALCON_KEY], np.float64)
            arm_results[arm] = {
                "per_session_r2": scores, "mean_r2": float(np.mean(values, dtype=np.float64)),
                "std_r2_population": float(np.std(values, ddof=0, dtype=np.float64)),
                "warmup_eval_valid_bins_excluded": warmup_excluded,
                "model_state_before_sha256": before, "model_state_after_sha256": after,
            }
            access_rows.append({
                "arm": arm, "recordings_opened": 13, "bytes_read": sum(int(path.stat().st_size) for path in paths),
                "files": [{"session": session, "falcon_key": key, "filename": path.name, "bytes": int(path.stat().st_size), "sha256": sha256_file(path)} for (session, key), path in zip(HELDIN_SESSION_TO_FALCON_KEY, paths, strict=True)],
            })
    finally:
        evaluator_module.tqdm = original_tqdm
    for index, (_session, _key) in enumerate(HELDIN_SESSION_TO_FALCON_KEY):
        _need(np.array_equal(cache_arrays[f"t0_{index}_target"], cache_arrays[f"c1_{index}_target"]), "T0/C1 minival target bytes drift")
        _need(np.array_equal(cache_arrays[f"t0_{index}_score_mask"], cache_arrays[f"c1_{index}_score_mask"]), "T0/C1 minival mask drift")
    per_session = {}
    for _session, key in HELDIN_SESSION_TO_FALCON_KEY:
        t0 = float(arm_results["t0"]["per_session_r2"][key])
        c1 = float(arm_results["c1"]["per_session_r2"][key])
        per_session[key] = {"t0": t0, "c1": c1, "delta_c1_minus_t0": c1 - t0}
    delta = arm_results["c1"]["mean_r2"] - arm_results["t0"]["mean_r2"]
    cache_sha = publish_npz(result_root.resolve() / "minival/prediction_cache.npz", **cache_arrays)
    cache_manifest_sha = publish_json(result_root.resolve() / "minival/prediction_cache.json", {
        "schema": f"{SCHEMA}_minival_prediction_cache", "arrays_file_sha256": cache_sha,
        "array_sha256": {name: array_sha256(value) for name, value in cache_arrays.items()},
        "array_shape": {name: list(value.shape) for name, value in cache_arrays.items()},
    })
    access_sha = publish_json(result_root.resolve() / "minival/target_access.json", {
        "schema": f"{SCHEMA}_minival_access", "scope": "held-in-minival only",
        "arms": access_rows, "total_recordings_opened": 26,
        "heldout_calib_scoring_recordings_opened": 0, "evalai_test_recordings_opened": 0,
        "optimizer_steps": 0, "backward_steps": 0, "model_updates": 0,
        "target_driven_selection": False, "evalai_submissions": 0,
    })
    body = {
        "schema": f"{SCHEMA}_minival_metrics", "status": STATUS_MINIVAL,
        "label": "local held-in-minival deployment sanity R²", "packages_sha256": packages_sha,
        "arms": arm_results, "per_session": per_session,
        "mean_delta_c1_minus_t0": delta,
        "prediction_cache_sha256": cache_sha, "prediction_cache_manifest_sha256": cache_manifest_sha,
        "target_access_sha256": access_sha, "checkpoint_or_budget_selection": False,
        "retraining_triggered": False, "optimizer_steps": 0, "backward_steps": 0,
        "model_updates": 0, "evalai_submissions": 0,
    }
    publish_json(result_root.resolve() / "minival/metrics.json", body)
    return body


def run_package_rehearsal(data_root: Path, result_root: Path) -> dict[str, Any]:
    from falcon_challenge.config import FalconConfig, FalconTask
    from third_party.falcon_challenge.h1_carrier_id_spint_decoder import H1CarrierIdSpintDecoder
    packages = _load_json(result_root.resolve() / "packages/packages.json", f"{SCHEMA}_packages")[0]
    paths = _minival_paths(data_root)
    config = FalconConfig(task=FalconTask.h1)
    rows = []
    for package in packages["packages"]:
        path = result_root.resolve() / package["relative"]
        _need(verify_sidecar(path) == package["sha256"], "rehearsal package SHA drift")
        cpu_a = H1CarrierIdSpintDecoder(config, path, batch_size=1, device="cpu")
        cpu_b = H1CarrierIdSpintDecoder(config, path, batch_size=1, device="cpu")
        cpu_a.reset([paths[0]]); cpu_b.reset([paths[0]])
        sample = np.zeros((1, 176), np.float32)
        cpu_before = cpu_a.model_state_sha256()
        cpu_pred_a = cpu_a.predict(sample); cpu_pred_b = cpu_b.predict(sample)
        _need(np.array_equal(cpu_pred_a, cpu_pred_b), "CPU package reload prediction drift")
        _need(cpu_a.model_state_sha256() == cpu_before == package["model_state_sha256"], "CPU package state mutation")
        gpu_a = H1CarrierIdSpintDecoder(config, path, batch_size=1, device="cuda:0")
        gpu_b = H1CarrierIdSpintDecoder(config, path, batch_size=1, device="cuda:0")
        gpu_a.reset([paths[0]]); gpu_b.reset([paths[0]])
        gpu_before = gpu_a.model_state_sha256()
        gpu_pred_a = gpu_a.predict(sample); gpu_pred_b = gpu_b.predict(sample)
        _need(np.array_equal(gpu_pred_a, gpu_pred_b), "GPU package reload prediction drift")
        _need(gpu_a.model_state_sha256() == gpu_before == package["model_state_sha256"], "GPU package state mutation")
        _need(np.allclose(cpu_pred_a, gpu_pred_a, rtol=2e-3, atol=2e-4), "CPU/GPU reload numerical drift")
        batch_decoder = H1CarrierIdSpintDecoder(config, path, batch_size=2, device="cuda:0")
        batch_decoder.reset([paths[0], paths[1]])
        batch_before = batch_decoder.model_state_sha256()
        batch_prediction = batch_decoder.predict(np.zeros((2, 176), np.float32))
        _need(batch_prediction.shape == (2, 7) and batch_decoder.model_state_sha256() == batch_before, "batch-size=2 compatibility/state drift")
        rows.append({
            "arm": package["arm"], "package_sha256": package["sha256"],
            "cpu_reload": True, "gpu_reload": True, "batch_sizes_tested": [1, 2],
            "cpu_repeat_exact": True, "gpu_repeat_exact": True,
            "cpu_gpu_allclose_rtol": 0.002, "cpu_gpu_allclose_atol": 0.0002,
            "model_state_immutable": True,
        })
    body = {
        "schema": f"{SCHEMA}_package_rehearsal", "status": STATUS_REHEARSAL,
        "packages": rows, "reset_predict_smoke": True,
        "serialization_deserialization": True, "package_reload_numerical_equivalence": True,
        "optimizer_steps": 0, "backward_steps": 0, "model_updates": 0,
        "evalai_test_recordings_opened": 0, "evalai_submissions": 0,
    }
    publish_json(result_root.resolve() / "packages/rehearsal.json", body)
    return body


def verify_terminal(result_root: Path) -> dict[str, Any]:
    root = result_root.resolve()
    attempt = load_attempt(root)
    attempt_sha = verify_sidecar(root / "attempt.json")
    authority, authority_sha = _load_json(root / "source_authority/authority.json", f"{SCHEMA}_source_authority")
    _need(authority["status"] == STATUS_SOURCE, "terminal source authority drift")
    pair, pair_sha = _load_json(root / "pair_integrity/paired_integrity.json", f"{SCHEMA}_pair_integrity")
    _need(pair["status"] == STATUS_PAIR, "terminal pair integrity drift")
    packages, packages_sha = _load_json(root / "packages/packages.json", f"{SCHEMA}_packages")
    _need(packages["status"] == STATUS_PACKAGES and len(packages["packages"]) == 2, "terminal package authority drift")
    for row in packages["packages"]:
        _need(verify_sidecar(root / row["relative"]) == row["sha256"], "terminal package file SHA drift")
    rehearsal, rehearsal_sha = _load_json(root / "packages/rehearsal.json", f"{SCHEMA}_package_rehearsal")
    _need(rehearsal["status"] == STATUS_REHEARSAL, "terminal package rehearsal drift")
    metrics, metrics_sha = _load_json(root / "minival/metrics.json", f"{SCHEMA}_minival_metrics")
    _need(metrics["status"] == STATUS_MINIVAL and metrics["label"] == "local held-in-minival deployment sanity R²", "terminal minival label/status drift")
    cache_manifest, cache_manifest_sha = _load_json(root / "minival/prediction_cache.json", f"{SCHEMA}_minival_prediction_cache")
    _need(cache_manifest_sha == metrics["prediction_cache_manifest_sha256"], "terminal prediction manifest SHA drift")
    cache_path = root / "minival/prediction_cache.npz"
    _need(verify_sidecar(cache_path) == metrics["prediction_cache_sha256"] == cache_manifest["arrays_file_sha256"], "terminal prediction cache SHA drift")
    with np.load(cache_path, allow_pickle=False) as values:
        arrays = {name: np.asarray(values[name]) for name in values.files}
    recomputed = {arm: {} for arm in ARMS}
    for index, (_session, key) in enumerate(HELDIN_SESSION_TO_FALCON_KEY):
        for arm in ARMS:
            prefix = f"{arm}_{index}"
            prediction = arrays[f"{prefix}_prediction"]
            target = arrays[f"{prefix}_target"]
            mask = np.asarray(arrays[f"{prefix}_score_mask"], bool)
            _need(array_sha256(prediction) == cache_manifest["array_sha256"][f"{prefix}_prediction"], "prediction tensor SHA drift")
            _need(np.array_equal(target, arrays[f"t0_{index}_target"]) and np.array_equal(mask, arrays[f"t0_{index}_score_mask"]), "paired minival scoring surface drift")
            recomputed[arm][key] = variance_weighted_r2(target[mask], prediction[mask])
            _need(math.isclose(recomputed[arm][key], metrics["per_session"][key][arm], rel_tol=0.0, abs_tol=1e-12), "minival metric recomputation drift")
    means = {arm: float(np.mean(list(recomputed[arm].values()), dtype=np.float64)) for arm in ARMS}
    stds = {arm: float(np.std(list(recomputed[arm].values()), ddof=0, dtype=np.float64)) for arm in ARMS}
    _need(math.isclose(means["c1"] - means["t0"], metrics["mean_delta_c1_minus_t0"], rel_tol=0.0, abs_tol=1e-12), "minival aggregate delta drift")
    training = {}
    total_target_updates = 0
    for arm in ARMS:
        terminal, terminal_sha = _load_json(root / "training" / arm / "terminal.json", SCHEMA)
        _need(terminal["status"] == STATUS_ARM and terminal["epoch_zero_based"] == 49, "training terminal drift")
        _need(verify_sidecar(root / terminal["checkpoint"]["relative"]) == terminal["checkpoint"]["sha256"], "training checkpoint SHA drift")
        total_target_updates += terminal["target_optimizer_steps"] + terminal["target_backward_steps"] + terminal["target_model_updates"]
        training[arm] = {"terminal_sha256": terminal_sha, "global_step": terminal["global_step"], "training_elapsed_seconds": terminal["training_elapsed_seconds"], "checkpoint_sha256": terminal["checkpoint"]["sha256"], "gpu": terminal["gpu"]}
    _need(total_target_updates == 0, "target optimizer/backward/update activity")
    body = {
        "schema": SCHEMA, "status": STATUS_TERMINAL, "finished_at_utc": utc_now(),
        "experiment_attempt_sha256": attempt_sha, "code_closure_sha256": attempt["code_closure_sha256"],
        "source_authority_sha256": authority_sha, "selected_q": authority["selected_q"],
        "selected_lambda": authority["selected_lambda"], "normalizer_sha256": authority["normalizer_sha256"],
        "pair_integrity_sha256": pair_sha, "training": training,
        "packages_sha256": packages_sha, "package_rehearsal_sha256": rehearsal_sha,
        "minival_metrics_sha256": metrics_sha, "minival_label": metrics["label"],
        "minival_per_session": metrics["per_session"], "minival_mean_r2": means,
        "minival_std_r2_population": stds, "minival_delta_c1_minus_t0": means["c1"] - means["t0"],
        "ready_for_evalai_hidden_test_without_model_change": True,
        "evalai_submission_authorized": False, "evalai_submissions": 0,
        "evalai_test_recordings_opened": 0, "official_heldout_score_accessed": False,
        "target_optimizer_steps": 0, "target_backward_steps": 0, "target_model_updates": 0,
        "post_minival_selection": False, "post_minival_retraining": False,
        "claim": "local held-in-minival deployment sanity R² only; not held-out R²",
    }
    terminal_sha = publish_json(root / "terminal.json", body)
    lines = [
        "# H1 CAL-AUG All-Source M3 Deployment V1", "", f"- Status: `{STATUS_TERMINAL}`",
        f"- Selected q/lambda: `{authority['selected_q']}` / `{authority['selected_lambda']}`",
        f"- T0 mean/std: `{means['t0']:.9f}` / `{stds['t0']:.9f}`",
        f"- C1 mean/std: `{means['c1']:.9f}` / `{stds['c1']:.9f}`",
        f"- Mean delta C1-T0: `{means['c1'] - means['t0']:+.9f}`",
        "- Metric label: `local held-in-minival deployment sanity R²`.",
        "- EvalAI submissions and official held-out score accesses: `0`.", "",
        "| FALCON session | T0 R² | C1 R² | Delta |", "|---|---:|---:|---:|",
    ]
    for key, row in metrics["per_session"].items():
        lines.append(f"| {key} | {row['t0']:.9f} | {row['c1']:.9f} | {row['delta_c1_minus_t0']:+.9f} |")
    lines.extend(["", f"Terminal SHA-256: `{terminal_sha}`"])
    publish_text(root / "EXPERIMENT_RECORD.md", "\n".join(lines) + "\n")
    return body


__all__ = (
    "SCHEMA", "STATUS_ARM", "STATUS_MINIVAL", "STATUS_PACKAGES", "STATUS_PAIR",
    "STATUS_REHEARSAL", "STATUS_SMOKE", "STATUS_SOURCE", "STATUS_TERMINAL",
    "build_packages", "common_config", "create_attempt", "load_attempt",
    "prepare_source_authority", "run_arm", "run_local_minival",
    "run_package_rehearsal", "verify_pair", "verify_terminal",
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT4_TERMINAL = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_prefix_cycle_v1_eval_a1/terminal.json"
M4_STOP_TERMINAL = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_all_source_heldout_v1/terminal.json"
M3_TERMINAL = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_prefix_cycle_m3_transfer_v1/terminal.json"
EXPERIMENT4_TERMINAL_SHA256 = "dc9e7ab44954d3d193f67f9bf8936aafdaf2b05be9968d5e0091c0b0ecf092fd"
M4_STOP_TERMINAL_SHA256 = "3ff971dc576958b13ace990bcca8aea2e8b999e2af2ed50f418296d05f8d5cfc"
M3_TERMINAL_SHA256 = "199a2fec864d7ae40d33ec911e43cd32e8623e5687ed9d44c5c9ac946a964429"


def validate_predecessors(result_root: Path) -> dict[str, Any]:
    rows = []
    expected = (
        ("experiment4_m4", EXPERIMENT4_TERMINAL, EXPERIMENT4_TERMINAL_SHA256, "PASS_H1_CAL_AUG_PREFIX_CYCLE_TRANSFER"),
        ("m4_feasibility_stop", M4_STOP_TERMINAL, M4_STOP_TERMINAL_SHA256, "STOP_H1_ALL_SOURCE_HELDOUT_M4_PROTOCOL_INFEASIBLE"),
        ("m3_secondary_diagnostic", M3_TERMINAL, M3_TERMINAL_SHA256, "COMPLETE_H1_CAL_AUG_PREFIX_CYCLE_M3_TRANSFER_V1"),
    )
    for name, path, digest, status in expected:
        _need(verify_sidecar(path) == digest, f"sealed predecessor SHA drift: {name}")
        body = json.loads(path.read_text(encoding="utf-8"))
        _need(body.get("status") == status, f"sealed predecessor status drift: {name}")
        rows.append({"name": name, "relative": str(path.relative_to(REPO_ROOT)), "sha256": digest, "status": status})
    m3 = json.loads(M3_TERMINAL.read_text(encoding="utf-8"))
    _need(m3.get("verdict") == "STRONG_M3_PREFIX_EXTRAPOLATION", "sealed M3 interpretation drift")
    body = {
        "schema": f"{SCHEMA}_predecessor_authority", "status": "PASS_SEALED_PREDECESSOR_AUTHORITIES",
        "predecessors": rows, "m3_interpretation": "STRONG_M3_PREFIX_EXTRAPOLATION",
        "heldin_calib_recordings_opened": 0, "heldin_minival_recordings_opened": 0,
        "heldout_calib_recordings_opened": 0, "evalai_test_recordings_opened": 0,
        "evalai_submissions": 0,
    }
    publish_json(result_root.resolve() / "predecessor_authority.json", body)
    return body
