"""Five-date matched H1 CAL-AUG M7/M5/M4 prefix-cycle experiment."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import stat
import struct
import time
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.h1_m4_eb_pilot import (
    EXPECTED_NEURONS,
    H1PilotRecord,
    array_sha256,
    fit_frozen_carrier,
    index_heldin_calib,
    interpolate_trial_identity,
    load_record,
)
from src.h1_hc_date_lodo_regen_v1 import (
    BATCH_SIZE,
    MODEL_PARAMETERS,
    SourceDataset,
    _publish_bytes,
    _finite_optimizer,
    _gpu_profile,
    _load_date_materialization,
    _new_model,
    model_config,
    publish_json,
    publish_npz,
    publish_text,
    variance_weighted_r2,
    verify_sidecar,
)
from src.h1_m4_cce_contract import (
    CONFIRMATORY_DATES,
    FIXED_EPOCHS,
    FIXED_SEED,
    NORMALIZER_FLOOR,
    WINDOW_SIZE,
    canonical_sha256,
    sha256_file,
    state_hash,
)
from src.data.h1_m4_cce_date_lodo import target_sessions_for_date


SCHEMA = "h1_cal_aug_prefix_cycle_v1"
ARMS = ("t0", "c1")
BUDGETS = (4, 5, 7)
C1_CYCLE = (7, 5, 4)
MAX_PREFIX = 7
STATUS_SOURCE = "PASS_H1_CAL_AUG_PREFIX_CYCLE_V1_SOURCE_AUTHORITY"
STATUS_ARM = "PASS_H1_CAL_AUG_PREFIX_CYCLE_V1_ARM_EPOCH49"
STATUS_SMOKE = "PASS_H1_CAL_AUG_PREFIX_CYCLE_V1_PAIRED_SMOKE"
STATUS_PAIRS = "PASS_H1_CAL_AUG_PREFIX_CYCLE_V1_TEN_MODEL_INTEGRITY"
STATUS_EVAL = "PASS_H1_CAL_AUG_PREFIX_CYCLE_V1_DATE_EVALUATED"
STATUS_PASS = "PASS_H1_CAL_AUG_PREFIX_CYCLE_TRANSFER"
STATUS_NO_TRANSFER = "COMPLETE_H1_CAL_AUG_PREFIX_CYCLE_NO_TRANSFER"
EXPERIMENT3_TERMINAL_SHA256 = "69ca9ac9eedabc6328bd0e6afa6556b40b479ecb6ffcfa6e66294580fd37258f"
EXPERIMENT3_COMMIT = "c9ad06f96d811fdf6be94c391128f0735060f4c6"
REGEN_TERMINAL_SHA256 = "470634334480c33fd8d4679baa470454984dadeb57a25b9dec8d05612c43b9ec"
REGEN_SOURCE_SHA256 = "6cf656048af20174c3cf164406e25051137bfe687b0dda93d06ff1835c80500e"
ARM_TIMEOUT_SECONDS = 8 * 60 * 60
RESOURCE_WAIT_SECONDS = 24 * 60 * 60


class CalAugError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise CalAugError(message)


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
        "outer_dates": list(CONFIRMATORY_DATES),
        "arms": list(ARMS),
        "t0_prefix": 7,
        "c1_cycle": list(C1_CYCLE),
        "evaluation_budgets": list(BUDGETS),
        "models": 10,
        "target_access": 0,
    }


def create_attempt(result_root: Path, closure: Mapping[str, str], head: str) -> dict[str, Any]:
    root = result_root.resolve()
    _need(not root.exists(), f"canonical result root is not fresh: {root}")
    body = {
        "schema": SCHEMA,
        "artifact": "attempt",
        "status": "ATTEMPT_BEFORE_DATA_AND_CUDA",
        "created_at_utc": utc_now(),
        "head": head,
        "closure": dict(closure),
        "code_closure_sha256": canonical_sha256(dict(closure)),
        "outer_dates": list(CONFIRMATORY_DATES),
        "arms": list(ARMS),
        "target_recordings_opened": 0,
        "target_bytes_read": 0,
        "cuda_initialized": False,
    }
    publish_json(root / "attempt.json", body)
    return body


def load_attempt(result_root: Path) -> dict[str, Any]:
    body, _ = _load_json(result_root.resolve() / "attempt.json", SCHEMA)
    _need(body.get("status") == "ATTEMPT_BEFORE_DATA_AND_CUDA", "attempt status drift")
    _need(body.get("target_recordings_opened") == body.get("target_bytes_read") == 0, "attempt target access")
    return body


def validate_predecessors(experiment3_root: Path, predecessor_root: Path) -> dict[str, Any]:
    exp3_terminal = experiment3_root.resolve() / "terminal.json"
    regen_terminal = predecessor_root.resolve() / "terminal.json"
    regen_source = predecessor_root.resolve() / "source_authority.json"
    _need(verify_sidecar(exp3_terminal) == EXPERIMENT3_TERMINAL_SHA256, "experiment-3 terminal drift")
    _need(verify_sidecar(regen_terminal) == REGEN_TERMINAL_SHA256, "regeneration terminal drift")
    _need(verify_sidecar(regen_source) == REGEN_SOURCE_SHA256, "regeneration source authority drift")
    exp3 = json.loads(exp3_terminal.read_text(encoding="utf-8"))
    regen = json.loads(regen_terminal.read_text(encoding="utf-8"))
    source = json.loads(regen_source.read_text(encoding="utf-8"))
    _need(exp3.get("status", "").endswith("NO_TRANSFER"), "experiment-3 is not a completed negative authority")
    _need(tuple(regen.get("date_order", ())) == CONFIRMATORY_DATES, "regeneration date set drift")
    _need(regen.get("target_bytes_read") == source.get("target_bytes_read") == 0, "predecessor target access drift")
    return {
        "schema": f"{SCHEMA}_predecessor_authority",
        "status": "PASS_H1_CAL_AUG_PREFIX_CYCLE_V1_PREDECESSORS",
        "experiment3_terminal_sha256": EXPERIMENT3_TERMINAL_SHA256,
        "experiment3_commit": EXPERIMENT3_COMMIT,
        "regeneration_terminal_sha256": REGEN_TERMINAL_SHA256,
        "regeneration_source_authority_sha256": REGEN_SOURCE_SHA256,
        "target_recordings_opened": 0,
        "target_bytes_read": 0,
    }


def _prefix_cycle(outer_date: str, epochs: int, batches: int) -> np.ndarray:
    result = np.empty((epochs, batches), dtype=np.int8)
    for epoch in range(epochs):
        token = hashlib.sha256(f"{SCHEMA}|prefix|{outer_date}|{epoch}".encode()).digest()
        offset = int.from_bytes(token[:8], "big") % len(C1_CYCLE)
        result[epoch] = np.asarray([C1_CYCLE[(index + offset) % 3] for index in range(batches)], dtype=np.int8)
    return result


def _m7_schedule(dataset: SourceDataset, order: np.ndarray, outer_date: str) -> np.ndarray:
    _need(len(order) % BATCH_SIZE == 0, "batch order is incomplete")
    names = np.asarray([dataset.windows[int(row)][0] for row in order], dtype=object)
    for offset in range(0, len(order), BATCH_SIZE):
        _need(len(set(names[offset:offset + BATCH_SIZE])) == 1, "batch is not session homogeneous")
    positions = np.arange(len(order), dtype=np.int64)
    schedule = np.empty((FIXED_EPOCHS, len(order)), dtype=np.int16)
    date_token = int(outer_date)
    for name, record in dataset.records.items():
        mask = names == name
        legal = len(record.trial_values) - MAX_PREFIX + 1
        _need(legal >= 2, f"{name}: M7 does not leave a causal query trial")
        local_pos = positions[mask]
        for epoch in range(FIXED_EPOCHS):
            values = (date_token + (epoch + 1) * 1_000_003 + (local_pos + 1) * 97_409 + order[mask] * 65_537) % legal
            schedule[epoch, mask] = values.astype(np.int16)
    return schedule


def prepare_source_authority(
    data_root: Path,
    predecessor_root: Path,
    experiment3_root: Path,
    result_root: Path,
) -> dict[str, Any]:
    load_attempt(result_root)
    root = result_root.resolve()
    predecessor = validate_predecessors(experiment3_root, predecessor_root)
    publish_json(root / "predecessor_authority.json", predecessor)
    rows = []
    for date in CONFIRMATORY_DATES:
        directory = root / "source_authority" / date
        records, _plan, cache_body, carriers, normalizer, order, _old_schedule, predecessor_sha = _load_date_materialization(
            data_root, predecessor_root, date
        )
        dataset = SourceDataset(records, cache_body, carriers, normalizer)
        m7 = _m7_schedule(dataset, order, date)
        prefixes = _prefix_cycle(date, FIXED_EPOCHS, len(order) // BATCH_SIZE)
        arrays_sha = publish_npz(directory / "schedule.npz", order=order, m7_starts=m7, c1_prefixes=prefixes)
        body = {
            "schema": f"{SCHEMA}_date_source_authority",
            "status": STATUS_SOURCE,
            "outer_date": date,
            "source_sessions": list(records),
            "source_input_sha256": [records[name].input_sha256 for name in records],
            "predecessor_date_source_authority_sha256": predecessor_sha,
            "predecessor_carrier_cache_sha256": cache_body["cache_sha256"],
            "normalizer_s_src": normalizer,
            "epochs": FIXED_EPOCHS,
            "batch_size": BATCH_SIZE,
            "batches_per_epoch": int(len(order) // BATCH_SIZE),
            "steps_per_arm": int(FIXED_EPOCHS * len(order) // BATCH_SIZE),
            "m7_prefix": True,
            "c1_cycle": list(C1_CYCLE),
            "schedule_file_sha256": arrays_sha,
            "array_sha256": {
                "order": array_sha256(order),
                "m7_starts": array_sha256(m7),
                "c1_prefixes": array_sha256(prefixes),
            },
            "minimum_trial_count": min(len(record.trial_values) for record in records.values()),
            "target_recordings_opened": 0,
            "target_bytes_read": 0,
        }
        authority_sha = publish_json(directory / "authority.json", body)
        rows.append({"outer_date": date, "relative": str((directory / "authority.json").relative_to(root)), "sha256": authority_sha})
    top = {
        "schema": f"{SCHEMA}_source_authority",
        "status": STATUS_SOURCE,
        "date_order": list(CONFIRMATORY_DATES),
        "dates": rows,
        "target_recordings_opened": 0,
        "target_bytes_read": 0,
    }
    publish_json(root / "source_authority.json", top)
    return top


def _load_schedule(result_root: Path, date: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, str]:
    directory = result_root.resolve() / "source_authority" / date
    body, digest = _load_json(directory / "authority.json", f"{SCHEMA}_date_source_authority")
    _need(body.get("status") == STATUS_SOURCE and body.get("target_bytes_read") == 0, "source authority drift")
    path = directory / "schedule.npz"
    _need(verify_sidecar(path) == body["schedule_file_sha256"], "schedule file drift")
    with np.load(path, allow_pickle=False) as values:
        order = np.asarray(values["order"], np.int64)
        starts = np.asarray(values["m7_starts"], np.int16)
        prefixes = np.asarray(values["c1_prefixes"], np.int8)
    for name, value in (("order", order), ("m7_starts", starts), ("c1_prefixes", prefixes)):
        _need(array_sha256(value) == body["array_sha256"][name], f"{name} digest drift")
    return body, order, starts, prefixes, digest


def _batch(dataset: SourceDataset, rows: Sequence[int], starts: Sequence[int], prefix: int):
    _need(prefix in BUDGETS and len(rows) == len(starts), "batch prefix/schema drift")
    names = {dataset.windows[int(row)][0] for row in rows}
    _need(len(names) == 1, "batch mixes sessions")
    xs, ys, identities, carrier_rows = [], [], [], []
    for row, support_start in zip(rows, starts, strict=True):
        name, start = dataset.windows[int(row)]
        values = dataset.records[name].trial_values[int(support_start):int(support_start) + MAX_PREFIX]
        _need(len(values) == MAX_PREFIX, "scheduled M7 prefix is incomplete")
        xs.append(dataset.neural[name][start:start + WINDOW_SIZE])
        ys.append(dataset.target[name][start:start + WINDOW_SIZE])
        identities.append(np.stack([dataset.identity[(name, float(value))] for value in values[:prefix]], axis=0))
        carrier_rows.append(dataset.carriers[(name, int(support_start))])
    return (
        np.ascontiguousarray(np.stack(xs), np.float32),
        np.ascontiguousarray(np.stack(ys), np.float32),
        np.ascontiguousarray(np.stack(identities), np.float32),
        np.ascontiguousarray(np.stack(carrier_rows), np.float32),
        next(iter(names)),
    )


def common_config() -> dict[str, Any]:
    config = model_config()
    return {
        "schema": f"{SCHEMA}_common_config",
        "base": config,
        "t0_prefix": 7,
        "c1_cycle": list(C1_CYCLE),
        "warm_start": False,
        "terminal_epoch_zero_based": 49,
        "arm_timeout_seconds": ARM_TIMEOUT_SECONDS,
    }


def run_arm(
    data_root: Path,
    predecessor_root: Path,
    result_root: Path,
    outer_date: str,
    arm: str,
    physical_gpu: int,
    *,
    smoke: bool = False,
    max_steps: int = 20,
) -> dict[str, Any]:
    _need(outer_date in CONFIRMATORY_DATES and arm in ARMS, "arm/date drift")
    cell_root = result_root.resolve() / ("smoke" if smoke else "pairs") / outer_date / arm
    _need(not cell_root.exists(), f"cell exists: {cell_root}")
    publish_json(cell_root / "attempt.json", {
        "schema": SCHEMA, "artifact": "arm_attempt", "outer_date": outer_date, "arm": arm,
        "smoke": smoke, "physical_gpu": int(physical_gpu), "created_at_utc": utc_now(),
        "target_recordings_opened": 0, "target_bytes_read": 0, "warm_start": False,
    })
    started = time.monotonic()
    try:
        import torch
        import src.models.components.h1_carrierid_spint as carrier_module

        _need(torch.cuda.is_available(), "training arm requires CUDA")
        attempt = load_attempt(result_root)
        authority, order, starts, prefixes, authority_sha = _load_schedule(result_root, outer_date)
        records, _plan, cache_body, carriers, normalizer, old_order, _old_schedule, predecessor_sha = _load_date_materialization(
            data_root, predecessor_root, outer_date
        )
        _need(np.array_equal(order, old_order), "predecessor batch order drift")
        dataset = SourceDataset(records, cache_body, carriers, normalizer)
        config_sha = publish_json(cell_root / "config.json", {**common_config(), "arm": arm})
        device = "cuda:0"
        model = _new_model(device)
        _need(sum(parameter.numel() for parameter in model.parameters()) == MODEL_PARAMETERS, "parameter count drift")
        model.train()
        initial_sha = state_hash(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=5.0e-5, weight_decay=0.0)
        original_uniform = carrier_module.random.uniform
        p_digest = hashlib.sha256()
        p_count = 0

        def tracked_uniform(low: float, high: float) -> float:
            nonlocal p_count
            value = original_uniform(low, high)
            p_digest.update(struct.pack("!d", float(value)))
            p_count += 1
            return value

        carrier_module.random.uniform = tracked_uniform
        global_step = 0
        epoch_rows = []
        loss_first = None
        loss_last = None
        prefix_counts = {4: 0, 5: 0, 7: 0}
        peak_reserved = 0
        training_started = time.monotonic()
        try:
            epochs = 1 if smoke else FIXED_EPOCHS
            for epoch in range(epochs):
                losses = []
                for batch_index, offset in enumerate(range(0, len(order), BATCH_SIZE)):
                    if smoke and global_step >= max_steps:
                        break
                    rows = order[offset:offset + BATCH_SIZE]
                    m7_starts = starts[epoch, offset:offset + BATCH_SIZE]
                    prefix = 7 if arm == "t0" else int(prefixes[epoch, batch_index])
                    neural, target, identity, carrier, _session = _batch(dataset, rows, m7_starts, prefix)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(
                        torch.as_tensor(neural, dtype=torch.float32, device=device),
                        calib_trialized_neural_features=torch.as_tensor(identity, dtype=torch.float32, device=device),
                        carrier=torch.as_tensor(carrier, dtype=torch.float32, device=device),
                    )
                    loss = torch.nn.functional.mse_loss(
                        prediction[:, -1:, :] / 20.0,
                        torch.as_tensor(target, dtype=torch.float32, device=device)[:, -1:, :],
                    )
                    _need(bool(torch.isfinite(loss)), "nonfinite loss")
                    loss.backward()
                    _need(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()), "nonfinite gradient")
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
                    print(f"EPOCH_END outer_date={outer_date} arm={arm} epoch={epoch} steps={len(losses)} mean_loss={row['mean_loss']:.9g}", flush=True)
                if smoke and global_step >= max_steps:
                    break
        finally:
            carrier_module.random.uniform = original_uniform
        training_elapsed = time.monotonic() - training_started
        expected_steps = max_steps if smoke else int(authority["steps_per_arm"])
        _need(global_step == expected_steps and p_count == global_step, "step/dropout count drift")
        _need(_finite_optimizer(optimizer), "Adam state is nonfinite")
        if smoke and arm == "c1":
            _need(all(prefix_counts[value] > 0 for value in C1_CYCLE), "smoke did not cover full prefix cycle")
        terminal_state = state_hash(model.state_dict())
        checkpoint = None
        if not smoke:
            checkpoint_path = cell_root / "epoch_049.ckpt"
            payload = {
                "schema": f"{SCHEMA}_checkpoint",
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "metadata": {
                    "outer_date": outer_date, "arm": arm, "epoch_zero_based": 49,
                    "global_step": global_step, "warm_start": False, "checkpoint_selection": False,
                    "initial_state_sha256": initial_sha, "terminal_state_sha256": terminal_state,
                    "dropout_probability_sha256": p_digest.hexdigest(), "dropout_probability_count": p_count,
                    "source_authority_sha256": authority_sha, "predecessor_source_authority_sha256": predecessor_sha,
                    "config_sha256": config_sha, "experiment_attempt_sha256": sha256_file(result_root.resolve() / "attempt.json"),
                    "code_closure_sha256": attempt["code_closure_sha256"],
                    "target_recordings_opened": 0, "target_bytes_read": 0,
                    "target_optimizer_steps": 0, "target_backward_steps": 0,
                },
            }
            import torch as _torch
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            _torch.save(payload, checkpoint_path)
            checkpoint_path.chmod(0o444)
            digest = sha256_file(checkpoint_path)
            _publish_bytes(
                checkpoint_path.with_name(checkpoint_path.name + ".sha256"),
                f"{digest}  {checkpoint_path.name}\n".encode("ascii"),
                sidecar=False,
            )
            checkpoint = {"relative": str(checkpoint_path.relative_to(result_root.resolve())), "sha256": digest}
        gpu = _gpu_profile(physical_gpu)
        body = {
            "schema": SCHEMA,
            "status": STATUS_SMOKE if smoke else STATUS_ARM,
            "outer_date": outer_date, "arm": arm, "smoke": smoke,
            "gpu": {**gpu, "physical_index": int(physical_gpu)},
            "source_authority_sha256": authority_sha,
            "predecessor_source_authority_sha256": predecessor_sha,
            "config_sha256": config_sha,
            "initial_state_sha256": initial_sha,
            "terminal_state_sha256": terminal_state,
            "global_step": global_step,
            "epoch_zero_based": 0 if smoke else 49,
            "dropout_probability_sha256": p_digest.hexdigest(),
            "dropout_probability_count": p_count,
            "prefix_step_counts": {str(key): value for key, value in prefix_counts.items()},
            "loss_first": loss_first, "loss_last": loss_last,
            "epochs": epoch_rows,
            "peak_memory_reserved_bytes": peak_reserved,
            "training_elapsed_seconds": training_elapsed,
            "checkpoint": checkpoint,
            "target_recordings_opened": 0, "target_bytes_read": 0,
            "target_optimizer_steps": 0, "target_backward_steps": 0,
            "elapsed_seconds": time.monotonic() - started,
            "finished_at_utc": utc_now(),
        }
        publish_json(cell_root / "terminal.json", body)
        return body
    except BaseException as error:
        try:
            publish_json(cell_root / "failure.json", {
                "schema": SCHEMA, "status": "FAIL_ARM_NO_AUTOMATIC_RETRY", "outer_date": outer_date,
                "arm": arm, "smoke": smoke, "physical_gpu": int(physical_gpu),
                "error_type": type(error).__name__, "error": str(error),
                "target_recordings_opened": 0, "target_bytes_read": 0,
                "elapsed_seconds": time.monotonic() - started,
            })
        except BaseException:
            pass
        raise


def verify_pair(result_root: Path, outer_date: str, *, smoke: bool = False) -> dict[str, Any]:
    root = result_root.resolve() / ("smoke" if smoke else "pairs") / outer_date
    rows = {}
    for arm in ARMS:
        terminal, digest = _load_json(root / arm / "terminal.json", SCHEMA)
        _need(terminal.get("status") == (STATUS_SMOKE if smoke else STATUS_ARM), f"{arm} terminal drift")
        rows[arm] = {**terminal, "terminal_sha256": digest}
    t0, c1 = rows["t0"], rows["c1"]
    for field in ("gpu", "source_authority_sha256", "initial_state_sha256", "global_step", "dropout_probability_sha256", "dropout_probability_count"):
        _need(t0[field] == c1[field], f"paired {field} mismatch")
    _need(t0["prefix_step_counts"]["7"] == t0["global_step"], "T0 did not remain M7")
    if not smoke:
        import torch
        for arm in ARMS:
            checkpoint = result_root.resolve() / rows[arm]["checkpoint"]["relative"]
            _need(verify_sidecar(checkpoint) == rows[arm]["checkpoint"]["sha256"], "checkpoint sidecar drift")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
            _need(isinstance(metadata, Mapping), "checkpoint metadata missing")
            _need(state_hash(payload["state_dict"]) == rows[arm]["terminal_state_sha256"], "checkpoint state drift")
            for field in ("outer_date", "arm", "global_step", "initial_state_sha256", "terminal_state_sha256",
                          "dropout_probability_sha256", "dropout_probability_count", "source_authority_sha256",
                          "config_sha256"):
                _need(metadata.get(field) == rows[arm].get(field), f"checkpoint metadata drift: {arm}/{field}")
            _need(metadata.get("epoch_zero_based") == 49 and metadata.get("warm_start") is False
                  and metadata.get("checkpoint_selection") is False, "checkpoint selection/epoch drift")
            for field in ("target_recordings_opened", "target_bytes_read", "target_optimizer_steps", "target_backward_steps"):
                _need(metadata.get(field) == 0, f"checkpoint target activity: {arm}/{field}")
            config, config_sha = _load_json(root / arm / "config.json", f"{SCHEMA}_common_config")
            _need(config_sha == rows[arm]["config_sha256"] and config.get("arm") == arm, "arm config drift")
    body = {
        "schema": f"{SCHEMA}_paired_integrity",
        "status": "PASS_PAIRED_SMOKE_INTEGRITY" if smoke else "PASS_PAIRED_EPOCH49_INTEGRITY",
        "outer_date": outer_date,
        "smoke": smoke,
        "gpu": t0["gpu"],
        "initial_state_sha256": t0["initial_state_sha256"],
        "dropout_probability_sha256": t0["dropout_probability_sha256"],
        "global_step_per_arm": t0["global_step"],
        "t0_terminal_sha256": rows["t0"]["terminal_sha256"],
        "c1_terminal_sha256": rows["c1"]["terminal_sha256"],
        "target_recordings_opened": 0,
        "target_bytes_read": 0,
    }
    publish_json(root / "paired_integrity.json", body)
    return body


def verify_all_pairs(result_root: Path) -> dict[str, Any]:
    rows = []
    for date in CONFIRMATORY_DATES:
        body = verify_pair(result_root, date, smoke=False)
        rows.append({"outer_date": date, "sha256": verify_sidecar(result_root.resolve() / "pairs" / date / "paired_integrity.json")})
    body = {
        "schema": f"{SCHEMA}_ten_model_integrity",
        "status": STATUS_PAIRS,
        "date_order": list(CONFIRMATORY_DATES),
        "pairs": rows,
        "models": 10,
        "target_recordings_opened": 0,
        "target_bytes_read": 0,
    }
    publish_json(result_root.resolve() / "paired_integrity.json", body)
    return body


def _load_plan_normalizer(predecessor_root: Path, date: str):
    from src.h1_hc_date_lodo_regen_v1 import RegenPlan
    directory = predecessor_root.resolve() / "source_authority" / date
    body, _ = _load_json(directory / "plan.json")
    path = directory / "plan.npz"
    _need(verify_sidecar(path) == body["arrays_file_sha256"], "plan arrays drift")
    with np.load(path, allow_pickle=False) as values:
        plan = RegenPlan(
            date, tuple(body["source_sessions"]), tuple(body["source_input_sha256"]),
            np.asarray(values["mean"], np.float64), np.asarray(values["scale"], np.float64),
            np.asarray(values["pcs"], np.float64), int(values["q"].item()), float(values["lambda"].item()),
            np.asarray(values["U"], np.float64), np.asarray(values["mu"], np.float64), float(values["tau2"].item()),
            str(body["selection_sha256"]), str(body["transform_sha256"]),
        )
    normalizer, _ = _load_json(directory / "normalizer.json")
    return plan, float(normalizer["s_src"])


def _target_records(data_root: Path, date: str, access: dict[str, Any]) -> dict[str, H1PilotRecord]:
    paths = index_heldin_calib(data_root)
    expected = target_sessions_for_date(date)
    records = {}
    files = []
    for session in expected:
        path = paths[session]
        size = int(path.stat().st_size)
        access["target_recordings_opened"] += 1
        access["target_bytes_read"] += size
        access["target_sessions_opened"].append(session)
        record = load_record(path)
        _need(record.date == date and record.session_name == session, "target partition drift")
        records[session] = record
        files.append({"session": session, "filename": path.name, "bytes": size, "sha256": record.input_sha256})
    access["files"] = files
    return records


def _load_arm_model(result_root: Path, date: str, arm: str, device: str):
    import torch
    terminal, terminal_sha = _load_json(result_root.resolve() / "pairs" / date / arm / "terminal.json", SCHEMA)
    _need(terminal.get("status") == STATUS_ARM and terminal.get("epoch_zero_based") == 49, "arm terminal drift")
    path = result_root.resolve() / terminal["checkpoint"]["relative"]
    _need(verify_sidecar(path) == terminal["checkpoint"]["sha256"], "arm checkpoint drift")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = _new_model(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    before = state_hash(model.state_dict())
    _need(before == terminal["terminal_state_sha256"], "checkpoint state drift")
    return model, before, terminal, terminal_sha


def _infer_budget(model: Any, record: H1PilotRecord, plan: Any, normalizer: float, budget: int, device: str):
    import torch
    _need(len(record.trial_values) >= budget + 1, f"{record.session_name}: no post-M{budget} query")
    support = tuple(float(value) for value in record.trial_values[:budget])
    next_value = float(record.trial_values[budget])
    boundary_rows = np.flatnonzero(record.eval_mask & np.isfinite(record.trial_num) & (record.trial_num == next_value))
    _need(boundary_rows.size > 0, "missing causal query boundary")
    boundary = int(boundary_rows[0])
    last_start = int(record.neural.shape[0] - WINDOW_SIZE)
    _need(last_start >= boundary, "no complete causal window")
    starts = np.arange(boundary, last_start + 1, dtype=np.int64)
    output_bins = starts + WINDOW_SIZE - 1
    score_mask = np.asarray(record.eval_mask[output_bins], bool)
    target = np.asarray(record.velocity[output_bins], np.float32)
    identity = np.ascontiguousarray(np.stack([interpolate_trial_identity(record, value) for value in support]), np.float32)
    carrier = np.ascontiguousarray(
        fit_frozen_carrier(record, plan, support[:4])["carrier"] / max(normalizer, NORMALIZER_FLOOR), np.float32
    )
    prediction = np.empty((len(starts), 7), np.float32)
    identity_one = torch.as_tensor(identity, dtype=torch.float32, device=device).unsqueeze(0)
    carrier_one = torch.as_tensor(carrier, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        for offset in range(0, len(starts), BATCH_SIZE):
            selected = starts[offset:offset + BATCH_SIZE]
            neural = np.ascontiguousarray(np.stack([record.neural[int(start):int(start) + WINDOW_SIZE] for start in selected]), np.float32)
            count = len(selected)
            output = model(
                torch.as_tensor(neural, dtype=torch.float32, device=device),
                calib_trialized_neural_features=identity_one.expand(count, -1, -1, -1),
                carrier=carrier_one.expand(count, -1, -1),
            )
            values = np.asarray((output[:, -1, :] / 20.0).cpu(), np.float32)
            _need(np.isfinite(values).all(), "nonfinite target prediction")
            prediction[offset:offset + count] = values
    _need(int(score_mask.sum()) > 1, "insufficient target score rows")
    return {
        "prediction": prediction,
        "target": target,
        "score_mask": score_mask,
        "output_bins": output_bins,
        "support": list(support),
        "next_trial": next_value,
        "identity_sha256": array_sha256(identity),
        "carrier_sha256": array_sha256(carrier),
    }


def run_evaluation_cell(data_root: Path, predecessor_root: Path, result_root: Path, date: str, physical_gpu: int) -> dict[str, Any]:
    directory = result_root.resolve() / "evaluation" / date
    _need(not directory.exists(), f"evaluation exists: {date}")
    publish_json(directory / "attempt.json", {
        "schema": SCHEMA, "artifact": "evaluation_attempt", "outer_date": date,
        "physical_gpu": int(physical_gpu), "created_at_utc": utc_now(),
        "target_recordings_opened": 0, "target_bytes_read": 0,
        "optimizer_steps": 0, "backward_steps": 0,
    })
    access = {"outer_date": date, "target_recordings_opened": 0, "target_bytes_read": 0, "target_sessions_opened": [], "files": []}
    started = time.monotonic()
    try:
        import torch
        _need(torch.cuda.is_available(), "evaluation requires CUDA")
        _load_json(result_root.resolve() / "paired_integrity.json", f"{SCHEMA}_ten_model_integrity")
        plan, normalizer = _load_plan_normalizer(predecessor_root, date)
        models = {}
        terminals = {}
        states = {}
        for arm in ARMS:
            models[arm], states[arm], terminals[arm], _ = _load_arm_model(result_root, date, arm, "cuda:0")
        records = _target_records(data_root, date, access)
        arrays = {}
        supports = {}
        metrics: dict[str, Any] = {"schema": f"{SCHEMA}_metrics", "outer_date": date, "budgets": {}}
        for budget in BUDGETS:
            by_arm = {}
            by_recording = {}
            for arm in ARMS:
                scores = {}
                for index, (session, record) in enumerate(records.items()):
                    cache = _infer_budget(models[arm], record, plan, normalizer, budget, "cuda:0")
                    prefix = f"m{budget}_{arm}_{index}"
                    arrays[f"{prefix}_prediction"] = cache["prediction"]
                    arrays[f"{prefix}_target"] = cache["target"]
                    arrays[f"{prefix}_score_mask"] = cache["score_mask"]
                    arrays[f"{prefix}_output_bins"] = cache["output_bins"]
                    mask = cache["score_mask"]
                    scores[session] = variance_weighted_r2(cache["target"][mask], cache["prediction"][mask])
                    supports[f"m{budget}_{arm}_{session}"] = {key: cache[key] for key in ("support", "next_trial", "identity_sha256", "carrier_sha256")}
                by_recording[arm] = scores
                by_arm[arm] = float(np.mean(list(scores.values()), dtype=np.float64))
            metrics["budgets"][str(budget)] = {
                "equal_recording_mean_r2": by_arm,
                "delta_c1_minus_t0": by_arm["c1"] - by_arm["t0"],
                "per_recording_r2": by_recording,
            }
        cache_path = directory / "prediction_cache.npz"
        cache_sha = publish_npz(cache_path, **arrays)
        cache_manifest_sha = publish_json(directory / "prediction_cache.json", {
            "schema": f"{SCHEMA}_prediction_cache", "outer_date": date,
            "sessions": list(records), "arrays_file_sha256": cache_sha,
            "array_sha256": {key: array_sha256(value) for key, value in arrays.items()},
            "array_shape": {key: list(value.shape) for key, value in arrays.items()},
            "support": supports,
        })
        metrics_sha = publish_json(directory / "metrics.json", metrics)
        for arm in ARMS:
            _need(state_hash(models[arm].state_dict()) == states[arm], f"{arm} state changed during evaluation")
        audit = {**access, "expected_sessions": list(target_sessions_for_date(date)), "authorized_outer_date_only": True,
                 "optimizer_steps": 0, "backward_steps": 0, "model_updates": 0, "target_driven_selection": False}
        audit_sha = publish_json(directory / "target_access.json", audit)
        body = {
            "schema": SCHEMA, "status": STATUS_EVAL, "outer_date": date,
            "gpu": {**_gpu_profile(physical_gpu), "physical_index": int(physical_gpu)},
            "metrics": metrics, "metrics_sha256": metrics_sha,
            "prediction_cache_sha256": cache_sha, "prediction_cache_manifest_sha256": cache_manifest_sha,
            "target_access": audit, "target_access_sha256": audit_sha,
            "model_state_immutable": True, "optimizer_steps": 0, "backward_steps": 0, "model_updates": 0,
            "selection_performed": False, "elapsed_seconds": time.monotonic() - started, "finished_at_utc": utc_now(),
        }
        publish_json(directory / "terminal.json", body)
        return body
    except BaseException as error:
        try:
            publish_json(directory / "failure.json", {
                "schema": SCHEMA, "status": "FAIL_EVALUATION_NO_AUTOMATIC_RETRY", "outer_date": date,
                "error_type": type(error).__name__, "error": str(error), "target_access": access,
                "optimizer_steps": 0, "backward_steps": 0, "elapsed_seconds": time.monotonic() - started,
            })
        except BaseException:
            pass
        raise


def transfer_decision(date_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    deltas = {str(budget): {date: float(date_metrics[date]["budgets"][str(budget)]["delta_c1_minus_t0"])
                            for date in CONFIRMATORY_DATES} for budget in BUDGETS}
    means = {budget: float(np.mean(list(rows.values()), dtype=np.float64)) for budget, rows in deltas.items()}
    positive = sum(value > 0.0 for value in deltas["4"].values())
    passed = means["4"] >= 0.01 and positive >= 4 and means["5"] >= -0.01 and means["7"] >= -0.01
    return {
        "verdict": "PASS_TRANSFER" if passed else "COMPLETE_NO_TRANSFER",
        "equal_date_delta_r2": means,
        "date_delta_r2": deltas,
        "m4_positive_dates": positive,
        "thresholds": {"m4_delta_min": 0.01, "m4_positive_dates_min": 4, "m5_delta_min": -0.01, "m7_delta_min": -0.01},
    }


def _verify_evaluation_cache(directory: Path, terminal: Mapping[str, Any], date: str) -> dict[str, Any]:
    manifest, manifest_sha = _load_json(directory / "prediction_cache.json", f"{SCHEMA}_prediction_cache")
    _need(manifest_sha == terminal.get("prediction_cache_manifest_sha256"), "prediction manifest SHA drift")
    path = directory / "prediction_cache.npz"
    _need(verify_sidecar(path) == terminal.get("prediction_cache_sha256") == manifest.get("arrays_file_sha256"),
          "prediction cache SHA drift")
    with np.load(path, allow_pickle=False) as values:
        arrays = {name: np.asarray(values[name]) for name in values.files}
    _need(set(arrays) == set(manifest.get("array_sha256", {})), "prediction cache array set drift")
    for name, value in arrays.items():
        _need(array_sha256(value) == manifest["array_sha256"][name], f"prediction array digest drift: {name}")
        _need(list(value.shape) == manifest["array_shape"][name], f"prediction array shape drift: {name}")
    sessions = target_sessions_for_date(date)
    recomputed: dict[str, Any] = {"schema": f"{SCHEMA}_metrics", "outer_date": date, "budgets": {}}
    for budget in BUDGETS:
        by_arm = {}
        by_recording = {}
        for arm in ARMS:
            scores = {}
            for index, session in enumerate(sessions):
                prefix = f"m{budget}_{arm}_{index}"
                prediction = arrays[f"{prefix}_prediction"]
                target = arrays[f"{prefix}_target"]
                mask = np.asarray(arrays[f"{prefix}_score_mask"], bool)
                _need(np.array_equal(target, arrays[f"m{budget}_{ARMS[0]}_{index}_target"]), "paired target bytes drift")
                _need(np.array_equal(mask, arrays[f"m{budget}_{ARMS[0]}_{index}_score_mask"]), "paired mask drift")
                _need(np.array_equal(arrays[f"{prefix}_output_bins"], arrays[f"m{budget}_{ARMS[0]}_{index}_output_bins"]),
                      "paired output-bin drift")
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


def verify_terminal(predecessor_root: Path, result_root: Path) -> dict[str, Any]:
    root = result_root.resolve()
    attempt = load_attempt(root)
    attempt_sha = verify_sidecar(root / "attempt.json")
    pair_body, pair_sha = _load_json(root / "paired_integrity.json", f"{SCHEMA}_ten_model_integrity")
    _need(pair_body.get("status") == STATUS_PAIRS and pair_body.get("models") == 10, "ten-model integrity drift")
    metrics = {}
    rows = []
    total_recordings = 0
    total_bytes = 0
    for date in CONFIRMATORY_DATES:
        terminal, terminal_sha = _load_json(root / "evaluation" / date / "terminal.json", SCHEMA)
        _need(terminal.get("status") == STATUS_EVAL and terminal.get("model_state_immutable") is True, "evaluation terminal drift")
        _need(terminal.get("optimizer_steps") == terminal.get("backward_steps") == terminal.get("model_updates") == 0, "target model update")
        audit = terminal["target_access"]
        _need(tuple(audit["target_sessions_opened"]) == target_sessions_for_date(date), "target isolation drift")
        _need(audit.get("target_driven_selection") is False, "target selection recorded")
        metrics[date] = _verify_evaluation_cache(root / "evaluation" / date, terminal, date)
        total_recordings += int(audit["target_recordings_opened"])
        total_bytes += int(audit["target_bytes_read"])
        rows.append({"outer_date": date, "terminal_sha256": terminal_sha, "gpu": terminal["gpu"], "metrics": terminal["metrics"]})
    decision = transfer_decision(metrics)
    status = STATUS_PASS if decision["verdict"] == "PASS_TRANSFER" else STATUS_NO_TRANSFER
    body = {
        "schema": SCHEMA, "status": status, "finished_at_utc": utc_now(),
        "date_order": list(CONFIRMATORY_DATES), "arms": list(ARMS), "budgets": list(BUDGETS),
        "decision": decision, "experiment_attempt_sha256": attempt_sha,
        "code_closure_sha256": attempt["code_closure_sha256"], "paired_integrity_sha256": pair_sha,
        "cells": rows, "target_recordings_opened": total_recordings, "target_bytes_read": total_bytes,
        "target_optimizer_steps": 0, "target_backward_steps": 0, "target_model_updates": 0,
        "target_driven_selection": False,
        "claim": "fixed H1 M7/M5/M4 CAL-AUG prefix-cycle on five matched date-LODO pairs only",
    }
    terminal_sha = publish_json(root / "terminal.json", body)
    lines = [
        "# H1 CAL-AUG Prefix-Cycle V1", "", f"- Status: `{status}`",
        f"- M4 equal-date delta: `{decision['equal_date_delta_r2']['4']:+.9f}`",
        f"- M4 positive dates: `{decision['m4_positive_dates']}/5`",
        f"- M5 safety delta: `{decision['equal_date_delta_r2']['5']:+.9f}`",
        f"- M7 safety delta: `{decision['equal_date_delta_r2']['7']:+.9f}`",
        "- Target optimizer/backward/model updates and target-driven selections: `0`.", "",
        "| Date | M4 delta C1-T0 | M5 delta | M7 delta | GPU UUID |", "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        d = row["outer_date"]
        values = row["metrics"]["budgets"]
        lines.append(f"| {d} | {values['4']['delta_c1_minus_t0']:+.9f} | {values['5']['delta_c1_minus_t0']:+.9f} | {values['7']['delta_c1_minus_t0']:+.9f} | `{row['gpu']['uuid']}` |")
    lines.extend(["", f"Terminal SHA-256: `{terminal_sha}`", ""])
    publish_text(root / "EXPERIMENT_RECORD.md", "\n".join(lines))
    return body


__all__ = (
    "ARMS", "ARM_TIMEOUT_SECONDS", "BUDGETS", "C1_CYCLE", "RESOURCE_WAIT_SECONDS", "SCHEMA",
    "STATUS_ARM", "STATUS_EVAL", "STATUS_NO_TRANSFER", "STATUS_PAIRS", "STATUS_PASS", "STATUS_SMOKE",
    "STATUS_SOURCE", "common_config", "create_attempt", "dry_plan", "load_attempt", "prepare_source_authority",
    "run_arm", "run_evaluation_cell", "transfer_decision", "validate_predecessors", "verify_all_pairs",
    "verify_pair", "verify_terminal",
)
