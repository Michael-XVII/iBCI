"""Five-date frozen H-C evaluation of causal activity-memory policies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import time
from typing import Any, Mapping

import numpy as np

from h1_date_lodo_activity_headroom_v1.evaluate import DATA_RELATIVE, _DateLodoDatasetAdapter, _bootstrap, _load_bound_authority, _load_model
from h1_date_lodo_activity_headroom_v1.plan import AUTHORITIES
from m1_h1_activity_headroom_v1.core import ActivityHeadroomError, array_digest, forward_with_cached_identity, identity_from_raw_trials, variance_weighted_r2
from m1_h1_activity_headroom_v1.h1 import BATCH_SIZE, OUTPUTS, SUPPORT_TRIALS, WINDOW, _output_trial_positions
from m1_h1_activity_headroom_v1.m1 import write_once

from .plan import ARM_ORDER, DATE_ORDER, PREDECESSOR_RELATIVE, PREDECESSOR_SHA256, decision, selection_for_arm


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ActivityHeadroomError(message)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_predecessor(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    path = root / PREDECESSOR_RELATIVE
    sidecar = path.with_name(path.name + ".sha256")
    _need(path.is_file() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o444 and _sha_file(path) == PREDECESSOR_SHA256, "predecessor body drift")
    _need(sidecar.is_file() and not sidecar.is_symlink() and stat.S_IMODE(sidecar.stat().st_mode) == 0o444 and sidecar.read_text(encoding="ascii") == f"{PREDECESSOR_SHA256}  {path.name}\n", "predecessor sidecar drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _need(payload.get("date_order") == list(DATE_ORDER) and payload.get("target_optimizer_backward_update") == 0, "predecessor contract drift")
    result = {}
    for date, row in zip(DATE_ORDER, payload.get("date_results", []), strict=True):
        _need(row.get("outer_date") == date, "predecessor date drift")
        arms = {item.get("arm"): item for item in row.get("results", []) if isinstance(item, Mapping)}
        _need("CAUSAL_GROWING_CAP30" in arms and "FULL_SESSION_ORACLE" in arms, "predecessor arm drift")
        result[date] = {name: dict(arms[name]) for name in ("CAUSAL_GROWING_CAP30", "FULL_SESSION_ORACLE")}
    return result


def _score_arm(*, model: Any, dataset: Any, activity: dict[str, np.ndarray], output_trials: tuple[int, ...], arm: str, device: str) -> dict[str, Any]:
    import torch

    predictions = np.empty((len(dataset), OUTPUTS), dtype=np.float32)
    targets = np.empty((len(dataset), OUTPUTS), dtype=np.float32)
    session_names = tuple(session for session, _ in dataset.window_indices)
    selections = tuple(selection_for_arm(arm, output_trial_index=output_trials[index]) for index in range(len(dataset)))
    groups: dict[tuple[str, tuple[int, ...]], list[int]] = {}
    for index, selection in enumerate(selections):
        groups.setdefault((session_names[index], selection), []).append(index)
    forwards = 0
    started = time.monotonic()
    with torch.no_grad():
        for (session, selection), rows in groups.items():
            carrier = np.asarray(dataset.support[session].carriers["full"], dtype=np.float32)
            identity = identity_from_raw_trials(model.net, activity[session], selection, family="h1", device=device, carrier=carrier)
            for offset in range(0, len(rows), BATCH_SIZE):
                batch_rows = rows[offset:offset + BATCH_SIZE]
                xs, ys = [], []
                for row in batch_rows:
                    current_session, start = dataset.window_indices[row]
                    _need(current_session == session, "grouped session drift")
                    end = int(start) + WINDOW
                    xs.append(dataset.records[session].neural[int(start):end])
                    ys.append(dataset.records[session].velocity[end - 1])
                output = forward_with_cached_identity(model.net, np.ascontiguousarray(np.stack(xs), dtype=np.float32), identity)
                predictions[np.asarray(batch_rows, dtype=np.int64)] = np.ascontiguousarray((output[:, -1, :] / float(model.hparams.behavior_scaling_factor)).detach().cpu().numpy(), dtype=np.float32)
                targets[np.asarray(batch_rows, dtype=np.int64)] = np.ascontiguousarray(np.stack(ys), dtype=np.float32)
                forwards += 1
    per_recording = {}
    for session in dataset.records:
        mask = np.asarray([name == session for name in session_names], dtype=bool)
        per_recording[session] = {"n_windows": int(mask.sum()), "r2": variance_weighted_r2(predictions[mask], targets[mask])}
    return {
        "arm": arm,
        "causal": True,
        "label_free": True,
        "h1_carrier_unchanged": True,
        "pooled_r2": variance_weighted_r2(predictions, targets),
        "equal_recording_mean_r2": float(np.mean([row["r2"] for row in per_recording.values()], dtype=np.float64)),
        "per_recording": per_recording,
        "n_windows": len(dataset),
        "prediction_sha256": array_digest(predictions),
        "target_sha256": array_digest(targets),
        "unique_activity_states": len(groups),
        "activity_cardinality_min": min(map(len, selections)),
        "activity_cardinality_max": max(map(len, selections)),
        "forward_batches": forwards,
        "elapsed_seconds": time.monotonic() - started,
    }


def _evaluate_date(root: Path, date: str, *, predecessor: Mapping[str, Mapping[str, Any]], device: str) -> dict[str, Any]:
    import torch
    from src.data.h1_carrierid_date_lodo_target import H1CarrierIdDateLodoStrictTargetDataset, load_outer_date_target_records, load_target_dependencies
    from src.data.h1_m4_eb_pilot import interpolate_trial_identity
    from src.h1_m4_cce_contract import state_hash

    authority = AUTHORITIES[date]
    terminal, checkpoint, config, source_manifest_path = _load_bound_authority(root, authority)
    plan, normalizer, _source_manifest = load_target_dependencies(source_manifest_path, outer_date=date)
    model, state_before = _load_model(checkpoint, config, terminal, device=device)
    records = load_outer_date_target_records(root / DATA_RELATIVE, outer_date=date)
    dataset = H1CarrierIdDateLodoStrictTargetDataset(records, plan, normalizer, outer_date=date)
    _need(dataset.manifest() == terminal["target"]["strict_dataset"], "strict target drift")
    activity = {}
    for session, record in records.items():
        value = np.ascontiguousarray(np.stack([interpolate_trial_identity(record, item) for item in record.trial_values]), dtype=np.float32)
        _need(np.array_equal(value[:SUPPORT_TRIALS], dataset.support[session].identity), "support activity drift")
        activity[session] = value
    adapted = _DateLodoDatasetAdapter(dataset)
    output_trials = _output_trial_positions(adapted)
    new_arms = {arm: _score_arm(model=model, dataset=adapted, activity=activity, output_trials=output_trials, arm=arm, device=device) for arm in ARM_ORDER}
    target_sha = predecessor["CAUSAL_GROWING_CAP30"]["target_sha256"]
    _need(predecessor["FULL_SESSION_ORACLE"]["target_sha256"] == target_sha and all(row["target_sha256"] == target_sha for row in new_arms.values()), "same-target digest drift")
    state_after = state_hash(model.state_dict())
    _need(state_before == state_after, "model state changed")
    del model
    torch.cuda.empty_cache()
    return {
        "outer_date": date,
        "sessions": list(records),
        "predecessor": {name: dict(predecessor[name]) for name in ("CAUSAL_GROWING_CAP30", "FULL_SESSION_ORACLE")},
        "new_arms": new_arms,
        "target_sha256": target_sha,
        "model_state_before_sha256": state_before,
        "model_state_after_sha256": state_after,
        "model_state_immutable": True,
    }


def run(root: Path, *, device: str) -> dict[str, Any]:
    import torch

    root = Path(root).resolve()
    _bootstrap(root)
    _need(device.startswith("cuda") and torch.cuda.is_available(), "representative activity evaluation requires CUDA")
    predecessor = _load_predecessor(root)
    rows = [_evaluate_date(root, date, predecessor=predecessor[date], device=device) for date in DATE_ORDER]
    result = decision(rows)
    return {
        "schema": "h1_causal_representative_activity_v1",
        "status": "COMPLETE_FROZEN_WEIGHT_FIVE_DATE_CAUSAL_ACTIVITY_MEMORY",
        "device": device,
        "date_order": list(DATE_ORDER),
        "predecessor": {"relative": PREDECESSOR_RELATIVE, "sha256": PREDECESSOR_SHA256},
        "date_results": rows,
        "decision": result,
        "verdict": result["verdict"],
        "target_optimizer_backward_update": 0,
        "formal_selection_claim": False,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }


__all__ = ("run", "write_once")
