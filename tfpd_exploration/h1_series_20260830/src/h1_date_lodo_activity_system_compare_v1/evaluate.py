"""Physical H-S evaluation paired to the immutable five-date H-C result."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import time
from typing import Any, Mapping

import numpy as np

from h1_date_lodo_activity_headroom_v1.evaluate import (
    DATA_RELATIVE,
    _DateLodoDatasetAdapter,
    _bootstrap,
    _load_bound_authority,
)
from h1_date_lodo_activity_headroom_v1.plan import AUTHORITIES as HC_AUTHORITIES
from m1_h1_activity_headroom_v1.core import (
    ActivityArm,
    ActivityHeadroomError,
    array_digest,
    forward_with_cached_identity,
    identity_from_raw_trials,
    selection_for_output_trial,
    variance_weighted_r2,
)
from m1_h1_activity_headroom_v1.h1 import (
    BATCH_SIZE,
    GROWING_CAP,
    OUTPUTS,
    SUPPORT_TRIALS,
    WINDOW,
    _evaluate_direct_static,
    _output_trial_positions,
)
from m1_h1_activity_headroom_v1.m1 import write_once

from .plan import (
    DATE_ORDER,
    HC_PREDECESSOR_RELATIVE,
    HC_PREDECESSOR_SHA256,
    HS_AUTHORITIES,
    HsAuthority,
    comparison_decision,
)


CHECKPOINT_METADATA_SCHEMA = "h1_carrierid_date_lodo_phase2_terminal_checkpoint_v1"


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ActivityHeadroomError(message)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _immutable_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o444


def _load_hc_predecessor(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    path = root / HC_PREDECESSOR_RELATIVE
    _need(_immutable_file(path) and _sha_file(path) == HC_PREDECESSOR_SHA256, "H-C predecessor result binding drift")
    sidecar = path.with_name(path.name + ".sha256")
    _need(_immutable_file(sidecar) and sidecar.read_text(encoding="ascii") == f"{HC_PREDECESSOR_SHA256}  {path.name}\n", "H-C predecessor sidecar drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _need(payload.get("schema") == "h1_date_lodo_activity_headroom_breadth_v1" and payload.get("date_order") == list(DATE_ORDER), "H-C predecessor schema/order drift")
    by_date: dict[str, dict[str, dict[str, Any]]] = {}
    for date, row in zip(DATE_ORDER, payload.get("date_results", []), strict=True):
        _need(row.get("outer_date") == date, "H-C predecessor date drift")
        authority = row.get("authority")
        _need(isinstance(authority, Mapping) and authority.get("terminal_sha256") == HC_AUTHORITIES[date].terminal_sha256 and authority.get("query_window_indices_sha256") == HC_AUTHORITIES[date].query_window_indices_sha256, "H-C predecessor authority drift")
        arms = {item.get("arm"): item for item in row.get("results", []) if isinstance(item, Mapping)}
        _need("STATIC_SUPPORT" in arms and "CAUSAL_GROWING_CAP30" in arms, "H-C predecessor comparison arms absent")
        by_date[date] = {name: dict(arms[name]) for name in ("STATIC_SUPPORT", "CAUSAL_GROWING_CAP30")}
    return payload, by_date


def _load_hs_model(checkpoint_path: Path, config_path: Path, terminal: Mapping[str, Any], *, device: str) -> tuple[Any, str]:
    import hydra
    from omegaconf import OmegaConf
    import torch
    from src.h1_m4_cce_contract import state_hash

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _need(isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping), "H-S checkpoint is malformed")
    expected_metadata = terminal["checkpoints"]["H-S"]["metadata"]
    _need(payload.get("h1_carrierid_date_lodo_phase2") == expected_metadata, "H-S checkpoint embedded provenance drift")
    config = OmegaConf.load(config_path)
    model = hydra.utils.instantiate(config.model)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(torch.device(device)); model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _need(model.arm == "H-S" and bool(model.hparams.decode_last_timestep_only) and bool(model.hparams.predict_scaled_behavior) and float(model.hparams.behavior_scaling_factor) == 20.0, "H-S model arm/output/scaling drift")
    return model, state_hash(model.state_dict())


def _validate_hs_authority(root: Path, terminal: Mapping[str, Any], authority: HsAuthority) -> tuple[Path, Path]:
    checkpoint = root / authority.checkpoint_relative
    config = root / authority.config_relative
    for label, path, expected in (("checkpoint", checkpoint, authority.checkpoint_sha256), ("config", config, authority.config_sha256)):
        _need(_immutable_file(path) and _sha_file(path) == expected, f"{authority.date} H-S {label} binding drift")
    row = terminal.get("checkpoints", {}).get("H-S")
    metadata = row.get("metadata") if isinstance(row, Mapping) else None
    _need(isinstance(row, Mapping) and row.get("sha256") == authority.checkpoint_sha256 and row.get("config_sha256") == authority.config_sha256 and isinstance(metadata, Mapping), f"{authority.date} H-S terminal checkpoint drift")
    _need(metadata.get("schema") == CHECKPOINT_METADATA_SCHEMA and metadata.get("arm") == "H-S" and metadata.get("outer_date") == authority.date and metadata.get("checkpoint_epoch_zero_based") == 49 and metadata.get("epochs_completed") == 50 and metadata.get("fresh_seed") == 42 and metadata.get("checkpoint_warm_start") is False and metadata.get("target_optimizer_steps") == 0 and metadata.get("target_backward_steps") == 0, f"{authority.date} H-S provenance drift")
    metrics = terminal.get("metrics", {}).get("h_s")
    _need(isinstance(metrics, Mapping) and metrics.get("query_window_indices_sha256") == HC_AUTHORITIES[authority.date].query_window_indices_sha256 and abs(float(metrics.get("pooled_r2")) - authority.accepted_static_pooled_r2) <= 1e-15, f"{authority.date} H-S accepted metric drift")
    return checkpoint, config


def _evaluate_hs_cached(*, model: Any, dataset: Any, trial_activity_by_session: dict[str, np.ndarray], output_trials: tuple[int, ...], arm: ActivityArm, device: str) -> dict[str, Any]:
    import torch

    _need(arm in {ActivityArm.STATIC_SUPPORT, ActivityArm.CAUSAL_GROWING_CAP30}, "H-S comparison arm drift")
    predictions = np.empty((len(dataset), OUTPUTS), dtype=np.float32)
    targets = np.empty((len(dataset), OUTPUTS), dtype=np.float32)
    session_names = tuple(session for session, _ in dataset.window_indices)
    selections = tuple(selection_for_output_trial(arm, output_trial_index=output_trials[index], total_trials=len(dataset.records[session_names[index]].trial_values), support_trials=SUPPORT_TRIALS, growing_cap=GROWING_CAP) for index in range(len(dataset)))
    groups: dict[tuple[str, tuple[int, ...]], list[int]] = {}
    for row, key in enumerate((session_names[index], selections[index]) for index in range(len(dataset))):
        groups.setdefault(key, []).append(row)
    forwards = 0
    started = time.monotonic()
    with torch.no_grad():
        for (session, selection), rows in groups.items():
            identity = identity_from_raw_trials(model.net, trial_activity_by_session[session], selection, family="m1", device=device)
            for offset in range(0, len(rows), BATCH_SIZE):
                batch_rows = rows[offset:offset + BATCH_SIZE]
                xs, ys = [], []
                for row in batch_rows:
                    current_session, start = dataset.window_indices[row]
                    _need(current_session == session, "H-S grouped session drift")
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
        "arm": arm.value,
        "system": "H-S",
        "causal": True,
        "label_free": True,
        "carrier_used": False,
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
        "_prediction": predictions,
        "_target": targets,
    }


def _evaluate_hs_date(root: Path, date: str, *, device: str, hc_rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    import torch
    from src.data.h1_carrierid_date_lodo_target import H1CarrierIdDateLodoStrictTargetDataset, load_outer_date_target_records, load_target_dependencies
    from src.data.h1_m4_eb_pilot import interpolate_trial_identity
    from src.h1_m4_cce_contract import state_hash

    hc_authority = HC_AUTHORITIES[date]
    terminal, _hc_checkpoint, _hc_config, source_manifest_path = _load_bound_authority(root, hc_authority)
    checkpoint, config = _validate_hs_authority(root, terminal, HS_AUTHORITIES[date])
    plan, normalizer, source_manifest = load_target_dependencies(source_manifest_path, outer_date=date)
    _need(source_manifest.get("outer_date") == date, "H-S source manifest date drift")
    model, state_before = _load_hs_model(checkpoint, config, terminal, device=device)
    records = load_outer_date_target_records(root / DATA_RELATIVE, outer_date=date)
    dataset = H1CarrierIdDateLodoStrictTargetDataset(records, plan, normalizer, outer_date=date)
    _need(dataset.manifest() == terminal["target"]["strict_dataset"], "H-S strict dataset differs from terminal")
    trial_activity: dict[str, np.ndarray] = {}
    for session, record in records.items():
        value = np.ascontiguousarray(np.stack([interpolate_trial_identity(record, item) for item in record.trial_values]), dtype=np.float32)
        _need(np.array_equal(value[:SUPPORT_TRIALS], dataset.support[session].identity), "H-S support identity reconstruction drift")
        trial_activity[session] = value
    adapted = _DateLodoDatasetAdapter(dataset)
    output_trials = _output_trial_positions(adapted)
    cached_static = _evaluate_hs_cached(model=model, dataset=adapted, trial_activity_by_session=trial_activity, output_trials=output_trials, arm=ActivityArm.STATIC_SUPPORT, device=device)
    growing = _evaluate_hs_cached(model=model, dataset=adapted, trial_activity_by_session=trial_activity, output_trials=output_trials, arm=ActivityArm.CAUSAL_GROWING_CAP30, device=device)
    direct = _evaluate_direct_static(model, adapted, device=device)
    _need(abs(float(direct["pooled_r2"]) - HS_AUTHORITIES[date].accepted_static_pooled_r2) <= 2e-7, "H-S static replay mismatch")
    max_abs = float(np.max(np.abs(direct["_prediction"].astype(np.float64) - cached_static["_prediction"].astype(np.float64))))
    r2_abs = abs(float(direct["pooled_r2"]) - float(cached_static["pooled_r2"]))
    _need(max_abs <= 2e-6 and r2_abs <= 2e-7, "H-S cached identity parity exceeds tolerance")
    direct.update({"system": "H-S", "carrier_used": False, "cached_identity_parity": {"max_abs_prediction": max_abs, "abs_r2": r2_abs, "tolerance_prediction": 2e-6, "tolerance_r2": 2e-7, "pass": True}})
    _need(direct["target_sha256"] == growing["target_sha256"] == hc_rows["STATIC_SUPPORT"]["target_sha256"] == hc_rows["CAUSAL_GROWING_CAP30"]["target_sha256"], "2x2 target digest mismatch")
    for row in (direct, growing):
        row.pop("_prediction", None); row.pop("_target", None)
    state_after = state_hash(model.state_dict())
    _need(state_before == state_after, "H-S model state changed during evaluation")
    del model
    torch.cuda.empty_cache()
    return {
        "outer_date": date,
        "sessions": list(records),
        "target_sha256": direct["target_sha256"],
        "systems": {
            "H-S": {"STATIC_SUPPORT": direct, "CAUSAL_GROWING_CAP30": growing},
            "H-C": {name: dict(hc_rows[name]) for name in ("STATIC_SUPPORT", "CAUSAL_GROWING_CAP30")},
        },
        "hs_authority": {
            "checkpoint_relative": HS_AUTHORITIES[date].checkpoint_relative,
            "checkpoint_sha256": HS_AUTHORITIES[date].checkpoint_sha256,
            "config_relative": HS_AUTHORITIES[date].config_relative,
            "config_sha256": HS_AUTHORITIES[date].config_sha256,
            "terminal_relative": hc_authority.terminal_relative,
            "terminal_sha256": hc_authority.terminal_sha256,
            "source_manifest_sha256": hc_authority.source_manifest_sha256,
            "query_window_indices_sha256": hc_authority.query_window_indices_sha256,
        },
        "model_state_before_sha256": state_before,
        "model_state_after_sha256": state_after,
        "model_state_immutable": True,
    }


def run(root: Path, *, device: str) -> dict[str, Any]:
    import torch

    root = Path(root).resolve()
    _bootstrap(root)
    _need(device.startswith("cuda") and torch.cuda.is_available(), "H1 2x2 comparison requires CUDA")
    _hc_payload, hc_by_date = _load_hc_predecessor(root)
    rows = [_evaluate_hs_date(root, date, device=device, hc_rows=hc_by_date[date]) for date in DATE_ORDER]
    decision = comparison_decision(rows)
    return {
        "schema": "h1_date_lodo_activity_system_compare_v1",
        "status": "COMPLETE_FROZEN_WEIGHT_FIVE_DATE_HS_HC_ACTIVITY_COMPARISON",
        "surface": "five_confirmatory_dates_strict_post_m4_support",
        "device": device,
        "date_order": list(DATE_ORDER),
        "hc_predecessor": {"relative": HC_PREDECESSOR_RELATIVE, "sha256": HC_PREDECESSOR_SHA256},
        "date_results": rows,
        "decision": decision,
        "verdict": decision["verdict"],
        "target_optimizer_backward_update": 0,
        "formal_selection_claim": False,
        "formal_heldout_opened": False,
        "minival_opened": False,
        "evalai_opened": False,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }


__all__ = ("run", "write_once")
