"""Physical five-date frozen-weight H1 activity-headroom evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from m1_h1_activity_headroom_v1.core import ARM_ORDER, ActivityHeadroomError, array_digest
from m1_h1_activity_headroom_v1.h1 import (
    SUPPORT_TRIALS,
    _evaluate_arm,
    _evaluate_direct_static,
    _output_trial_positions,
)
from m1_h1_activity_headroom_v1.m1 import write_once

from .plan import AUTHORITIES, DATE_ORDER, DateAuthority, breadth_decision


DATA_RELATIVE = "SPINT-main/data/000954"
CHECKPOINT_METADATA_SCHEMA = "h1_carrierid_date_lodo_phase2_terminal_checkpoint_v1"
TERMINAL_SCHEMA = "h1_carrierid_date_lodo_phase2_terminal_evaluation_v1"


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


def _bootstrap(root: Path) -> None:
    project = root / "SPINT-main"
    _need(project.is_dir(), "H1 project root is absent")
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))


class _DateLodoDatasetAdapter:
    """Expose the date-LODO carrier under the frozen fold0 evaluator name."""

    def __init__(self, dataset: Any) -> None:
        self._dataset = dataset
        self.records = dataset.records
        self.window_indices = dataset.window_indices
        self.support = {
            session: SimpleNamespace(carriers={"full": item.normalized_carrier})
            for session, item in dataset.support.items()
        }

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> Any:
        return self._dataset[index]


def _load_bound_authority(root: Path, authority: DateAuthority) -> tuple[dict[str, Any], Path, Path, Path]:
    terminal_path = root / authority.terminal_relative
    checkpoint_path = root / authority.checkpoint_relative
    config_path = root / authority.config_relative
    source_manifest_path = root / authority.source_manifest_relative
    for label, path, expected in (
        ("terminal", terminal_path, authority.terminal_sha256),
        ("checkpoint", checkpoint_path, authority.checkpoint_sha256),
        ("config", config_path, authority.config_sha256),
        ("source manifest", source_manifest_path, authority.source_manifest_sha256),
    ):
        _need(_immutable_file(path), f"{authority.date} {label} is absent, mutable, or symlinked")
        _need(_sha_file(path) == expected, f"{authority.date} {label} SHA drift")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    _need(
        isinstance(terminal, dict)
        and terminal.get("schema") == TERMINAL_SCHEMA
        and terminal.get("status") == f"PASS_H1_CARRIERID_DATE_LODO_PHASE2_{authority.date}_HS_HC_EVALUATED"
        and terminal.get("outer_date") == authority.date,
        f"{authority.date} terminal schema/status/date drift",
    )
    checkpoint_row = terminal.get("checkpoints", {}).get("H-C")
    metadata = checkpoint_row.get("metadata") if isinstance(checkpoint_row, Mapping) else None
    _need(
        isinstance(checkpoint_row, Mapping)
        and checkpoint_row.get("sha256") == authority.checkpoint_sha256
        and checkpoint_row.get("config_sha256") == authority.config_sha256
        and isinstance(metadata, Mapping)
        and metadata.get("schema") == CHECKPOINT_METADATA_SCHEMA
        and metadata.get("arm") == "H-C"
        and metadata.get("outer_date") == authority.date
        and metadata.get("checkpoint_epoch_zero_based") == 49
        and metadata.get("epochs_completed") == 50
        and metadata.get("fresh_seed") == 42
        and metadata.get("checkpoint_warm_start") is False
        and metadata.get("target_optimizer_steps") == 0
        and metadata.get("target_backward_steps") == 0
        and metadata.get("phase1_source_manifest_sha256") == authority.source_manifest_sha256,
        f"{authority.date} terminal H-C checkpoint provenance drift",
    )
    target = terminal.get("target")
    strict = target.get("strict_dataset") if isinstance(target, Mapping) else None
    _need(
        isinstance(strict, Mapping)
        and strict.get("outer_date") == authority.date
        and strict.get("window_indices_sha256") == authority.query_window_indices_sha256,
        f"{authority.date} strict target authority drift",
    )
    metrics = terminal.get("metrics", {}).get("h_c")
    _need(
        isinstance(metrics, Mapping)
        and metrics.get("query_window_indices_sha256") == authority.query_window_indices_sha256
        and abs(float(metrics.get("pooled_r2")) - authority.accepted_static_pooled_r2) <= 1.0e-15,
        f"{authority.date} accepted H-C score drift",
    )
    return terminal, checkpoint_path, config_path, source_manifest_path


def _load_model(checkpoint_path: Path, config_path: Path, terminal: Mapping[str, Any], *, device: str) -> tuple[Any, str]:
    import hydra
    from omegaconf import OmegaConf
    import torch
    from src.h1_m4_cce_contract import state_hash

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _need(isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping), "H1 checkpoint is malformed")
    expected_metadata = terminal["checkpoints"]["H-C"]["metadata"]
    _need(payload.get("h1_carrierid_date_lodo_phase2") == expected_metadata, "checkpoint embedded provenance drift")
    config = OmegaConf.load(config_path)
    model = hydra.utils.instantiate(config.model)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(torch.device(device)); model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _need(
        model.arm == "H-C"
        and bool(model.hparams.decode_last_timestep_only)
        and bool(model.hparams.predict_scaled_behavior)
        and float(model.hparams.behavior_scaling_factor) == 20.0,
        "H1 model arm/output/scaling contract drift",
    )
    return model, state_hash(model.state_dict())


def _evaluate_date(root: Path, authority: DateAuthority, *, device: str) -> dict[str, Any]:
    import torch
    from src.data.h1_carrierid_date_lodo_target import (
        H1CarrierIdDateLodoStrictTargetDataset,
        load_outer_date_target_records,
        load_target_dependencies,
    )
    from src.data.h1_m4_eb_pilot import interpolate_trial_identity
    from src.h1_m4_cce_contract import state_hash

    terminal, checkpoint_path, config_path, source_manifest_path = _load_bound_authority(root, authority)
    plan, normalizer, source_manifest = load_target_dependencies(source_manifest_path, outer_date=authority.date)
    _need(source_manifest.get("outer_date") == authority.date, "source manifest outer-date drift")
    model, state_before = _load_model(checkpoint_path, config_path, terminal, device=device)
    records = load_outer_date_target_records(root / DATA_RELATIVE, outer_date=authority.date)
    dataset = H1CarrierIdDateLodoStrictTargetDataset(records, plan, normalizer, outer_date=authority.date)
    _need(dataset.manifest() == terminal["target"]["strict_dataset"], "strict target dataset differs from terminal receipt")

    trial_activity_by_session: dict[str, np.ndarray] = {}
    trial_counts: dict[str, int] = {}
    for session, record in records.items():
        activity = np.ascontiguousarray(
            np.stack([interpolate_trial_identity(record, value) for value in record.trial_values]),
            dtype=np.float32,
        )
        _need(np.array_equal(activity[:SUPPORT_TRIALS], dataset.support[session].identity), "support identity reconstruction drift")
        trial_activity_by_session[session] = activity
        trial_counts[session] = int(activity.shape[0])

    evaluator_dataset = _DateLodoDatasetAdapter(dataset)
    output_trials = _output_trial_positions(evaluator_dataset)
    cached_results = [
        _evaluate_arm(
            model=model,
            dataset=evaluator_dataset,
            trial_activity_by_session=trial_activity_by_session,
            output_trials=output_trials,
            arm=arm,
            device=device,
        )
        for arm in ARM_ORDER
    ]
    direct = _evaluate_direct_static(model, evaluator_dataset, device=device)
    _need(abs(float(direct["pooled_r2"]) - authority.accepted_static_pooled_r2) <= 2.0e-7, "static replay score mismatch")
    cached = cached_results[0]
    max_abs = float(np.max(np.abs(direct["_prediction"].astype(np.float64) - cached["_prediction"].astype(np.float64))))
    r2_abs = abs(float(direct["pooled_r2"]) - float(cached["pooled_r2"]))
    _need(max_abs <= 2.0e-6 and r2_abs <= 2.0e-7, "cached identity parity exceeds tolerance")
    direct["cached_identity_parity"] = {
        "max_abs_prediction": max_abs,
        "abs_r2": r2_abs,
        "tolerance_prediction": 2.0e-6,
        "tolerance_r2": 2.0e-7,
        "pass": True,
    }
    results = [direct, *cached_results[1:]]
    direct_target_digest = array_digest(direct["_target"])
    for row in results:
        _need(row["target_sha256"] == direct_target_digest, "four-arm target digest drift")
        row.pop("_prediction", None)
        row.pop("_target", None)
    state_after = state_hash(model.state_dict())
    _need(state_before == state_after, "H1 model state changed during target evaluation")
    result = {
        "outer_date": authority.date,
        "authority": {
            "terminal_relative": authority.terminal_relative,
            "terminal_sha256": authority.terminal_sha256,
            "checkpoint_relative": authority.checkpoint_relative,
            "checkpoint_sha256": authority.checkpoint_sha256,
            "config_relative": authority.config_relative,
            "config_sha256": authority.config_sha256,
            "source_manifest_relative": authority.source_manifest_relative,
            "source_manifest_sha256": authority.source_manifest_sha256,
            "query_window_indices_sha256": authority.query_window_indices_sha256,
            "accepted_static_pooled_r2": authority.accepted_static_pooled_r2,
        },
        "sessions": list(records),
        "trial_counts": trial_counts,
        "results": results,
        "model_state_before_sha256": state_before,
        "model_state_after_sha256": state_after,
        "model_state_immutable": True,
    }
    del model
    torch.cuda.empty_cache()
    return result


def run(root: Path, *, device: str) -> dict[str, Any]:
    import torch

    root = Path(root).resolve()
    _bootstrap(root)
    _need(device.startswith("cuda") and torch.cuda.is_available(), "five-date H1 experiment requires CUDA")
    date_rows = [_evaluate_date(root, AUTHORITIES[date], device=device) for date in DATE_ORDER]
    decision = breadth_decision(date_rows)
    return {
        "schema": "h1_date_lodo_activity_headroom_breadth_v1",
        "status": "COMPLETE_FROZEN_WEIGHT_FIVE_DATE_ACTIVITY_HEADROOM",
        "surface": "five_confirmatory_dates_strict_post_m4_support",
        "device": device,
        "date_order": list(DATE_ORDER),
        "date_results": date_rows,
        "decision": decision,
        "verdict": decision["verdict"],
        "target_optimizer_backward_update": 0,
        "formal_heldout_opened": False,
        "minival_opened": False,
        "evalai_opened": False,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }


__all__ = ("run", "write_once")
