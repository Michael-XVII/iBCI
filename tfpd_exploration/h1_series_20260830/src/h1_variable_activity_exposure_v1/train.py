"""Physical source-only fine-tuning for H1 variable activity exposure."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable

import numpy as np

from .plan import (
    DATA_RELATIVE,
    EB_RECEIPT_RELATIVE,
    RAW_RECEIPT_RELATIVE,
    SEALED_CHECKPOINT_RELATIVE,
    SEALED_CHECKPOINT_SHA256,
    SEALED_CONFIG_RELATIVE,
    SEALED_CONFIG_SHA256,
    SOURCE_AUTHORITY_RECEIPT_SHA256,
    SOURCE_AUTHORITY_RELATIVE,
    TRAINING_PLAN,
    VariableActivityTrainingPlan,
)


class VariableActivityTrainingError(RuntimeError):
    pass


@dataclass(frozen=True)
class BatchDirective:
    epoch: int
    session: str
    row_indices: tuple[int, ...]
    activity_trials: int
    replay_m4: bool

    def __post_init__(self) -> None:
        if not (
            type(self.epoch) is int and self.epoch >= 0
            and isinstance(self.session, str) and self.session.startswith("ses-")
            and self.row_indices and len(self.row_indices) <= TRAINING_PLAN.batch_size
            and all(type(index) is int and index >= 0 for index in self.row_indices)
            and type(self.activity_trials) is int and self.activity_trials >= TRAINING_PLAN.support_trials
            and type(self.replay_m4) is bool
            and self.replay_m4 == (self.activity_trials == TRAINING_PLAN.support_trials)
        ):
            raise VariableActivityTrainingError("variable-activity batch directive drift")


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise VariableActivityTrainingError(message)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap(root: Path) -> None:
    project = root / "SPINT-main"
    _need(project.is_dir(), "SPINT-main root is absent")
    occupied = sys.modules.get("src")
    if occupied is not None:
        paths = tuple(str(item) for item in getattr(occupied, "__path__", ()))
        module_file = str(getattr(occupied, "__file__", ""))
        _need(str(project) in module_file or any(str(project) in item for item in paths),
              "top-level src namespace is occupied by a non-SPINT package")
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))


def build_epoch_directives(
    *,
    epoch: int,
    rows_by_session: dict[str, tuple[int, ...]],
    trial_counts: dict[str, int],
    plan: VariableActivityTrainingPlan = TRAINING_PLAN,
) -> tuple[BatchDirective, ...]:
    """Build one deterministic session-homogeneous 50/50 replay epoch."""

    _need(type(epoch) is int and 0 <= epoch < plan.epochs, "epoch outside frozen plan")
    directives: list[BatchDirective] = []
    for session in sorted(rows_by_session):
        rows = list(rows_by_session[session])
        count = int(trial_counts[session])
        _need(count >= plan.support_trials + 1 and rows, "source session lacks variable activity support")
        rng = random.Random(f"{plan.seed}|h1-var-activity|{epoch}|{session}")
        rng.shuffle(rows)
        chunks = [tuple(rows[offset : offset + plan.batch_size]) for offset in range(0, len(rows), plan.batch_size)]
        for local_index, chunk in enumerate(chunks):
            replay = local_index % plan.replay_period == 0
            if replay:
                cardinality = plan.support_trials
            else:
                token = hashlib.sha256(
                    f"{plan.seed}|cardinality|{epoch}|{session}|{local_index}".encode("utf-8")
                ).digest()
                cardinality = plan.support_trials + 1 + int.from_bytes(token[:8], "big") % (
                    count - plan.support_trials
                )
            directives.append(BatchDirective(epoch, session, chunk, cardinality, replay))
    random.Random(f"{plan.seed}|global-batch-order|{epoch}").shuffle(directives)
    _need(directives and any(row.replay_m4 for row in directives)
          and any(not row.replay_m4 for row in directives), "epoch lacks both replay and variable batches")
    return tuple(directives)


def _prepare_source(root: Path) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    _bootstrap(root)
    from src.data.h1_m4_eb_normalized_v2 import (
        H1M4EBNormalizedV2SourceDataset,
        fit_source_normalizer_from_cache,
    )
    from src.data.h1_m4_eb_pilot import (
        H1_M4_FOLD0_SOURCE,
        interpolate_trial_identity,
        load_immutable_source_authority,
        load_source_records,
    )

    records = load_source_records(root / DATA_RELATIVE)
    plan, _plan_manifest, cache, authority = load_immutable_source_authority(
        records,
        root / SOURCE_AUTHORITY_RELATIVE,
        SOURCE_AUTHORITY_RECEIPT_SHA256,
    )
    normalizer, _raw = fit_source_normalizer_from_cache(cache)
    dataset = H1M4EBNormalizedV2SourceDataset(records, cache, normalizer)
    rows_by_session: dict[str, list[int]] = {session: [] for session in H1_M4_FOLD0_SOURCE}
    for index, (session, _start) in enumerate(dataset.window_indices):
        rows_by_session[session].append(index)
    trial_identity = {
        session: np.ascontiguousarray(np.stack([
            interpolate_trial_identity(records[session], value)
            for value in records[session].trial_values
        ]), dtype=np.float32)
        for session in H1_M4_FOLD0_SOURCE
    }
    carriers = {
        session: np.ascontiguousarray(
            normalizer.normalize(cache.get(session, 0).carrier), dtype=np.float32,
        )
        for session in H1_M4_FOLD0_SOURCE
    }
    for session in H1_M4_FOLD0_SOURCE:
        _need(
            trial_identity[session].shape[1:] == (
                TRAINING_PLAN.max_trial_length,
                TRAINING_PLAN.units,
            )
            and carriers[session].shape == (TRAINING_PLAN.units, 4)
            and tuple(cache.get(session, 0).trial_values)
            == tuple(records[session].trial_values[: TRAINING_PLAN.support_trials]),
            "source first-four carrier/activity alignment drift",
        )
    source = {
        "records": records,
        "dataset": dataset,
        "rows_by_session": {key: tuple(value) for key, value in rows_by_session.items()},
        "trial_identity": trial_identity,
        "carriers": carriers,
        "trial_counts": {key: int(value.shape[0]) for key, value in trial_identity.items()},
    }
    evidence = {
        "source_authority": authority,
        "source_sessions": list(H1_M4_FOLD0_SOURCE),
        "source_window_indices_sha256": dataset.window_indices_sha256,
        "normalizer_sha256": normalizer.normalizer_sha256,
        "carrier_cache_sha256": cache.manifest["cache_sha256"],
        "target_sessions_opened": [],
        "formal_heldout_opened": False,
        "minival_opened": False,
        "evalai_opened": False,
    }
    return source, evidence, {"plan": plan, "normalizer": normalizer}


def _load_warm_start(root: Path, device: str) -> tuple[Any, str, dict[str, Any]]:
    import torch
    from scripts.h1_carrierid_evaluate import (
        _instantiate,
        _load_carrierid_checkpoint,
        _validate_carrierid_config,
    )
    from src.h1_m4_eb_normalized_v2_contract import state_hash

    checkpoint = root / SEALED_CHECKPOINT_RELATIVE
    config_path = root / SEALED_CONFIG_RELATIVE
    _need(
        checkpoint.is_file() and not checkpoint.is_symlink()
        and config_path.is_file() and not config_path.is_symlink()
        and _sha_file(checkpoint) == SEALED_CHECKPOINT_SHA256
        and _sha_file(config_path) == SEALED_CONFIG_SHA256,
        "sealed H-C warm-start binding drift",
    )
    config = _validate_carrierid_config(config_path, "full")
    payload, metadata = _load_carrierid_checkpoint(checkpoint, config_path, "full")
    model = _instantiate(config, payload, torch.device(device))
    _need(bool(model.hparams.decode_last_timestep_only)
          and bool(model.hparams.predict_scaled_behavior)
          and float(model.hparams.behavior_scaling_factor) == TRAINING_PLAN.behavior_scale,
          "sealed H-C output/loss contract drift")
    return model, state_hash(model.state_dict()), dict(metadata)


def run_training(
    root: Path,
    *,
    device: str,
    max_steps: int | None = None,
    checkpoint_output: Path | None = None,
) -> dict[str, Any]:
    import torch

    root = Path(root).resolve()
    _bootstrap(root)
    from src.h1_m4_eb_normalized_v2_contract import array_sha256, state_hash

    _need(device.startswith("cuda") and torch.cuda.is_available(), "H1 variable activity requires CUDA")
    _need(max_steps is None or type(max_steps) is int and max_steps > 0, "invalid smoke step bound")
    torch.manual_seed(TRAINING_PLAN.seed)
    torch.cuda.manual_seed_all(TRAINING_PLAN.seed)
    source, source_evidence, _bindings = _prepare_source(root)
    model, state_before, warm_metadata = _load_warm_start(root, device)
    _need(
        warm_metadata.get("normalizer_sha256") == source_evidence["normalizer_sha256"]
        and warm_metadata.get("source_cache_sha256") == source_evidence["carrier_cache_sha256"],
        "warm-start checkpoint is not bound to the prepared source carrier authority",
    )
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=TRAINING_PLAN.learning_rate, weight_decay=TRAINING_PLAN.weight_decay,
    )
    identity_gpu = {
        session: torch.as_tensor(value, dtype=torch.float32, device=device)
        for session, value in source["trial_identity"].items()
    }
    carrier_gpu = {
        session: torch.as_tensor(value, dtype=torch.float32, device=device)
        for session, value in source["carriers"].items()
    }
    dataset = source["dataset"]
    losses: list[float] = []
    epoch_evidence: list[dict[str, Any]] = []
    cardinality_counts: dict[int, int] = {}
    steps = 0
    nonfinite = False
    gradient_nonzero_steps = 0
    started = time.monotonic()
    for epoch in range(TRAINING_PLAN.epochs):
        directives = build_epoch_directives(
            epoch=epoch,
            rows_by_session=source["rows_by_session"],
            trial_counts=source["trial_counts"],
        )
        epoch_losses: list[float] = []
        for directive in directives:
            if max_steps is not None and steps >= max_steps:
                break
            session = directive.session
            starts = [int(dataset.window_indices[index][1]) for index in directive.row_indices]
            neural = np.ascontiguousarray(np.stack([
                dataset.neural_data[session][start : start + TRAINING_PLAN.window_size]
                for start in starts
            ]), dtype=np.float32)
            target = np.ascontiguousarray(np.stack([
                dataset.covariate_data[session][start + TRAINING_PLAN.window_size - 1]
                for start in starts
            ]), dtype=np.float32)
            batch = len(starts)
            identity = identity_gpu[session][: directive.activity_trials].unsqueeze(0).expand(
                batch, -1, -1, -1,
            )
            carrier = carrier_gpu[session].unsqueeze(0).expand(batch, -1, -1)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                torch.as_tensor(neural, dtype=torch.float32, device=device),
                calib_trialized_neural_features=identity,
                carrier=carrier,
            )
            prediction = output[:, -1, :] / TRAINING_PLAN.behavior_scale
            loss = torch.nn.functional.mse_loss(
                prediction,
                torch.as_tensor(target, dtype=torch.float32, device=device),
            )
            if not torch.isfinite(loss):
                nonfinite = True
                raise VariableActivityTrainingError("nonfinite H1 variable-activity loss")
            loss.backward()
            finite_gradients = all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
            nonzero_gradient = any(
                parameter.grad is not None and bool(torch.count_nonzero(parameter.grad).item())
                for parameter in model.parameters()
            )
            _need(finite_gradients and nonzero_gradient, "H1 variable-activity gradient gate failed")
            gradient_nonzero_steps += 1
            optimizer.step()
            value = float(loss.detach().cpu())
            losses.append(value)
            epoch_losses.append(value)
            cardinality_counts[directive.activity_trials] = cardinality_counts.get(directive.activity_trials, 0) + 1
            steps += 1
        if epoch_losses:
            epoch_evidence.append({
                "epoch": epoch,
                "steps": len(epoch_losses),
                "mean_loss": float(np.mean(epoch_losses, dtype=np.float64)),
                "first_loss": epoch_losses[0],
                "last_loss": epoch_losses[-1],
            })
        if max_steps is not None and steps >= max_steps:
            break
    _need(steps > 0 and gradient_nonzero_steps == steps and 4 in cardinality_counts
          and any(key > 4 for key in cardinality_counts), "training did not exercise both cardinality regimes")
    state_after = state_hash(model.state_dict())
    checkpoint_evidence = None
    if checkpoint_output is not None:
        output = Path(checkpoint_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        _need(not output.exists(), "variable-activity checkpoint output already exists")
        payload = {
            "schema": "h1_variable_activity_exposure_checkpoint_v1",
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "warm_start_state_sha256": state_before,
            "trained_state_sha256": state_after,
            "steps": steps,
            "epochs_completed": len(epoch_evidence),
            "training_plan": TRAINING_PLAN.__dict__,
            "source_evidence": source_evidence,
        }
        torch.save(payload, output)
        output.chmod(0o444)
        checkpoint_evidence = {
            "path": str(output),
            "sha256": _sha_file(output),
            "state_sha256": state_after,
            "mode": "0444",
        }
    return {
        "schema": "h1_variable_activity_exposure_training_v1",
        "status": "SMOKE_COMPLETE" if max_steps is not None else "FULL_TRAINING_COMPLETE",
        "device": device,
        "training_plan": TRAINING_PLAN.__dict__,
        "warm_start": {
            "checkpoint_relative": SEALED_CHECKPOINT_RELATIVE,
            "checkpoint_sha256": SEALED_CHECKPOINT_SHA256,
            "state_sha256": state_before,
            "normalizer_sha256": warm_metadata["normalizer_sha256"],
            "source_cache_sha256": warm_metadata["source_cache_sha256"],
        },
        "source": source_evidence,
        "steps": steps,
        "gradient_nonzero_steps": gradient_nonzero_steps,
        "nonfinite": nonfinite,
        "cardinality_counts": {str(key): value for key, value in sorted(cardinality_counts.items())},
        "carrier_sha256_by_session": {
            session: array_sha256(value) for session, value in source["carriers"].items()
        },
        "epoch_evidence": epoch_evidence,
        "mean_loss": float(np.mean(losses, dtype=np.float64)),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "state_after_sha256": state_after,
        "checkpoint": checkpoint_evidence,
        "target_optimizer_backward_update": 0,
        "elapsed_seconds": time.monotonic() - started,
        "cuda_initialized": bool(torch.cuda.is_initialized()),
    }


def write_json_once(path: Path, payload: dict[str, Any]) -> tuple[Path, str]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    with path.open("xb") as handle:
        handle.write(body)
    sidecar = path.with_name(path.name + ".sha256")
    with sidecar.open("x", encoding="utf-8") as handle:
        handle.write(f"{digest}  {path.name}\n")
    path.chmod(0o444)
    sidecar.chmod(0o444)
    return path, digest


__all__ = (
    "BatchDirective",
    "VariableActivityTrainingError",
    "build_epoch_directives",
    "run_training",
    "write_json_once",
)
