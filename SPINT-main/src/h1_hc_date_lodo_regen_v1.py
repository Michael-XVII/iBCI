"""Source-only H1 five-date H-C regeneration successor.

This module is additive.  It does not accept the missing historical raw/EB
receipts as authority and it never opens an outer-date recording while
preparing or training a source cell.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import stat
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np

from src.data.h1_carrierid_date_lodo_source import (
    load_source_records_with_target_filename_index,
)
from src.data.h1_m4_cce_date_lodo import source_sessions_for_date
from src.data.h1_m4_eb_pilot import (
    EXPECTED_NEURONS,
    H1PilotRecord,
    array_sha256,
    carrier_sha256,
    fit_frozen_carrier,
    interpolate_trial_identity,
    session_date,
)
from src.h1_m4_cce_contract import (
    CONFIRMATORY_DATES,
    FIXED_EPOCHS,
    FIXED_SEED,
    NORMALIZER_FLOOR,
    NORMALIZER_FORMULA,
    SUPPORT_TRIALS,
    WINDOW_SIZE,
    canonical_sha256,
    sha256_file,
    state_hash,
)


SCHEMA = "h1_hc_date_lodo_regen_v1"
STATUS_SOURCE = "PASS_H1_HC_DATE_LODO_REGEN_V1_SOURCE_AUTHORITY"
STATUS_SMOKE = "PASS_H1_HC_DATE_LODO_REGEN_V1_GPU_SMOKE"
STATUS_CELL = "PASS_H1_HC_DATE_LODO_REGEN_V1_SOURCE_CELL"
STATUS_TERMINAL = "COMPLETE_H1_HC_DATE_LODO_REGEN_V1_SOURCE_ONLY_AUTHORITY"
Q_GRID = (4, 8, 12, 16)
LAMBDA_GRID = (1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0)
BATCH_SIZE = 32
MODEL_PARAMETERS = 10_947_836
CHECKPOINT_SCHEMA = "h1_hc_date_lodo_regen_v1_checkpoint"


class RegenError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RegenError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _encoded_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _publish_bytes(path: Path, payload: bytes, *, sidecar: bool = True) -> str:
    """Publish once using a hard link, chmod 0444, and optional SHA sidecar."""
    output = path.resolve()
    if output.exists() or output.is_symlink() or os.path.lexists(str(output)):
        raise FileExistsError(f"refusing existing artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    _need(stat.S_IMODE(output.stat().st_mode) == 0o444, f"artifact is not 0444: {output}")
    digest = hashlib.sha256(payload).hexdigest()
    if sidecar:
        _publish_bytes(output.with_name(output.name + ".sha256"), f"{digest}  {output.name}\n".encode("ascii"), sidecar=False)
    return digest


def publish_json(path: Path, value: Mapping[str, Any]) -> str:
    return _publish_bytes(path, _encoded_json(value))


def publish_text(path: Path, value: str) -> str:
    return _publish_bytes(path, value.encode("utf-8"))


def publish_npz(path: Path, **arrays: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.savez(buffer, **{key: np.asarray(value) for key, value in arrays.items()})
    return _publish_bytes(path, buffer.getvalue())


def publish_npy(path: Path, value: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(value), allow_pickle=False)
    return _publish_bytes(path, buffer.getvalue())


def verify_sidecar(path: Path) -> str:
    _need(path.is_file() and not path.is_symlink(), f"missing artifact: {path}")
    _need(stat.S_IMODE(path.stat().st_mode) == 0o444, f"artifact is not 0444: {path}")
    digest = sha256_file(path)
    sidecar = path.with_name(path.name + ".sha256")
    _need(sidecar.is_file() and not sidecar.is_symlink(), f"missing sidecar: {sidecar}")
    _need(stat.S_IMODE(sidecar.stat().st_mode) == 0o444, f"sidecar is not 0444: {sidecar}")
    _need(sidecar.read_text(encoding="ascii") == f"{digest}  {path.name}\n", f"sidecar mismatch: {sidecar}")
    return digest


def variance_weighted_r2(truth: np.ndarray, estimate: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    _need(truth.shape == estimate.shape and truth.ndim == 2 and truth.shape[0] > 1, "R2 arrays are malformed")
    residual = float(np.square(truth - estimate, dtype=np.float64).sum(dtype=np.float64))
    centered = truth - truth.mean(axis=0, keepdims=True, dtype=np.float64)
    total = float(np.square(centered, dtype=np.float64).sum(dtype=np.float64))
    _need(math.isfinite(residual) and math.isfinite(total) and total > 0.0, "variance-weighted R2 is undefined")
    return float(1.0 - residual / total)


def select_candidate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = {(q, lam) for q in Q_GRID for lam in LAMBDA_GRID}
    observed = {(int(row["q"]), float(row["lambda"])) for row in rows}
    _need(observed == expected and len(rows) == len(expected), "candidate grid is incomplete or duplicated")
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["equal_date_mean_r2"]),
            -float(row["worst_date_r2"]),
            int(row["q"]),
            -float(row["lambda"]),
        ),
    )
    return dict(ranked[0])


def _support_rates(record: H1PilotRecord) -> np.ndarray:
    trials = [record.blocks_for(value) for value in record.trial_values[:SUPPORT_TRIALS]]
    _need(len(trials) == SUPPORT_TRIALS and all(item.rates.shape[0] >= 2 for item in trials), f"{record.session_name}: illegal M4 support")
    return np.concatenate([item.rates for item in trials], axis=0).astype(np.float64)


def _fit_basis(records: Mapping[str, H1PilotRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rates = np.concatenate([_support_rates(record) for record in records.values()], axis=0)
    mean = rates.mean(axis=0, dtype=np.float64)
    scale = np.maximum(rates.std(axis=0, dtype=np.float64), 1.0e-6)
    _, _, right = np.linalg.svd((rates - mean[None, :]) / scale[None, :], full_matrices=False)
    pcs = np.asarray(right[:16], dtype=np.float64)
    _need(pcs.shape == (16, EXPECTED_NEURONS), "source PCA shape drift")
    return mean, scale, pcs


def _ridge_fit(record: H1PilotRecord, mean: np.ndarray, scale: np.ndarray, pcs: np.ndarray, q: int, lam: float) -> np.ndarray:
    trials = [record.blocks_for(value) for value in record.trial_values[:SUPPORT_TRIALS]]
    rates = np.concatenate([item.rates for item in trials], axis=0).astype(np.float64)
    labels = np.concatenate([item.velocity for item in trials], axis=0).astype(np.float64)
    projected = ((rates - mean[None, :]) / scale[None, :]) @ pcs[:q].T
    design = np.column_stack((np.ones(len(projected), dtype=np.float64), projected))
    regularizer = np.eye(design.shape[1], dtype=np.float64) * float(lam)
    regularizer[0, 0] = 0.0
    return np.linalg.solve(design.T @ design + regularizer, design.T @ labels)


def _ridge_score(record: H1PilotRecord, mean: np.ndarray, scale: np.ndarray, pcs: np.ndarray, q: int, lam: float) -> float:
    beta = _ridge_fit(record, mean, scale, pcs, q, lam)
    later = [record.blocks_for(value) for value in record.trial_values[SUPPORT_TRIALS:]]
    _need(bool(later), f"{record.session_name}: no post-M4 analytic query trials")
    rates = np.concatenate([item.rates for item in later], axis=0).astype(np.float64)
    truth = np.concatenate([item.velocity for item in later], axis=0).astype(np.float64)
    projected = ((rates - mean[None, :]) / scale[None, :]) @ pcs[:q].T
    estimate = np.column_stack((np.ones(len(projected), dtype=np.float64), projected)) @ beta
    return variance_weighted_r2(truth, estimate)


@dataclass(frozen=True)
class RegenPlan:
    outer_date: str
    source_sessions: tuple[str, ...]
    source_input_sha256: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    pcs: np.ndarray
    q: int
    ridge_lambda: float
    U: np.ndarray
    mu: np.ndarray
    tau2: float
    selection_sha256: str
    transform_sha256: str


def _raw_rows(record: H1PilotRecord, plan: Any) -> np.ndarray:
    beta = _ridge_fit(record, plan.mean, plan.scale, plan.pcs, plan.q, plan.ridge_lambda)
    return np.asarray((plan.pcs[:plan.q].T @ beta[1:]) / plan.scale[:, None], dtype=np.float64)


def _make_final_plan(records: Mapping[str, H1PilotRecord], outer_date: str, selected: Mapping[str, Any], selection_sha: str) -> RegenPlan:
    mean, scale, pcs = _fit_basis(records)
    provisional = SimpleNamespace(mean=mean, scale=scale, pcs=pcs, q=int(selected["q"]), ridge_lambda=float(selected["lambda"]))
    pooled = np.concatenate([_raw_rows(record, provisional) for record in records.values()], axis=0)
    _, _, right = np.linalg.svd(pooled, full_matrices=False)
    U = np.asarray(right[:4].T, dtype=np.float64)
    carriers = pooled @ U
    mu = carriers.mean(axis=0, dtype=np.float64)
    tau2 = float(np.square(carriers - mu[None, :], dtype=np.float64).sum(dtype=np.float64) / (carriers.shape[0] * 4))
    _need(U.shape == (7, 4) and mu.shape == (4,) and math.isfinite(tau2) and tau2 > 1.0e-12, "EB prior is invalid")
    source = tuple(records)
    hashes = tuple(records[name].input_sha256 for name in source)
    body = {
        "schema": f"{SCHEMA}_plan",
        "outer_date": outer_date,
        "source_sessions": list(source),
        "source_input_sha256": list(hashes),
        "q": int(selected["q"]),
        "lambda": float(selected["lambda"]),
        "selection_sha256": selection_sha,
        "array_sha256": {name: array_sha256(value) for name, value in {"mean": mean, "scale": scale, "pcs": pcs, "U": U, "mu": mu}.items()},
        "tau2": tau2,
    }
    return RegenPlan(outer_date, source, hashes, mean, scale, pcs, int(selected["q"]), float(selected["lambda"]), U, mu, tau2, selection_sha, canonical_sha256(body))


def _legal_starts(record: H1PilotRecord) -> tuple[int, ...]:
    starts: list[int] = []
    for start in range(len(record.trial_values) - SUPPORT_TRIALS + 1):
        values = record.trial_values[start:start + SUPPORT_TRIALS]
        if all(record.blocks_for(value).rates.shape[0] >= 2 for value in values):
            for value in values:
                record.eval_trial_neural(value)
            starts.append(start)
    _need(bool(starts), f"{record.session_name}: no legal contiguous M4 support")
    return tuple(starts)


def _window_indices(records: Mapping[str, H1PilotRecord]) -> tuple[tuple[str, int], ...]:
    return tuple((name, int(index)) for name, record in records.items() for index, valid in enumerate(record.eval_mask) if bool(valid))


def _window_hash(indices: Sequence[tuple[str, int]]) -> str:
    digest = hashlib.sha256()
    for session, start in indices:
        digest.update(session.encode("ascii")); digest.update(int(start).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def _build_schedule(records: Mapping[str, H1PilotRecord], starts_by_session: Mapping[str, Sequence[int]], outer_date: str) -> tuple[np.ndarray, np.ndarray, tuple[tuple[str, int], ...]]:
    windows = _window_indices(records)
    grouped: dict[str, list[int]] = {name: [] for name in records}
    for index, (name, _start) in enumerate(windows):
        grouped[name].append(index)
    batches: list[list[int]] = []
    for name in records:
        token = int.from_bytes(hashlib.sha256(f"{FIXED_SEED}|{outer_date}|carrierid-date-lodo-batch|{name}".encode()).digest()[:8], "big")
        permutation = random.Random(token).sample(grouped[name], len(grouped[name]))
        batches.extend(permutation[offset:offset + BATCH_SIZE] for offset in range(0, len(permutation), BATCH_SIZE) if len(permutation[offset:offset + BATCH_SIZE]) == BATCH_SIZE)
    token = int.from_bytes(hashlib.sha256(f"{FIXED_SEED}|{outer_date}|carrierid-date-lodo-batches".encode()).digest()[:8], "big")
    batches = random.Random(token).sample(batches, len(batches))
    order = np.asarray([index for batch in batches for index in batch], dtype=np.int64)
    session_vector = np.asarray([windows[int(index)][0] for index in order], dtype=object)
    schedule = np.empty((FIXED_EPOCHS, len(order)), dtype=np.int16)
    for name in records:
        positions = np.flatnonzero(session_vector == name)
        legal = np.asarray(starts_by_session[name], dtype=np.int16)
        token = hashlib.sha256(f"{FIXED_SEED}|{outer_date}|carrierid-date-lodo-m4|{name}".encode()).digest()
        draws = np.random.default_rng(int.from_bytes(token[:8], "big")).integers(0, len(legal), size=(FIXED_EPOCHS, len(positions)))
        schedule[:, positions] = legal[draws]
    _need(len(order) > 0 and len(order) % BATCH_SIZE == 0, "source batch order is empty or incomplete")
    return order, schedule, windows


def _date_selection(records: Mapping[str, H1PilotRecord], outer_date: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dates = tuple(sorted({session_date(name) for name in records}))
    rows: list[dict[str, Any]] = []
    basis = {}
    for held_date in dates:
        training = {name: record for name, record in records.items() if session_date(name) != held_date}
        held = {name: record for name, record in records.items() if session_date(name) == held_date}
        _need(bool(training) and bool(held), "source-inner date fold is empty")
        basis[held_date] = (*_fit_basis(training), held)
    for q in Q_GRID:
        for lam in LAMBDA_GRID:
            by_date = {}
            by_recording = {}
            for held_date in dates:
                mean, scale, pcs, held = basis[held_date]
                scores = {name: _ridge_score(record, mean, scale, pcs, q, lam) for name, record in held.items()}
                by_recording[held_date] = scores
                by_date[held_date] = float(np.mean(list(scores.values()), dtype=np.float64))
            values = list(by_date.values())
            rows.append({
                "q": q, "lambda": lam,
                "equal_date_mean_r2": float(np.mean(values, dtype=np.float64)),
                "worst_date_r2": float(min(values)),
                "r2_by_date": by_date,
                "r2_by_recording": by_recording,
            })
    selected = select_candidate(rows)
    return selected, rows


def dry_plan() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "DRY_NO_WRITE_NO_DATA_NO_CUDA",
        "outer_dates": list(CONFIRMATORY_DATES),
        "q_grid": list(Q_GRID),
        "lambda_grid": list(LAMBDA_GRID),
        "selection": "source-inner date-LODO equal-date analytic variance-weighted R2",
        "training": {"seed": 42, "epochs": 50, "terminal_epoch_zero_based": 49, "batch_size": 32, "fp32": True, "deterministic": False},
        "target_access": 0,
    }


def create_attempt(result_root: Path, closure: Mapping[str, str], head: str) -> dict[str, Any]:
    root = result_root.resolve()
    _need(not root.exists() and not root.is_symlink() and not os.path.lexists(str(root)), f"canonical result root is not fresh: {root}")
    body = {
        "schema": SCHEMA,
        "artifact": "attempt",
        "status": "ATTEMPT_BEFORE_DATA_AND_CUDA",
        "created_at_utc": utc_now(),
        "head": head,
        "closure": dict(closure),
        "outer_dates": list(CONFIRMATORY_DATES),
        "target_recordings_opened": 0,
        "target_bytes_read": 0,
        "cuda_initialized": False,
    }
    publish_json(root / "attempt.json", body)
    return body


def load_attempt(result_root: Path) -> dict[str, Any]:
    path = result_root.resolve() / "attempt.json"
    verify_sidecar(path)
    body = json.loads(path.read_text(encoding="utf-8"))
    _need(body.get("schema") == SCHEMA and body.get("status") == "ATTEMPT_BEFORE_DATA_AND_CUDA", "attempt schema/status drift")
    _need(body.get("target_recordings_opened") == 0 and body.get("target_bytes_read") == 0, "attempt records target access")
    return body


def prepare_source_authority(data_root: Path, result_root: Path) -> dict[str, Any]:
    load_attempt(result_root)
    root = result_root.resolve()
    date_rows = []
    for outer_date in CONFIRMATORY_DATES:
        directory = root / "source_authority" / outer_date
        _need(not directory.exists(), f"source authority date exists: {outer_date}")
        started = time.monotonic()
        records, audit = load_source_records_with_target_filename_index(data_root, outer_date)
        _need(tuple(records) == tuple(source_sessions_for_date(outer_date)), "source roster drift")
        selected, candidates = _date_selection(records, outer_date)
        selection = {
            "schema": f"{SCHEMA}_selection",
            "outer_date": outer_date,
            "candidate_grid": {"q": list(Q_GRID), "lambda": list(LAMBDA_GRID)},
            "metric": "float64 variance-weighted R2; equal recording within date; equal date governing",
            "tie_break": ["higher equal_date_mean_r2", "higher worst_date_r2", "smaller q", "larger lambda"],
            "candidates": candidates,
            "selected": selected,
            "target_recordings_opened": 0,
            "target_bytes_read": 0,
        }
        selection_sha = publish_json(directory / "selection.json", selection)
        plan = _make_final_plan(records, outer_date, selected, selection_sha)
        plan_arrays_sha = publish_npz(directory / "plan.npz", mean=plan.mean, scale=plan.scale, pcs=plan.pcs, q=np.asarray(plan.q, np.int64), **{"lambda": np.asarray(plan.ridge_lambda, np.float64)}, U=plan.U, mu=plan.mu, tau2=np.asarray(plan.tau2, np.float64))
        plan_body = {
            "schema": f"{SCHEMA}_plan",
            "outer_date": outer_date,
            "source_sessions": list(plan.source_sessions),
            "source_input_sha256": list(plan.source_input_sha256),
            "q": plan.q,
            "lambda": plan.ridge_lambda,
            "tau2": plan.tau2,
            "selection_sha256": selection_sha,
            "transform_sha256": plan.transform_sha256,
            "arrays_file_sha256": plan_arrays_sha,
            "array_sha256": {name: array_sha256(value) for name, value in {"mean": plan.mean, "scale": plan.scale, "pcs": plan.pcs, "U": plan.U, "mu": plan.mu}.items()},
        }
        plan_sha = publish_json(directory / "plan.json", plan_body)
        entries = []
        carriers = []
        starts_by_session = {}
        for name, record in records.items():
            starts = _legal_starts(record); starts_by_session[name] = starts
            for start in starts:
                values = tuple(float(value) for value in record.trial_values[start:start + SUPPORT_TRIALS])
                carrier = np.asarray(fit_frozen_carrier(record, plan, values)["carrier"], dtype=np.float64)
                entries.append({"session": name, "start_index": start, "trial_values": list(values), "carrier_sha256": carrier_sha256(carrier)})
                carriers.append(carrier)
        carrier_array = np.stack(carriers, axis=0)
        cache_arrays_sha = publish_npz(directory / "carrier_cache.npz", carriers=carrier_array)
        cache_body = {
            "schema": f"{SCHEMA}_carrier_cache",
            "outer_date": outer_date,
            "transform_sha256": plan.transform_sha256,
            "shape": list(carrier_array.shape),
            "entries": entries,
            "tensor_sha256": array_sha256(carrier_array),
            "arrays_file_sha256": cache_arrays_sha,
        }
        cache_body["cache_sha256"] = canonical_sha256(cache_body)
        cache_sha = publish_json(directory / "carrier_cache.json", cache_body)
        scalar = float(np.sqrt(np.mean(np.square(carrier_array, dtype=np.float64), dtype=np.float64)))
        _need(math.isfinite(scalar) and scalar >= 0.0, "source RMS normalizer is invalid")
        normalized = carrier_array / max(scalar, NORMALIZER_FLOOR)
        normalizer = {
            "schema": f"{SCHEMA}_normalizer",
            "formula": NORMALIZER_FORMULA,
            "floor": NORMALIZER_FLOOR,
            "s_src": scalar,
            "cache_sha256": cache_sha,
            "normalized_tensor_sha256": array_sha256(normalized),
        }
        normalizer_sha = publish_json(directory / "normalizer.json", normalizer)
        order, schedule, windows = _build_schedule(records, starts_by_session, outer_date)
        order_sha = publish_npy(directory / "batch_order.npy", order)
        schedule_file_sha = publish_npy(directory / "schedule.npy", schedule)
        schedule_body = {
            "schema": f"{SCHEMA}_schedule",
            "outer_date": outer_date,
            "epochs": FIXED_EPOCHS,
            "batch_size": BATCH_SIZE,
            "batches_per_epoch": int(len(order) // BATCH_SIZE),
            "steps_total": int(FIXED_EPOCHS * len(order) // BATCH_SIZE),
            "window_indices_sha256": _window_hash(windows),
            "batch_order_tensor_sha256": array_sha256(order),
            "batch_order_file_sha256": order_sha,
            "schedule_tensor_sha256": array_sha256(schedule),
            "schedule_file_sha256": schedule_file_sha,
        }
        schedule_sha = publish_json(directory / "schedule.json", schedule_body)
        authority = {
            "schema": f"{SCHEMA}_date_source_authority",
            "status": STATUS_SOURCE,
            "outer_date": outer_date,
            "source_sessions": list(records),
            "source_files": [{"session": name, "sha256": records[name].input_sha256} for name in records],
            "target_filenames_indexed_only": list(audit.target_filenames),
            "target_sessions_indexed": list(audit.target_sessions_indexed),
            "target_recordings_opened": 0,
            "target_bytes_read": 0,
            "selection_sha256": selection_sha,
            "plan_sha256": plan_sha,
            "carrier_cache_sha256": cache_sha,
            "normalizer_sha256": normalizer_sha,
            "schedule_sha256": schedule_sha,
            "elapsed_seconds": time.monotonic() - started,
        }
        authority_sha = publish_json(directory / "authority.json", authority)
        date_rows.append({"outer_date": outer_date, "relative": str((directory / "authority.json").relative_to(root)), "sha256": authority_sha})
    top = {
        "schema": f"{SCHEMA}_source_authority",
        "status": STATUS_SOURCE,
        "created_at_utc": utc_now(),
        "date_order": list(CONFIRMATORY_DATES),
        "dates": date_rows,
        "target_recordings_opened": 0,
        "target_bytes_read": 0,
    }
    publish_json(root / "source_authority.json", top)
    return top


def _load_json(path: Path, schema: str | None = None) -> tuple[dict[str, Any], str]:
    digest = verify_sidecar(path)
    body = json.loads(path.read_text(encoding="utf-8"))
    if schema is not None:
        _need(body.get("schema") == schema, f"schema drift: {path}")
    return body, digest


def _load_date_materialization(data_root: Path, result_root: Path, outer_date: str):
    directory = result_root.resolve() / "source_authority" / outer_date
    authority, authority_sha = _load_json(directory / "authority.json", f"{SCHEMA}_date_source_authority")
    _need(authority.get("status") == STATUS_SOURCE and authority.get("target_bytes_read") == 0 and authority.get("target_recordings_opened") == 0, "date authority is not source-only")
    records, audit = load_source_records_with_target_filename_index(data_root, outer_date)
    audit_body = audit.manifest()
    _need(audit_body["target_bytes_read"] == 0 and audit_body["target_recordings_opened"] == 0, "source reload accessed target bytes")
    _need(tuple(records) == tuple(authority["source_sessions"]), "source reload roster drift")
    expected_files = {row["session"]: row["sha256"] for row in authority["source_files"]}
    _need(all(records[name].input_sha256 == expected_files[name] for name in records), "source NWB SHA drift")
    plan_body, plan_sha = _load_json(directory / "plan.json", f"{SCHEMA}_plan")
    _need(plan_sha == authority["plan_sha256"], "plan authority SHA drift")
    _need(verify_sidecar(directory / "plan.npz") == plan_body["arrays_file_sha256"], "plan arrays file SHA drift")
    with np.load(directory / "plan.npz", allow_pickle=False) as arrays:
        plan = RegenPlan(
            outer_date, tuple(records), tuple(records[name].input_sha256 for name in records),
            np.asarray(arrays["mean"], np.float64), np.asarray(arrays["scale"], np.float64), np.asarray(arrays["pcs"], np.float64),
            int(arrays["q"].item()), float(arrays["lambda"].item()), np.asarray(arrays["U"], np.float64), np.asarray(arrays["mu"], np.float64),
            float(arrays["tau2"].item()), str(plan_body["selection_sha256"]), str(plan_body["transform_sha256"]),
        )
    _need(int(plan_body["q"]) == plan.q and float(plan_body["lambda"]) == plan.ridge_lambda, "plan scalar drift")
    cache_body, cache_sha = _load_json(directory / "carrier_cache.json", f"{SCHEMA}_carrier_cache")
    _need(cache_sha == authority["carrier_cache_sha256"], "cache authority SHA drift")
    _need(verify_sidecar(directory / "carrier_cache.npz") == cache_body["arrays_file_sha256"], "cache arrays file SHA drift")
    with np.load(directory / "carrier_cache.npz", allow_pickle=False) as arrays:
        carriers = np.asarray(arrays["carriers"], np.float64)
    _need(array_sha256(carriers) == cache_body["tensor_sha256"], "carrier cache tensor drift")
    normalizer, normalizer_sha = _load_json(directory / "normalizer.json", f"{SCHEMA}_normalizer")
    _need(normalizer_sha == authority["normalizer_sha256"], "normalizer authority SHA drift")
    schedule_body, schedule_sha = _load_json(directory / "schedule.json", f"{SCHEMA}_schedule")
    _need(schedule_sha == authority["schedule_sha256"], "schedule authority SHA drift")
    _need(verify_sidecar(directory / "batch_order.npy") == schedule_body["batch_order_file_sha256"], "batch order file SHA drift")
    _need(verify_sidecar(directory / "schedule.npy") == schedule_body["schedule_file_sha256"], "schedule file SHA drift")
    order = np.asarray(np.load(directory / "batch_order.npy", allow_pickle=False), np.int64)
    schedule = np.asarray(np.load(directory / "schedule.npy", allow_pickle=False), np.int16)
    _need(array_sha256(order) == schedule_body["batch_order_tensor_sha256"] and array_sha256(schedule) == schedule_body["schedule_tensor_sha256"], "schedule tensor drift")
    _need(schedule.shape == (FIXED_EPOCHS, len(order)) and len(order) % BATCH_SIZE == 0, "schedule shape/batch schema drift")
    return records, plan, cache_body, carriers, float(normalizer["s_src"]), order, schedule, authority_sha


class SourceDataset:
    def __init__(self, records: Mapping[str, H1PilotRecord], cache_body: Mapping[str, Any], carriers: np.ndarray, normalizer: float):
        self.records = dict(records)
        self.windows = _window_indices(records)
        self.prehistory = WINDOW_SIZE - 1
        self.neural = {name: np.pad(record.neural, ((self.prehistory, 0), (0, 0)), constant_values=0.0) for name, record in records.items()}
        self.target = {name: np.pad(record.velocity, ((self.prehistory, 0), (0, 0)), constant_values=0.0) for name, record in records.items()}
        self.identity = {(name, float(value)): interpolate_trial_identity(record, value) for name, record in records.items() for value in record.trial_values}
        denominator = max(float(normalizer), NORMALIZER_FLOOR)
        self.carriers = {}
        for row, carrier in zip(cache_body["entries"], carriers, strict=True):
            self.carriers[(str(row["session"]), int(row["start_index"]))] = np.asarray(carrier / denominator, np.float32)

    def batch(self, window_rows: Sequence[int], support_starts: Sequence[int]):
        names = {self.windows[int(index)][0] for index in window_rows}
        _need(len(names) == 1, "training batch mixes source sessions")
        xs, ys, identities, carrier_rows = [], [], [], []
        for index, support_start in zip(window_rows, support_starts, strict=True):
            name, start = self.windows[int(index)]
            values = self.records[name].trial_values[int(support_start):int(support_start) + SUPPORT_TRIALS]
            _need(len(values) == SUPPORT_TRIALS, "scheduled support is incomplete")
            xs.append(self.neural[name][start:start + WINDOW_SIZE])
            ys.append(self.target[name][start:start + WINDOW_SIZE])
            identities.append(np.stack([self.identity[(name, float(value))] for value in values], axis=0))
            carrier_rows.append(self.carriers[(name, int(support_start))])
        return np.stack(xs), np.stack(ys), np.stack(identities), np.stack(carrier_rows), next(iter(names))


def model_config() -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA}_resolved_config",
        "model": "src.models.components.h1_carrierid_spint.H1CarrierIdSpint",
        "model_kwargs": {
            "carrier_hidden_dim": 32, "carrier_dim": 4, "carrier_trial_length": 1024, "zero_carrier": False,
            "model_dim": 1024, "num_covariates": 7, "window_size": 700, "num_heads": 64, "num_layers": 1,
            "num_id_layers": 3, "use_learnable_id": True, "learnable_id_type": "mlp", "learnable_rep": True,
            "dropout_rate": 0.0, "dynamic_dropout": True, "dynamic_dropout_low": 0.0, "dynamic_dropout_high": 1.0,
            "tf_drop_rate": 0.1, "readin_layer_type": "mlp",
        },
        "optimizer": {"name": "torch.optim.Adam", "lr": 5.0e-5, "weight_decay": 0.0},
        "loss": "last-bin MSE after prediction/20",
        "seed": 42, "batch_size": 32, "epochs": 50, "terminal_epoch_zero_based": 49,
        "precision": "float32", "deterministic": False, "warm_start": False,
    }


def _seed_all() -> None:
    random.seed(FIXED_SEED); np.random.seed(FIXED_SEED)
    import torch
    torch.manual_seed(FIXED_SEED); torch.cuda.manual_seed_all(FIXED_SEED)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _new_model(device: str):
    import torch
    from src.models.components.h1_carrierid_spint import H1CarrierIdSpint
    _seed_all()
    model = H1CarrierIdSpint(**model_config()["model_kwargs"])
    _need(sum(parameter.numel() for parameter in model.parameters()) == MODEL_PARAMETERS, "H-C parameter count drift")
    return model.to(torch.device(device))


def _gpu_profile(physical_gpu: int) -> dict[str, Any]:
    import torch
    properties = torch.cuda.get_device_properties(0)
    uuid = subprocess.check_output(
        ["nvidia-smi", "-i", str(int(physical_gpu)), "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    return {"visible_index": 0, "uuid": uuid, "name": properties.name, "memory_total_bytes": int(properties.total_memory)}


def _finite_optimizer(optimizer: Any) -> bool:
    import torch
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all().item():
                return False
    return True


def run_cell(data_root: Path, result_root: Path, outer_date: str, physical_gpu: int, *, smoke: bool = False, smoke_steps: int = 20) -> dict[str, Any]:
    _need(outer_date in CONFIRMATORY_DATES, "cell outer date is not canonical")
    cell_id = f"smoke_{outer_date}" if smoke else outer_date
    directory = result_root.resolve() / ("smoke" if smoke else "cells") / cell_id
    _need(not directory.exists(), f"cell directory exists: {directory}")
    attempt = {
        "schema": SCHEMA, "artifact": "cell_attempt", "cell_id": cell_id, "outer_date": outer_date,
        "smoke": smoke, "physical_gpu": physical_gpu, "created_at_utc": utc_now(),
        "target_recordings_opened": 0, "target_bytes_read": 0, "warm_start": False,
    }
    publish_json(directory / "attempt.json", attempt)
    started = time.monotonic()
    try:
        # Importing torch and the first CUDA query deliberately occur only
        # after the immutable cell attempt has been published.
        import torch
        _need(torch.cuda.is_available(), "cell requires CUDA")
        top_attempt, top_attempt_sha = _load_json(result_root.resolve() / "attempt.json", SCHEMA)
        closure_sha = canonical_sha256(top_attempt["closure"])
        records, _plan, cache_body, carriers, normalizer, order, schedule, authority_sha = _load_date_materialization(data_root, result_root, outer_date)
        config_sha = publish_json(directory / "config.json", model_config())
        dataset = SourceDataset(records, cache_body, carriers, normalizer)
        device = "cuda:0"
        model = _new_model(device); model.train()
        initial_sha = state_hash(model.state_dict())
        optimizer = torch.optim.Adam(model.parameters(), lr=5.0e-5, weight_decay=0.0)
        losses = []; global_step = 0; gradients_finite = True
        epochs = 1 if smoke else FIXED_EPOCHS
        for epoch in range(epochs):
            starts = schedule[epoch]
            for offset in range(0, len(order), BATCH_SIZE):
                if smoke and global_step >= smoke_steps:
                    break
                rows = order[offset:offset + BATCH_SIZE]; support = starts[offset:offset + BATCH_SIZE]
                neural, target, identity, carrier, _session = dataset.batch(rows, support)
                neural_t = torch.as_tensor(neural, dtype=torch.float32, device=device)
                target_t = torch.as_tensor(target, dtype=torch.float32, device=device)
                identity_t = torch.as_tensor(identity, dtype=torch.float32, device=device)
                carrier_t = torch.as_tensor(carrier, dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(neural_t, calib_trialized_neural_features=identity_t, carrier=carrier_t)
                loss = torch.nn.functional.mse_loss(prediction[:, -1:, :] / 20.0, target_t[:, -1:, :])
                _need(torch.isfinite(loss).item(), "training loss is nonfinite")
                loss.backward()
                gradients_finite = gradients_finite and all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in model.parameters())
                _need(gradients_finite, "training gradient is nonfinite")
                optimizer.step(); global_step += 1; losses.append(float(loss.detach().cpu()))
            if smoke and global_step >= smoke_steps:
                break
            print(json.dumps({"outer_date": outer_date, "epoch_zero_based": epoch, "global_step": global_step, "mean_loss": float(np.mean(losses[-max(1, len(order)//BATCH_SIZE):], dtype=np.float64))}), flush=True)
        _need(global_step == (smoke_steps if smoke else FIXED_EPOCHS * len(order) // BATCH_SIZE), "cell step count drift")
        _need(_finite_optimizer(optimizer), "Adam state is nonfinite")
        terminal_state = state_hash(model.state_dict())
        gpu = _gpu_profile(physical_gpu)
        base = {
            "schema": SCHEMA, "cell_id": cell_id, "outer_date": outer_date, "smoke": smoke,
            "physical_gpu": physical_gpu, "gpu": gpu, "source_authority_sha256": authority_sha,
            "experiment_attempt_sha256": top_attempt_sha, "code_closure_sha256": closure_sha,
            "config_sha256": config_sha, "initial_state_sha256": initial_sha, "terminal_state_sha256": terminal_state,
            "global_step": global_step, "epoch_zero_based": 0 if smoke else 49,
            "loss_first": losses[0], "loss_last": losses[-1], "all_losses_finite": True,
            "all_gradients_finite": gradients_finite, "adam_state_finite": True,
            "target_recordings_opened": 0, "target_bytes_read": 0, "target_optimizer_steps": 0,
            "elapsed_seconds": time.monotonic() - started, "finished_at_utc": utc_now(),
        }
        if smoke:
            body = {**base, "status": STATUS_SMOKE}
            publish_json(directory / "terminal.json", body)
            return body
        metadata = {
            "schema": CHECKPOINT_SCHEMA, "outer_date": outer_date, "fresh_seed": 42,
            "checkpoint_epoch_zero_based": 49, "epochs_completed": 50,
            "selected_by": "fixed_terminal_epoch_no_validation_or_target_selection",
            "warm_start": False, "config_sha256": config_sha, "source_authority_sha256": authority_sha,
            "experiment_attempt_sha256": top_attempt_sha, "code_closure_sha256": closure_sha,
            "initial_state_sha256": initial_sha, "terminal_state_sha256": terminal_state,
            "global_step": global_step, "target_recordings_opened": 0, "target_bytes_read": 0,
            "target_optimizer_steps": 0, "target_backward_steps": 0,
        }
        checkpoint = directory / "epoch_049.ckpt"
        torch.save({"state_dict": model.state_dict(), "metadata": metadata}, checkpoint)
        os.chmod(checkpoint, 0o444)
        checkpoint_sha = sha256_file(checkpoint)
        _publish_bytes(checkpoint.with_name(checkpoint.name + ".sha256"), f"{checkpoint_sha}  {checkpoint.name}\n".encode("ascii"), sidecar=False)
        body = {**base, "status": STATUS_CELL, "checkpoint_relative": str(checkpoint.relative_to(result_root.resolve())), "checkpoint_sha256": checkpoint_sha, "checkpoint_metadata": metadata}
        publish_json(directory / "terminal.json", body)
        return body
    except BaseException as error:
        failure = {
            "schema": SCHEMA, "artifact": "cell_failure", "status": "FAIL_CELL_NO_AUTOMATIC_RETRY",
            "cell_id": cell_id, "outer_date": outer_date, "smoke": smoke, "physical_gpu": physical_gpu,
            "error_type": type(error).__name__, "error": str(error), "target_recordings_opened": 0,
            "target_bytes_read": 0, "finished_at_utc": utc_now(), "elapsed_seconds": time.monotonic() - started,
        }
        try:
            publish_json(directory / "failure.json", failure)
        except BaseException:
            pass
        raise


def validate_checkpoint_contract(
    metadata: Mapping[str, Any], terminal: Mapping[str, Any], outer_date: str,
    attempt_sha: str, closure_sha: str, source_authority_sha: str,
) -> None:
    _need(metadata.get("schema") == CHECKPOINT_SCHEMA, f"checkpoint schema drift: {outer_date}")
    _need(metadata.get("outer_date") == outer_date, f"checkpoint outer date drift: {outer_date}")
    _need(metadata.get("checkpoint_epoch_zero_based") == 49 and metadata.get("epochs_completed") == 50, f"checkpoint epoch drift: {outer_date}")
    _need(metadata.get("warm_start") is False and metadata.get("fresh_seed") == 42, f"checkpoint warm-start/seed drift: {outer_date}")
    _need(metadata.get("selected_by") == "fixed_terminal_epoch_no_validation_or_target_selection", f"checkpoint selection drift: {outer_date}")
    _need(metadata.get("experiment_attempt_sha256") == attempt_sha and metadata.get("code_closure_sha256") == closure_sha, f"checkpoint code closure drift: {outer_date}")
    _need(metadata.get("source_authority_sha256") == source_authority_sha, f"checkpoint source authority drift: {outer_date}")
    _need(metadata.get("config_sha256") == terminal.get("config_sha256"), f"checkpoint configuration drift: {outer_date}")
    _need(metadata.get("initial_state_sha256") == terminal.get("initial_state_sha256"), f"checkpoint initial state drift: {outer_date}")
    _need(metadata.get("terminal_state_sha256") == terminal.get("terminal_state_sha256"), f"checkpoint terminal state drift: {outer_date}")
    _need(int(metadata.get("global_step", 0)) == int(terminal.get("global_step", -1)) > 0, f"checkpoint step drift: {outer_date}")
    for field in ("target_recordings_opened", "target_bytes_read", "target_optimizer_steps", "target_backward_steps"):
        _need(metadata.get(field) == 0, f"checkpoint records target activity ({field}): {outer_date}")


def verify_terminal(result_root: Path) -> dict[str, Any]:
    import torch
    root = result_root.resolve(); attempt = load_attempt(root)
    attempt_sha = verify_sidecar(root / "attempt.json")
    closure_sha = canonical_sha256(attempt["closure"])
    source, source_sha = _load_json(root / "source_authority.json", f"{SCHEMA}_source_authority")
    _need(source.get("status") == STATUS_SOURCE and source.get("target_bytes_read") == 0 and source.get("target_recordings_opened") == 0, "top source authority drift")
    _need(tuple(source.get("date_order", ())) == CONFIRMATORY_DATES, "top source date set/order drift")
    source_rows = {row.get("outer_date"): row for row in source.get("dates", ())}
    _need(tuple(source_rows) == CONFIRMATORY_DATES, "top source authority is not the complete five-date set")
    date_authorities = {}
    for date in CONFIRMATORY_DATES:
        date_path = root / str(source_rows[date]["relative"])
        _need(verify_sidecar(date_path) == source_rows[date]["sha256"], f"top source date SHA drift: {date}")
        date_authorities[date] = json.loads(date_path.read_text(encoding="utf-8"))
    rows = []
    initial = set()
    for date in CONFIRMATORY_DATES:
        directory = root / "cells" / date
        terminal, terminal_sha = _load_json(directory / "terminal.json")
        _need(terminal.get("status") == STATUS_CELL and terminal.get("outer_date") == date, f"cell terminal drift: {date}")
        _need(terminal.get("epoch_zero_based") == 49 and terminal.get("smoke") is False, f"cell epoch/smoke drift: {date}")
        _need(terminal.get("target_recordings_opened") == 0 and terminal.get("target_bytes_read") == 0 and terminal.get("target_optimizer_steps") == 0, f"cell target access: {date}")
        checkpoint = root / terminal["checkpoint_relative"]
        checkpoint_sha = verify_sidecar(checkpoint)
        _need(checkpoint_sha == terminal["checkpoint_sha256"], f"checkpoint SHA drift: {date}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
        _need(isinstance(metadata, Mapping) and dict(metadata) == terminal["checkpoint_metadata"], f"checkpoint metadata drift: {date}")
        validate_checkpoint_contract(metadata, terminal, date, attempt_sha, closure_sha, source_rows[date]["sha256"])
        _need(state_hash(payload["state_dict"]) == terminal["terminal_state_sha256"], f"checkpoint state drift: {date}")
        config, config_sha = _load_json(directory / "config.json", f"{SCHEMA}_resolved_config")
        _need(config_sha == terminal["config_sha256"] == metadata["config_sha256"], f"config SHA drift: {date}")
        _need(config == model_config(), f"resolved config contract drift: {date}")
        schedule_body, schedule_sha = _load_json(root / "source_authority" / date / "schedule.json", f"{SCHEMA}_schedule")
        _need(schedule_sha == date_authorities[date]["schedule_sha256"], f"schedule authority drift: {date}")
        _need(terminal["global_step"] == schedule_body["steps_total"], f"terminal optimizer step count drift: {date}")
        initial.add(terminal["initial_state_sha256"])
        rows.append({"outer_date": date, "terminal_relative": str((directory / "terminal.json").relative_to(root)), "terminal_sha256": terminal_sha, "checkpoint_relative": terminal["checkpoint_relative"], "checkpoint_sha256": checkpoint_sha, "config_sha256": config_sha, "source_authority_sha256": terminal["source_authority_sha256"], "global_step": terminal["global_step"], "gpu": terminal["gpu"]})
    _need(len(initial) == 1, "five date cells do not share the canonical initial state")
    body = {
        "schema": SCHEMA, "status": STATUS_TERMINAL, "finished_at_utc": utc_now(),
        "date_order": list(CONFIRMATORY_DATES), "source_authority_sha256": source_sha,
        "experiment_attempt_sha256": attempt_sha, "code_closure_sha256": closure_sha,
        "canonical_initial_state_sha256": next(iter(initial)), "cells": rows,
        "target_recordings_opened": 0, "target_bytes_read": 0, "target_optimizer_steps": 0,
        "claim": "new source-only H-C checkpoint authority; historical checkpoint bytes and target scores are not reproduced",
    }
    terminal_sha = publish_json(root / "terminal.json", body)
    lines = [
        "# H1 five-date H-C Regeneration Successor V1", "", f"- Status: `{STATUS_TERMINAL}`",
        "- Scope: source-only checkpoint authority; no outer-date target recording was opened.",
        "- Selection: source-inner date-LODO over q={4,8,12,16}, lambda={1e-3,1e-2,1e-1,1,10}.",
        "- Training: H1CarrierIdSpint h=32, seed 42, FP32, 50 epochs, fixed epoch 49, no validation or warm start.", "",
        "| Outer date | Steps | Checkpoint SHA-256 | GPU UUID |", "|---|---:|---|---|",
    ]
    lines.extend(f"| {row['outer_date']} | {row['global_step']} | `{row['checkpoint_sha256']}` | `{row['gpu']['uuid']}` |" for row in rows)
    lines.extend(["", f"Terminal SHA-256: `{terminal_sha}`", ""])
    publish_text(root / "EXPERIMENT_RECORD.md", "\n".join(lines))
    return body


__all__ = (
    "BATCH_SIZE", "CHECKPOINT_SCHEMA", "LAMBDA_GRID", "MODEL_PARAMETERS", "Q_GRID", "RegenError", "SCHEMA",
    "STATUS_CELL", "STATUS_SMOKE", "STATUS_SOURCE", "STATUS_TERMINAL", "create_attempt", "dry_plan",
    "load_attempt", "model_config", "prepare_source_authority", "publish_json", "run_cell", "select_candidate",
    "validate_checkpoint_contract", "variance_weighted_r2", "verify_sidecar", "verify_terminal",
)
