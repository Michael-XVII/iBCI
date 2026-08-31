"""Receipt, grid, and gate laws for H1 masked dense-auxiliary V1."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "h1_masked_dense_aux_v1"
SOURCE_DATES = ("19250108", "19250113", "19250115", "19250119", "19250120")
LAMBDAS = (0.0, 0.1, 0.3, 1.0)


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sidecar_path(path: str | Path) -> Path:
    value = Path(path)
    return value.with_name(value.name + ".sha256")


def write_immutable_bytes(path: str | Path, content: bytes) -> tuple[Path, str]:
    destination = Path(path)
    sidecar = sidecar_path(destination)
    if destination.exists() or destination.is_symlink() or sidecar.exists() or sidecar.is_symlink():
        raise FileExistsError(f"immutable artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        digest = hashlib.sha256(content).hexdigest()
        os.chmod(temporary, 0o444)
        os.replace(temporary, destination)
        side_content = f"{digest}  {destination.name}\n".encode("ascii")
        side_descriptor, side_name = tempfile.mkstemp(prefix=f".{sidecar.name}.", dir=destination.parent)
        side_temporary = Path(side_name)
        with os.fdopen(side_descriptor, "wb") as stream:
            stream.write(side_content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(side_temporary, 0o444)
        os.replace(side_temporary, sidecar)
        return destination, digest
    finally:
        if temporary.exists():
            temporary.unlink()


def write_immutable_json(path: str | Path, payload: Mapping[str, Any]) -> tuple[Path, str]:
    return write_immutable_bytes(path, canonical_bytes(payload))


def verify_immutable(path: str | Path) -> str:
    artifact = Path(path)
    sidecar = sidecar_path(artifact)
    digest = sha256_file(artifact)
    expected = sidecar.read_text(encoding="ascii").split()[0]
    if expected != digest:
        raise ValueError(f"sidecar mismatch: {artifact}")
    if artifact.stat().st_mode & 0o222 or sidecar.stat().st_mode & 0o222:
        raise ValueError(f"artifact is writable: {artifact}")
    return digest


def source_cell_specs() -> tuple[dict[str, Any], ...]:
    return tuple(
        {"cell_id": f"source_{date}_lambda_{lam:g}", "validation_date": date, "lambda": lam}
        for date in SOURCE_DATES for lam in LAMBDAS
    )


def select_source_lambda(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    indexed = {(str(row["validation_date"]), float(row["lambda"])): float(row["r2_mean"]) for row in rows}
    expected = {(date, lam) for date in SOURCE_DATES for lam in LAMBDAS}
    if set(indexed) != expected:
        raise ValueError("source selection requires exactly the frozen 5x4 terminal grid")
    candidate_means = {
        lam: sum(indexed[(date, lam)] for date in SOURCE_DATES) / len(SOURCE_DATES)
        for lam in LAMBDAS if lam > 0
    }
    selected = min(candidate_means, key=lambda lam: (-candidate_means[lam], lam))
    deltas = {date: indexed[(date, selected)] - indexed[(date, 0.0)] for date in SOURCE_DATES}
    mean_delta = sum(deltas.values()) / len(deltas)
    positive = sum(value > 0 for value in deltas.values())
    worst = min(deltas.values())
    passed = mean_delta >= 0.01 and positive >= 4 and worst >= -0.02
    return {
        "schema": SCHEMA,
        "selected_lambda": selected,
        "candidate_equal_date_mean_r2": candidate_means,
        "paired_delta_r2_by_date": deltas,
        "mean_delta_r2": mean_delta,
        "positive_dates": positive,
        "worst_date_delta_r2": worst,
        "thresholds": {"mean_delta_r2_min": 0.01, "positive_dates_min": 4, "worst_date_delta_r2_min": -0.02},
        "source_gate_passed": passed,
        "verdict": "PASS_SOURCE_GATE_AUTHORIZE_OUTER" if passed else "STOP_SOURCE_GATE_NO_TARGET_ACCESS",
    }


def outer_gate(t0: Mapping[str, float], selected: Mapping[str, float]) -> dict[str, Any]:
    if set(t0) != set(selected) or len(t0) != 2:
        raise ValueError("outer gate requires the same exact two recordings")
    deltas = {name: float(selected[name]) - float(t0[name]) for name in sorted(t0)}
    mean_delta = sum(deltas.values()) / 2
    positive = sum(value > 0 for value in deltas.values())
    passed = mean_delta >= 0.01 and positive == 2
    return {
        "paired_delta_r2_by_recording": deltas,
        "equal_recording_mean_delta_r2": mean_delta,
        "positive_recordings": positive,
        "outer_gate_passed": passed,
        "verdict": "PASS_MASKED_DENSE_AUX_V1_OUTER" if passed else "FAIL_MASKED_DENSE_AUX_V1_OUTER",
    }
