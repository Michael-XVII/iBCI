"""Metadata-only H1 held-out feasibility access for the all-source successor.

This module deliberately does not import or call ``falcon_challenge.load_nwb``.
The only NWB datasets read by :func:`read_heldout_trial_metadata` are
``TrialNum`` and either ``eval_mask`` or the legacy ``Blacklist`` fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.data.h1_m4_eb_pilot import session_from_path


H1_HELDOUT_SESSIONS: tuple[str, ...] = (
    "ses-19250126T113454",
    "ses-19250126T114029",
    "ses-19250127T120333",
    "ses-19250127T120826",
    "ses-19250129T112555",
    "ses-19250129T113059",
    "ses-19250202T113958",
    "ses-19250202T114452",
    "ses-19250203T113515",
    "ses-19250203T114018",
    "ses-19250206T112219",
    "ses-19250206T112712",
    "ses-19250209T111826",
    "ses-19250209T112327",
)
HELDOUT_DIRECTORY = "sub-HumanPitt-held-out-calib"
M4_MINIMUM_LEGAL_TRIALS = 5


class HeldoutMetadataError(RuntimeError):
    pass


@dataclass(frozen=True)
class HeldoutTrialMetadata:
    session_name: str
    legal_trial_count: int
    m4_evaluable: bool
    validity_field: str


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise HeldoutMetadataError(message)


def index_heldout_calib(data_root: str | Path) -> dict[str, Path]:
    """Index exactly the registered 14 public held-out-calibration NWBs."""
    root = Path(data_root).resolve()
    _need(root.name == "000954" and root.is_dir(), f"expected existing 000954 data root, got {root}")
    directory = (root / HELDOUT_DIRECTORY).resolve()
    _need(directory.is_dir(), f"held-out calibration directory missing: {directory}")
    try:
        directory.relative_to(root)
    except ValueError as error:
        raise HeldoutMetadataError("held-out directory escapes data root") from error
    observed: dict[str, Path] = {}
    for candidate in sorted(directory.glob("*.nwb")):
        resolved = candidate.resolve()
        try:
            resolved.relative_to(directory)
        except ValueError as error:
            raise HeldoutMetadataError(f"held-out NWB escapes registered directory: {candidate}") from error
        name = session_from_path(resolved)
        _need(name not in observed, f"duplicate held-out session: {name}")
        observed[name] = resolved
    _need(set(observed) == set(H1_HELDOUT_SESSIONS) and len(observed) == 14,
          f"held-out roster drift: {sorted(observed)}")
    return {name: observed[name] for name in H1_HELDOUT_SESSIONS}


def _validate_registered_path(path: Path, data_root: Path) -> tuple[Path, str]:
    root = data_root.resolve()
    resolved = path.resolve()
    directory = (root / HELDOUT_DIRECTORY).resolve()
    _need(resolved.parent == directory, f"metadata loader accepts only direct held-out-calib files: {resolved}")
    _need(resolved.suffix == ".nwb" and resolved.is_file(), f"expected held-out NWB file: {resolved}")
    session = session_from_path(resolved)
    _need(session in H1_HELDOUT_SESSIONS, f"unregistered held-out session: {session}")
    return resolved, session


def _legal_trial_count(trial_num: np.ndarray, eval_mask: np.ndarray) -> int:
    labels = np.asarray(trial_num, dtype=np.float64).reshape(-1)
    mask = np.asarray(eval_mask, dtype=bool).reshape(-1)
    _need(labels.shape == mask.shape, "TrialNum/evaluation-validity length mismatch")
    ordered = labels[mask & np.isfinite(labels)]
    _need(ordered.size > 0, "no finite evaluation-valid TrialNum values")
    _need(not np.any(np.diff(ordered) < 0.0), "TrialNum is not chronological on evaluation-valid bins")
    count = 1 + int(np.count_nonzero(np.diff(ordered) != 0.0))
    _need(count > 0, "legal trial count is empty")
    return count


def read_heldout_trial_metadata(
    path: str | Path,
    data_root: str | Path,
    *,
    io_factory: Callable[..., Any] | None = None,
) -> HeldoutTrialMetadata:
    """Read only TrialNum and evaluation-validity data from one held-out NWB."""
    resolved, session = _validate_registered_path(Path(path), Path(data_root))
    if io_factory is None:
        from pynwb import NWBHDF5IO

        io_factory = NWBHDF5IO
    with io_factory(str(resolved), "r", load_namespaces=True) as io:
        nwb = io.read()
        acquisitions = nwb.acquisition
        _need("TrialNum" in acquisitions, f"{session}: TrialNum acquisition missing")
        trial_num = np.asarray(acquisitions["TrialNum"].data[:], dtype=np.float64)
        if "eval_mask" in acquisitions:
            eval_mask = np.asarray(acquisitions["eval_mask"].data[:], dtype=bool)
            validity_field = "eval_mask"
        else:
            _need("Blacklist" in acquisitions, f"{session}: eval_mask/Blacklist acquisition missing")
            eval_mask = ~np.asarray(acquisitions["Blacklist"].data[:], dtype=bool)
            validity_field = "Blacklist"
    count = _legal_trial_count(trial_num, eval_mask)
    return HeldoutTrialMetadata(
        session_name=session,
        legal_trial_count=count,
        m4_evaluable=count >= M4_MINIMUM_LEGAL_TRIALS,
        validity_field=validity_field,
    )


def audit_registered_heldout_metadata(
    data_root: str | Path,
    *,
    io_factory: Callable[..., Any] | None = None,
) -> tuple[HeldoutTrialMetadata, ...]:
    paths = index_heldout_calib(data_root)
    rows = tuple(
        read_heldout_trial_metadata(paths[session], data_root, io_factory=io_factory)
        for session in H1_HELDOUT_SESSIONS
    )
    _need(tuple(row.session_name for row in rows) == H1_HELDOUT_SESSIONS, "held-out audit order drift")
    return rows


__all__ = (
    "HELDOUT_DIRECTORY",
    "H1_HELDOUT_SESSIONS",
    "M4_MINIMUM_LEGAL_TRIALS",
    "HeldoutMetadataError",
    "HeldoutTrialMetadata",
    "audit_registered_heldout_metadata",
    "index_heldout_calib",
    "read_heldout_trial_metadata",
)
