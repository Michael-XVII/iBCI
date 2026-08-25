"""Multi-session DataModule for DANDI 000688 (Perich et al.) sorted SUA.

Extends the MC_Maze single-file loader to:
- scan multiple NWB files per subject
- chronological session-level train/val/test splits
- per-session variable unit counts and calibration tensors
- cursor_vel interpolation from irregular timestamps
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import fcntl
import lightning.pytorch as pl
import numpy as np
import torch
from pynwb import NWBHDF5IO
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader, Dataset, Sampler

from mc_maze.datamodule import bin_spikes

logger = logging.getLogger(__name__)

SESSION_DATE_RE = re.compile(r"ses-(?:CO|RT)-(\d{8})")
CACHE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class SessionRecord:
    name: str
    neural: np.ndarray
    behavior: np.ndarray
    calib_trials: np.ndarray
    valid_starts: np.ndarray
    source_unit_count: int | None = None
    channel_ids: np.ndarray | None = None
    signal_view: str = "sua"
    side_features: np.ndarray | None = None
    electrode_ids: np.ndarray | None = None


@contextmanager
def _exclusive_cache_lock(cache_path: Path):
    """Serialize construction of one cache entry across concurrent workers."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_suffix(f"{cache_path.suffix}.lock")
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _source_fingerprint(nwb_path: Path) -> dict[str, int | str]:
    stat = nwb_path.stat()
    return {
        "path": str(nwb_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _cache_key(payload: dict) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _write_npz_atomically(cache_path: Path, **arrays: np.ndarray) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".npz", prefix=f".{cache_path.stem}.",
        dir=cache_path.parent, delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_cached_session(cache_path: Path) -> SessionRecord:
    with np.load(cache_path, allow_pickle=False) as cache:
        neural = cache["neural"].astype(np.float32, copy=False)
        return SessionRecord(
            name=str(cache["name"].item()),
            neural=neural,
            behavior=cache["behavior"].astype(np.float32, copy=False),
            calib_trials=cache["calib_trials"].astype(np.float32, copy=False),
            valid_starts=cache["valid_starts"].astype(np.int64, copy=False),
            source_unit_count=(
                int(cache["source_unit_count"].item())
                if "source_unit_count" in cache.files
                else neural.shape[1]
            ),
            channel_ids=(
                cache["channel_ids"].astype(np.int64, copy=False)
                if "channel_ids" in cache.files
                else np.arange(neural.shape[1], dtype=np.int64)
            ),
            signal_view=(
                str(cache["signal_view"].item())
                if "signal_view" in cache.files
                else "sua"
            ),
            side_features=(
                cache["side_features"].astype(np.float32, copy=False)
                if "side_features" in cache.files
                else None
            ),
            electrode_ids=(
                cache["electrode_ids"].astype(np.int64, copy=False)
                if "electrode_ids" in cache.files
                else None
            ),
        )


def _session_cache_path(
    cache_dir: Path,
    nwb_path: Path,
    *,
    bin_size_ms: int,
    window_size: int,
    calibration_n_trials: int,
    max_trial_length: int,
    pad_value: float,
    interpolate_trials: bool,
    behavior_mean: np.ndarray,
    behavior_std: np.ndarray,
    trial_result_filter: str,
    exclude_calibration_trials_from_windows: bool,
    signal_view: str,
) -> Path:
    payload = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "source": _source_fingerprint(nwb_path),
        "bin_size_ms": bin_size_ms,
        "window_size": window_size,
        "calibration_n_trials": calibration_n_trials,
        "max_trial_length": max_trial_length,
        "pad_value": pad_value,
        "interpolate_trials": interpolate_trials,
        "behavior_mean": behavior_mean.astype(np.float32).tolist(),
        "behavior_std": behavior_std.astype(np.float32).tolist(),
        "trial_result_filter": trial_result_filter,
        "exclude_calibration_trials_from_windows": exclude_calibration_trials_from_windows,
    }
    if signal_view != "sua":
        payload["signal_view"] = signal_view
    key = _cache_key(payload)[:20]
    return cache_dir / "sessions" / f"{session_name_from_path(nwb_path)}_{key}.npz"


def _behavior_stats_cache_path(
    cache_dir: Path,
    train_files: Sequence[Path],
    bin_size_ms: int,
) -> Path:
    payload = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "kind": "behavior_stats",
        "bin_size_ms": bin_size_ms,
        "train_sources": [_source_fingerprint(path) for path in train_files],
    }
    return cache_dir / "behavior_stats" / f"{_cache_key(payload)[:20]}.npz"


def session_name_from_path(nwb_path: Path) -> str:
    stem = nwb_path.name.replace("_behavior+ecephys.nwb", "")
    return stem


def session_date_key(nwb_path: Path) -> str:
    match = SESSION_DATE_RE.search(nwb_path.name)
    if not match:
        raise ValueError(f"Cannot parse session date from {nwb_path.name}")
    return match.group(1)


def _validate_max_units_exclusive(max_units_exclusive: int | None) -> None:
    if max_units_exclusive is None:
        return
    if isinstance(max_units_exclusive, bool) or not isinstance(max_units_exclusive, int):
        raise ValueError(
            "max_units_exclusive must be a positive integer or None; "
            f"got {max_units_exclusive!r}"
        )
    if max_units_exclusive <= 0:
        raise ValueError(
            "max_units_exclusive must be a positive integer or None; "
            f"got {max_units_exclusive!r}"
        )


def _validate_signal_view(signal_view: str) -> None:
    if signal_view not in {"sua", "pseudo_mua"}:
        raise ValueError(
            "signal_view must be one of {'sua', 'pseudo_mua'}; "
            f"got {signal_view!r}"
        )


def electrode_ids_from_units(units_df) -> np.ndarray:
    """Extract one electrode id per sorted unit from an NWB units table."""
    if "electrodes" not in units_df:
        raise ValueError("NWB units table has no electrodes column for pseudo_mua")
    electrode_ids: list[int] = []
    for unit_index, electrode_region in enumerate(units_df["electrodes"]):
        region_index = getattr(electrode_region, "index", None)
        if region_index is None or len(region_index) != 1:
            raise ValueError(
                "pseudo_mua requires exactly one electrode per sorted unit; "
                f"unit row {unit_index} had {0 if region_index is None else len(region_index)}"
            )
        electrode_ids.append(int(region_index[0]))
    return np.asarray(electrode_ids, dtype=np.int64)


def pool_spikes_by_electrode(
    binned_spikes: np.ndarray,
    electrode_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum sorted-unit spike counts into deterministic electrode channels."""
    if binned_spikes.ndim != 2:
        raise ValueError(f"Expected binned spikes [time, units], got {binned_spikes.shape}")
    if electrode_ids.ndim != 1 or electrode_ids.shape[0] != binned_spikes.shape[1]:
        raise ValueError(
            "electrode_ids must contain exactly one value per unit; got "
            f"{electrode_ids.shape} for {binned_spikes.shape[1]} units"
        )
    channel_ids, inverse = np.unique(electrode_ids, return_inverse=True)
    pooled = np.zeros((binned_spikes.shape[0], channel_ids.size), dtype=np.float32)
    np.add.at(pooled.T, inverse, binned_spikes.T)
    return pooled, channel_ids


def nwb_unit_count(nwb_path: Path) -> int:
    """Return an NWB file's unit-table row count without loading spike data."""
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        if nwb.units is None:
            raise ValueError(f"NWB file has no units table: {nwb_path}")
        return len(nwb.units)


def discover_nwb_files(
    data_dir: Path,
    task: str = "CO",
    max_units_exclusive: int | None = None,
) -> list[Path]:
    """Discover chronological task files, optionally retaining only units < threshold."""
    _validate_max_units_exclusive(max_units_exclusive)
    pattern = f"*ses-{task.upper()}-*_behavior+ecephys.nwb"
    files = sorted(data_dir.glob(pattern), key=session_date_key)
    if not files:
        raise FileNotFoundError(
            f"No NWB files matching {pattern} under {data_dir}"
        )

    if max_units_exclusive is None:
        return files

    kept_files: list[Path] = []
    excluded: list[tuple[str, int]] = []
    for nwb_path in files:
        unit_count = nwb_unit_count(nwb_path)
        if unit_count < max_units_exclusive:
            kept_files.append(nwb_path)
        else:
            excluded.append((session_name_from_path(nwb_path), unit_count))

    logger.info(
        "Discovered %d NWB sessions; units < %d retained %d and excluded %d",
        len(files),
        max_units_exclusive,
        len(kept_files),
        len(excluded),
    )
    if excluded:
        logger.info(
            "Excluded sessions for units < %d: %s",
            max_units_exclusive,
            ", ".join(f"{name} (units={count})" for name, count in excluded),
        )
    if not kept_files:
        raise ValueError(
            f"No NWB sessions remain after filtering for units < "
            f"{max_units_exclusive} (discovered {len(files)} sessions)"
        )
    return kept_files


def chronological_session_split(
    files: Sequence[Path],
    split_counts: tuple[int, int, int],
    *,
    max_units_exclusive: int | None = None,
) -> tuple[list[Path], list[Path], list[Path]]:
    n_train, n_val, n_test = split_counts
    expected = n_train + n_val + n_test
    if expected != len(files):
        filter_description = (
            f" after filtering for units < {max_units_exclusive}"
            if max_units_exclusive is not None
            else ""
        )
        raise ValueError(
            f"split_counts sum to {expected} but found {len(files)} sessions"
            f"{filter_description}"
        )
    train_files = list(files[:n_train])
    val_files = list(files[n_train : n_train + n_val])
    test_files = list(files[n_train + n_val :])
    return train_files, val_files, test_files


def load_frozen_train_val_manifest(
    manifest_path: Path,
    data_dir: Path,
) -> tuple[list[Path], list[Path], list[str]]:
    """Resolve a strict train/validation manifest without discovering test NWBs.

    This is the isolation path for validation-only pilots: only the 27 train
    and 6 validation files listed in the manifest are returned as paths.  Test
    session names remain receipt strings and are never resolved/opened here.
    """
    try:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read frozen session manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Frozen session manifest must be a schema_version=1 object")
    splits = payload.get("session_splits")
    if not isinstance(splits, dict):
        raise ValueError("Frozen session manifest must contain session_splits")
    expected = {"train": 27, "val": 6, "test": 6}
    names: dict[str, list[str]] = {}
    for split, count in expected.items():
        rows = splits.get(split)
        if not isinstance(rows, list) or len(rows) != count or not all(isinstance(row, str) for row in rows):
            raise ValueError(f"Frozen manifest session_splits.{split} must contain {count} names")
        if len(set(rows)) != count:
            raise ValueError(f"Frozen manifest session_splits.{split} contains duplicates")
        names[split] = rows
    if len(set(names["train"] + names["val"] + names["test"])) != 39:
        raise ValueError("Frozen manifest train/val/test names must be disjoint")
    resolved: dict[str, list[Path]] = {}
    for split in ("train", "val"):
        paths = []
        for name in names[split]:
            path = (data_dir / f"{name}_behavior+ecephys.nwb").resolve()
            if path.parent != data_dir.resolve() or not path.is_file():
                raise FileNotFoundError(f"Frozen manifest {split} session is missing: {path}")
            paths.append(path)
        resolved[split] = paths
    return resolved["train"], resolved["val"], names["test"]


def load_dandi688_session(
    nwb_path: Path,
    *,
    bin_size_ms: int,
    window_size: int,
    calibration_n_trials: int,
    max_trial_length: int,
    pad_value: float,
    interpolate_trials: bool,
    behavior_mean: np.ndarray,
    behavior_std: np.ndarray,
    trial_result_filter: str = "R",
    exclude_calibration_trials_from_windows: bool = False,
    cache_dir: Path | None = None,
    signal_view: str = "sua",
) -> SessionRecord:
    """Load one session, reusing a versioned preprocessed cache when configured."""
    _validate_signal_view(signal_view)
    if cache_dir is None:
        return _load_dandi688_session_uncached(
            nwb_path,
            bin_size_ms=bin_size_ms,
            window_size=window_size,
            calibration_n_trials=calibration_n_trials,
            max_trial_length=max_trial_length,
            pad_value=pad_value,
            interpolate_trials=interpolate_trials,
            behavior_mean=behavior_mean,
            behavior_std=behavior_std,
            trial_result_filter=trial_result_filter,
            exclude_calibration_trials_from_windows=exclude_calibration_trials_from_windows,
            signal_view=signal_view,
        )

    cache_path = _session_cache_path(
        cache_dir,
        nwb_path,
        bin_size_ms=bin_size_ms,
        window_size=window_size,
        calibration_n_trials=calibration_n_trials,
        max_trial_length=max_trial_length,
        pad_value=pad_value,
        interpolate_trials=interpolate_trials,
        behavior_mean=behavior_mean,
        behavior_std=behavior_std,
        trial_result_filter=trial_result_filter,
        exclude_calibration_trials_from_windows=exclude_calibration_trials_from_windows,
        signal_view=signal_view,
    )
    with _exclusive_cache_lock(cache_path):
        if cache_path.is_file():
            try:
                record = _load_cached_session(cache_path)
                logger.info("Loaded cached session %s from %s", record.name, cache_path)
                return record
            except (KeyError, OSError, ValueError) as exc:
                logger.warning("Discarding unreadable session cache %s: %s", cache_path, exc)
                cache_path.unlink(missing_ok=True)
        record = _load_dandi688_session_uncached(
            nwb_path,
            bin_size_ms=bin_size_ms,
            window_size=window_size,
            calibration_n_trials=calibration_n_trials,
            max_trial_length=max_trial_length,
            pad_value=pad_value,
            interpolate_trials=interpolate_trials,
            behavior_mean=behavior_mean,
            behavior_std=behavior_std,
            trial_result_filter=trial_result_filter,
            exclude_calibration_trials_from_windows=exclude_calibration_trials_from_windows,
            signal_view=signal_view,
        )
        _write_npz_atomically(
            cache_path,
            name=np.asarray(record.name),
            neural=record.neural,
            behavior=record.behavior,
            calib_trials=record.calib_trials,
            valid_starts=record.valid_starts,
            source_unit_count=np.asarray(record.source_unit_count, dtype=np.int64),
            channel_ids=record.channel_ids,
            signal_view=np.asarray(record.signal_view),
        )
        logger.info("Cached session %s at %s", record.name, cache_path)
        return record


def _load_dandi688_session_uncached(
    nwb_path: Path,
    *,
    bin_size_ms: int,
    window_size: int,
    calibration_n_trials: int,
    max_trial_length: int,
    pad_value: float,
    interpolate_trials: bool,
    behavior_mean: np.ndarray,
    behavior_std: np.ndarray,
    trial_result_filter: str,
    exclude_calibration_trials_from_windows: bool,
    signal_view: str,
) -> SessionRecord:
    bin_size_s = bin_size_ms / 1000.0
    session_name = session_name_from_path(nwb_path)

    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        units_df = nwb.units.to_dataframe()
        source_unit_count = len(units_df)

        all_spikes = np.concatenate(units_df["spike_times"].values)
        t_min = float(all_spikes.min())
        t_max = float(all_spikes.max())
        bin_edges = np.arange(t_min, t_max + bin_size_s, bin_size_s)
        num_bins = len(bin_edges) - 1

        binned_spikes = np.zeros((num_bins, source_unit_count), dtype=np.float32)
        for i, (_, unit) in enumerate(units_df.iterrows()):
            binned_spikes[:, i] = bin_spikes(unit["spike_times"], bin_edges)

        channel_ids = np.arange(source_unit_count, dtype=np.int64)
        if signal_view == "pseudo_mua":
            binned_spikes, channel_ids = pool_spikes_by_electrode(
                binned_spikes,
                electrode_ids_from_units(units_df),
            )
        n_units = binned_spikes.shape[1]

        vel_series = nwb.processing["behavior"]["Velocity"].time_series["cursor_vel"]
        cursor_vel = vel_series.data[:]
        vel_times = vel_series.timestamps[:]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        binned_vel = np.zeros((num_bins, cursor_vel.shape[1]), dtype=np.float32)
        for c in range(cursor_vel.shape[1]):
            fn = interp1d(
                vel_times,
                cursor_vel[:, c],
                kind="linear",
                bounds_error=False,
                fill_value=0.0,
            )
            binned_vel[:, c] = fn(bin_centers)

        binned_vel = (binned_vel - behavior_mean) / behavior_std

        trials_df = nwb.intervals["trials"].to_dataframe()
        trial_info = []
        for _, trial in trials_df.iterrows():
            if trial["result"] != trial_result_filter:
                continue
            start_bin = int(np.searchsorted(bin_edges, trial["start_time"]))
            stop_bin = int(np.searchsorted(bin_edges, trial["stop_time"]))
            start_bin = max(0, start_bin)
            stop_bin = min(num_bins, stop_bin)
            if stop_bin - start_bin >= window_size:
                trial_info.append({"start": start_bin, "stop": stop_bin})

        calib_trials = _build_calib_trials(
            binned_spikes,
            trial_info,
            calibration_n_trials,
            max_trial_length,
            n_units,
            pad_value,
            interpolate_trials,
        )
        window_trials = (
            trial_info[calibration_n_trials:]
            if exclude_calibration_trials_from_windows
            else trial_info
        )
        valid_starts = _compute_valid_starts(window_trials, window_size)
        if exclude_calibration_trials_from_windows and not valid_starts.size:
            raise ValueError(
                f"{session_name}: no usable windows remain after reserving the first "
                f"{calibration_n_trials} rewarded trials for calibration"
            )

    logger.info(
        "%s: signal_view=%s source_units=%d channels=%d trials=%d bins=%d windows=%d",
        session_name,
        signal_view,
        source_unit_count,
        n_units,
        len(trial_info),
        num_bins,
        len(valid_starts),
    )
    return SessionRecord(
        name=session_name,
        neural=binned_spikes,
        behavior=binned_vel,
        calib_trials=calib_trials,
        valid_starts=valid_starts,
        source_unit_count=source_unit_count,
        channel_ids=channel_ids,
        signal_view=signal_view,
    )


def _build_calib_trials(
    binned_spikes: np.ndarray,
    trials: list[dict],
    m_trials: int,
    t_bins: int,
    n_units: int,
    pad_value: float,
    interpolate_trials: bool,
) -> np.ndarray:
    calib = np.full((m_trials, t_bins, n_units), pad_value, dtype=np.float32)
    selected = trials[:m_trials]
    for i, trial in enumerate(selected):
        start, stop = trial["start"], trial["stop"]
        trial_data = binned_spikes[start:stop]
        trial_len = len(trial_data)
        if interpolate_trials and trial_len != t_bins:
            x_orig = np.linspace(0, 1, trial_len)
            x_new = np.linspace(0, 1, t_bins)
            for n in range(n_units):
                fn = interp1d(
                    x_orig,
                    trial_data[:, n],
                    kind="cubic",
                    bounds_error=False,
                    fill_value=pad_value,
                )
                calib[i, :, n] = fn(x_new)
        else:
            length = min(trial_len, t_bins)
            calib[i, :length] = trial_data[:length]
    return calib


def _compute_valid_starts(trials: list[dict], window_size: int) -> np.ndarray:
    starts = []
    for trial in trials:
        for s in range(trial["start"], trial["stop"] - window_size + 1):
            starts.append(s)
    return np.array(starts, dtype=np.int64)


def list_datamodule_rewarded_trials(
    nwb_path: Path,
    *,
    bin_size_ms: int,
    window_size: int,
    trial_result_filter: str = "R",
) -> list[dict[str, float]]:
    """Chronological rewarded trials with the same duration filter as session loading.

    Each entry also carries ``target_dir`` (radians, center-out task target angle) when the
    NWB trials table has that column and the row's value is finite, else ``None``. This is
    read from the exact same ``trials_df`` row already visited by the filter loop below, so
    it does not change which trials are selected or their order -- callers that only need
    the original ``start_time``/``stop_time``/``start``/``stop`` fields (e.g.
    ``calibration_pool_end_time``) are unaffected. Added for E3 directional tuning side
    features (E3_E4_ENCODER_PROGRAM.md section 1), which need each pool trial's target
    direction and must reuse this filter rather than reimplementing it.
    """
    bin_size_s = bin_size_ms / 1000.0
    with NWBHDF5IO(str(nwb_path), "r") as io:
        nwb = io.read()
        if nwb.units is None:
            raise ValueError(f"NWB file has no units table: {nwb_path}")
        units_df = nwb.units.to_dataframe()
        all_spikes = np.concatenate(units_df["spike_times"].values)
        t_min = float(all_spikes.min())
        t_max = float(all_spikes.max())
        bin_edges = np.arange(t_min, t_max + bin_size_s, bin_size_s)
        num_bins = len(bin_edges) - 1

        trials_df = nwb.intervals["trials"].to_dataframe()
        trial_info: list[dict[str, float]] = []
        for _, trial in trials_df.iterrows():
            if trial["result"] != trial_result_filter:
                continue
            start_bin = int(np.searchsorted(bin_edges, trial["start_time"]))
            stop_bin = int(np.searchsorted(bin_edges, trial["stop_time"]))
            start_bin = max(0, start_bin)
            stop_bin = min(num_bins, stop_bin)
            if stop_bin - start_bin >= window_size:
                target_dir = trial.get("target_dir")
                if target_dir is None or not np.isfinite(target_dir):
                    target_dir = None
                else:
                    target_dir = float(target_dir)
                go_cue_time = trial.get("go_cue_time")
                if go_cue_time is None or not np.isfinite(go_cue_time):
                    go_cue_time = None
                else:
                    go_cue_time = float(go_cue_time)
                target_on_time = trial.get("target_on_time")
                if target_on_time is None or not np.isfinite(target_on_time):
                    target_on_time = None
                else:
                    target_on_time = float(target_on_time)
                trial_info.append(
                    {
                        "start_time": float(trial["start_time"]),
                        "stop_time": float(trial["stop_time"]),
                        "start": float(start_bin),
                        "stop": float(stop_bin),
                        "target_dir": target_dir,
                        "go_cue_time": go_cue_time,
                        "target_on_time": target_on_time,
                    }
                )
    return trial_info


def calibration_pool_end_time(
    nwb_path: Path,
    *,
    pool_size: int,
    bin_size_ms: int,
    window_size: int,
    trial_result_filter: str = "R",
) -> float:
    """Stop time of the pool_size-th datamodule-aligned rewarded trial."""
    trials = list_datamodule_rewarded_trials(
        nwb_path,
        bin_size_ms=bin_size_ms,
        window_size=window_size,
        trial_result_filter=trial_result_filter,
    )
    if len(trials) < pool_size:
        raise ValueError(
            f"{session_name_from_path(nwb_path)}: only {len(trials)} rewarded trials "
            f"pass the datamodule filter; pool_size={pool_size} required"
        )
    return float(trials[pool_size - 1]["stop_time"])


def fit_behavior_stats(
    train_files: Sequence[Path],
    bin_size_ms: int,
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute train-only behavior normalization, with an optional disk cache."""
    if cache_dir is None:
        return _fit_behavior_stats_uncached(train_files, bin_size_ms)

    cache_path = _behavior_stats_cache_path(cache_dir, train_files, bin_size_ms)
    with _exclusive_cache_lock(cache_path):
        if cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as cache:
                    mean = cache["mean"].astype(np.float32, copy=False)
                    std = cache["std"].astype(np.float32, copy=False)
                logger.info("Loaded cached train behavior statistics from %s", cache_path)
                return mean, std
            except (KeyError, OSError, ValueError) as exc:
                logger.warning("Discarding unreadable behavior-statistics cache %s: %s", cache_path, exc)
                cache_path.unlink(missing_ok=True)
        mean, std = _fit_behavior_stats_uncached(train_files, bin_size_ms)
        _write_npz_atomically(cache_path, mean=mean, std=std)
        logger.info("Cached train behavior statistics at %s", cache_path)
        return mean, std


def _fit_behavior_stats_uncached(
    train_files: Sequence[Path],
    bin_size_ms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-dimension mean/std of binned cursor_vel on train sessions only."""
    bin_size_s = bin_size_ms / 1000.0
    chunks: list[np.ndarray] = []
    for nwb_path in train_files:
        with NWBHDF5IO(str(nwb_path), "r") as io:
            nwb = io.read()
            vel_series = nwb.processing["behavior"]["Velocity"].time_series["cursor_vel"]
            cursor_vel = vel_series.data[:]
            vel_times = vel_series.timestamps[:]
            t_min = float(vel_times.min())
            t_max = float(vel_times.max())
            bin_edges = np.arange(t_min, t_max + bin_size_s, bin_size_s)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            num_bins = len(bin_centers)
            binned_vel = np.zeros((num_bins, cursor_vel.shape[1]), dtype=np.float32)
            for c in range(cursor_vel.shape[1]):
                fn = interp1d(
                    vel_times,
                    cursor_vel[:, c],
                    kind="linear",
                    bounds_error=False,
                    fill_value=0.0,
                )
                binned_vel[:, c] = fn(bin_centers)
            chunks.append(binned_vel)
    stacked = np.concatenate(chunks, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


class Dandi688MultiSessionDataset(Dataset):
    """Windowed multi-session dataset with per-session calibration tensors."""

    def __init__(
        self,
        sessions: dict[str, SessionRecord],
        window_size: int,
        random_calibration: bool = False,
        calibration_n_trials: int = 10,
    ):
        self.sessions = sessions
        self.window_size = window_size
        self.random_calibration = random_calibration
        self.calibration_n_trials = calibration_n_trials
        self.window_indices: list[tuple[str, int]] = []

        for session_name, record in sessions.items():
            for start in record.valid_starts:
                self.window_indices.append((session_name, int(start)))

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, idx: int):
        session_name, start = self.window_indices[idx]
        record = self.sessions[session_name]
        end = start + self.window_size
        neural = torch.from_numpy(record.neural[start:end]).float()
        behavior = torch.from_numpy(record.behavior[start:end]).float()
        calib = torch.from_numpy(record.calib_trials.copy()).float()
        if self.random_calibration:
            n_trials = calib.shape[0]
            if n_trials > self.calibration_n_trials:
                pick = np.random.choice(
                    n_trials, size=self.calibration_n_trials, replace=False
                )
                calib = calib[pick]
        if record.side_features is not None:
            side = torch.from_numpy(record.side_features).float()
            if record.electrode_ids is not None:
                electrode_ids = torch.from_numpy(record.electrode_ids).long()
                return neural, behavior, calib, session_name, side, electrode_ids
            return neural, behavior, calib, session_name, side
        return neural, behavior, calib, session_name


class SessionBatchSampler(Sampler):
    """Yield batches where every sample belongs to the same session."""

    def __init__(
        self,
        dataset: Dandi688MultiSessionDataset,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.session_to_indices: dict[str, list[int]] = {}
        for idx, (session_name, _) in enumerate(dataset.window_indices):
            self.session_to_indices.setdefault(session_name, []).append(idx)
        self.batched_indices = self._build_batches(self.seed)

    def _build_batches(self, seed: int) -> list[list[int]]:
        session_batches: dict[str, list[list[int]]] = {}
        for session_name, indices in self.session_to_indices.items():
            session_indices = list(indices)
            if self.shuffle:
                session_indices = random.Random(seed).sample(
                    session_indices, len(session_indices)
                )
            batches = []
            for i in range(0, len(session_indices), self.batch_size):
                batch = session_indices[i : i + self.batch_size]
                if len(batch) == self.batch_size:
                    batches.append(batch)
            session_batches[session_name] = batches
        batched = [batch for batches in session_batches.values() for batch in batches]
        if self.shuffle:
            batched = random.Random(seed).sample(batched, len(batched))
        return batched

    def __iter__(self):
        for batch_indices in self.batched_indices:
            yield batch_indices

    def __len__(self) -> int:
        return len(self.batched_indices)


class Dandi688MultiSessionDataModule(pl.LightningDataModule):
    """Lightning DataModule for DANDI 000688 multi-session SUA decoding."""

    def __init__(
        self,
        data_dir: str,
        task: str = "CO",
        split_counts: tuple[int, int, int] = (37, 8, 8),
        batch_size: int = 32,
        window_size: int = 50,
        calibration_n_trials: int = 10,
        max_trial_length: int = 100,
        bin_size_ms: int = 20,
        pad_value: float = -1.0,
        num_workers: int = 4,
        pin_memory: bool = True,
        random_calibration: bool = True,
        interpolate_trials: bool = True,
        trial_result_filter: str = "R",
        seed: int = 42,
        max_units_exclusive: int | None = None,
        cache_dir: str | None = None,
        signal_view: str = "sua",
        side_feature_group: str | None = None,
        side_feature_pool_size: int = 50,
        side_permutation_seed: int | None = None,
        train_val_manifest_path: str | None = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.task = task.upper()
        self.split_counts = split_counts
        self.batch_size = batch_size
        self.window_size = window_size
        self.calibration_n_trials = calibration_n_trials
        self.max_trial_length = max_trial_length
        self.bin_size_ms = bin_size_ms
        self.pad_value = pad_value
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.random_calibration = random_calibration
        self.interpolate_trials = interpolate_trials
        self.trial_result_filter = trial_result_filter
        self.seed = seed
        _validate_max_units_exclusive(max_units_exclusive)
        self.max_units_exclusive = max_units_exclusive
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        _validate_signal_view(signal_view)
        self.signal_view = signal_view
        self.side_feature_group = side_feature_group
        self.side_feature_pool_size = side_feature_pool_size
        self.side_permutation_seed = side_permutation_seed
        self.train_val_manifest_path = (
            Path(train_val_manifest_path).expanduser().resolve()
            if train_val_manifest_path else None
        )
        # T4/T8 may be constructed at pseudo-MUA electrode resolution.  Waveform
        # features remain sorted-unit only and are rejected by unit_side_features
        # with a precise error if requested.
        if signal_view == "pseudo_mua" and side_feature_group not in {
            None,
            "t4",
            "t4r",
            "t4rq",
            "t8",
            "ts4",
            "ts8",
        }:
            raise ValueError(
                "pseudo_mua supports only T4/T8 side features (or none); "
                f"got {side_feature_group!r}"
            )

        self.train_dataset: Optional[Dandi688MultiSessionDataset] = None
        self.val_dataset: Optional[Dandi688MultiSessionDataset] = None
        self.test_dataset: Optional[Dandi688MultiSessionDataset] = None
        self.session_splits: dict[str, list[str]] = {}
        self.session_files: dict[str, list[Path]] = {}
        self.session_unit_counts: dict[str, int] = {}
        self.session_channel_counts: dict[str, int] = {}
        self._behavior_stats: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._side_feature_stats: Optional[tuple[np.ndarray, np.ndarray]] = None
        self._template_ridge_receipt: Optional[dict[str, object]] = None
        self._template_ridge_profile: Optional[np.ndarray] = None
        self._t4r_posterior_receipt: Optional[dict[str, object]] = None
        self._splits_initialized = False

    def setup(self, stage: Optional[str] = None):
        if stage not in (None, "fit", "validate", "test", "predict"):
            raise ValueError(f"Unsupported setup stage: {stage!r}")
        self._initialize_splits()
        needs_fit = stage in (None, "fit", "validate")
        needs_test = stage in (None, "test", "predict")

        if needs_fit or needs_test:
            behavior_mean, behavior_std = self._get_behavior_stats()
            side_stats = self._get_side_feature_stats()

        def load_split(
            files: Sequence[Path],
            *,
            exclude_calibration_trials_from_windows: bool,
        ) -> dict[str, SessionRecord]:
            sessions: dict[str, SessionRecord] = {}
            for nwb_path in files:
                record = load_dandi688_session(
                    nwb_path,
                    bin_size_ms=self.bin_size_ms,
                    window_size=self.window_size,
                    calibration_n_trials=self.calibration_n_trials,
                    max_trial_length=self.max_trial_length,
                    pad_value=self.pad_value,
                    interpolate_trials=self.interpolate_trials,
                    behavior_mean=behavior_mean,
                    behavior_std=behavior_std,
                    trial_result_filter=self.trial_result_filter,
                    exclude_calibration_trials_from_windows=(
                        exclude_calibration_trials_from_windows
                    ),
                    cache_dir=self.cache_dir,
                    signal_view=self.signal_view,
                )
                if self.side_feature_group is not None:
                    from mc_maze.unit_side_features import (
                        base_feature_group,
                        confidence_component_shuffle,
                        is_electrode_shuffle_control,
                        is_feature_shuffle_control,
                        is_template_ridge_zero_control,
                        load_session_electrode_ids,
                        load_unit_side_features,
                        permute_electrode_ids,
                        permute_t4c_component,
                        uses_electrode_ids,
                        uses_electrode_relation_membership,
                    )

                    assert side_stats is not None
                    side_mean, side_std = side_stats
                    feature_shuffle_seed = (
                        self.side_permutation_seed
                        if is_feature_shuffle_control(self.side_feature_group)
                        else None
                    )
                    side_features, _ = load_unit_side_features(
                        nwb_path,
                        feature_group=base_feature_group(self.side_feature_group),
                        pool_size=self.side_feature_pool_size,
                        mean=side_mean,
                        std=side_std,
                        cache_dir=self.cache_dir,
                        permutation_seed=feature_shuffle_seed,
                        bin_size_ms=self.bin_size_ms,
                        window_size=self.window_size,
                        trial_result_filter=self.trial_result_filter,
                        signal_view=self.signal_view,
                        template_profile=self._template_ridge_profile,
                        posterior_prior=self._t4r_posterior_receipt,
                    )
                    if is_template_ridge_zero_control(self.side_feature_group):
                        side_features = np.zeros_like(side_features, dtype=np.float32)
                    component_shuffle = confidence_component_shuffle(self.side_feature_group)
                    if component_shuffle is not None:
                        if self.side_permutation_seed is None:
                            raise ValueError(
                                f"{self.side_feature_group} requires a non-None side_permutation_seed"
                            )
                        side_features = permute_t4c_component(
                            side_features,
                            component=component_shuffle,
                            permutation_seed=self.side_permutation_seed,
                        )
                    if side_features.shape[0] != record.neural.shape[1]:
                        raise ValueError(
                            f"{record.name}: side_features units {side_features.shape[0]} "
                            f"!= neural channels {record.neural.shape[1]}"
                        )
                    electrode_ids = None
                    if uses_electrode_relation_membership(self.side_feature_group):
                        # Relation tokens use equality only, never an absolute-ID
                        # lookup.  pseudo-MUA is already one channel per physical
                        # electrode, so the record's deterministic channel IDs
                        # give the required singleton boundary without reopening
                        # a sorted-unit membership table.
                        electrode_ids = (
                            record.channel_ids.astype(np.int64, copy=False)
                            if self.signal_view == "pseudo_mua"
                            else load_session_electrode_ids(nwb_path)
                        )
                        if electrode_ids.shape[0] != record.neural.shape[1]:
                            raise ValueError(
                                f"{record.name}: relation membership units {electrode_ids.shape[0]} "
                                f"!= neural channels {record.neural.shape[1]}"
                            )
                        if is_electrode_shuffle_control(self.side_feature_group):
                            if self.side_permutation_seed is None:
                                raise ValueError(
                                    "t4rel_membership_shuffled requires a non-None "
                                    "side_permutation_seed"
                                )
                            from mc_maze.sua_auxiliary_stage0 import (
                                deterministic_membership_shuffle,
                            )

                            electrode_ids = deterministic_membership_shuffle(
                                electrode_ids, seed=self.side_permutation_seed
                            )
                    elif uses_electrode_ids(self.side_feature_group):
                        electrode_ids = load_session_electrode_ids(nwb_path)
                        if electrode_ids.shape[0] != record.neural.shape[1]:
                            raise ValueError(
                                f"{record.name}: electrode_ids units {electrode_ids.shape[0]} "
                                f"!= neural channels {record.neural.shape[1]}"
                            )
                        if is_electrode_shuffle_control(self.side_feature_group):
                            assert self.side_permutation_seed is not None
                            electrode_ids = permute_electrode_ids(
                                electrode_ids,
                                permutation_seed=self.side_permutation_seed,
                            )
                    record = SessionRecord(
                        name=record.name,
                        neural=record.neural,
                        behavior=record.behavior,
                        calib_trials=record.calib_trials,
                        valid_starts=record.valid_starts,
                        source_unit_count=record.source_unit_count,
                        channel_ids=record.channel_ids,
                        signal_view=record.signal_view,
                        side_features=side_features,
                        electrode_ids=electrode_ids,
                    )
                sessions[record.name] = record
                self.session_channel_counts[record.name] = record.neural.shape[1]
            return sessions

        if needs_fit and self.train_dataset is None:
            train_sessions = load_split(
                self.session_files["train"],
                exclude_calibration_trials_from_windows=False,
            )
            self.train_dataset = Dandi688MultiSessionDataset(
                train_sessions,
                window_size=self.window_size,
                random_calibration=self.random_calibration,
                calibration_n_trials=self.calibration_n_trials,
            )
        if needs_fit and self.val_dataset is None:
            val_sessions = load_split(
                self.session_files["val"],
                exclude_calibration_trials_from_windows=True,
            )
            self.val_dataset = Dandi688MultiSessionDataset(
                val_sessions,
                window_size=self.window_size,
                random_calibration=False,
                calibration_n_trials=self.calibration_n_trials,
            )
        if needs_test and self.test_dataset is None:
            test_sessions = load_split(
                self.session_files["test"],
                exclude_calibration_trials_from_windows=True,
            )
            self.test_dataset = Dandi688MultiSessionDataset(
                test_sessions,
                window_size=self.window_size,
                random_calibration=False,
                calibration_n_trials=self.calibration_n_trials,
            )

    def _initialize_splits(self) -> None:
        if self._splits_initialized:
            return
        if self.train_val_manifest_path is not None:
            train_files, val_files, test_names = load_frozen_train_val_manifest(
                self.train_val_manifest_path, self.data_dir
            )
            if self.split_counts != (27, 6, 6) or self.max_units_exclusive != 100:
                raise ValueError(
                    "Frozen train/validation manifest is only authorized for split_counts=(27,6,6), "
                    "max_units_exclusive=100"
                )
            test_files: list[Path] = []
            counted_files = train_files + val_files
        else:
            all_files = discover_nwb_files(
                self.data_dir,
                task=self.task,
                max_units_exclusive=self.max_units_exclusive,
            )
            train_files, val_files, test_files = chronological_session_split(
                all_files,
                self.split_counts,
                max_units_exclusive=self.max_units_exclusive,
            )
            test_names = [session_name_from_path(p) for p in test_files]
            counted_files = all_files
        self.session_files = {
            "train": train_files,
            "val": val_files,
            "test": test_files,
        }
        self.session_splits = {
            "train": [session_name_from_path(p) for p in train_files],
            "val": [session_name_from_path(p) for p in val_files],
            "test": test_names,
        }
        self.session_unit_counts = {
            session_name_from_path(nwb_path): nwb_unit_count(nwb_path)
            for nwb_path in counted_files
        }
        self._splits_initialized = True
        logger.info("Session split: %s", self.session_splits)

    def _get_behavior_stats(self) -> tuple[np.ndarray, np.ndarray]:
        if self._behavior_stats is None:
            self._behavior_stats = fit_behavior_stats(
                self.session_files["train"], self.bin_size_ms, cache_dir=self.cache_dir
            )
        return self._behavior_stats

    def _get_side_feature_stats(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self.side_feature_group is None:
            return None
        if self._side_feature_stats is None:
            from mc_maze.unit_side_features import (
                TEMPLATE_RIDGE_FEATURE_NAMES,
                base_feature_group,
                fit_side_feature_stats,
            )

            feature_group = base_feature_group(self.side_feature_group)
            if feature_group in {"t4r", "t4rq"}:
                side_mean, side_std, receipt = fit_side_feature_stats(
                    self.session_files["train"],
                    feature_group=feature_group,
                    pool_size=self.side_feature_pool_size,
                    cache_dir=self.cache_dir,
                    bin_size_ms=self.bin_size_ms,
                    window_size=self.window_size,
                    trial_result_filter=self.trial_result_filter,
                    signal_view=self.signal_view,
                    return_t4r_receipt=True,
                )
                self._side_feature_stats = (side_mean, side_std)
                self._t4r_posterior_receipt = receipt
            elif feature_group in TEMPLATE_RIDGE_FEATURE_NAMES:
                side_mean, side_std, receipt = fit_side_feature_stats(
                    self.session_files["train"],
                    feature_group=feature_group,
                    pool_size=self.side_feature_pool_size,
                    cache_dir=self.cache_dir,
                    bin_size_ms=self.bin_size_ms,
                    window_size=self.window_size,
                    trial_result_filter=self.trial_result_filter,
                    signal_view=self.signal_view,
                    return_template_receipt=True,
                )
                self._side_feature_stats = (side_mean, side_std)
                self._template_ridge_receipt = receipt
                self._template_ridge_profile = np.asarray(receipt["profile"], dtype=np.float32)
            else:
                self._side_feature_stats = fit_side_feature_stats(
                    self.session_files["train"],
                    feature_group=feature_group,
                    pool_size=self.side_feature_pool_size,
                    cache_dir=self.cache_dir,
                    bin_size_ms=self.bin_size_ms,
                    window_size=self.window_size,
                    trial_result_filter=self.trial_result_filter,
                    signal_view=self.signal_view,
                )
        return self._side_feature_stats

    def _dataloader(self, dataset: Dandi688MultiSessionDataset, shuffle: bool) -> DataLoader:
        sampler = SessionBatchSampler(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            seed=self.seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("Train dataset is not initialized; call setup('fit') first")
        return self._dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset is not initialized; call setup('fit') first")
        dl = self._dataloader(self.val_dataset, shuffle=False)
        return [dl, dl]

    def test_dataloader(self):
        if self.test_dataset is None:
            raise RuntimeError("Test dataset is not initialized; call setup('test') first")
        # The streaming calibration module exposes separate test_heldin and
        # test_heldout metric namespaces. The DANDI test split is the official
        # held-out set, so feed the same frozen loader through both indices.
        dl = self._dataloader(self.test_dataset, shuffle=False)
        return [dl, dl]


# Backward-compatible alias requested in ROADMAP
MultiSessionDataModule = Dandi688MultiSessionDataModule
