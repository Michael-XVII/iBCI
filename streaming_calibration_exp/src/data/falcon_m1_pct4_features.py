"""M1-only phase-conditioned tuning carrier (PCT4).

PCT4 is a calibration-only analytic side feature for native FALCON M1.  It uses
M1 trial events and target directions from the calibration NWB, plus the same
EMG timestamp grid that the official loader uses to bin spikes.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from pynwb import NWBHDF5IO


PCT4_DIM = 4
PCT4_FEATURE_NAMES = (
    "reach_cos_phi",
    "reach_sin_phi",
    "post_contact_cos_phi",
    "post_contact_sin_phi",
)
PCT4_ESTIMATOR_VERSION = "m1-pct4-v1-event-aligned-bin-end"


@dataclass(frozen=True)
class M1PCT4Metadata:
    nwb_path: str
    target_angles: np.ndarray
    move_onset_times: np.ndarray
    contact_times: np.ndarray
    stop_times: np.ndarray
    bin_timestamps: np.ndarray


def _task_name(task: object) -> str:
    return str(getattr(task, "name", getattr(task, "value", task))).split(".")[-1].lower()


def _require_m1(task: object) -> None:
    if _task_name(task) != "m1":
        raise ValueError(f"PCT4-v1 supports native FALCON M1 only, got {task!r}")


def calibration_m1_pct4_metadata(nwb_path: Path, task: object = "m1") -> M1PCT4Metadata:
    """Read event times, target angles, and official M1 bin timestamps."""
    _require_m1(task)
    with NWBHDF5IO(str(nwb_path), "r", load_namespaces=True) as io:
        nwbfile = io.read()
        trials = nwbfile.trials
        if trials is None:
            raise ValueError(f"Calibration file has no trials table: {nwb_path}")
        frame = trials.to_dataframe()
        required = {"move_onset_time", "contact_time", "stop_time", "tgt_loc"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"M1 PCT4 missing trial columns {missing}: {nwb_path}")
        if "preprocessed_emg" not in nwbfile.acquisition:
            raise ValueError(f"M1 PCT4 needs preprocessed_emg timestamps: {nwb_path}")
        raw_emg = nwbfile.acquisition["preprocessed_emg"]
        muscles = list(raw_emg.time_series)
        if not muscles:
            raise ValueError(f"M1 PCT4 found no EMG time series: {nwb_path}")
        bin_timestamps = np.asarray(
            raw_emg.get_timeseries(muscles[0]).timestamps[:], dtype=np.float64
        )

    target_angles = np.deg2rad(np.asarray(frame["tgt_loc"], dtype=np.float64))
    metadata = M1PCT4Metadata(
        nwb_path=str(nwb_path),
        target_angles=target_angles.astype(np.float32),
        move_onset_times=np.asarray(frame["move_onset_time"], dtype=np.float64),
        contact_times=np.asarray(frame["contact_time"], dtype=np.float64),
        stop_times=np.asarray(frame["stop_time"], dtype=np.float64),
        bin_timestamps=bin_timestamps,
    )
    validate_m1_pct4_metadata(metadata, source=str(nwb_path))
    return metadata


def validate_m1_pct4_metadata(metadata: M1PCT4Metadata, *, source: str) -> None:
    """Fail closed on malformed M1 event metadata."""
    n_trials = metadata.target_angles.shape[0]
    arrays = (
        metadata.move_onset_times,
        metadata.contact_times,
        metadata.stop_times,
    )
    if any(arr.shape != (n_trials,) for arr in arrays):
        raise ValueError(f"M1 PCT4 event/angle shape mismatch for {source}")
    if n_trials < 3:
        raise ValueError(f"M1 PCT4 needs at least 3 trials for {source}")
    if not np.all(np.isfinite(metadata.target_angles)):
        raise ValueError(f"M1 PCT4 target angles contain NaN/Inf for {source}")
    if not all(np.all(np.isfinite(arr)) for arr in arrays):
        raise ValueError(f"M1 PCT4 event times contain NaN/Inf for {source}")
    if np.any(metadata.contact_times <= metadata.move_onset_times):
        raise ValueError(f"M1 PCT4 has non-positive reach phase for {source}")
    if np.any(metadata.stop_times <= metadata.contact_times):
        raise ValueError(f"M1 PCT4 has non-positive post-contact phase for {source}")
    if metadata.bin_timestamps.ndim != 1 or metadata.bin_timestamps.size == 0:
        raise ValueError(f"M1 PCT4 bin timestamps must be a non-empty vector for {source}")
    if np.any(np.diff(metadata.bin_timestamps) <= 0):
        raise ValueError(f"M1 PCT4 bin timestamps must be strictly increasing for {source}")


def phase_window_trial_sums(
    neural: np.ndarray,
    eval_mask: np.ndarray,
    bin_timestamps: np.ndarray,
    phase_start_times: np.ndarray,
    phase_stop_times: np.ndarray,
    *,
    source: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Sum raw binned spikes inside event windows using retained eval bins only."""
    x = np.asarray(neural, dtype=np.float64)
    mask = np.asarray(eval_mask, dtype=bool).reshape(-1)
    timestamps = np.asarray(bin_timestamps, dtype=np.float64).reshape(-1)
    starts = np.asarray(phase_start_times, dtype=np.float64).reshape(-1)
    stops = np.asarray(phase_stop_times, dtype=np.float64).reshape(-1)
    if x.ndim != 2:
        raise ValueError(f"M1 PCT4 neural must be [T,N] for {source}, got {x.shape}")
    if x.shape[0] != mask.size or x.shape[0] != timestamps.size:
        raise ValueError(
            f"M1 PCT4 raw shape mismatch for {source}: neural={x.shape}, "
            f"eval_mask={mask.shape}, timestamps={timestamps.shape}"
        )
    if starts.shape != stops.shape:
        raise ValueError(f"M1 PCT4 phase start/stop shape mismatch for {source}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"M1 PCT4 neural contains NaN/Inf for {source}")

    sums = np.zeros((starts.size, x.shape[1]), dtype=np.float64)
    lengths = np.zeros(starts.size, dtype=np.int64)
    for trial_idx, (start, stop) in enumerate(zip(starts, stops)):
        lo = int(np.searchsorted(timestamps, start, side="left"))
        hi = int(np.searchsorted(timestamps, stop, side="left"))
        if hi <= lo:
            raise ValueError(f"M1 PCT4 empty timestamp window trial {trial_idx} for {source}")
        keep = mask[lo:hi]
        length = int(keep.sum())
        if length <= 0:
            raise ValueError(f"M1 PCT4 no retained bins trial {trial_idx} for {source}")
        sums[trial_idx] = x[lo:hi][keep].sum(axis=0, dtype=np.float64)
        lengths[trial_idx] = length
    return sums, lengths


def _direction_design(angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(angles, dtype=np.float64).reshape(-1)
    usable = np.isfinite(values)
    theta = values[usable]
    design = np.stack([np.ones(theta.shape[0]), np.cos(theta), np.sin(theta)], axis=1)
    return design, usable


def _signed_direction_coefficients(
    sums: np.ndarray,
    lengths: np.ndarray,
    target_angles: np.ndarray,
    *,
    source: str,
) -> np.ndarray:
    values = np.asarray(sums, dtype=np.float64)
    trial_lengths = np.asarray(lengths, dtype=np.float64).reshape(-1)
    angles = np.asarray(target_angles, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[0] != trial_lengths.size or values.shape[0] != angles.size:
        raise ValueError(
            f"M1 PCT4 shape mismatch for {source}: sums={values.shape}, "
            f"lengths={trial_lengths.shape}, angles={angles.shape}"
        )
    if np.any(trial_lengths <= 0):
        raise ValueError(f"M1 PCT4 non-positive trial lengths for {source}")
    design, usable = _direction_design(angles)
    if design.shape[0] < 3:
        raise ValueError(f"M1 PCT4 needs >=3 directional trials for {source}")
    rank = int(np.linalg.matrix_rank(design))
    if rank != 3:
        raise ValueError(f"M1 PCT4 direction design is rank {rank}, not 3, for {source}")
    rates = values[usable] / trial_lengths[usable, None]
    coefficients, _, fitted_rank, _ = np.linalg.lstsq(design, rates, rcond=None)
    if int(fitted_rank) != 3:
        raise ValueError(f"M1 PCT4 least-squares rank {fitted_rank}, not 3, for {source}")
    _, a, c = coefficients
    signed = np.stack([a, c], axis=-1)
    if not np.all(np.isfinite(signed)):
        raise ValueError(f"M1 PCT4 coefficients contain NaN/Inf for {source}")
    return signed


def deterministic_label_permutation(num_trials: int, *, session_name: str, seed: int) -> np.ndarray:
    """Stable non-identity trial-label permutation for PCT4-LS."""
    if num_trials < 2:
        raise ValueError("PCT4-LS requires at least two trials")
    digest = hashlib.sha256(f"m1-pct4-ls-v1:{seed}:{session_name}".encode()).digest()
    generator = np.random.RandomState(int.from_bytes(digest[:4], "little"))
    permutation = generator.permutation(num_trials)
    if np.array_equal(permutation, np.arange(num_trials)):
        permutation = np.roll(permutation, 1)
    return permutation.astype(np.int64, copy=False)


def deterministic_pct4_row_permutation(
    num_channels: int, *, session_name: str, seed: int
) -> np.ndarray:
    """Stable non-identity channel-row permutation for PCT4-RS."""
    if num_channels < 2:
        raise ValueError("PCT4-RS requires at least two native-MUA channels")
    digest = hashlib.sha256(f"m1-pct4-rs-v1:{seed}:{session_name}".encode()).digest()
    generator = np.random.RandomState(int.from_bytes(digest[:4], "little"))
    permutation = generator.permutation(num_channels)
    if np.array_equal(permutation, np.arange(num_channels)):
        permutation = np.roll(permutation, 1)
    return permutation.astype(np.int64, copy=False)


def pct4_from_phase_sums(
    reach_sums: np.ndarray,
    reach_lengths: np.ndarray,
    post_sums: np.ndarray,
    post_lengths: np.ndarray,
    target_angles: np.ndarray,
    *,
    source: str,
    label_shuffle_seed: int | None = None,
    session_name: str | None = None,
) -> np.ndarray:
    """Fit PCT4 from precomputed reach/post phase sums."""
    angles = np.asarray(target_angles, dtype=np.float64).reshape(-1)
    if label_shuffle_seed is not None:
        if session_name is None:
            raise ValueError("PCT4-LS requires session_name for deterministic permutation")
        perm = deterministic_label_permutation(
            angles.size, session_name=session_name, seed=int(label_shuffle_seed)
        )
        angles = angles[perm]
    reach = _signed_direction_coefficients(
        reach_sums, reach_lengths, angles, source=f"{source}:reach"
    )
    post = _signed_direction_coefficients(
        post_sums, post_lengths, angles, source=f"{source}:post"
    )
    features = np.concatenate([reach, post], axis=-1).astype(np.float32)
    if features.shape[-1] != PCT4_DIM:
        raise RuntimeError(f"M1 PCT4 internal dimension error for {source}: {features.shape}")
    return features


def fit_train_pct4_stats(
    session_reach_sums: Mapping[str, np.ndarray],
    session_reach_lengths: Mapping[str, np.ndarray],
    session_post_sums: Mapping[str, np.ndarray],
    session_post_lengths: Mapping[str, np.ndarray],
    session_target_angles: Mapping[str, np.ndarray],
    session_names: Sequence[str],
    calibration_n_trials: int,
    *,
    feature_group: str = "pct4",
    label_shuffle_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit z-score statistics from source sessions only."""
    if calibration_n_trials < 1:
        raise ValueError("M1 PCT4 requires calibration_n_trials >= 1")
    group = str(feature_group).lower()
    if group == "pct4_z4":
        return np.zeros(PCT4_DIM, dtype=np.float32), np.ones(PCT4_DIM, dtype=np.float32)
    chunks: list[np.ndarray] = []
    for name in session_names:
        max_start = session_reach_sums[name].shape[0] - calibration_n_trials
        if max_start < 0:
            raise ValueError(f"Session {name} has fewer trials than calibration_n_trials")
        raw = pct4_from_phase_sums(
            session_reach_sums[name][:calibration_n_trials],
            session_reach_lengths[name][:calibration_n_trials],
            session_post_sums[name][:calibration_n_trials],
            session_post_lengths[name][:calibration_n_trials],
            session_target_angles[name][:calibration_n_trials],
            source=f"{name}[0:{calibration_n_trials}]",
            label_shuffle_seed=label_shuffle_seed if group == "pct4_ls" else None,
            session_name=name,
        )
        chunks.append(raw)
    if not chunks:
        raise ValueError("No train calibration windows were available for M1 PCT4 stats")
    values = np.concatenate(chunks, axis=0)
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    std[std <= 1.0e-6] = 1.0
    return mean, std
