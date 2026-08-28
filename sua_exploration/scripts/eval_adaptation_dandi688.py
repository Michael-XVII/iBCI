"""P3 Step 2: frozen gradient-free streaming calibration on DANDI 000688.

Loads a Step-1 cross-session checkpoint (shared decoder + NeuronID encoder
trained end-to-end on train sessions). The deployment protocol for every
held-out test session is strictly gradient-free: calibration spikes are read
in a forward pass to compute identity/statistics, then both encoder and
decoder remain frozen while decoding.

The machine-readable ``gradient_free_calibrated`` configuration is the deployment result.
Optional encoder-only and encoder+decoder finetuning configurations are
explicit diagnostic-oracle baselines: they use held-out behavior labels and
backward gradients, so they are not adaptation strategies or Step 2 targets.

Leakage control (strict):
  - calibration always uses trials[0:calib_n].
  - default deployment evaluation uses the disjoint trials[calib_n:].
  - deployment reads no held-out behavior labels for any weight update.
  - optional diagnostic oracle finetuning uses trials[0:K], and evaluation is
    always on trials[max(K, calib_n):] to remain disjoint from all inputs.
  - behavior standardization reuses Step-1 train-session stats.

Outputs results/p3_step2_adaptation_<out_name>.json with per-session and mean
R^2 plus explicit protocol and configuration-role metadata.
"""
from __future__ import annotations

import argparse
import copy
import json
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pynwb import NWBHDF5IO
from scipy.interpolate import interp1d
from torch.utils.data import DataLoader
from torchmetrics.regression import R2Score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mc_maze.datamodule import MCMazeSessionDataset, bin_spikes
from mc_maze.multisession_datamodule import (
    _cache_key,
    _build_calib_trials,
    _compute_valid_starts,
    _exclusive_cache_lock,
    _source_fingerprint,
    _write_npz_atomically,
    chronological_session_split,
    electrode_ids_from_units,
    discover_nwb_files,
    fit_behavior_stats,
    nwb_unit_count,
    pool_spikes_by_electrode,
    session_name_from_path,
)
from dandi688_gradient_free_protocol import (
    assert_state_dict_unchanged,
    canonical_direction_key,
    complete_formal_receipt,
    create_formal_receipt,
    formal_receipt_path,
    select_calibration_trial_indices,
    sha256_file,
)

_sce_root = Path(__file__).resolve().parents[2] / "streaming_calibration_exp"
sys.path.insert(0, str(_sce_root))
from src.models.streaming_calibration_module import StreamingCalibrationLitModule

DEFAULT_TEACHER = (
    Path(__file__).resolve().parents[1]
    / "checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt"
)
BEHAVIOR_SCALING_FACTOR = 5.0
WINDOW_SIZE = 50
TRIAL_LENGTH = 100
CALIB_N = 10
ID_HIDDEN_DIM = 128
HIDDEN_DIM = 64
PAD_VALUE = -1.0


def checkpoint_architecture_kwargs(checkpoint: dict) -> dict:
    """Reconstruct optional student topology fields from checkpoint hparams.

    Validation and formal evaluation must instantiate the same module topology
    before strict state loading.  The defaults preserve every legacy coupled
    checkpoint that predates these fields.
    """
    hyper_parameters = checkpoint.get("hyper_parameters") or {}
    return {
        "fixed_slot_count": int(hyper_parameters.get("fixed_slot_count", 0)),
        "fixed_slot_dim": int(hyper_parameters.get("fixed_slot_dim", 32)),
        "fixed_slot_mode": str(hyper_parameters.get("fixed_slot_mode", "soft")),
        "fixed_slot_fusion": str(
            hyper_parameters.get("fixed_slot_fusion", "film")
        ),
        "fixed_slot_temperature": float(
            hyper_parameters.get("fixed_slot_temperature", 1.0)
        ),
        "decoder_mode": str(hyper_parameters.get("decoder_mode", "coupled")),
        "decoupled_key_mode": str(
            hyper_parameters.get("decoupled_key_mode", "e_t4")
        ),
        "decoupled_key_dim": int(
            hyper_parameters.get("decoupled_key_dim", 32)
        ),
        "decoupled_value_dim": int(
            hyper_parameters.get("decoupled_value_dim", 32)
        ),
        "decoupled_num_heads": int(
            hyper_parameters.get("decoupled_num_heads", 2)
        ),
        "decoupled_key_permutation_seed": hyper_parameters.get(
            "decoupled_key_permutation_seed"
        ),
        "side_dim": int(hyper_parameters.get("side_dim", 0)),
        "electrode_embed_dim": int(
            hyper_parameters.get("electrode_embed_dim", 0)
        ),
        "num_electrodes": int(hyper_parameters.get("num_electrodes", 0)),
        "analytic_residual_mode": str(
            hyper_parameters.get("analytic_residual_mode", "none")
        ),
        "analytic_ridge_lambda": float(
            hyper_parameters.get("analytic_ridge_lambda", 0.0)
        ),
        "analytic_gain": float(hyper_parameters.get("analytic_gain", 1.0)),
        "analytic_side_mean": hyper_parameters.get("analytic_side_mean"),
        "analytic_side_std": hyper_parameters.get("analytic_side_std"),
        "analytic_bin_size_ms": int(
            hyper_parameters.get("analytic_bin_size_ms", 20)
        ),
        "analytic_shuffle_seed": hyper_parameters.get("analytic_shuffle_seed"),
        "analytic_zero_residual_init": bool(
            hyper_parameters.get("analytic_zero_residual_init", False)
        ),
        "analytic_residual_frame": str(
            hyper_parameters.get("analytic_residual_frame", "direct")
        ),
        "analytic_local_frame_epsilon": float(
            hyper_parameters.get("analytic_local_frame_epsilon", 1.0e-6)
        ),
    }


def parse_split_counts(text: str) -> tuple[int, int, int]:
    raw_parts = text.split(",")
    if len(raw_parts) != 3:
        raise ValueError("split_counts must be three comma-separated non-negative integers")
    try:
        parts = [int(part.strip()) for part in raw_parts]
    except ValueError as exc:
        raise ValueError(
            "split_counts must be three comma-separated non-negative integers"
        ) from exc
    if any(part < 0 for part in parts):
        raise ValueError("split_counts must be three comma-separated non-negative integers")
    return parts[0], parts[1], parts[2]


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value)!r} to an evaluation-record cache")


def load_session_with_trials(
    nwb_path: Path,
    bin_size_ms: int,
    window_size: int,
    calib_n: int,
    max_trial_length: int,
    pad_value: float,
    behavior_mean: np.ndarray,
    behavior_std: np.ndarray,
    trial_result_filter: str = "R",
    cache_dir: Path | None = None,
    signal_view: str = "sua",
) -> dict:
    """Load an evaluation record, optionally from a versioned local cache."""
    if signal_view not in {"sua", "pseudo_mua"}:
        raise ValueError(f"Unsupported signal_view: {signal_view!r}")
    if cache_dir is None:
        return _load_session_with_trials_uncached(
            nwb_path,
            bin_size_ms,
            window_size,
            calib_n,
            max_trial_length,
            pad_value,
            behavior_mean,
            behavior_std,
            trial_result_filter,
            signal_view,
        )

    cache_payload = {
        "cache_schema": "evaluation_record_v1",
        "kind": "evaluation_record",
        "source": _source_fingerprint(nwb_path),
        "bin_size_ms": bin_size_ms,
        "window_size": window_size,
        "calib_n": calib_n,
        "max_trial_length": max_trial_length,
        "pad_value": pad_value,
        "behavior_mean": behavior_mean.astype(np.float32).tolist(),
        "behavior_std": behavior_std.astype(np.float32).tolist(),
        "trial_result_filter": trial_result_filter,
        "signal_view": signal_view,
    }
    cache_path = cache_dir / "evaluation_records" / f"{session_name_from_path(nwb_path)}_{_cache_key(cache_payload)[:20]}.npz"
    with _exclusive_cache_lock(cache_path):
        if cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as cache:
                    return {
                        "name": str(cache["name"].item()),
                        "n_units": int(cache["n_units"].item()),
                        "neural": cache["neural"].astype(np.float32, copy=False),
                        "behavior": cache["behavior"].astype(np.float32, copy=False),
                        "trials": json.loads(str(cache["trials_json"].item())),
                        "calib_trials": cache["calib_trials"].astype(np.float32, copy=False),
                        "source_unit_count": int(cache["source_unit_count"].item()),
                        "signal_view": str(cache["signal_view"].item()),
                    }
            except (KeyError, OSError, ValueError, json.JSONDecodeError):
                cache_path.unlink(missing_ok=True)
        record = _load_session_with_trials_uncached(
            nwb_path,
            bin_size_ms,
            window_size,
            calib_n,
            max_trial_length,
            pad_value,
            behavior_mean,
            behavior_std,
            trial_result_filter,
            signal_view,
        )
        _write_npz_atomically(
            cache_path,
            name=np.asarray(record["name"]),
            n_units=np.asarray(record["n_units"], dtype=np.int64),
            neural=record["neural"],
            behavior=record["behavior"],
            trials_json=np.asarray(json.dumps(record["trials"], sort_keys=True, default=_json_default)),
            calib_trials=record["calib_trials"],
            source_unit_count=np.asarray(record["source_unit_count"], dtype=np.int64),
            signal_view=np.asarray(record["signal_view"]),
        )
        return record


def _load_session_with_trials_uncached(
    nwb_path: Path,
    bin_size_ms: int,
    window_size: int,
    calib_n: int,
    max_trial_length: int,
    pad_value: float,
    behavior_mean: np.ndarray,
    behavior_std: np.ndarray,
    trial_result_filter: str,
    signal_view: str,
) -> dict:
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
        if signal_view == "pseudo_mua":
            binned_spikes, _ = pool_spikes_by_electrode(
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
            start_bin = max(0, int(np.searchsorted(bin_edges, trial["start_time"])))
            stop_bin = min(num_bins, int(np.searchsorted(bin_edges, trial["stop_time"])))
            if stop_bin - start_bin >= window_size:
                trial_info.append({
                    "start": start_bin,
                    "stop": stop_bin,
                    "trial_index": int(trial.name),
                    "target_dir": trial.get("target_dir"),
                    "target_id": trial.get("target_id"),
                })

    calib_trials = _build_calib_trials(
        binned_spikes,
        trial_info,
        calib_n,
        max_trial_length,
        n_units,
        pad_value,
        interpolate_trials=True,
    )
    return {
        "name": session_name,
        "n_units": n_units,
        "neural": binned_spikes,
        "behavior": binned_vel,
        "trials": trial_info,
        "calib_trials": calib_trials,
        "source_unit_count": source_unit_count,
        "signal_view": signal_view,
    }


def build_calib_trials_for_indices(rec: dict, indices: list[int], calib_n: int) -> np.ndarray:
    if len(indices) != calib_n:
        raise ValueError(f"Expected {calib_n} selected calibration trials, got {len(indices)}")
    selected = [rec["trials"][index] for index in indices]
    return _build_calib_trials(
        rec["neural"], selected, calib_n, TRIAL_LENGTH, rec["n_units"], PAD_VALUE, True
    )


def make_subset_dataset(rec: dict, trials: list[dict], session_name: str) -> MCMazeSessionDataset:
    valid_starts = _compute_valid_starts(trials, WINDOW_SIZE)
    return MCMazeSessionDataset(
        neural_data=rec["neural"],
        behavior_data=rec["behavior"],
        valid_starts=valid_starts,
        calib_trials=rec["calib_trials"],
        window_size=WINDOW_SIZE,
        session_name=session_name,
        side_features=rec.get("side_features"),
        electrode_ids=rec.get("electrode_ids"),
    )


# --- Side-feature support (UNIT_SIDE_FEATURE_ABLATION.md) --------------------------------
# load_session_with_trials()/MCMazeSessionDataset predate B3S; B3S's per-unit side features
# were only ever wired through the *training* path (Dandi688MultiSessionDataModule +
# StreamingCalibrationLitModule). Evaluating a B3S checkpoint with side_dim > 0 through this
# script (and therefore through select_gradient_free_protocol_dandi688.py and
# eval_epoch_window_dandi688.py, which both reuse load_session_with_trials/
# make_subset_dataset/eval_r2 from here) previously had no way to compute or attach those
# features at all -- SideFeatureEarlyPoolEncoder.finalize_identity raises
# "B3S requires side_features when side_dim > 0" the moment such a checkpoint is evaluated.
# The two helpers below close that gap without changing behavior for any side_dim == 0
# checkpoint (variant B3/B15/B15P/B15D/B16, or B3S with --side_features none): every new
# parameter is optional and every new code path is only reached when a checkpoint's own
# run_metadata.json records a real side-feature group.
def load_side_feature_stats_for_run_metadata(
    run_metadata: dict,
    train_files: list[Path],
    cache_dir: Path | None,
) -> tuple[str, str, int, int | None, np.ndarray, np.ndarray, np.ndarray | None] | None:
    """Resolve and fit the side-feature configuration a training run used, or return None.

    Returns ``(side_feature_group, waveform_feature_group, pool_size, permutation_seed,
    mean, std, template_profile)`` ready to pass to ``attach_side_features`` for each session, or ``None`` if
    ``run_metadata`` shows no side features were used (the ``side_features.group`` field
    train_variant_dandi688.py always writes is ``"none"`` in that case). Reuses
    ``fit_side_feature_stats``'s on-disk cache (keyed by feature_group/pool_size/bin_size_ms/
    window_size/trial_result_filter/train sources) so this recovers the exact same train-only
    statistics the checkpoint was trained with (UNIT_SIDE_FEATURE_ABLATION.md section 6.1:
    z-score stats must be fit from train sessions only and must not be refit per session),
    not an independent recomputation that could drift from training-time normalization.
    """
    side_meta = run_metadata.get("side_features") or {}
    group = side_meta.get("group", "none")
    if group == "none":
        return None
    from mc_maze.unit_side_features import (
        TEMPLATE_RIDGE_FEATURE_NAMES,
        base_feature_group,
        fit_side_feature_stats,
    )

    waveform_group = base_feature_group(group)
    signal_view = str(run_metadata.get("signal_view", "sua"))
    pool_size = int(side_meta["pool_size"])
    permutation_seed = side_meta.get("permutation_seed")
    template_profile = None
    if waveform_group in TEMPLATE_RIDGE_FEATURE_NAMES:
        mean, std, receipt = fit_side_feature_stats(
            train_files,
            feature_group=waveform_group,
            pool_size=pool_size,
            cache_dir=cache_dir,
            bin_size_ms=20,
            window_size=WINDOW_SIZE,
            trial_result_filter="R",
            signal_view=signal_view,
            return_template_receipt=True,
        )
        template_profile = np.asarray(receipt["profile"], dtype=np.float32)
    else:
        mean, std = fit_side_feature_stats(
            train_files,
            feature_group=waveform_group,
            pool_size=pool_size,
            cache_dir=cache_dir,
            bin_size_ms=20,
            window_size=WINDOW_SIZE,
            trial_result_filter="R",
            signal_view=signal_view,
        )
    return group, waveform_group, pool_size, permutation_seed, mean, std, template_profile


def attach_side_features(
    rec: dict,
    nwb_path: Path,
    *,
    side_feature_group: str,
    waveform_feature_group: str,
    pool_size: int,
    permutation_seed: int | None,
    mean: np.ndarray,
    std: np.ndarray,
    cache_dir: Path | None,
    template_profile: np.ndarray | None = None,
) -> dict:
    """Return a copy of an ``rec`` from ``load_session_with_trials`` with side features
    (and, for F3/FS3 and the T4-substrate electrode designs A/D/C -- t4e/t4gate/t4anchor,
    docs/ELECTRODE_ANCHOR_DESIGNS.md -- electrode ids) attached."""
    from mc_maze.unit_side_features import (
        is_electrode_shuffle_control,
        is_feature_shuffle_control,
        confidence_component_shuffle,
        is_template_ridge_zero_control,
        load_session_electrode_ids,
        load_unit_side_features,
        permute_electrode_ids,
        permute_t4c_component,
        uses_electrode_ids,
        uses_electrode_relation_membership,
    )

    feature_shuffle_seed = permutation_seed if is_feature_shuffle_control(side_feature_group) else None
    side_features, _ = load_unit_side_features(
        nwb_path,
        feature_group=waveform_feature_group,
        pool_size=pool_size,
        mean=mean,
        std=std,
        cache_dir=cache_dir,
        permutation_seed=feature_shuffle_seed,
        bin_size_ms=20,
        window_size=WINDOW_SIZE,
        trial_result_filter="R",
        signal_view=str(rec.get("signal_view", "sua")),
        template_profile=template_profile,
    )
    if is_template_ridge_zero_control(side_feature_group):
        side_features = np.zeros_like(side_features, dtype=np.float32)
    component_shuffle = confidence_component_shuffle(side_feature_group)
    if component_shuffle is not None:
        if permutation_seed is None:
            raise ValueError(f"{side_feature_group} requires a non-None permutation_seed")
        side_features = permute_t4c_component(
            side_features, component=component_shuffle, permutation_seed=permutation_seed
        )
    if side_features.shape[0] != rec["n_units"]:
        raise ValueError(
            f"{rec['name']}: side_features units {side_features.shape[0]} != "
            f"n_units {rec['n_units']}"
        )
    updated = {**rec, "side_features": side_features}
    if uses_electrode_relation_membership(side_feature_group):
        # In pseudo-MUA, each retained channel is exactly one physical electrode;
        # equality labels need not preserve its absolute NWB index to exercise
        # the required all-singleton relation boundary.
        electrode_ids = (
            np.arange(rec["n_units"], dtype=np.int64)
            if str(rec.get("signal_view", "sua")) == "pseudo_mua"
            else load_session_electrode_ids(nwb_path)
        )
        if electrode_ids.shape[0] != rec["n_units"]:
            raise ValueError(
                f"{rec['name']}: relation membership units {electrode_ids.shape[0]} != "
                f"n_units {rec['n_units']}"
            )
        if is_electrode_shuffle_control(side_feature_group):
            if permutation_seed is None:
                raise ValueError(
                    "t4rel_membership_shuffled requires a non-None permutation_seed"
                )
            from mc_maze.sua_auxiliary_stage0 import deterministic_membership_shuffle

            electrode_ids = deterministic_membership_shuffle(
                electrode_ids, seed=permutation_seed
            )
        updated["electrode_ids"] = electrode_ids
    elif uses_electrode_ids(side_feature_group):
        electrode_ids = load_session_electrode_ids(nwb_path)
        if electrode_ids.shape[0] != rec["n_units"]:
            raise ValueError(
                f"{rec['name']}: electrode_ids units {electrode_ids.shape[0]} != "
                f"n_units {rec['n_units']}"
            )
        if is_electrode_shuffle_control(side_feature_group):
            if permutation_seed is None:
                raise ValueError(
                    f"{side_feature_group!r} is an electrode-shuffle control and requires a "
                    "non-None permutation_seed"
                )
            electrode_ids = permute_electrode_ids(
                electrode_ids, permutation_seed=permutation_seed
            )
        updated["electrode_ids"] = electrode_ids
    return updated


def _unpack_loader_batch(batch):
    """DataLoader batches over MCMazeSessionDataset are 4-tuples
    ``(neural, behavior, calib, session_name)`` normally, or 5-tuples with a trailing
    ``side_features`` when the dataset carries per-unit side features, or 6-tuples with
    ``electrode_ids`` for F3. Returns ``(neural, behavior, calib, side_features,
    electrode_ids)`` with ``None`` for absent optional tensors."""
    if len(batch) == 6:
        neural, behavior, calib, _session_name, side_features, electrode_ids = batch
        return neural, behavior, calib, side_features, electrode_ids
    if len(batch) == 5:
        neural, behavior, calib, _session_name, side_features = batch
        return neural, behavior, calib, side_features, None
    neural, behavior, calib, _session_name = batch
    return neural, behavior, calib, None, None


def zero_identity_for_neural(neural: torch.Tensor) -> torch.Tensor:
    """Return a non-learned [batch, neurons, window] zero identity tensor."""
    if neural.ndim != 3:
        raise ValueError(f"Expected neural [batch, window, neurons], got {tuple(neural.shape)}")
    return torch.zeros(
        neural.shape[0], neural.shape[2], neural.shape[1], device=neural.device, dtype=neural.dtype
    )


def decode_last_behavior(raw_output: torch.Tensor) -> torch.Tensor:
    """Apply the shared decoder-output scaling exactly once for every control."""
    return raw_output[:, -1:, :] / BEHAVIOR_SCALING_FACTOR


@torch.no_grad()
def eval_r2_with_zero_identity(model, dataset: MCMazeSessionDataset, device: torch.device) -> float:
    """Evaluate decoder with a zero identity without invoking the identity encoder."""
    model.eval()
    r2 = R2Score(multioutput="variance_weighted").to(device)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    for batch in loader:
        neural, behavior, _calib, _side_features, _electrode_ids = _unpack_loader_batch(batch)
        neural = neural.to(device)
        behavior = behavior.to(device)
        identity = zero_identity_for_neural(neural)
        if model.student.decoder_mode == "coupled":
            y = model.student.decode_with_identity(neural, identity)
        else:
            direct_key_features = (
                None
                if model.student.decoupled_key_mode == "x_only"
                else identity.new_zeros(
                    identity.shape[0],
                    identity.shape[1],
                    model.student.decoupled_direct_feature_dim,
                )
            )
            y = model.student.decode_with_decoupled_identity(
                neural,
                identity,
                decoder_key_features=direct_key_features,
            )
        y = decode_last_behavior(y)
        r2.update(y.flatten(start_dim=0, end_dim=1), behavior[:, -1:, :].flatten(start_dim=0, end_dim=1))
    return float(r2.compute().item())


@torch.no_grad()
def eval_r2_with_learned_prior(
    model,
    dataset: MCMazeSessionDataset,
    device: torch.device,
) -> float:
    model.eval()
    r2 = R2Score(multioutput="variance_weighted").to(device)
    if getattr(model, "population_identity", None) is None:
        raise ValueError("Model does not expose a learned population identity for prior inference")
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    for batch in loader:
        neural, behavior, _calib, _side_features, _electrode_ids = _unpack_loader_batch(batch)
        neural = neural.to(device)
        behavior = behavior.to(device)
        identity = model.population_identity.to(device)
        y = model.student.decode_with_identity(neural, identity)
        y = decode_last_behavior(y)
        r2.update(y.flatten(start_dim=0, end_dim=1), behavior[:, -1:, :].flatten(start_dim=0, end_dim=1))
    return float(r2.compute().item())


@torch.no_grad()
def eval_r2(model, dataset: MCMazeSessionDataset, device: torch.device) -> float:
    model.eval()
    r2 = R2Score(multioutput="variance_weighted").to(device)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    for batch in loader:
        neural, behavior, calib, side_features, electrode_ids = _unpack_loader_batch(batch)
        neural = neural.to(device)
        behavior = behavior.to(device)
        calib = calib.to(device)
        if side_features is not None:
            side_features = side_features.to(device)
        if electrode_ids is not None:
            electrode_ids = electrode_ids.to(device)
        decoder_key_features = model.decoder_key_features(side_features)
        y, _ = model.student(
            neural,
            calib_trials=calib,
            side_features=side_features,
            decoder_key_features=decoder_key_features,
            electrode_ids=electrode_ids,
        )
        y = decode_last_behavior(y)
        target = behavior[:, -1:, :]
        r2.update(y.flatten(start_dim=0, end_dim=1), target.flatten(start_dim=0, end_dim=1))
    return float(r2.compute().item())


def finetune(
    model,
    dataset: MCMazeSessionDataset,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    tune_decoder: bool,
) -> list[float]:
    model.train()
    if tune_decoder:
        for p in model.student.parameters():
            p.requires_grad = True
    else:
        model.student.freeze_decoder()
    trainable = [p for p in model.student.parameters() if p.requires_grad]
    if not trainable:
        return []
    opt = torch.optim.Adam(trainable, lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    epoch_losses: list[float] = []
    for _ in range(epochs):
        run_loss = 0.0
        n_batches = 0
        for batch in loader:
            neural, behavior, calib, side_features, electrode_ids = _unpack_loader_batch(batch)
            neural = neural.to(device)
            behavior = behavior.to(device)
            calib = calib.to(device)
            if side_features is not None:
                side_features = side_features.to(device)
            if electrode_ids is not None:
                electrode_ids = electrode_ids.to(device)
            decoder_key_features = model.decoder_key_features(side_features)
            y, _ = model.student(
                neural,
                calib_trials=calib,
                side_features=side_features,
                decoder_key_features=decoder_key_features,
                electrode_ids=electrode_ids,
            )
            y = y[:, -1:, :] / BEHAVIOR_SCALING_FACTOR
            target = behavior[:, -1:, :]
            loss = F.mse_loss(y, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += float(loss.item())
            n_batches += 1
        epoch_losses.append(run_loss / max(n_batches, 1))
    model.eval()
    return epoch_losses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Step-1 cross-session checkpoint to adapt.",
    )
    parser.add_argument("--teacher_ckpt", type=str, default=str(DEFAULT_TEACHER))
    parser.add_argument("--variant", type=str, default="B3", choices=["B3", "B15P", "B15D", "B15", "B16"])
    parser.add_argument(
        "--data_dir",
        type=str,
        default="sua_exploration/data/dandi_000688/sub-C",
    )
    parser.add_argument("--task", type=str, default="CO", choices=["CO", "RT"])
    parser.add_argument("--split_counts", type=str, default="37,8,8")
    parser.add_argument(
        "--max_units_exclusive",
        type=int,
        default=None,
        help="Optionally retain only sessions with strictly fewer units than this value.",
    )
    parser.add_argument(
        "--ks",
        type=str,
        default="5,10,20",
        help="Trial counts for opt-in diagnostic-oracle finetuning only.",
    )
    parser.add_argument(
        "--finetune_epochs",
        type=int,
        default=30,
        help="Epochs for opt-in diagnostic-oracle finetuning only.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for opt-in diagnostic-oracle finetuning only.",
    )
    parser.add_argument(
        "--run_diagnostic_oracle_baselines",
        action="store_true",
        help=(
            "Also run label-using backward-gradient finetuning baselines. "
            "Not part of the gradient-free Step 2 deployment protocol."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_sessions",
        type=int,
        default=None,
        help="Limit number of test sessions (for smoke testing). None = all.",
    )
    parser.add_argument("--out_name", type=str, default=None)
    parser.add_argument("--protocol_lock", type=str, default=None,
                        help="Locked validation-selected gradient-free protocol JSON.")
    parser.add_argument("--confirm_formal_test_once", action="store_true",
                        help="Required acknowledgement that this lock can consume its one formal test.")
    args = parser.parse_args()

    import lightning.pytorch as pl

    protocol_lock = None
    if not args.protocol_lock:
        raise ValueError("Formal evaluation is lock-only: --protocol_lock is required")
    if not args.confirm_formal_test_once:
        raise ValueError("--confirm_formal_test_once is required for the single formal test")
    if args.run_diagnostic_oracle_baselines or args.max_sessions is not None:
        raise ValueError("Formal lock-only evaluation forbids oracle baselines and --max_sessions")
    if args.protocol_lock:
        lock_path = Path(args.protocol_lock).expanduser().resolve()
        protocol_lock = json.loads(lock_path.read_text())
        if protocol_lock.get("schema_version") != 1 or protocol_lock.get("purpose") != "locked_gradient_free_formal_test_protocol" or not protocol_lock.get("validation_complete"):
            raise ValueError("--protocol_lock must be a complete locked_gradient_free_formal_test_protocol JSON")
        locked_ckpt = Path(protocol_lock["ckpt"]).resolve()
        if args.ckpt is not None and Path(args.ckpt).expanduser().resolve() != locked_ckpt:
            raise ValueError("--ckpt must match the authoritative protocol-lock checkpoint")
        args.ckpt = str(locked_ckpt)
        args.variant = protocol_lock["variant"]
        args.task = protocol_lock["task"]
        args.split_counts = ",".join(str(value) for value in protocol_lock["split_counts"])
        args.max_units_exclusive = protocol_lock["max_units_exclusive"]
        args.teacher_ckpt = protocol_lock["teacher_ckpt"]
        args.data_dir = protocol_lock["data_dir"]
        args.seed = protocol_lock["seed"]
        selected_protocol = protocol_lock["selected_protocol"]
        if selected_protocol.get("selection_mode") not in {"first", "direction_coverage"}:
            raise ValueError("Protocol-lock has an unsupported selection_mode")
        if not 0 < int(selected_protocol.get("calibration_n", 0)) <= int(selected_protocol.get("pool_size", 0)):
            raise ValueError("Protocol-lock must satisfy 0 < calibration_n <= pool_size")
        source_path = Path(protocol_lock["source_validation_result"])
        if not source_path.is_file() or sha256_file(source_path) != protocol_lock["source_validation_result_sha256"]:
            raise ValueError("Protocol-lock source validation result is missing or has a SHA-256 mismatch")
        source = json.loads(source_path.read_text())
        if source.get("purpose") != "validation_only_gradient_free_protocol_selection" or not source.get("validation_complete"):
            raise ValueError("Protocol-lock source is not a complete validation-only selection result")
        consistency_fields = (
            "ckpt_sha256",
            "teacher_ckpt_sha256",
            "data_dir",
            "variant",
            "task",
            "seed",
            "split_counts",
            "max_units_exclusive",
            "selected_protocol",
            "training_run_metadata",
            "training_run_metadata_sha256",
            "outcome_interpretation",
            "formal_test_scope_id",
        )
        mismatches = [
            field for field in consistency_fields
            if source.get(field) != protocol_lock.get(field)
        ]
        if mismatches:
            raise ValueError(
                "Protocol-lock disagrees with its source validation result for: "
                + ", ".join(mismatches)
            )
    pl.seed_everything(args.seed, workers=True)
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Step-1 checkpoint does not exist: {ckpt_path}")
    if protocol_lock and sha256_file(ckpt_path) != protocol_lock["ckpt_sha256"]:
        raise ValueError("Protocol-lock checkpoint SHA-256 does not match --ckpt")
    teacher_ckpt = Path(args.teacher_ckpt).expanduser().resolve()
    if not teacher_ckpt.is_file():
        raise FileNotFoundError(f"Teacher checkpoint does not exist: {teacher_ckpt}")
    if protocol_lock and sha256_file(teacher_ckpt) != protocol_lock["teacher_ckpt_sha256"]:
        raise ValueError("Protocol-lock teacher checkpoint SHA-256 does not match")
    run_metadata_path = Path(protocol_lock["training_run_metadata"])
    if not run_metadata_path.is_file() or sha256_file(run_metadata_path) != protocol_lock["training_run_metadata_sha256"]:
        raise ValueError("Protocol-lock training run metadata is missing or has a SHA-256 mismatch")
    run_metadata = json.loads(run_metadata_path.read_text())

    split_counts = parse_split_counts(args.split_counts)
    ks = sorted({int(k) for k in args.ks.split(",")})
    max_k = max(ks)
    calib_n = int(protocol_lock["selected_protocol"]["calibration_n"]) if protocol_lock else CALIB_N
    pool_size = int(protocol_lock["selected_protocol"]["pool_size"]) if protocol_lock else calib_n
    selection_mode = protocol_lock["selected_protocol"]["selection_mode"] if protocol_lock else "first"
    out_name = args.out_name or f"{args.variant.lower()}_dandi688_{args.task.lower()}"
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"p3_step2_adaptation_{out_name}_seed{args.seed}.json"
    if out_path.exists():
        raise FileExistsError(f"Formal result already exists and cannot be overwritten: {out_path}")
    formal_test_scope_id = protocol_lock["formal_test_scope_id"]
    receipt_path = create_formal_receipt(
        results_dir, formal_test_scope_id, lock_path, sha256_file(lock_path)
    )

    data_dir = Path(args.data_dir).expanduser().resolve()
    all_files = discover_nwb_files(
        data_dir,
        task=args.task,
        max_units_exclusive=args.max_units_exclusive,
    )
    train_files, val_files, test_files = chronological_session_split(
        all_files,
        split_counts,
        max_units_exclusive=args.max_units_exclusive,
    )
    session_files = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }
    session_splits = {
        split: [session_name_from_path(path) for path in files]
        for split, files in session_files.items()
    }
    session_unit_counts = {
        session_name_from_path(path): nwb_unit_count(path)
        for path in all_files
    }
    current_scope = {
        "data_dir": str(data_dir), "task": args.task, "split_counts": list(split_counts),
        "max_units_exclusive": args.max_units_exclusive,
        "test_sessions": session_splits["test"],
        "test_session_unit_counts": {
            name: session_unit_counts[name] for name in session_splits["test"]
        },
    }
    current_scope_id = hashlib.sha256(
        json.dumps(current_scope, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if current_scope_id != formal_test_scope_id:
        raise ValueError("Current formal test scope differs from the protocol lock")
    if session_splits != source.get("session_splits") or session_unit_counts != source.get("session_unit_counts"):
        raise ValueError("Current discovered session split or unit counts drift from validation source")
    cache_dir_raw = source.get("cache_dir")
    cache_dir = (
        Path(cache_dir_raw).expanduser().resolve()
        if cache_dir_raw is not None
        else None
    )
    signal_view = str(
        source.get("signal_view", run_metadata.get("signal_view", "sua"))
    )
    if signal_view != str(run_metadata.get("signal_view", "sua")):
        raise ValueError("Formal source signal_view differs from training metadata")
    behavior_mean, behavior_std = fit_behavior_stats(
        train_files, bin_size_ms=20, cache_dir=cache_dir
    )
    side_feature_config = load_side_feature_stats_for_run_metadata(
        run_metadata, train_files, cache_dir
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Step-1 checkpoint: {ckpt_path}")
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except Exception:
        # Locally-trained trusted Lightning checkpoint; weights_only=True rejects
        # callback/hparam objects, so fall back to full unpickling here.
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = StreamingCalibrationLitModule(
        task="mc_maze",
        variant=args.variant,
        teacher_ckpt_path=str(teacher_ckpt),
        window_size=WINDOW_SIZE,
        trial_length=TRIAL_LENGTH,
        id_hidden_dim=ID_HIDDEN_DIM,
        hidden_dim=HIDDEN_DIM,
        pad_value=PAD_VALUE,
        freeze_decoder=False,
        loss_mode="task_only",
        decode_last_timestep_only=True,
        predict_scaled_behavior=True,
        behavior_scaling_factor=BEHAVIOR_SCALING_FACTOR,
        **checkpoint_architecture_kwargs(ckpt),
        compile=False,
    )
    model.setup("fit")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.to(device)
    model.eval()
    initial_state = copy.deepcopy(model.state_dict())
    print("Model ready; initial state cached.")

    common_bin_kwargs = dict(
        bin_size_ms=20,
        window_size=WINDOW_SIZE,
        calib_n=calib_n,
        max_trial_length=TRIAL_LENGTH,
        pad_value=PAD_VALUE,
        behavior_mean=behavior_mean,
        behavior_std=behavior_std,
    )

    configs = ["gradient_free_calibrated", "zero_identity_no_calibration"]
    config_roles = {
        "gradient_free_calibrated": "deployment_gradient_free_streaming_calibration",
        "zero_identity_no_calibration": "strict_non_learned_zero_identity_control",
    }
    if args.run_diagnostic_oracle_baselines:
        for k in ks:
            config_name = f"enc_ft_k{k}"
            configs.append(config_name)
            config_roles[config_name] = "diagnostic_oracle_label_gradient_baseline"
        for k in ks:
            config_name = f"encdec_ft_k{k}"
            configs.append(config_name)
            config_roles[config_name] = "diagnostic_oracle_label_gradient_baseline"
    deployment_eval_trial_start_index = pool_size if protocol_lock else calib_n
    eval_trial_start_index = (
        max(max_k, calib_n)
        if args.run_diagnostic_oracle_baselines
        else deployment_eval_trial_start_index
    )

    per_session: dict[str, dict[str, float]] = {}
    paired_delta_vs_no_calibration: dict[str, float] = {}
    formal_trial_selections: dict[str, dict] = {}
    for nwb_path in test_files:
        rec = load_session_with_trials(
            nwb_path,
            **common_bin_kwargs,
            cache_dir=cache_dir,
            signal_view=signal_view,
        )
        if side_feature_config is not None:
            (
                side_feature_group,
                waveform_feature_group,
                side_pool_size,
                permutation_seed,
                side_mean,
                side_std,
                template_profile,
            ) = side_feature_config
            rec = attach_side_features(
                rec,
                nwb_path,
                side_feature_group=side_feature_group,
                waveform_feature_group=waveform_feature_group,
                pool_size=side_pool_size,
                permutation_seed=permutation_seed,
                mean=side_mean,
                std=side_std,
                cache_dir=cache_dir,
                template_profile=template_profile,
            )
        sname = rec["name"]
        n_trials = len(rec["trials"])
        print(f"\n[{sname}] units={rec['n_units']} trials={n_trials}")
        if n_trials <= eval_trial_start_index:
            raise ValueError(
                f"{sname} has {n_trials} usable trials, but the "
                f"{'diagnostic-oracle' if args.run_diagnostic_oracle_baselines else 'deployment'} "
                f"protocol reserves trials[0:{eval_trial_start_index}] before evaluation; "
                "no disjoint evaluation trials remain"
            )

        if protocol_lock:
            selected_indices = select_calibration_trial_indices(
                rec["trials"], calib_n, pool_size, selection_mode
            )
            rec["calib_trials"] = build_calib_trials_for_indices(rec, selected_indices, calib_n)
            formal_trial_selections[sname] = {
                "usable_trial_list_indices": selected_indices,
                "original_trial_indices": [rec["trials"][index]["trial_index"] for index in selected_indices],
                "direction_keys": [repr(canonical_direction_key(rec["trials"][index])) for index in selected_indices],
                "selection_mode": selection_mode,
            }
        eval_trials = rec["trials"][eval_trial_start_index:]
        eval_ds = make_subset_dataset(rec, eval_trials, sname)
        if len(eval_ds) == 0:
            raise ValueError(
                f"{sname} has disjoint evaluation trials[{eval_trial_start_index}:] "
                "but none provide a usable evaluation window"
            )
        print(
            f"  eval set = trials[{eval_trial_start_index}:] -> {len(eval_ds)} windows"
        )

        per_session[sname] = {}
        r2_no_calibration = eval_r2_with_zero_identity(model, eval_ds, device)
        r2_calibrated = eval_r2(model, eval_ds, device)
        per_session[sname]["zero_identity_no_calibration"] = r2_no_calibration
        per_session[sname]["gradient_free_calibrated"] = r2_calibrated
        paired_delta_vs_no_calibration[sname] = r2_calibrated - r2_no_calibration
        print(f"  zero_identity_no_calibration R2 = {r2_no_calibration:+.4f}")
        print(f"  gradient_free_calibrated R2 = {r2_calibrated:+.4f}")

        if args.run_diagnostic_oracle_baselines:
            for k in ks:
                adapt_ds = make_subset_dataset(rec, rec["trials"][:k], sname)
                if len(adapt_ds) == 0:
                    raise ValueError(
                        f"{sname} diagnostic oracle trials[0:{k}] provide no usable "
                        "adaptation windows"
                    )
                for tune_dec, tag in ((False, "enc_ft"), (True, "encdec_ft")):
                    model.load_state_dict(initial_state)
                    losses = finetune(
                        model,
                        adapt_ds,
                        device,
                        epochs=args.finetune_epochs,
                        lr=args.lr,
                        batch_size=args.batch_size,
                        tune_decoder=tune_dec,
                    )
                    r2 = eval_r2(model, eval_ds, device)
                    cfg = f"{tag}_k{k}"
                    per_session[sname][cfg] = r2
                    print(
                        f"  DIAGNOSTIC ORACLE {cfg:14s} R2 = {r2:+.4f} "
                        f"(final_ft_loss={losses[-1]:.4f})"
                    )

        model.load_state_dict(initial_state)

    config_means: dict[str, float] = {}
    for cfg in configs:
        vals = [per_session[s][cfg] for s in per_session]
        config_means[cfg] = float(np.mean(vals)) if vals else float("nan")
    mean_paired_delta = float(np.mean(list(paired_delta_vs_no_calibration.values())))
    positive_session_count = sum(delta > 0 for delta in paired_delta_vs_no_calibration.values())

    payload = {
        "schema_version": 1,
        "step": "P3_step2_adaptation_comparison",
        "created_at": datetime.now().astimezone().isoformat(),
        "ckpt": str(ckpt_path),
        "ckpt_sha256": sha256_file(ckpt_path),
        "protocol_lock": str(Path(args.protocol_lock).resolve()) if args.protocol_lock else None,
        "formal_test_scope_id": formal_test_scope_id,
        "formal_test_receipt": str(receipt_path),
        "variant": args.variant,
        "task": args.task,
        "seed": args.seed,
        "split_counts": list(split_counts),
        "max_units_exclusive": args.max_units_exclusive,
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "signal_view": signal_view,
        "session_splits": session_splits,
        "session_unit_counts": session_unit_counts,
        "session_files": {
            split: [str(path) for path in files]
            for split, files in session_files.items()
        },
        "test_sessions": [session_name_from_path(path) for path in test_files],
        "deployment_protocol": {
            "name": "gradient_free_streaming_calibration",
            "deployment_config": "gradient_free_calibrated",
            "calibration_input": (
                "held-out session spikes plus calibration-block target labels/rates "
                "for the frozen side-feature estimator"
                if side_feature_config is not None
                else "held-out session spikes only"
            ),
            "uses_calibration_target_labels_for_side_features": (
                side_feature_config is not None
            ),
            "disjoint_trial_policy": {
                "calibration_trials": "selected calibration trials within pool[0:pool_size]" if protocol_lock else "trials[0:calibration_n_trials]",
                "default_evaluation_trials": "trials[pool_size:]" if protocol_lock else "trials[calibration_n_trials:]",
                "default_evaluation_start_index": deployment_eval_trial_start_index,
                "pool_size": pool_size if protocol_lock else None,
                "selection_mode": selection_mode if protocol_lock else None,
            },
            "weight_updates": "none; encoder and decoder remain frozen",
            "backward_gradients": False,
            "held_out_behavior_labels_used_for_updates": False,
            "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        },
        "diagnostic_oracle_baselines": {
            "enabled": args.run_diagnostic_oracle_baselines,
            "description": (
                "Optional label-using, backward-gradient finetuning baselines; "
                "not a deployment protocol or Step 2 target."
            ),
            "uses_held_out_behavior_labels": args.run_diagnostic_oracle_baselines,
            "uses_backward_gradients": args.run_diagnostic_oracle_baselines,
        },
        "config_roles": config_roles,
        "ks": ks,
        "max_k": max_k,
        "eval_trial_start_index": eval_trial_start_index,
        "formal_held_out_test": {
            "is_formal_test": protocol_lock is not None,
            "default_configuration": "gradient_free_calibrated",
            "default_backward_gradients": False,
            "default_disjoint_trial_policy": (
                "selected calibration trials within pool[0:pool_size], evaluation trials[pool_size:]"
                if protocol_lock else "calibration trials[0:N], evaluation trials[N:]"
            ),
            "diagnostic_oracle_evaluation_start_index": (
                max(max_k, calib_n) if args.run_diagnostic_oracle_baselines else None
            ),
        },
        "formal_trial_selections": formal_trial_selections if protocol_lock else None,
        "zero_identity_no_calibration": {
            "description": "Non-learned all-zero identity control; no calibration spikes or identity encoder.",
            "limitation": "Not a learned population prior.",
        },
        "paired_delta_gradient_free_calibrated_minus_zero_identity_no_calibration": paired_delta_vs_no_calibration,
        "mean_paired_delta_gradient_free_calibrated_minus_zero_identity_no_calibration": mean_paired_delta,
        "outcome_interpretation": protocol_lock["outcome_interpretation"],
        "formal_outcome": {
            "positive_session_count": positive_session_count,
            "supports_gradient_free_calibration": mean_paired_delta > 0 and positive_session_count >= 4,
            "supports_usable_cross_session_decoding": config_means["gradient_free_calibrated"] > 0,
        },
        "model_state_unchanged": True,
        "calibration_n_trials": calib_n,
        "finetune_epochs": args.finetune_epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "behavior_standardization": "step1_train_session_stats",
        "configs": configs,
        "per_session": per_session,
        "config_means": config_means,
    }
    out_path = results_dir / f"p3_step2_adaptation_{out_name}_seed{args.seed}.json"
    if out_path.exists():
        raise FileExistsError(f"Formal result already exists and cannot be overwritten: {out_path}")
    assert_state_dict_unchanged(initial_state, model.state_dict())
    with out_path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    complete_formal_receipt(receipt_path, out_path)

    print(f"\n=== Summary (mean R^2 across {len(test_files)} test sessions) ===")
    for cfg in configs:
        print(f"  {cfg:14s}: {config_means[cfg]:+.4f}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
