"""FALCON benchmark data module - part of "SPINT: Spatial Permutation-Invariant Neural Transformer for Consistent Intracortical Motor Decoding".
Scaffolding adapted from the Hydra template (ashleve/lightning-hydra-template).
Copyright (c) 2024-2026 University of Washington. Developed in UW NeuroAI Lab by Trung Le.
"""
import random
from collections import OrderedDict
from typing import Any, Dict, List, Optional
from falcon_challenge.config import FalconConfig, FalconTask
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import torch.distributed as dist

import os
from pathlib import Path
import numpy as np
import lightning.pytorch as pl
from falcon_challenge.dataloaders import load_nwb
from scipy.interpolate import interp1d
import logging
from third_party.catalyst.distributed_sampler import DistributedSamplerWrapper
from third_party.falcon_challenge.filtering import (
    apply_exponential_filter,
    NEURAL_TAU_MS,
)
from src.data.validation_protocol import loso_split, rotation_5_2_split
from src.data.falcon_t4_features import (
    T4_DIM,
    calibration_target_angles,
    deterministic_row_permutation,
    fit_train_t4_stats,
    t4_from_trial_sums,
    validate_trial_label_alignment,
)
from src.data.falcon_m1_pct4_features import (
    PCT4_DIM,
    calibration_m1_pct4_metadata,
    deterministic_pct4_row_permutation,
    fit_train_pct4_stats,
    pct4_from_phase_sums,
    phase_window_trial_sums,
)


T4_SIDE_GROUPS = {"t4", "ts4"}
PCT4_SIDE_GROUPS = {"pct4", "pct4_rs", "pct4_ls", "pct4_z4"}
ALL_SIDE_GROUPS = {"none"} | T4_SIDE_GROUPS | PCT4_SIDE_GROUPS


class FalconDataset(Dataset):
    def __init__(
        self, 
        sessions_dict,
        calib_sessions_dict,
        window_size=50,
        split=None,
        calibration_n_trials=1.0,
        random_calibration=False,
        smooth_calibration=True,
        max_trial_length=256,
        use_calib_intertrials=False,
        trial_feature_type='raw',
        remove_still_times=False,
        remove_calib_still_times=False,
        use_calib_active_segments=False,
        calib_n_active_segments=1,
        interpolate_trials=False,
        interpolate_trials_kind='linear',
        pad_value=-1.0,
        side_feature_group='none',
        side_feature_shuffle_seed=0,
        side_feature_mean=None,
        side_feature_std=None,
        task='m1',
    ):
        """
        Initializes the FalconDataModule.
        Args:
            sessions_dict (OrderedDict): Dictionary containing multi-session data.
            calib_sessions_dict (OrderedDict): Dictionary containing multi-session calibration data.
            window_size (int, optional): The window size (W). Defaults to 50.
            calibration_n_trials (float, optional): If between 0 and 1: ratio of session neural trials used for calibration. If greater than 1: number of trials in session neural data used for calibration. Defaults to 1.0.
            random_calibration (bool, optional): Whether to randomly sample calibration windows. Defaults to False.
            split (str, optional): Split of this dataset, can be 'train', 'val_heldin', 'val_heldout' or None. Defaults to None.
            smooth_calibration (bool, optional): Whether to smooth calibration data. Defaults to True.
            max_trial_length (int, optional): Maximum length of a trial padded to calculate FFT. Defaults to 256.
            use_calib_intertrials (bool, optional): Whether to use intertrial data in calibration. Defaults to False.
            trial_feature_type (str, optional): The type of trial features to extract for learning neuron identity. Can be 'raw' or 'fft'. Defaults to 'raw'.
        """
        self.calibration_n_trials = calibration_n_trials
        self.random_calibration = random_calibration
        self.trial_feature_type = trial_feature_type
        self.task = str(getattr(task, "name", getattr(task, "value", task))).split(".")[-1].lower()
        self.side_feature_group = str(side_feature_group).lower()
        if self.side_feature_group not in ALL_SIDE_GROUPS:
            raise ValueError(f"Unsupported FALCON side_feature_group {side_feature_group!r}")
        if self.side_feature_group in PCT4_SIDE_GROUPS:
            if self.task != 'm1':
                raise ValueError(f"PCT4-v1 supports native FALCON M1 only, got task={task!r}")
            if smooth_calibration:
                raise ValueError("PCT4-v1 freezes the raw-bin estimator and requires smooth_calibration=false")
        self.side_feature_shuffle_seed = int(side_feature_shuffle_seed)
        self.side_feature_mean = None if side_feature_mean is None else np.asarray(side_feature_mean, dtype=np.float32)
        self.side_feature_std = None if side_feature_std is None else np.asarray(side_feature_std, dtype=np.float32)
        side_dim = PCT4_DIM if self.side_feature_group in PCT4_SIDE_GROUPS else T4_DIM
        if self.side_feature_mean is not None and self.side_feature_mean.shape != (side_dim,):
            raise ValueError("Native FALCON side-feature mean must have shape (4,)")
        if self.side_feature_std is not None and self.side_feature_std.shape != (side_dim,):
            raise ValueError("Native FALCON side-feature std must have shape (4,)")
        self.split = split
        self.window_size = window_size
        pre_history = window_size - 1
        self.use_calib_active_segments = use_calib_active_segments
        self.calib_n_active_segments = calib_n_active_segments
        self.neural_data = {}
        self.covariate_data = {}
        self.eval_mask = {}
        self.trial_change = {}
        self.trial_start_indices = {}
        for session_name, data_dict in sessions_dict.items():
            neural = data_dict["neural"]
            covariates = data_dict["covariates"]
            eval_mask = data_dict["eval_mask"]
            trial_change = data_dict["trial_change"]
            if smooth_calibration:
                neural = apply_exponential_filter(neural, tau=NEURAL_TAU_MS, bin_size=20).astype(np.float32)
            still_times = np.all(np.abs(covariates) < 0.001, axis=1)
            if remove_still_times:
                eval_mask = eval_mask & ~still_times
            self.neural_data[session_name] = np.pad(neural, ((pre_history, 0), (0, 0)), constant_values=0.0, mode='constant') # TxN
            self.covariate_data[session_name] = np.pad(covariates, ((pre_history, 0), (0, 0)), constant_values=0.0, mode='constant') # TxC
            self.eval_mask[session_name] = np.pad(eval_mask, (pre_history, 0), constant_values=False, mode='constant') # T
            self.trial_change[session_name] = np.pad(trial_change, (pre_history, 0), constant_values=False, mode='constant') # T
            self.trial_start_indices[session_name] = np.where(self.trial_change[session_name] == True)[0] # M

        self.calib_neural = {}
        self.calib_covariates = {}
        self.calib_trial_change = {}
        self.calib_target_angles_raw = {}
        self.calib_pct4_reach_sums = {}
        self.calib_pct4_reach_lengths = {}
        self.calib_pct4_post_sums = {}
        self.calib_pct4_post_lengths = {}
        self.calib_pct4_target_angles = {}
        self.calib_neural_active_segments = {}
        self.calib_covariates_active_segments = {}
        for session_name, data_dict in calib_sessions_dict.items():
            calib_neural = data_dict["neural"]
            calib_covariates = data_dict["covariates"]
            calib_trial_change = data_dict["trial_change"]
            if smooth_calibration:
                calib_neural = apply_exponential_filter(calib_neural, tau=NEURAL_TAU_MS, bin_size=20).astype(np.float32)
            
            calib_eval_mask = data_dict["eval_mask"]
            calib_still_times = np.all(np.abs(calib_covariates) < 0.001, axis=1)
            calib_active_times = ~calib_still_times
            calib_target_angles = None
            if self.side_feature_group != 'none':
                calib_target_angles = np.asarray(data_dict['trial_target_angles'], dtype=np.float32)
                validate_trial_label_alignment(
                    calib_trial_change, calib_target_angles, source=f"{session_name} raw calibration"
                )
            if self.side_feature_group in PCT4_SIDE_GROUPS:
                metadata = data_dict.get('m1_pct4_metadata')
                raw_neural = data_dict.get('m1_pct4_raw_neural')
                raw_eval_mask = data_dict.get('m1_pct4_raw_eval_mask')
                if metadata is None or raw_neural is None or raw_eval_mask is None:
                    raise ValueError(f"Missing M1 PCT4 raw metadata for {session_name}")
                reach_sums, reach_lengths = phase_window_trial_sums(
                    raw_neural,
                    raw_eval_mask,
                    metadata.bin_timestamps,
                    metadata.move_onset_times,
                    metadata.contact_times,
                    source=f"{session_name}:reach",
                )
                post_sums, post_lengths = phase_window_trial_sums(
                    raw_neural,
                    raw_eval_mask,
                    metadata.bin_timestamps,
                    metadata.contact_times,
                    metadata.stop_times,
                    source=f"{session_name}:post",
                )
                self.calib_pct4_reach_sums[session_name] = reach_sums.astype(np.float32)
                self.calib_pct4_reach_lengths[session_name] = reach_lengths
                self.calib_pct4_post_sums[session_name] = post_sums.astype(np.float32)
                self.calib_pct4_post_lengths[session_name] = post_lengths
                self.calib_pct4_target_angles[session_name] = metadata.target_angles.astype(np.float32, copy=False)

            # find active segments in calibration data (for H1)
            calib_active_segments = []
            start_idx = None
            for idx, val in enumerate(calib_active_times):
                if val and start_idx is None:
                    start_idx = idx
                elif not val and start_idx is not None:
                    calib_active_segments.append(slice(start_idx, idx))
                    start_idx = None
            # if start_idx is not None:
            if start_idx is not None and start_idx > np.where(calib_trial_change == True)[0][0]: # only select the segments that start after the first trial change
                calib_active_segments.append(slice(start_idx, len(calib_active_times)))
            calib_neural_active_segments = np.full((len(calib_active_segments), max_trial_length, calib_neural.shape[-1]), -1.0, dtype=np.float32)
            calib_covariates_active_segments = np.full((len(calib_active_segments), max_trial_length, calib_covariates.shape[-1]), -1.0, dtype=np.float32)
            for i, segment in enumerate(calib_active_segments):
                segment_length = segment.stop - segment.start
                if segment_length > max_trial_length:
                    segment = slice(segment.start, segment.start + max_trial_length)
                calib_neural_active_segments[i, :segment_length, :] = calib_neural[segment]
                calib_covariates_active_segments[i, :segment_length, :] = calib_covariates[segment]
            self.calib_neural_active_segments[session_name] = calib_neural_active_segments
            self.calib_covariates_active_segments[session_name] = calib_covariates_active_segments

            if not use_calib_intertrials:
                if remove_calib_still_times:
                    calib_eval_mask = calib_eval_mask & calib_active_times
                calib_neural = calib_neural[calib_eval_mask]
                calib_covariates = calib_covariates[calib_eval_mask]
                if calib_target_angles is not None:
                    calib_target_angles = calib_target_angles[
                        np.asarray(calib_eval_mask, dtype=bool)[np.flatnonzero(calib_trial_change)]
                    ]
                calib_trial_change = calib_trial_change[calib_eval_mask]
            self.calib_neural[session_name] = calib_neural
            self.calib_covariates[session_name] = calib_covariates
            self.calib_trial_change[session_name] = calib_trial_change
            if self.side_feature_group != 'none':
                validate_trial_label_alignment(
                    calib_trial_change, calib_target_angles, source=f"{session_name} filtered calibration"
                )
                self.calib_target_angles_raw[session_name] = calib_target_angles

        self.calib_trialized_neural = {}
        self.calib_n_trials = {}
        self.calib_trial_start_indices = {}
        self.calib_trialized_neural_features = {}
        self.calib_trial_spike_sums = {}
        self.calib_trial_lengths = {}
        self.calib_trial_target_angles = {}
        self._side_feature_cache = {}
        for session_name, trial_change in self.calib_trial_change.items():
            calib_neural = self.calib_neural[session_name]
            calib_covariates = self.calib_covariates[session_name]
            trial_starts = np.where(trial_change == True)[0]
            target_angles = self.calib_target_angles_raw.get(
                session_name, np.asarray([], dtype=np.float32)
            )
            if self.side_feature_group != 'none' and target_angles.shape != (trial_starts.shape[0],):
                raise ValueError(
                    f"Native FALCON T4 label/trial mismatch for {session_name}: "
                    f"angles={target_angles.shape}, starts={trial_starts.shape}"
                )
            calib_trialized_neural = []
            calib_trialized_covariates = []
            trial_start_indices = []
            trial_spike_sums = []
            trial_lengths = []
            for i in range(trial_starts.shape[0]):
                start_idx = trial_starts[i]
                end_idx = trial_starts[i + 1] if i + 1 < trial_starts.shape[0] else calib_neural.shape[0]
                trial_neural = calib_neural[start_idx:end_idx, :]
                valid_length = min(trial_neural.shape[0], max_trial_length)
                if valid_length <= 0:
                    raise ValueError(f"Empty calibration trial for {session_name} at index {i}")
                trial_spike_sums.append(trial_neural[:valid_length].sum(axis=0, dtype=np.float64))
                trial_lengths.append(valid_length)
                if interpolate_trials:
                    # interpolate trial_neural and trial_covariates to max_trial_length
                    x_original = np.linspace(0, 1, trial_neural.shape[0])
                    x_target = np.linspace(0, 1, max_trial_length)
                    interpolator_neural = interp1d(x_original, trial_neural, axis=0, kind=interpolate_trials_kind, fill_value="extrapolate")
                    trial_neural = interpolator_neural(x_target).astype(np.float32)
                elif trial_neural.shape[0] < max_trial_length:
                    # pad trial_neural and trial_covariates to max_trial_length
                    trial_neural = np.pad(trial_neural, ((0, max_trial_length - trial_neural.shape[0]), (0, 0)), constant_values=pad_value, mode='constant') # TtxN
                else:
                    # truncate trial_neural and trial_covariates to max_trial_length
                    trial_neural = trial_neural[:max_trial_length, :] # TtxN
                calib_trialized_neural.append(trial_neural)
                trial_start_indices.append(start_idx)
            self.calib_trialized_neural[session_name] = np.array(calib_trialized_neural) # MxTtxN
            self.calib_trial_start_indices[session_name] = np.array(trial_start_indices) # M
            self.calib_trial_spike_sums[session_name] = np.asarray(trial_spike_sums, dtype=np.float32)
            self.calib_trial_lengths[session_name] = np.asarray(trial_lengths, dtype=np.int64)
            if self.side_feature_group != 'none':
                self.calib_trial_target_angles[session_name] = target_angles
            if self.calibration_n_trials < 1.0: # if self.calibration_n_trials is a ratio:
                self.calib_n_trials[session_name] = int(self.calibration_n_trials * self.calib_trialized_neural[session_name].shape[0]) # M'
            else: # else self.calibration_n_trials is number of trials to be sampled:
                self.calib_n_trials[session_name] = self.calibration_n_trials # M'    
            if trial_feature_type == 'raw':
                self.calib_trialized_neural_features[session_name] = self.calib_trialized_neural[session_name] # MxTtxN
            else:
                raise ValueError(f"Unsupported trial feature type: {trial_feature_type}")


        # Precompute all possible (session_name, start_idx) pairs
        self.window_indices = []
        for (session_name, data) in self.neural_data.items():
            T = data.shape[0]
            for start_idx in range(0, T - window_size + 1):
                if self.eval_mask[session_name][start_idx + window_size - 1]: # if last timestep in the window does not belong to an intertrial period
                    self.window_indices.append((session_name, start_idx))

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, idx):
        session_name, start_idx = self.window_indices[idx]
        end_idx = start_idx + self.window_size

        # Extract windows
        neural_window = self.neural_data[session_name][start_idx:end_idx] # W x N
        covariate_window = self.covariate_data[session_name][start_idx:end_idx] # W x C
        if self.use_calib_active_segments: # for H1
            neural_active_segments = self.calib_neural_active_segments[session_name] # M x Tt x N
            if self.random_calibration:
                selected_indices = np.random.choice(neural_active_segments.shape[0], size=self.calib_n_active_segments, replace=False)
                calib_trialized_neural_features = neural_active_segments[selected_indices] # M' x Tt x N
            else:
                calib_trialized_neural_features = neural_active_segments[:self.calib_n_active_segments] # M' x Tt x N
        else:
            calib_n_trials = self.calib_n_trials[session_name] # M'
            calib_total_n_trials = self.calib_trial_start_indices[session_name].shape[0] # M

            # prepare trial features:
            if self.random_calibration: # can only be True in train split
                calib_start_trial_idx = random.randint(0, calib_total_n_trials - calib_n_trials)
            else:
                calib_start_trial_idx = 0

            calib_trialized_neural_features = self.calib_trialized_neural_features[session_name][calib_start_trial_idx:calib_start_trial_idx + calib_n_trials] # M' x (Tt//2 + 1) x N if fft or M' x Tt x N if raw
            if self.trial_feature_type == 'fft':
                calib_trialized_neural_features = np.mean(calib_trialized_neural_features, axis=0, keepdims=False) # M'x(Tt//2 + 1)xN -> (Tt//2 + 1)xN
                calib_trialized_neural_features = np.transpose(calib_trialized_neural_features, (1, 0)).astype(np.float32) # (Tt//2 + 1)xN -> Nx(Tt//2 + 1)

        if self.side_feature_group == 'none':
            return neural_window, covariate_window, calib_trialized_neural_features, session_name
        if self.side_feature_group in T4_SIDE_GROUPS:
            side_features = self._native_t4_side_features(session_name, calib_start_trial_idx, calib_n_trials)
        else:
            side_features = self._native_pct4_side_features(session_name, calib_start_trial_idx, calib_n_trials)
        return (
            neural_window,
            covariate_window,
            calib_trialized_neural_features,
            session_name,
            side_features,
        )

    def set_native_t4_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        if self.side_feature_group == 'none':
            raise ValueError("Cannot set side-feature statistics when side features are disabled")
        self.side_feature_mean = np.asarray(mean, dtype=np.float32)
        self.side_feature_std = np.asarray(std, dtype=np.float32)
        side_dim = PCT4_DIM if self.side_feature_group in PCT4_SIDE_GROUPS else T4_DIM
        if self.side_feature_mean.shape != (side_dim,) or self.side_feature_std.shape != (side_dim,):
            raise ValueError("Native FALCON side-feature normalization must have shape (4,)")
        self._side_feature_cache.clear()

    def set_native_pct4_normalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.set_native_t4_normalization(mean, std)

    def native_t4_statistics_inputs(self, session_names):
        return (
            {name: self.calib_trial_spike_sums[name] for name in session_names},
            {name: self.calib_trial_lengths[name] for name in session_names},
            {name: self.calib_trial_target_angles[name] for name in session_names},
        )

    def native_pct4_statistics_inputs(self, session_names):
        return (
            {name: self.calib_pct4_reach_sums[name] for name in session_names},
            {name: self.calib_pct4_reach_lengths[name] for name in session_names},
            {name: self.calib_pct4_post_sums[name] for name in session_names},
            {name: self.calib_pct4_post_lengths[name] for name in session_names},
            {name: self.calib_pct4_target_angles[name] for name in session_names},
        )

    def _native_t4_side_features(self, session_name, start_trial_idx, calib_n_trials):
        if self.side_feature_mean is None or self.side_feature_std is None:
            raise RuntimeError("Native FALCON T4 statistics must be fitted from train sessions before use")
        key = (self.side_feature_group, session_name, int(start_trial_idx), int(calib_n_trials))
        if key not in self._side_feature_cache:
            stop = start_trial_idx + calib_n_trials
            raw = t4_from_trial_sums(
                self.calib_trial_spike_sums[session_name][start_trial_idx:stop],
                self.calib_trial_lengths[session_name][start_trial_idx:stop],
                self.calib_trial_target_angles[session_name][start_trial_idx:stop],
                source=f"{session_name}[{start_trial_idx}:{stop}]",
            )
            values = ((raw - self.side_feature_mean) / self.side_feature_std).astype(np.float32)
            if self.side_feature_group == 'ts4':
                values = values[deterministic_row_permutation(
                    values.shape[0], session_name=session_name, seed=self.side_feature_shuffle_seed
                )]
            self._side_feature_cache[key] = values
        return self._side_feature_cache[key]

    def _native_pct4_side_features(self, session_name, start_trial_idx, calib_n_trials):
        key = (self.side_feature_group, session_name, int(start_trial_idx), int(calib_n_trials))
        if key not in self._side_feature_cache:
            stop = start_trial_idx + calib_n_trials
            if self.side_feature_group == 'pct4_z4':
                num_channels = self.calib_pct4_reach_sums[session_name].shape[1]
                values = np.zeros((num_channels, PCT4_DIM), dtype=np.float32)
            else:
                raw = pct4_from_phase_sums(
                    self.calib_pct4_reach_sums[session_name][start_trial_idx:stop],
                    self.calib_pct4_reach_lengths[session_name][start_trial_idx:stop],
                    self.calib_pct4_post_sums[session_name][start_trial_idx:stop],
                    self.calib_pct4_post_lengths[session_name][start_trial_idx:stop],
                    self.calib_pct4_target_angles[session_name][start_trial_idx:stop],
                    source=f"{session_name}[{start_trial_idx}:{stop}]",
                    label_shuffle_seed=(
                        self.side_feature_shuffle_seed if self.side_feature_group == 'pct4_ls' else None
                    ),
                    session_name=session_name,
                )
                if self.side_feature_mean is None or self.side_feature_std is None:
                    raise RuntimeError("M1 PCT4 statistics must be fitted from train sessions before use")
                values = ((raw - self.side_feature_mean) / self.side_feature_std).astype(np.float32)
                if self.side_feature_group == 'pct4_rs':
                    values = values[deterministic_pct4_row_permutation(
                        values.shape[0], session_name=session_name, seed=self.side_feature_shuffle_seed
                    )]
            self._side_feature_cache[key] = values
        return self._side_feature_cache[key]

class SessionBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle=False,
        seed=42,
        balance_sessions=False,
        reshuffle_each_epoch=False,
    ):
        """
        Args:
            dataset (FalconDataset): The dataset object.
            batch_size (int): The number of windows per batch.
            shuffle (bool, optional): Whether to shuffle the indices. Defaults to False.
            seed (int, optional): Seed for the fixed session-local and batch-order shuffle.
                Defaults to 42 for backward compatibility.
            balance_sessions (bool or float, optional): Strength of interpolation from
                empirical session batch counts (0/False) to equal counts (1/True), while
                preserving the original total number of batches. Shorter sessions are
                cycled deterministically. Defaults to False.
            reshuffle_each_epoch (bool, optional): Rebuild the session-local and global
                batch order with ``seed + epoch`` on each iterator. Defaults to False for
                backward compatibility.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = int(seed)
        self.balance_strength = float(balance_sessions)
        if not 0.0 <= self.balance_strength <= 1.0:
            raise ValueError("balance_sessions must be in [0, 1]")
        self.balance_sessions = self.balance_strength > 0.0
        self.reshuffle_each_epoch = bool(reshuffle_each_epoch)
        self.epoch = 0

        # Group indices by session
        self.session_to_indices = {}
        for idx, (session_name, _) in enumerate(dataset.window_indices):
            if session_name not in self.session_to_indices:
                self.session_to_indices[session_name] = []
            self.session_to_indices[session_name].append(idx)

        self.original_session_batch_counts = {
            session_name: len(indices) // self.batch_size
            for session_name, indices in self.session_to_indices.items()
        }
        self.batched_indices = self._build_batches(self.seed)

    def _build_batches(self, seed):
        session_batches = {}
        for session_name, original_indices in self.session_to_indices.items():
            session_indices = list(original_indices)
            if self.shuffle:
                session_indices = random.Random(seed).sample(session_indices, len(session_indices))
            batches = []
            for i in range(0, len(session_indices), self.batch_size):
                batch = session_indices[i:i + self.batch_size]
                if len(batch) == self.batch_size:  # Drop the last batch if it's smaller than batch_size
                    batches.append(batch)
            session_batches[session_name] = batches

        if self.balance_sessions and session_batches:
            total_batches = sum(len(batches) for batches in session_batches.values())
            session_names = [
                session_name
                for session_name, batches in session_batches.items()
                if batches
            ]
            if not session_names:
                return []
            equal_count = total_batches / len(session_names)
            raw_target_counts = {
                session_name: (
                    (1.0 - self.balance_strength) * len(session_batches[session_name])
                    + self.balance_strength * equal_count
                )
                for session_name in session_names
            }
            target_counts = {
                session_name: int(raw_target_counts[session_name])
                for session_name in session_names
            }
            remaining = total_batches - sum(target_counts.values())
            priority = sorted(
                session_names,
                key=lambda session_name: (
                    -(raw_target_counts[session_name] - target_counts[session_name]),
                    session_names.index(session_name),
                ),
            )
            for session_name in priority[:remaining]:
                target_counts[session_name] += 1

            session_schedule = []
            scheduled_counts = {session_name: 0 for session_name in session_names}
            while len(session_schedule) < total_batches:
                for session_name in session_names:
                    if scheduled_counts[session_name] < target_counts[session_name]:
                        session_schedule.append(session_name)
                        scheduled_counts[session_name] += 1
            if self.shuffle:
                session_schedule = random.Random(seed).sample(
                    session_schedule, len(session_schedule)
                )
            session_offsets = {session_name: 0 for session_name in session_names}
            batched_indices = []
            for session_name in session_schedule:
                batches = session_batches[session_name]
                offset = session_offsets[session_name]
                batched_indices.append(batches[offset % len(batches)])
                session_offsets[session_name] += 1
            return batched_indices

        batched_indices = [
            batch for batches in session_batches.values() for batch in batches
        ]
        if self.shuffle:
            batched_indices = random.Random(seed).sample(
                batched_indices, len(batched_indices)
            )
        return batched_indices

    def __iter__(self):
        seed = self.seed + self.epoch if self.reshuffle_each_epoch else self.seed
        batched_indices = (
            self._build_batches(seed) if self.reshuffle_each_epoch else self.batched_indices
        )
        if self.reshuffle_each_epoch:
            self.epoch += 1
        for batch_indices in batched_indices:
            yield batch_indices

    def __len__(self):
        return len(self.batched_indices)


class FalconDataModule(pl.LightningDataModule):
    """`LightningDataModule` for the FALCON dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        task: str,
        data_dir: str,
        heldin_session_names: list[str] = [''],
        batch_size: int = 64,
        window_size: int = 50,
        calibration_n_trials: float = 1.0,
        random_calibration: bool = False,
        smooth_calibration: bool = True,
        max_trial_length: int = 256,
        standardize_covariates: bool = False,
        use_intertrials: bool = True,
        use_calib_intertrials: bool = False,
        trial_feature_type: str = 'raw',
        remove_still_times: bool = False,
        remove_calib_still_times: bool = False,
        use_calib_active_segments: bool = False,
        calib_n_active_segments: int = 1,
        interpolate_trials: bool = False,
        interpolate_trials_kind: str = 'linear',
        pad_value: float = -1.0,
        num_workers: int | None = os.cpu_count() - 1,
        pin_memory: bool = False,
        validation_protocol: str = "minival",
        loso_fold: int | None = None,
        rotation_id: int = 0,
        include_heldout_in_fit: bool = False,
        include_heldout_in_test: bool = False,
        sampler_seed: int = 42,
        balance_session_batches: float | bool = False,
        reshuffle_train_sampler_each_epoch: bool = False,
        side_feature_group: str = "none",
        side_feature_shuffle_seed: int = 0,
        ) -> None:
        """
        Initialize a `FALCONDataModule`.

        :param task: The task to be performed.
        :param data_dir: The data directory.
        :param batch_size: The batch size. Defaults to `64`.
        :param window_size: The size of the window. Defaults to `50`.
        :param calibration_n_trials: If between 0 and 1: ratio of session neural trials used for calibration. If greater than 1: number of trials in session neural data used for calibration. Defaults to 1.0.
        :param random_calibration: Whether to randomly sample calibration windows. Defaults to False.
        :param smooth_calibration: Whether to apply smoothing to calibration. Defaults to `True`.
        :param max_trial_length: The maximum length of a trial padded to calculate FFT. Defaults to `256`.
        :param standardize_covariates: Whether to standardize covariates. Defaults to `False`.
        :param use_intertrials: Whether to use intertrial data. Defaults to `True`.
        :param use_calib_intertrials: Whether to use intertrial data in calibration. Defaults to `False`.
        :param trial_feature_type: The type of trial features to extract for learning neuron identity. Can be 'raw' or 'fft'. Defaults to 'raw'.
        :param num_workers: The number of workers. Defaults to `os.cpu_count() - 1`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        data_dir = Path(data_dir)
        num_workers = num_workers if num_workers is not None else os.cpu_count() - 1
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.batch_size_per_device = batch_size

    @staticmethod
    def _subset_sessions(sessions: OrderedDict, names: List[str]) -> OrderedDict:
        return OrderedDict((name, sessions[name]) for name in names if name in sessions)

    def _resolve_train_val_sessions(self, all_heldin_sessions: List[str]) -> tuple[List[str], List[str]]:
        protocol = self.hparams.validation_protocol
        if protocol == "minival":
            return list(all_heldin_sessions), list(all_heldin_sessions)
        if protocol == "loso":
            if self.hparams.loso_fold is None:
                raise ValueError("validation_protocol=loso requires data.loso_fold in [0, 6]")
            train_sessions, heldout_session = loso_split(all_heldin_sessions, int(self.hparams.loso_fold))
            return train_sessions, [heldout_session]
        if protocol == "rotation_5_2":
            return rotation_5_2_split(all_heldin_sessions, int(self.hparams.rotation_id))
        raise ValueError(f"Unknown validation_protocol: {protocol}")

    def _needs_heldout_data(self, stage: Optional[str]) -> bool:
        if self.hparams.include_heldout_in_fit:
            return True
        return stage == "test" and self.hparams.include_heldout_in_test

    def get_split_manifest(self) -> Dict[str, Any]:
        fold_id = self.hparams.loso_fold
        if self.hparams.validation_protocol == "rotation_5_2":
            fold_id = self.hparams.rotation_id
        manifest = {
            "validation_protocol": self.hparams.validation_protocol,
            "fold_id": fold_id,
            "train_sessions": list(getattr(self, "train_session_names", [])),
            "validation_sessions": list(getattr(self, "val_heldin_session_names", [])),
            "heldout_evaluated_in_fit": bool(self.hparams.include_heldout_in_fit),
            "heldout_evaluated_in_test": bool(self.hparams.include_heldout_in_test),
        }
        normalization = getattr(self, "native_t4_normalization", None)
        if normalization is not None:
            encoded = {
                "feature_group": normalization["feature_group"],
                "train_sessions": list(normalization["train_sessions"]),
                "mean": np.asarray(normalization["mean"], dtype=np.float32).tolist(),
                "std": np.asarray(normalization["std"], dtype=np.float32).tolist(),
            }
            import hashlib, json
            encoded["sha256"] = hashlib.sha256(json.dumps(encoded, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            manifest["native_t4_normalization"] = encoded
        return manifest

    def _build_heldout_dataset(
        self, covariates_mean, covariates_std, task_config,
        *, side_feature_group: str, side_feature_mean, side_feature_std,
    ) -> None:
        if getattr(self, "val_heldout_dataset", None) is not None:
            return
        val_calib_heldout_files = sorted([f for f in self.hparams.data_dir.rglob("*held-out-calib*.nwb")])
        self.val_calib_heldout_sessions = OrderedDict()
        for f in val_calib_heldout_files:
            session_name = f.name.split("_")[1].split(".")[0]
            self.val_calib_heldout_sessions[session_name] = self.prepare_session_data(
                f,
                task_config.task,
                standardize_covariates=self.hparams.standardize_covariates,
                covariates_mean=covariates_mean,
                covariates_std=covariates_std,
                use_intertrials=self.hparams.use_intertrials,
                # T4 labels are read solely from this held-out *calibration* NWB's
                # trials.tgt_loc.  Query/evaluation covariates never enter feature
                # construction; they remain present only as the supervised test target.
                include_trial_targets=side_feature_group != 'none',
            )
        self.val_heldout_dataset = FalconDataset(
            sessions_dict=self.val_calib_heldout_sessions,
            calib_sessions_dict=self.val_calib_heldout_sessions,
            window_size=self.hparams.window_size,
            split="val_heldout",
            calibration_n_trials=self.hparams.calibration_n_trials,
            random_calibration=False,
            smooth_calibration=self.hparams.smooth_calibration,
            max_trial_length=self.hparams.max_trial_length,
            use_calib_intertrials=self.hparams.use_calib_intertrials,
            trial_feature_type=self.hparams.trial_feature_type,
            remove_still_times=self.hparams.remove_still_times,
            remove_calib_still_times=self.hparams.remove_calib_still_times,
            use_calib_active_segments=self.hparams.use_calib_active_segments,
            calib_n_active_segments=self.hparams.calib_n_active_segments,
            interpolate_trials=self.hparams.interpolate_trials,
            interpolate_trials_kind=self.hparams.interpolate_trials_kind,
            pad_value=self.hparams.pad_value,
            side_feature_group=side_feature_group,
            side_feature_shuffle_seed=self.hparams.side_feature_shuffle_seed,
            side_feature_mean=side_feature_mean,
            side_feature_std=side_feature_std,
            task=self.hparams.task,
        )
        self.val_heldout_batch_sampler = SessionBatchSampler(
            self.val_heldout_dataset, self.batch_size_per_device, shuffle=False
        )
        if dist.is_available() and dist.is_initialized():
            self.val_heldout_batch_sampler = DistributedSamplerWrapper(
                self.val_heldout_batch_sampler,
                shuffle=False,
            )
        logging.info(f"Validation heldout dataset: {len(self.val_heldout_dataset)} windows")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        side_feature_group = str(self.hparams.side_feature_group).lower()
        if side_feature_group not in ALL_SIDE_GROUPS:
            raise ValueError(f"Unsupported FALCON side_feature_group {self.hparams.side_feature_group!r}")
        if side_feature_group != 'none' and self.hparams.random_calibration:
            raise ValueError(
                "Native FALCON side features currently require random_calibration=false: arbitrary "
                "short M1 support windows can be direction-rank-deficient. Matched cells "
                "must all use the deterministic first-support protocol."
            )
        task_name = str(self.hparams.task).lower()
        if side_feature_group in PCT4_SIDE_GROUPS:
            if task_name != 'm1':
                raise ValueError(f"PCT4-v1 supports native FALCON M1 only, got task={self.hparams.task!r}")
            if self.hparams.smooth_calibration:
                raise ValueError("PCT4-v1 requires smooth_calibration=false")

        task_config = FalconConfig(task=FalconTask.__dict__[self.hparams.task],)
        train_calib_heldin_files = sorted([f for f in self.hparams.data_dir.rglob('*held-in-calib*.nwb') if any(session_name in f.name for session_name in self.hparams.heldin_session_names)])
        val_heldin_files = sorted([f for f in self.hparams.data_dir.rglob('*held-in-minival*.nwb') if any(session_name in f.name for session_name in self.hparams.heldin_session_names)])

        logging.info(f"Data directory: {self.hparams.data_dir}")
        logging.info(f"Train calibration heldin files: {train_calib_heldin_files}")
        logging.info(f"Val heldin files: {val_heldin_files}")
        if self._needs_heldout_data(stage):
            logging.info("Held-out dataset will be loaded for this stage.")
        else:
            logging.info("Held-out dataset skipped for this stage.")

        self.train_calib_heldin_sessions = OrderedDict()
        self.val_heldin_sessions = OrderedDict()
        for i, f in enumerate(train_calib_heldin_files):
            session_name = f.name.split('_')[1].split('.')[0]
            if i == 0:
                self.train_calib_heldin_sessions[session_name] = self.prepare_session_data(f, 
                                                                                           task_config.task, 
                                                                                           standardize_covariates=self.hparams.standardize_covariates,
                                                                                           use_intertrials=self.hparams.use_intertrials,
                                                                                           include_trial_targets=str(self.hparams.side_feature_group).lower() != 'none',
                                                                                           )
                covariates_mean = self.train_calib_heldin_sessions[session_name]['covariates_mean']
                covariates_std = self.train_calib_heldin_sessions[session_name]['covariates_std']
            else:
                self.train_calib_heldin_sessions[session_name] = self.prepare_session_data(
                f, 
                task_config.task,
                standardize_covariates=self.hparams.standardize_covariates,
                covariates_mean=covariates_mean, 
                covariates_std=covariates_std,
                use_intertrials=self.hparams.use_intertrials,
                include_trial_targets=str(self.hparams.side_feature_group).lower() != 'none',
            )
        for f in val_heldin_files:
            session_name = f.name.split('_')[1].split('.')[0]
            self.val_heldin_sessions[session_name] = self.prepare_session_data(
                f, 
                task_config.task, 
                standardize_covariates=self.hparams.standardize_covariates,
                covariates_mean=covariates_mean, 
                covariates_std=covariates_std, 
                use_intertrials=self.hparams.use_intertrials,
                include_trial_targets=False,
            )
        all_heldin_sessions = list(self.train_calib_heldin_sessions.keys())
        train_sessions, val_heldin_sessions = self._resolve_train_val_sessions(all_heldin_sessions)
        self.train_session_names = train_sessions
        self.val_heldin_session_names = val_heldin_sessions
        logging.info(f"Validation protocol: {self.hparams.validation_protocol}")
        logging.info(f"Train held-in sessions ({len(train_sessions)}): {train_sessions}")
        logging.info(f"Val held-in sessions ({len(val_heldin_sessions)}): {val_heldin_sessions}")

        train_query_sessions = self._subset_sessions(self.train_calib_heldin_sessions, train_sessions)
        train_calib_sessions = train_query_sessions
        val_query_sessions = self._subset_sessions(self.val_heldin_sessions, val_heldin_sessions)
        val_calib_sessions = self._subset_sessions(self.train_calib_heldin_sessions, val_heldin_sessions)

        self.train_dataset = FalconDataset(
            sessions_dict=train_query_sessions,
            calib_sessions_dict=train_calib_sessions,
            window_size=self.hparams.window_size,
            split='train',
            calibration_n_trials=self.hparams.calibration_n_trials,
            random_calibration=self.hparams.random_calibration,
            smooth_calibration=self.hparams.smooth_calibration,
            max_trial_length=self.hparams.max_trial_length,
            use_calib_intertrials=self.hparams.use_calib_intertrials,
            trial_feature_type=self.hparams.trial_feature_type,
            remove_still_times=self.hparams.remove_still_times,
            remove_calib_still_times=self.hparams.remove_calib_still_times,
            use_calib_active_segments=self.hparams.use_calib_active_segments,
            calib_n_active_segments=self.hparams.calib_n_active_segments,
            interpolate_trials=self.hparams.interpolate_trials,
            interpolate_trials_kind=self.hparams.interpolate_trials_kind,
            pad_value=self.hparams.pad_value,
            side_feature_group=side_feature_group,
            side_feature_shuffle_seed=self.hparams.side_feature_shuffle_seed,
            task=self.hparams.task,
        )
        side_feature_mean = side_feature_std = None
        if side_feature_group in T4_SIDE_GROUPS:
            sums, lengths, angles = self.train_dataset.native_t4_statistics_inputs(train_sessions)
            side_feature_mean, side_feature_std = fit_train_t4_stats(
                sums, lengths, angles, train_sessions, int(self.hparams.calibration_n_trials)
            )
            self.train_dataset.set_native_t4_normalization(side_feature_mean, side_feature_std)
        elif side_feature_group in PCT4_SIDE_GROUPS:
            reach_sums, reach_lengths, post_sums, post_lengths, angles = self.train_dataset.native_pct4_statistics_inputs(train_sessions)
            side_feature_mean, side_feature_std = fit_train_pct4_stats(
                reach_sums,
                reach_lengths,
                post_sums,
                post_lengths,
                angles,
                train_sessions,
                int(self.hparams.calibration_n_trials),
                feature_group=side_feature_group,
                label_shuffle_seed=self.hparams.side_feature_shuffle_seed,
            )
            self.train_dataset.set_native_pct4_normalization(side_feature_mean, side_feature_std)
        if side_feature_group != 'none':
            self.native_t4_normalization = {
                'feature_group': side_feature_group,
                'mean': side_feature_mean,
                'std': side_feature_std,
                'train_sessions': list(train_sessions),
            }
        self.val_heldin_dataset = FalconDataset(
            sessions_dict=val_query_sessions,
            calib_sessions_dict=val_calib_sessions,
            window_size=self.hparams.window_size,
            split='val_heldin',
            calibration_n_trials=self.hparams.calibration_n_trials,
            random_calibration=False,
            smooth_calibration=self.hparams.smooth_calibration,
            max_trial_length=self.hparams.max_trial_length,
            use_calib_intertrials=self.hparams.use_calib_intertrials,
            trial_feature_type=self.hparams.trial_feature_type,
            remove_still_times=self.hparams.remove_still_times,
            remove_calib_still_times=self.hparams.remove_calib_still_times,
            use_calib_active_segments=self.hparams.use_calib_active_segments,
            calib_n_active_segments=self.hparams.calib_n_active_segments,
            interpolate_trials=self.hparams.interpolate_trials,
            interpolate_trials_kind=self.hparams.interpolate_trials_kind,
            pad_value=self.hparams.pad_value,
            side_feature_group=side_feature_group,
            side_feature_shuffle_seed=self.hparams.side_feature_shuffle_seed,
            side_feature_mean=side_feature_mean,
            side_feature_std=side_feature_std,
            task=self.hparams.task,
        )
        self.val_heldout_dataset = None

        logging.info(f"Training dataset: {len(self.train_dataset)} windows")
        logging.info(f"Validation heldin dataset: {len(self.val_heldin_dataset)} windows")

        self.train_batch_sampler = SessionBatchSampler(
            self.train_dataset,
            self.batch_size_per_device,
            shuffle=True,
            seed=self.hparams.sampler_seed,
            balance_sessions=self.hparams.balance_session_batches,
            reshuffle_each_epoch=self.hparams.reshuffle_train_sampler_each_epoch,
        )
        logging.info(
            "Train sampler full-batch counts by session: %s; balance_strength=%s; "
            "reshuffle_each_epoch=%s; batches_per_epoch=%d",
            self.train_batch_sampler.original_session_batch_counts,
            self.train_batch_sampler.balance_strength,
            self.hparams.reshuffle_train_sampler_each_epoch,
            len(self.train_batch_sampler),
        )
        self.val_heldin_batch_sampler = SessionBatchSampler(self.val_heldin_dataset, self.batch_size_per_device, shuffle=False)

        if self._needs_heldout_data(stage):
            self._build_heldout_dataset(
                covariates_mean, covariates_std, task_config,
                side_feature_group=side_feature_group,
                side_feature_mean=side_feature_mean,
                side_feature_std=side_feature_std,
            )

        if dist.is_available() and dist.is_initialized():
            logging.info(f"World size: {dist.get_world_size()}")
            logging.info(f"Rank: {dist.get_rank()}")
            self.train_batch_sampler = DistributedSamplerWrapper(
                self.train_batch_sampler,
                shuffle=True,
            )
            self.val_heldin_batch_sampler = DistributedSamplerWrapper(
                self.val_heldin_batch_sampler,
                shuffle=False,
            )

    def prepare_session_data(self, session_data_file, task, standardize_covariates=False, covariates_mean=None, covariates_std=None, use_intertrials=True, include_trial_targets=False):
        session_data_dict = {}
        neural, covariates, trial_change, eval_mask = self.load_data(session_data_file, task, use_intertrials=use_intertrials)
        if include_trial_targets:
            # The target source is the calibration NWB trials table, never query/evaluation
            # covariates.  Validate before and after optional intertrial filtering so a
            # change in FALCON's trialization cannot silently attach wrong labels.
            raw_neural, _, raw_trial_change, raw_eval_mask = self.load_data(
                session_data_file, task, use_intertrials=True
            )
            target_angles = calibration_target_angles(Path(session_data_file), task)
            validate_trial_label_alignment(raw_trial_change, target_angles, source=str(session_data_file))
            if use_intertrials:
                retained_angles = target_angles
            else:
                retained_angles = target_angles[
                    np.asarray(raw_eval_mask, dtype=bool)[np.flatnonzero(raw_trial_change)]
                ]
            validate_trial_label_alignment(trial_change, retained_angles, source=str(session_data_file))
            session_data_dict['trial_target_angles'] = retained_angles.astype(np.float32, copy=False)
            if str(getattr(task, "name", getattr(task, "value", task))).split(".")[-1].lower() == 'm1':
                session_data_dict['m1_pct4_metadata'] = calibration_m1_pct4_metadata(Path(session_data_file), task)
                session_data_dict['m1_pct4_raw_neural'] = raw_neural.astype(np.float32, copy=False)
                session_data_dict['m1_pct4_raw_eval_mask'] = np.asarray(raw_eval_mask, dtype=bool)
        session_data_dict['neural'] = neural.astype(np.float32)
        covariates = covariates.astype(np.float32)
        standardized_covariates, covariates_mean, covariates_std = self.standardize(covariates, covariates_mean, covariates_std)
        session_data_dict['covariates'] = standardized_covariates if standardize_covariates else covariates
        session_data_dict['covariates_mean'] = covariates_mean
        session_data_dict['covariates_std'] = covariates_std
        session_data_dict['trial_change'] = trial_change
        session_data_dict['eval_mask'] = eval_mask
        return session_data_dict
        
    def load_data(self, file, task, use_intertrials=True):
        neural, covariates, trial_change, eval_mask = load_nwb(file, task)
        if np.isnan(neural).any() or np.isnan(covariates).any() or np.isnan(trial_change).any():
            raise ValueError(f"NaN values found in the data from file {file}")
        if use_intertrials:
            return neural, covariates, trial_change, eval_mask
        else:
            return neural[eval_mask], covariates[eval_mask], trial_change[eval_mask], eval_mask[eval_mask] # eval_mask becomes an all-True shorter vector

    def standardize(self, data, mean=None, std=None):
        mean = np.mean(data, axis=0) if mean is None else mean
        std = np.std(data, axis=0) if std is None else std
        std[std == 0] = 1
        standardized_data = (data - mean) / std
        return standardized_data, mean, std

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_sampler=self.train_batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            # shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any] | list[DataLoader[Any]]:
        """Validation dataloader(s). Held-out is excluded during fit by default."""
        heldin_loader = DataLoader(
            dataset=self.val_heldin_dataset,
            batch_sampler=self.val_heldin_batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
        if not self.hparams.include_heldout_in_fit:
            return heldin_loader
        return [
            heldin_loader,
            DataLoader(
                dataset=self.val_heldout_dataset,
                batch_sampler=self.val_heldout_batch_sampler,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
            ),
        ]
    
    
    def test_dataloader(self) -> DataLoader[Any] | list[DataLoader[Any]]:
        """Test dataloader(s). Held-out is excluded unless include_heldout_in_test=true."""
        heldin_loader = DataLoader(
            dataset=self.val_heldin_dataset,
            batch_sampler=self.val_heldin_batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
        if not self.hparams.include_heldout_in_test or self.val_heldout_dataset is None:
            return heldin_loader
        return [
            heldin_loader,
            DataLoader(
                dataset=self.val_heldout_dataset,
                batch_sampler=self.val_heldout_batch_sampler,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
            ),
        ]
    
