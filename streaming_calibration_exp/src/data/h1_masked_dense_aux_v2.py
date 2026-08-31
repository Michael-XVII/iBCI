"""Additive H1 V2 route: final-legal training subset, legacy metric domain."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from src.data.falcon_datamodule import FalconDataModule, SessionBatchSampler
from src.data.h1_masked_dense_aux_v1 import H1MaskedDenseAuxDataModule
from src.data.h1_window_mask_contract_v1 import (
    append_window_valid,
    build_window_valid,
    window_valid_digest,
)


def _digest_arrays(*values: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode())
        digest.update(repr(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


class FinalLegalWindowDataset(Dataset):
    """Filter training windows by the frozen final-position admission law."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        window_size = int(dataset.window_size)
        kept_original_indices: list[int] = []
        rows_by_session: dict[str, list[tuple[int, int]]] = defaultdict(list)
        original_by_session: dict[str, list[int]] = defaultdict(list)
        excluded_by_session: dict[str, list[int]] = defaultdict(list)
        trial_original: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        trial_retained: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        trial_ids: dict[str, np.ndarray] = {}

        for original_index, (raw_session, raw_start) in enumerate(dataset.window_indices):
            session, start = str(raw_session), int(raw_start)
            final = start + window_size - 1
            original_by_session[session].append(start)
            if not bool(dataset.eval_mask[session][final]):
                raise RuntimeError("established loader admitted a final eval_mask=False window")
            if session not in trial_ids:
                trial_ids[session] = np.cumsum(dataset.trial_change[session].astype(np.int64))
            trial_id = int(trial_ids[session][final])
            trial_original[session][trial_id] += 1
            final_still = bool(np.all(np.abs(dataset.covariate_data[session][final]) < 0.001))
            if final_still:
                excluded_by_session[session].append(start)
                continue
            kept_original_indices.append(original_index)
            rows_by_session[session].append((original_index, start))
            trial_retained[session][trial_id] += 1

        self.kept_original_indices = tuple(kept_original_indices)
        self.window_indices = [dataset.window_indices[index] for index in self.kept_original_indices]
        masks_by_original: dict[int, torch.Tensor] = {}
        self.mask_audit: dict[str, dict[str, Any]] = {}
        for session in sorted(original_by_session):
            rows = rows_by_session[session]
            starts = np.asarray([start for _, start in rows], dtype=np.int64)
            masks = build_window_valid(
                dataset.eval_mask[session], dataset.covariate_data[session],
                dataset.trial_change[session], starts, window_size,
            )
            for (original_index, _), mask in zip(rows, masks, strict=True):
                masks_by_original[original_index] = mask
            original = np.asarray(original_by_session[session], dtype=np.int64)
            excluded = np.asarray(excluded_by_session[session], dtype=np.int64)
            original_trials = dict(sorted(trial_original[session].items()))
            retained_trials = dict(sorted(trial_retained[session].items()))
            missing_trials = [trial for trial in original_trials if retained_trials.get(trial, 0) == 0]
            self.mask_audit[session] = {
                "window_size": window_size,
                "original_windows": int(original.size),
                "retained_windows": int(starts.size),
                "excluded_final_still_windows": int(excluded.size),
                "retention_fraction": float(starts.size / original.size) if original.size else 0.0,
                "final_all_true": bool(masks[:, -1].all().item()) if starts.size else False,
                "legal_count_min": int(masks.sum(dim=1).min().item()) if starts.size else None,
                "legal_count_max": int(masks.sum(dim=1).max().item()) if starts.size else None,
                "original_trials_with_windows": original_trials,
                "retained_windows_by_trial": retained_trials,
                "trials_losing_all_windows": missing_trials,
                "original_window_start_sha256": _digest_arrays(original),
                "retained_window_start_sha256": _digest_arrays(starts),
                "excluded_window_start_sha256": _digest_arrays(excluded),
                "window_valid_sha256": window_valid_digest(starts, masks),
            }
        self._masks = tuple(masks_by_original[index] for index in self.kept_original_indices)
        if len(self._masks) != len(self.window_indices):
            raise RuntimeError("V2 mask/index map is incomplete")

    def __getattr__(self, name: str):
        if name in {"dataset", "kept_original_indices", "window_indices", "_masks", "mask_audit"}:
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __len__(self) -> int:
        return len(self.kept_original_indices)

    def __getitem__(self, index: int):
        return append_window_valid(
            self.dataset[self.kept_original_indices[index]], self._masks[index]
        )


class H1MaskedDenseAuxV2DataModule(H1MaskedDenseAuxDataModule):
    """Filter training only; validation remains the established four-field loader."""

    def setup(self, stage: str | None = None) -> None:
        FalconDataModule.setup(self, stage)
        if dist.is_available() and dist.is_initialized():
            raise RuntimeError("H1 masked dense V2 is registered for single-GPU cells only")
        self.train_original_dataset = self.train_dataset
        self.train_dataset = FinalLegalWindowDataset(self.train_original_dataset)
        self.train_batch_sampler = SessionBatchSampler(
            self.train_dataset, self.batch_size_per_device, shuffle=True,
            seed=self.hparams.sampler_seed,
            balance_sessions=self.hparams.balance_session_batches,
            reshuffle_each_epoch=self.hparams.reshuffle_train_sampler_each_epoch,
        )
        if getattr(self, "val_heldout_dataset", None) is not None:
            raise RuntimeError("formal held-out construction is forbidden")

    def window_mask_audit(self) -> dict[str, Any]:
        validation_counts: dict[str, int] = defaultdict(int)
        validation_starts: dict[str, list[int]] = defaultdict(list)
        for session, start in self.val_heldin_dataset.window_indices:
            validation_counts[str(session)] += 1
            validation_starts[str(session)].append(int(start))
        return {
            "protocol": "final-legal training subset; legacy unfiltered last-bin validation",
            "allowed_sessions": list(self.allowed_sessions),
            "validation_date": self.validation_date,
            "train_sessions": list(self.train_session_names),
            "validation_sessions": list(self.val_heldin_session_names),
            "training": dict(self.train_dataset.mask_audit),
            "legacy_validation": {
                session: {"windows": validation_counts[session],
                          "window_start_sha256": _digest_arrays(np.asarray(starts, dtype=np.int64))}
                for session, starts in sorted(validation_starts.items())
            },
            "formal_heldout_constructed": False,
        }
