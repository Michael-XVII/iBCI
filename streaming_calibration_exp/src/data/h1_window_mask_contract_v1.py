"""Typed H1 window-validity contract for masked dense supervision.

This module is deliberately additive.  The established ``FalconDataset``
batch remains unchanged unless ``append_window_valid`` is called explicitly.
"""
from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


_DIGEST_SCHEMA = b"h1-window-valid-v1\0"


def _session_arrays(
    eval_mask: np.ndarray,
    covariates: np.ndarray,
    trial_change: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    evaluation = np.asarray(eval_mask)
    targets = np.asarray(covariates)
    boundaries = np.asarray(trial_change)
    if evaluation.ndim != 1 or evaluation.dtype != np.bool_:
        raise TypeError("eval_mask must be a one-dimensional bool array")
    if boundaries.ndim != 1 or boundaries.dtype != np.bool_:
        raise TypeError("trial_change must be a one-dimensional bool array")
    if targets.ndim != 2 or not np.issubdtype(targets.dtype, np.number):
        raise TypeError("covariates must be a two-dimensional numeric array")
    if not (evaluation.shape[0] == targets.shape[0] == boundaries.shape[0]):
        raise ValueError("session arrays must have the same time dimension")
    return evaluation, targets, boundaries


def _window_starts(
    window_starts: Sequence[int] | np.ndarray,
    *,
    session_length: int,
    window_size: int,
) -> np.ndarray:
    if type(window_size) is not int or window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    starts = np.asarray(window_starts)
    if starts.ndim != 1 or not np.issubdtype(starts.dtype, np.integer):
        raise TypeError("window_starts must be a one-dimensional integer sequence")
    starts = starts.astype(np.int64, copy=False)
    if np.unique(starts).size != starts.size:
        raise ValueError("window_starts must not contain duplicates")
    if starts.size and (int(starts.min()) < 0 or int(starts.max()) + window_size > session_length):
        raise ValueError("window_starts contains an out-of-bounds window")
    return starts


def build_window_valid(
    eval_mask: np.ndarray,
    covariates: np.ndarray,
    trial_change: np.ndarray,
    window_starts: Sequence[int] | np.ndarray,
    window_size: int,
) -> torch.Tensor:
    """Return the deterministic bool ``[B, W]`` supervision mask.

    A position is legal exactly when it is evaluable, is not a still-time row,
    and has the same cumulative trial identity as the window's final position.
    The existing loader's pre-history rows are excluded by their False
    ``eval_mask`` values.  Every supplied start is treated as an admitted
    window, so a final position that is not legal is a contract violation.
    """
    evaluation, targets, boundaries = _session_arrays(
        eval_mask, covariates, trial_change
    )
    starts = _window_starts(
        window_starts,
        session_length=evaluation.shape[0],
        window_size=window_size,
    )
    if starts.size == 0:
        return torch.empty((0, window_size), dtype=torch.bool)

    still_time = np.all(np.abs(targets) < 0.001, axis=1)
    legal = evaluation & ~still_time
    trial_id = np.cumsum(boundaries.astype(np.int64, copy=False))
    offsets = np.arange(window_size, dtype=np.int64)
    positions = starts[:, None] + offsets[None, :]
    final_trial_id = trial_id[positions[:, -1]][:, None]
    valid = legal[positions] & (trial_id[positions] == final_trial_id)
    if not np.all(valid[:, -1]):
        bad_starts = starts[~valid[:, -1]].tolist()
        raise ValueError(
            "admitted windows must have a legal final position; "
            f"invalid starts={bad_starts}"
        )
    return torch.from_numpy(np.ascontiguousarray(valid, dtype=np.bool_))


def window_valid_digest(
    window_starts: Sequence[int] | np.ndarray,
    window_valid: torch.Tensor,
) -> str:
    """Hash masks in canonical start order, independent of sampling order."""
    starts = np.asarray(window_starts)
    if starts.ndim != 1 or not np.issubdtype(starts.dtype, np.integer):
        raise TypeError("window_starts must be a one-dimensional integer sequence")
    starts = starts.astype(np.int64, copy=False)
    if np.unique(starts).size != starts.size:
        raise ValueError("window_starts must not contain duplicates")
    if not isinstance(window_valid, torch.Tensor) or window_valid.dtype != torch.bool:
        raise TypeError("window_valid must be a torch.bool tensor")
    if window_valid.ndim != 2 or window_valid.shape[0] != starts.size:
        raise ValueError("window_valid must have shape [len(window_starts), W]")
    values = window_valid.detach().cpu().contiguous().numpy()
    order = np.argsort(starts, kind="stable")
    digest = hashlib.sha256(_DIGEST_SCHEMA)
    digest.update(struct.pack("<QQ", starts.size, values.shape[1]))
    for index in order:
        digest.update(struct.pack("<q", int(starts[index])))
        digest.update(values[index].tobytes(order="C"))
    return digest.hexdigest()


def masked_dense_aux_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_valid: torch.Tensor,
) -> torch.Tensor:
    """Compute the contract's channel-normalized masked dense MSE term."""
    if not isinstance(pred, torch.Tensor) or not isinstance(target, torch.Tensor):
        raise TypeError("pred and target must be torch tensors")
    if pred.shape != target.shape or pred.ndim != 3:
        raise ValueError("pred and target must have identical [B, W, C] shape")
    if not pred.is_floating_point() or not target.is_floating_point():
        raise TypeError("pred and target must be floating-point tensors")
    if not isinstance(window_valid, torch.Tensor) or window_valid.dtype != torch.bool:
        raise TypeError("window_valid must be a torch.bool tensor")
    if window_valid.shape != pred.shape[:2]:
        raise ValueError("window_valid must match pred's [B, W] dimensions")
    if window_valid.device != pred.device or target.device != pred.device:
        raise ValueError("pred, target, and window_valid must share one device")
    legal_count = int(window_valid.sum().item())
    if legal_count == 0 or pred.shape[-1] == 0:
        raise ValueError("masked dense MSE requires at least one legal scalar target")
    squared_error = ((pred - target) ** 2).sum(dim=-1)
    weights = window_valid.to(dtype=squared_error.dtype)
    return (squared_error * weights).sum() / (legal_count * pred.shape[-1])


def append_window_valid(sample: tuple[Any, ...], window_valid: torch.Tensor) -> tuple[Any, ...]:
    """Opt in to an appended per-sample bool mask without mutating old fields."""
    if not isinstance(sample, tuple):
        raise TypeError("sample must be the established tuple batch item")
    if not isinstance(window_valid, torch.Tensor) or window_valid.dtype != torch.bool:
        raise TypeError("window_valid must be a torch.bool tensor")
    if window_valid.ndim != 1:
        raise ValueError("per-sample window_valid must have shape [W]")
    return (*sample, window_valid.detach().clone())
