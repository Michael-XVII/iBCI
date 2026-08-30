"""Shared, model-agnostic rules for the M1/H1 activity-headroom experiment."""
from __future__ import annotations

from enum import Enum
import hashlib
import json
from typing import Any, Sequence

import numpy as np


class ActivityHeadroomError(RuntimeError):
    """Fail closed on an activity intervention or metric contract drift."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ActivityHeadroomError(message)


class ActivityArm(str, Enum):
    STATIC_SUPPORT = "STATIC_SUPPORT"
    ROLLING_FIXED_M = "ROLLING_FIXED_M"
    CAUSAL_GROWING_CAP30 = "CAUSAL_GROWING_CAP30"
    FULL_SESSION_ORACLE = "FULL_SESSION_ORACLE"


ARM_ORDER = tuple(ActivityArm)


def selection_for_output_trial(
    arm: ActivityArm,
    *,
    output_trial_index: int,
    total_trials: int,
    support_trials: int,
    growing_cap: int = 30,
) -> tuple[int, ...]:
    """Return activity-trial indices available at one output trial.

    ``output_trial_index`` is zero based.  A causal arm may use only indices
    strictly smaller than it.  The frozen support is retained while fewer than
    ``support_trials`` have completed, matching the existing deployment
    contract rather than inventing a partially calibrated model state.
    """

    _need(isinstance(arm, ActivityArm), "activity arm type drift")
    _need(type(output_trial_index) is int and output_trial_index >= -1,
          "output trial index drift")
    _need(type(total_trials) is int and total_trials >= 1,
          "total trial count drift")
    _need(type(support_trials) is int and 1 <= support_trials <= total_trials,
          "support trial count drift")
    _need(type(growing_cap) is int and growing_cap >= support_trials,
          "growing cap drift")

    if arm is ActivityArm.FULL_SESSION_ORACLE:
        return tuple(range(total_trials))
    if arm is ActivityArm.STATIC_SUPPORT or output_trial_index < support_trials:
        return tuple(range(support_trials))
    stop = min(output_trial_index, total_trials)
    if arm is ActivityArm.ROLLING_FIXED_M:
        start = max(0, stop - support_trials)
    else:
        start = max(0, stop - growing_cap)
    selected = tuple(range(start, stop))
    _need(len(selected) >= support_trials, "causal activity selection lost support cardinality")
    _need(all(index < output_trial_index for index in selected),
          "causal activity selection read current/future trial")
    return selected


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")) + "\n").encode("utf-8")


def array_digest(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    _need(array.ndim >= 1 and np.isfinite(array).all(), "array digest needs finite non-scalar")
    header = json.dumps(
        {"dtype": str(array.dtype), "shape": list(array.shape)},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def variance_weighted_r2(prediction: Any, target: Any) -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    _need(pred.ndim == truth.ndim == 2 and pred.shape == truth.shape and pred.shape[0] > 0,
          "R2 input topology drift")
    _need(np.isfinite(pred).all() and np.isfinite(truth).all(), "R2 input is nonfinite")
    sse = float(np.square(truth - pred).sum(dtype=np.float64))
    centered = truth - truth.mean(axis=0, keepdims=True, dtype=np.float64)
    tss = float(np.square(centered).sum(dtype=np.float64))
    _need(np.isfinite(sse) and np.isfinite(tss) and tss > 0.0, "R2 is undefined")
    return float(1.0 - sse / tss)


def grouped_indices(selections: Sequence[tuple[int, ...]]) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    groups: dict[tuple[int, ...], list[int]] = {}
    for row, selection in enumerate(selections):
        _need(isinstance(selection, tuple) and selection, "empty activity selection")
        groups.setdefault(selection, []).append(row)
    return tuple((selection, tuple(rows)) for selection, rows in groups.items())


def encode_trial_activity(net: Any, trial_activity: Any, *, family: str, device: str, chunk_size: int = 16) -> Any:
    """Apply the frozen pre-pool map once per trial.

    The returned tensor is [trials, units, hidden].  Pooling a chosen subset
    later is algebraically identical to the production per-batch identity path.
    """

    import torch

    trials = np.ascontiguousarray(np.asarray(trial_activity), dtype=np.float32)
    _need(trials.ndim == 3 and trials.shape[0] > 0 and np.isfinite(trials).all(),
          "trial activity tensor drift")
    _need(family in {"m1", "h1"} and type(chunk_size) is int and chunk_size > 0,
          "trial encoder contract drift")
    outputs = []
    with torch.no_grad():
        for start in range(0, trials.shape[0], chunk_size):
            value = torch.as_tensor(trials[start:start + chunk_size], dtype=torch.float32, device=device)
            temporal = value.permute(0, 2, 1)
            encoded = net.fc_id_in(temporal) if family == "m1" else net.carrier_pre_pool(temporal)
            _need(encoded.ndim == 3 and bool(torch.isfinite(encoded).all()),
                  "encoded activity tensor drift")
            outputs.append(encoded)
    return torch.cat(outputs, dim=0)


def identity_from_encoded_trials(
    net: Any,
    encoded_trials: Any,
    selection: tuple[int, ...],
    *,
    family: str,
    carrier: Any | None = None,
) -> Any:
    import torch

    _need(family in {"m1", "h1"} and selection and len(set(selection)) == len(selection),
          "identity selection drift")
    index = torch.as_tensor(selection, dtype=torch.long, device=encoded_trials.device)
    pooled = encoded_trials.index_select(0, index).mean(dim=0)
    if family == "m1":
        identity = net.fc_id_out(pooled)
    else:
        _need(carrier is not None, "H1 identity requires the fixed carrier")
        fixed = torch.as_tensor(carrier, dtype=pooled.dtype, device=pooled.device)
        _need(tuple(fixed.shape) == (pooled.shape[0], net.carrier_dim), "H1 fixed carrier shape drift")
        effective = torch.zeros_like(fixed) if net.zero_carrier else fixed
        identity = net.carrier_post_pool(torch.cat((pooled, effective), dim=-1))
    _need(identity.ndim == 2 and bool(torch.isfinite(identity).all()), "identity tensor drift")
    return identity


def identity_from_raw_trials(
    net: Any,
    trial_activity: Any,
    selection: tuple[int, ...],
    *,
    family: str,
    device: str,
    carrier: Any | None = None,
) -> Any:
    """Run the production pre-pool graph once for one complete activity state.

    Unlike per-trial embedding caching, this preserves the production MLP's
    selected-trial batching/reduction order.  Only the redundant repetition of
    the same identity across evaluation rows is removed.
    """
    import torch

    trials = np.ascontiguousarray(np.asarray(trial_activity), dtype=np.float32)
    _need(trials.ndim == 3 and selection and len(set(selection)) == len(selection)
          and min(selection) >= 0 and max(selection) < trials.shape[0]
          and family in {"m1", "h1"},
          "raw activity-state identity contract drift")
    selected = torch.as_tensor(
        np.ascontiguousarray(trials[np.asarray(selection, dtype=np.int64)]),
        dtype=torch.float32, device=device,
    ).unsqueeze(0)
    temporal = selected.permute(0, 1, 3, 2)
    with torch.no_grad():
        if family == "m1":
            pooled = net.fc_id_in(temporal).mean(dim=1)[0]
            identity = net.fc_id_out(pooled)
        else:
            _need(carrier is not None, "H1 raw activity identity requires fixed carrier")
            pooled = net.carrier_pre_pool(temporal).mean(dim=1)[0]
            fixed = torch.as_tensor(carrier, dtype=pooled.dtype, device=pooled.device)
            _need(tuple(fixed.shape) == (pooled.shape[0], net.carrier_dim),
                  "H1 raw activity fixed carrier shape drift")
            effective = torch.zeros_like(fixed) if net.zero_carrier else fixed
            identity = net.carrier_post_pool(torch.cat((pooled, effective), dim=-1))
    _need(identity.ndim == 2 and bool(torch.isfinite(identity).all()),
          "raw activity-state identity tensor drift")
    return identity


def forward_with_cached_identity(net: Any, x: Any, identity: Any) -> Any:
    """Run the unchanged eval graph after its deterministic identity projection."""

    import torch

    _need(net.training is False, "cached-identity forward requires eval mode")
    src = torch.as_tensor(x, dtype=torch.float32, device=identity.device)
    _need(src.ndim == 3 and identity.ndim == 2
          and src.shape[2] == identity.shape[0] and src.shape[1] == identity.shape[1],
          "cached-identity forward input shape drift")
    src = src.permute(0, 2, 1) + identity.unsqueeze(0)
    src = net.fc_in(src)
    rep = net.fc_in(net.rep).to(src)
    transformed, _ = net.transformer(rep.repeat(src.size(0), 1, 1), src)
    output = net.fc_out(transformed).permute(0, 2, 1)
    _need(output.ndim == 3 and bool(torch.isfinite(output).all()), "cached-identity output drift")
    return output
