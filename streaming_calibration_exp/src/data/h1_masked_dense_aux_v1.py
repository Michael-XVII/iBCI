"""Additive H1 source-date data route with typed window-valid masks."""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.falcon_datamodule import FalconDataModule
from src.data.h1_window_mask_contract_v1 import (
    append_window_valid,
    build_window_valid,
    window_valid_digest,
)


OUTER_DATE = "19250101"
SOURCE_DATES = ("19250108", "19250113", "19250115", "19250119", "19250120")
TARGET_SESSIONS = (
    "ses-19250101T111740",
    "ses-19250101T112404",
)
SOURCE_SESSIONS = (
    "ses-19250108T110520",
    "ses-19250108T111022",
    "ses-19250108T111455",
    "ses-19250113T120811",
    "ses-19250113T121303",
    "ses-19250115T110633",
    "ses-19250115T111328",
    "ses-19250119T113543",
    "ses-19250119T114045",
    "ses-19250120T115044",
    "ses-19250120T115537",
)
_SESSION_RE = re.compile(r"^ses-(\d{8})T")


def session_date(session_name: str) -> str:
    match = _SESSION_RE.match(str(session_name))
    if match is None:
        raise ValueError(f"invalid H1 session name: {session_name!r}")
    return match.group(1)


def source_date_split(
    session_names: Sequence[str], validation_date: str | None
) -> tuple[list[str], list[str]]:
    observed = tuple(sorted(session_names))
    if observed != tuple(sorted(SOURCE_SESSIONS)):
        raise ValueError("source-date split requires exactly the frozen 11 source sessions")
    if validation_date is None:
        return list(SOURCE_SESSIONS), list(SOURCE_SESSIONS)
    if validation_date not in SOURCE_DATES:
        raise ValueError(f"validation_date must be one of {SOURCE_DATES}")
    validation = [name for name in SOURCE_SESSIONS if session_date(name) == validation_date]
    train = [name for name in SOURCE_SESSIONS if session_date(name) != validation_date]
    if not validation or set(train) & set(validation):
        raise ValueError("invalid grouped source-date split")
    return train, validation


def _array_digest(values: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


class WindowValidDataset(Dataset):
    """Opt-in wrapper that appends a precomputed bool mask to a Falcon sample."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        starts_by_session: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for dataset_index, (session, start) in enumerate(dataset.window_indices):
            starts_by_session[str(session)].append((dataset_index, int(start)))
        self._masks: list[torch.Tensor | None] = [None] * len(dataset.window_indices)
        self.mask_audit: dict[str, dict[str, Any]] = {}
        for session, rows in sorted(starts_by_session.items()):
            starts = np.asarray([start for _, start in rows], dtype=np.int64)
            masks = build_window_valid(
                dataset.eval_mask[session],
                dataset.covariate_data[session],
                dataset.trial_change[session],
                starts,
                int(dataset.window_size),
            )
            for (dataset_index, _), mask in zip(rows, masks, strict=True):
                self._masks[dataset_index] = mask
            counts = masks.sum(dim=1).cpu().numpy().astype(np.int64)
            positions = starts[:, None] + np.arange(dataset.window_size)[None, :]
            padded = positions < int(dataset.window_size) - 1
            still = np.all(np.abs(dataset.covariate_data[session]) < 0.001, axis=1)
            intertrial = ~np.asarray(dataset.eval_mask[session], dtype=bool)
            self.mask_audit[session] = {
                "windows": int(starts.size),
                "window_size": int(dataset.window_size),
                "legal_count_min": int(counts.min()) if counts.size else None,
                "legal_count_max": int(counts.max()) if counts.size else None,
                "legal_count_mean": float(counts.mean()) if counts.size else None,
                "legal_fraction_mean": float(counts.mean() / dataset.window_size) if counts.size else None,
                "final_all_true": bool(masks[:, -1].all().item()) if starts.size else True,
                "padded_positions_observed": int(padded.sum()),
                "padded_positions_legal": int(masks.cpu().numpy()[padded].sum()),
                "still_positions_observed": int(still[positions].sum()),
                "still_positions_legal": int(masks.cpu().numpy()[still[positions]].sum()),
                "intertrial_positions_observed": int(intertrial[positions].sum()),
                "intertrial_positions_legal": int(masks.cpu().numpy()[intertrial[positions]].sum()),
                "window_start_sha256": _array_digest([starts]),
                "window_valid_sha256": window_valid_digest(starts, masks),
                "existing_session_arrays_sha256": _array_digest([
                    dataset.neural_data[session],
                    dataset.covariate_data[session],
                    dataset.eval_mask[session],
                    dataset.trial_change[session],
                ]),
            }
        if any(mask is None for mask in self._masks):
            raise RuntimeError("window mask precomputation did not cover every dataset item")

    @property
    def window_indices(self):
        return self.dataset.window_indices

    def __getattr__(self, name: str):
        if name in {"dataset", "_masks", "mask_audit"}:
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        mask = self._masks[index]
        assert mask is not None
        return append_window_valid(self.dataset[index], mask)


class H1MaskedDenseAuxDataModule(FalconDataModule):
    """Source-bounded H1 DataModule that leaves the established module untouched."""

    def __init__(
        self,
        *args,
        validation_date: str | None = None,
        allowed_sessions: Sequence[str] = SOURCE_SESSIONS,
        **kwargs,
    ) -> None:
        allowed = tuple(allowed_sessions)
        if allowed not in {SOURCE_SESSIONS, TARGET_SESSIONS}:
            raise ValueError("allowed_sessions must be the frozen source or fold-0 target roster")
        if allowed == TARGET_SESSIONS and validation_date is not None:
            raise ValueError("outer target evaluation cannot request source validation")
        self.validation_date = validation_date
        self.allowed_sessions = allowed
        kwargs["heldin_session_names"] = list(allowed)
        kwargs["validation_protocol"] = "minival"
        kwargs["include_heldout_in_fit"] = False
        kwargs["include_heldout_in_test"] = False
        kwargs["num_workers"] = 0
        super().__init__(*args, **kwargs)

    def _resolve_train_val_sessions(self, all_heldin_sessions):
        observed = tuple(sorted(all_heldin_sessions))
        if observed != tuple(sorted(self.allowed_sessions)):
            raise ValueError(
                f"loaded H1 roster drift: expected={self.allowed_sessions}, observed={observed}"
            )
        if self.allowed_sessions == TARGET_SESSIONS:
            return list(TARGET_SESSIONS), list(TARGET_SESSIONS)
        return source_date_split(observed, self.validation_date)

    def setup(self, stage: str | None = None) -> None:
        super().setup(stage)
        if not isinstance(self.train_dataset, WindowValidDataset):
            self.train_dataset = WindowValidDataset(self.train_dataset)
        if not isinstance(self.val_heldin_dataset, WindowValidDataset):
            self.val_heldin_dataset = WindowValidDataset(self.val_heldin_dataset)
        if getattr(self, "val_heldout_dataset", None) is not None:
            raise RuntimeError("masked-dense H1 route must never construct formal held-out data")

    def window_mask_audit(self) -> dict[str, Any]:
        return {
            "allowed_sessions": list(self.allowed_sessions),
            "validation_date": self.validation_date,
            "train_sessions": list(self.train_session_names),
            "validation_sessions": list(self.val_heldin_session_names),
            "train": dict(self.train_dataset.mask_audit),
            "validation": dict(self.val_heldin_dataset.mask_audit),
            "formal_heldout_constructed": False,
        }
