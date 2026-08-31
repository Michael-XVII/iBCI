"""CPU-only adversarial tests for the additive H1 window-mask contract."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
from torch.utils.data._utils.collate import default_collate

from src.data.h1_window_mask_contract_v1 import (
    append_window_valid,
    build_window_valid,
    masked_dense_aux_mse,
    window_valid_digest,
)


ACTIVE = np.asarray([0.25, -0.5], dtype=np.float32)


def _single_window(
    eval_mask: list[bool],
    covariates: list[list[float]],
    trial_change: list[bool],
) -> torch.Tensor:
    return build_window_valid(
        np.asarray(eval_mask, dtype=bool),
        np.asarray(covariates, dtype=np.float32),
        np.asarray(trial_change, dtype=bool),
        [0],
        len(eval_mask),
    )[0]


@pytest.mark.parametrize(
    ("eval_mask", "covariates", "trial_change", "expected"),
    [
        pytest.param(
            [True] * 5,
            [ACTIVE.tolist(), ACTIVE.tolist(), [0.0, 0.0], ACTIVE.tolist(), ACTIVE.tolist()],
            [True, False, False, False, False],
            [True, True, False, True, True],
            id="still-time",
        ),
        pytest.param(
            [True, True, False, True, True],
            [ACTIVE.tolist()] * 5,
            [True, False, False, False, False],
            [True, True, False, True, True],
            id="intertrial-gap",
        ),
        pytest.param(
            [False, False, False, False, True],
            [[0.0, 0.0]] * 4 + [ACTIVE.tolist()],
            [False, False, False, False, True],
            [False, False, False, False, True],
            id="short-first-trial-padding",
        ),
        pytest.param(
            [True] * 5,
            [ACTIVE.tolist()] * 5,
            [True, False, True, False, False],
            [False, False, True, True, True],
            id="trial-change-mid-window",
        ),
    ],
)
def test_hand_computed_adversarial_masks(
    eval_mask: list[bool],
    covariates: list[list[float]],
    trial_change: list[bool],
    expected: list[bool],
):
    actual = _single_window(eval_mask, covariates, trial_change)
    assert actual.dtype == torch.bool
    assert actual.tolist() == expected
    assert actual[-1].item() is True


def test_real_h1_width_700_preserves_padding_and_final_admission_invariant():
    window_size = 700
    eval_mask = np.zeros(704, dtype=bool)
    eval_mask[699:] = True
    covariates = np.zeros((704, 7), dtype=np.float32)
    covariates[699:] = 0.25
    trial_change = np.zeros(704, dtype=bool)
    trial_change[699] = True

    valid = build_window_valid(
        eval_mask, covariates, trial_change, [0, 4], window_size
    )

    assert valid.shape == (2, 700)
    assert valid.dtype == torch.bool
    assert valid[0, :-1].sum().item() == 0
    assert valid[0, -1].item() is True
    assert valid[1, :695].sum().item() == 0
    assert valid[1, 695:].all().item() is True
    assert valid[:, -1].all().item() is True


def test_mask_and_digest_are_invariant_to_window_sampling_order():
    eval_mask = np.ones(8, dtype=bool)
    covariates = np.full((8, 2), 0.25, dtype=np.float32)
    trial_change = np.asarray([True, False, False, False, True, False, False, False])
    starts = np.asarray([0, 1, 3], dtype=np.int64)
    permuted = starts[[2, 0, 1]]

    ordered_mask = build_window_valid(eval_mask, covariates, trial_change, starts, 4)
    permuted_mask = build_window_valid(eval_mask, covariates, trial_change, permuted, 4)
    restored = permuted_mask[torch.tensor([1, 2, 0])]

    assert torch.equal(ordered_mask, restored)
    assert window_valid_digest(starts, ordered_mask) == window_valid_digest(
        permuted, permuted_mask
    )


def test_mask_and_session_digest_are_deterministic():
    eval_mask = np.ones(7, dtype=bool)
    covariates = np.full((7, 3), 0.5, dtype=np.float32)
    covariates[2] = 0.0
    trial_change = np.asarray([True, False, False, True, False, False, False])
    starts = [0, 2]

    first = build_window_valid(eval_mask, covariates, trial_change, starts, 5)
    second = build_window_valid(
        eval_mask.copy(), covariates.copy(), trial_change.copy(), list(starts), 5
    )

    assert torch.equal(first, second)
    assert window_valid_digest(starts, first) == window_valid_digest(starts, second)


def test_auxiliary_loss_matches_dense_and_last_bin_boundaries():
    pred = torch.tensor(
        [[[1.0, 2.0], [2.0, 0.0], [3.0, -1.0]],
         [[0.0, 1.0], [1.0, 4.0], [2.0, 2.0]]],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [[[0.0, 0.0], [1.0, 1.0], [2.0, 1.0]],
         [[1.0, 1.0], [1.0, 2.0], [0.0, 2.0]]],
        dtype=torch.float32,
    )
    all_true = torch.ones(pred.shape[:2], dtype=torch.bool)
    last_only = torch.zeros(pred.shape[:2], dtype=torch.bool)
    last_only[:, -1] = True

    dense_aux = masked_dense_aux_mse(pred, target, all_true)
    last_aux = masked_dense_aux_mse(pred, target, last_only)
    naive_dense = torch.mean((pred - target) ** 2)
    governing_last = torch.mean((pred[:, -1:, :] - target[:, -1:, :]) ** 2)

    assert torch.equal(dense_aux, naive_dense)
    assert torch.equal(last_aux, governing_last)
    lam_zero_total = governing_last + 0.0 * dense_aux
    assert torch.equal(lam_zero_total, governing_last)
    lam = 0.3
    last_only_total = governing_last + lam * last_aux
    assert torch.allclose(last_only_total, (1.0 + lam) * governing_last)


def _tensor_digest(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(repr(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def test_opt_in_collation_preserves_existing_batch_field_bytes_and_order():
    old_samples = [
        (
            np.full((5, 3), index + 0.25, dtype=np.float32),
            np.full((5, 2), index - 0.5, dtype=np.float32),
            np.full((2, 4, 3), index + 1.0, dtype=np.float32),
            f"session-{index}",
        )
        for index in range(2)
    ]
    masks = [
        torch.tensor([False, False, True, True, True]),
        torch.tensor([False, True, False, True, True]),
    ]
    extended_samples = [
        append_window_valid(sample, mask)
        for sample, mask in zip(old_samples, masks, strict=True)
    ]

    old_batch = default_collate(old_samples)
    extended_batch = default_collate(extended_samples)

    assert len(old_batch) == 4
    assert len(extended_batch) == 5
    for old_field, extended_field in zip(old_batch[:3], extended_batch[:3], strict=True):
        assert _tensor_digest(old_field) == _tensor_digest(extended_field)
        assert torch.equal(old_field, extended_field)
    assert old_batch[3] == extended_batch[3]
    assert extended_batch[4].shape == (2, 5)
    assert extended_batch[4].dtype == torch.bool


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"window_starts": [0, 0]}, "duplicates"),
        ({"window_starts": [-1]}, "out-of-bounds"),
        ({"window_starts": [0], "eval_mask": [True, True, True, True, False]}, "legal final"),
    ],
)
def test_contract_rejects_non_admitted_or_ambiguous_windows(kwargs, match):
    values = {
        "eval_mask": [True] * 5,
        "covariates": [ACTIVE.tolist()] * 5,
        "trial_change": [True, False, False, False, False],
        "window_starts": [0],
        "window_size": 5,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        build_window_valid(
            np.asarray(values["eval_mask"], dtype=bool),
            np.asarray(values["covariates"], dtype=np.float32),
            np.asarray(values["trial_change"], dtype=bool),
            values["window_starts"],
            values["window_size"],
        )


def test_cpu_contract_suite_does_not_initialize_cuda():
    assert torch.cuda.is_initialized() is False
