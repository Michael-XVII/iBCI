"""CPU/no-data tests for the final-legal H1 masked dense V2 route."""
from __future__ import annotations

from functools import partial

import numpy as np
import pytest
import torch
from torch.utils.data._utils.collate import default_collate

from src.data.h1_masked_dense_aux_v2 import FinalLegalWindowDataset
from src.h1_masked_dense_aux_v2_protocol import evaluate_attrition
from src.models.h1_masked_dense_aux_v2 import H1MaskedDenseAuxV2LitModule


SESSION = "ses-19250108T110520"


class _Dataset:
    window_size = 4

    def __init__(self):
        self.window_indices = [(SESSION, 0), (SESSION, 1), (SESSION, 2)]
        self.neural_data = {SESSION: np.arange(24, dtype=np.float32).reshape(8, 3)}
        self.covariate_data = {SESSION: np.asarray([
            [0.2, 0.1], [0.2, 0.1], [0.2, 0.1], [0.3, 0.1],
            [0.0, 0.0], [0.4, 0.1], [0.4, 0.1], [0.4, 0.1],
        ], dtype=np.float32)}
        self.eval_mask = {SESSION: np.ones(8, dtype=bool)}
        self.trial_change = {SESSION: np.asarray(
            [True, False, False, False, False, False, False, False], dtype=bool
        )}

    def __len__(self):
        return len(self.window_indices)

    def __getitem__(self, index):
        session, start = self.window_indices[index]
        stop = start + self.window_size
        return (
            self.neural_data[session][start:stop],
            self.covariate_data[session][start:stop],
            np.ones((2, 4, 3), dtype=np.float32), session,
        )


def test_final_legal_subset_filters_only_invalid_endpoint_and_preserves_old_fields():
    base = _Dataset()
    subset = FinalLegalWindowDataset(base)
    assert subset.window_indices == [(SESSION, 0), (SESSION, 2)]
    assert subset.kept_original_indices == (0, 2)
    for public_index, original_index in enumerate((0, 2)):
        old = default_collate([base[original_index]])
        new = default_collate([subset[public_index]])
        assert len(new) == 5
        for old_field, new_field in zip(old[:3], new[:3], strict=True):
            assert torch.equal(old_field, new_field)
            assert old_field.numpy().tobytes() == new_field.numpy().tobytes()
        assert old[3] == new[3]
        assert new[4].dtype == torch.bool
        assert new[4][:, -1].all().item() is True
    audit = subset.mask_audit[SESSION]
    assert audit["original_windows"] == 3
    assert audit["retained_windows"] == 2
    assert audit["excluded_final_still_windows"] == 1
    assert audit["retention_fraction"] == pytest.approx(2 / 3)
    assert audit["trials_losing_all_windows"] == []


def test_attrition_gate_requires_retention_trial_coverage_and_legal_finals():
    passing = {SESSION: {
        "retention_fraction": 0.5, "retained_windows": 10,
        "trials_losing_all_windows": [], "final_all_true": True,
    }}
    assert evaluate_attrition(passing)["attrition_gate_passed"] is True
    for change in (
        {"retention_fraction": 0.24},
        {"retained_windows": 0},
        {"trials_losing_all_windows": [2]},
        {"final_all_true": False},
    ):
        row = dict(passing[SESSION])
        row.update(change)
        assert evaluate_attrition({SESSION: row})["attrition_gate_passed"] is False


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x, calib_trialized_neural_features=None):
        return x[..., :2] * 20.0 + self.anchor * 0.0


def test_v2_training_requires_mask_but_validation_preserves_four_field_metric_domain():
    module = H1MaskedDenseAuxV2LitModule(
        task="h1", net=_Net(), decode_last_timestep_only=True,
        predict_scaled_behavior=True, behavior_scaling_factor=20.0,
        optimizer=partial(torch.optim.Adam, lr=5e-5), scheduler=None,
        compile=False, dense_aux_lambda=0.3,
    )
    neural = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10
    target = neural[..., :2].clone()
    calibration = torch.ones((2, 2, 3, 4))
    sessions = [SESSION, SESSION]
    mask = torch.ones((2, 3), dtype=torch.bool)
    train = module.loss_step((neural, target, calibration, sessions, mask), apply_aux=True)
    assert train["loss"].item() == 0.0
    loss, prediction, expected, observed_sessions = module.legacy_validation_outputs(
        (neural, target, calibration, sessions)
    )
    assert loss.item() == 0.0
    assert prediction.shape == expected.shape == (2, 1, 2)
    assert observed_sessions == sessions
    with pytest.raises(ValueError, match="four-field"):
        module.legacy_validation_outputs((neural, target, calibration, sessions, mask))


def test_v2_cpu_suite_does_not_initialize_cuda():
    assert torch.cuda.is_initialized() is False
