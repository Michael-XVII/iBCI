"""CPU/no-data tests for the additive H1 masked dense-auxiliary cell."""
from __future__ import annotations

from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data._utils.collate import default_collate

from src.data.h1_masked_dense_aux_v1 import (
    OUTER_DATE,
    SOURCE_DATES,
    SOURCE_SESSIONS,
    TARGET_SESSIONS,
    WindowValidDataset,
    H1MaskedDenseAuxDataModule,
    session_date,
    source_date_split,
)
from src.models.h1_masked_dense_aux_v1 import H1MaskedDenseAuxLitModule
from src.h1_masked_dense_aux_protocol_v1 import (
    outer_gate,
    select_source_lambda,
    verify_immutable,
    write_immutable_json,
)


class _SyntheticFalconDataset:
    window_size = 5

    def __init__(self) -> None:
        self.window_indices = [("ses-19250108T110520", 0), ("ses-19250108T110520", 2)]
        self.neural_data = {
            "ses-19250108T110520": np.arange(24, dtype=np.float32).reshape(8, 3)
        }
        self.covariate_data = {
            "ses-19250108T110520": np.asarray(
                [[0.2, -0.1], [0.2, -0.1], [0.0, 0.0], [0.3, 0.1],
                 [0.3, 0.1], [0.4, 0.1], [0.4, 0.1], [0.4, 0.1]],
                dtype=np.float32,
            )
        }
        self.eval_mask = {"ses-19250108T110520": np.ones(8, dtype=bool)}
        self.trial_change = {
            "ses-19250108T110520": np.asarray(
                [True, False, False, True, False, False, False, False], dtype=bool
            )
        }

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(self, index: int):
        session, start = self.window_indices[index]
        stop = start + self.window_size
        return (
            self.neural_data[session][start:stop],
            self.covariate_data[session][start:stop],
            np.ones((2, 5, 3), dtype=np.float32),
            session,
        )


def test_wrapper_is_opt_in_and_preserves_all_old_collated_fields():
    old = _SyntheticFalconDataset()
    wrapped = WindowValidDataset(old)
    old_batch = default_collate([old[0], old[1]])
    new_batch = default_collate([wrapped[0], wrapped[1]])
    assert len(old_batch) == 4
    assert len(new_batch) == 5
    for old_field, new_field in zip(old_batch[:3], new_batch[:3], strict=True):
        assert torch.equal(old_field, new_field)
        assert old_field.numpy().tobytes() == new_field.numpy().tobytes()
    assert old_batch[3] == new_batch[3]
    assert new_batch[4].shape == (2, 5)
    assert new_batch[4].dtype == torch.bool
    assert new_batch[4][:, -1].all().item() is True
    audit = wrapped.mask_audit["ses-19250108T110520"]
    assert audit["windows"] == 2
    assert audit["final_all_true"] is True
    assert len(audit["window_valid_sha256"]) == 64


def test_frozen_source_date_split_is_grouped_and_excludes_outer_target():
    assert OUTER_DATE not in SOURCE_DATES
    assert not set(TARGET_SESSIONS) & set(SOURCE_SESSIONS)
    for date in SOURCE_DATES:
        train, validation = source_date_split(SOURCE_SESSIONS, date)
        assert validation
        assert all(session_date(name) == date for name in validation)
        assert all(session_date(name) != date for name in train)
        assert set(train) | set(validation) == set(SOURCE_SESSIONS)
        assert not set(train) & set(validation)
    train, validation = source_date_split(SOURCE_SESSIONS, None)
    assert tuple(train) == SOURCE_SESSIONS
    assert tuple(validation) == SOURCE_SESSIONS


def test_source_split_rejects_roster_drift_and_bad_dates():
    with pytest.raises(ValueError, match="exactly the frozen"):
        source_date_split(SOURCE_SESSIONS[:-1], SOURCE_DATES[0])
    with pytest.raises(ValueError, match="validation_date"):
        source_date_split(SOURCE_SESSIONS, OUTER_DATE)
    with pytest.raises(ValueError, match="invalid H1 session"):
        session_date("held-out-calib")


class _FixedDenseNet(torch.nn.Module):
    def __init__(self, prediction_target_scale: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("prediction_target_scale", prediction_target_scale)
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.calibration_seen = None

    def forward(self, x, calib_trialized_neural_features=None):
        self.calibration_seen = calib_trialized_neural_features
        return self.prediction_target_scale.expand(x.shape[0], -1, -1) + self.anchor * 0.0


def _module(dense_aux_lambda: float) -> H1MaskedDenseAuxLitModule:
    prediction = torch.tensor(
        [[[20.0, 0.0], [40.0, -20.0], [60.0, 20.0]]], dtype=torch.float32
    )
    return H1MaskedDenseAuxLitModule(
        task="h1", net=_FixedDenseNet(prediction), decode_last_timestep_only=True,
        predict_scaled_behavior=True, behavior_scaling_factor=20.0,
        optimizer=partial(torch.optim.Adam, lr=5e-5, weight_decay=0.0),
        scheduler=None, compile=False, dense_aux_lambda=dense_aux_lambda,
    )


def _batch(mask: torch.Tensor):
    neural = torch.zeros((2, 3, 4), dtype=torch.float32)
    target = torch.tensor(
        [[[0.0, 0.0], [1.0, 1.0], [2.0, -1.0]],
         [[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]]], dtype=torch.float32,
    )
    calibration = torch.ones((2, 2, 3, 4), dtype=torch.float32)
    sessions = ["ses-19250108T110520", "ses-19250108T110520"]
    return neural, target, calibration, sessions, mask


def test_lambda_zero_is_exact_last_bin_loss_and_validation_never_applies_aux():
    mask = torch.tensor([[True, True, True], [False, True, True]], dtype=torch.bool)
    module = _module(0.0)
    out = module.loss_step(_batch(mask), apply_aux=True)
    expected = torch.mean((out["behavior_pred"] - out["behavior_target"]) ** 2)
    assert torch.equal(out["loss"], expected)
    assert torch.equal(out["governing_loss"], expected)
    assert out["auxiliary_loss"].item() > 0
    assert module.net.calibration_seen is not None
    validation = module.loss_step(_batch(mask), apply_aux=False)
    assert torch.equal(validation["loss"], validation["governing_loss"])


def test_positive_lambda_obeys_registered_formula_and_last_bin_interface():
    mask = torch.tensor([[True, True, True], [False, True, True]], dtype=torch.bool)
    module = _module(0.3)
    out = module.loss_step(_batch(mask), apply_aux=True)
    assert torch.allclose(out["loss"], out["governing_loss"] + 0.3 * out["auxiliary_loss"])
    assert out["behavior_pred"].shape == (2, 1, 2)
    assert out["behavior_target"].shape == (2, 1, 2)


@pytest.mark.parametrize("value", [-1.0, 0.2, 2.0])
def test_dense_aux_lambda_is_fail_closed(value):
    with pytest.raises(ValueError, match="dense_aux_lambda"):
        _module(value)


def test_cpu_no_data_suite_does_not_initialize_cuda():
    assert torch.cuda.is_initialized() is False


def test_receipt_lifecycle_is_atomic_immutable_and_refuses_overwrite(tmp_path):
    path = tmp_path / "attempt.json"
    _, digest = write_immutable_json(path, {"schema": "test", "status": "ATTEMPT"})
    assert verify_immutable(path) == digest
    assert path.stat().st_mode & 0o222 == 0
    assert (tmp_path / "attempt.json.sha256").stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError, match="already exists"):
        write_immutable_json(path, {"schema": "test", "status": "RETRY"})


def test_source_and_outer_gates_follow_preregistered_boundaries():
    rows = []
    for date_index, date in enumerate(SOURCE_DATES):
        for lam in (0.0, 0.1, 0.3, 1.0):
            improvement = 0.0 if lam == 0 else {0.1: 0.011, 0.3: 0.02, 1.0: 0.015}[lam]
            rows.append({"validation_date": date, "lambda": lam, "r2_mean": date_index + improvement})
    selection = select_source_lambda(rows)
    assert selection["selected_lambda"] == 0.3
    assert selection["source_gate_passed"] is True
    outer = outer_gate({"a": 0.2, "b": 0.3}, {"a": 0.211, "b": 0.312})
    assert outer["outer_gate_passed"] is True


def test_datamodule_preserves_path_typed_data_root(tmp_path):
    dm = H1MaskedDenseAuxDataModule(
        task="h1", data_dir=tmp_path, allowed_sessions=SOURCE_SESSIONS,
        validation_date=SOURCE_DATES[0], num_workers=0,
    )
    assert isinstance(dm.hparams.data_dir, Path)
    assert dm.hparams.data_dir == tmp_path
