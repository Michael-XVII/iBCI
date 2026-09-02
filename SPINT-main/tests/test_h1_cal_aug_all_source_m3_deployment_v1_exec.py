from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import src.h1_cal_aug_all_source_m3_deployment_v1_exec as execution
from src.h1_cal_aug_all_source_m3_deployment_v1_exec import (
    AllSourceExecutionError,
    STATUS_PAIR,
    STATUS_SMOKE,
    STATUS_TERMINAL,
    _m7_schedule,
    _prefix_schedule,
    common_config,
    verify_pair,
)
from third_party.falcon_challenge.h1_carrier_id_spint_decoder import H1CarrierIdSpintDecoder


def test_all_source_m7_and_c1_schedules_are_deterministic_balanced_and_exclude_m3() -> None:
    records = {
        "ses-a": SimpleNamespace(trial_values=tuple(range(9))),
        "ses-b": SimpleNamespace(trial_values=tuple(range(10))),
    }
    windows = tuple(("ses-a" if index < 16 else "ses-b", index) for index in range(32))
    dataset = SimpleNamespace(records=records, windows=windows)
    order = np.arange(32, dtype=np.int64)
    first = _m7_schedule(dataset, order)
    second = _m7_schedule(dataset, order)
    assert np.array_equal(first, second) and first.shape == (50, 32)
    prefixes = _prefix_schedule(32)
    assert prefixes.shape == (50, 32) and set(np.unique(prefixes)) == {4, 5, 7}
    assert 3 not in prefixes
    for row in prefixes:
        counts = [int(np.sum(row == value)) for value in (7, 5, 4)]
        assert max(counts) - min(counts) <= 1


def test_common_config_is_exact_and_has_no_selection_or_warm_start() -> None:
    config = common_config()
    assert config["model_parameters"] == 10_947_836
    assert config["t0_identity_prefix"] == 7
    assert config["c1_identity_cycle"] == [7, 5, 4]
    assert config["training_carrier_trials"] == 4 and config["deployment_trials"] == 3
    assert config["epochs"] == 50 and config["checkpoint_epoch_zero_based"] == 49
    assert config["batch_size"] == 32 and config["prediction_divisor"] == 20.0
    assert config["validation_selection"] is config["early_stopping"] is config["warm_start"] is config["target_fitting"] is False


class _FakeTaskConfig:
    n_channels = 176

    @staticmethod
    def hash_dataset(stem: str) -> str:
        return stem


class _FakeModel:
    def __init__(self) -> None:
        self.kwargs = None

    def eval(self):
        return self

    def __call__(self, neural, **kwargs):
        self.kwargs = kwargs
        batch = neural.shape[0]
        return torch.full((batch, 5, 7), 40.0, dtype=torch.float32)


def test_carrier_decoder_reset_and_predict_pass_both_identity_and_carrier() -> None:
    decoder = H1CarrierIdSpintDecoder.__new__(H1CarrierIdSpintDecoder)
    decoder._task_config = _FakeTaskConfig()
    decoder.batch_size = 1
    decoder.window_size = 700
    decoder.prediction_divisor = 20.0
    decoder.device = torch.device("cpu")
    decoder.model = _FakeModel()
    decoder.sessions = {
        "S0_set_1": {
            "identity": np.ones((3, 1024, 176), np.float32),
            "carrier": np.ones((176, 4), np.float32),
        }
    }
    decoder.observation_buffer = np.zeros((700, 1, 176), np.float32)
    decoder.history_count = np.zeros(1, np.int64)
    decoder.local_identity = None
    decoder.local_carrier = None
    decoder.local_keys = ()
    decoder.reset([Path("S0_set_1")])
    prediction = decoder.predict(np.zeros((1, 176), np.float32))
    assert prediction.shape == (1, 7) and np.all(prediction == 2.0)
    assert tuple(decoder.model.kwargs["calib_trialized_neural_features"].shape) == (1, 3, 1024, 176)
    assert tuple(decoder.model.kwargs["carrier"].shape) == (1, 176, 4)
    assert decoder.history_count.tolist() == [1]


def _smoke_row() -> dict:
    return {
        "status": STATUS_SMOKE,
        "source_authority_sha256": "a", "schedule_sha256": "b",
        "batch_order_sha256": "c", "m7_schedule_sha256": "d",
        "source_tensor_surface_sha256": "e", "carrier_cache_sha256": "f",
        "normalized_carrier_sha256": "g", "common_config_sha256": "h",
        "initial_state_sha256": "i", "global_step": 12,
        "dropout_probability_sha256": "j", "dropout_probability_count": 12,
        "target_optimizer_steps": 0, "target_backward_steps": 0, "target_model_updates": 0,
        "heldin_minival_recordings_opened": 0, "heldout_calib_recordings_opened": 0,
        "evalai_test_recordings_opened": 0,
    }


def test_pair_gate_accepts_only_identical_registered_surfaces(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rows = {"t0": _smoke_row(), "c1": _smoke_row()}
    monkeypatch.setattr(execution, "_load_json", lambda path, schema=None: (rows[path.parent.name], path.parent.name + "-sha"))
    published = {}
    monkeypatch.setattr(execution, "publish_json", lambda path, body: published.update({str(path): body}) or "sha")
    body = verify_pair(tmp_path, smoke=True)
    assert body["status"] == STATUS_SMOKE
    assert body["initial_state_identical"] is body["carrier_bytes_matched"] is body["query_target_bytes_matched"] is True
    assert body["target_optimizer_steps"] == body["target_backward_steps"] == body["target_model_updates"] == 0
    rows["c1"] = {**rows["c1"], "dropout_probability_sha256": "changed"}
    with pytest.raises(AllSourceExecutionError, match="dropout_probability"):
        verify_pair(tmp_path, smoke=True)


def test_execution_runner_has_no_evalai_or_remote_test_phase() -> None:
    root = Path(__file__).resolve().parents[2]
    runner = root / "tfpd_exploration/h1_series_20260830/scripts/run_h1_cal_aug_all_source_m3_deployment_v1_exec.py"
    text = runner.read_text(encoding="utf-8")
    assert "--detached-supervisor" in text and "TQDM_DISABLE" in text
    for forbidden in ("evalai push", "--submit", "--remote-test", "phase=\"test\"", "phase='test'"):
        assert forbidden not in text


def test_terminal_status_is_local_stop_boundary() -> None:
    assert STATUS_PAIR == "PASS_H1_ALL_SOURCE_M3_DEPLOYMENT_PAIR_INTEGRITY"
    assert STATUS_TERMINAL == "COMPLETE_LOCAL_H1_ALL_SOURCE_M3_DEPLOYMENT_READY_NO_EVALAI_SUBMISSION"
