from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.data.h1_m4_cce_date_lodo import target_sessions_for_date
import src.h1_cal_aug_prefix_cycle_m3_transfer_v1 as protocol
from src.h1_cal_aug_prefix_cycle_m3_transfer_v1 import (
    ARMS,
    M3TransferProtocolError,
    aggregate_m3_results,
    dry_plan,
    m3_causal_surface,
    m3_support_and_query_trial,
    materialize_m3_calibration,
    ordered_legal_trials,
    scale_last_bin_prediction,
    score_m3,
)
from src.h1_m4_cce_contract import CONFIRMATORY_DATES


def test_dry_plan_is_review_only_and_has_no_execution_surface() -> None:
    plan = dry_plan()
    assert plan["status"] == "DRY_REVIEW_ONLY_NO_WRITE_NO_DATA_NO_CUDA"
    assert tuple(plan["outer_dates"]) == CONFIRMATORY_DATES
    assert tuple(plan["arms"]) == ARMS
    assert plan["calibration_trials"] == 3
    assert plan["first_scoring_trial_ordinal"] == 4
    assert plan["window_bins"] == 700 and plan["prediction_divisor"] == 20.0
    assert plan["score_dtype"] == "float64" and plan["outputs"] == 7
    assert plan["checkpoint_epoch_zero_based"] == 49
    for field in ("optimizer_steps", "backward_steps", "model_updates", "heldout_calib_files_opened", "target_files_opened"):
        assert plan[field] == 0
    assert plan["retraining"] is plan["cuda_initialized"] is plan["execution_entrypoints_enabled"] is False
    assert plan["m4_governing_result_unchanged"] is True


def test_earliest_three_are_calibration_and_fourth_is_first_query() -> None:
    support, query = m3_support_and_query_trial((11.0, 12.0, 13.0, 14.0, 15.0))
    assert support == (11.0, 12.0, 13.0)
    assert query == 14.0
    with pytest.raises(M3TransferProtocolError, match="requires a fourth"):
        m3_support_and_query_trial((11.0, 12.0, 13.0))


def test_ordered_legal_trials_uses_only_eval_valid_chronological_values() -> None:
    labels = np.repeat(np.arange(1, 6, dtype=np.float64), 4)
    mask = np.ones(20, dtype=bool)
    mask[4:8] = False
    assert ordered_legal_trials(labels, mask) == (1.0, 3.0, 4.0, 5.0)
    labels[10] = 0.0
    with pytest.raises(M3TransferProtocolError, match="chronological"):
        ordered_legal_trials(labels, np.ones(20, dtype=bool))


def test_causal_surface_starts_at_trial_four_and_keeps_mask_gaps() -> None:
    trial_num = np.repeat(np.arange(1, 6, dtype=np.float64), 800)
    eval_mask = np.ones(4000, dtype=bool)
    eval_mask[3250:3260] = False
    surface = m3_causal_surface(trial_num, eval_mask, len(trial_num))
    assert surface["support"] == [1.0, 2.0, 3.0]
    assert surface["query_trial"] == 4.0
    assert surface["boundary_bin"] == 2400
    starts = surface["starts"]
    outputs = surface["output_bins"]
    assert starts[0] == 2400 and np.all(starts >= 2400)
    assert np.array_equal(outputs, starts + 699)
    assert np.array_equal(surface["score_mask"], eval_mask[outputs])
    assert len(starts) == 901
    assert (~surface["score_mask"]).any()


def test_causal_surface_rejects_three_trial_heldout_shape() -> None:
    trial_num = np.repeat(np.arange(1, 4, dtype=np.float64), 800)
    with pytest.raises(M3TransferProtocolError, match="requires a fourth"):
        m3_causal_surface(trial_num, np.ones(len(trial_num), dtype=bool), len(trial_num))


def test_materialization_calls_three_trial_deployment_carrier(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}
    record = SimpleNamespace(trial_values=(1.0, 2.0, 3.0, 4.0), num_neurons=5)

    def fake_identity(received: object, value: float) -> np.ndarray:
        assert received is record
        calls.setdefault("identity_trials", []).append(value)
        return np.full((1024, 5), value, dtype=np.float32)

    def fake_carrier(received: object, plan: object, values: tuple[float, ...]) -> dict[str, np.ndarray]:
        assert received is record and plan == "frozen-plan"
        calls["carrier_trials"] = values
        return {"carrier": np.full((5, 4), 6.0, dtype=np.float64)}

    monkeypatch.setattr(protocol, "interpolate_trial_identity", fake_identity)
    monkeypatch.setattr(protocol, "fit_deployment_carrier", fake_carrier)
    result = materialize_m3_calibration(record, "frozen-plan", 3.0)
    assert calls == {"identity_trials": [1.0, 2.0, 3.0], "carrier_trials": (1.0, 2.0, 3.0)}
    assert result["support"] == [1.0, 2.0, 3.0] and result["query_trial"] == 4.0
    assert result["identity"].shape == (3, 1024, 5)
    assert result["carrier"].shape == (5, 4)
    assert np.all(result["carrier"] == np.float32(2.0))


def test_last_bin_scaling_is_fixed_prediction_divide_20() -> None:
    output = np.arange(2 * 9 * 7, dtype=np.float32).reshape(2, 9, 7)
    scaled = scale_last_bin_prediction(output)
    assert scaled.dtype == np.float32 and scaled.shape == (2, 7)
    assert np.array_equal(scaled, output[:, -1, :] / np.float32(20.0))


def test_score_is_masked_float64_seven_output_variance_weighted(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, np.ndarray] = {}

    def fake_r2(target: np.ndarray, prediction: np.ndarray) -> float:
        seen["target"] = target
        seen["prediction"] = prediction
        return 0.25

    monkeypatch.setattr(protocol, "variance_weighted_r2", fake_r2)
    target = np.arange(35, dtype=np.float32).reshape(5, 7)
    prediction = target + 1.0
    mask = np.asarray([True, False, True, False, True])
    assert score_m3(target, prediction, mask) == 0.25
    assert seen["target"].dtype == seen["prediction"].dtype == np.float64
    assert np.array_equal(seen["target"], target[mask].astype(np.float64))


def test_aggregate_reports_per_recording_equal_date_delta_and_positive_count() -> None:
    deltas = (0.1, -0.2, 0.3, 0.0, 0.4)
    rows = {}
    for date, delta in zip(CONFIRMATORY_DATES, deltas):
        rows[date] = {
            session: {"t0": 0.2 + index / 100.0, "c1": 0.2 + index / 100.0 + delta}
            for index, session in enumerate(target_sessions_for_date(date))
        }
    result = aggregate_m3_results(rows)
    assert result["status"] == "COMPLETE_SECONDARY_M3_TRANSFER_DIAGNOSTIC"
    assert result["positive_date_count"] == 3
    assert result["equal_date_mean_delta_r2"] == pytest.approx(np.mean(deltas))
    for date, delta in zip(CONFIRMATORY_DATES, deltas):
        assert result["dates"][date]["delta_c1_minus_t0"] == pytest.approx(delta)
        assert tuple(result["dates"][date]["per_recording_r2"]) == target_sessions_for_date(date)
        for row in result["dates"][date]["per_recording_r2"].values():
            assert row["delta_c1_minus_t0"] == pytest.approx(delta)
    assert result["governing_m4_result_modified"] is result["selection_performed"] is False


def test_aggregate_rejects_incomplete_date_or_recording_roster() -> None:
    with pytest.raises(M3TransferProtocolError, match="date roster"):
        aggregate_m3_results({})
    rows = {date: {} for date in CONFIRMATORY_DATES}
    with pytest.raises(M3TransferProtocolError, match="recording roster"):
        aggregate_m3_results(rows)
    rows = {
        date: {session: {"t0": 0.0, "wrong": 0.0} for session in target_sessions_for_date(date)}
        for date in CONFIRMATORY_DATES
    }
    with pytest.raises(M3TransferProtocolError, match="arm roster"):
        aggregate_m3_results(rows)


def test_review_cli_has_no_data_cuda_training_or_evaluation_entrypoint() -> None:
    root = Path(__file__).resolve().parents[2]
    runner_path = root / "tfpd_exploration/h1_series_20260830/scripts/run_h1_cal_aug_prefix_cycle_m3_transfer_v1.py"
    runner = runner_path.read_text(encoding="utf-8")
    assert "--dry-run" in runner
    for forbidden in ("--train", "--evaluate", "--prepare", "--smoke", "torch", "pynwb", "load_nwb", "NWBHDF5IO"):
        assert forbidden not in runner


def test_work_order_records_no_leakage_and_heldout_limit() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "tfpd_exploration/h1_series_20260830/H1_CAL_AUG_PREFIX_CYCLE_M3_TRANSFER_V1_WORK_ORDER.md"
    text = path.read_text(encoding="utf-8")
    assert "secondary transfer diagnostic" in text
    assert "no independent trial 4+ post-calibration scoring surface" in text
    assert "invalid to calculate governing R² on those same three carrier-fitting trials" in text
    assert "evaluation/test stream independent of the three calibration trials" in text
    assert "M2 carrier successor" in text
    assert "No H1 held-out-calib file is in scope" in text
    assert "cannot replace, reinterpret, reopen, or modify the sealed M4 governing result" in text
