from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.h1_cal_aug_all_source_m3_deployment_v1_contract import (
    C1_CYCLE,
    DeploymentContractError,
    HELDIN_SESSION_TO_FALCON_KEY,
    HELDOUT_SESSION_TO_FALCON_KEY,
    classify_local_inventory,
    deterministic_prefix_cycle,
    dry_plan,
    frozen_training_contract,
    session_mapping,
    validate_balanced_cycle,
    validate_decoder_payload,
    validate_m3_deployment,
)


def _inventory() -> list[str]:
    rows = []
    for session, _ in HELDIN_SESSION_TO_FALCON_KEY:
        rows.append(f"sub-HumanPitt-held-in-calib/sub-HumanPitt-held-in-calib_{session}.nwb")
        rows.append(f"sub-HumanPitt-held-in-minival/sub-HumanPitt-held-in-minival_{session}.nwb")
    for session, _ in HELDOUT_SESSION_TO_FALCON_KEY:
        rows.append(f"sub-HumanPitt-held-out-calib/sub-HumanPitt-held-out-calib_{session}.nwb")
    return rows


def test_exact_session_rosters_and_falcon_keys_are_one_to_one() -> None:
    assert len(HELDIN_SESSION_TO_FALCON_KEY) == 13
    assert len(HELDOUT_SESSION_TO_FALCON_KEY) == 14
    mapping = session_mapping()
    assert len(mapping) == len(set(mapping.values())) == 27
    assert {key.split("_")[0] for _, key in HELDIN_SESSION_TO_FALCON_KEY} == {f"S{i}" for i in range(6)}
    assert {key.split("_")[0] for _, key in HELDOUT_SESSION_TO_FALCON_KEY} == {f"S{i}" for i in range(6, 13)}


def test_local_inventory_has_heldin_minival_but_no_heldout_score_stream() -> None:
    result = classify_local_inventory(_inventory())
    assert result["complete_public_inventory"] is True
    assert result["counts"] == {
        "held_in_calib": 13,
        "held_in_minival": 13,
        "held_out_calib": 14,
        "held_out_eval_or_test": 0,
    }
    assert result["local_heldin_minival_scoring_available"] is True
    assert result["local_heldout_postcalibration_scoring_available"] is False
    assert result["official_heldout_score_requires_evalai"] is True


def test_frozen_training_recipe_keeps_m3_out_of_cycle() -> None:
    contract = frozen_training_contract()
    assert contract["model"] == "H1CarrierIdSpint"
    assert contract["hidden_size"] == 32 and contract["window_bins"] == 700
    assert contract["t0_identity_prefix"] == 7
    assert tuple(contract["c1_identity_cycle"]) == C1_CYCLE == (7, 5, 4)
    assert contract["training_carrier_trials"] == 4 and contract["m3_in_training_cycle"] is False
    assert contract["seed"] == 42 and contract["batch_size"] == 32
    assert contract["optimizer"] == "Adam" and contract["learning_rate"] == 5e-5
    assert contract["weight_decay"] == 0.0 and contract["precision"] == "fp32"
    assert contract["epochs"] == 50 and contract["checkpoint_epoch_zero_based"] == 49
    assert contract["loss"] == "last_bin_mse" and contract["prediction_divisor"] == 20.0
    for field in ("validation_checkpoint_selection", "early_stopping", "warm_start", "target_fitting", "budget_sweep", "hyperparameter_sweep"):
        assert contract[field] is False


def test_cycle_is_deterministic_balanced_and_excludes_m3() -> None:
    first = deterministic_prefix_cycle("all-source|epoch=0", 32)
    second = deterministic_prefix_cycle("all-source|epoch=0", 32)
    assert first == second and 3 not in first
    validate_balanced_cycle(first)
    with pytest.raises(DeploymentContractError, match="M7/M5/M4"):
        validate_balanced_cycle((7, 5, 3))


def test_m3_requires_independent_scoring_stream_and_zero_updates() -> None:
    local = validate_m3_deployment((1, 2, 3), "held-in-minival")
    remote = validate_m3_deployment((1, 2, 3), "evalai-test")
    for result in (local, remote):
        assert result["carrier_function"] == "fit_deployment_carrier"
        assert result["calibration_rows_scored"] == 0
        assert result["optimizer_steps"] == result["backward_steps"] == result["parameter_updates"] == 0
    with pytest.raises(DeploymentContractError, match="cannot be used"):
        validate_m3_deployment((1, 2, 3), "held-out-calib")
    with pytest.raises(DeploymentContractError, match="earliest three"):
        validate_m3_deployment((2, 3, 4), "evalai-test")


def test_carrier_aware_decoder_payload_requires_m3_identity_and_carrier() -> None:
    keys = tuple(key for _, key in HELDOUT_SESSION_TO_FALCON_KEY)
    payload = {
        "schema": "h1_cal_aug_all_source_m3_deployment_v1_decoder_payload",
        "window_bins": 700,
        "prediction_divisor": 20.0,
        "sessions": {
            key: {"identity_shape": [3, 1024, 176], "carrier_shape": [176, 4], "calibration_trials": [1, 2, 3]}
            for key in keys
        },
    }
    validate_decoder_payload(payload, keys)
    payload["sessions"][keys[0]]["carrier_shape"] = [176, 3]
    with pytest.raises(DeploymentContractError, match="carrier shape"):
        validate_decoder_payload(payload, keys)


def test_dry_plan_is_zero_side_effect_and_distinguishes_score_surfaces() -> None:
    plan = dry_plan()
    assert plan["status"] == "DRY_REVIEW_ONLY_NO_WRITE_NO_FILE_ACCESS_NO_DATA_NO_CUDA"
    assert plan["source_recordings"] == 13 and plan["heldout_calibration_recordings"] == 14
    assert plan["deployment_budget"] == 3
    assert "sanity only" in plan["local_independent_stream"]
    assert "EvalAI" in plan["official_heldout_stream"]
    assert plan["official_heldout_score_locally_available"] is False
    for field in ("writes", "filesystem_reads", "nwb_files_opened", "evalai_submissions"):
        assert plan[field] == 0
    for field in ("cuda_initialized", "training_started", "inference_started", "r2_calculated", "carrier_fitted", "decoder_packaged", "result_root_created"):
        assert plan[field] is False


def test_review_cli_is_only_dry_run_and_imports_no_heavy_runtime() -> None:
    root = Path(__file__).resolve().parents[2]
    runner = root / "tfpd_exploration/h1_series_20260830/scripts/run_h1_cal_aug_all_source_m3_deployment_v1.py"
    text = runner.read_text(encoding="utf-8")
    assert "--dry-run" in text
    for forbidden in ("--train", "--evaluate", "--package", "--submit", "--prepare", "--smoke", "torch", "pynwb", "h5py", "load_nwb", "NWBHDF5IO"):
        assert forbidden not in text
    completed = subprocess.run([sys.executable, str(runner), "--dry-run"], check=True, capture_output=True, text=True)
    output = json.loads(completed.stdout)
    assert output == dry_plan()


def test_contract_module_is_stdlib_only() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "SPINT-main/src/h1_cal_aug_all_source_m3_deployment_v1_contract.py"
    text = path.read_text(encoding="utf-8")
    for forbidden in ("import torch", "import numpy", "import pynwb", "import h5py", "load_nwb", "NWBHDF5IO"):
        assert forbidden not in text


def test_work_order_records_local_and_official_boundaries() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "tfpd_exploration/h1_series_20260830/H1_CAL_AUG_ALL_SOURCE_M3_DEPLOYMENT_V1_WORK_ORDER.md"
    text = path.read_text(encoding="utf-8")
    assert "not held-out generalization and not the governing official score" in text
    assert "Only the remote hidden `test/eval` stream can produce the governing held-out score" in text
    assert "Those three trials are calibration-only" in text
    assert "M3 is forbidden from the training cycle" in text
    assert "does not pass the required H-C carrier" in text
    assert "No EvalAI call, credential access, upload, or submission is permitted" in text
