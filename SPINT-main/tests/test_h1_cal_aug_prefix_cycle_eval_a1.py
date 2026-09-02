from __future__ import annotations

import json
from pathlib import Path
import stat

import pytest

from src.h1_cal_aug_prefix_cycle_eval_a1 import (
    SCHEMA,
    TRAINING_SEAL_COMMIT,
    collect_training_authority,
    create_attempt,
    dry_plan,
    load_attempt,
)
from src.h1_hc_date_lodo_regen_v1 import verify_sidecar
from src.h1_m4_cce_contract import CONFIRMATORY_DATES


def test_dry_plan_is_evaluation_only_and_zero_access() -> None:
    plan = dry_plan()
    assert plan["schema"] == SCHEMA
    assert plan["status"] == "DRY_NO_WRITE_NO_DATA_NO_CUDA"
    assert plan["amendment"] == "evaluation-only; no retraining"
    assert tuple(plan["outer_dates"]) == CONFIRMATORY_DATES
    assert plan["training_seal_commit"] == TRAINING_SEAL_COMMIT
    assert plan["target_access"] == 0


def test_attempt_is_publish_once_0444_and_before_cuda_target(tmp_path: Path) -> None:
    root = tmp_path / "result"
    body = create_attempt(root, {"x": "0" * 64}, "1" * 40)
    path = root / "attempt.json"
    assert body["status"] == "ATTEMPT_BEFORE_TRAINING_AUTHORITY_CUDA_AND_TARGET"
    assert body["cuda_initialized"] is False
    assert body["target_recordings_opened"] == body["target_bytes_read"] == 0
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert load_attempt(root) == body
    verify_sidecar(path)
    with pytest.raises(RuntimeError, match="not fresh"):
        create_attempt(root, {"x": "0" * 64}, "1" * 40)


def test_attempt_rejects_writable_receipt(tmp_path: Path) -> None:
    root = tmp_path / "result"
    create_attempt(root, {"x": "0" * 64}, "1" * 40)
    (root / "attempt.json").chmod(0o644)
    with pytest.raises(RuntimeError, match="0444"):
        load_attempt(root)


def test_training_authority_rejects_missing_sealed_attempt(tmp_path: Path) -> None:
    with pytest.raises((FileNotFoundError, RuntimeError)):
        collect_training_authority(tmp_path / "missing")


def test_runner_orders_attempt_before_authority_and_evaluation() -> None:
    runner = (
        Path(__file__).resolve().parents[2]
        / "tfpd_exploration/h1_series_20260830/scripts/run_h1_cal_aug_prefix_cycle_eval_a1.py"
    )
    text = runner.read_text(encoding="utf-8")
    attempt = text.index("create_attempt(args.result_root")
    cpu_gate = text.index("_cpu_gate(args.result_root)", attempt)
    authority = text.index("prepare_authority(args.training_root", cpu_gate)
    evaluator = text.index("_evaluate_all(args, gpus)", authority)
    assert attempt < cpu_gate < authority < evaluator
    assert "run_arm" not in text
    assert "optimizer" not in text.lower()
    assert "tqdm" not in text and "progress" not in text.lower()


def test_expected_failure_receipt_shape_is_not_a_scientific_terminal() -> None:
    failure = {
        "schema": "h1_cal_aug_prefix_cycle_v1",
        "status": "FAIL_IMMUTABLE_NO_AUTOMATIC_RETRY",
        "phase": "training",
        "error_type": "FileExistsError",
        "target_recordings_opened": 0,
        "target_bytes_read": 0,
    }
    assert failure["status"].startswith("FAIL_")
    assert not failure["status"].startswith("COMPLETE_")
    assert json.loads(json.dumps(failure))["error_type"] == "FileExistsError"
