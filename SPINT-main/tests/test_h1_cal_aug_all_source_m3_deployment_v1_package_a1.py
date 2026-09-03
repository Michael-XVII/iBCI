from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.data.h1_m4_eb_pilot import PilotDataError, _ordered_eval_trials
from src.h1_cal_aug_all_source_m3_deployment_v1_package_a1 import (
    CHECKPOINT_SHA256,
    PREDECESSOR_FAILURE_SHA256,
    PREDECESSOR_PAIR_SHA256,
    PREDECESSOR_SOURCE_SHA256,
    PackageA1Error,
    SCHEMA,
    STATUS_TERMINAL,
    ordered_m3_eval_trials,
)


def test_m3_parser_accepts_exactly_three_without_m4_query_requirement() -> None:
    labels = np.repeat(np.asarray([11.0, 12.0, 13.0]), 8)
    mask = np.ones(len(labels), dtype=bool)
    assert ordered_m3_eval_trials(labels, mask) == (11.0, 12.0, 13.0)
    with pytest.raises(PilotDataError, match="at least five"):
        _ordered_eval_trials(labels, mask)


def test_m3_parser_is_chronological_eval_valid_and_requires_three() -> None:
    labels = np.repeat(np.asarray([1.0, 2.0, 3.0, 4.0]), 4)
    mask = np.ones(len(labels), dtype=bool)
    mask[4:8] = False
    assert ordered_m3_eval_trials(labels, mask) == (1.0, 3.0, 4.0)
    with pytest.raises(PackageA1Error, match="at least three"):
        ordered_m3_eval_trials(np.repeat([1.0, 2.0], 4), np.ones(8, bool))
    bad = labels.copy(); bad[9] = 0.0
    with pytest.raises(PackageA1Error, match="chronological"):
        ordered_m3_eval_trials(bad, np.ones(len(bad), bool))


def test_successor_binds_exact_failed_root_and_valid_checkpoints() -> None:
    assert PREDECESSOR_FAILURE_SHA256 == "71a6915f7f1273d2dd78b71b47c5456957acaf020f26de6fc2c33c0c2511576a"
    assert PREDECESSOR_SOURCE_SHA256 == "8ea4bb1174c00ab713843cd7561562d43f81509eaaea6ea12ee80cd4eba95de7"
    assert PREDECESSOR_PAIR_SHA256 == "b2a4fd570b152028e3e3ab99bbf1bbb11b6ed49d6907cc26772974d0ef4a7e9d"
    assert CHECKPOINT_SHA256 == {
        "t0": "6d4d14226b706951274982438b588527beb442200aad2f50f9d18b68e54a9648",
        "c1": "0f406a8e69fdb57cf6a5480149f04ab3500e7fad849d36db38042edbadb2cd06",
    }


def test_successor_source_does_not_reuse_sealed_m4_parser() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "SPINT-main/src/h1_cal_aug_all_source_m3_deployment_v1_package_a1.py").read_text(encoding="utf-8")
    assert "_ordered_eval_trials" not in source
    assert "ordered_m3_eval_trials(trial_num, mask)" in source
    assert "fit_deployment_carrier(record, plan, support)" in source
    assert 'len(values) == 3' in source


def test_runner_is_packaging_only_and_has_no_evalai_or_training_phase() -> None:
    root = Path(__file__).resolve().parents[2]
    runner = root / "tfpd_exploration/h1_series_20260830/scripts/run_h1_cal_aug_all_source_m3_deployment_v1_package_a1.py"
    text = runner.read_text(encoding="utf-8")
    assert "--detached-supervisor" in text
    for forbidden in ("--train", "--arm-cell", "run_arm(", "evalai push", "--submit", "phase=\"test\""):
        assert forbidden not in text
    completed = subprocess.run([sys.executable, str(runner), "--dry-run"], check=True, capture_output=True, text=True)
    body = json.loads(completed.stdout)
    assert body["schema"] == SCHEMA and body["retraining"] is False
    assert body["evalai_submissions"] == 0


def test_amendment_preserves_failed_root_and_metric_boundary() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "tfpd_exploration/h1_series_20260830/docs/AMENDMENT_H1_CAL_AUG_ALL_SOURCE_M3_DEPLOYMENT_V1_PACKAGE_A1.md").read_text(encoding="utf-8")
    assert "does not modify the sealed M4 helper or predecessor root" in text
    assert "performs no training" in text
    assert "local held-in-minival deployment sanity R²" in text
    assert "not held-out R²" in text
    assert "remote test" in text and "remain forbidden" in text


def test_terminal_status_remains_no_evalai_stop_boundary() -> None:
    assert STATUS_TERMINAL == "COMPLETE_LOCAL_H1_ALL_SOURCE_M3_DEPLOYMENT_READY_NO_EVALAI_SUBMISSION"
