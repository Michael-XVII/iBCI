from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.h1_cal_aug_all_source_m3_test_heldout_compat_v1 import (
    LABEL,
    R2_ABS_TOL,
    SCHEMA,
    VAL_METRICS_SHA256,
    VAL_TERMINAL_SHA256,
)


def test_frozen_val_predecessor_and_tolerance() -> None:
    assert VAL_TERMINAL_SHA256 == "a508f83bb9c22fe4d21329e7b02debae337209d3a6822f690a47b2385b20f5b4"
    assert VAL_METRICS_SHA256 == "335b95ee1465f7687347a86de5161d506a3e9b6426a3606ab976622a23fcb03b"
    assert R2_ABS_TOL == 2e-4
    assert "test-loop" in LABEL


def test_test_heldout_std_semantics_is_sample_std() -> None:
    values = np.asarray([1.0, 2.0, 4.0], dtype=np.float64)
    sample = float(np.std(values, ddof=1, dtype=np.float64))
    population = float(np.std(values, ddof=0, dtype=np.float64))
    assert sample > population
    assert np.isclose(sample, np.asarray(values).std(ddof=1))


def test_runner_dry_run_is_no_write_no_data_no_cuda(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    runner = root / "tfpd_exploration/h1_series_20260830/scripts/run_h1_cal_aug_all_source_m3_test_heldout_compat_v1.py"
    result_root = tmp_path / "absent"
    completed = subprocess.run([sys.executable, str(runner), "--dry-run", "--result-root", str(result_root)], check=True, capture_output=True, text=True)
    body = json.loads(completed.stdout)
    assert body["schema"] == SCHEMA
    assert body["writes"] == body["nwb_files_opened"] == body["cuda_initializations"] == 0
    assert body["training"] is False and body["evalai_submissions"] == 0
    assert not result_root.exists()


def test_source_is_strict_evaluation_only_and_independent_inference() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "SPINT-main/src/h1_cal_aug_all_source_m3_test_heldout_compat_v1.py").read_text()
    runner = (root / "tfpd_exploration/h1_series_20260830/scripts/run_h1_cal_aug_all_source_m3_test_heldout_compat_v1.py").read_text()
    for forbidden in ("optimizer.step", ".backward(", "evalai push", "phase=\"test\""):
        assert forbidden not in source
    for forbidden in ("--train", "evalai push"):
        assert forbidden not in runner
    assert "H1CarrierIdSpint(**payload" in source
    assert "padded[index:index + WINDOW_SIZE]" in source
    assert "np.allclose(prediction, old_prediction" in source
    assert 'np.std(values, ddof=1' in source


def test_work_order_freezes_population_and_fail_closed_equivalence() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "tfpd_exploration/h1_series_20260830/docs/WORKORDER_H1_CAL_AUG_ALL_SOURCE_M3_TEST_HELDOUT_COMPAT_V1.md").read_text()
    for phrase in ("W=700", "zero-prefix", "batches of 32", "sample standard", "fails closed", "official FALCON hidden-test R²"):
        assert phrase in text

