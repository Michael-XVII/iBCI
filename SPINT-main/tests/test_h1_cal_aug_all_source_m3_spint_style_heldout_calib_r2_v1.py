from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.h1_cal_aug_all_source_m3_spint_style_heldout_calib_r2_v1 import (
    CHECKPOINT_SHA256,
    LABEL,
    PACKAGE_A1_TERMINAL_SHA256,
    SCHEMA,
    SpintStyleHeldoutError,
    legacy_spint_score_mask,
)
from src.h1_hc_date_lodo_regen_v1 import variance_weighted_r2


def test_legacy_spint_mask_matches_per_session_batch32_drop_last() -> None:
    mask = np.zeros(90, dtype=bool)
    mask[[0, 2, *range(5, 75)]] = True
    result = legacy_spint_score_mask(mask, batch_size=32)
    indices = np.flatnonzero(mask)
    assert result.sum() == 64
    assert np.array_equal(np.flatnonzero(result), indices[:64])
    assert result[0] and result[2]  # no W700 full-history exclusion: legacy zero padding applies


def test_legacy_spint_mask_rejects_incomplete_population() -> None:
    with pytest.raises(SpintStyleHeldoutError, match="insufficient"):
        legacy_spint_score_mask(np.asarray([True, False]), batch_size=32)
    with pytest.raises(SpintStyleHeldoutError, match="positive"):
        legacy_spint_score_mask(np.ones(64, bool), batch_size=0)


def test_float64_variance_weighted_r2_is_per_recording() -> None:
    target = np.arange(70, dtype=np.float64).reshape(10, 7)
    prediction = target.copy()
    assert variance_weighted_r2(target, prediction) == 1.0
    shifted = prediction.copy(); shifted[:, 0] += 1.0
    value = variance_weighted_r2(target, shifted)
    assert isinstance(value, float) and np.isfinite(value) and value < 1.0


def test_frozen_predecessor_and_claim_boundary() -> None:
    assert PACKAGE_A1_TERMINAL_SHA256 == "4137495462a299e948beb58be578c739cc211330de4769992c03e743d7c7bf26"
    assert CHECKPOINT_SHA256 == {
        "t0": "6d4d14226b706951274982438b588527beb442200aad2f50f9d18b68e54a9648",
        "c1": "0f406a8e69fdb57cf6a5480149f04ab3500e7fad849d36db38042edbadb2cd06",
    }
    assert "held-out-calib validation" in LABEL


def test_runner_dry_run_is_no_write_no_data_no_cuda(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    runner = root / "tfpd_exploration/h1_series_20260830/scripts/run_h1_cal_aug_all_source_m3_spint_style_heldout_calib_r2_v1.py"
    result_root = tmp_path / "must_not_exist"
    completed = subprocess.run(
        [sys.executable, str(runner), "--dry-run", "--result-root", str(result_root)],
        check=True, capture_output=True, text=True,
    )
    body = json.loads(completed.stdout)
    assert body["schema"] == SCHEMA
    assert body["writes"] == body["nwb_files_opened"] == body["cuda_initializations"] == 0
    assert body["evalai_submissions"] == 0 and not result_root.exists()


def test_source_has_no_training_evalai_or_full_history_filter() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "SPINT-main/src/h1_cal_aug_all_source_m3_spint_style_heldout_calib_r2_v1.py").read_text()
    runner = (root / "tfpd_exploration/h1_series_20260830/scripts/run_h1_cal_aug_all_source_m3_spint_style_heldout_calib_r2_v1.py").read_text()
    for forbidden in ("optimizer.step", ".backward(", "evalai push", "phase=\"test\"", "complete_history"):
        assert forbidden not in source
    for forbidden in ("--train", "evalai push"):
        assert forbidden not in runner
    assert "H1CarrierIdSpintDecoder" in source
    assert "legacy_spint_score_mask(eval_mask)" in source


def test_work_order_records_calibration_reuse_and_nonofficial_boundary() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "tfpd_exploration/h1_series_20260830/docs/WORKORDER_H1_CAL_AUG_ALL_SOURCE_M3_SPINT_STYLE_HELDOUT_CALIB_R2_V1.md").read_text()
    assert "not the FALCON" in text and "official held-out R²" in text
    assert "calibration-reuse" in text
    assert "drop only the final incomplete per-session batch" in text
    assert "No EvalAI" in text
