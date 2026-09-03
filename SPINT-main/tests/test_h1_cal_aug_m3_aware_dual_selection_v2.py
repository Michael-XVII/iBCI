from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from src.h1_cal_aug_m3_aware_dual_selection_v2_contract import (
    BATCHES_PER_EPOCH, C2_CYCLE, EPOCHS, GLOBAL_STEPS, V2ContractError,
    dry_plan, prefix_schedule, select_epoch,
)
from src.h1_cal_aug_m3_aware_dual_selection_v2_exec import _surface_metrics


def test_cycle_is_deterministic_balanced_and_contains_m3() -> None:
    first = prefix_schedule()
    second = prefix_schedule()
    assert first == second and len(first) == EPOCHS
    assert sum(map(len, first)) == GLOBAL_STEPS
    for row in first:
        assert len(row) == BATCHES_PER_EPOCH and set(row) == set(C2_CYCLE)
        counts = Counter(row)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_selection_tie_break_is_frozen() -> None:
    rows = [
        {"epoch_zero_based": epoch, "mean": 0.1, "worst_session_r2": 0.0, "session_std_population": 0.2}
        for epoch in range(50)
    ]
    rows[8].update(mean=0.2, worst_session_r2=-0.2, session_std_population=0.3)
    rows[9].update(mean=0.2, worst_session_r2=-0.1, session_std_population=0.4)
    rows[10].update(mean=0.2, worst_session_r2=-0.1, session_std_population=0.1)
    rows[11].update(mean=0.2, worst_session_r2=-0.1, session_std_population=0.1)
    assert select_epoch(rows, "mean")["epoch_zero_based"] == 10
    with pytest.raises(V2ContractError, match="all 50"):
        select_epoch(rows[:-1], "mean")


def test_dry_run_is_no_write_no_data_no_cuda() -> None:
    plan = dry_plan()
    assert plan["writes"] == plan["nwb_files_opened"] == plan["docker_builds"] == plan["evalai_submissions"] == 0
    assert plan["cuda_initialized"] is False
    assert plan["new_training_arms"] == ["c2"]


def test_runner_forbids_progress_and_remote_actions() -> None:
    root = Path(__file__).resolve().parents[2]
    runner = root / "tfpd_exploration/h1_series_20260830/scripts/run_h1_cal_aug_m3_aware_dual_selection_v2.py"
    text = runner.read_text(encoding="utf-8")
    assert "TQDM_DISABLE" in text and "--detached-supervisor" in text
    for forbidden in ("evalai push", "docker build", "tail -f", "capture-pane", "early_stopping"):
        assert forbidden not in text


def test_grouping_merges_set_recordings_before_r2() -> None:
    import numpy as np
    mapping = (("r0a", "S0_set_1"), ("r0b", "S0_set_2"), ("r1a", "S1_set_1"))
    targets = {
        "S0_set_1": np.tile(np.arange(4, dtype=np.float64)[:, None], (1, 7)),
        "S0_set_2": np.tile(np.arange(4, 8, dtype=np.float64)[:, None], (1, 7)),
        "S1_set_1": np.tile(np.arange(8, 12, dtype=np.float64)[:, None], (1, 7)),
    }
    predictions = {key: value.copy() for key, value in targets.items()}
    masks = {key: np.ones(4, bool) for key in targets}
    result = _surface_metrics(predictions, targets, masks, mapping)
    assert result["r2_mean"] == 1.0 and result["r2_std_population"] == 0.0
    assert set(result["per_session_r2"]) == {"S0", "S1"}
