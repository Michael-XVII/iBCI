from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from src.h1_cal_aug_m3_aware_dual_selection_v2_eval_a1 import (
    EvaluationRepairError, dry_plan, ordered_m3_trials,
)


def test_m3_trial_loader_accepts_exactly_three_and_preserves_order() -> None:
    trials = np.asarray([1, 1, 2, 2, 3, 3], np.float64)
    assert ordered_m3_trials(trials, np.ones(6, bool)) == (1.0, 2.0, 3.0)


def test_m3_trial_loader_rejects_fewer_than_three_or_nonchronological() -> None:
    with pytest.raises(EvaluationRepairError, match="at least three"):
        ordered_m3_trials(np.asarray([1, 1, 2, 2]), np.ones(4, bool))
    with pytest.raises(EvaluationRepairError, match="nonchronological"):
        ordered_m3_trials(np.asarray([1, 2, 1, 3]), np.ones(4, bool))


def test_dry_run_is_strictly_evaluation_only() -> None:
    plan = dry_plan()
    assert plan["new_training"] is False and plan["frozen_c2_checkpoints"] == 50
    assert plan["m3_loader_minimum_trials"] == 3 and plan["m4_loader_used"] is False
    assert plan["optimizer_steps"] == plan["backward_steps"] == plan["model_updates"] == 0


def test_repair_does_not_call_incompatible_m4_trial_helper() -> None:
    module = Path(__file__).resolve().parents[1] / "src/h1_cal_aug_m3_aware_dual_selection_v2_eval_a1.py"
    text = module.read_text(encoding="utf-8")
    assert "_ordered_eval_trials" not in text
    assert "load_heldout_m3_record" in text
