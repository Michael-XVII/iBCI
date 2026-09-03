from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.h1_cal_aug_m3_aware_dual_selection_v2_evalai_a1 import (
    CANDIDATES, CHALLENGE_ID, PHASE_ID, PHASE_SLUG, PREDECESSOR_COMMIT,
    PREDECESSOR_TERMINAL_SHA256, SubmissionA1Error, collect_results,
)


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tfpd_exploration/h1_series_20260830/scripts/run_h1_cal_aug_m3_aware_dual_selection_v2_evalai_a1.py"


def test_frozen_candidate_order_and_authority() -> None:
    assert PREDECESSOR_COMMIT == "ae14a232d1dfc84de0661916da34fdb9596753c2"
    assert PREDECESSOR_TERMINAL_SHA256.startswith("d0735087")
    assert [row["key"] for row in CANDIDATES] == ["c2_e49", "c2_hi_e45", "c2_ho_e15"]
    assert [row["epoch_zero_based"] for row in CANDIDATES] == [49, 45, 15]
    assert len({row["checkpoint_sha256"] for row in CANDIDATES}) == 3
    assert len({row["model_state_sha256"] for row in CANDIDATES}) == 3


def test_phase_is_exact_previous_official_phase() -> None:
    assert (CHALLENGE_ID, PHASE_ID, PHASE_SLUG) == (2319, 4599, "few-shot-test-2319")


def test_dry_run_is_zero_write_cuda_docker_evalai(tmp_path: Path) -> None:
    result = tmp_path / "must_not_exist"
    completed = subprocess.run([sys.executable, str(RUNNER), "--dry-run", "--result-root", str(result)], check=True, capture_output=True, text=True)
    body = json.loads(completed.stdout)
    assert body["writes"] == body["docker_commands"] == body["evalai_submissions"] == 0
    assert body["training"] is False and body["optimizer_steps"] == body["backward_steps"] == body["model_updates"] == 0
    assert not result.exists()


def test_work_order_freezes_primary_and_contrasts() -> None:
    text = (ROOT / "tfpd_exploration/h1_series_20260830/docs/WORKORDER_H1_CAL_AUG_M3_AWARE_DUAL_SELECTION_V2_EVALAI_A1.md").read_text()
    assert "C2-HO-E15" in text and "performance-oriented primary successor" in text
    assert "C2-HO-E15 minus" in text and "V1-C1-E49" in text
    assert "training=false" in text and "three EvalAI submissions" in text


def test_score_gate_requires_three_submissions(tmp_path: Path) -> None:
    root = tmp_path / "result"
    (root / "submission").mkdir(parents=True)
    path = root / "submission/submissions.json"
    body = {"schema": "h1_cal_aug_m3_aware_dual_selection_v2_evalai_submission_a1_submissions", "submissions": [{"submission_id": 1}]}
    path.write_text(json.dumps(body) + "\n")
    import hashlib
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_name(path.name + ".sha256").write_text(f"{digest}  {path.name}\n")
    path.chmod(0o444); path.with_name(path.name + ".sha256").chmod(0o444)
    with pytest.raises(SubmissionA1Error, match="three submissions"):
        collect_results(root, tmp_path / "logs", Path("evalai"))


def test_no_training_or_checkpoint_selection_code_path() -> None:
    module = (ROOT / "SPINT-main/src/h1_cal_aug_m3_aware_dual_selection_v2_evalai_a1.py").read_text()
    for forbidden in ("optimizer.step", ".backward(", "load_state_dict(checkpoint", "early_stopping"):
        assert forbidden not in module
    assert '"training": False' in module and '"post_selection_retraining": False' in module


def test_host_smoke_amendment_keeps_exact_repeats_and_tolerant_cross_batch() -> None:
    module = (ROOT / "SPINT-main/src/h1_cal_aug_m3_aware_dual_selection_v2_evalai_a1.py").read_text()
    amendment = (ROOT / "tfpd_exploration/h1_series_20260830/docs/AMENDMENT_H1_CAL_AUG_M3_AWARE_DUAL_SELECTION_V2_EVALAI_A1_HOST_SMOKE.md").read_text()
    assert "np.array_equal(first, second)" in module
    assert "np.allclose(cpu1, cpu8" in module and "np.allclose(gpu1, gpu8" in module
    assert "Packages, checkpoints" in amendment
