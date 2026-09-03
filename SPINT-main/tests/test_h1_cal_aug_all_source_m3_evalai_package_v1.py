from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from src.h1_cal_aug_all_source_m3_evalai_package_v1 import (
    A1_TERMINAL_SHA256,
    CHECKPOINT_SHA256,
    IMAGE_TAGS,
    PACKAGE_SHA256,
    SCHEMA,
)


def test_frozen_submission_inputs() -> None:
    assert A1_TERMINAL_SHA256 == "4137495462a299e948beb58be578c739cc211330de4769992c03e743d7c7bf26"
    assert CHECKPOINT_SHA256["t0"].startswith("6d4d1422") and CHECKPOINT_SHA256["c1"].startswith("0f406a8e")
    assert PACKAGE_SHA256["t0"].startswith("d2ad4bea") and PACKAGE_SHA256["c1"].startswith("bfd02e51")
    assert set(IMAGE_TAGS) == {"t0", "c1"}


def test_container_entrypoint_has_smoke_and_remote_test_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    sample = (root / "SPINT-main/third_party/falcon_challenge/h1_carrier_id_spint_sample.py").read_text()
    dockerfile = (root / "SPINT-main/third_party/falcon_challenge/h1_carrier_id_spint_sample.Dockerfile").read_text()
    assert "H1CarrierIdSpintDecoder" in sample and '"smoke", "local", "remote"' in sample
    assert "model_state_immutable" in sample
    for value in ("EVALUATION_LOC=remote", "TASK=h1", "PHASE=test", "BATCH_SIZE=${BATCH_SIZE}"):
        assert value in dockerfile
    assert "ibci.h1.package.sha256" in dockerfile and "ibci.h1.checkpoint.sha256" in dockerfile


def test_runner_dry_run_has_no_docker_data_cuda_or_submission(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    runner = root / "tfpd_exploration/h1_series_20260830/scripts/run_h1_cal_aug_all_source_m3_evalai_package_v1.py"
    result_root = tmp_path / "absent"
    completed = subprocess.run([sys.executable, str(runner), "--dry-run", "--result-root", str(result_root)], check=True, capture_output=True, text=True)
    body = json.loads(completed.stdout)
    assert body["schema"] == SCHEMA
    assert body["writes"] == body["dataset_files_opened"] == body["cuda_initializations"] == body["docker_commands"] == 0
    assert body["evalai_submissions"] == body["docker_pushes"] == 0
    assert not result_root.exists()


def test_packager_has_no_training_scoring_or_submission_command() -> None:
    root = Path(__file__).resolve().parents[2]
    module = (root / "SPINT-main/src/h1_cal_aug_all_source_m3_evalai_package_v1.py").read_text()
    runner = (root / "tfpd_exploration/h1_series_20260830/scripts/run_h1_cal_aug_all_source_m3_evalai_package_v1.py").read_text()
    for forbidden in ("optimizer.step", ".backward(", "evalai push", "docker push", "FalconEvaluator"):
        assert forbidden not in module
    for forbidden in ("--train", "evalai push", "docker push"):
        assert forbidden not in runner
    assert '"--evaluation", "smoke"' in module


def test_work_order_stops_before_submission_and_scores() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / "tfpd_exploration/h1_series_20260830/docs/WORKORDER_H1_CAL_AUG_ALL_SOURCE_M3_EVALAI_PACKAGE_V1.md").read_text()
    assert "EvalAI submission is forbidden" in text
    assert "held-out scoring" in text and "official hidden-test access" in text
    assert "submission-ready only" in text

