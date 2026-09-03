#!/usr/bin/env python3
"""Prepare and locally smoke frozen H1 M3 EvalAI Docker packages without submission."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
SPINT_ROOT = REPO_ROOT / "SPINT-main"
if str(SPINT_ROOT) not in sys.path:
    sys.path.insert(0, str(SPINT_ROOT))

from src.h1_cal_aug_all_source_m3_evalai_package_v1 import (  # noqa: E402
    SCHEMA,
    build_and_smoke,
    create_attempt,
    load_attempt,
    prepare_packages,
    verify_terminal,
)
from src.h1_hc_date_lodo_regen_v1 import publish_json, publish_text  # noqa: E402
from src.h1_m4_cce_contract import sha256_file  # noqa: E402


DEFAULT_PACKAGE_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_all_source_m3_deployment_v1_package_a1"
DEFAULT_STAGING_ROOT = SPINT_ROOT / "local_data/h1_m3_evalai_v1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_all_source_m3_evalai_package_v1"
DEFAULT_LOG_ROOT = REPO_ROOT / "logs/h1_cal_aug_all_source_m3_evalai_package_v1"
WORK_ORDER = REPO_ROOT / "tfpd_exploration/h1_series_20260830/docs/WORKORDER_H1_CAL_AUG_ALL_SOURCE_M3_EVALAI_PACKAGE_V1.md"
MODULE = SPINT_ROOT / "src/h1_cal_aug_all_source_m3_evalai_package_v1.py"
TEST_FILE = SPINT_ROOT / "tests/test_h1_cal_aug_all_source_m3_evalai_package_v1.py"
SAMPLE = SPINT_ROOT / "third_party/falcon_challenge/h1_carrier_id_spint_sample.py"
DOCKERFILE = SPINT_ROOT / "third_party/falcon_challenge/h1_carrier_id_spint_sample.Dockerfile"


def closure() -> dict[str, str]:
    paths = (
        WORK_ORDER, MODULE, TEST_FILE, SAMPLE, DOCKERFILE, Path(__file__).resolve(),
        SPINT_ROOT / "third_party/falcon_challenge/h1_carrier_id_spint_decoder.py",
        SPINT_ROOT / "src/models/components/h1_carrierid_spint.py",
        SPINT_ROOT / "environment.yaml",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"EvalAI package closure incomplete: {missing}")
    return {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths}


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def assert_closure(result_root: Path) -> None:
    attempt = load_attempt(result_root)
    if attempt["closure"] != closure() or attempt["git_head"] != git_head():
        raise RuntimeError("Git HEAD/code closure differs from immutable attempt")


def cpu_gate(result_root: Path) -> None:
    command = [sys.executable, "-m", "pytest", "-q", str(TEST_FILE)]
    environment = dict(os.environ)
    environment.update({"PYTHONNOUSERSITE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "CUDA_VISIBLE_DEVICES": ""})
    completed = subprocess.run(command, cwd=SPINT_ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log_sha = publish_text(result_root.resolve() / "cpu_gate.log", completed.stdout)
    publish_json(result_root.resolve() / "cpu_gate.json", {
        "schema": f"{SCHEMA}_cpu_gate",
        "status": "PASS_H1_M3_EVALAI_PACKAGE_CPU_GATE" if completed.returncode == 0 else "FAIL_H1_M3_EVALAI_PACKAGE_CPU_GATE",
        "returncode": completed.returncode,
        "command": command,
        "log_sha256": log_sha,
        "cuda_visible_devices": "",
        "dataset_files_opened": 0,
        "docker_commands": 0,
        "training": False,
        "evalai_submissions": 0,
    })
    if completed.returncode:
        raise RuntimeError("EvalAI package CPU gate failed")


def publish_failure(result_root: Path, phase: str, error: BaseException) -> None:
    path = result_root.resolve() / f"{phase}_failure.json"
    if path.exists() or not result_root.exists():
        return
    publish_json(path, {
        "schema": SCHEMA,
        "status": "FAIL_H1_M3_EVALAI_PACKAGE_NO_AUTOMATIC_RETRY",
        "phase": phase,
        "error_type": type(error).__name__,
        "error": str(error),
        "training": False,
        "docker_pushes": 0,
        "evalai_submissions": 0,
    })


def dry_plan() -> dict:
    return {
        "schema": SCHEMA,
        "status": "DRY_RUN_NO_WRITE_NO_DATA_NO_CUDA_NO_DOCKER",
        "writes": 0,
        "dataset_files_opened": 0,
        "cuda_initializations": 0,
        "docker_commands": 0,
        "training": False,
        "checkpoint_selection": False,
        "docker_pushes": 0,
        "evalai_submissions": 0,
        "phases": ["cpu-gate", "attempt", "prepare-packages", "build-images", "container-smoke", "terminal"],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    phases = result.add_mutually_exclusive_group()
    phases.add_argument("--dry-run", action="store_true")
    phases.add_argument("--initialize", action="store_true")
    phases.add_argument("--detached-supervisor", action="store_true")
    result.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    result.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING_ROOT)
    result.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    result.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not any((args.initialize, args.detached_supervisor)):
        print(json.dumps(dry_plan(), indent=2, sort_keys=True))
        return 0
    phase = "initialize"
    try:
        if args.initialize:
            create_attempt(args.result_root, args.package_root, closure(), git_head())
            cpu_gate(args.result_root)
            return 0
        assert_closure(args.result_root)
        phase = "prepare_packages"
        prepare_packages(args.package_root, args.staging_root, args.result_root)
        phase = "docker_build_and_smoke"
        build_and_smoke(SPINT_ROOT, args.package_root, args.staging_root, args.result_root, args.log_root)
        phase = "terminal"
        verify_terminal(args.result_root)
        return 0
    except BaseException as error:
        publish_failure(args.result_root, phase, error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
