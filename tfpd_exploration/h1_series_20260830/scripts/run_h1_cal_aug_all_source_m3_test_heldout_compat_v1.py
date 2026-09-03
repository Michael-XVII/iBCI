#!/usr/bin/env python3
"""Run strict local test_heldout compatibility for frozen H1 M3 packages."""
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

from src.h1_cal_aug_all_source_m3_test_heldout_compat_v1 import (  # noqa: E402
    SCHEMA,
    create_attempt,
    load_attempt,
    run_compatibility,
    validate_predecessor,
    verify_terminal,
)
from src.h1_hc_date_lodo_regen_v1 import publish_json, publish_text  # noqa: E402
from src.h1_m4_cce_contract import sha256_file  # noqa: E402


DEFAULT_DATA_ROOT = Path("/data/ial-dataset/ial-mohd/000954")
DEFAULT_PACKAGE_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_all_source_m3_deployment_v1_package_a1"
DEFAULT_VAL_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_all_source_m3_spint_style_heldout_calib_r2_v1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_all_source_m3_test_heldout_compat_v1"
DEFAULT_LOG_ROOT = REPO_ROOT / "logs/h1_cal_aug_all_source_m3_test_heldout_compat_v1"
WORK_ORDER = REPO_ROOT / "tfpd_exploration/h1_series_20260830/docs/WORKORDER_H1_CAL_AUG_ALL_SOURCE_M3_TEST_HELDOUT_COMPAT_V1.md"
MODULE = SPINT_ROOT / "src/h1_cal_aug_all_source_m3_test_heldout_compat_v1.py"
TEST_FILE = SPINT_ROOT / "tests/test_h1_cal_aug_all_source_m3_test_heldout_compat_v1.py"


def closure() -> dict[str, str]:
    paths = (
        WORK_ORDER, MODULE, TEST_FILE, Path(__file__).resolve(),
        SPINT_ROOT / "src/h1_cal_aug_all_source_m3_spint_style_heldout_calib_r2_v1.py",
        SPINT_ROOT / "src/data/falcon_datamodule.py",
        SPINT_ROOT / "src/models/falcon_module.py",
        SPINT_ROOT / "src/models/components/h1_carrierid_spint.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"test-heldout compatibility closure incomplete: {missing}")
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
        "status": "PASS_H1_M3_TEST_HELDOUT_COMPAT_CPU_GATE" if completed.returncode == 0 else "FAIL_H1_M3_TEST_HELDOUT_COMPAT_CPU_GATE",
        "returncode": completed.returncode,
        "command": command,
        "log_sha256": log_sha,
        "cuda_visible_devices": "",
        "nwb_files_opened": 0,
        "training": False,
        "evalai_submissions": 0,
    })
    if completed.returncode:
        raise RuntimeError("test-heldout compatibility CPU gate failed")


def publish_failure(result_root: Path, phase: str, error: BaseException) -> None:
    path = result_root.resolve() / f"{phase}_failure.json"
    if path.exists() or not result_root.exists():
        return
    publish_json(path, {
        "schema": SCHEMA,
        "status": "FAIL_H1_M3_TEST_HELDOUT_COMPAT_NO_AUTOMATIC_RETRY",
        "phase": phase,
        "error_type": type(error).__name__,
        "error": str(error),
        "training": False,
        "evalai_submissions": 0,
    })


def dry_plan() -> dict:
    return {
        "schema": SCHEMA,
        "status": "DRY_RUN_NO_WRITE_NO_DATA_NO_CUDA",
        "writes": 0,
        "nwb_files_opened": 0,
        "cuda_initializations": 0,
        "training": False,
        "checkpoint_selection": False,
        "evalai_submissions": 0,
        "phases": ["cpu-gate", "attempt", "predecessor-authority", "independent-test-loop", "equivalence-gate", "terminal"],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    phases = result.add_mutually_exclusive_group()
    phases.add_argument("--dry-run", action="store_true")
    phases.add_argument("--initialize", action="store_true")
    phases.add_argument("--detached-supervisor", action="store_true")
    phases.add_argument("--verify-terminal", action="store_true")
    result.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    result.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    result.add_argument("--val-result-root", type=Path, default=DEFAULT_VAL_ROOT)
    result.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    result.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    result.add_argument("--device", default="cuda:0")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not any((args.initialize, args.detached_supervisor, args.verify_terminal)):
        print(json.dumps(dry_plan(), indent=2, sort_keys=True))
        return 0
    phase = "initialize"
    try:
        if args.initialize:
            create_attempt(args.result_root, args.package_root, args.val_result_root, closure(), git_head())
            cpu_gate(args.result_root)
            return 0
        assert_closure(args.result_root)
        if args.verify_terminal:
            verify_terminal(args.result_root)
            return 0
        phase = "predecessor_authority"
        validate_predecessor(args.package_root, args.val_result_root, args.result_root)
        phase = "independent_test_loop"
        run_compatibility(args.package_root, args.val_result_root, args.data_root, args.result_root, device=args.device)
        phase = "terminal"
        verify_terminal(args.result_root)
        return 0
    except BaseException as error:
        publish_failure(args.result_root, phase, error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
