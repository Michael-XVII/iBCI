#!/usr/bin/env python3
"""Detached packaging successor for H1 all-source M3 deployment V1."""
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

from src.h1_cal_aug_all_source_m3_deployment_v1_package_a1 import (  # noqa: E402
    SCHEMA,
    build_packages,
    create_attempt,
    load_attempt,
    publish_json,
    run_local_minival,
    run_package_rehearsal,
    verify_terminal,
)
from src.h1_hc_date_lodo_regen_v1 import publish_text  # noqa: E402
from src.h1_m4_cce_contract import sha256_file  # noqa: E402


DEFAULT_DATA_ROOT = Path("/data/ial-dataset/ial-mohd/000954")
DEFAULT_PREDECESSOR_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_all_source_m3_deployment_v1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_all_source_m3_deployment_v1_package_a1"
DEFAULT_LOG_ROOT = REPO_ROOT / "logs/h1_cal_aug_all_source_m3_deployment_v1_package_a1"
AMENDMENT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/docs/AMENDMENT_H1_CAL_AUG_ALL_SOURCE_M3_DEPLOYMENT_V1_PACKAGE_A1.md"
TEST_FILE = SPINT_ROOT / "tests/test_h1_cal_aug_all_source_m3_deployment_v1_package_a1.py"


def _closure() -> dict[str, str]:
    paths = (
        AMENDMENT, Path(__file__).resolve(), TEST_FILE,
        SPINT_ROOT / "src/h1_cal_aug_all_source_m3_deployment_v1_package_a1.py",
        SPINT_ROOT / "src/h1_cal_aug_all_source_m3_deployment_v1_exec.py",
        SPINT_ROOT / "src/h1_cal_aug_all_source_m3_deployment_v1_contract.py",
        SPINT_ROOT / "third_party/falcon_challenge/h1_carrier_id_spint_decoder.py",
        SPINT_ROOT / "src/models/components/h1_carrierid_spint.py",
        SPINT_ROOT / "src/data/h1_m4_eb_pilot.py",
        SPINT_ROOT / "src/h1_m4_cce_contract.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"A1 code closure incomplete: {missing}")
    return {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths}


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _assert_closure(result_root: Path) -> None:
    attempt = load_attempt(result_root)
    if attempt["closure"] != _closure() or attempt["head"] != _head():
        raise RuntimeError("A1 Git HEAD/code closure differs from immutable attempt")


def _cpu_gate(result_root: Path) -> None:
    command = [sys.executable, "-m", "pytest", "-q", str(TEST_FILE)]
    environment = dict(os.environ)
    environment.update({"PYTHONNOUSERSITE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "CUDA_VISIBLE_DEVICES": ""})
    completed = subprocess.run(command, cwd=SPINT_ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log_sha = publish_text(result_root.resolve() / "cpu_gate.log", completed.stdout)
    publish_json(result_root.resolve() / "cpu_gate.json", {
        "schema": f"{SCHEMA}_cpu_gate", "status": "PASS_PACKAGE_A1_CPU_GATE" if completed.returncode == 0 else "FAIL_PACKAGE_A1_CPU_GATE",
        "returncode": completed.returncode, "command": command, "log_sha256": log_sha,
        "cuda_visible_devices": "", "nwb_files_opened": 0, "training": False,
        "evalai_submissions": 0,
    })
    if completed.returncode:
        raise RuntimeError("A1 CPU gate failed")


def _publish_failure(result_root: Path, phase: str, error: BaseException) -> None:
    path = result_root.resolve() / f"{phase}_failure.json"
    if path.exists() or not result_root.exists():
        return
    publish_json(path, {"schema": SCHEMA, "status": "FAIL_PACKAGE_A1_NO_AUTOMATIC_RETRY", "phase": phase, "error_type": type(error).__name__, "error": str(error), "training": False, "evalai_submissions": 0})


def dry_plan() -> dict:
    return {
        "schema": SCHEMA, "status": "DRY_PACKAGE_A1_NO_WRITE_NO_DATA_NO_CUDA",
        "reuses_epoch49_checkpoints": True, "retraining": False,
        "phases": ["package", "local-held-in-minival", "package-rehearsal", "terminal"],
        "evalai_submissions": 0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_mutually_exclusive_group()
    phases.add_argument("--dry-run", action="store_true")
    phases.add_argument("--initialize", action="store_true")
    phases.add_argument("--detached-supervisor", action="store_true")
    phases.add_argument("--verify-terminal", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--predecessor-root", type=Path, default=DEFAULT_PREDECESSOR_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not any((args.initialize, args.detached_supervisor, args.verify_terminal)):
        print(json.dumps(dry_plan(), indent=2, sort_keys=True)); return 0
    phase = "initialize"
    try:
        if args.initialize:
            create_attempt(args.result_root, args.predecessor_root, _closure(), _head())
            _cpu_gate(args.result_root); return 0
        _assert_closure(args.result_root)
        if args.verify_terminal:
            verify_terminal(args.predecessor_root, args.result_root); return 0
        phase = "packages"; build_packages(args.predecessor_root, args.data_root, args.result_root)
        phase = "minival"; run_local_minival(args.data_root, args.result_root, device="cuda:0")
        phase = "package_rehearsal"; run_package_rehearsal(args.data_root, args.result_root)
        phase = "terminal_verification"; verify_terminal(args.predecessor_root, args.result_root)
        return 0
    except BaseException as error:
        _publish_failure(args.result_root, phase, error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
