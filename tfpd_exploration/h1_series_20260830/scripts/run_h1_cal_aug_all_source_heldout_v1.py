#!/usr/bin/env python3
"""CLI for the pre-training H1 all-source held-out M4 metadata gate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SPINT_ROOT = REPO_ROOT / "SPINT-main"
if str(SPINT_ROOT) not in sys.path:
    sys.path.insert(0, str(SPINT_ROOT))

from src.h1_cal_aug_all_source_heldout_v1 import (  # noqa: E402
    SCHEMA,
    create_attempt,
    dry_plan,
    load_attempt,
    publish_json,
    run_metadata_feasibility_audit,
    validate_predecessor,
    verify_metadata_terminal,
)
from src.h1_hc_date_lodo_regen_v1 import publish_text  # noqa: E402
from src.h1_m4_cce_contract import sha256_file  # noqa: E402


DEFAULT_DATA_ROOT = Path("/data/ial-dataset/ial-mohd/000954")
DEFAULT_PREDECESSOR_ROOT = (
    REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_prefix_cycle_v1_eval_a1"
)
DEFAULT_RESULT_ROOT = (
    REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_all_source_heldout_v1"
)
WORK_ORDER = REPO_ROOT / "tfpd_exploration/h1_series_20260830/H1_CAL_AUG_ALL_SOURCE_HELDOUT_V1_WORK_ORDER.md"
AMENDMENT = (
    REPO_ROOT
    / "tfpd_exploration/h1_series_20260830/docs/AMENDMENT_H1_CAL_AUG_ALL_SOURCE_HELDOUT_V1_METADATA_FEASIBILITY.md"
)
TEST_FILE = SPINT_ROOT / "tests/test_h1_cal_aug_all_source_heldout_v1.py"


def _closure() -> dict[str, str]:
    paths = (
        WORK_ORDER,
        AMENDMENT,
        Path(__file__).resolve(),
        SPINT_ROOT / "src/data/h1_cal_aug_all_source_heldout_v1.py",
        SPINT_ROOT / "src/h1_cal_aug_all_source_heldout_v1.py",
        TEST_FILE,
        SPINT_ROOT / "src/h1_cal_aug_prefix_cycle_v1.py",
        SPINT_ROOT / "src/h1_cal_aug_prefix_cycle_eval_a1.py",
        SPINT_ROOT / "src/data/h1_m4_eb_pilot.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"metadata feasibility closure is incomplete: {missing}")
    return {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths}


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _load_json(path: Path) -> dict[str, Any]:
    from src.h1_hc_date_lodo_regen_v1 import verify_sidecar

    verify_sidecar(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_closure(result_root: Path) -> None:
    attempt = load_attempt(result_root)
    if attempt.get("head") != _head() or attempt.get("closure") != _closure():
        raise RuntimeError("current Git HEAD/code closure differs from immutable metadata attempt")


def _cpu_gate(result_root: Path) -> None:
    command = [sys.executable, "-m", "pytest", "-q", str(TEST_FILE)]
    environment = dict(os.environ)
    environment.update({
        "PYTHONNOUSERSITE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "CUDA_VISIBLE_DEVICES": "",
    })
    completed = subprocess.run(
        command,
        cwd=SPINT_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_sha = publish_text(result_root.resolve() / "cpu_gate.log", completed.stdout)
    body = {
        "schema": f"{SCHEMA}_cpu_gate",
        "status": "PASS_H1_ALL_SOURCE_HELDOUT_METADATA_CPU_GATE" if completed.returncode == 0 else "FAIL_CPU_GATE",
        "command": command,
        "returncode": completed.returncode,
        "log_sha256": log_sha,
        "cuda_visible_devices": "",
        "nwb_files_opened": 0,
        "heldout_neural_arrays_read": 0,
        "heldout_behavior_arrays_read": 0,
    }
    publish_json(result_root.resolve() / "cpu_gate.json", body)
    if completed.returncode:
        raise RuntimeError("metadata feasibility CPU gate failed")


def _publish_failure(result_root: Path, phase: str, error: BaseException) -> None:
    root = result_root.resolve()
    path = root / f"{phase}_failure.json"
    if not root.exists() or path.exists():
        return
    publish_json(path, {
        "schema": SCHEMA,
        "status": "FAIL_H1_ALL_SOURCE_HELDOUT_METADATA_AUDIT",
        "phase": phase,
        "error_type": type(error).__name__,
        "error": str(error),
        "gpu_training_started": False,
        "prediction_performed": False,
        "r2_calculated": False,
        "heldout_neural_arrays_read": 0,
        "heldout_behavior_arrays_read": 0,
        "metadata_access_may_be_partial": phase == "metadata_audit",
    })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_mutually_exclusive_group()
    phases.add_argument("--dry-run", action="store_true")
    phases.add_argument("--audit-heldout-metadata", action="store_true")
    phases.add_argument("--verify-metadata-terminal", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--predecessor-root", type=Path, default=DEFAULT_PREDECESSOR_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not (args.audit_heldout_metadata or args.verify_metadata_terminal):
        print(json.dumps(dry_plan(), indent=2, sort_keys=True))
        return 0
    phase = "unknown"
    try:
        if args.audit_heldout_metadata:
            phase = "attempt"
            create_attempt(args.result_root, _closure(), _head())
            phase = "cpu_gate"
            _cpu_gate(args.result_root)
            phase = "predecessor_authority"
            predecessor = validate_predecessor(args.predecessor_root)
            publish_json(args.result_root.resolve() / "predecessor_authority.json", predecessor)
            phase = "metadata_audit"
            terminal = run_metadata_feasibility_audit(args.data_root, args.result_root)
            phase = "metadata_verification"
            verify_metadata_terminal(args.result_root)
            print(json.dumps({
                "status": terminal["status"],
                "m4_evaluable_recordings": terminal["m4_evaluable_recordings"],
                "registered_recordings": terminal["registered_recordings"],
                "continuation_to_training_allowed": terminal["continuation_to_training_allowed"],
                "gpu_training_started": False,
            }, sort_keys=True))
        else:
            phase = "metadata_verification"
            _assert_closure(args.result_root)
            terminal = verify_metadata_terminal(args.result_root)
            print(json.dumps(terminal, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        _publish_failure(args.result_root, phase, error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
