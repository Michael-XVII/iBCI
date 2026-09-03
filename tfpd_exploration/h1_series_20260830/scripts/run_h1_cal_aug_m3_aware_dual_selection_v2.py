#!/usr/bin/env python3
"""Detached fail-closed executor for H1 CAL-AUG M3-Aware Dual-Selection V2."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
SPINT_ROOT = REPO_ROOT / "SPINT-main"
if str(SPINT_ROOT) not in sys.path:
    sys.path.insert(0, str(SPINT_ROOT))

from src.h1_cal_aug_m3_aware_dual_selection_v2_contract import SCHEMA, dry_plan  # noqa: E402
from src.h1_cal_aug_m3_aware_dual_selection_v2_exec import (  # noqa: E402
    create_attempt, load_attempt, prepare_source_authority, publish_json,
    publish_text, run_c2_training, run_offline_validation, verify_terminal,
    verify_training_integrity,
)
from src.h1_m4_cce_contract import sha256_file  # noqa: E402


DEFAULT_DATA_ROOT = Path("/data/ial-dataset/ial-mohd/000954")
DEFAULT_PREDECESSOR_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_all_source_m3_deployment_v1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_m3_aware_dual_selection_v2"
DEFAULT_LOG_ROOT = REPO_ROOT / "logs/h1_cal_aug_m3_aware_dual_selection_v2"
WORK_ORDER = REPO_ROOT / "tfpd_exploration/h1_series_20260830/H1_CAL_AUG_M3_AWARE_DUAL_SELECTION_V2_WORK_ORDER.md"
TEST = SPINT_ROOT / "tests/test_h1_cal_aug_m3_aware_dual_selection_v2.py"
MIN_GPU_FREE_MIB = 8192
MIN_DISK_FREE_BYTES = 4 * 1024**3


def _closure() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(), WORK_ORDER, TEST,
        SPINT_ROOT / "src/h1_cal_aug_m3_aware_dual_selection_v2_contract.py",
        SPINT_ROOT / "src/h1_cal_aug_m3_aware_dual_selection_v2_exec.py",
        SPINT_ROOT / "src/h1_cal_aug_all_source_m3_deployment_v1_contract.py",
        SPINT_ROOT / "src/h1_cal_aug_all_source_m3_deployment_v1_exec.py",
        SPINT_ROOT / "third_party/falcon_challenge/h1_carrier_id_spint_decoder.py",
        SPINT_ROOT / "src/h1_cal_aug_prefix_cycle_v1.py",
        SPINT_ROOT / "src/h1_hc_date_lodo_regen_v1.py",
        SPINT_ROOT / "src/data/h1_m4_eb_pilot.py",
        SPINT_ROOT / "src/models/components/h1_carrierid_spint.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"code closure incomplete: {missing}")
    return {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths}


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _assert_closure(result_root: Path) -> None:
    attempt = load_attempt(result_root)
    if attempt["git_head"] != _head() or attempt["closure"] != _closure():
        raise RuntimeError("Git HEAD/code closure differs from immutable attempt")


def _cpu_gate(result_root: Path) -> None:
    command = [sys.executable, "-m", "pytest", "-q", str(TEST)]
    environment = dict(os.environ)
    environment.update({"PYTHONNOUSERSITE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "CUDA_VISIBLE_DEVICES": "", "TQDM_DISABLE": "1"})
    completed = subprocess.run(command, cwd=SPINT_ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log_sha = publish_text(result_root.resolve() / "cpu_gate.log", completed.stdout)
    publish_json(result_root.resolve() / "cpu_gate.json", {
        "schema": f"{SCHEMA}_cpu_gate", "status": "PASS_CPU_NO_DATA_GATE" if completed.returncode == 0 else "FAIL_CPU_GATE",
        "returncode": completed.returncode, "command": command, "log_sha256": log_sha,
        "nwb_files_opened": 0, "cuda_initialized": False, "docker_builds": 0, "evalai_submissions": 0,
    })
    if completed.returncode:
        raise RuntimeError("CPU gate failed")


def _precheck(result_root: Path, gpu: int) -> None:
    query = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=index,uuid,name,memory.total,memory.free", "--format=csv,noheader,nounits"], text=True,
    ).strip()
    fields = [field.strip() for field in query.split(",")]
    if len(fields) != 5:
        raise RuntimeError("GPU precheck output drift")
    free_mib = int(fields[4])
    disk = shutil.disk_usage(result_root.resolve().parent)
    passed = free_mib >= MIN_GPU_FREE_MIB and disk.free >= MIN_DISK_FREE_BYTES
    body = {
        "schema": f"{SCHEMA}_precheck", "status": "PASS_ONE_TIME_FAIL_FAST_PRECHECK" if passed else "FAIL_ONE_TIME_PRECHECK",
        "physical_gpu": gpu, "gpu_uuid": fields[1], "gpu_name": fields[2], "gpu_total_mib": int(fields[3]),
        "gpu_free_mib": free_mib, "minimum_gpu_free_mib": MIN_GPU_FREE_MIB,
        "disk_free_bytes": int(disk.free), "minimum_disk_free_bytes": MIN_DISK_FREE_BYTES,
        "resource_polling": False,
    }
    publish_json(result_root.resolve() / "precheck.json", body)
    if not passed:
        raise RuntimeError(f"one-time resource precheck failed: {body}")


def _failure(result_root: Path, phase: str, error: BaseException) -> None:
    path = result_root.resolve() / f"{phase}_failure.json"
    if result_root.exists() and not path.exists():
        publish_json(path, {"schema": SCHEMA, "status": "FAIL_IMMUTABLE_NO_AUTOMATIC_RETRY", "phase": phase, "error_type": type(error).__name__, "error": str(error), "automatic_retry": False, "docker_builds": 0, "evalai_submissions": 0})


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    phases = result.add_mutually_exclusive_group()
    phases.add_argument("--dry-run", action="store_true")
    phases.add_argument("--detached-supervisor", action="store_true")
    phases.add_argument("--verify-terminal", action="store_true")
    result.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    result.add_argument("--predecessor-root", type=Path, default=DEFAULT_PREDECESSOR_ROOT)
    result.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    result.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    result.add_argument("--physical-gpu", type=int, default=0)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.detached_supervisor and not args.verify_terminal:
        print(json.dumps(dry_plan(), indent=2, sort_keys=True))
        return 0
    if args.verify_terminal:
        _assert_closure(args.result_root)
        verify_terminal(args.result_root)
        return 0
    phase = "attempt"
    try:
        create_attempt(args.result_root, args.predecessor_root, _closure(), _head())
        phase = "cpu_gate"; _cpu_gate(args.result_root)
        phase = "precheck"; _precheck(args.result_root, args.physical_gpu)
        _assert_closure(args.result_root)
        os.environ.update({"PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": str(args.physical_gpu), "TQDM_DISABLE": "1"})
        print("START phase=source_authority", flush=True)
        phase = "source_authority"; prepare_source_authority(args.data_root, args.predecessor_root, args.result_root)
        print("START phase=c2_training", flush=True)
        phase = "c2_training"; run_c2_training(args.data_root, args.predecessor_root, args.result_root, args.physical_gpu)
        print("START phase=training_integrity", flush=True)
        phase = "training_integrity"; verify_training_integrity(args.predecessor_root, args.result_root)
        print("START phase=offline_dual_selection", flush=True)
        phase = "offline_dual_selection"; run_offline_validation(args.data_root, args.predecessor_root, args.result_root, device="cuda:0")
        print("START phase=terminal_verification", flush=True)
        phase = "terminal_verification"; terminal = verify_terminal(args.result_root)
        print(f"TERMINAL status={terminal['status']}", flush=True)
        return 0
    except BaseException as error:
        _failure(args.result_root, phase, error)
        print(f"ERROR phase={phase} type={type(error).__name__} message={error}", flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
