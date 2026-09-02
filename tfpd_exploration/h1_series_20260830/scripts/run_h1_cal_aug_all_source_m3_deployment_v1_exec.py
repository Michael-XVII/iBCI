#!/usr/bin/env python3
"""Fail-closed local executor for H1 all-source M3 deployment V1."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SPINT_ROOT = REPO_ROOT / "SPINT-main"
if str(SPINT_ROOT) not in sys.path:
    sys.path.insert(0, str(SPINT_ROOT))

from src.h1_cal_aug_all_source_m3_deployment_v1_contract import ARMS, dry_plan  # noqa: E402
from src.h1_cal_aug_all_source_m3_deployment_v1_exec import (  # noqa: E402
    SCHEMA,
    build_packages,
    create_attempt,
    load_attempt,
    prepare_source_authority,
    publish_json,
    run_arm,
    run_local_minival,
    run_package_rehearsal,
    verify_pair,
    verify_terminal,
)
from src.h1_hc_date_lodo_regen_v1 import publish_text, seal_existing_log, verify_sidecar  # noqa: E402
from src.h1_m4_cce_contract import sha256_file  # noqa: E402


DEFAULT_DATA_ROOT = Path("/data/ial-dataset/ial-mohd/000954")
DEFAULT_RESULT_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_all_source_m3_deployment_v1"
DEFAULT_LOG_ROOT = REPO_ROOT / "logs/h1_cal_aug_all_source_m3_deployment_v1"
WORK_ORDER = REPO_ROOT / "tfpd_exploration/h1_series_20260830/H1_CAL_AUG_ALL_SOURCE_M3_DEPLOYMENT_V1_WORK_ORDER.md"
AMENDMENT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/docs/AMENDMENT_H1_CAL_AUG_ALL_SOURCE_M3_DEPLOYMENT_V1_EXECUTION.md"
TESTS = (
    SPINT_ROOT / "tests/test_h1_cal_aug_all_source_m3_deployment_v1_contract.py",
    SPINT_ROOT / "tests/test_h1_cal_aug_all_source_m3_deployment_v1_exec.py",
)
ARM_TIMEOUT_SECONDS = 12 * 60 * 60
RESOURCE_WAIT_SECONDS = 24 * 60 * 60


def _closure() -> dict[str, str]:
    paths = (
        WORK_ORDER, AMENDMENT, Path(__file__).resolve(),
        SPINT_ROOT / "src/h1_cal_aug_all_source_m3_deployment_v1_contract.py",
        SPINT_ROOT / "src/h1_cal_aug_all_source_m3_deployment_v1_exec.py",
        SPINT_ROOT / "third_party/falcon_challenge/h1_carrier_id_spint_decoder.py",
        *TESTS,
        SPINT_ROOT / "src/h1_hc_date_lodo_regen_v1.py",
        SPINT_ROOT / "src/h1_cal_aug_prefix_cycle_v1.py",
        SPINT_ROOT / "src/models/components/h1_carrierid_spint.py",
        SPINT_ROOT / "src/data/h1_m4_eb_pilot.py",
        SPINT_ROOT / "src/h1_m4_cce_contract.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"code closure incomplete: {missing}")
    return {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths}


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _assert_closure(result_root: Path) -> None:
    attempt = load_attempt(result_root)
    if attempt["closure"] != _closure() or attempt["head"] != _head():
        raise RuntimeError("Git HEAD/code closure differs from immutable attempt")


def _cpu_gate(result_root: Path) -> None:
    command = [sys.executable, "-m", "pytest", "-q", *[str(path) for path in TESTS]]
    environment = dict(os.environ)
    environment.update({"PYTHONNOUSERSITE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "CUDA_VISIBLE_DEVICES": ""})
    completed = subprocess.run(command, cwd=SPINT_ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log_sha = publish_text(result_root.resolve() / "cpu_gate.log", completed.stdout)
    publish_json(result_root.resolve() / "cpu_gate.json", {
        "schema": f"{SCHEMA}_cpu_gate", "status": "PASS_CPU_NO_DATA_GATE" if completed.returncode == 0 else "FAIL_CPU_GATE",
        "returncode": completed.returncode, "command": command, "log_sha256": log_sha,
        "cuda_visible_devices": "", "nwb_files_opened": 0, "cuda_initialized": False,
        "evalai_submissions": 0,
    })
    if completed.returncode:
        raise RuntimeError("CPU gate failed")


def _environment(gpu: int) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update({"PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": str(gpu), "TQDM_DISABLE": "1"})
    return environment


def _gpu_free_mib(gpu: int) -> int:
    return int(subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())


def _wait_gpu(gpu: int) -> None:
    started = time.monotonic()
    while _gpu_free_mib(gpu) < 8192:
        if time.monotonic() - started >= RESOURCE_WAIT_SECONDS:
            raise TimeoutError(f"GPU{gpu} free-memory wait exceeded 24 hours")
        time.sleep(60)


def _run_logged(command: list[str], path: Path, gpu: int, timeout: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as output:
            completed = subprocess.run(command, cwd=REPO_ROOT, env=_environment(gpu), stdout=output, stderr=subprocess.STDOUT, text=True, check=False, timeout=timeout)
        return completed.returncode
    finally:
        if path.is_file() and path.stat().st_mode & 0o222:
            seal_existing_log(path)


def _base(args: argparse.Namespace) -> list[str]:
    return ["--data-root", str(args.data_root), "--result-root", str(args.result_root), "--log-root", str(args.log_root)]


def _run_pair_processes(args: argparse.Namespace, *, smoke: bool) -> None:
    failures = []
    lock = threading.Lock()
    def worker(arm: str, gpu: int) -> None:
        try:
            _wait_gpu(gpu)
            command = [sys.executable, str(Path(__file__).resolve()), "--arm-cell", "--arm", arm, "--physical-gpu", str(gpu), *_base(args)]
            if smoke:
                command.extend(["--smoke-cell", "--smoke-steps", str(args.smoke_steps)])
            log = args.log_root.resolve() / ("smoke" if smoke else "training") / f"{arm}.log"
            rc = _run_logged(command, log, gpu, ARM_TIMEOUT_SECONDS)
            if rc:
                raise RuntimeError(f"{arm} {'smoke' if smoke else 'training'} exited {rc}")
        except BaseException as error:
            with lock:
                failures.append(f"{arm}/GPU{gpu}: {type(error).__name__}: {error}")
    threads = [threading.Thread(target=worker, args=(arm, gpu), daemon=False) for arm, gpu in zip(ARMS, (0, 1), strict=True)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    if failures:
        raise RuntimeError(f"paired phase failed without retry: {failures}")
    verify_pair(args.result_root, smoke=smoke)


def _publish_failure(result_root: Path, phase: str, error: BaseException) -> None:
    path = result_root.resolve() / f"{phase}_failure.json"
    if path.exists() or not result_root.exists():
        return
    publish_json(path, {"schema": SCHEMA, "status": "FAIL_IMMUTABLE_NO_AUTOMATIC_RETRY", "phase": phase, "error_type": type(error).__name__, "error": str(error), "evalai_submissions": 0})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_mutually_exclusive_group()
    phases.add_argument("--dry-run", action="store_true")
    phases.add_argument("--initialize", action="store_true")
    phases.add_argument("--prepare-source-authority", action="store_true")
    phases.add_argument("--detached-supervisor", action="store_true")
    phases.add_argument("--verify-terminal", action="store_true")
    phases.add_argument("--arm-cell", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--smoke-cell", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--smoke-steps", type=int, default=12)
    parser.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--physical-gpu", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not any((args.initialize, args.prepare_source_authority, args.detached_supervisor, args.verify_terminal, args.arm_cell)):
        print(json.dumps(dry_plan(), indent=2, sort_keys=True)); return 0
    if args.arm_cell:
        if args.arm is None or args.physical_gpu is None:
            raise RuntimeError("internal arm cell requires arm/GPU")
        _assert_closure(args.result_root)
        print(f"START arm={args.arm} smoke={args.smoke_cell}", flush=True)
        body = run_arm(args.data_root, args.result_root, args.arm, args.physical_gpu, smoke=args.smoke_cell, max_steps=args.smoke_steps)
        print(f"TERMINAL status={body['status']}", flush=True)
        return 0
    phase = "initialize"
    try:
        if args.initialize:
            create_attempt(args.result_root, _closure(), _head()); _cpu_gate(args.result_root); return 0
        _assert_closure(args.result_root)
        if args.prepare_source_authority:
            prepare_source_authority(args.data_root, args.result_root); return 0
        if args.verify_terminal:
            verify_terminal(args.result_root); return 0
        phase = "source_authority"; prepare_source_authority(args.data_root, args.result_root)
        phase = "paired_smoke"; _run_pair_processes(args, smoke=True)
        phase = "training"; _run_pair_processes(args, smoke=False)
        phase = "packages"; build_packages(args.data_root, args.result_root)
        phase = "minival"; run_local_minival(args.data_root, args.result_root, device="cuda:0")
        phase = "package_rehearsal"; run_package_rehearsal(args.data_root, args.result_root)
        phase = "terminal_verification"; verify_terminal(args.result_root)
        return 0
    except BaseException as error:
        _publish_failure(args.result_root, phase, error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
