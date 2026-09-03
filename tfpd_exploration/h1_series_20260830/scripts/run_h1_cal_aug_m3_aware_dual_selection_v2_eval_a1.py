#!/usr/bin/env python3
"""Evaluation-only dual-GPU repair executor for H1 M3-aware V2."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading


REPO_ROOT = Path(__file__).resolve().parents[3]
SPINT_ROOT = REPO_ROOT / "SPINT-main"
if str(SPINT_ROOT) not in sys.path:
    sys.path.insert(0, str(SPINT_ROOT))

from src.h1_cal_aug_m3_aware_dual_selection_v2_eval_a1 import (  # noqa: E402
    SCHEMA, combine_and_verify, create_attempt, dry_plan, load_attempt,
    prepare_calibration_payloads, run_surface, validate_training_predecessor,
)
from src.h1_hc_date_lodo_regen_v1 import publish_json, publish_text, seal_existing_log  # noqa: E402
from src.h1_m4_cce_contract import sha256_file  # noqa: E402


DATA_ROOT = Path("/data/ial-dataset/ial-mohd/000954")
V1_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_all_source_m3_deployment_v1"
TRAINING_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_m3_aware_dual_selection_v2_a1"
RESULT_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_m3_aware_dual_selection_v2_eval_a1"
LOG_ROOT = REPO_ROOT / "logs/h1_cal_aug_m3_aware_dual_selection_v2_eval_a1"
WORK_ORDER = REPO_ROOT / "tfpd_exploration/h1_series_20260830/H1_CAL_AUG_M3_AWARE_DUAL_SELECTION_V2_WORK_ORDER.md"
AMENDMENT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/docs/AMENDMENT_H1_CAL_AUG_M3_AWARE_DUAL_SELECTION_V2_EVAL_A1.md"
TESTS = (
    SPINT_ROOT / "tests/test_h1_cal_aug_m3_aware_dual_selection_v2.py",
    SPINT_ROOT / "tests/test_h1_cal_aug_m3_aware_dual_selection_v2_eval_a1.py",
)


def _closure() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(), WORK_ORDER, AMENDMENT, *TESTS,
        SPINT_ROOT / "src/h1_cal_aug_m3_aware_dual_selection_v2_contract.py",
        SPINT_ROOT / "src/h1_cal_aug_m3_aware_dual_selection_v2_exec.py",
        SPINT_ROOT / "src/h1_cal_aug_m3_aware_dual_selection_v2_eval_a1.py",
        SPINT_ROOT / "src/h1_cal_aug_all_source_m3_deployment_v1_contract.py",
        SPINT_ROOT / "src/h1_cal_aug_all_source_m3_deployment_v1_exec.py",
        SPINT_ROOT / "third_party/falcon_challenge/h1_carrier_id_spint_decoder.py",
        SPINT_ROOT / "src/h1_hc_date_lodo_regen_v1.py",
        SPINT_ROOT / "src/data/h1_m4_eb_pilot.py",
        SPINT_ROOT / "src/models/components/h1_carrierid_spint.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"evaluation code closure incomplete: {missing}")
    return {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths}


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _assert_closure(result_root: Path) -> None:
    attempt = load_attempt(result_root)
    if attempt["git_head"] != _head() or attempt["closure"] != _closure():
        raise RuntimeError("Git HEAD/code closure differs from immutable evaluation attempt")


def _cpu_gate(result_root: Path) -> None:
    command = [sys.executable, "-m", "pytest", "-q", *map(str, TESTS)]
    environment = dict(os.environ)
    environment.update({"PYTHONNOUSERSITE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "CUDA_VISIBLE_DEVICES": "", "TQDM_DISABLE": "1"})
    completed = subprocess.run(command, cwd=SPINT_ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log_sha = publish_text(result_root.resolve() / "cpu_gate.log", completed.stdout)
    publish_json(result_root.resolve() / "cpu_gate.json", {"schema": f"{SCHEMA}_cpu_gate",
        "status": "PASS_CPU_NO_DATA_GATE" if completed.returncode == 0 else "FAIL_CPU_GATE",
        "returncode": completed.returncode, "log_sha256": log_sha, "nwb_files_opened": 0,
        "cuda_initialized": False, "training": False, "evalai_submissions": 0})
    if completed.returncode:
        raise RuntimeError("evaluation CPU gate failed")


def _precheck(result_root: Path) -> None:
    rows = []
    for gpu in (0, 1):
        query = subprocess.check_output(["nvidia-smi", "-i", str(gpu),
            "--query-gpu=index,uuid,name,memory.total,memory.free", "--format=csv,noheader,nounits"], text=True).strip()
        fields = [value.strip() for value in query.split(",")]
        if len(fields) != 5:
            raise RuntimeError("GPU precheck output drift")
        rows.append({"physical_gpu": gpu, "uuid": fields[1], "name": fields[2],
                     "total_mib": int(fields[3]), "free_mib": int(fields[4])})
    free = shutil.disk_usage(result_root.resolve().parent).free
    passed = all(row["free_mib"] >= 8192 for row in rows) and free >= 1024**3
    publish_json(result_root.resolve() / "precheck.json", {"schema": f"{SCHEMA}_precheck",
        "status": "PASS_ONE_TIME_DUAL_GPU_PRECHECK" if passed else "FAIL_ONE_TIME_PRECHECK",
        "gpus": rows, "disk_free_bytes": int(free), "gpu_resource_polling": False})
    if not passed:
        raise RuntimeError("dual-GPU fail-fast precheck failed")


def _surface_process(args: argparse.Namespace, surface: str, gpu: int, failures: list[str], lock: threading.Lock) -> None:
    log = args.log_root.resolve() / f"{surface}.log"
    command = [sys.executable, str(Path(__file__).resolve()), "--surface-cell", "--surface", surface,
               "--physical-gpu", str(gpu), "--data-root", str(args.data_root),
               "--v1-root", str(args.v1_root), "--training-root", str(args.training_root),
               "--result-root", str(args.result_root), "--log-root", str(args.log_root)]
    environment = dict(os.environ)
    environment.update({"PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1", "TQDM_DISABLE": "1", "CUDA_VISIBLE_DEVICES": str(gpu)})
    try:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x", encoding="utf-8") as output:
            completed = subprocess.run(command, cwd=REPO_ROOT, env=environment, stdout=output, stderr=subprocess.STDOUT, text=True, check=False)
        seal_existing_log(log)
        if completed.returncode:
            raise RuntimeError(f"{surface}/GPU{gpu} exited {completed.returncode}")
    except BaseException as error:
        with lock:
            failures.append(f"{surface}/GPU{gpu}: {type(error).__name__}: {error}")


def _run_parallel_surfaces(args: argparse.Namespace) -> None:
    failures: list[str] = []
    lock = threading.Lock()
    threads = [
        threading.Thread(target=_surface_process, args=(args, "hi", 0, failures, lock), daemon=False),
        threading.Thread(target=_surface_process, args=(args, "ho", 1, failures, lock), daemon=False),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise RuntimeError(f"evaluation surface failure without retry: {failures}")


def _failure(result_root: Path, phase: str, error: BaseException) -> None:
    path = result_root.resolve() / f"{phase}_failure.json"
    if result_root.exists() and not path.exists():
        publish_json(path, {"schema": SCHEMA, "status": "FAIL_EVALUATION_A1_NO_AUTOMATIC_RETRY",
            "phase": phase, "error_type": type(error).__name__, "error": str(error),
            "training": False, "optimizer_steps": 0, "backward_steps": 0,
            "model_updates": 0, "evalai_submissions": 0})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_mutually_exclusive_group()
    phases.add_argument("--dry-run", action="store_true")
    phases.add_argument("--detached-supervisor", action="store_true")
    phases.add_argument("--surface-cell", action="store_true", help=argparse.SUPPRESS)
    phases.add_argument("--verify-terminal", action="store_true")
    parser.add_argument("--surface", choices=("hi", "ho"), help=argparse.SUPPRESS)
    parser.add_argument("--physical-gpu", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--v1-root", type=Path, default=V1_ROOT)
    parser.add_argument("--training-root", type=Path, default=TRAINING_ROOT)
    parser.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--log-root", type=Path, default=LOG_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.surface_cell:
        if args.surface is None or args.physical_gpu is None:
            raise RuntimeError("surface cell requires surface/GPU")
        _assert_closure(args.result_root)
        print(f"START surface={args.surface} physical_gpu={args.physical_gpu}", flush=True)
        body = run_surface(args.surface, args.data_root, args.training_root, args.v1_root, args.result_root, args.physical_gpu)
        print(f"TERMINAL surface={args.surface} status={body['status']}", flush=True)
        return 0
    if args.verify_terminal:
        _assert_closure(args.result_root)
        combine_and_verify(args.result_root)
        return 0
    if not args.detached_supervisor:
        print(json.dumps(dry_plan(), indent=2, sort_keys=True))
        return 0
    phase = "attempt"
    try:
        create_attempt(args.result_root, args.training_root, _closure(), _head())
        phase = "cpu_gate"; _cpu_gate(args.result_root)
        phase = "precheck"; _precheck(args.result_root)
        _assert_closure(args.result_root)
        print("START phase=predecessor_authority", flush=True)
        phase = "predecessor_authority"; validate_training_predecessor(args.training_root, args.v1_root, args.result_root)
        print("START phase=m3_calibration_authority", flush=True)
        phase = "m3_calibration_authority"; prepare_calibration_payloads(args.data_root, args.v1_root, args.result_root)
        print("START phase=parallel_hi_ho", flush=True)
        phase = "parallel_hi_ho"; _run_parallel_surfaces(args)
        print("START phase=dual_selection_terminal", flush=True)
        phase = "dual_selection_terminal"; terminal = combine_and_verify(args.result_root)
        print(f"TERMINAL status={terminal['status']}", flush=True)
        return 0
    except BaseException as error:
        _failure(args.result_root, phase, error)
        print(f"ERROR phase={phase} type={type(error).__name__} message={error}", flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
