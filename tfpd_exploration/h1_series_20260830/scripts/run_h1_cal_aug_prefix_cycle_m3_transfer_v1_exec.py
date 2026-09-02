#!/usr/bin/env python3
"""Fail-closed formal runner for H1 CAL-AUG Prefix-Cycle M3 Transfer V1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SPINT_ROOT = REPO_ROOT / "SPINT-main"
if str(SPINT_ROOT) not in sys.path:
    sys.path.insert(0, str(SPINT_ROOT))

from src.h1_cal_aug_prefix_cycle_m3_transfer_v1 import dry_plan  # noqa: E402
from src.h1_cal_aug_prefix_cycle_m3_transfer_v1_exec import (  # noqa: E402
    SCHEMA,
    STATUS_CELL,
    create_attempt,
    load_attempt,
    prepare_predecessor_authority,
    publish_json,
    run_evaluation_cell,
    verify_terminal,
)
from src.h1_hc_date_lodo_regen_v1 import publish_text, seal_existing_log, verify_sidecar  # noqa: E402
from src.h1_m4_cce_contract import CONFIRMATORY_DATES, sha256_file  # noqa: E402


DEFAULT_DATA_ROOT = Path("/data/ial-dataset/ial-mohd/000954")
DEFAULT_REGEN_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_hc_date_lodo_regen_v1_detached_a2"
DEFAULT_EXPERIMENT3_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_causal_output_ema_v1"
DEFAULT_TRAINING_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_prefix_cycle_v1"
DEFAULT_EVAL_A1_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_prefix_cycle_v1_eval_a1"
DEFAULT_M4_AUDIT_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_all_source_heldout_v1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_prefix_cycle_m3_transfer_v1"
DEFAULT_LOG_ROOT = REPO_ROOT / "logs/h1_cal_aug_prefix_cycle_m3_transfer_v1"
WORK_ORDER = REPO_ROOT / "tfpd_exploration/h1_series_20260830/H1_CAL_AUG_PREFIX_CYCLE_M3_TRANSFER_V1_WORK_ORDER.md"
EXECUTION_AMENDMENT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/docs/AMENDMENT_H1_CAL_AUG_PREFIX_CYCLE_M3_TRANSFER_V1_EXECUTION.md"
REVIEW_TEST = SPINT_ROOT / "tests/test_h1_cal_aug_prefix_cycle_m3_transfer_v1.py"
EXEC_TEST = SPINT_ROOT / "tests/test_h1_cal_aug_prefix_cycle_m3_transfer_v1_exec.py"
EVALUATION_TIMEOUT_SECONDS = 8 * 60 * 60
RESOURCE_WAIT_SECONDS = 24 * 60 * 60


def _closure() -> dict[str, str]:
    paths = (
        WORK_ORDER,
        EXECUTION_AMENDMENT,
        Path(__file__).resolve(),
        SPINT_ROOT / "src/h1_cal_aug_prefix_cycle_m3_transfer_v1.py",
        SPINT_ROOT / "src/h1_cal_aug_prefix_cycle_m3_transfer_v1_exec.py",
        REVIEW_TEST,
        EXEC_TEST,
        SPINT_ROOT / "src/h1_cal_aug_prefix_cycle_eval_a1.py",
        SPINT_ROOT / "src/h1_cal_aug_prefix_cycle_v1.py",
        SPINT_ROOT / "src/h1_hc_date_lodo_regen_v1.py",
        SPINT_ROOT / "src/data/h1_m4_eb_pilot.py",
        SPINT_ROOT / "src/models/components/h1_carrierid_spint.py",
        SPINT_ROOT / "src/h1_m4_cce_contract.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"M3 execution closure is incomplete: {missing}")
    return {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths}


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _load_json(path: Path) -> dict[str, Any]:
    verify_sidecar(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_closure(result_root: Path) -> None:
    attempt = load_attempt(result_root)
    if attempt.get("head") != _head() or attempt.get("closure") != _closure():
        raise RuntimeError("current Git HEAD/code closure differs from immutable M3 attempt")


def _run_cpu_gate_no_write() -> tuple[str, str, list[str]]:
    command = [sys.executable, "-m", "pytest", "-q", str(REVIEW_TEST), str(EXEC_TEST)]
    environment = dict(os.environ)
    environment.update({"PYTHONNOUSERSITE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "CUDA_VISIBLE_DEVICES": ""})
    completed = subprocess.run(
        command,
        cwd=SPINT_ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    digest = hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest()
    if completed.returncode:
        raise RuntimeError(f"formal M3 CPU gate failed:\n{completed.stdout}")
    return completed.stdout, digest, command


def _publish_cpu_gate(result_root: Path, stdout: str, digest: str, command: list[str]) -> None:
    log_sha = publish_text(result_root.resolve() / "cpu_gate.log", stdout)
    if log_sha != digest:
        raise RuntimeError("CPU gate captured stdout SHA drift")
    publish_json(result_root.resolve() / "cpu_gate.json", {
        "schema": f"{SCHEMA}_cpu_gate",
        "status": "PASS_H1_CAL_AUG_PREFIX_CYCLE_M3_TRANSFER_V1_CPU_GATE",
        "command": command,
        "returncode": 0,
        "log_sha256": log_sha,
        "executed_before_attempt": True,
        "cuda_visible_devices": "",
        "nwb_files_opened": 0,
        "checkpoint_files_opened": 0,
        "target_recordings_opened": 0,
        "heldout_calib_recordings_opened": 0,
    })


def _publish_failure(result_root: Path, phase: str, error: BaseException) -> None:
    root = result_root.resolve()
    path = root / f"{phase}_failure.json"
    if not root.exists() or path.exists():
        return
    publish_json(path, {
        "schema": SCHEMA,
        "status": "FAIL_H1_M3_IMMUTABLE_NO_AUTOMATIC_RETRY",
        "phase": phase,
        "error_type": type(error).__name__,
        "error": str(error),
        "optimizer_steps": 0,
        "backward_steps": 0,
        "parameter_updates": 0,
        "automatic_retry": False,
    })


def _parse_gpus(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values or len(values) > 4 or len(values) != len(set(values)) or any(value < 0 for value in values):
        raise RuntimeError("--physical-gpus must contain one to four distinct non-negative indices")
    return values


def _gpu_row(index: int) -> dict[str, Any]:
    output = subprocess.check_output([
        "nvidia-smi", "-i", str(index),
        "--query-gpu=index,uuid,name,memory.total,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    values = [value.strip() for value in output.split(",")]
    if len(values) != 7:
        raise RuntimeError(f"cannot parse GPU profile: {output}")
    return {
        "physical_index": int(values[0]), "uuid": values[1], "name": values[2],
        "memory_total_mib": int(values[3]), "memory_free_mib": int(values[4]),
        "memory_used_mib": int(values[5]), "utilization_percent": int(values[6]),
    }


def _environment(gpu: int) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update({"PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": str(gpu)})
    return environment


def _base_args(args: argparse.Namespace) -> list[str]:
    return [
        "--data-root", str(args.data_root),
        "--regen-root", str(args.regen_root),
        "--experiment3-root", str(args.experiment3_root),
        "--training-root", str(args.training_root),
        "--eval-a1-root", str(args.eval_a1_root),
        "--m4-audit-root", str(args.m4_audit_root),
        "--result-root", str(args.result_root),
        "--log-root", str(args.log_root),
    ]


def _eval_command(args: argparse.Namespace, date: str, gpu: int) -> list[str]:
    return [
        sys.executable, str(Path(__file__).resolve()), "--eval-cell",
        "--outer-date", date, "--physical-gpu", str(gpu), *_base_args(args),
    ]


def _run_logged(command: list[str], log_path: Path, gpu: int) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("x", encoding="utf-8") as output:
            completed = subprocess.run(
                command, cwd=REPO_ROOT, env=_environment(gpu), text=True,
                stdout=output, stderr=subprocess.STDOUT, check=False, timeout=EVALUATION_TIMEOUT_SECONDS,
            )
        return completed.returncode
    finally:
        if log_path.is_file() and (log_path.stat().st_mode & 0o222):
            seal_existing_log(log_path)


def _evaluate_all(args: argparse.Namespace, gpus: tuple[int, ...]) -> None:
    pending: queue.Queue[str] = queue.Queue()
    for date in CONFIRMATORY_DATES:
        pending.put(date)
    stop = threading.Event()
    failures: list[str] = []
    lock = threading.Lock()

    def worker(gpu: int) -> None:
        wait_started = time.monotonic()
        try:
            while not stop.is_set():
                if pending.empty():
                    return
                if _gpu_row(gpu)["memory_free_mib"] < 8192:
                    if time.monotonic() - wait_started >= RESOURCE_WAIT_SECONDS:
                        raise TimeoutError(f"GPU {gpu} resource wait exceeded 24 hours")
                    stop.wait(60.0)
                    continue
                try:
                    date = pending.get_nowait()
                except queue.Empty:
                    return
                wait_started = time.monotonic()
                returncode = _run_logged(
                    _eval_command(args, date, gpu),
                    args.log_root.resolve() / "evaluation" / f"{date}.log",
                    gpu,
                )
                if returncode:
                    raise RuntimeError(f"M3 evaluation failed: {date}")
                terminal = _load_json(args.result_root.resolve() / "evaluation" / date / "terminal.json")
                if terminal.get("status") != STATUS_CELL:
                    raise RuntimeError(f"M3 cell terminal drift: {date}")
                pending.task_done()
        except BaseException as error:
            with lock:
                failures.append(f"GPU{gpu}: {type(error).__name__}: {error}")
            stop.set()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures or not pending.empty():
        raise RuntimeError(f"parallel M3 evaluation failed; errors={failures}; pending={list(pending.queue)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_mutually_exclusive_group()
    phases.add_argument("--dry-run", action="store_true")
    phases.add_argument("--prepare-authority", action="store_true")
    phases.add_argument("--evaluate", action="store_true")
    phases.add_argument("--verify-terminal", action="store_true")
    phases.add_argument("--detached-supervisor", action="store_true")
    phases.add_argument("--eval-cell", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--regen-root", type=Path, default=DEFAULT_REGEN_ROOT)
    parser.add_argument("--experiment3-root", type=Path, default=DEFAULT_EXPERIMENT3_ROOT)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--eval-a1-root", type=Path, default=DEFAULT_EVAL_A1_ROOT)
    parser.add_argument("--m4-audit-root", type=Path, default=DEFAULT_M4_AUDIT_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--physical-gpus", default="0,1")
    parser.add_argument("--allow-shared-gpus", action="store_true")
    parser.add_argument("--outer-date", choices=CONFIRMATORY_DATES, help=argparse.SUPPRESS)
    parser.add_argument("--physical-gpu", type=int, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not any((args.prepare_authority, args.evaluate, args.verify_terminal, args.detached_supervisor, args.eval_cell)):
        print(json.dumps({**dry_plan(), "formal_execution_layer_available": True}, indent=2, sort_keys=True))
        return 0
    if args.eval_cell:
        if args.outer_date is None or args.physical_gpu is None:
            raise RuntimeError("internal M3 cell requires date and physical GPU")
        _assert_closure(args.result_root)
        print(f"START M3_EVALUATION outer_date={args.outer_date}", flush=True)
        body = run_evaluation_cell(
            args.data_root, args.regen_root, args.training_root, args.result_root, args.outer_date, args.physical_gpu
        )
        print(f"TERMINAL status={body['status']}", flush=True)
        return 0
    phase = "unknown"
    try:
        if args.prepare_authority:
            phase = "cpu_gate"
            stdout, digest, command = _run_cpu_gate_no_write()
            phase = "attempt"
            create_attempt(args.result_root, _closure(), _head(), digest)
            _publish_cpu_gate(args.result_root, stdout, digest, command)
            phase = "predecessor_authority"
            prepare_predecessor_authority(
                args.training_root, args.eval_a1_root, args.m4_audit_root,
                args.regen_root, args.experiment3_root, args.result_root,
            )
            return 0
        _assert_closure(args.result_root)
        gpus = _parse_gpus(args.physical_gpus)
        if not args.allow_shared_gpus and any(_gpu_row(gpu)["utilization_percent"] > 5 for gpu in gpus):
            raise RuntimeError("busy GPUs require explicit --allow-shared-gpus")
        if args.evaluate:
            phase = "evaluation"
            _evaluate_all(args, gpus)
        elif args.verify_terminal:
            phase = "terminal_verification"
            verify_terminal(
                args.training_root, args.eval_a1_root, args.m4_audit_root,
                args.regen_root, args.experiment3_root, args.result_root,
            )
        elif args.detached_supervisor:
            phase = "evaluation"
            _evaluate_all(args, gpus)
            phase = "terminal_verification"
            verify_terminal(
                args.training_root, args.eval_a1_root, args.m4_audit_root,
                args.regen_root, args.experiment3_root, args.result_root,
            )
        return 0
    except BaseException as error:
        _publish_failure(args.result_root, phase, error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
