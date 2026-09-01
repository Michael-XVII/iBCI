#!/usr/bin/env python3
"""Fail-closed CLI for H1 CAL-AUG Prefix-Cycle V1."""
from __future__ import annotations

import argparse
import json
import math
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

from src.h1_cal_aug_prefix_cycle_v1 import (  # noqa: E402
    ARMS,
    ARM_TIMEOUT_SECONDS,
    RESOURCE_WAIT_SECONDS,
    SCHEMA,
    STATUS_EVAL,
    STATUS_SMOKE,
    create_attempt,
    dry_plan,
    load_attempt,
    prepare_source_authority,
    publish_json,
    run_arm,
    run_evaluation_cell,
    validate_predecessors,
    verify_all_pairs,
    verify_pair,
    verify_terminal,
)
from src.h1_hc_date_lodo_regen_v1 import seal_existing_log, verify_sidecar  # noqa: E402
from src.h1_m4_cce_contract import CONFIRMATORY_DATES, sha256_file  # noqa: E402


DEFAULT_DATA_ROOT = Path("/data/ial-dataset/ial-mohd/000954")
DEFAULT_PREDECESSOR_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_hc_date_lodo_regen_v1_detached_a2"
DEFAULT_EXPERIMENT3_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_causal_output_ema_v1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_prefix_cycle_v1"
DEFAULT_LOG_ROOT = REPO_ROOT / "logs/h1_cal_aug_prefix_cycle_v1"
WORK_ORDER = REPO_ROOT / "tfpd_exploration/h1_series_20260830/H1_CAL_AUG_PREFIX_CYCLE_V1_WORK_ORDER.md"
TEST_FILE = SPINT_ROOT / "tests/test_h1_cal_aug_prefix_cycle_v1.py"


def _closure() -> dict[str, str]:
    paths = (
        WORK_ORDER,
        Path(__file__).resolve(),
        SPINT_ROOT / "src/h1_cal_aug_prefix_cycle_v1.py",
        TEST_FILE,
        SPINT_ROOT / "src/h1_hc_date_lodo_regen_v1.py",
        SPINT_ROOT / "src/h1_causal_output_ema_v1.py",
        SPINT_ROOT / "src/models/components/h1_carrierid_spint.py",
        SPINT_ROOT / "src/data/h1_m4_eb_pilot.py",
        SPINT_ROOT / "src/h1_m4_cce_contract.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"code closure is incomplete: {missing}")
    return {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths}


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _load_json(path: Path) -> dict[str, Any]:
    verify_sidecar(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_closure(result_root: Path) -> None:
    attempt = load_attempt(result_root)
    if attempt.get("closure") != _closure() or attempt.get("head") != _head():
        raise RuntimeError("current Git HEAD/code closure differs from immutable attempt")


def _publish_failure(result_root: Path, phase: str, error: BaseException) -> None:
    path = result_root.resolve() / f"{phase}_failure.json"
    if path.exists() or not result_root.resolve().exists():
        return
    publish_json(path, {
        "schema": SCHEMA,
        "status": "FAIL_IMMUTABLE_NO_AUTOMATIC_RETRY",
        "phase": phase,
        "error_type": type(error).__name__,
        "error": str(error),
        "target_recordings_opened": 0 if phase not in {"evaluation", "terminal_verification"} else "see_cell_audits",
        "target_bytes_read": 0 if phase not in {"evaluation", "terminal_verification"} else "see_cell_audits",
    })


def _cpu_gate(result_root: Path) -> None:
    command = [sys.executable, "-m", "pytest", "-q", str(TEST_FILE)]
    environment = dict(os.environ)
    environment.update({"PYTHONNOUSERSITE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "CUDA_VISIBLE_DEVICES": ""})
    completed = subprocess.run(command, cwd=SPINT_ROOT, env=environment, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    from src.h1_hc_date_lodo_regen_v1 import publish_text
    log_sha = publish_text(result_root.resolve() / "cpu_gate.log", completed.stdout)
    body = {
        "schema": f"{SCHEMA}_cpu_gate",
        "status": "PASS_H1_CAL_AUG_PREFIX_CYCLE_V1_CPU_GATE" if completed.returncode == 0 else "FAIL_CPU_GATE",
        "command": command, "returncode": completed.returncode, "log_sha256": log_sha,
        "cuda_visible_devices": "", "nwb_files_opened": 0,
        "target_recordings_opened": 0, "target_bytes_read": 0,
    }
    publish_json(result_root.resolve() / "cpu_gate.json", body)
    if completed.returncode:
        raise RuntimeError("CPU gate failed")


def _parse_gpus(raw: str | None) -> tuple[int, ...]:
    if raw is None:
        raise RuntimeError("--physical-gpus is required")
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values or len(values) > 4 or len(set(values)) != len(values) or any(value < 0 for value in values):
        raise RuntimeError("--physical-gpus must contain one to four distinct indices")
    return values


def _gpu_row(index: int) -> dict[str, Any]:
    output = subprocess.check_output([
        "nvidia-smi", "-i", str(index),
        "--query-gpu=index,uuid,name,memory.total,memory.free,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    fields = [value.strip() for value in output.split(",")]
    if len(fields) != 7:
        raise RuntimeError(f"cannot parse GPU profile: {output}")
    return {
        "physical_index": int(fields[0]), "uuid": fields[1], "name": fields[2],
        "memory_total_mib": int(fields[3]), "memory_free_mib": int(fields[4]),
        "memory_used_mib": int(fields[5]), "utilization_percent": int(fields[6]),
    }


def _environment(gpu: int) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update({"PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": str(gpu)})
    return environment


def _base_args(args: argparse.Namespace) -> list[str]:
    return [
        "--data-root", str(args.data_root), "--predecessor-root", str(args.predecessor_root),
        "--experiment3-root", str(args.experiment3_root), "--result-root", str(args.result_root),
        "--log-root", str(args.log_root),
    ]


def _run_logged(command: list[str], log_path: Path, gpu: int, timeout: int) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("x", encoding="utf-8") as output:
            completed = subprocess.run(command, cwd=REPO_ROOT, env=_environment(gpu), text=True,
                                       stdout=output, stderr=subprocess.STDOUT, check=False, timeout=timeout)
        return completed.returncode
    finally:
        if log_path.is_file() and (log_path.stat().st_mode & 0o222):
            seal_existing_log(log_path)


def _arm_command(args: argparse.Namespace, date: str, arm: str, gpu: int, *, smoke: bool) -> list[str]:
    command = [sys.executable, str(Path(__file__).resolve()), "--arm-cell", "--outer-date", date,
               "--arm", arm, "--physical-gpu", str(gpu), *_base_args(args)]
    if smoke:
        command.extend(("--smoke-cell", "--smoke-steps", str(args.smoke_steps)))
    return command


def _eval_command(args: argparse.Namespace, date: str, gpu: int) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), "--eval-cell", "--outer-date", date,
            "--physical-gpu", str(gpu), *_base_args(args)]


def _wait_memory(gpu: int, threshold_mib: int, stop: threading.Event) -> dict[str, Any] | None:
    started = time.monotonic()
    while not stop.is_set():
        row = _gpu_row(gpu)
        if row["memory_free_mib"] >= threshold_mib:
            return row
        if time.monotonic() - started >= RESOURCE_WAIT_SECONDS:
            raise TimeoutError(f"GPU {gpu} resource wait exceeded {RESOURCE_WAIT_SECONDS}s")
        stop.wait(60.0)
    return None


def _run_smoke(args: argparse.Namespace, gpus: tuple[int, ...]) -> dict[str, Any]:
    wait_started = time.monotonic()
    gpu = None
    while gpu is None:
        gpu = next((value for value in gpus if _gpu_row(value)["memory_free_mib"] >= 8192), None)
        if gpu is not None:
            break
        if time.monotonic() - wait_started >= RESOURCE_WAIT_SECONDS:
            raise TimeoutError("paired-smoke GPU resource wait exceeded 24 hours")
        time.sleep(60.0)
    for arm in ARMS:
        log = args.log_root.resolve() / "smoke" / f"{arm}.log"
        rc = _run_logged(_arm_command(args, CONFIRMATORY_DATES[0], arm, gpu, smoke=True), log, gpu, ARM_TIMEOUT_SECONDS)
        if rc:
            raise RuntimeError(f"paired smoke arm failed: {arm}")
    body = verify_pair(args.result_root, CONFIRMATORY_DATES[0], smoke=True)
    if body.get("gpu", {}).get("physical_index") != gpu:
        raise RuntimeError("smoke GPU binding drift")
    return body


def _throughput_probe(args: argparse.Namespace) -> dict[str, Any]:
    rows = [_load_json(args.result_root.resolve() / "smoke" / CONFIRMATORY_DATES[0] / arm / "terminal.json") for arm in ARMS]
    source_rows = {}
    for date in CONFIRMATORY_DATES:
        source_rows[date] = _load_json(args.result_root.resolve() / "source_authority" / date / "authority.json")
    seconds_per_step = max(row["training_elapsed_seconds"] / row["global_step"] for row in rows)
    projected = {date: float(seconds_per_step * source_rows[date]["steps_per_arm"]) for date in CONFIRMATORY_DATES}
    peak = max(int(row["peak_memory_reserved_bytes"]) for row in rows)
    threshold_mib = max(8192, int(math.ceil(1.5 * peak / (1024 * 1024))))
    body = {
        "schema": f"{SCHEMA}_throughput_probe",
        "status": "PASS_H1_CAL_AUG_PREFIX_CYCLE_V1_THROUGHPUT_PROBE" if max(projected.values()) <= ARM_TIMEOUT_SECONDS else "STOP_THROUGHPUT_BUDGET",
        "smoke_steps_per_arm": rows[0]["global_step"], "seconds_per_step_worst_arm": seconds_per_step,
        "projected_seconds_per_arm_by_date": projected, "arm_timeout_seconds": ARM_TIMEOUT_SECONDS,
        "peak_memory_reserved_bytes": peak, "required_free_memory_mib": threshold_mib,
        "target_recordings_opened": 0, "target_bytes_read": 0,
    }
    publish_json(args.result_root.resolve() / "throughput_probe.json", body)
    if body["status"].startswith("STOP"):
        raise RuntimeError("throughput projection exceeds pre-registered arm timeout")
    return body


def _verify_final(args: argparse.Namespace) -> dict[str, Any]:
    live = validate_predecessors(args.experiment3_root, args.predecessor_root)
    receipt = _load_json(args.result_root.resolve() / "predecessor_authority.json")
    if live != receipt:
        raise RuntimeError("live predecessor authorities differ from immutable receipt")
    return verify_terminal(args.predecessor_root, args.result_root)


def _parallel_dates(args: argparse.Namespace, gpus: tuple[int, ...], *, evaluation: bool) -> None:
    threshold = int(_load_json(args.result_root.resolve() / "throughput_probe.json")["required_free_memory_mib"])
    pending: queue.Queue[str] = queue.Queue()
    for date in CONFIRMATORY_DATES:
        pending.put(date)
    stop = threading.Event()
    failures: list[str] = []
    lock = threading.Lock()

    def worker(gpu: int) -> None:
        try:
            wait_started = time.monotonic()
            while not stop.is_set():
                if pending.empty():
                    return
                profile = _gpu_row(gpu)
                if profile["memory_free_mib"] < threshold:
                    if time.monotonic() - wait_started >= RESOURCE_WAIT_SECONDS:
                        raise TimeoutError(f"GPU {gpu} resource wait exceeded {RESOURCE_WAIT_SECONDS}s")
                    stop.wait(60.0)
                    continue
                try:
                    date = pending.get_nowait()
                except queue.Empty:
                    return
                wait_started = time.monotonic()
                if evaluation:
                    log = args.log_root.resolve() / "evaluation" / f"{date}.log"
                    rc = _run_logged(_eval_command(args, date, gpu), log, gpu, ARM_TIMEOUT_SECONDS)
                    if rc:
                        raise RuntimeError(f"evaluation failed: {date}")
                    terminal = _load_json(args.result_root.resolve() / "evaluation" / date / "terminal.json")
                    if terminal.get("status") != STATUS_EVAL:
                        raise RuntimeError(f"evaluation terminal drift: {date}")
                else:
                    for arm in ARMS:
                        if stop.is_set():
                            pending.put(date)
                            return
                        log = args.log_root.resolve() / "pairs" / date / f"{arm}.log"
                        rc = _run_logged(_arm_command(args, date, arm, gpu, smoke=False), log, gpu, ARM_TIMEOUT_SECONDS)
                        if rc:
                            raise RuntimeError(f"training failed: {date}/{arm}")
                    verify_pair(args.result_root, date, smoke=False)
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
        raise RuntimeError(f"parallel phase failed; errors={failures}; pending={list(pending.queue)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_mutually_exclusive_group()
    phases.add_argument("--dry-run", action="store_true")
    phases.add_argument("--prepare-source-authority", action="store_true")
    phases.add_argument("--smoke", action="store_true")
    phases.add_argument("--throughput-probe", action="store_true")
    phases.add_argument("--train", action="store_true")
    phases.add_argument("--evaluate", action="store_true")
    phases.add_argument("--verify-terminal", action="store_true")
    phases.add_argument("--detached-supervisor", action="store_true")
    phases.add_argument("--arm-cell", action="store_true", help=argparse.SUPPRESS)
    phases.add_argument("--eval-cell", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--predecessor-root", type=Path, default=DEFAULT_PREDECESSOR_ROOT)
    parser.add_argument("--experiment3-root", type=Path, default=DEFAULT_EXPERIMENT3_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--physical-gpus", default="0,1,2,3")
    parser.add_argument("--allow-shared-gpus", action="store_true")
    parser.add_argument("--outer-date", choices=CONFIRMATORY_DATES, help=argparse.SUPPRESS)
    parser.add_argument("--arm", choices=ARMS, help=argparse.SUPPRESS)
    parser.add_argument("--physical-gpu", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--smoke-cell", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--smoke-steps", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not any((args.prepare_source_authority, args.smoke, args.throughput_probe, args.train, args.evaluate,
                args.verify_terminal, args.detached_supervisor, args.arm_cell, args.eval_cell)):
        print(json.dumps(dry_plan(), indent=2, sort_keys=True))
        return 0
    if args.arm_cell:
        if args.outer_date is None or args.arm is None or args.physical_gpu is None:
            raise RuntimeError("internal arm cell requires date, arm, and physical GPU")
        _assert_closure(args.result_root)
        print(f"START outer_date={args.outer_date} arm={args.arm} smoke={args.smoke_cell}", flush=True)
        body = run_arm(args.data_root, args.predecessor_root, args.result_root, args.outer_date, args.arm,
                       args.physical_gpu, smoke=args.smoke_cell, max_steps=args.smoke_steps)
        print(f"TERMINAL status={body['status']}", flush=True)
        return 0
    if args.eval_cell:
        if args.outer_date is None or args.physical_gpu is None:
            raise RuntimeError("internal eval cell requires date and physical GPU")
        _assert_closure(args.result_root)
        print(f"START evaluation outer_date={args.outer_date}", flush=True)
        body = run_evaluation_cell(args.data_root, args.predecessor_root, args.result_root, args.outer_date, args.physical_gpu)
        print(f"TERMINAL status={body['status']}", flush=True)
        return 0
    phase = "unknown"
    try:
        if args.prepare_source_authority:
            phase = "source_authority"
            create_attempt(args.result_root, _closure(), _head())
            _cpu_gate(args.result_root)
            prepare_source_authority(args.data_root, args.predecessor_root, args.experiment3_root, args.result_root)
        else:
            _assert_closure(args.result_root)
            gpus = _parse_gpus(args.physical_gpus)
            if not args.allow_shared_gpus and any(_gpu_row(gpu)["utilization_percent"] > 5 for gpu in gpus):
                raise RuntimeError("busy GPUs require explicit --allow-shared-gpus")
            if args.smoke:
                phase = "smoke"; _run_smoke(args, gpus)
            elif args.throughput_probe:
                phase = "throughput_probe"; _throughput_probe(args)
            elif args.train:
                phase = "training"; _parallel_dates(args, gpus, evaluation=False); verify_all_pairs(args.result_root)
            elif args.evaluate:
                phase = "evaluation"; _parallel_dates(args, gpus, evaluation=True)
            elif args.verify_terminal:
                phase = "terminal_verification"; _verify_final(args)
            elif args.detached_supervisor:
                if not (args.result_root.resolve() / "smoke" / CONFIRMATORY_DATES[0] / "paired_integrity.json").exists():
                    phase = "smoke"; _run_smoke(args, gpus)
                if not (args.result_root.resolve() / "throughput_probe.json").exists():
                    phase = "throughput_probe"; _throughput_probe(args)
                phase = "training"; _parallel_dates(args, gpus, evaluation=False); verify_all_pairs(args.result_root)
                phase = "evaluation"; _parallel_dates(args, gpus, evaluation=True)
                phase = "terminal_verification"; _verify_final(args)
        return 0
    except BaseException as error:
        _publish_failure(args.result_root, phase, error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
