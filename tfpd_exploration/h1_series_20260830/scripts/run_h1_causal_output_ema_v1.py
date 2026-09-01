#!/usr/bin/env python3
"""Fail-closed CLI for H1 five-date causal output EMA V1."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SPINT_ROOT = REPO_ROOT / "SPINT-main"
if str(SPINT_ROOT) not in sys.path:
    sys.path.insert(0, str(SPINT_ROOT))

from src.h1_causal_output_ema_v1 import (  # noqa: E402
    SCHEMA,
    STATUS_CELL,
    STATUS_SMOKE,
    create_attempt,
    dry_plan,
    load_attempt,
    publish_predecessor_authority,
    run_evaluation_cell,
    run_smoke_cell,
    validate_predecessor,
    verify_terminal,
)
from src.h1_hc_date_lodo_regen_v1 import publish_json, publish_text, seal_existing_log, verify_sidecar  # noqa: E402
from src.h1_m4_cce_contract import CONFIRMATORY_DATES, sha256_file  # noqa: E402


DEFAULT_DATA_ROOT = Path("/data/ial-dataset/ial-mohd/000954")
DEFAULT_PREDECESSOR_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_hc_date_lodo_regen_v1_detached_a2"
DEFAULT_RESULT_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_causal_output_ema_v1"
DEFAULT_LOG_ROOT = REPO_ROOT / "logs/h1_causal_output_ema_v1"
WORK_ORDER = REPO_ROOT / "tfpd_exploration/h1_series_20260830/H1_CAUSAL_OUTPUT_EMA_V1_WORK_ORDER.md"
TEST_FILE = SPINT_ROOT / "tests/test_h1_causal_output_ema_v1.py"


def _closure() -> dict[str, str]:
    paths = (
        WORK_ORDER,
        Path(__file__).resolve(),
        SPINT_ROOT / "src/h1_causal_output_ema_v1.py",
        TEST_FILE,
        SPINT_ROOT / "src/h1_hc_date_lodo_regen_v1.py",
        SPINT_ROOT / "src/models/components/h1_carrierid_spint.py",
        SPINT_ROOT / "src/data/h1_m4_eb_pilot.py",
        SPINT_ROOT / "src/data/h1_m4_cce_date_lodo.py",
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


def _assert_code_closure(result_root: Path) -> None:
    attempt = load_attempt(result_root)
    if attempt.get("closure") != _closure():
        raise RuntimeError("current code closure differs from immutable experiment attempt")
    if attempt.get("head") != _head():
        raise RuntimeError("current Git HEAD differs from immutable experiment attempt")


def _access_totals(result_root: Path) -> tuple[int, int]:
    recordings = 0
    byte_count = 0
    cells = result_root.resolve() / "cells"
    if not cells.is_dir():
        return recordings, byte_count
    for date in CONFIRMATORY_DATES:
        for name in ("terminal.json", "failure.json"):
            path = cells / date / name
            if not path.is_file():
                continue
            try:
                body = json.loads(path.read_text(encoding="utf-8"))
                audit = body.get("target_access", {})
                recordings += int(audit.get("target_recordings_opened", 0))
                byte_count += int(audit.get("target_bytes_read", 0))
            except BaseException:
                pass
            break
    return recordings, byte_count


def _publish_failure(result_root: Path, phase: str, error: BaseException) -> None:
    root = result_root.resolve()
    path = root / f"{phase}_failure.json"
    if path.exists() or not root.exists():
        return
    recordings, byte_count = _access_totals(root)
    publish_json(
        path,
        {
            "schema": SCHEMA,
            "artifact": f"{phase}_failure",
            "status": "FAIL_IMMUTABLE_NO_AUTOMATIC_RETRY",
            "error_type": type(error).__name__,
            "error": str(error),
            "target_recordings_opened": recordings,
            "target_bytes_read": byte_count,
            "target_optimizer_steps": 0,
            "target_backward_steps": 0,
        },
    )


def _cpu_gate(result_root: Path) -> None:
    command = [sys.executable, "-m", "pytest", "-q", str(TEST_FILE)]
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
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
        "status": "PASS_H1_CAUSAL_OUTPUT_EMA_V1_CPU_GATE" if completed.returncode == 0 else "FAIL_CPU_GATE",
        "command": command,
        "returncode": completed.returncode,
        "log_sha256": log_sha,
        "cuda_visible_devices": "",
        "data_files_opened": 0,
        "nwb_files_opened": 0,
        "target_recordings_opened": 0,
        "target_bytes_read": 0,
    }
    publish_json(result_root.resolve() / "cpu_gate.json", body)
    if completed.returncode != 0:
        raise RuntimeError(f"CPU gate failed; see {result_root / 'cpu_gate.log'}")


def _require_cpu_gate(result_root: Path) -> None:
    body = _load_json(result_root.resolve() / "cpu_gate.json")
    if body.get("status") != "PASS_H1_CAUSAL_OUTPUT_EMA_V1_CPU_GATE":
        raise RuntimeError("runtime phase requires a PASS CPU/no-data gate")


def _gpu_row(index: int) -> dict[str, Any]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    fields = [value.strip() for value in output.split(",")]
    if len(fields) != 5:
        raise RuntimeError(f"cannot parse GPU profile: {output}")
    row = {
        "physical_index": int(fields[0]),
        "uuid": fields[1],
        "name": fields[2],
        "memory_used_mib": int(fields[3]),
        "utilization_percent": int(fields[4]),
    }
    if row["memory_used_mib"] >= 1024 or row["utilization_percent"] > 5:
        raise RuntimeError(f"GPU {index} is not idle: {row}")
    return row


def _parse_gpus(raw: str | None, *, require_two: bool) -> tuple[int, ...]:
    if raw is None:
        raise RuntimeError("--physical-gpus is required")
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    expected_count = 2 if require_two else None
    if not values or len(values) > 2 or len(set(values)) != len(values) or any(value < 0 for value in values):
        raise RuntimeError("--physical-gpus must contain one or two distinct nonnegative indices")
    if expected_count is not None and len(values) != expected_count:
        raise RuntimeError("five-date evaluation requires exactly two physical GPUs")
    return values


def _cell_env(gpu: int) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": str(gpu),
        }
    )
    return environment


def _cell_command(args: argparse.Namespace, date: str, gpu: int) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--cell-date",
        date,
        "--physical-gpu",
        str(gpu),
        "--data-root",
        str(args.data_root),
        "--predecessor-root",
        str(args.predecessor_root),
        "--result-root",
        str(args.result_root),
    ]


def _smoke_command(args: argparse.Namespace, gpu: int) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--smoke-cell",
        "--physical-gpu",
        str(gpu),
        "--predecessor-root",
        str(args.predecessor_root),
        "--result-root",
        str(args.result_root),
    ]


def _prepare_predecessor(args: argparse.Namespace) -> None:
    path = args.result_root.resolve() / "predecessor_authority.json"
    if path.exists():
        body = _load_json(path)
        if body != validate_predecessor(args.predecessor_root):
            raise RuntimeError("existing predecessor authority receipt drift")
        return
    publish_predecessor_authority(args.predecessor_root, args.result_root)


def _run_smoke(args: argparse.Namespace, gpus: tuple[int, ...]) -> None:
    _require_cpu_gate(args.result_root)
    _prepare_predecessor(args)
    gpu = gpus[0]
    profile = _gpu_row(gpu)
    args.log_root.mkdir(parents=True, exist_ok=True)
    log_path = args.log_root.resolve() / "smoke.log"
    with log_path.open("x", encoding="utf-8") as output:
        completed = subprocess.run(
            _smoke_command(args, gpu),
            cwd=REPO_ROOT,
            env=_cell_env(gpu),
            text=True,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=False,
        )
    log_sha = seal_existing_log(log_path)
    if completed.returncode != 0:
        raise RuntimeError(f"GPU smoke failed on {profile['uuid']}; log SHA {log_sha}")
    terminal = _load_json(args.result_root.resolve() / "smoke" / "terminal.json")
    if terminal.get("status") != STATUS_SMOKE or terminal.get("gpu", {}).get("uuid") != profile["uuid"]:
        raise RuntimeError("GPU smoke terminal/profile binding drift")


def _require_smoke(result_root: Path) -> None:
    terminal = _load_json(result_root.resolve() / "smoke" / "terminal.json")
    if terminal.get("status") != STATUS_SMOKE or terminal.get("target_recordings_opened") != 0:
        raise RuntimeError("evaluation requires a zero-target-access PASS GPU smoke")


def _run_evaluation(args: argparse.Namespace, gpus: tuple[int, ...]) -> None:
    _require_cpu_gate(args.result_root)
    _require_smoke(args.result_root)
    for gpu in gpus:
        _gpu_row(gpu)
    args.log_root.mkdir(parents=True, exist_ok=True)
    queues = {
        gpus[0]: ["19250108", "19250115", "19250120"],
        gpus[1]: ["19250113", "19250119"],
    }
    running: dict[int, tuple[str, subprocess.Popen[str], Any, Path]] = {}
    failed = False
    while any(queues.values()) or running:
        for gpu in gpus:
            if gpu in running or not queues[gpu] or failed:
                continue
            date = queues[gpu].pop(0)
            log_path = args.log_root.resolve() / f"{date}.log"
            output = log_path.open("x", encoding="utf-8")
            process = subprocess.Popen(
                _cell_command(args, date, gpu),
                cwd=REPO_ROOT,
                env=_cell_env(gpu),
                text=True,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            running[gpu] = (date, process, output, log_path)
        completed_gpus: list[int] = []
        for gpu, (_date, process, output, log_path) in running.items():
            returncode = process.poll()
            if returncode is None:
                continue
            process.wait()
            output.close()
            seal_existing_log(log_path)
            completed_gpus.append(gpu)
            if returncode != 0:
                failed = True
        for gpu in completed_gpus:
            del running[gpu]
        if running and not completed_gpus:
            time.sleep(2.0)
        if failed and not running:
            break
    pending = [date for gpu in gpus for date in queues[gpu]]
    if failed or pending:
        raise RuntimeError(f"one or more evaluation cells failed; unlaunched dates: {pending}")
    for date in CONFIRMATORY_DATES:
        terminal = _load_json(args.result_root.resolve() / "cells" / date / "terminal.json")
        if terminal.get("status") != STATUS_CELL:
            raise RuntimeError(f"date cell lacks PASS terminal: {date}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_mutually_exclusive_group()
    phases.add_argument("--dry-run", action="store_true")
    phases.add_argument("--cpu-gate", action="store_true")
    phases.add_argument("--smoke", action="store_true")
    phases.add_argument("--evaluate", action="store_true")
    phases.add_argument("--verify-terminal", action="store_true")
    phases.add_argument("--detached-supervisor", action="store_true")
    phases.add_argument("--smoke-cell", action="store_true", help=argparse.SUPPRESS)
    phases.add_argument("--cell-date", choices=CONFIRMATORY_DATES, help=argparse.SUPPRESS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--predecessor-root", type=Path, default=DEFAULT_PREDECESSOR_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--physical-gpus", help="one or two comma-separated physical GPU indices")
    parser.add_argument("--physical-gpu", type=int, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not any((args.cpu_gate, args.smoke, args.evaluate, args.verify_terminal, args.detached_supervisor, args.smoke_cell, args.cell_date)):
        print(json.dumps(dry_plan(), indent=2, sort_keys=True))
        return 0
    if args.smoke_cell:
        if args.physical_gpu is None:
            raise RuntimeError("internal --smoke-cell requires --physical-gpu")
        _assert_code_closure(args.result_root)
        print(f"START smoke physical_gpu={args.physical_gpu}", flush=True)
        body = run_smoke_cell(args.predecessor_root, args.result_root, args.physical_gpu)
        print(f"TERMINAL {body['status']}", flush=True)
        return 0
    if args.cell_date is not None:
        if args.physical_gpu is None:
            raise RuntimeError("internal --cell-date requires --physical-gpu")
        _assert_code_closure(args.result_root)
        print(f"START outer_date={args.cell_date} physical_gpu={args.physical_gpu}", flush=True)
        body = run_evaluation_cell(args.data_root, args.predecessor_root, args.result_root, args.cell_date, args.physical_gpu)
        print(f"TERMINAL outer_date={args.cell_date} status={body['status']}", flush=True)
        return 0
    phase = "unknown"
    try:
        if args.cpu_gate:
            phase = "cpu_gate"
            create_attempt(args.result_root, _closure(), _head())
            _cpu_gate(args.result_root)
        elif args.smoke:
            phase = "smoke"
            _assert_code_closure(args.result_root)
            gpus = _parse_gpus(args.physical_gpus, require_two=False)
            _run_smoke(args, gpus)
        elif args.evaluate:
            phase = "evaluation"
            _assert_code_closure(args.result_root)
            gpus = _parse_gpus(args.physical_gpus, require_two=True)
            _run_evaluation(args, gpus)
        elif args.verify_terminal:
            phase = "terminal_verification"
            _assert_code_closure(args.result_root)
            verify_terminal(args.predecessor_root, args.result_root)
        elif args.detached_supervisor:
            phase = "predecessor_validation"
            _assert_code_closure(args.result_root)
            gpus = _parse_gpus(args.physical_gpus, require_two=True)
            _prepare_predecessor(args)
            if not (args.result_root.resolve() / "smoke" / "terminal.json").exists():
                phase = "smoke"
                _run_smoke(args, gpus)
            phase = "evaluation"
            _run_evaluation(args, gpus)
            phase = "terminal_verification"
            verify_terminal(args.predecessor_root, args.result_root)
        return 0
    except BaseException as error:
        _publish_failure(args.result_root, phase, error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
