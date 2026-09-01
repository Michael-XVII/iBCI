#!/usr/bin/env python3
"""Fail-closed CLI for H1 five-date source-only H-C regeneration V1."""
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

from src.h1_hc_date_lodo_regen_v1 import (  # noqa: E402
    SCHEMA,
    STATUS_SMOKE,
    create_attempt,
    dry_plan,
    prepare_source_authority,
    publish_json,
    publish_text,
    run_cell,
    seal_existing_log,
    verify_sidecar,
    verify_terminal,
)
from src.h1_m4_cce_contract import CONFIRMATORY_DATES, sha256_file  # noqa: E402


DEFAULT_DATA_ROOT = Path("/data/ial-dataset/ial-mohd/000954")
DEFAULT_RESULT_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_hc_date_lodo_regen_v1_detached_a1"
DEFAULT_LOG_ROOT = REPO_ROOT / "logs/h1_hc_date_lodo_regen_v1_detached_a1"
WORK_ORDER = REPO_ROOT / "tfpd_exploration/h1_series_20260830/H1_HC_DATE_LODO_REGEN_V1_WORK_ORDER.md"
AMENDMENT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/H1_HC_DATE_LODO_REGEN_V1_AMENDMENT_1.md"
TEST_FILE = SPINT_ROOT / "tests/test_h1_hc_date_lodo_regen_v1.py"


def _closure() -> dict[str, str]:
    paths = (
        WORK_ORDER,
        AMENDMENT,
        Path(__file__).resolve(),
        SPINT_ROOT / "src/h1_hc_date_lodo_regen_v1.py",
        TEST_FILE,
        SPINT_ROOT / "src/models/components/h1_carrierid_spint.py",
        SPINT_ROOT / "src/data/h1_carrierid_date_lodo_source.py",
        SPINT_ROOT / "src/data/h1_m4_eb_pilot.py",
        SPINT_ROOT / "src/h1_m4_cce_contract.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"code closure is incomplete: {missing}")
    return {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths}


def _head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def _load_json(path: Path) -> dict[str, Any]:
    verify_sidecar(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_code_closure(result_root: Path) -> None:
    attempt = _load_json(result_root.resolve() / "attempt.json")
    if attempt.get("closure") != _closure():
        raise RuntimeError("current code closure differs from the immutable experiment attempt")
    if attempt.get("head") != _head():
        raise RuntimeError("current Git HEAD differs from the immutable experiment attempt")


def _publish_failure(result_root: Path, phase: str, error: BaseException) -> None:
    path = result_root.resolve() / f"{phase}_failure.json"
    if path.exists() or not result_root.resolve().exists():
        return
    publish_json(
        path,
        {
            "schema": SCHEMA,
            "artifact": f"{phase}_failure",
            "status": "FAIL_IMMUTABLE_NO_AUTOMATIC_RETRY",
            "error_type": type(error).__name__,
            "error": str(error),
            "target_recordings_opened": 0,
            "target_bytes_read": 0,
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
        "status": "PASS_H1_HC_DATE_LODO_REGEN_V1_CPU_GATE" if completed.returncode == 0 else "FAIL_CPU_GATE",
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


def _parse_gpus(raw: str | None) -> tuple[int, ...]:
    if raw is None:
        raise RuntimeError("--gpus is required for --smoke and --train")
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values or len(values) > 2 or len(set(values)) != len(values) or any(value < 0 for value in values):
        raise RuntimeError("--gpus must contain one or two distinct nonnegative physical GPU indices")
    return values


def _cell_command(args: argparse.Namespace, date: str, gpu: int, *, smoke: bool) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--cell",
        "--outer-date",
        date,
        "--physical-gpu",
        str(gpu),
        "--data-root",
        str(args.data_root),
        "--result-root",
        str(args.result_root),
    ]
    if smoke:
        command.extend(("--smoke-cell", "--smoke-steps", str(args.smoke_steps)))
    return command


def _cell_env(gpu: int) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update({"PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": str(gpu)})
    return environment


def _run_smoke(args: argparse.Namespace) -> None:
    source = _load_json(args.result_root.resolve() / "source_authority.json")
    if source.get("target_bytes_read") != 0:
        raise RuntimeError("source authority records target access")
    gpu = _parse_gpus(args.gpus)[0]
    profile = _gpu_row(gpu)
    command = _cell_command(args, CONFIRMATORY_DATES[0], gpu, smoke=True)
    args.log_root.mkdir(parents=True, exist_ok=True)
    log_path = args.log_root.resolve() / "smoke.log"
    with log_path.open("x", encoding="utf-8") as output:
        completed = subprocess.run(
            command,
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
    terminal = _load_json(args.result_root.resolve() / "smoke" / f"smoke_{CONFIRMATORY_DATES[0]}" / "terminal.json")
    if terminal.get("status") != STATUS_SMOKE or terminal.get("gpu", {}).get("uuid") != profile["uuid"]:
        raise RuntimeError("GPU smoke terminal/profile binding drift")


def _run_training(args: argparse.Namespace) -> None:
    smoke = _load_json(args.result_root.resolve() / "smoke" / f"smoke_{CONFIRMATORY_DATES[0]}" / "terminal.json")
    if smoke.get("status") != STATUS_SMOKE or smoke.get("target_bytes_read") != 0:
        raise RuntimeError("training requires a source-only PASS smoke")
    gpus = _parse_gpus(args.gpus)
    for gpu in gpus:
        _gpu_row(gpu)
    args.log_root.mkdir(parents=True, exist_ok=True)
    pending = list(CONFIRMATORY_DATES)
    running: dict[str, tuple[subprocess.Popen[str], int, Any, Path]] = {}
    failed = False
    while pending or running:
        while pending and len(running) < len(gpus) and not failed:
            date = pending.pop(0)
            gpu = next(value for value in gpus if value not in {item[1] for item in running.values()})
            log_path = args.log_root.resolve() / f"{date}.log"
            output = log_path.open("x", encoding="utf-8")
            process = subprocess.Popen(
                _cell_command(args, date, gpu, smoke=False),
                cwd=REPO_ROOT,
                env=_cell_env(gpu),
                text=True,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            running[date] = (process, gpu, output, log_path)
        completed_dates = []
        for date, (process, _gpu, output, log_path) in running.items():
            returncode = process.poll()
            if returncode is None:
                continue
            process.wait()
            output.close()
            seal_existing_log(log_path)
            completed_dates.append(date)
            if returncode != 0:
                failed = True
        for date in completed_dates:
            del running[date]
        if running and not completed_dates:
            time.sleep(2.0)
        if failed and not running:
            break
    if failed or pending:
        raise RuntimeError(f"one or more training cells failed; unlaunched dates: {pending}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_mutually_exclusive_group()
    phases.add_argument("--dry-run", action="store_true")
    phases.add_argument("--prepare-source-authority", action="store_true")
    phases.add_argument("--smoke", action="store_true")
    phases.add_argument("--train", action="store_true")
    phases.add_argument("--verify-terminal", action="store_true")
    phases.add_argument("--detached-supervisor", action="store_true", help=argparse.SUPPRESS)
    phases.add_argument("--cell", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--gpus", help="one or two comma-separated physical GPU indices")
    parser.add_argument("--outer-date", choices=CONFIRMATORY_DATES, help=argparse.SUPPRESS)
    parser.add_argument("--physical-gpu", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--smoke-cell", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--smoke-steps", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not any((args.prepare_source_authority, args.smoke, args.train, args.verify_terminal, args.detached_supervisor, args.cell)):
        print(json.dumps(dry_plan(), indent=2, sort_keys=True))
        return 0
    if args.cell:
        if args.outer_date is None or args.physical_gpu is None:
            raise RuntimeError("internal --cell requires --outer-date and --physical-gpu")
        _assert_code_closure(args.result_root)
        run_cell(
            args.data_root,
            args.result_root,
            args.outer_date,
            args.physical_gpu,
            smoke=args.smoke_cell,
            smoke_steps=args.smoke_steps,
        )
        return 0
    phase = "unknown"
    try:
        if args.prepare_source_authority:
            phase = "source_authority"
            create_attempt(args.result_root, _closure(), _head())
            _cpu_gate(args.result_root)
            prepare_source_authority(args.data_root, args.result_root)
        elif args.smoke:
            phase = "smoke"
            _assert_code_closure(args.result_root)
            _run_smoke(args)
        elif args.train:
            phase = "training"
            _assert_code_closure(args.result_root)
            _run_training(args)
        elif args.verify_terminal:
            phase = "terminal_verification"
            _assert_code_closure(args.result_root)
            verify_terminal(args.result_root)
        elif args.detached_supervisor:
            phase = "detached_training"
            _assert_code_closure(args.result_root)
            _run_training(args)
            phase = "terminal_verification"
            verify_terminal(args.result_root)
        return 0
    except BaseException as error:
        _publish_failure(args.result_root, phase, error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
