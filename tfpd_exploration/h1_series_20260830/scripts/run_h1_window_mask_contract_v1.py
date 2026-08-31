#!/usr/bin/env python3
"""Run the H1 window-mask CPU gate once with immutable receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime, timezone
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[3]
BUNDLE_ROOT = Path(__file__).resolve().parents[1]
STREAMING_ROOT = REPO_ROOT / "streaming_calibration_exp"
DEFAULT_RESULT_ROOT = BUNDLE_ROOT / "results/h1_window_mask_contract_v1"
EXPECTED_BRANCH = "exp/h1-window-mask-contract-v1"
BASELINE_SHA = "21d0881f50a8d88d78fb5d5b941a6bf470019e9f"
PASS_STATUS = "PASS_WINDOW_MASK_CONTRACT_V1_CPU_GATE"
FAIL_STATUS = "FAIL_WINDOW_MASK_CONTRACT_V1_CPU_GATE"


EXECUTED_CLOSURE = {
    "contract": STREAMING_ROOT / "src/data/h1_window_mask_contract_v1.py",
    "contract_tests": STREAMING_ROOT / "tests/test_h1_window_mask_contract_v1.py",
    "sampler_regression": STREAMING_ROOT / "tests/test_falcon_sampler.py",
    "runner": Path(__file__).resolve(),
    "design": BUNDLE_ROOT / "docs/DESIGN_H1_WINDOW_MASK_CONTRACT_V1_20260830.md",
    "work_order": BUNDLE_ROOT / "docs/WORKORDER_H1_WINDOW_MASK_CONTRACT_V1_20260831.md",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def _write_atomic_immutable(path: Path, payload: object) -> str:
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    sidecar = path.with_name(path.name + ".sha256")
    sidecar_body = f"{digest}  {path.name}\n".encode("ascii")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{sidecar.name}.", suffix=".tmp", dir=sidecar.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(sidecar_body)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        os.replace(temporary, sidecar)
    finally:
        if temporary.exists():
            temporary.unlink()
    return digest


def _pytest_summary(path: Path) -> dict[str, object]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    summary: dict[str, object] = {}
    for key in ("tests", "failures", "errors", "skipped"):
        summary[key] = sum(int(suite.attrib.get(key, "0")) for suite in suites)
    summary["time_seconds"] = sum(float(suite.attrib.get("time", "0")) for suite in suites)
    return summary


def _validate_context(python_executable: Path, result_root: Path) -> None:
    if _git("rev-parse", "--abbrev-ref", "HEAD") != EXPECTED_BRANCH:
        raise RuntimeError(f"runner requires branch {EXPECTED_BRANCH}")
    if _git("rev-parse", "HEAD") != BASELINE_SHA:
        raise RuntimeError(f"runner requires uncommitted additive work over {BASELINE_SHA}")
    if not python_executable.is_file():
        raise RuntimeError(f"Python executable is missing: {python_executable}")
    missing = [str(path) for path in EXECUTED_CLOSURE.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"executed closure is incomplete: {missing}")
    if os.path.lexists(result_root):
        raise RuntimeError(f"result root must be fresh: {result_root}")


def run(python_executable: Path, result_root: Path) -> int:
    _validate_context(python_executable, result_root)
    result_root.mkdir(parents=True, exist_ok=False)
    junit_path = result_root / "pytest.xml"
    command = [
        str(python_executable),
        "-m",
        "pytest",
        "-q",
        "tests/test_h1_window_mask_contract_v1.py",
        "tests/test_falcon_sampler.py",
        f"--junitxml={junit_path}",
    ]
    closure = {
        name: {"path": str(path.relative_to(REPO_ROOT)), "sha256": _sha256(path)}
        for name, path in EXECUTED_CLOSURE.items()
    }
    attempt = {
        "schema": "h1_window_mask_contract_v1_attempt",
        "status": "ATTEMPT_H1_WINDOW_MASK_CONTRACT_V1_CPU_GATE",
        "created_at_utc": _utc_now(),
        "branch": EXPECTED_BRANCH,
        "baseline_git_sha": BASELINE_SHA,
        "python_executable": str(python_executable),
        "command": command,
        "working_directory": str(STREAMING_ROOT),
        "executed_closure": closure,
        "scope": {
            "synthetic_sessions_only": True,
            "h1_data_root_opened": False,
            "nwb_files_opened": 0,
            "cuda_visible_devices": "",
            "cuda_constructed": False,
            "gpu_allocated": False,
            "training_steps": 0,
            "automatic_retry": False,
        },
    }
    attempt_sha = _write_atomic_immutable(result_root / "attempt.json", attempt)
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    started_at = _utc_now()
    try:
        completed = subprocess.run(
            command,
            cwd=STREAMING_ROOT,
            env=environment,
            check=False,
        )
        returncode = int(completed.returncode)
        if junit_path.is_file():
            junit_path.chmod(0o444)
            junit_sha = _sha256(junit_path)
            (result_root / "pytest.xml.sha256").write_text(
                f"{junit_sha}  pytest.xml\n", encoding="ascii"
            )
            (result_root / "pytest.xml.sha256").chmod(0o444)
            pytest_summary = _pytest_summary(junit_path)
        else:
            junit_sha = None
            pytest_summary = None
        passed = (
            returncode == 0
            and isinstance(pytest_summary, dict)
            and pytest_summary.get("tests") == 17
            and pytest_summary.get("failures") == 0
            and pytest_summary.get("errors") == 0
            and pytest_summary.get("skipped") == 0
        )
        receipt = {
            "schema": "h1_window_mask_contract_v1_terminal_or_failure",
            "status": PASS_STATUS if passed else FAIL_STATUS,
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "attempt": {"path": "attempt.json", "sha256": attempt_sha},
            "returncode": returncode,
            "pytest": {
                "summary": pytest_summary,
                "junit_path": "pytest.xml" if junit_sha else None,
                "junit_sha256": junit_sha,
            },
            "scope_confirmation": {
                "h1_data_root_opened": False,
                "nwb_files_opened": 0,
                "cuda_visible_devices": "",
                "cuda_constructed": False,
                "gpu_allocated": False,
                "training_steps": 0,
            },
            "gpu_cell_prerequisite_satisfied": bool(passed),
            "later_gpu_cells_authorized_by_this_receipt": False,
            "authorization_note": (
                "A passing CPU gate is necessary but does not replace the separate "
                "work order and explicit authorization required for any GPU cell."
            ),
        }
        name = "terminal.json" if passed else "failure.json"
        _write_atomic_immutable(result_root / name, receipt)
        return returncode if returncode != 0 else (0 if passed else 1)
    except BaseException as error:
        failure = {
            "schema": "h1_window_mask_contract_v1_terminal_or_failure",
            "status": FAIL_STATUS,
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "attempt": {"path": "attempt.json", "sha256": attempt_sha},
            "exception_type": type(error).__name__,
            "exception": str(error),
            "gpu_cell_prerequisite_satisfied": False,
            "later_gpu_cells_authorized_by_this_receipt": False,
        }
        _write_atomic_immutable(result_root / "failure.json", failure)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=Path("/home/ial-mohd/workspace/envs/spint/bin/python"),
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    args = parser.parse_args()
    raise SystemExit(run(args.python_executable.resolve(), args.result_root.resolve()))


if __name__ == "__main__":
    main()
