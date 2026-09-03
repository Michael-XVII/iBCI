#!/usr/bin/env python3
"""Additive proxy-reachability repair for H1 M3 EvalAI package rehearsal."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[3]
SPINT_ROOT = REPO_ROOT / "SPINT-main"
if str(SPINT_ROOT) not in sys.path:
    sys.path.insert(0, str(SPINT_ROOT))

from src.h1_cal_aug_all_source_m3_evalai_package_v1 import (  # noqa: E402
    SCHEMA,
    build_and_smoke,
    create_attempt,
    load_attempt,
    prepare_packages,
    verify_terminal,
)
from src.h1_hc_date_lodo_regen_v1 import publish_json, publish_text  # noqa: E402
from src.h1_m4_cce_contract import sha256_file  # noqa: E402


DEFAULT_PACKAGE_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_all_source_m3_deployment_v1_package_a1"
DEFAULT_STAGING_ROOT = SPINT_ROOT / "local_data/h1_m3_evalai_v1"
DEFAULT_RESULT_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_all_source_m3_evalai_package_v1_a1"
DEFAULT_LOG_ROOT = REPO_ROOT / "logs/h1_cal_aug_all_source_m3_evalai_package_v1_a1"
AMENDMENT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/docs/AMENDMENT_H1_CAL_AUG_ALL_SOURCE_M3_EVALAI_PACKAGE_V1_A1.md"
TEST_FILE = SPINT_ROOT / "tests/test_h1_cal_aug_all_source_m3_evalai_package_v1_a1.py"
BASE_RUNNER = REPO_ROOT / "tfpd_exploration/h1_series_20260830/scripts/run_h1_cal_aug_all_source_m3_evalai_package_v1.py"


def closure() -> dict[str, str]:
    paths = (
        AMENDMENT, TEST_FILE, Path(__file__).resolve(), BASE_RUNNER,
        REPO_ROOT / "tfpd_exploration/h1_series_20260830/docs/WORKORDER_H1_CAL_AUG_ALL_SOURCE_M3_EVALAI_PACKAGE_V1.md",
        SPINT_ROOT / "src/h1_cal_aug_all_source_m3_evalai_package_v1.py",
        SPINT_ROOT / "third_party/falcon_challenge/h1_carrier_id_spint_sample.py",
        SPINT_ROOT / "third_party/falcon_challenge/h1_carrier_id_spint_sample.Dockerfile",
        SPINT_ROOT / "third_party/falcon_challenge/h1_carrier_id_spint_decoder.py",
        SPINT_ROOT / "src/models/components/h1_carrierid_spint.py",
        SPINT_ROOT / "environment.yaml",
    )
    return {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in paths}


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def cpu_gate(result_root: Path) -> None:
    command = [sys.executable, "-m", "pytest", "-q", str(TEST_FILE)]
    env = dict(os.environ); env.update({"PYTHONNOUSERSITE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "CUDA_VISIBLE_DEVICES": ""})
    completed = subprocess.run(command, cwd=SPINT_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log_sha = publish_text(result_root.resolve() / "cpu_gate.log", completed.stdout)
    publish_json(result_root.resolve() / "cpu_gate.json", {
        "schema": f"{SCHEMA}_a1_cpu_gate",
        "status": "PASS_H1_M3_EVALAI_PACKAGE_A1_CPU_GATE" if completed.returncode == 0 else "FAIL_H1_M3_EVALAI_PACKAGE_A1_CPU_GATE",
        "returncode": completed.returncode, "log_sha256": log_sha,
        "nwb_files_opened": 0, "docker_commands": 0, "training": False, "evalai_submissions": 0,
    })
    if completed.returncode:
        raise RuntimeError("A1 CPU gate failed")


def assert_closure(result_root: Path) -> None:
    attempt = load_attempt(result_root)
    if attempt["closure"] != closure() or attempt["git_head"] != git_head():
        raise RuntimeError("A1 Git HEAD/code closure differs from attempt")


def port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def with_proxy_forwarder(action) -> None:
    if port_open(7897):
        raise RuntimeError("port 7897 unexpectedly occupied; refuse ambiguous proxy repair")
    proxy = subprocess.Popen(["socat", "TCP-LISTEN:7897,bind=127.0.0.1,reuseaddr,fork", "TCP:127.0.0.1:17897"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(20):
            if proxy.poll() is not None:
                raise RuntimeError("temporary proxy forwarder exited")
            if port_open(7897):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("temporary proxy forwarder did not listen")
        action()
    finally:
        proxy.terminate()
        try:
            proxy.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proxy.kill(); proxy.wait(timeout=5)


def publish_failure(result_root: Path, phase: str, error: BaseException) -> None:
    path = result_root.resolve() / f"{phase}_failure.json"
    if path.exists() or not result_root.exists():
        return
    publish_json(path, {"schema": SCHEMA, "status": "FAIL_H1_M3_EVALAI_PACKAGE_A1_NO_AUTOMATIC_RETRY", "phase": phase, "error_type": type(error).__name__, "error": str(error), "training": False, "docker_pushes": 0, "evalai_submissions": 0})


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    phases = result.add_mutually_exclusive_group()
    phases.add_argument("--dry-run", action="store_true")
    phases.add_argument("--initialize", action="store_true")
    phases.add_argument("--detached-supervisor", action="store_true")
    result.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    result.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING_ROOT)
    result.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    result.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if not any((args.initialize, args.detached_supervisor)):
        print(json.dumps({"schema": f"{SCHEMA}_a1", "status": "DRY_A1_NO_WRITE_NO_DATA_NO_CUDA_NO_DOCKER", "proxy_forwarder_started": False, "evalai_submissions": 0}, indent=2, sort_keys=True)); return 0
    phase = "initialize"
    try:
        if args.initialize:
            create_attempt(args.result_root, args.package_root, closure(), git_head()); cpu_gate(args.result_root); return 0
        assert_closure(args.result_root)
        phase = "prepare_packages"; prepare_packages(args.package_root, args.staging_root, args.result_root)
        phase = "docker_build_and_smoke"
        with_proxy_forwarder(lambda: build_and_smoke(SPINT_ROOT, args.package_root, args.staging_root, args.result_root, args.log_root))
        phase = "terminal"; verify_terminal(args.result_root); return 0
    except BaseException as error:
        publish_failure(args.result_root, phase, error); raise


if __name__ == "__main__":
    raise SystemExit(main())
