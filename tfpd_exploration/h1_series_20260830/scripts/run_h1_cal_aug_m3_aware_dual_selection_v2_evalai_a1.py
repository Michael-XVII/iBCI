#!/usr/bin/env python3
"""Execute frozen V2 A1 package, Docker, submission, and result phases."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
SPINT_ROOT = REPO_ROOT / "SPINT-main"
if str(SPINT_ROOT) not in sys.path:
    sys.path.insert(0, str(SPINT_ROOT))

from src.h1_cal_aug_m3_aware_dual_selection_v2_evalai_a1 import (  # noqa: E402
    SCHEMA, build_and_smoke_images, collect_results, create_attempt,
    create_submission_manifest, host_smoke, prepare_packages, submit_all,
    verify_predecessor, verify_terminal,
)
from src.h1_hc_date_lodo_regen_v1 import publish_json, publish_text, verify_sidecar  # noqa: E402
from src.h1_m4_cce_contract import canonical_sha256, sha256_file  # noqa: E402


EVAL_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_m3_aware_dual_selection_v2_eval_a1"
TRAIN_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_m3_aware_dual_selection_v2_a1"
V1_PACKAGE_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_all_source_m3_deployment_v1_package_a1"
RESULT_ROOT = REPO_ROOT / "tfpd_exploration/h1_series_20260830/results/h1_cal_aug_m3_aware_dual_selection_v2_evalai_submission_a1"
LOG_ROOT = REPO_ROOT / "logs/h1_cal_aug_m3_aware_dual_selection_v2_evalai_submission_a1"
STAGING_ROOT = SPINT_ROOT / "local_data/h1_m3_evalai_v1"
EVALAI = REPO_ROOT.parent / "envs/falcon/bin/evalai"
WORK_ORDER = REPO_ROOT / "tfpd_exploration/h1_series_20260830/docs/WORKORDER_H1_CAL_AUG_M3_AWARE_DUAL_SELECTION_V2_EVALAI_A1.md"
MODULE = SPINT_ROOT / "src/h1_cal_aug_m3_aware_dual_selection_v2_evalai_a1.py"
TEST = SPINT_ROOT / "tests/test_h1_cal_aug_m3_aware_dual_selection_v2_evalai_a1.py"


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def closure() -> dict[str, str]:
    files = (
        WORK_ORDER, MODULE, TEST, Path(__file__).resolve(),
        SPINT_ROOT / "third_party/falcon_challenge/h1_carrier_id_spint_decoder.py",
        SPINT_ROOT / "third_party/falcon_challenge/h1_carrier_id_spint_sample.py",
        SPINT_ROOT / "third_party/falcon_challenge/h1_carrier_id_spint_sample.Dockerfile",
        SPINT_ROOT / "src/models/components/h1_carrierid_spint.py", SPINT_ROOT / "environment.yaml",
    )
    return {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in files}


def assert_attempt(result_root: Path) -> dict:
    attempt = json.loads((result_root / "attempt.json").read_text(encoding="utf-8"))
    verify_sidecar(result_root / "attempt.json")
    if attempt["code_closure"] != closure() or attempt["git_head"] != git_head():
        raise RuntimeError("immutable attempt code closure/Git HEAD drift")
    return attempt


def cpu_gate(result_root: Path) -> None:
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONNOUSERSITE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "TQDM_DISABLE": "1"})
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q", str(TEST)], cwd=SPINT_ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log_sha = publish_text(result_root / "cpu_gate.log", completed.stdout)
    publish_json(result_root / "cpu_gate.json", {
        "schema": f"{SCHEMA}_cpu_gate", "status": "PASS_V2_A1_CPU_GATE" if completed.returncode == 0 else "FAIL_V2_A1_CPU_GATE",
        "returncode": completed.returncode, "log_sha256": log_sha, "training": False,
        "cuda_initializations": 0, "docker_commands": 0, "evalai_submissions": 0,
    })
    if completed.returncode:
        raise RuntimeError("CPU gate failed")


def initialize(args) -> None:
    predecessor = verify_predecessor(REPO_ROOT, EVAL_ROOT, TRAIN_ROOT, V1_PACKAGE_ROOT)
    create_attempt(args.result_root, {"git_head": git_head(), "code_closure": closure(), "code_closure_sha256": canonical_sha256(closure())})
    publish_json(args.result_root / "predecessor_authority.json", predecessor)
    cpu_gate(args.result_root)


def dry_run() -> dict:
    return {
        "schema": SCHEMA, "status": "DRY_RUN_NO_WRITE_NO_CUDA_NO_DOCKER_NO_EVALAI",
        "writes": 0, "training": False, "optimizer_steps": 0, "backward_steps": 0,
        "model_updates": 0, "docker_commands": 0, "evalai_submissions": 0,
        "candidate_order": ["C2-E49", "C2-HI-E45", "C2-HO-E15"],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    phases = result.add_mutually_exclusive_group()
    phases.add_argument("--dry-run", action="store_true")
    phases.add_argument("--initialize", action="store_true")
    phases.add_argument("--prepare-packages", action="store_true")
    phases.add_argument("--host-smoke", action="store_true")
    phases.add_argument("--build-docker", action="store_true")
    phases.add_argument("--seal-manifest", action="store_true")
    phases.add_argument("--submit-all", action="store_true")
    phases.add_argument("--collect-results", action="store_true")
    phases.add_argument("--verify-terminal", action="store_true")
    result.add_argument("--result-root", type=Path, default=RESULT_ROOT)
    result.add_argument("--log-root", type=Path, default=LOG_ROOT)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if not any((args.initialize, args.prepare_packages, args.host_smoke, args.build_docker, args.seal_manifest, args.submit_all, args.collect_results, args.verify_terminal)):
        print(json.dumps(dry_run(), indent=2, sort_keys=True)); return 0
    os.environ["TQDM_DISABLE"] = "1"
    if args.initialize:
        initialize(args); return 0
    assert_attempt(args.result_root)
    if args.prepare_packages:
        prepare_packages(args.result_root, V1_PACKAGE_ROOT)
    elif args.host_smoke:
        host_smoke(args.result_root)
    elif args.build_docker:
        build_and_smoke_images(args.result_root, args.log_root, SPINT_ROOT, STAGING_ROOT)
    elif args.seal_manifest:
        create_submission_manifest(args.result_root)
    elif args.submit_all:
        submit_all(REPO_ROOT, args.result_root, args.log_root, EVALAI)
    elif args.collect_results:
        collect_results(args.result_root, args.log_root, EVALAI)
    elif args.verify_terminal:
        verify_terminal(args.result_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
