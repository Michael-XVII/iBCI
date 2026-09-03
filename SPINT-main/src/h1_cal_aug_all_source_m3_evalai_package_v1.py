"""Submission-readiness packaging for frozen H1 M3 T0/C1 decoders; never submits."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping

import numpy as np

from src.h1_hc_date_lodo_regen_v1 import publish_json, verify_sidecar
from src.h1_m4_cce_contract import sha256_file, state_hash


SCHEMA = "h1_cal_aug_all_source_m3_evalai_package_v1"
STATUS_PREPARED = "PASS_H1_M3_EVALAI_PACKAGES_PREPARED"
STATUS_DOCKER = "PASS_H1_M3_EVALAI_DOCKER_SMOKE"
STATUS_TERMINAL = "COMPLETE_H1_M3_EVALAI_SUBMISSION_READY_NO_SUBMISSION"
A1_SCHEMA = "h1_cal_aug_all_source_m3_deployment_v1_package_a1"
A1_TERMINAL_SHA256 = "4137495462a299e948beb58be578c739cc211330de4769992c03e743d7c7bf26"
PACKAGE_SHA256 = {
    "t0": "d2ad4beacbe5dd1bc26ec8764fcdb7745df83e795d743d9af5600f525860d760",
    "c1": "bfd02e51d2c0309a74b5e835668105f5102b55fe41458d519fce8895f1db5411",
}
CHECKPOINT_SHA256 = {
    "t0": "6d4d14226b706951274982438b588527beb442200aad2f50f9d18b68e54a9648",
    "c1": "0f406a8e69fdb57cf6a5480149f04ab3500e7fad849d36db38042edbadb2cd06",
}
IMAGE_TAGS = {"t0": "h1-cal-aug-m3-t0:epoch49", "c1": "h1-cal-aug-m3-c1:epoch49"}


class EvalAIPackageError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise EvalAIPackageError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path, schema: str | None = None) -> tuple[dict[str, Any], str]:
    digest = verify_sidecar(path)
    body = json.loads(path.read_text(encoding="utf-8"))
    if schema is not None:
        _need(body.get("schema") == schema, f"schema drift: {path}")
    return body, digest


def create_attempt(result_root: Path, package_root: Path, closure: Mapping[str, str], head: str) -> dict[str, Any]:
    root = result_root.resolve()
    _need(not root.exists(), f"EvalAI package result root is not fresh: {root}")
    body = {
        "schema": SCHEMA,
        "artifact": "attempt",
        "status": "ATTEMPT_BEFORE_PACKAGE_COPY_DOCKER_OR_CUDA",
        "created_at_utc": utc_now(),
        "git_head": head,
        "closure": dict(closure),
        "package_root": str(package_root.resolve()),
        "training": False,
        "checkpoint_selection": False,
        "optimizer_steps": 0,
        "backward_steps": 0,
        "model_updates": 0,
        "dataset_files_opened": 0,
        "evalai_credentials_accessed": False,
        "docker_pushes": 0,
        "evalai_submissions": 0,
    }
    publish_json(root / "attempt.json", body)
    return body


def load_attempt(result_root: Path) -> dict[str, Any]:
    body, _ = _load(result_root.resolve() / "attempt.json", SCHEMA)
    _need(body.get("status") == "ATTEMPT_BEFORE_PACKAGE_COPY_DOCKER_OR_CUDA", "attempt status drift")
    _need(body.get("training") is False and body.get("checkpoint_selection") is False, "attempt selection/training drift")
    return body


def prepare_packages(package_root: Path, staging_root: Path, result_root: Path) -> dict[str, Any]:
    import torch

    terminal, terminal_sha = _load(package_root.resolve() / "terminal.json", A1_SCHEMA)
    _need(terminal_sha == A1_TERMINAL_SHA256, "Package-A1 terminal SHA drift")
    authority, authority_sha = _load(package_root.resolve() / "packages/packages.json", "h1_cal_aug_all_source_m3_deployment_v1_packages")
    _need(authority_sha == terminal["packages_sha256"], "Package-A1 package authority drift")
    staging = staging_root.resolve()
    staging.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in authority["packages"]:
        arm = row["arm"]
        source = package_root.resolve() / row["relative"]
        _need(verify_sidecar(source) == row["sha256"] == PACKAGE_SHA256[arm], f"{arm} package SHA drift")
        _need(row["checkpoint_sha256"] == CHECKPOINT_SHA256[arm], f"{arm} checkpoint SHA drift")
        payload = torch.load(source, map_location="cpu", weights_only=False)
        _need(state_hash(payload["state_dict"]) == row["model_state_sha256"], f"{arm} model state drift")
        _need(len(payload["sessions"]) == 27 and payload["window_size"] == 700 and payload["prediction_divisor"] == 20.0, f"{arm} deployment payload drift")
        destination = staging / f"{arm}.pt"
        if destination.exists():
            _need(sha256_file(destination) == row["sha256"], f"existing staged {arm} differs")
        else:
            shutil.copyfile(source, destination)
            os.chmod(destination, 0o444)
        _need(sha256_file(destination) == row["sha256"], f"staged {arm} copy drift")
        rows.append({
            "arm": arm,
            "staged_path": str(destination),
            "package_sha256": row["sha256"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "model_state_sha256": row["model_state_sha256"],
            "session_payloads": len(payload["sessions"]),
            "heldout_m3_payloads": 14,
        })
    body = {
        "schema": f"{SCHEMA}_prepared_packages",
        "status": STATUS_PREPARED,
        "package_a1_terminal_sha256": terminal_sha,
        "package_authority_sha256": authority_sha,
        "source_authority_sha256": terminal["source_authority_sha256"],
        "calibration_authority_sha256": terminal["calibration_authority_sha256"],
        "selected_q": terminal["selected_q"],
        "selected_lambda": terminal["selected_lambda"],
        "packages": rows,
        "training": False,
        "checkpoint_selection": False,
        "dataset_files_opened": 0,
        "optimizer_steps": 0,
        "backward_steps": 0,
        "model_updates": 0,
        "docker_pushes": 0,
        "evalai_submissions": 0,
    }
    publish_json(result_root.resolve() / "packages/prepared.json", body)
    return body


def _run(command: list[str], log_path: Path, *, cwd: Path) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise EvalAIPackageError(f"command failed ({completed.returncode}): {' '.join(command)}; log={log_path}")
    return completed


def _json_output(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise EvalAIPackageError("container smoke emitted no JSON result")


def build_and_smoke(spint_root: Path, package_root: Path, staging_root: Path, result_root: Path, log_root: Path) -> dict[str, Any]:
    from falcon_challenge.config import FalconConfig, FalconTask
    from third_party.falcon_challenge.h1_carrier_id_spint_decoder import H1CarrierIdSpintDecoder

    prepared, prepared_sha = _load(result_root.resolve() / "packages/prepared.json", f"{SCHEMA}_prepared_packages")
    _need(prepared["status"] == STATUS_PREPARED, "packages not prepared")
    dockerfile = spint_root.resolve() / "third_party/falcon_challenge/h1_carrier_id_spint_sample.Dockerfile"
    results = []
    for row in prepared["packages"]:
        arm = row["arm"]
        tag = IMAGE_TAGS[arm]
        build = [
            "docker", "build", "--build-arg", f"ARM={arm}",
            "--build-arg", f"PACKAGE_SHA256={row['package_sha256']}",
            "--build-arg", f"CHECKPOINT_SHA256={row['checkpoint_sha256']}",
            "--build-arg", "BATCH_SIZE=8", "-t", tag, "-f", str(dockerfile), ".",
        ]
        _run(build, log_root.resolve() / f"build_{arm}.log", cwd=spint_root.resolve())
        inspect_output = subprocess.check_output(["docker", "image", "inspect", tag], text=True)
        inspect = json.loads(inspect_output)[0]
        labels = inspect["Config"]["Labels"]
        env = dict(item.split("=", 1) for item in inspect["Config"]["Env"] if "=" in item)
        _need(labels["ibci.h1.arm"] == arm and labels["ibci.h1.package.sha256"] == row["package_sha256"], f"{arm} image label drift")
        _need(labels["ibci.h1.checkpoint.sha256"] == row["checkpoint_sha256"], f"{arm} image checkpoint label drift")
        _need(env.get("TASK") == "h1" and env.get("PHASE") == "test" and env.get("EVALUATION_LOC") == "remote" and env.get("BATCH_SIZE") == "8", f"{arm} image runtime contract drift")

        host = H1CarrierIdSpintDecoder(FalconConfig(task=FalconTask.h1), package_root.resolve() / f"packages/{arm}.pt", batch_size=8, device="cpu")
        host_before = host.model_state_sha256()
        host.reset([Path("sub-HumanPitt-held-out-calib_ses-19250126T113454")])
        host_prediction = np.concatenate([host.predict(np.zeros((1, 176), np.float32)) for _ in range(3)], axis=0)
        _need(host.model_state_sha256() == host_before == row["model_state_sha256"], f"{arm} host smoke state drift")

        base_run = ["docker", "run", "--rm", "--entrypoint", "python", tag, "/decode.py", "--evaluation", "smoke", "--model-path", "/data/decoder.pt", "--batch-size", "8"]
        cpu_completed = _run(base_run + ["--device", "cpu"], log_root.resolve() / f"smoke_cpu_{arm}.log", cwd=spint_root.resolve())
        cpu = _json_output(cpu_completed.stdout)
        gpu_command = ["docker", "run", "--rm", "--gpus", "device=0", "--entrypoint", "python", tag, "/decode.py", "--evaluation", "smoke", "--model-path", "/data/decoder.pt", "--batch-size", "8", "--device", "cuda:0"]
        gpu_completed = _run(gpu_command, log_root.resolve() / f"smoke_gpu_{arm}.log", cwd=spint_root.resolve())
        gpu = _json_output(gpu_completed.stdout)
        for name, smoke in (("cpu", cpu), ("gpu", gpu)):
            _need(smoke["status"] == "PASS_H1_CARRIER_ID_SPINT_CONTAINER_SMOKE" and smoke["arm"] == arm, f"{arm} {name} smoke status drift")
            _need(smoke["checkpoint_sha256"] == row["checkpoint_sha256"] and smoke["model_state_sha256"] == row["model_state_sha256"], f"{arm} {name} provenance drift")
            _need(smoke["model_state_immutable"] is True, f"{arm} {name} state mutation")
        cpu_prediction = np.asarray(cpu["prediction"], dtype=np.float32)
        gpu_prediction = np.asarray(gpu["prediction"], dtype=np.float32)
        _need(np.array_equal(host_prediction, cpu_prediction), f"{arm} host/container CPU prediction drift")
        _need(np.allclose(host_prediction, gpu_prediction, rtol=2e-3, atol=2e-4), f"{arm} host/container GPU prediction drift")
        results.append({
            "arm": arm,
            "image_tag": tag,
            "image_id": inspect["Id"],
            "image_repo_digests": inspect.get("RepoDigests") or [],
            "package_sha256": row["package_sha256"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "model_state_sha256": row["model_state_sha256"],
            "runtime": {"task": "h1", "phase": "test", "evaluation": "remote", "batch_size": 8},
            "cpu_smoke": cpu,
            "gpu_smoke": gpu,
            "host_container_cpu_exact": True,
            "host_container_gpu_allclose": True,
            "model_state_immutable": True,
        })
    body = {
        "schema": f"{SCHEMA}_docker",
        "status": STATUS_DOCKER,
        "prepared_packages_sha256": prepared_sha,
        "images": results,
        "package_reload_numerical_equivalence": True,
        "container_cpu_gpu_smoke": True,
        "dataset_files_opened": 0,
        "heldout_scores_computed": 0,
        "training": False,
        "checkpoint_selection": False,
        "optimizer_steps": 0,
        "backward_steps": 0,
        "model_updates": 0,
        "evalai_credentials_accessed": False,
        "docker_pushes": 0,
        "evalai_submissions": 0,
    }
    publish_json(result_root.resolve() / "docker/authority.json", body)
    return body


def verify_terminal(result_root: Path) -> dict[str, Any]:
    root = result_root.resolve()
    attempt = load_attempt(root)
    attempt_sha = verify_sidecar(root / "attempt.json")
    prepared, prepared_sha = _load(root / "packages/prepared.json", f"{SCHEMA}_prepared_packages")
    docker, docker_sha = _load(root / "docker/authority.json", f"{SCHEMA}_docker")
    _need(prepared["status"] == STATUS_PREPARED and docker["status"] == STATUS_DOCKER, "terminal input status drift")
    _need(len(docker["images"]) == 2 and {row["arm"] for row in docker["images"]} == {"t0", "c1"}, "terminal image set drift")
    _need(docker["dataset_files_opened"] == docker["heldout_scores_computed"] == docker["docker_pushes"] == docker["evalai_submissions"] == 0, "forbidden external/data activity")
    body = {
        "schema": SCHEMA,
        "status": STATUS_TERMINAL,
        "finished_at_utc": utc_now(),
        "attempt_sha256": attempt_sha,
        "git_head": attempt["git_head"],
        "prepared_packages_sha256": prepared_sha,
        "docker_authority_sha256": docker_sha,
        "source_authority_sha256": prepared["source_authority_sha256"],
        "calibration_authority_sha256": prepared["calibration_authority_sha256"],
        "selected_q": prepared["selected_q"],
        "selected_lambda": prepared["selected_lambda"],
        "images": [{key: row[key] for key in ("arm", "image_tag", "image_id", "package_sha256", "checkpoint_sha256", "model_state_sha256")} for row in docker["images"]],
        "package_reload_numerical_equivalence": True,
        "container_cpu_gpu_smoke": True,
        "submission_ready_without_model_change": True,
        "training": False,
        "checkpoint_selection": False,
        "optimizer_steps": 0,
        "backward_steps": 0,
        "model_updates": 0,
        "dataset_files_opened": 0,
        "heldout_scores_computed": 0,
        "evalai_credentials_accessed": False,
        "docker_pushes": 0,
        "evalai_submissions": 0,
        "official_hidden_test_accessed": False,
    }
    publish_json(root / "terminal.json", body)
    return body


__all__ = (
    "SCHEMA", "STATUS_DOCKER", "STATUS_PREPARED", "STATUS_TERMINAL",
    "EvalAIPackageError", "build_and_smoke", "create_attempt", "load_attempt",
    "prepare_packages", "verify_terminal",
)
