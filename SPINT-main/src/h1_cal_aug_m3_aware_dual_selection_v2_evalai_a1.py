"""Frozen packaging and official EvalAI submission authority for H1 V2 A1."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import uuid
from typing import Any, Mapping

import numpy as np

from src.h1_m4_cce_contract import sha256_file, state_hash


SCHEMA = "h1_cal_aug_m3_aware_dual_selection_v2_evalai_submission_a1"
STATUS_TERMINAL = "COMPLETE_H1_M3_AWARE_DUAL_SELECTION_V2_EVALAI_OFFICIAL_A1"
PREDECESSOR_COMMIT = "ae14a232d1dfc84de0661916da34fdb9596753c2"
PREDECESSOR_TERMINAL_SHA256 = "d0735087091bd1681804e76c96f10f9f262efc3295fd32aa2f24498a51a4a31a"
V1_PACKAGE_SHA256 = "bfd02e51d2c0309a74b5e835668105f5102b55fe41458d519fce8895f1db5411"
V1_OFFICIAL_HELDOUT_R2_MEAN = 0.28413945277226266
CHALLENGE_ID = 2319
PHASE_ID = 4599
PHASE_SLUG = "few-shot-test-2319"
CPU_GPU_RTOL = 2e-3
CPU_GPU_ATOL = 2e-4

CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "key": "c2_e49", "label": "C2-E49", "epoch_zero_based": 49,
        "checkpoint_sha256": "2264b627c83ec629c122d33b758831d66cae4b4397d49a96aaba9e6c0616e0c9",
        "model_state_sha256": "6ddcafb60c783cd6d5edcd8c11bb1ec98c3d4dc01da47b5e6202492cefb3495a",
        "selection_receipt": None, "image_tag": "h1-m3aware-v2-c2-e49:a1",
    },
    {
        "key": "c2_hi_e45", "label": "C2-HI-E45", "epoch_zero_based": 45,
        "checkpoint_sha256": "83d56e8ce82a41ec711e0b2ec5e57d66488fae18252189123a107d5523d9352f",
        "model_state_sha256": "bf619e5f33e24e7eabb0ee0488e105316b776a82eaefd4e2c902f73b4501182b",
        "selection_receipt": "selection/c2_hi.json", "image_tag": "h1-m3aware-v2-c2-hi-e45:a1",
    },
    {
        "key": "c2_ho_e15", "label": "C2-HO-E15", "epoch_zero_based": 15,
        "checkpoint_sha256": "ce46267eb220142b8ef1f2e5acf05194650796ac2752594995b40d2ad0950215",
        "model_state_sha256": "d22988987c307662c6c7caaa4947b153aa215d74c3e3b020d3e2ce681709b70a",
        "selection_receipt": "selection/c2_ho.json", "image_tag": "h1-m3aware-v2-c2-ho-e15:a1",
    },
)


class SubmissionA1Error(RuntimeError):
    pass


def _publish_file(path: Path, data: bytes) -> None:
    output = path.resolve()
    if output.exists():
        raise SubmissionA1Error(f"refuse to overwrite immutable artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data); handle.flush(); os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    _need(stat.S_IMODE(output.stat().st_mode) == 0o444, f"artifact mode drift: {output}")


def _publish_bytes(path: Path, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()
    _publish_file(path, data)
    _publish_file(path.with_name(path.name + ".sha256"), f"{digest}  {path.name}\n".encode("utf-8"))
    return digest


def publish_json(path: Path, value: Mapping[str, Any]) -> str:
    return _publish_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))


def publish_text(path: Path, value: str) -> str:
    return _publish_bytes(path, value.encode("utf-8"))


def verify_sidecar(path: Path) -> str:
    candidate = path.resolve(); sidecar = candidate.with_name(candidate.name + ".sha256")
    _need(candidate.is_file() and sidecar.is_file(), f"artifact/sidecar missing: {candidate}")
    _need(stat.S_IMODE(candidate.stat().st_mode) == stat.S_IMODE(sidecar.stat().st_mode) == 0o444, f"artifact/sidecar is not mode 0444: {candidate}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    actual = sha256_file(candidate)
    _need(len(fields) == 2 and fields[0] == actual and fields[1] == candidate.name, f"sidecar mismatch: {candidate}")
    return actual


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise SubmissionA1Error(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path, schema: str | None = None) -> tuple[dict[str, Any], str]:
    digest = verify_sidecar(path)
    body = json.loads(path.read_text(encoding="utf-8"))
    if schema is not None:
        _need(body.get("schema") == schema, f"schema drift: {path}")
    return body, digest


def _verify_legacy_binary_sidecar(path: Path, expected: str) -> dict[str, Any]:
    """Verify predecessor bytes without mutating Git-restored sidecar mode."""
    sidecar = path.with_name(path.name + ".sha256")
    _need(path.is_file() and sidecar.is_file(), f"legacy package/sidecar missing: {path}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    _need(len(fields) == 2 and fields[1] == path.name, f"legacy package sidecar format drift: {sidecar}")
    actual = sha256_file(path)
    _need(fields[0] == actual == expected, f"legacy package SHA drift: {path}")
    return {
        "body_mode": oct(path.stat().st_mode & 0o777),
        "sidecar_mode": oct(sidecar.stat().st_mode & 0o777),
        "sha256": actual,
        "sidecar_content_verified": True,
        "predecessor_files_modified": False,
    }


def verify_predecessor(repo_root: Path, eval_root: Path, train_root: Path, v1_package_root: Path) -> dict[str, Any]:
    import torch

    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PREDECESSOR_COMMIT, "HEAD"],
        cwd=repo_root, check=False,
    )
    _need(ancestor.returncode == 0, "current HEAD does not contain sealed predecessor commit")
    terminal, terminal_sha = _load(eval_root / "terminal.json")
    _need(terminal_sha == PREDECESSOR_TERMINAL_SHA256, "V2 evaluation terminal SHA drift")
    _need(terminal.get("status") == "COMPLETE_H1_CAL_AUG_M3_AWARE_DUAL_SELECTION_V2_EVAL_A1_NO_SUBMISSION", "V2 evaluation terminal status drift")
    training, training_sha = _load(train_root / "training/c2/terminal.json")
    _need(training.get("status") == "PASS_H1_M3_AWARE_V2_C2_ALL_EPOCHS", "C2 training terminal status drift")
    rows = {int(row["epoch_zero_based"]): row for row in training["checkpoints"]}
    resolved = []
    for candidate in CANDIDATES:
        epoch = int(candidate["epoch_zero_based"])
        row = rows[epoch]
        if candidate["selection_receipt"]:
            selection, selection_sha = _load(eval_root / candidate["selection_receipt"])
            selected = selection["selected"]
            _need(int(selected["epoch_zero_based"]) == epoch, f"{candidate['key']} selection epoch drift")
            _need(selected["checkpoint_sha256"] == candidate["checkpoint_sha256"], f"{candidate['key']} selection checkpoint drift")
        else:
            selection_sha = None
        _need(row["sha256"] == candidate["checkpoint_sha256"], f"{candidate['key']} terminal checkpoint drift")
        _need(row["state_sha256"] == candidate["model_state_sha256"], f"{candidate['key']} terminal state drift")
        checkpoint = train_root / row["relative"]
        _need(verify_sidecar(checkpoint) == row["sha256"], f"{candidate['key']} checkpoint sidecar drift")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        _need(state_hash(payload["state_dict"]) == row["state_sha256"], f"{candidate['key']} checkpoint state drift")
        resolved.append({**candidate, "checkpoint_path": str(checkpoint.resolve()), "checkpoint_relative": row["relative"], "global_step": row["global_step"], "selection_receipt_sha256": selection_sha})

    template_path = v1_package_root / "packages/c1.pt"
    template_file_authority = _verify_legacy_binary_sidecar(template_path, V1_PACKAGE_SHA256)
    template = torch.load(template_path, map_location="cpu", weights_only=False)
    _need(template.get("schema") == "h1_cal_aug_all_source_m3_deployment_v1_package", "V1 deployment package schema drift")
    _need(len(template.get("sessions", {})) == 27 and template.get("window_size") == 700 and template.get("prediction_divisor") == 20.0, "V1 deployment surface drift")
    return {
        "schema": f"{SCHEMA}_predecessor_authority",
        "status": "PASS_V2_A1_FROZEN_PREDECESSOR",
        "predecessor_commit": PREDECESSOR_COMMIT,
        "predecessor_terminal_sha256": terminal_sha,
        "training_terminal_sha256": training_sha,
        "v1_template_package_sha256": V1_PACKAGE_SHA256,
        "v1_template_file_authority": template_file_authority,
        "source_authority_sha256": template["source_authority_sha256"],
        "calibration_authority_sha256": template["calibration_authority_sha256"],
        "candidates": resolved,
        "training": False,
        "checkpoint_reselection": False,
        "carrier_refitting": False,
        "optimizer_steps": 0,
        "backward_steps": 0,
        "model_updates": 0,
    }


def create_attempt(result_root: Path, body: Mapping[str, Any]) -> None:
    _need(not result_root.exists(), f"result root is not fresh: {result_root}")
    publish_json(result_root / "attempt.json", {
        "schema": SCHEMA,
        "artifact": "attempt",
        "status": "ATTEMPT_BEFORE_PACKAGE_CUDA_DOCKER_OR_EVALAI",
        "created_at_utc": utc_now(),
        **dict(body),
        "training": False,
        "optimizer_steps": 0,
        "backward_steps": 0,
        "model_updates": 0,
        "post_selection_retraining": False,
        "docker_images": 0,
        "evalai_submissions": 0,
    })


def prepare_packages(result_root: Path, v1_package_root: Path) -> dict[str, Any]:
    import torch

    predecessor, predecessor_sha = _load(result_root / "predecessor_authority.json", f"{SCHEMA}_predecessor_authority")
    template = torch.load(v1_package_root / "packages/c1.pt", map_location="cpu", weights_only=False)
    rows = []
    for candidate in predecessor["candidates"]:
        checkpoint = torch.load(candidate["checkpoint_path"], map_location="cpu", weights_only=False)
        package = dict(template)
        package.update({
            "arm": candidate["key"],
            "state_dict": checkpoint["state_dict"],
            "model_state_sha256": candidate["model_state_sha256"],
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "optimizer_steps": 0,
            "backward_steps": 0,
            "model_updates": 0,
            "evalai_submissions": 0,
        })
        _need(state_hash(package["state_dict"]) == candidate["model_state_sha256"], f"{candidate['key']} package state drift")
        buffer = io.BytesIO()
        torch.save(package, buffer)
        destination = result_root / "packages" / f"{candidate['key']}.pt"
        package_sha = _publish_bytes(destination, buffer.getvalue())
        reloaded = torch.load(destination, map_location="cpu", weights_only=False)
        _need(state_hash(reloaded["state_dict"]) == candidate["model_state_sha256"], f"{candidate['key']} serialized state drift")
        rows.append({
            **{key: candidate[key] for key in ("key", "label", "epoch_zero_based", "checkpoint_sha256", "model_state_sha256", "image_tag", "global_step")},
            "relative": f"packages/{candidate['key']}.pt",
            "package_sha256": package_sha,
            "source_authority_sha256": package["source_authority_sha256"],
            "calibration_authority_sha256": package["calibration_authority_sha256"],
            "session_payloads": len(package["sessions"]),
        })
    _need(len({row["source_authority_sha256"] for row in rows}) == 1, "candidate source authority mismatch")
    _need(len({row["calibration_authority_sha256"] for row in rows}) == 1, "candidate calibration authority mismatch")
    body = {
        "schema": f"{SCHEMA}_packages",
        "status": "PASS_V2_A1_THREE_FROZEN_PACKAGES",
        "predecessor_authority_sha256": predecessor_sha,
        "v1_template_package_sha256": V1_PACKAGE_SHA256,
        "candidates": rows,
        "shared_deployment_semantics": ["H1CarrierIdSpint", "frozen all-source H-C authority", "earliest-M3 calibration", "W700", "last-bin", "prediction /20", "27-session mapping", "FALCON H1 remote test"],
        "candidate_specific_input": "state_dict and checkpoint/state provenance only",
        "training": False,
        "optimizer_steps": 0,
        "backward_steps": 0,
        "model_updates": 0,
    }
    publish_json(result_root / "packages/authority.json", body)
    return body


def _prediction(package_path: Path, device: str, batch_size: int) -> tuple[np.ndarray, str, bool]:
    import torch
    from falcon_challenge.config import FalconConfig, FalconTask
    from third_party.falcon_challenge.h1_carrier_id_spint_decoder import H1CarrierIdSpintDecoder

    decoder = H1CarrierIdSpintDecoder(FalconConfig(task=FalconTask.h1), package_path, batch_size=batch_size, device=device)
    before = decoder.model_state_sha256()
    tag = Path("sub-HumanPitt-held-out-calib_ses-19250126T113454")
    # Match the already validated container smoke input exactly.
    observations = [np.zeros((1, 176), dtype=np.float32) for _ in range(3)]
    decoder.reset([tag])
    first = np.concatenate([decoder.predict(value) for value in observations], axis=0)
    after_first = decoder.model_state_sha256()
    decoder.reset([tag])
    second = np.concatenate([decoder.predict(value) for value in observations], axis=0)
    after_second = decoder.model_state_sha256()
    exact = np.array_equal(first, second)
    _need(exact, f"{package_path.name} repeated {device} predictions differ")
    _need(before == after_first == after_second, f"{package_path.name} {device} model state mutation")
    _need(np.isfinite(first).all(), f"{package_path.name} {device} nonfinite prediction")
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    return first, before, exact


def host_smoke(result_root: Path) -> dict[str, Any]:
    packages, packages_sha = _load(result_root / "packages/authority.json", f"{SCHEMA}_packages")
    rows = []
    for row in packages["candidates"]:
        package_path = result_root / row["relative"]
        _need(verify_sidecar(package_path) == row["package_sha256"], f"{row['key']} package SHA drift")
        cpu1, cpu_state, cpu_exact = _prediction(package_path, "cpu", 1)
        cpu8, cpu8_state, cpu8_exact = _prediction(package_path, "cpu", 8)
        gpu1, gpu_state, gpu_exact = _prediction(package_path, "cuda:0", 1)
        gpu8, gpu8_state, gpu8_exact = _prediction(package_path, "cuda:0", 8)
        _need(np.allclose(cpu1, cpu8, rtol=CPU_GPU_RTOL, atol=CPU_GPU_ATOL), f"{row['key']} CPU batch-size drift")
        _need(np.allclose(gpu1, gpu8, rtol=CPU_GPU_RTOL, atol=CPU_GPU_ATOL), f"{row['key']} GPU batch-size drift")
        _need(np.allclose(cpu1, gpu1, rtol=CPU_GPU_RTOL, atol=CPU_GPU_ATOL), f"{row['key']} CPU/GPU prediction drift")
        _need(cpu_state == cpu8_state == gpu_state == gpu8_state == row["model_state_sha256"], f"{row['key']} reload state drift")
        rows.append({
            "key": row["key"], "package_sha256": row["package_sha256"], "model_state_sha256": cpu_state,
            "cpu_repeated_exact": cpu_exact and cpu8_exact, "gpu_repeated_exact": gpu_exact and gpu8_exact,
            "batch_size_1_8_compatible": True, "cpu_gpu_allclose": True,
            "cpu_gpu_rtol": CPU_GPU_RTOL, "cpu_gpu_atol": CPU_GPU_ATOL,
            "cpu_prediction": cpu1.tolist(), "gpu_prediction": gpu1.tolist(), "model_state_immutable": True,
        })
    body = {
        "schema": f"{SCHEMA}_host_smoke", "status": "PASS_V2_A1_HOST_CPU_GPU_RELOAD_RESET_PREDICT",
        "packages_authority_sha256": packages_sha, "candidates": rows, "training": False,
        "optimizer_steps": 0, "backward_steps": 0, "model_updates": 0,
    }
    publish_json(result_root / "host_smoke.json", body)
    return body


def _run(command: list[str], log_path: Path, cwd: Path) -> str:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["TQDM_DISABLE"] = "1"
    completed = subprocess.run(command, cwd=cwd, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    publish_text(log_path, completed.stdout)
    _need(completed.returncode == 0, f"command failed ({completed.returncode}); log={log_path}")
    return completed.stdout


def _last_json(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            return json.loads(line)
    raise SubmissionA1Error("smoke output contains no JSON")


def _fresh_log(path: Path) -> Path:
    return path if not path.exists() else path.with_name(path.stem + "_repair" + path.suffix)


def build_and_smoke_images(result_root: Path, log_root: Path, spint_root: Path, staging_root: Path) -> dict[str, Any]:
    packages, packages_sha = _load(result_root / "packages/authority.json", f"{SCHEMA}_packages")
    host, host_sha = _load(result_root / "host_smoke.json", f"{SCHEMA}_host_smoke")
    host_rows = {row["key"]: row for row in host["candidates"]}
    staging_root.mkdir(parents=True, exist_ok=True)
    dockerfile = spint_root / "third_party/falcon_challenge/h1_carrier_id_spint_sample.Dockerfile"
    images = []
    for row in packages["candidates"]:
        staged = staging_root / f"{row['key']}.pt"
        if not staged.exists():
            shutil.copyfile(result_root / row["relative"], staged)
            os.chmod(staged, 0o444)
        _need(sha256_file(staged) == row["package_sha256"], f"{row['key']} staged package drift")
        build = [
            "docker", "build", "--progress=plain", "--build-arg", f"ARM={row['key']}",
            "--build-arg", f"PACKAGE_SHA256={row['package_sha256']}",
            "--build-arg", f"CHECKPOINT_SHA256={row['checkpoint_sha256']}",
            "--build-arg", "BATCH_SIZE=8", "-t", row["image_tag"], "-f", str(dockerfile), ".",
        ]
        reused = False
        try:
            inspected = json.loads(subprocess.check_output(["docker", "image", "inspect", row["image_tag"]], text=True, stderr=subprocess.DEVNULL))[0]
            existing_labels = inspected["Config"]["Labels"]
            _need(existing_labels.get("ibci.h1.package.sha256") == row["package_sha256"] and existing_labels.get("ibci.h1.checkpoint.sha256") == row["checkpoint_sha256"], f"{row['key']} existing image identity drift")
            reused = True
        except subprocess.CalledProcessError:
            _run(build, _fresh_log(log_root / f"build_{row['key']}.log"), spint_root)
            inspected = json.loads(subprocess.check_output(["docker", "image", "inspect", row["image_tag"]], text=True))[0]
        labels = inspected["Config"]["Labels"]
        _need(labels["ibci.h1.package.sha256"] == row["package_sha256"], f"{row['key']} image package label drift")
        _need(labels["ibci.h1.checkpoint.sha256"] == row["checkpoint_sha256"], f"{row['key']} image checkpoint label drift")
        common = ["docker", "run", "--rm", "--entrypoint", "python", row["image_tag"], "/decode.py", "--evaluation", "smoke", "--model-path", "/data/decoder.pt", "--batch-size", "8"]
        cpu = _last_json(_run(common + ["--device", "cpu"], _fresh_log(log_root / f"smoke_cpu_{row['key']}.log"), spint_root))
        gpu = _last_json(_run(["docker", "run", "--rm", "--gpus", "device=0", "--entrypoint", "python", row["image_tag"], "/decode.py", "--evaluation", "smoke", "--model-path", "/data/decoder.pt", "--batch-size", "8", "--device", "cuda:0"], _fresh_log(log_root / f"smoke_gpu_{row['key']}.log"), spint_root))
        expected = host_rows[row["key"]]
        host_cpu = np.asarray(expected["cpu_prediction"], np.float32)
        container_cpu = np.asarray(cpu["prediction"], np.float32)
        container_gpu = np.asarray(gpu["prediction"], np.float32)
        _need(np.allclose(host_cpu, container_cpu, rtol=CPU_GPU_RTOL, atol=CPU_GPU_ATOL), f"{row['key']} host/container CPU drift")
        _need(np.allclose(host_cpu, container_gpu, rtol=CPU_GPU_RTOL, atol=CPU_GPU_ATOL), f"{row['key']} host/container GPU drift")
        for smoke in (cpu, gpu):
            _need(smoke["checkpoint_sha256"] == row["checkpoint_sha256"], f"{row['key']} container checkpoint drift")
            _need(smoke["model_state_sha256"] == row["model_state_sha256"] and smoke["model_state_immutable"] is True, f"{row['key']} container state drift")
        images.append({
            **{key: row[key] for key in ("key", "label", "epoch_zero_based", "checkpoint_sha256", "model_state_sha256", "package_sha256", "image_tag")},
            "image_id": inspected["Id"], "repo_digests": inspected.get("RepoDigests") or [],
            "host_container_cpu_allclose": True, "host_container_gpu_allclose": True,
            "container_cpu_smoke": cpu, "container_gpu_smoke": gpu, "image_reused_after_identity_check": reused,
        })
    body = {
        "schema": f"{SCHEMA}_docker_authority", "status": "PASS_V2_A1_THREE_IMMUTABLE_IMAGES",
        "packages_authority_sha256": packages_sha, "host_smoke_sha256": host_sha,
        "docker_recipe": str(dockerfile.relative_to(spint_root)), "identical_recipe_and_entrypoint": True,
        "images": images, "training": False, "optimizer_steps": 0, "backward_steps": 0,
        "model_updates": 0, "docker_images": 3, "evalai_submissions": 0,
    }
    publish_json(result_root / "docker/authority.json", body)
    return body


def create_submission_manifest(result_root: Path) -> dict[str, Any]:
    docker, docker_sha = _load(result_root / "docker/authority.json", f"{SCHEMA}_docker_authority")
    _need(len(docker["images"]) == 3, "three frozen images required")
    order = []
    for ordinal, (candidate, image) in enumerate(zip(CANDIDATES, docker["images"], strict=True), start=1):
        _need(candidate["key"] == image["key"] and candidate["image_tag"] == image["image_tag"], "image order drift")
        order.append({"ordinal": ordinal, **{key: image[key] for key in ("key", "label", "epoch_zero_based", "checkpoint_sha256", "model_state_sha256", "package_sha256", "image_tag", "image_id")}})
    body = {
        "schema": f"{SCHEMA}_manifest", "status": "SEALED_BEFORE_FIRST_V2_SCORE_ACCESS",
        "created_at_utc": utc_now(), "docker_authority_sha256": docker_sha,
        "challenge_id": CHALLENGE_ID, "challenge_phase": PHASE_SLUG, "challenge_phase_id": PHASE_ID,
        "submission_visibility": "private", "submission_order": order,
        "submission_settings_identical": True,
        "endpoints": {
            "primary": "Official Held Out R2 Mean",
            "secondary": ["Official Held Out R2 Std.", "Official Held In R2 Mean", "Normalized Latency"],
        },
        "primary_successor": "C2-HO-E15",
        "contrasts": [
            "C2-E49 - V1-C1-E49", "C2-HI-E45 - C2-E49", "C2-HO-E15 - C2-E49",
            "C2-HO-E15 - C2-HI-E45", "C2-HO-E15 - V1-C1-E49",
        ],
        "v1_c1_official_held_out_r2_mean": V1_OFFICIAL_HELDOUT_R2_MEAN,
        "immutable_rules": {
            "training": False, "finetuning": False, "checkpoint_reselection": False,
            "docker_rebuild_after_score": False, "hyperparameter_change": False,
            "carrier_refitting": False, "model_updates": False, "automatic_successor": False,
            "inspect_first_score_before_all_submissions": False,
        },
    }
    publish_json(result_root / "submission/manifest.json", body)
    return body


def verify_manifest_committed(repo_root: Path, manifest_path: Path) -> str:
    relative = str(manifest_path.resolve().relative_to(repo_root.resolve()))
    _need(subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=repo_root).returncode == 0, "manifest differs from HEAD")
    tracked = subprocess.check_output(["git", "ls-files", "--error-unmatch", relative], cwd=repo_root, text=True).strip()
    _need(tracked == relative, "manifest is not committed")
    head_bytes = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=repo_root)
    import hashlib
    _need(hashlib.sha256(head_bytes).hexdigest() == sha256_file(manifest_path), "committed manifest bytes drift")
    return sha256_file(manifest_path)


def submit_all(repo_root: Path, result_root: Path, log_root: Path, evalai_executable: Path) -> dict[str, Any]:
    manifest, manifest_sha = _load(result_root / "submission/manifest.json", f"{SCHEMA}_manifest")
    _need(verify_manifest_committed(repo_root, result_root / "submission/manifest.json") == manifest_sha, "manifest commit gate drift")
    submissions = []
    for row in manifest["submission_order"]:
        output = _run([str(evalai_executable), "push", row["image_tag"], "--phase", PHASE_SLUG, "--private"], log_root / f"submit_{row['key']}.log", repo_root)
        matches = re.findall(r"submission\s+(\d+)", output, flags=re.IGNORECASE)
        _need(matches, f"{row['key']} EvalAI output contains no submission ID")
        submissions.append({
            "ordinal": row["ordinal"], "key": row["key"], "label": row["label"],
            "submission_id": int(matches[-1]), "status": "submitted_score_not_accessed",
            "image_tag": row["image_tag"], "image_id": row["image_id"],
            "checkpoint_sha256": row["checkpoint_sha256"], "package_sha256": row["package_sha256"],
        })
    body = {
        "schema": f"{SCHEMA}_submissions", "status": "THREE_SUBMISSIONS_CREATED_BEFORE_SCORE_ACCESS",
        "created_at_utc": utc_now(), "submission_manifest_sha256": manifest_sha,
        "challenge_phase_id": PHASE_ID, "submissions": submissions,
        "scores_accessed_before_all_three_submitted": False,
        "training": False, "optimizer_steps": 0, "backward_steps": 0, "model_updates": 0,
        "docker_images": 3, "evalai_submissions": 3,
    }
    publish_json(result_root / "submission/submissions.json", body)
    return body


def collect_results(result_root: Path, log_root: Path, evalai_executable: Path) -> dict[str, Any]:
    submissions, submissions_sha = _load(result_root / "submission/submissions.json", f"{SCHEMA}_submissions")
    _need(len(submissions["submissions"]) == 3, "cannot access score before three submissions exist")
    status_rows = []
    for row in submissions["submissions"]:
        output = _run([str(evalai_executable), "submission", str(row["submission_id"])], log_root / f"status_{row['key']}.log", result_root)
        match = re.search(r"Submission Status\s*:\s*([^\n]+)", output, flags=re.IGNORECASE)
        _need(match is not None, f"{row['key']} status parse failed")
        status_rows.append(match.group(1).strip().lower())
    _need(all(status == "finished" for status in status_rows), f"not all submissions finished: {status_rows}")
    result_rows = []
    for row, status in zip(submissions["submissions"], status_rows, strict=True):
        output = _run([str(evalai_executable), "submission", str(row["submission_id"]), "result"], log_root / f"result_{row['key']}.log", result_root)
        payload = json.loads(next(line for line in output.splitlines() if line.strip().startswith("{")))
        metrics = payload["test_split_h1"]
        for key in ("Held Out R2 Mean", "Held Out R2 Std.", "Held In R2 Mean", "Normalized Latency"):
            _need(key in metrics, f"{row['key']} missing official metric: {key}")
        result_rows.append({**row, "status": status, "metrics": metrics})
    by_key = {row["key"]: row["metrics"]["Held Out R2 Mean"] for row in result_rows}
    contrasts = {
        "C2-E49 - V1-C1-E49": by_key["c2_e49"] - V1_OFFICIAL_HELDOUT_R2_MEAN,
        "C2-HI-E45 - C2-E49": by_key["c2_hi_e45"] - by_key["c2_e49"],
        "C2-HO-E15 - C2-E49": by_key["c2_ho_e15"] - by_key["c2_e49"],
        "C2-HO-E15 - C2-HI-E45": by_key["c2_ho_e15"] - by_key["c2_hi_e45"],
        "C2-HO-E15 - V1-C1-E49": by_key["c2_ho_e15"] - V1_OFFICIAL_HELDOUT_R2_MEAN,
    }
    body = {
        "schema": f"{SCHEMA}_official_results", "status": "SEALED_THREE_OFFICIAL_H1_RESULTS",
        "finished_at_utc": utc_now(), "submissions_sha256": submissions_sha,
        "challenge_id": CHALLENGE_ID, "challenge_phase": PHASE_SLUG, "challenge_phase_id": PHASE_ID,
        "primary_endpoint": "Held Out R2 Mean", "primary_successor": "C2-HO-E15",
        "v1_c1_official_held_out_r2_mean": V1_OFFICIAL_HELDOUT_R2_MEAN,
        "submissions": result_rows, "preregistered_contrasts": contrasts,
    }
    publish_json(result_root / "official_results.json", body)
    return body


def verify_terminal(result_root: Path) -> dict[str, Any]:
    manifest, manifest_sha = _load(result_root / "submission/manifest.json", f"{SCHEMA}_manifest")
    submissions, submissions_sha = _load(result_root / "submission/submissions.json", f"{SCHEMA}_submissions")
    results, results_sha = _load(result_root / "official_results.json", f"{SCHEMA}_official_results")
    docker, docker_sha = _load(result_root / "docker/authority.json", f"{SCHEMA}_docker_authority")
    _need(len(manifest["submission_order"]) == len(submissions["submissions"]) == len(results["submissions"]) == len(docker["images"]) == 3, "terminal candidate count drift")
    body = {
        "schema": SCHEMA, "status": STATUS_TERMINAL, "finished_at_utc": utc_now(),
        "submission_manifest_sha256": manifest_sha, "submissions_sha256": submissions_sha,
        "official_results_sha256": results_sha, "docker_authority_sha256": docker_sha,
        "submission_ids": [row["submission_id"] for row in submissions["submissions"]],
        "primary_successor": "C2-HO-E15", "preregistered_contrasts": results["preregistered_contrasts"],
        "training": False, "optimizer_steps": 0, "backward_steps": 0, "model_updates": 0,
        "post_selection_retraining": False, "docker_images": 3, "evalai_submissions": 3,
        "automatic_successor": False,
    }
    publish_json(result_root / "terminal.json", body)
    lines = [
        "# H1 CAL-AUG M3-Aware Dual-Selection V2 EvalAI Submission A1", "",
        f"Status: `{STATUS_TERMINAL}`", "",
        f"Phase: `{PHASE_SLUG}` (ID {PHASE_ID}). All three frozen images were submitted before score access.", "",
        "## Official results", "",
    ]
    for row in results["submissions"]:
        metrics = row["metrics"]
        lines.append(f"- {row['label']} (submission {row['submission_id']}): Held Out R2 Mean `{metrics['Held Out R2 Mean']}`, Held Out R2 Std. `{metrics['Held Out R2 Std.']}`, Held In R2 Mean `{metrics['Held In R2 Mean']}`, Normalized Latency `{metrics['Normalized Latency']}`.")
    lines.extend(["", "No training, optimizer, backward pass, model update, post-selection retraining, Docker rebuild after score, or automatic successor occurred.", ""])
    publish_text(result_root / "EXPERIMENT_RECORD.md", "\n".join(lines))
    return body


__all__ = (
    "CANDIDATES", "CHALLENGE_ID", "PHASE_ID", "PHASE_SLUG", "PREDECESSOR_COMMIT",
    "PREDECESSOR_TERMINAL_SHA256", "SCHEMA", "STATUS_TERMINAL", "V1_OFFICIAL_HELDOUT_R2_MEAN",
    "build_and_smoke_images", "collect_results", "create_attempt", "create_submission_manifest",
    "host_smoke", "prepare_packages", "submit_all", "verify_manifest_committed", "verify_predecessor", "verify_terminal",
)
