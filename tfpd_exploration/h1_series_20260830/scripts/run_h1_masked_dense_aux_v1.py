#!/usr/bin/env python3
"""One-shot, receipt-driven H1 masked dense-auxiliary experiment supervisor."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import lightning.pytorch as pl


ROOT = Path(__file__).resolve().parents[3]
STREAMING = ROOT / "streaming_calibration_exp"
if str(STREAMING) not in sys.path:
    sys.path.insert(0, str(STREAMING))

from src.data.h1_masked_dense_aux_v1 import (  # noqa: E402
    H1MaskedDenseAuxDataModule,
    SOURCE_DATES,
    SOURCE_SESSIONS,
    TARGET_SESSIONS,
)
from src.h1_masked_dense_aux_protocol_v1 import (  # noqa: E402
    SCHEMA,
    outer_gate,
    select_source_lambda,
    sha256_file,
    source_cell_specs,
    verify_immutable,
    write_immutable_bytes,
    write_immutable_json,
)
from src.models.components.spint import SpintModel  # noqa: E402
from src.models.h1_masked_dense_aux_v1 import H1MaskedDenseAuxLitModule  # noqa: E402


BASE_COMMIT = "d1c66774a8b1e081972b72e9d4f5a89829b4c700"
BRANCH = "exp/h1-masked-dense-aux-v1"
PYTHON = Path("/home/ial-mohd/workspace/envs/spint/bin/python")
DEFAULT_DATA = Path("/home/ial-mohd/dataset/ial-mohd/000954")
RESULT_ROOT = ROOT / "tfpd_exploration/h1_series_20260830/results/h1_masked_dense_aux_v1"
LOG_DIR = ROOT / "logs/h1_masked_dense_aux_v1"
GPU_ALLOWLIST = (0, 1, 2, 3)
MAX_PARALLEL_GPUS = 2
EXPERIMENT1 = ROOT / "tfpd_exploration/h1_series_20260830/results/h1_window_mask_contract_v1"
CONFIG = STREAMING / "configs/experiment/h1_masked_dense_aux_v1.yaml"
WORKORDER = ROOT / "tfpd_exploration/h1_series_20260830/docs/WORKORDER_H1_MASKED_DENSE_AUX_V1_20260831.md"
AMENDMENT = ROOT / "tfpd_exploration/h1_series_20260830/docs/AMENDMENT_H1_MASKED_DENSE_AUX_V1_GPU_AND_PATH_REPAIR_20260831.md"
CLOSURE = (
    Path(__file__).resolve(),
    STREAMING / "src/data/h1_window_mask_contract_v1.py",
    STREAMING / "src/data/h1_masked_dense_aux_v1.py",
    STREAMING / "src/models/h1_masked_dense_aux_v1.py",
    STREAMING / "src/h1_masked_dense_aux_protocol_v1.py",
    STREAMING / "tests/test_h1_window_mask_contract_v1.py",
    STREAMING / "tests/test_h1_masked_dense_aux_v1.py",
    CONFIG,
    WORKORDER,
    AMENDMENT,
)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_text(command: list[str], *, cwd: Path = ROOT, env: Mapping[str, str] | None = None) -> str:
    return subprocess.check_output(command, cwd=cwd, env=env, text=True).strip()


def git_head() -> str:
    return run_text(["git", "rev-parse", "HEAD"])


def closure_manifest() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in CLOSURE}


def closure_sha256(manifest: Mapping[str, str]) -> str:
    encoded = json.dumps(dict(manifest), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def tensor_batch_digest(batch) -> str:
    digest = hashlib.sha256(b"h1-masked-dense-batch-v1\0")
    for value in batch:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(repr(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(json.dumps(list(value), separators=(",", ":")).encode())
    return digest.hexdigest()


def state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256(b"h1-masked-dense-checkpoint-state-v1\0")
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def build_dm(data_root: Path, validation_date: str | None, *, target: bool = False):
    return H1MaskedDenseAuxDataModule(
        task="h1", data_dir=data_root,
        allowed_sessions=TARGET_SESSIONS if target else SOURCE_SESSIONS,
        validation_date=validation_date, batch_size=32, window_size=700,
        calibration_n_trials=2, random_calibration=True, smooth_calibration=False,
        max_trial_length=1024, standardize_covariates=False, use_intertrials=True,
        use_calib_intertrials=False, trial_feature_type="raw", remove_still_times=False,
        remove_calib_still_times=False, use_calib_active_segments=True,
        calib_n_active_segments=1, interpolate_trials=True,
        interpolate_trials_kind="cubic", pad_value=-1.0, pin_memory=False,
        sampler_seed=42, balance_session_batches=False,
        reshuffle_train_sampler_each_epoch=False, side_feature_group="none",
    )


def build_model(lam: float) -> H1MaskedDenseAuxLitModule:
    net = SpintModel(
        model_dim=1024, num_covariates=7, window_size=700, num_heads=64,
        num_layers=1, num_id_layers=3, use_learnable_id=True,
        learnable_id_type="mlp", learnable_rep=True, dropout_rate=0.0,
        dynamic_dropout=True, dynamic_dropout_low=0.0, dynamic_dropout_high=1.0,
        tf_drop_rate=0.1, readin_layer_type="mlp",
    )
    return H1MaskedDenseAuxLitModule(
        task="h1", net=net, decode_last_timestep_only=True,
        predict_scaled_behavior=True, behavior_scaling_factor=20.0,
        optimizer=partial(torch.optim.Adam, lr=5e-5, weight_decay=0.0),
        scheduler=None, compile=False, dense_aux_lambda=lam,
    )


def fixed_seed() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def gpu_rows() -> dict[int, dict[str, Any]]:
    output = run_text([
        "nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    rows = {}
    for line in output.splitlines():
        index, uuid, name, memory, utilization = [part.strip() for part in line.split(",")]
        rows[int(index)] = {"index": int(index), "uuid": uuid, "name": name,
                            "memory_used_mib": int(memory), "utilization_percent": int(utilization)}
    return rows


def gpu_is_idle(index: int) -> tuple[bool, dict[str, Any]]:
    row = gpu_rows()[index]
    return row["memory_used_mib"] < 1024 and row["utilization_percent"] < 10, row


def wait_for_gpu(index: int) -> dict[str, Any]:
    while True:
        idle, row = gpu_is_idle(index)
        if idle:
            return row
        print(f"[{utcnow()}] physical GPU {index} no longer idle; waiting 30s: {row}", flush=True)
        time.sleep(30)


def wait_for_two_idle_gpus() -> tuple[int, int]:
    while True:
        rows = gpu_rows()
        idle = [index for index in GPU_ALLOWLIST
                if rows[index]["memory_used_mib"] < 1024 and rows[index]["utilization_percent"] < 10]
        if len(idle) >= MAX_PARALLEL_GPUS:
            return idle[0], idle[1]
        print(f"[{utcnow()}] fewer than two authorized GPUs are idle; waiting 30s: {rows}", flush=True)
        time.sleep(30)


def preflight(log_path: Path) -> None:
    if RESULT_ROOT.exists():
        raise FileExistsError(f"one-shot result root already exists: {RESULT_ROOT}")
    if run_text(["git", "branch", "--show-current"]) != BRANCH:
        raise RuntimeError(f"preflight requires branch {BRANCH}")
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"], cwd=ROOT, check=True)
    exp1_terminal = EXPERIMENT1 / "terminal.json"
    exp1_sha = verify_immutable(exp1_terminal)
    exp1 = load_json(exp1_terminal)
    if exp1.get("status") != "PASS_WINDOW_MASK_CONTRACT_V1_CPU_GATE":
        raise RuntimeError("experiment-1 CPU contract gate is not PASS")
    closure = closure_manifest()
    attempt = {
        "schema": SCHEMA, "artifact": "attempt", "status": "ATTEMPT_CPU_NO_DATA_GATE",
        "created_at_utc": utcnow(), "branch": BRANCH, "head_before_preflight_commit": git_head(),
        "required_base_commit": BASE_COMMIT, "python": str(PYTHON),
        "python_version": sys.version, "closure": closure,
        "closure_sha256": closure_sha256(closure),
        "experiment1_terminal_sha256": exp1_sha,
        "scope": {"h1_data_opened": False, "cuda_initialized": torch.cuda.is_initialized(),
                  "gpu_allocated": False, "formal_heldout_opened": False},
    }
    write_immutable_json(RESULT_ROOT / "attempt.json", attempt)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON), "-m", "pytest", "-q",
        "tests/test_h1_window_mask_contract_v1.py",
        "tests/test_h1_masked_dense_aux_v1.py",
        "tests/test_falcon_sampler.py",
    ]
    environment = dict(os.environ)
    environment.update({"PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": ""})
    started = time.monotonic()
    with log_path.open("wb") as stream:
        process = subprocess.run(command, cwd=STREAMING, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    receipt = {
        "schema": SCHEMA, "artifact": "cpu_gate", "finished_at_utc": utcnow(),
        "command": command, "cwd": str(STREAMING), "log_path": str(log_path),
        "log_sha256": sha256_file(log_path), "returncode": process.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "experiment1_terminal_sha256": exp1_sha,
        "closure_sha256": closure_sha256(closure),
        "scope": {"h1_data_opened": False, "cuda_visible_devices": "",
                  "cuda_initialized": torch.cuda.is_initialized(), "training_steps": 0},
        "status": "PASS_H1_MASKED_DENSE_AUX_V1_CPU_NO_DATA_GATE" if process.returncode == 0
                  else "FAIL_H1_MASKED_DENSE_AUX_V1_CPU_NO_DATA_GATE",
    }
    write_immutable_json(RESULT_ROOT / "cpu_gate.json", receipt)
    if process.returncode:
        write_immutable_json(RESULT_ROOT / "failure.json", {
            "schema": SCHEMA, "status": "FAIL_CPU_GATE_NO_DATA_NO_GPU", "returncode": process.returncode,
            "target_opened": False, "finished_at_utc": utcnow(),
        })
        raise SystemExit(process.returncode)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def closure_amendment_gate(log_path: Path) -> None:
    """Rebind pure pre-commit formatting fixes without replacing the first PASS."""
    original_path = RESULT_ROOT / "cpu_gate.json"
    original_sha = verify_immutable(original_path)
    original = load_json(original_path)
    if original["status"] != "PASS_H1_MASKED_DENSE_AUX_V1_CPU_NO_DATA_GATE":
        raise RuntimeError("original CPU/no-data gate is not PASS")
    destination = RESULT_ROOT / "cpu_gate_closure_v2.json"
    if destination.exists():
        raise FileExistsError("closure amendment gate is one-shot")
    command = [
        str(PYTHON), "-m", "pytest", "-q",
        "tests/test_h1_window_mask_contract_v1.py",
        "tests/test_h1_masked_dense_aux_v1.py",
        "tests/test_falcon_sampler.py",
    ]
    environment = dict(os.environ)
    environment.update({"PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": ""})
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as stream:
        process = subprocess.run(command, cwd=STREAMING, env=environment,
                                 stdout=stream, stderr=subprocess.STDOUT)
    closure = closure_manifest()
    receipt = {
        "schema": SCHEMA, "artifact": "cpu_gate_closure_amendment_v2",
        "status": "PASS_H1_MASKED_DENSE_AUX_V1_CPU_NO_DATA_GATE" if process.returncode == 0
                  else "FAIL_H1_MASKED_DENSE_AUX_V1_CPU_NO_DATA_GATE",
        "reason": "post-PASS pre-commit whitespace normalization; no experiment/data/GPU attempt",
        "original_cpu_gate_path": str(original_path), "original_cpu_gate_sha256": original_sha,
        "finished_at_utc": utcnow(), "command": command, "returncode": process.returncode,
        "log_path": str(log_path), "log_sha256": sha256_file(log_path),
        "elapsed_seconds": time.monotonic() - started, "closure": closure,
        "closure_sha256": closure_sha256(closure),
        "scope": {"h1_data_opened": False, "cuda_visible_devices": "",
                  "cuda_initialized": torch.cuda.is_initialized(), "training_steps": 0},
    }
    write_immutable_json(destination, receipt)
    if process.returncode:
        raise SystemExit(process.returncode)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def repair_gate(log_path: Path) -> None:
    """Authorize one repaired attempt after the recorded pre-data type failure."""
    failure_path = RESULT_ROOT / "failure.json"
    failure_sha = verify_immutable(failure_path)
    failure = load_json(failure_path)
    expected_fragment = "'str' object has no attribute 'rglob'"
    if failure.get("target_opened") is not False or expected_fragment not in failure.get("error", ""):
        raise RuntimeError("historical failure is not the bounded pre-data Path-type failure")
    closure = closure_manifest()
    write_immutable_json(RESULT_ROOT / "repair_attempt_v2.json", {
        "schema": SCHEMA, "artifact": "repair_attempt_v2", "status": "ATTEMPT_REPAIR_CPU_NO_DATA_GATE",
        "created_at_utc": utcnow(), "historical_failure_path": str(failure_path),
        "historical_failure_sha256": failure_sha, "historical_nwb_opened": False,
        "historical_cuda_initialized": False, "repair": "preserve pathlib.Path in DataModule hparams",
        "gpu_authorization_amendment": {"allowed": [0, 1, 2, 3], "max_parallel": 2,
                                        "gpu_2_3_authorized_by_user": True},
        "closure": closure, "closure_sha256": closure_sha256(closure),
    })
    command = [str(PYTHON), "-m", "pytest", "-q",
               "tests/test_h1_window_mask_contract_v1.py",
               "tests/test_h1_masked_dense_aux_v1.py", "tests/test_falcon_sampler.py"]
    environment = dict(os.environ)
    environment.update({"PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": ""})
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as stream:
        process = subprocess.run(command, cwd=STREAMING, env=environment,
                                 stdout=stream, stderr=subprocess.STDOUT)
    receipt = {
        "schema": SCHEMA, "artifact": "cpu_gate_closure_repair_v3",
        "status": "PASS_H1_MASKED_DENSE_AUX_V1_CPU_NO_DATA_GATE" if process.returncode == 0
                  else "FAIL_H1_MASKED_DENSE_AUX_V1_CPU_NO_DATA_GATE",
        "finished_at_utc": utcnow(), "command": command, "returncode": process.returncode,
        "log_path": str(log_path), "log_sha256": sha256_file(log_path),
        "elapsed_seconds": time.monotonic() - started, "closure": closure,
        "closure_sha256": closure_sha256(closure),
        "scope": {"h1_data_opened": False, "cuda_visible_devices": "",
                  "cuda_initialized": torch.cuda.is_initialized(), "training_steps": 0},
    }
    gate_path, gate_sha = write_immutable_json(RESULT_ROOT / "cpu_gate_closure_v3.json", receipt)
    if process.returncode:
        raise SystemExit(process.returncode)
    write_immutable_json(RESULT_ROOT / "resume_authority_v2.json", {
        "schema": SCHEMA, "artifact": "resume_authority_v2",
        "status": "PASS_REPAIRED_SUPERVISOR_MAY_OPEN_SOURCE_DATA",
        "created_at_utc": utcnow(), "repair_gate_path": str(gate_path),
        "repair_gate_sha256": gate_sha, "historical_failure_sha256": failure_sha,
        "target_open_authorized_now": False, "gpu_2_3_user_authorized": True,
    })
    print(json.dumps(receipt, indent=2, sort_keys=True))


def fit_cell(args: argparse.Namespace) -> None:
    result_dir = Path(args.result_dir).resolve()
    if result_dir.exists():
        raise FileExistsError(f"cell result already exists; retries forbidden: {result_dir}")
    result_dir.mkdir(parents=True)
    physical_gpu = int(os.environ["H1_PHYSICAL_GPU"])
    if physical_gpu not in GPU_ALLOWLIST or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("cell GPU environment is outside the user-authorized physical 0-3 allowlist")
    closure = closure_manifest()
    write_immutable_json(result_dir / "attempt.json", {
        "schema": SCHEMA, "artifact": "cell_attempt", "cell_id": args.cell_id,
        "created_at_utc": utcnow(), "lambda": args.lam,
        "validation_date": args.validation_date, "smoke": args.smoke,
        "final_all_source": args.final_all_source, "physical_gpu": physical_gpu,
        "gpu_before_data": gpu_rows()[physical_gpu], "head": git_head(),
        "closure_sha256": closure_sha256(closure), "target_opened": False,
        "formal_heldout_opened": False,
    })
    started = time.monotonic()
    try:
        fixed_seed()
        dm = build_dm(Path(args.data_root), None if args.final_all_source else args.validation_date)
        dm.setup("fit")
        first_batch = next(iter(dm.train_dataloader()))
        batch_sha = tensor_batch_digest(first_batch)
        fixed_seed()
        model = build_model(args.lam)
        epochs = 1 if args.smoke else 50
        trainer = pl.Trainer(
            accelerator="gpu", devices=1, precision="32-true", min_epochs=epochs,
            max_epochs=epochs, logger=False, enable_checkpointing=False,
            enable_model_summary=False, num_sanity_val_steps=0,
            limit_train_batches=20 if args.smoke else 1.0,
            limit_val_batches=0 if (args.smoke or args.final_all_source) else 1.0,
            check_val_every_n_epoch=1, deterministic=False,
        )
        trainer.fit(model, train_dataloaders=dm.train_dataloader(),
                    val_dataloaders=dm.val_dataloader())
        expected_epoch = 0 if args.smoke else 49
        if model.last_completed_epoch != expected_epoch:
            raise RuntimeError(f"terminal epoch drift: {model.last_completed_epoch} != {expected_epoch}")
        if not model.all_training_terms_finite or not model.all_gradients_finite:
            raise RuntimeError("nonfinite training term or gradient")
        if not model.saw_nonzero_auxiliary:
            raise RuntimeError("masked dense auxiliary term was never nonzero")
        net_state = {name: value.detach().cpu() for name, value in model.net.state_dict().items()}
        if any(tensor.is_floating_point() and not torch.isfinite(tensor).all().item()
               for tensor in net_state.values()):
            raise RuntimeError("nonfinite terminal network state")
        checkpoint_payload = {
            "schema": SCHEMA, "cell_id": args.cell_id, "lambda": args.lam,
            "validation_date": args.validation_date, "epoch_zero_based": expected_epoch,
            "global_step": int(trainer.global_step), "net_state_dict": net_state,
            "metadata": {"seed": 42, "warm_start": False, "target_optimizer_steps": 0,
                         "selected_by": "fixed_terminal_epoch_no_selection"},
        }
        buffer = io.BytesIO()
        torch.save(checkpoint_payload, buffer)
        checkpoint_path, checkpoint_sha = write_immutable_bytes(result_dir / "epoch_terminal.pt", buffer.getvalue())
        receipt = {
            "schema": SCHEMA, "artifact": "cell_terminal", "status": "PASS_CELL",
            "cell_id": args.cell_id, "lambda": args.lam,
            "validation_date": args.validation_date, "smoke": args.smoke,
            "final_all_source": args.final_all_source, "physical_gpu": physical_gpu,
            "gpu_uuid": gpu_rows()[physical_gpu]["uuid"], "seed": 42,
            "batch_sha256": batch_sha, "initial_net_state_sha256": model.initial_net_state_sha256,
            "terminal_net_state_sha256": state_sha256(net_state),
            "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha,
            "epoch_zero_based": expected_epoch, "global_step": int(trainer.global_step),
            "last_training_terms": model.last_training_terms,
            "all_training_terms_finite": model.all_training_terms_finite,
            "all_gradients_finite": model.all_gradients_finite,
            "auxiliary_nonzero": model.saw_nonzero_auxiliary,
            "last_bin_r2_by_recording": model.last_validation_r2_by_session,
            "last_bin_r2_equal_recording_mean": model.last_validation_r2_mean,
            "held_source_date_r2_by_epoch": model.validation_r2_history,
            "data_audit": dm.window_mask_audit(), "elapsed_seconds": time.monotonic() - started,
            "finished_at_utc": utcnow(), "formal_heldout_opened": False, "target_opened": False,
        }
        if not args.smoke and not args.final_all_source and receipt["last_bin_r2_equal_recording_mean"] is None:
            raise RuntimeError("source screen cell did not produce terminal last-bin validation R2")
        if not args.smoke and not args.final_all_source:
            epochs_seen = [row["epoch_zero_based"] for row in model.validation_r2_history]
            if epochs_seen != list(range(50)):
                raise RuntimeError(f"held-source-date R2 history is not exact epochs 0..49: {epochs_seen}")
        write_immutable_json(result_dir / "terminal.json", receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)
    except BaseException as error:
        failure = {
            "schema": SCHEMA, "artifact": "cell_failure", "status": "FAIL_CELL_NO_RETRY",
            "cell_id": args.cell_id, "lambda": args.lam, "validation_date": args.validation_date,
            "smoke": args.smoke, "final_all_source": args.final_all_source,
            "physical_gpu": physical_gpu, "target_opened": False,
            "formal_heldout_opened": False, "error_type": type(error).__name__,
            "error": str(error), "traceback": traceback.format_exc(), "finished_at_utc": utcnow(),
        }
        if not (result_dir / "failure.json").exists():
            write_immutable_json(result_dir / "failure.json", failure)
        raise


def source_data_audit(data_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    dm = build_dm(data_root, None)
    dm.setup("fit")
    audit = dm.window_mask_audit()
    for split in ("train", "validation"):
        for session, row in audit[split].items():
            if not row["final_all_true"]:
                raise RuntimeError(f"{split}/{session}: final-position contract contradiction")
            for field in ("padded_positions_legal", "still_positions_legal", "intertrial_positions_legal"):
                if row[field] != 0:
                    raise RuntimeError(f"{split}/{session}: illegal positions survived mask: {field}={row[field]}")
    receipt = {
        "schema": SCHEMA, "artifact": "source_data_audit",
        "status": "PASS_SOURCE_ONLY_H1_DATA_AUDIT", "created_at_utc": utcnow(),
        "data_root": str(data_root), "source_dates": list(SOURCE_DATES),
        "source_sessions": list(SOURCE_SESSIONS), "target_sessions_opened": [],
        "formal_heldout_opened": False, "audit": audit,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_immutable_json(RESULT_ROOT / "data_audit.json", receipt)
    return receipt


def cell_command(spec: Mapping[str, Any], result_dir: Path, data_root: Path,
                 *, smoke: bool = False, final_all_source: bool = False) -> list[str]:
    command = [
        str(PYTHON), str(Path(__file__).resolve()), "fit",
        "--cell-id", str(spec["cell_id"]), "--lam", str(spec["lambda"]),
        "--validation-date", str(spec.get("validation_date") or SOURCE_DATES[0]),
        "--result-dir", str(result_dir), "--data-root", str(data_root),
    ]
    if smoke:
        command.append("--smoke")
    if final_all_source:
        command.append("--final-all-source")
    return command


def launch_cell(spec: Mapping[str, Any], physical_gpu: int, data_root: Path,
                *, smoke: bool = False, final_all_source: bool = False) -> dict[str, Any]:
    gpu_before = wait_for_gpu(physical_gpu)
    category = "smoke" if smoke else "final" if final_all_source else "source_cells"
    result_dir = RESULT_ROOT / category / str(spec["cell_id"])
    log_path = LOG_DIR / f"{spec['cell_id']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = cell_command(spec, result_dir, data_root, smoke=smoke, final_all_source=final_all_source)
    environment = dict(os.environ)
    environment.update({"PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": str(physical_gpu),
                        "H1_PHYSICAL_GPU": str(physical_gpu)})
    print(f"[{utcnow()}] launch {spec['cell_id']} on physical GPU {physical_gpu}: {gpu_before}", flush=True)
    with log_path.open("wb") as stream:
        process = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode != 0:
        raise RuntimeError(f"cell failed without retry: {spec['cell_id']} rc={process.returncode} log={log_path}")
    terminal = result_dir / "terminal.json"
    terminal_sha = verify_immutable(terminal)
    receipt = load_json(terminal)
    receipt["terminal_receipt_path"] = str(terminal)
    receipt["terminal_receipt_sha256"] = terminal_sha
    receipt["launcher_log_path"] = str(log_path)
    receipt["launcher_log_sha256"] = sha256_file(log_path)
    return receipt


def run_parallel(specs: list[Mapping[str, Any]], data_root: Path,
                 physical_gpus: tuple[int, int], *, final_all_source: bool = False) -> list[dict[str, Any]]:
    assignments = {gpu: specs[offset::len(physical_gpus)] for offset, gpu in enumerate(physical_gpus)}
    outputs: list[dict[str, Any]] = []
    errors: list[BaseException] = []
    lock = threading.Lock()
    stop = threading.Event()

    def worker(gpu: int) -> None:
        for spec in assignments[gpu]:
            if stop.is_set():
                return
            try:
                receipt = launch_cell(spec, gpu, data_root, final_all_source=final_all_source)
                with lock:
                    outputs.append(receipt)
            except BaseException as error:
                with lock:
                    errors.append(error)
                stop.set()
                return

    threads = [threading.Thread(target=worker, args=(gpu,), name=f"gpu-{gpu}") for gpu in physical_gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    if len(outputs) != len(specs):
        raise RuntimeError(f"parallel grid incomplete: {len(outputs)} != {len(specs)}")
    return sorted(outputs, key=lambda row: row["cell_id"])


def variance_weighted_r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.double()
    target = target.double()
    residual = ((target - pred) ** 2).sum(dim=0)
    total = ((target - target.mean(dim=0)) ** 2).sum(dim=0)
    if torch.any(total <= 0):
        raise RuntimeError("outer recording has a constant behavior channel")
    return float((1.0 - residual.sum() / total.sum()).item())


def evaluate_outer(data_root: Path, final_receipts: list[Mapping[str, Any]], selected_lam: float) -> dict[str, Any]:
    fixed_seed()
    dm = build_dm(data_root, None, target=True)
    dm.setup("fit")
    by_lambda = {float(row["lambda"]): row for row in final_receipts}
    scores: dict[float, dict[str, float]] = {}
    checkpoint_bindings = {}
    for lam in (0.0, selected_lam):
        row = by_lambda[lam]
        checkpoint = Path(str(row["checkpoint_path"]))
        if verify_immutable(checkpoint) != row["checkpoint_sha256"]:
            raise RuntimeError("final checkpoint receipt mismatch before target evaluation")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = build_model(lam)
        model.net.load_state_dict(payload["net_state_dict"])
        before = state_sha256(model.net.state_dict())
        if before != row["terminal_net_state_sha256"]:
            raise RuntimeError("loaded final network state differs from terminal receipt")
        model = model.cuda().eval()
        predictions: dict[str, list[torch.Tensor]] = {name: [] for name in TARGET_SESSIONS}
        targets: dict[str, list[torch.Tensor]] = {name: [] for name in TARGET_SESSIONS}
        with torch.no_grad():
            for neural, target, calibration, sessions, _mask in dm.val_dataloader():
                if len(set(sessions)) != 1:
                    raise RuntimeError("outer batch crosses recording boundary")
                session = sessions[0]
                output = model(neural.cuda(), calib_trialized_neural_features=calibration.cuda())
                output = output[:, -1, :] / 20.0
                predictions[session].append(output.cpu())
                targets[session].append(target[:, -1, :].cpu())
        scores[lam] = {
            name: variance_weighted_r2(torch.cat(predictions[name]), torch.cat(targets[name]))
            for name in TARGET_SESSIONS
        }
        after = state_sha256(model.net.state_dict())
        if before != after:
            raise RuntimeError("outer inference mutated source-trained network state")
        checkpoint_bindings[str(lam)] = {"path": str(checkpoint), "sha256": row["checkpoint_sha256"],
                                            "state_sha256_before_after": before}
        del model
        torch.cuda.empty_cache()
    gate = outer_gate(scores[0.0], scores[selected_lam])
    receipt = {
        "schema": SCHEMA, "artifact": "outer_eval", "created_at_utc": utcnow(),
        "status": "PASS_ONE_SHOT_OUTER_EVALUATION_COMPLETED", "selected_lambda": selected_lam,
        "r2_by_arm_and_recording": {str(lam): values for lam, values in scores.items()},
        "equal_recording_mean_r2": {str(lam): sum(values.values()) / 2 for lam, values in scores.items()},
        "gate": gate, "target_sessions_opened_once": list(TARGET_SESSIONS),
        "target_data_audit": dm.window_mask_audit(), "checkpoint_bindings": checkpoint_bindings,
        "target_optimizer_steps": 0, "target_backward_steps": 0, "formal_heldout_opened": False,
    }
    write_immutable_json(RESULT_ROOT / "outer_eval.json", receipt)
    return receipt


def write_record(status: str, *, selection: Mapping[str, Any] | None = None,
                 outer: Mapping[str, Any] | None = None, error: str | None = None) -> tuple[Path, str]:
    lines = [
        "# H1 Masked Dense-Auxiliary V1 — Experiment Record", "",
        f"- Status: `{status}`", f"- Branch: `{BRANCH}`", f"- Execution commit: `{git_head()}`",
        f"- Required experiment-1 ancestor: `{BASE_COMMIT}`",
        f"- Interpreter: `{PYTHON}`", f"- Result root: `{RESULT_ROOT}`",
        f"- Aggregate/per-cell logs: `{ROOT / 'logs'}`", "",
        "## Scope and protocol", "",
        "The five source dates were screened by grouped leave-one-date-out. Every epoch 0–49 evaluated the held-source-date recordings and each cell receipt stores per-recording and equal-recording-mean last-bin variance-weighted R². Only epoch 49 governs lambda selection. The two 19250101 recordings are the one-shot outer fold; the fourteen formal held-out recordings were never opened.", "",
        "The frozen recipe is W=700, two calibration trials, batch 32, Adam 5e-5, FP32, seed 42, 50 epochs, no early stopping, raw unsmoothed input, random calibration, active calibration segments, and cubic interpolation. Training loss is last-bin MSE plus lambda times contract-masked dense MSE; all validation and report metrics are last-bin only.", "",
    ]
    if selection is not None:
        lines.extend(["## Source selection", "", "```json",
                      json.dumps(dict(selection), indent=2, sort_keys=True), "```", ""])
    if outer is not None:
        lines.extend(["## One-shot outer result", "", "```json",
                      json.dumps(dict(outer), indent=2, sort_keys=True), "```", ""])
    if error is not None:
        lines.extend(["## Failure", "", error, ""])
    lines.extend(["## GPU authorization conclusion", "",
                  "GPU training was authorized only after the experiment-1 contract gate, this experiment's CPU/no-data gate, source data audit, and two smoke cells passed. Following the user amendment, physical GPUs 0–3 were eligible, but the supervisor selected at most two idle devices and checked them again before every cell. A failed source gate does not authorize outer target access.", ""])
    destination = RESULT_ROOT / "EXPERIMENT_RECORD.md"
    if destination.exists():
        destination = RESULT_ROOT / "EXPERIMENT_RECORD_v2.md"
    return write_immutable_bytes(destination, "\n".join(lines).encode())


def supervise(data_root: Path) -> None:
    target_opened = False
    try:
        if run_text(["git", "branch", "--show-current"]) != BRANCH:
            raise RuntimeError(f"supervisor requires branch {BRANCH}")
        subprocess.run(["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"], cwd=ROOT, check=True)
        if run_text(["git", "status", "--porcelain", "--untracked-files=no"]):
            raise RuntimeError("tracked worktree must be clean before the one-shot supervisor")
        repaired_gate = RESULT_ROOT / "cpu_gate_closure_v3.json"
        amended_gate = RESULT_ROOT / "cpu_gate_closure_v2.json"
        cpu_gate_path = (repaired_gate if repaired_gate.exists() else amended_gate
                         if amended_gate.exists() else RESULT_ROOT / "cpu_gate.json")
        verify_immutable(RESULT_ROOT / "attempt.json")
        verify_immutable(cpu_gate_path)
        cpu_gate = load_json(cpu_gate_path)
        if cpu_gate["status"] != "PASS_H1_MASKED_DENSE_AUX_V1_CPU_NO_DATA_GATE":
            raise RuntimeError("CPU/no-data gate is not PASS")
        if (RESULT_ROOT / "failure.json").exists():
            resume_path = RESULT_ROOT / "resume_authority_v2.json"
            verify_immutable(resume_path)
            if load_json(resume_path).get("status") != "PASS_REPAIRED_SUPERVISOR_MAY_OPEN_SOURCE_DATA":
                raise RuntimeError("pre-data repair does not have additive resume authority")
        closure = closure_manifest()
        if closure_sha256(closure) != cpu_gate["closure_sha256"]:
            raise RuntimeError("execution closure changed after CPU/no-data gate")
        source_data_audit(data_root)

        initial_gpu_rows = gpu_rows()
        for index in GPU_ALLOWLIST:
            if index not in initial_gpu_rows:
                raise RuntimeError(f"allowed physical GPU missing: {index}")
        active_gpus = wait_for_two_idle_gpus()
        write_immutable_json(RESULT_ROOT / "gpu_allocation.json", {
            "schema": SCHEMA, "artifact": "gpu_allocation", "created_at_utc": utcnow(),
            "allowlist": list(GPU_ALLOWLIST), "selected_physical_gpus": list(active_gpus),
            "max_parallel_cells": MAX_PARALLEL_GPUS,
            "initial_rows": {str(i): initial_gpu_rows[i] for i in GPU_ALLOWLIST},
            "gpu_2_3_user_authorized": True, "idle_rule": "memory_used_mib<1024 and utilization_percent<10",
            "user_local_gpu_override_authorized": True,
        })

        smoke_specs = [
            {"cell_id": "smoke_t0", "validation_date": SOURCE_DATES[0], "lambda": 0.0},
            {"cell_id": "smoke_lambda_1", "validation_date": SOURCE_DATES[0], "lambda": 1.0},
        ]
        smoke = [launch_cell(spec, active_gpus[0], data_root, smoke=True) for spec in smoke_specs]
        if len({row["batch_sha256"] for row in smoke}) != 1:
            raise RuntimeError("T0/lambda1 smoke batch digest mismatch")
        if len({row["initial_net_state_sha256"] for row in smoke}) != 1:
            raise RuntimeError("T0/lambda1 smoke initialization mismatch")
        if not all(row["auxiliary_nonzero"] and row["all_gradients_finite"] for row in smoke):
            raise RuntimeError("smoke finite/nonzero auxiliary gate failed")
        write_immutable_json(RESULT_ROOT / "smoke.json", {
            "schema": SCHEMA, "artifact": "smoke_gate", "status": "PASS_GPU_SMOKE_T0_AND_LAMBDA1",
            "created_at_utc": utcnow(), "cells": smoke,
        })

        source_receipts = run_parallel(list(source_cell_specs()), data_root, active_gpus)
        for date in SOURCE_DATES:
            matched = [row for row in source_receipts if row["validation_date"] == date]
            if len(matched) != 4 or len({row["batch_sha256"] for row in matched}) != 1:
                raise RuntimeError(f"{date}: four matched arms do not share first batch")
            if len({row["initial_net_state_sha256"] for row in matched}) != 1:
                raise RuntimeError(f"{date}: four matched arms do not share initialization")
            if any(len(row["held_source_date_r2_by_epoch"]) != 50 for row in matched):
                raise RuntimeError(f"{date}: missing per-epoch held-source R2 history")
        selection_rows = [
            {"validation_date": row["validation_date"], "lambda": row["lambda"],
             "r2_mean": row["last_bin_r2_equal_recording_mean"],
             "terminal_receipt_path": row["terminal_receipt_path"],
             "terminal_receipt_sha256": row["terminal_receipt_sha256"]}
            for row in source_receipts
        ]
        selection = select_source_lambda(selection_rows)
        selection["created_at_utc"] = utcnow()
        selection["terminal_epoch_zero_based"] = 49
        selection["every_epoch_held_source_r2_recorded"] = True
        write_immutable_json(RESULT_ROOT / "selection.json", selection)
        if not selection["source_gate_passed"]:
            record_path, record_sha = write_record(selection["verdict"], selection=selection)
            write_immutable_json(RESULT_ROOT / "terminal.json", {
                "schema": SCHEMA, "status": selection["verdict"], "finished_at_utc": utcnow(),
                "selection": selection, "target_sessions_opened": [], "target_bytes_read": 0,
                "formal_heldout_opened": False, "record_path": str(record_path), "record_sha256": record_sha,
            })
            print(json.dumps(selection, indent=2, sort_keys=True), flush=True)
            return

        selected_lam = float(selection["selected_lambda"])
        final_specs = [
            {"cell_id": "final_all_source_t0", "lambda": 0.0, "validation_date": None},
            {"cell_id": f"final_all_source_lambda_{selected_lam:g}", "lambda": selected_lam, "validation_date": None},
        ]
        final_receipts = run_parallel(final_specs, data_root, active_gpus, final_all_source=True)
        if len({row["batch_sha256"] for row in final_receipts}) != 1:
            raise RuntimeError("final paired fits do not share first batch")
        if len({row["initial_net_state_sha256"] for row in final_receipts}) != 1:
            raise RuntimeError("final paired fits do not share initialization")
        wait_for_gpu(active_gpus[0])
        torch.cuda.set_device(active_gpus[0])
        target_opened = True
        outer = evaluate_outer(data_root, final_receipts, selected_lam)
        status = outer["gate"]["verdict"]
        record_path, record_sha = write_record(status, selection=selection, outer=outer)
        write_immutable_json(RESULT_ROOT / "terminal.json", {
            "schema": SCHEMA, "status": status, "finished_at_utc": utcnow(),
            "selection": selection, "outer_gate": outer["gate"],
            "target_sessions_opened_once": list(TARGET_SESSIONS), "formal_heldout_opened": False,
            "record_path": str(record_path), "record_sha256": record_sha,
        })
        print(json.dumps({"status": status, "selection": selection, "outer": outer}, indent=2, sort_keys=True), flush=True)
    except BaseException as error:
        failure = {
            "schema": SCHEMA, "status": "FAIL_H1_MASKED_DENSE_AUX_V1_NO_RETRY",
            "finished_at_utc": utcnow(), "target_opened": target_opened,
            "formal_heldout_opened": False, "error_type": type(error).__name__,
            "error": str(error), "traceback": traceback.format_exc(),
        }
        failure_path = RESULT_ROOT / "failure.json"
        if failure_path.exists():
            failure_path = RESULT_ROOT / "failure_v2.json"
        if not failure_path.exists():
            write_immutable_json(failure_path, failure)
        if not (RESULT_ROOT / "EXPERIMENT_RECORD_v2.md").exists():
            write_record(failure["status"], error=f"`{type(error).__name__}: {error}`")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--log", type=Path, required=True)
    amendment_parser = subparsers.add_parser("closure-amendment-gate")
    amendment_parser.add_argument("--log", type=Path, required=True)
    repair_parser = subparsers.add_parser("repair-gate")
    repair_parser.add_argument("--log", type=Path, required=True)
    supervisor_parser = subparsers.add_parser("supervise")
    supervisor_parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--cell-id", required=True)
    fit_parser.add_argument("--lam", type=float, choices=(0.0, 0.1, 0.3, 1.0), required=True)
    fit_parser.add_argument("--validation-date", choices=SOURCE_DATES, required=True)
    fit_parser.add_argument("--result-dir", type=Path, required=True)
    fit_parser.add_argument("--data-root", type=Path, required=True)
    fit_parser.add_argument("--smoke", action="store_true")
    fit_parser.add_argument("--final-all-source", action="store_true")
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(args.log.resolve())
    elif args.command == "closure-amendment-gate":
        closure_amendment_gate(args.log.resolve())
    elif args.command == "repair-gate":
        repair_gate(args.log.resolve())
    elif args.command == "fit":
        fit_cell(args)
    else:
        supervise(args.data_root.resolve())


if __name__ == "__main__":
    main()
