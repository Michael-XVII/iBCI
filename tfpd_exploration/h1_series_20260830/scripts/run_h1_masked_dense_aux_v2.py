#!/usr/bin/env python3
"""One-shot H1 masked dense-auxiliary V2 final-legal-subset runner."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[3]
STREAMING = ROOT / "streaming_calibration_exp"
SCRIPTS = Path(__file__).resolve().parent
for import_root in (STREAMING, SCRIPTS):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import run_h1_masked_dense_aux_v1 as engine  # noqa: E402
from src.data.h1_masked_dense_aux_v1 import (  # noqa: E402
    SOURCE_DATES,
    SOURCE_SESSIONS,
    TARGET_SESSIONS,
)
from src.data.h1_masked_dense_aux_v2 import H1MaskedDenseAuxV2DataModule  # noqa: E402
from src.h1_masked_dense_aux_protocol_v1 import (  # noqa: E402
    outer_gate,
    select_source_lambda,
    sha256_file,
    source_cell_specs,
    verify_immutable,
    write_immutable_bytes,
    write_immutable_json,
)
from src.h1_masked_dense_aux_v2_protocol import evaluate_attrition  # noqa: E402
from src.models.components.spint import SpintModel  # noqa: E402
from src.models.h1_masked_dense_aux_v2 import H1MaskedDenseAuxV2LitModule  # noqa: E402


SCHEMA = "h1_masked_dense_aux_v2_final_legal_subset"
BASE_COMMIT = "b1dfc9d8b6516eefe41f9dbbe32525b5a72e10fa"
BRANCH = "exp/h1-masked-dense-aux-v2-final-legal-subset"
PYTHON = Path("/home/ial-mohd/workspace/envs/spint/bin/python")
DEFAULT_DATA = Path("/home/ial-mohd/dataset/ial-mohd/000954")
RESULT_ROOT = ROOT / "tfpd_exploration/h1_series_20260830/results/h1_masked_dense_aux_v2_final_legal_subset"
LOG_DIR = ROOT / "logs/h1_masked_dense_aux_v2_final_legal_subset"
EXPERIMENT1 = ROOT / "tfpd_exploration/h1_series_20260830/results/h1_window_mask_contract_v1"
V1_RESULT = ROOT / "tfpd_exploration/h1_series_20260830/results/h1_masked_dense_aux_v1"
CONFIG = STREAMING / "configs/experiment/h1_masked_dense_aux_v2_final_legal_subset.yaml"
WORKORDER = ROOT / "tfpd_exploration/h1_series_20260830/docs/WORKORDER_H1_MASKED_DENSE_AUX_V2_FINAL_LEGAL_SUBSET_20260831.md"
CLOSURE = (
    Path(__file__).resolve(),
    Path(engine.__file__).resolve(),
    STREAMING / "src/data/h1_window_mask_contract_v1.py",
    STREAMING / "src/data/h1_masked_dense_aux_v1.py",
    STREAMING / "src/data/h1_masked_dense_aux_v2.py",
    STREAMING / "src/models/h1_masked_dense_aux_v1.py",
    STREAMING / "src/models/h1_masked_dense_aux_v2.py",
    STREAMING / "src/h1_masked_dense_aux_protocol_v1.py",
    STREAMING / "src/h1_masked_dense_aux_v2_protocol.py",
    STREAMING / "tests/test_h1_window_mask_contract_v1.py",
    STREAMING / "tests/test_h1_masked_dense_aux_v1.py",
    STREAMING / "tests/test_h1_masked_dense_aux_v2.py",
    STREAMING / "tests/test_falcon_sampler.py",
    CONFIG,
    WORKORDER,
)


def configure_engine() -> None:
    """Bind the tested V1 execution engine to the additive V2 route."""
    engine.BASE_COMMIT = BASE_COMMIT
    engine.BRANCH = BRANCH
    engine.RESULT_ROOT = RESULT_ROOT
    engine.LOG_DIR = LOG_DIR
    engine.CONFIG = CONFIG
    engine.WORKORDER = WORKORDER
    engine.CLOSURE = CLOSURE
    engine.build_dm = build_dm
    engine.build_model = build_model
    engine.source_data_audit = source_data_audit
    engine.cell_command = cell_command
    engine.evaluate_outer = evaluate_outer
    engine.write_record = write_record


def build_dm(data_root: Path, validation_date: str | None, *, target: bool = False):
    return H1MaskedDenseAuxV2DataModule(
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


def build_model(lam: float) -> H1MaskedDenseAuxV2LitModule:
    net = SpintModel(
        model_dim=1024, num_covariates=7, window_size=700, num_heads=64,
        num_layers=1, num_id_layers=3, use_learnable_id=True,
        learnable_id_type="mlp", learnable_rep=True, dropout_rate=0.0,
        dynamic_dropout=True, dynamic_dropout_low=0.0, dynamic_dropout_high=1.0,
        tf_drop_rate=0.1, readin_layer_type="mlp",
    )
    return H1MaskedDenseAuxV2LitModule(
        task="h1", net=net, decode_last_timestep_only=True,
        predict_scaled_behavior=True, behavior_scaling_factor=20.0,
        optimizer=partial(torch.optim.Adam, lr=5e-5, weight_decay=0.0),
        scheduler=None, compile=False, dense_aux_lambda=lam,
    )


def preflight(log_path: Path) -> None:
    if RESULT_ROOT.exists():
        raise FileExistsError(f"one-shot result root already exists: {RESULT_ROOT}")
    if engine.run_text(["git", "branch", "--show-current"]) != BRANCH:
        raise RuntimeError(f"preflight requires branch {BRANCH}")
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"],
                   cwd=ROOT, check=True)
    exp1_path = EXPERIMENT1 / "terminal.json"
    exp1_sha = verify_immutable(exp1_path)
    if engine.load_json(exp1_path).get("status") != "PASS_WINDOW_MASK_CONTRACT_V1_CPU_GATE":
        raise RuntimeError("experiment-1 contract gate is not PASS")
    v1_failure_path = V1_RESULT / "failure_v2.json"
    v1_failure_sha = verify_immutable(v1_failure_path)
    v1_failure = engine.load_json(v1_failure_path)
    if "admitted windows must have a legal final position" not in v1_failure.get("error", ""):
        raise RuntimeError("V1 terminal receipt is not the registered final-still contradiction")
    closure = engine.closure_manifest()
    closure_sha = engine.closure_sha256(closure)
    write_immutable_json(RESULT_ROOT / "attempt.json", {
        "schema": SCHEMA, "artifact": "attempt",
        "status": "ATTEMPT_V2_CPU_NO_DATA_GATE", "created_at_utc": engine.utcnow(),
        "branch": BRANCH, "head_before_execution_commit": engine.git_head(),
        "required_v1_terminal_commit": BASE_COMMIT, "python": str(PYTHON),
        "python_version": sys.version, "closure": closure, "closure_sha256": closure_sha,
        "experiment1_terminal_sha256": exp1_sha,
        "v1_terminal_failure_path": str(v1_failure_path),
        "v1_terminal_failure_sha256": v1_failure_sha,
        "registered_change": "filter training windows to final-eval-true and final-nonstill only",
        "validation_domain": "unchanged legacy four-field last-bin windows",
        "scope": {"h1_data_opened": False, "cuda_initialized": torch.cuda.is_initialized(),
                  "gpu_allocated": False, "target_opened": False,
                  "formal_heldout_opened": False},
    })
    command = [
        str(PYTHON), "-m", "pytest", "-q",
        "tests/test_h1_window_mask_contract_v1.py",
        "tests/test_h1_masked_dense_aux_v1.py",
        "tests/test_h1_masked_dense_aux_v2.py",
        "tests/test_falcon_sampler.py",
    ]
    environment = dict(os.environ)
    environment.update({"PYTHONNOUSERSITE": "1", "CUDA_VISIBLE_DEVICES": ""})
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("wb") as stream:
        process = subprocess.run(command, cwd=STREAMING, env=environment,
                                 stdout=stream, stderr=subprocess.STDOUT)
    passed = process.returncode == 0 and not torch.cuda.is_initialized()
    receipt = {
        "schema": SCHEMA, "artifact": "cpu_gate", "finished_at_utc": engine.utcnow(),
        "status": "PASS_H1_MASKED_DENSE_AUX_V2_CPU_NO_DATA_GATE" if passed
                  else "FAIL_H1_MASKED_DENSE_AUX_V2_CPU_NO_DATA_GATE",
        "command": command, "cwd": str(STREAMING), "returncode": process.returncode,
        "log_path": str(log_path), "log_sha256": sha256_file(log_path),
        "elapsed_seconds": time.monotonic() - started,
        "closure_sha256": closure_sha, "experiment1_terminal_sha256": exp1_sha,
        "v1_terminal_failure_sha256": v1_failure_sha,
        "scope": {"h1_data_opened": False, "cuda_visible_devices": "",
                  "cuda_initialized": torch.cuda.is_initialized(), "training_steps": 0,
                  "target_opened": False, "formal_heldout_opened": False},
    }
    write_immutable_json(RESULT_ROOT / "cpu_gate.json", receipt)
    if not passed:
        write_immutable_json(RESULT_ROOT / "failure.json", {
            "schema": SCHEMA, "status": "FAIL_V2_CPU_GATE_NO_DATA_NO_GPU",
            "returncode": process.returncode, "target_opened": False,
            "formal_heldout_opened": False, "finished_at_utc": engine.utcnow(),
        })
        raise SystemExit(process.returncode or 1)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def source_data_audit(data_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    dm = build_dm(data_root, None)
    dm.setup("fit")
    audit = dm.window_mask_audit()
    gate = evaluate_attrition(audit["training"])
    receipt = {
        "schema": SCHEMA, "artifact": "source_data_attrition_audit",
        "status": gate["verdict"], "created_at_utc": engine.utcnow(),
        "data_root": str(data_root), "source_dates": list(SOURCE_DATES),
        "source_sessions": list(SOURCE_SESSIONS), "target_sessions_opened": [],
        "formal_heldout_opened": False, "training_population_audit": audit["training"],
        "legacy_validation_audit": audit["legacy_validation"], "attrition_gate": gate,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_immutable_json(RESULT_ROOT / "data_audit.json", receipt)
    if not gate["attrition_gate_passed"]:
        raise RuntimeError(
            "V2 source attrition gate failed; GPU smoke is forbidden; "
            f"see {RESULT_ROOT / 'data_audit.json'}"
        )
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


def evaluate_outer(data_root: Path, final_receipts: list[Mapping[str, Any]],
                   selected_lam: float) -> dict[str, Any]:
    engine.fixed_seed()
    dm = build_dm(data_root, None, target=True)
    dm.setup("fit")
    by_lambda = {float(row["lambda"]): row for row in final_receipts}
    scores: dict[float, dict[str, float]] = {}
    checkpoint_bindings: dict[str, Any] = {}
    for lam in (0.0, selected_lam):
        row = by_lambda[lam]
        checkpoint = Path(str(row["checkpoint_path"]))
        if verify_immutable(checkpoint) != row["checkpoint_sha256"]:
            raise RuntimeError("final checkpoint receipt mismatch before target evaluation")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = build_model(lam)
        model.net.load_state_dict(payload["net_state_dict"])
        before = engine.state_sha256(model.net.state_dict())
        if before != row["terminal_net_state_sha256"]:
            raise RuntimeError("loaded final network state differs from terminal receipt")
        model = model.cuda().eval()
        predictions: dict[str, list[torch.Tensor]] = {name: [] for name in TARGET_SESSIONS}
        targets: dict[str, list[torch.Tensor]] = {name: [] for name in TARGET_SESSIONS}
        with torch.no_grad():
            for neural, target, calibration, sessions in dm.val_dataloader():
                if len(set(sessions)) != 1:
                    raise RuntimeError("outer batch crosses recording boundary")
                session = sessions[0]
                output = model(neural.cuda(), calib_trialized_neural_features=calibration.cuda())
                predictions[session].append((output[:, -1, :] / 20.0).cpu())
                targets[session].append(target[:, -1, :].cpu())
        scores[lam] = {
            name: engine.variance_weighted_r2(
                torch.cat(predictions[name]), torch.cat(targets[name])
            ) for name in TARGET_SESSIONS
        }
        after = engine.state_sha256(model.net.state_dict())
        if before != after:
            raise RuntimeError("outer inference mutated source-trained network state")
        checkpoint_bindings[str(lam)] = {
            "path": str(checkpoint), "sha256": row["checkpoint_sha256"],
            "state_sha256_before_after": before,
        }
        del model
        torch.cuda.empty_cache()
    gate = outer_gate(scores[0.0], scores[selected_lam])
    gate["verdict"] = ("PASS_MASKED_DENSE_AUX_V2_OUTER" if gate["outer_gate_passed"]
                       else "FAIL_MASKED_DENSE_AUX_V2_OUTER")
    receipt = {
        "schema": SCHEMA, "artifact": "outer_eval", "created_at_utc": engine.utcnow(),
        "status": "PASS_ONE_SHOT_OUTER_EVALUATION_COMPLETED",
        "selected_lambda": selected_lam,
        "metric_population": "unchanged legacy unfiltered last-bin windows",
        "r2_by_arm_and_recording": {str(lam): values for lam, values in scores.items()},
        "equal_recording_mean_r2": {
            str(lam): sum(values.values()) / 2 for lam, values in scores.items()
        },
        "gate": gate, "target_sessions_opened_once": list(TARGET_SESSIONS),
        "target_data_audit": dm.window_mask_audit(),
        "checkpoint_bindings": checkpoint_bindings, "target_optimizer_steps": 0,
        "target_backward_steps": 0, "formal_heldout_opened": False,
    }
    write_immutable_json(RESULT_ROOT / "outer_eval.json", receipt)
    return receipt


def write_record(status: str, *, selection: Mapping[str, Any] | None = None,
                 outer: Mapping[str, Any] | None = None,
                 error: str | None = None) -> tuple[Path, str]:
    lines = [
        "# H1 Masked Dense-Auxiliary V2 — Experiment Record", "",
        f"- Status: `{status}`", f"- Branch: `{BRANCH}`",
        f"- Execution commit: `{engine.git_head()}`",
        f"- Required V1 terminal commit: `{BASE_COMMIT}`",
        f"- Interpreter: `{PYTHON}`", f"- Result root: `{RESULT_ROOT}`",
        f"- Per-cell logs: `{LOG_DIR}`", "", "## Registered V2 change", "",
        "Training uses the final-legal subset: final eval-mask true and non-still. "
        "T0 and all positive-lambda arms use identical filtered indices and samplers. "
        "Held-source validation and one-shot outer evaluation retain the original "
        "unfiltered four-field loader and last-bin variance-weighted R2 population.", "",
        "The source attrition gate precedes CUDA: every recording must retain at least "
        "25%, no recording may be empty, every represented trial must retain a window, "
        "and every retained final must satisfy the frozen contract.", "",
        "For each 50-epoch source cell, held-source-date last-bin R2 is evaluated and "
        "recorded at every epoch 0 through 49; only epoch 49 governs selection.", "",
    ]
    if selection is not None:
        lines.extend(["## Source selection", "", "```json",
                      json.dumps(dict(selection), indent=2, sort_keys=True), "```", ""])
    if outer is not None:
        lines.extend(["## One-shot outer result", "", "```json",
                      json.dumps(dict(outer), indent=2, sort_keys=True), "```", ""])
    if error is not None:
        lines.extend(["## Failure", "", error, ""])
    lines.extend([
        "## GPU authorization conclusion", "",
        "Physical GPUs 0–3 are user-authorized, with at most two idle devices used. "
        "GPU smoke is authorized only after the CPU/no-data and source attrition gates. "
        "Target fold-0 remains closed until the source lambda gate passes; the fourteen "
        "formal held-out recordings are never opened.", "",
    ])
    destination = RESULT_ROOT / "EXPERIMENT_RECORD.md"
    return write_immutable_bytes(destination, "\n".join(lines).encode())


def supervise(data_root: Path) -> None:
    target_opened = False
    try:
        if engine.run_text(["git", "branch", "--show-current"]) != BRANCH:
            raise RuntimeError(f"supervisor requires branch {BRANCH}")
        subprocess.run(["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"],
                       cwd=ROOT, check=True)
        if engine.run_text(["git", "status", "--porcelain", "--untracked-files=no"]):
            raise RuntimeError("tracked worktree must be clean before the one-shot supervisor")
        verify_immutable(RESULT_ROOT / "attempt.json")
        verify_immutable(RESULT_ROOT / "cpu_gate.json")
        cpu_gate = engine.load_json(RESULT_ROOT / "cpu_gate.json")
        if cpu_gate["status"] != "PASS_H1_MASKED_DENSE_AUX_V2_CPU_NO_DATA_GATE":
            raise RuntimeError("V2 CPU/no-data gate is not PASS")
        closure = engine.closure_manifest()
        if engine.closure_sha256(closure) != cpu_gate["closure_sha256"]:
            raise RuntimeError("execution closure changed after CPU/no-data gate")

        source_data_audit(data_root)
        initial_gpu_rows = engine.gpu_rows()
        for index in engine.GPU_ALLOWLIST:
            if index not in initial_gpu_rows:
                raise RuntimeError(f"allowed physical GPU missing: {index}")
        active_gpus = engine.wait_for_two_idle_gpus()
        write_immutable_json(RESULT_ROOT / "gpu_allocation.json", {
            "schema": SCHEMA, "artifact": "gpu_allocation",
            "created_at_utc": engine.utcnow(), "allowlist": list(engine.GPU_ALLOWLIST),
            "selected_physical_gpus": list(active_gpus),
            "max_parallel_cells": engine.MAX_PARALLEL_GPUS,
            "initial_rows": {str(i): initial_gpu_rows[i] for i in engine.GPU_ALLOWLIST},
            "gpu_2_3_user_authorized": True,
            "idle_rule": "memory_used_mib<1024 and utilization_percent<10",
        })

        smoke_specs = [
            {"cell_id": "smoke_t0", "validation_date": SOURCE_DATES[0], "lambda": 0.0},
            {"cell_id": "smoke_lambda_1", "validation_date": SOURCE_DATES[0], "lambda": 1.0},
        ]
        smoke = [engine.launch_cell(spec, active_gpus[0], data_root, smoke=True)
                 for spec in smoke_specs]
        if len({row["batch_sha256"] for row in smoke}) != 1:
            raise RuntimeError("T0/lambda1 smoke batch digest mismatch")
        if len({row["initial_net_state_sha256"] for row in smoke}) != 1:
            raise RuntimeError("T0/lambda1 smoke initialization mismatch")
        if not all(row["auxiliary_nonzero"] and row["all_gradients_finite"] for row in smoke):
            raise RuntimeError("smoke finite/nonzero auxiliary gate failed")
        write_immutable_json(RESULT_ROOT / "smoke.json", {
            "schema": SCHEMA, "artifact": "smoke_gate",
            "status": "PASS_V2_GPU_SMOKE_T0_AND_LAMBDA1",
            "created_at_utc": engine.utcnow(), "cells": smoke,
        })

        source_receipts = engine.run_parallel(list(source_cell_specs()), data_root, active_gpus)
        for date in SOURCE_DATES:
            matched = [row for row in source_receipts if row["validation_date"] == date]
            if len(matched) != 4 or len({row["batch_sha256"] for row in matched}) != 1:
                raise RuntimeError(f"{date}: four matched arms do not share first batch")
            if len({row["initial_net_state_sha256"] for row in matched}) != 1:
                raise RuntimeError(f"{date}: four matched arms do not share initialization")
            if any([entry["epoch_zero_based"] for entry in row["held_source_date_r2_by_epoch"]]
                   != list(range(50)) for row in matched):
                raise RuntimeError(f"{date}: held-source R2 is not exact epochs 0..49")
        selection_rows = [
            {"validation_date": row["validation_date"], "lambda": row["lambda"],
             "r2_mean": row["last_bin_r2_equal_recording_mean"],
             "terminal_receipt_path": row["terminal_receipt_path"],
             "terminal_receipt_sha256": row["terminal_receipt_sha256"]}
            for row in source_receipts
        ]
        selection = select_source_lambda(selection_rows)
        selection.update({"schema": SCHEMA, "created_at_utc": engine.utcnow(),
                          "terminal_epoch_zero_based": 49,
                          "every_epoch_held_source_r2_recorded": True})
        write_immutable_json(RESULT_ROOT / "selection.json", selection)
        if not selection["source_gate_passed"]:
            record_path, record_sha = write_record(selection["verdict"], selection=selection)
            write_immutable_json(RESULT_ROOT / "terminal.json", {
                "schema": SCHEMA, "status": selection["verdict"],
                "finished_at_utc": engine.utcnow(), "selection": selection,
                "target_sessions_opened": [], "target_bytes_read": 0,
                "formal_heldout_opened": False, "record_path": str(record_path),
                "record_sha256": record_sha,
            })
            return

        selected_lam = float(selection["selected_lambda"])
        final_specs = [
            {"cell_id": "final_all_source_t0", "lambda": 0.0, "validation_date": None},
            {"cell_id": f"final_all_source_lambda_{selected_lam:g}",
             "lambda": selected_lam, "validation_date": None},
        ]
        final_receipts = engine.run_parallel(final_specs, data_root, active_gpus,
                                             final_all_source=True)
        if len({row["batch_sha256"] for row in final_receipts}) != 1:
            raise RuntimeError("final paired fits do not share first batch")
        if len({row["initial_net_state_sha256"] for row in final_receipts}) != 1:
            raise RuntimeError("final paired fits do not share initialization")
        engine.wait_for_gpu(active_gpus[0])
        torch.cuda.set_device(active_gpus[0])
        target_opened = True
        outer = evaluate_outer(data_root, final_receipts, selected_lam)
        status = outer["gate"]["verdict"]
        record_path, record_sha = write_record(status, selection=selection, outer=outer)
        write_immutable_json(RESULT_ROOT / "terminal.json", {
            "schema": SCHEMA, "status": status, "finished_at_utc": engine.utcnow(),
            "selection": selection, "outer_gate": outer["gate"],
            "target_sessions_opened_once": list(TARGET_SESSIONS),
            "formal_heldout_opened": False, "record_path": str(record_path),
            "record_sha256": record_sha,
        })
    except BaseException as error:
        failure = {
            "schema": SCHEMA, "status": "FAIL_H1_MASKED_DENSE_AUX_V2_NO_RETRY",
            "finished_at_utc": engine.utcnow(), "target_opened": target_opened,
            "formal_heldout_opened": False, "error_type": type(error).__name__,
            "error": str(error), "traceback": traceback.format_exc(),
        }
        if not (RESULT_ROOT / "failure.json").exists():
            write_immutable_json(RESULT_ROOT / "failure.json", failure)
        if not (RESULT_ROOT / "EXPERIMENT_RECORD.md").exists():
            write_record(failure["status"], error=f"`{type(error).__name__}: {error}`")
        raise


def main() -> None:
    configure_engine()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight_parser = commands.add_parser("preflight")
    preflight_parser.add_argument("--log", type=Path, required=True)
    supervisor_parser = commands.add_parser("supervise")
    supervisor_parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    fit_parser = commands.add_parser("fit")
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
    elif args.command == "fit":
        engine.fit_cell(args)
    else:
        supervise(args.data_root.resolve())


if __name__ == "__main__":
    main()
