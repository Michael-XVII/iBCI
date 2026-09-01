"""Five-date causal output EMA evaluation for regenerated H1 H-C checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import stat
import subprocess
import time
from typing import Any, Mapping, Sequence

import numpy as np

from src.data.h1_m4_cce_date_lodo import target_sessions_for_date
from src.data.h1_m4_eb_pilot import (
    EXPECTED_NEURONS,
    H1PilotRecord,
    array_sha256,
    fit_frozen_carrier,
    index_heldin_calib,
    interpolate_trial_identity,
    load_record,
)
from src.h1_hc_date_lodo_regen_v1 import (
    CHECKPOINT_SCHEMA,
    RegenPlan,
    model_config,
    publish_json,
    publish_npz,
    publish_text,
    validate_checkpoint_contract,
    variance_weighted_r2,
    verify_sidecar,
)
from src.h1_m4_cce_contract import (
    CONFIRMATORY_DATES,
    NORMALIZER_FLOOR,
    NORMALIZER_FORMULA,
    SUPPORT_TRIALS,
    WINDOW_SIZE,
    canonical_sha256,
    reject_nonpublic_heldin_scope,
    sha256_file,
    state_hash,
)


SCHEMA = "h1_causal_output_ema_v1"
STATUS_SMOKE = "PASS_H1_CAUSAL_OUTPUT_EMA_V1_GPU_SMOKE"
STATUS_CELL = "PASS_H1_CAUSAL_OUTPUT_EMA_V1_DATE_EVALUATED"
STATUS_PASS = "COMPLETE_H1_CAUSAL_OUTPUT_EMA_V1_PASS_TRANSFER"
STATUS_NO_TRANSFER = "COMPLETE_H1_CAUSAL_OUTPUT_EMA_V1_NO_TRANSFER"
ALPHAS = (0.0, 0.3, 0.5, 0.7, 0.9)
GOVERNING_ALPHA = 0.7
MIN_EQUAL_DATE_DELTA = 0.01
MIN_POSITIVE_DATES = 4
BATCH_SIZE = 32
MODEL_PARAMETERS = 10_947_836

PREDECESSOR_TERMINAL_SHA256 = "470634334480c33fd8d4679baa470454984dadeb57a25b9dec8d05612c43b9ec"
PREDECESSOR_SOURCE_SHA256 = "6cf656048af20174c3cf164406e25051137bfe687b0dda93d06ff1835c80500e"
PREDECESSOR_CLOSURE_SHA256 = "0727459079dd401f896bc714e85da46c5e5d08820aea3932a507d04cc85a4a74"
PREDECESSOR_ATTEMPT_SHA256 = "e0d39b534cb0800f1ca027ce1893c7f023648391380b94d723f531b1b9fa67b4"
PREDECESSOR_CONFIG_SHA256 = "d9fd74381bbc9769a116f47657f5762225ef56258714fe9de20f0498e0496f4f"
PREDECESSOR_INITIAL_STATE_SHA256 = "bc6dc8a0543c760811f770206c7ee22ae35eaf970c6dad0ec259a84172e4d04b"
PREDECESSOR = {
    "19250108": {
        "terminal_sha256": "6ec102fd871b76a0c37978f7694d6a6c7b4bd55a6df611a8f8445538ebad5fb3",
        "checkpoint_sha256": "d08e9488757a5e9672c2bbafaac218bf2043e41f17b8ffe02ceae9c28b65d4de",
        "source_authority_sha256": "c740fcf56980b4d308d2030ad0b606a7a8d282abee70aaca4f56e17eaf7edb93",
        "global_step": 167800,
        "terminal_state_sha256": "06ae7856c71a6586685b508d9bcc8abf084472f380aea3454cbedd9e012509f9",
    },
    "19250113": {
        "terminal_sha256": "e509abc4ce54bd84673e35143607c76303e19a0e999c577e2b3bee28c3243a8a",
        "checkpoint_sha256": "76e275b4f79a90e8223d444e610a7780dec1204574787b096f524e1466a4910c",
        "source_authority_sha256": "6e12eb63a93c4de9e548548d01d2ff85f26900bda7dee63f6fea4188680f2b7d",
        "global_step": 171100,
        "terminal_state_sha256": "03dd75975edabf8b064cd0496d872b401e20748c03de7c63862d19c98b3bafed",
    },
    "19250115": {
        "terminal_sha256": "2e6fc4ead6351c708363c61395a475bcfb854941a72f32bcd363e1f1da74dfac",
        "checkpoint_sha256": "470fa3fc3e023fd98fb0ac52a304bb75e167b2e15caa5afae5e94e6111da0249",
        "source_authority_sha256": "5949d3fb1c10880ca1db69a983f332afd0f1fc340531d5836ae22f566a3e27b5",
        "global_step": 172700,
        "terminal_state_sha256": "3cb305725b7ac7abe947eba1d1207f3285480071ed9368b6b69a0271ec6d7af3",
    },
    "19250119": {
        "terminal_sha256": "4c78f0e7ba1027f1b4965ef53c885548fe3e15ae4d57d417ca6947fd5e445b75",
        "checkpoint_sha256": "3cc84b8f956fe6d7e2b581ec2fc73aa7099bf1aac4d20fd63ee2f73cc20ac637",
        "source_authority_sha256": "fba7d5054104aded142875cacda1cf2636b918dc9c340edd099cf7b3cae6cde4",
        "global_step": 170200,
        "terminal_state_sha256": "446a8a21dc2a7b85c7afe01b61a87e0abe28cd97c65cb0cd65b613d9f3f4468e",
    },
    "19250120": {
        "terminal_sha256": "49d536369ae892670f317fcd33ec7899007a44d63e50c8d0ea16bddb6e63044d",
        "checkpoint_sha256": "405ae97bf31513466f6be8583f07ac2ca354ba1fbf256e2bec97728675386af4",
        "source_authority_sha256": "fc4d6a6ae756aea9247018bc1d9368a0cc28d4d062eaa224aadc16039fab06f8",
        "global_step": 170950,
        "terminal_state_sha256": "f8e455cfdf217ea34e37fc0ed309a44da500d142f148f198a2a8706bf7f8528c",
    },
}


class EmaExperimentError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise EmaExperimentError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _alpha_key(alpha: float) -> str:
    return f"{float(alpha):.1f}"


def dry_plan() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "DRY_NO_WRITE_NO_DATA_NO_CUDA",
        "date_order": list(CONFIRMATORY_DATES),
        "alphas": list(ALPHAS),
        "governing_alpha": GOVERNING_ALPHA,
        "target_access": 0,
        "cuda_queries": 0,
        "writes": 0,
    }


def create_attempt(result_root: Path, closure: Mapping[str, str], head: str) -> dict[str, Any]:
    root = result_root.resolve()
    _need(not root.exists(), f"canonical result root is not fresh: {root}")
    body = {
        "schema": SCHEMA,
        "artifact": "experiment_attempt",
        "status": "ATTEMPT_PUBLISHED_BEFORE_CUDA_OR_TARGET_INDEX",
        "created_at_utc": utc_now(),
        "head": str(head),
        "closure": dict(closure),
        "code_closure_sha256": canonical_sha256(closure),
        "date_order": list(CONFIRMATORY_DATES),
        "alphas": list(ALPHAS),
        "governing_alpha": GOVERNING_ALPHA,
        "ema_formula": "s[0]=prediction[0]; s[t]=alpha*s[t-1]+(1-alpha)*prediction[t]",
        "reset": "recording_boundary_only",
        "target_recordings_opened": 0,
        "target_bytes_read": 0,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
    }
    publish_json(root / "attempt.json", body)
    return body


def load_attempt(result_root: Path) -> dict[str, Any]:
    path = result_root.resolve() / "attempt.json"
    verify_sidecar(path)
    body = json.loads(path.read_text(encoding="utf-8"))
    _need(body.get("schema") == SCHEMA, "experiment attempt schema drift")
    _need(body.get("date_order") == list(CONFIRMATORY_DATES), "attempt date order drift")
    _need(body.get("alphas") == list(ALPHAS), "attempt alpha grid drift")
    _need(body.get("target_recordings_opened") == 0 and body.get("target_bytes_read") == 0, "attempt records target access")
    return body


def causal_ema(prediction: np.ndarray, alpha: float) -> np.ndarray:
    values = np.asarray(prediction)
    _need(values.ndim == 2 and values.shape[0] > 0 and values.shape[1] == 7, "EMA prediction array is malformed")
    _need(float(alpha) in ALPHAS, "EMA alpha is not pre-registered")
    _need(np.isfinite(values).all(), "EMA prediction contains nonfinite values")
    source = np.asarray(values, dtype=np.float64)
    output = np.empty_like(source, dtype=np.float64)
    output[0] = source[0]
    for index in range(1, len(source)):
        output[index] = float(alpha) * output[index - 1] + (1.0 - float(alpha)) * source[index]
    _need(np.isfinite(output).all(), "EMA output contains nonfinite values")
    if float(alpha) == 0.0:
        _need(np.array_equal(output, source), "alpha zero is not raw-prediction identity")
    return output


def score_cache(cache: Mapping[str, np.ndarray], sessions: Sequence[str]) -> dict[str, Any]:
    per_recording: dict[str, Any] = {}
    pooled_truth: list[np.ndarray] = []
    pooled_by_alpha: dict[str, list[np.ndarray]] = {_alpha_key(alpha): [] for alpha in ALPHAS}
    for index, session in enumerate(sessions):
        raw = np.asarray(cache[f"raw_{index}"], dtype=np.float32)
        target = np.asarray(cache[f"target_{index}"], dtype=np.float32)
        score_mask = np.asarray(cache[f"score_mask_{index}"], dtype=bool)
        bins = np.asarray(cache[f"output_bins_{index}"], dtype=np.int64)
        _need(raw.shape == target.shape and raw.ndim == 2 and raw.shape[1] == 7, f"{session}: cached prediction/target shape drift")
        _need(score_mask.shape == (len(raw),) and bins.shape == (len(raw),), f"{session}: cached mask/bin shape drift")
        _need(len(raw) > 1 and int(score_mask.sum()) > 1 and np.all(np.diff(bins) == 1), f"{session}: noncontinuous or empty scoring cache")
        _need(np.isfinite(raw).all() and np.isfinite(target[score_mask]).all(), f"{session}: nonfinite cache")
        truth = np.asarray(target[score_mask], dtype=np.float64)
        pooled_truth.append(truth)
        alpha_r2: dict[str, float] = {}
        for alpha in ALPHAS:
            key = _alpha_key(alpha)
            filtered = causal_ema(raw, alpha)
            scored = filtered[score_mask]
            alpha_r2[key] = variance_weighted_r2(truth, scored)
            pooled_by_alpha[key].append(scored)
        raw_r2 = alpha_r2[_alpha_key(0.0)]
        per_recording[str(session)] = {
            "continuous_outputs": int(len(raw)),
            "scored_outputs": int(score_mask.sum()),
            "first_output_bin": int(bins[0]),
            "last_output_bin": int(bins[-1]),
            "r2_by_alpha": alpha_r2,
            "delta_r2_by_alpha": {key: float(value - raw_r2) for key, value in alpha_r2.items()},
            "raw_prediction_sha256": array_sha256(raw),
            "target_output_sha256": array_sha256(target),
            "score_mask_sha256": array_sha256(score_mask),
            "output_bins_sha256": array_sha256(bins),
        }
    alpha_metrics: dict[str, Any] = {}
    for alpha in ALPHAS:
        key = _alpha_key(alpha)
        recording_scores = [float(per_recording[session]["r2_by_alpha"][key]) for session in sessions]
        alpha_metrics[key] = {
            "alpha": float(alpha),
            "equal_recording_mean_r2": float(np.mean(recording_scores, dtype=np.float64)),
            "pooled_r2": variance_weighted_r2(np.concatenate(pooled_truth), np.concatenate(pooled_by_alpha[key])),
        }
    raw_mean = float(alpha_metrics[_alpha_key(0.0)]["equal_recording_mean_r2"])
    for row in alpha_metrics.values():
        row["delta_equal_recording_mean_r2_vs_raw"] = float(row["equal_recording_mean_r2"] - raw_mean)
    return {
        "schema": f"{SCHEMA}_metrics",
        "alpha_order": list(ALPHAS),
        "governing_alpha": GOVERNING_ALPHA,
        "per_recording": per_recording,
        "alpha_metrics": alpha_metrics,
        "selection_performed": False,
        "float64_ema_and_r2": True,
    }


def transfer_decision(date_metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _need(len(date_metrics) == len(CONFIRMATORY_DATES), "decision requires all five dates")
    dates = tuple(str(row.get("outer_date")) for row in date_metrics)
    _need(dates == CONFIRMATORY_DATES, "decision date order drift")
    key = _alpha_key(GOVERNING_ALPHA)
    deltas = [float(row["metrics"]["alpha_metrics"][key]["delta_equal_recording_mean_r2_vs_raw"]) for row in date_metrics]
    _need(all(math.isfinite(value) for value in deltas), "decision contains nonfinite delta")
    equal_date_delta = float(np.mean(deltas, dtype=np.float64))
    positive_dates = int(sum(value > 0.0 for value in deltas))
    passed = equal_date_delta >= MIN_EQUAL_DATE_DELTA and positive_dates >= MIN_POSITIVE_DATES
    return {
        "verdict": "PASS_TRANSFER" if passed else "COMPLETE_NO_TRANSFER",
        "governing_alpha": GOVERNING_ALPHA,
        "equal_date_mean_delta_r2": equal_date_delta,
        "positive_dates": positive_dates,
        "required_equal_date_mean_delta_r2": MIN_EQUAL_DATE_DELTA,
        "required_positive_dates": MIN_POSITIVE_DATES,
        "date_delta_r2": {date: value for date, value in zip(dates, deltas, strict=True)},
    }


def _load_json(path: Path, schema: str | None = None) -> tuple[dict[str, Any], str]:
    digest = verify_sidecar(path)
    body = json.loads(path.read_text(encoding="utf-8"))
    _need(isinstance(body, dict), f"JSON artifact is not an object: {path}")
    if schema is not None:
        _need(body.get("schema") == schema, f"schema drift: {path}")
    return body, digest


@dataclass(frozen=True)
class DateAuthority:
    outer_date: str
    terminal: dict[str, Any]
    terminal_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    config: dict[str, Any]
    config_sha256: str
    source_authority: dict[str, Any]
    source_authority_sha256: str
    plan: RegenPlan
    plan_sha256: str
    normalizer_s_src: float
    normalizer_sha256: str


def _load_date_authority(predecessor_root: Path, outer_date: str, *, validate_state: bool) -> DateAuthority:
    root = predecessor_root.resolve()
    expected = PREDECESSOR[outer_date]
    cell_dir = root / "cells" / outer_date
    terminal, terminal_sha = _load_json(cell_dir / "terminal.json")
    _need(terminal_sha == expected["terminal_sha256"], f"predecessor cell terminal SHA drift: {outer_date}")
    _need(terminal.get("status") == "PASS_H1_HC_DATE_LODO_REGEN_V1_SOURCE_CELL", f"predecessor cell status drift: {outer_date}")
    _need(terminal.get("outer_date") == outer_date and terminal.get("global_step") == expected["global_step"], f"predecessor cell date/step drift: {outer_date}")
    _need(terminal.get("initial_state_sha256") == PREDECESSOR_INITIAL_STATE_SHA256, f"predecessor initial state drift: {outer_date}")
    _need(terminal.get("terminal_state_sha256") == expected["terminal_state_sha256"], f"predecessor terminal state drift: {outer_date}")
    _need(terminal.get("source_authority_sha256") == expected["source_authority_sha256"], f"predecessor source SHA drift: {outer_date}")
    checkpoint_path = root / str(terminal.get("checkpoint_relative", ""))
    checkpoint_sha = verify_sidecar(checkpoint_path)
    _need(checkpoint_sha == terminal.get("checkpoint_sha256") == expected["checkpoint_sha256"], f"predecessor checkpoint SHA drift: {outer_date}")
    config, config_sha = _load_json(cell_dir / "config.json", "h1_hc_date_lodo_regen_v1_resolved_config")
    _need(config_sha == PREDECESSOR_CONFIG_SHA256 == terminal.get("config_sha256"), f"predecessor config SHA drift: {outer_date}")
    _need(config == model_config(), f"predecessor config content drift: {outer_date}")
    source_dir = root / "source_authority" / outer_date
    source, source_sha = _load_json(source_dir / "authority.json", "h1_hc_date_lodo_regen_v1_date_source_authority")
    _need(source_sha == expected["source_authority_sha256"], f"predecessor date source authority drift: {outer_date}")
    _need(source.get("outer_date") == outer_date and source.get("target_recordings_opened") == 0 and source.get("target_bytes_read") == 0, f"predecessor source access drift: {outer_date}")
    _need(tuple(source.get("target_sessions_indexed", ())) == target_sessions_for_date(outer_date), f"predecessor target roster drift: {outer_date}")
    plan_body, plan_sha = _load_json(source_dir / "plan.json", "h1_hc_date_lodo_regen_v1_plan")
    _need(plan_sha == source.get("plan_sha256"), f"predecessor plan receipt drift: {outer_date}")
    plan_arrays_path = source_dir / "plan.npz"
    _need(verify_sidecar(plan_arrays_path) == plan_body.get("arrays_file_sha256"), f"predecessor plan array file drift: {outer_date}")
    with np.load(plan_arrays_path, allow_pickle=False) as values:
        _need(set(values.files) == {"mean", "scale", "pcs", "q", "lambda", "U", "mu", "tau2"}, f"predecessor plan array schema drift: {outer_date}")
        arrays = {name: np.asarray(values[name], dtype=np.float64) for name in ("mean", "scale", "pcs", "U", "mu")}
        for name, value in arrays.items():
            _need(array_sha256(value) == plan_body["array_sha256"][name], f"predecessor plan array SHA drift: {outer_date}:{name}")
        plan = RegenPlan(
            outer_date=outer_date,
            source_sessions=tuple(str(value) for value in plan_body["source_sessions"]),
            source_input_sha256=tuple(str(value) for value in plan_body["source_input_sha256"]),
            mean=arrays["mean"],
            scale=arrays["scale"],
            pcs=arrays["pcs"],
            q=int(values["q"].item()),
            ridge_lambda=float(values["lambda"].item()),
            U=arrays["U"],
            mu=arrays["mu"],
            tau2=float(values["tau2"].item()),
            selection_sha256=str(plan_body["selection_sha256"]),
            transform_sha256=str(plan_body["transform_sha256"]),
        )
    _need(plan.q == int(plan_body["q"]) and plan.ridge_lambda == float(plan_body["lambda"]), f"predecessor plan scalar drift: {outer_date}")
    _need(plan.mean.shape == plan.scale.shape == (EXPECTED_NEURONS,) and plan.U.shape == (7, 4) and plan.mu.shape == (4,), f"predecessor plan shape drift: {outer_date}")
    normalizer, normalizer_sha = _load_json(source_dir / "normalizer.json", "h1_hc_date_lodo_regen_v1_normalizer")
    _need(normalizer_sha == source.get("normalizer_sha256"), f"predecessor normalizer SHA drift: {outer_date}")
    _need(normalizer.get("formula") == NORMALIZER_FORMULA and float(normalizer.get("floor")) == NORMALIZER_FLOOR, f"predecessor normalizer contract drift: {outer_date}")
    s_src = float(normalizer.get("s_src"))
    _need(math.isfinite(s_src) and s_src >= 0.0, f"predecessor normalizer value drift: {outer_date}")
    if validate_state:
        import torch

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata") if isinstance(payload, Mapping) else None
        _need(isinstance(metadata, Mapping) and dict(metadata) == terminal.get("checkpoint_metadata"), f"predecessor embedded metadata drift: {outer_date}")
        _need(metadata.get("schema") == CHECKPOINT_SCHEMA, f"predecessor checkpoint schema drift: {outer_date}")
        validate_checkpoint_contract(
            metadata,
            terminal,
            outer_date,
            PREDECESSOR_ATTEMPT_SHA256,
            PREDECESSOR_CLOSURE_SHA256,
            expected["source_authority_sha256"],
        )
        _need(state_hash(payload["state_dict"]) == expected["terminal_state_sha256"], f"predecessor checkpoint state drift: {outer_date}")
    return DateAuthority(
        outer_date,
        terminal,
        terminal_sha,
        checkpoint_path,
        checkpoint_sha,
        config,
        config_sha,
        source,
        source_sha,
        plan,
        plan_sha,
        s_src,
        normalizer_sha,
    )


def validate_predecessor(predecessor_root: Path) -> dict[str, Any]:
    root = predecessor_root.resolve()
    terminal, terminal_sha = _load_json(root / "terminal.json")
    _need(terminal_sha == PREDECESSOR_TERMINAL_SHA256, "predecessor top terminal SHA drift")
    _need(terminal.get("status") == "COMPLETE_H1_HC_DATE_LODO_REGEN_V1_SOURCE_ONLY_AUTHORITY", "predecessor top terminal status drift")
    _need(terminal.get("date_order") == list(CONFIRMATORY_DATES), "predecessor top date order drift")
    _need(terminal.get("source_authority_sha256") == PREDECESSOR_SOURCE_SHA256, "predecessor top source SHA drift")
    _need(terminal.get("experiment_attempt_sha256") == PREDECESSOR_ATTEMPT_SHA256, "predecessor top attempt SHA drift")
    _need(terminal.get("code_closure_sha256") == PREDECESSOR_CLOSURE_SHA256, "predecessor top closure SHA drift")
    _need(terminal.get("canonical_initial_state_sha256") == PREDECESSOR_INITIAL_STATE_SHA256, "predecessor canonical initial state drift")
    source, source_sha = _load_json(root / "source_authority.json", "h1_hc_date_lodo_regen_v1_source_authority")
    _need(source_sha == PREDECESSOR_SOURCE_SHA256, "predecessor source authority file SHA drift")
    attempt_sha = verify_sidecar(root / "attempt.json")
    _need(attempt_sha == PREDECESSOR_ATTEMPT_SHA256, "predecessor attempt file SHA drift")
    top_rows = {str(row.get("outer_date")): row for row in terminal.get("cells", ())}
    _need(tuple(top_rows) == CONFIRMATORY_DATES, "predecessor top cell set/order drift")
    rows = []
    for date in CONFIRMATORY_DATES:
        authority = _load_date_authority(root, date, validate_state=True)
        expected = PREDECESSOR[date]
        top = top_rows[date]
        for field in ("terminal_sha256", "checkpoint_sha256", "source_authority_sha256", "global_step"):
            _need(top.get(field) == expected[field], f"predecessor top cell {field} drift: {date}")
        rows.append(
            {
                "outer_date": date,
                "cell_terminal_sha256": authority.terminal_sha256,
                "checkpoint_sha256": authority.checkpoint_sha256,
                "config_sha256": authority.config_sha256,
                "source_authority_sha256": authority.source_authority_sha256,
                "plan_sha256": authority.plan_sha256,
                "normalizer_sha256": authority.normalizer_sha256,
                "global_step": expected["global_step"],
                "terminal_state_sha256": expected["terminal_state_sha256"],
            }
        )
    return {
        "schema": f"{SCHEMA}_predecessor_authority",
        "status": "PASS_EXACT_FIVE_DATE_REGENERATION_PREDECESSOR",
        "predecessor_root": str(root),
        "terminal_sha256": terminal_sha,
        "source_authority_sha256": source_sha,
        "attempt_sha256": attempt_sha,
        "code_closure_sha256": PREDECESSOR_CLOSURE_SHA256,
        "canonical_initial_state_sha256": PREDECESSOR_INITIAL_STATE_SHA256,
        "date_order": list(CONFIRMATORY_DATES),
        "cells": rows,
        "target_recordings_opened": 0,
        "target_bytes_read": 0,
    }


def publish_predecessor_authority(predecessor_root: Path, result_root: Path) -> dict[str, Any]:
    body = validate_predecessor(predecessor_root)
    publish_json(result_root.resolve() / "predecessor_authority.json", body)
    return body


def _require_predecessor_receipt(predecessor_root: Path, result_root: Path) -> dict[str, Any]:
    body, _ = _load_json(result_root.resolve() / "predecessor_authority.json", f"{SCHEMA}_predecessor_authority")
    _need(body.get("status") == "PASS_EXACT_FIVE_DATE_REGENERATION_PREDECESSOR", "predecessor receipt status drift")
    _need(body.get("predecessor_root") == str(predecessor_root.resolve()), "predecessor receipt root drift")
    _need(body.get("terminal_sha256") == PREDECESSOR_TERMINAL_SHA256, "predecessor receipt terminal drift")
    return body


def _target_records(
    data_root: Path,
    outer_date: str,
    access: dict[str, Any],
    *,
    record_loader=load_record,
) -> dict[str, H1PilotRecord]:
    reject_nonpublic_heldin_scope(data_root)
    paths = index_heldin_calib(data_root)
    expected = target_sessions_for_date(outer_date)
    records: dict[str, H1PilotRecord] = {}
    files: list[dict[str, Any]] = []
    for session in expected:
        path = paths[session]
        size = int(path.stat().st_size)
        access["target_recordings_opened"] += 1
        access["target_bytes_read"] += size
        access["target_sessions_opened"].append(session)
        record = record_loader(path)
        _need(record.session_name == session and record.date == outer_date, f"outer-date target loader partition drift: {session}")
        records[session] = record
        files.append({"session": session, "filename": path.name, "bytes": size, "sha256": record.input_sha256})
    _need(tuple(records) == expected, "outer-date target order drift")
    access["files"] = files
    return records


def _support_boundary(record: H1PilotRecord) -> tuple[tuple[float, ...], int]:
    _need(len(record.trial_values) >= SUPPORT_TRIALS + 1, f"{record.session_name}: fewer than five trials")
    support = tuple(float(value) for value in record.trial_values[:SUPPORT_TRIALS])
    _need(all(record.blocks_for(value).rates.shape[0] >= 2 for value in support), f"{record.session_name}: illegal M4 support")
    fifth = float(record.trial_values[SUPPORT_TRIALS])
    indices = np.flatnonzero(record.eval_mask & np.isfinite(record.trial_num) & (record.trial_num == fifth))
    _need(indices.size > 0, f"{record.session_name}: fifth trial has no eval-valid boundary")
    return support, int(indices[0])


def _load_model(authority: DateAuthority, device: str):
    import torch
    from src.models.components.h1_carrierid_spint import H1CarrierIdSpint

    payload = torch.load(authority.checkpoint_path, map_location="cpu", weights_only=False)
    _need(isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping), "predecessor checkpoint is malformed")
    _need(payload.get("metadata") == authority.terminal.get("checkpoint_metadata"), "predecessor checkpoint embedded provenance drift")
    model = H1CarrierIdSpint(**authority.config["model_kwargs"])
    _need(sum(parameter.numel() for parameter in model.parameters()) == MODEL_PARAMETERS, "H-C parameter count drift")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(torch.device(device))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = state_hash(model.state_dict())
    _need(before == PREDECESSOR[authority.outer_date]["terminal_state_sha256"], "loaded model state differs from predecessor")
    return model, before


def _gpu_profile(physical_gpu: int) -> dict[str, Any]:
    import torch

    properties = torch.cuda.get_device_properties(0)
    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", "-i", str(int(physical_gpu)), "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    return {
        "physical_index": int(physical_gpu),
        "visible_index": 0,
        "uuid": gpu_uuid,
        "name": properties.name,
        "memory_total_bytes": int(properties.total_memory),
    }


def run_smoke_cell(predecessor_root: Path, result_root: Path, physical_gpu: int) -> dict[str, Any]:
    directory = result_root.resolve() / "smoke"
    _need(not directory.exists(), f"smoke directory exists: {directory}")
    publish_json(
        directory / "attempt.json",
        {
            "schema": SCHEMA,
            "artifact": "smoke_attempt",
            "created_at_utc": utc_now(),
            "physical_gpu": int(physical_gpu),
            "target_recordings_opened": 0,
            "target_bytes_read": 0,
        },
    )
    started = time.monotonic()
    try:
        import torch

        _need(torch.cuda.is_available(), "GPU smoke requires CUDA")
        _require_predecessor_receipt(predecessor_root, result_root)
        authority = _load_date_authority(predecessor_root, CONFIRMATORY_DATES[0], validate_state=True)
        model, before = _load_model(authority, "cuda:0")
        with torch.no_grad():
            neural = torch.zeros(1, WINDOW_SIZE, EXPECTED_NEURONS, dtype=torch.float32, device="cuda:0")
            identity = torch.zeros(1, SUPPORT_TRIALS, 1024, EXPECTED_NEURONS, dtype=torch.float32, device="cuda:0")
            carrier = torch.zeros(1, EXPECTED_NEURONS, 4, dtype=torch.float32, device="cuda:0")
            output = model(neural, calib_trialized_neural_features=identity, carrier=carrier)
            prediction = np.asarray((output[:, -1, :] / 20.0).cpu(), dtype=np.float32)
        _need(prediction.shape == (1, 7) and np.isfinite(prediction).all(), "GPU smoke prediction is nonfinite")
        after = state_hash(model.state_dict())
        _need(before == after, "model state changed during GPU smoke")
        body = {
            "schema": SCHEMA,
            "status": STATUS_SMOKE,
            "finished_at_utc": utc_now(),
            "outer_date": CONFIRMATORY_DATES[0],
            "gpu": _gpu_profile(physical_gpu),
            "checkpoint_sha256": authority.checkpoint_sha256,
            "model_state_before_sha256": before,
            "model_state_after_sha256": after,
            "prediction_finite": True,
            "prediction_shape": list(prediction.shape),
            "optimizer_steps": 0,
            "backward_steps": 0,
            "target_recordings_opened": 0,
            "target_bytes_read": 0,
            "elapsed_seconds": time.monotonic() - started,
        }
        publish_json(directory / "terminal.json", body)
        return body
    except BaseException as error:
        try:
            publish_json(
                directory / "failure.json",
                {
                    "schema": SCHEMA,
                    "status": "FAIL_GPU_SMOKE_NO_AUTOMATIC_RETRY",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "target_recordings_opened": 0,
                    "target_bytes_read": 0,
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
        except BaseException:
            pass
        raise


def _infer_recording(model: Any, record: H1PilotRecord, authority: DateAuthority, *, device: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch

    support, boundary = _support_boundary(record)
    last_start = int(record.neural.shape[0] - WINDOW_SIZE)
    _need(last_start >= boundary, f"{record.session_name}: no complete post-M4 window")
    starts = np.arange(boundary, last_start + 1, dtype=np.int64)
    output_bins = starts + WINDOW_SIZE - 1
    score_mask = np.asarray(record.eval_mask[output_bins], dtype=bool)
    target = np.asarray(record.velocity[output_bins], dtype=np.float32)
    _need(int(score_mask.sum()) > 1 and np.isfinite(target[score_mask]).all(), f"{record.session_name}: invalid scoring targets")
    identity = np.ascontiguousarray(
        np.stack([interpolate_trial_identity(record, value) for value in support]),
        dtype=np.float32,
    )
    denominator = max(authority.normalizer_s_src, NORMALIZER_FLOOR)
    carrier = np.ascontiguousarray(fit_frozen_carrier(record, authority.plan, support)["carrier"] / denominator, dtype=np.float32)
    _need(identity.shape == (SUPPORT_TRIALS, 1024, EXPECTED_NEURONS), f"{record.session_name}: identity shape drift")
    _need(carrier.shape == (EXPECTED_NEURONS, 4) and np.isfinite(carrier).all(), f"{record.session_name}: carrier shape/finite drift")
    prediction = np.empty((len(starts), 7), dtype=np.float32)
    identity_one = torch.as_tensor(identity, dtype=torch.float32, device=device).unsqueeze(0)
    carrier_one = torch.as_tensor(carrier, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        for offset in range(0, len(starts), BATCH_SIZE):
            batch_starts = starts[offset : offset + BATCH_SIZE]
            neural = np.ascontiguousarray(np.stack([record.neural[int(start) : int(start) + WINDOW_SIZE] for start in batch_starts]), dtype=np.float32)
            neural_t = torch.as_tensor(neural, dtype=torch.float32, device=device)
            count = len(batch_starts)
            output = model(
                neural_t,
                calib_trialized_neural_features=identity_one.expand(count, -1, -1, -1),
                carrier=carrier_one.expand(count, -1, -1),
            )
            values = np.asarray((output[:, -1, :] / 20.0).cpu(), dtype=np.float32)
            _need(values.shape == (count, 7) and np.isfinite(values).all(), f"{record.session_name}: nonfinite model output")
            prediction[offset : offset + count] = values
    cache = {
        "raw": prediction,
        "target": target,
        "score_mask": score_mask,
        "output_bins": output_bins,
    }
    support_row = {
        "support_trials": list(support),
        "fifth_trial": float(record.trial_values[SUPPORT_TRIALS]),
        "support_boundary_bin": boundary,
        "first_output_bin": int(output_bins[0]),
        "last_output_bin": int(output_bins[-1]),
        "continuous_outputs": int(len(output_bins)),
        "scored_outputs": int(score_mask.sum()),
        "identity_sha256": array_sha256(identity),
        "normalized_carrier_sha256": array_sha256(carrier),
    }
    return cache, support_row


def run_evaluation_cell(data_root: Path, predecessor_root: Path, result_root: Path, outer_date: str, physical_gpu: int) -> dict[str, Any]:
    _need(outer_date in CONFIRMATORY_DATES, "evaluation date is not canonical")
    directory = result_root.resolve() / "cells" / outer_date
    _need(not directory.exists(), f"evaluation cell directory exists: {directory}")
    publish_json(
        directory / "attempt.json",
        {
            "schema": SCHEMA,
            "artifact": "date_evaluation_attempt",
            "outer_date": outer_date,
            "physical_gpu": int(physical_gpu),
            "created_at_utc": utc_now(),
            "target_recordings_opened": 0,
            "target_bytes_read": 0,
            "optimizer_steps": 0,
            "backward_steps": 0,
        },
    )
    access: dict[str, Any] = {
        "outer_date": outer_date,
        "target_recordings_opened": 0,
        "target_bytes_read": 0,
        "target_sessions_opened": [],
        "files": [],
    }
    started = time.monotonic()
    try:
        import torch

        _need(torch.cuda.is_available(), "evaluation cell requires CUDA")
        _require_predecessor_receipt(predecessor_root, result_root)
        authority = _load_date_authority(predecessor_root, outer_date, validate_state=True)
        model, before = _load_model(authority, "cuda:0")
        gpu = _gpu_profile(physical_gpu)
        records = _target_records(data_root, outer_date, access)
        sessions = tuple(records)
        arrays: dict[str, np.ndarray] = {}
        support: dict[str, Any] = {}
        for index, session in enumerate(sessions):
            cache, support_row = _infer_recording(model, records[session], authority, device="cuda:0")
            arrays[f"raw_{index}"] = cache["raw"]
            arrays[f"target_{index}"] = cache["target"]
            arrays[f"score_mask_{index}"] = cache["score_mask"]
            arrays[f"output_bins_{index}"] = cache["output_bins"]
            support[session] = support_row
        cache_path = directory / "prediction_cache.npz"
        cache_sha = publish_npz(cache_path, **arrays)
        cache_body = {
            "schema": f"{SCHEMA}_prediction_cache",
            "outer_date": outer_date,
            "sessions": list(sessions),
            "arrays_file": cache_path.name,
            "arrays_file_sha256": cache_sha,
            "array_sha256": {name: array_sha256(value) for name, value in arrays.items()},
            "array_shape": {name: list(value.shape) for name, value in arrays.items()},
            "support": support,
            "continuous_post_m4": True,
            "score_mask_applied_after_filtering": True,
            "reset": "recording_boundary_only",
        }
        cache_manifest_sha = publish_json(directory / "prediction_cache.json", cache_body)
        metrics = score_cache(arrays, sessions)
        metrics = {**metrics, "outer_date": outer_date, "sessions": list(sessions)}
        metrics_sha = publish_json(directory / "metrics.json", metrics)
        after = state_hash(model.state_dict())
        _need(before == after == PREDECESSOR[outer_date]["terminal_state_sha256"], "model state changed during target evaluation")
        audit = {
            **access,
            "authorized_outer_date_only": True,
            "expected_sessions": list(target_sessions_for_date(outer_date)),
            "optimizer_steps": 0,
            "backward_steps": 0,
            "model_updates": 0,
            "target_driven_selection": False,
        }
        audit_sha = publish_json(directory / "target_access.json", audit)
        body = {
            "schema": SCHEMA,
            "status": STATUS_CELL,
            "outer_date": outer_date,
            "finished_at_utc": utc_now(),
            "gpu": gpu,
            "predecessor_terminal_sha256": PREDECESSOR_TERMINAL_SHA256,
            "predecessor_cell_terminal_sha256": authority.terminal_sha256,
            "checkpoint_sha256": authority.checkpoint_sha256,
            "config_sha256": authority.config_sha256,
            "source_authority_sha256": authority.source_authority_sha256,
            "plan_sha256": authority.plan_sha256,
            "normalizer_sha256": authority.normalizer_sha256,
            "prediction_cache_sha256": cache_sha,
            "prediction_cache_manifest_sha256": cache_manifest_sha,
            "metrics_sha256": metrics_sha,
            "target_access_sha256": audit_sha,
            "target_access": audit,
            "metrics": metrics,
            "model_state_before_sha256": before,
            "model_state_after_sha256": after,
            "model_state_immutable": True,
            "optimizer_steps": 0,
            "backward_steps": 0,
            "model_updates": 0,
            "selection_performed": False,
            "elapsed_seconds": time.monotonic() - started,
        }
        publish_json(directory / "terminal.json", body)
        return body
    except BaseException as error:
        try:
            publish_json(
                directory / "failure.json",
                {
                    "schema": SCHEMA,
                    "status": "FAIL_DATE_EVALUATION_NO_AUTOMATIC_RETRY",
                    "outer_date": outer_date,
                    "physical_gpu": int(physical_gpu),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "target_access": access,
                    "optimizer_steps": 0,
                    "backward_steps": 0,
                    "elapsed_seconds": time.monotonic() - started,
                    "finished_at_utc": utc_now(),
                },
            )
        except BaseException:
            pass
        raise


def _load_cache(directory: Path, manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
    path = directory / str(manifest.get("arrays_file", ""))
    _need(verify_sidecar(path) == manifest.get("arrays_file_sha256"), "prediction cache file SHA drift")
    with np.load(path, allow_pickle=False) as values:
        arrays = {name: np.asarray(values[name]) for name in values.files}
    _need(set(arrays) == set(manifest.get("array_sha256", {})), "prediction cache array set drift")
    for name, value in arrays.items():
        _need(array_sha256(value) == manifest["array_sha256"][name], f"prediction cache tensor SHA drift: {name}")
        _need(list(value.shape) == manifest["array_shape"][name], f"prediction cache tensor shape drift: {name}")
    return arrays


def verify_terminal(predecessor_root: Path, result_root: Path) -> dict[str, Any]:
    root = result_root.resolve()
    attempt = load_attempt(root)
    attempt_sha = verify_sidecar(root / "attempt.json")
    predecessor = validate_predecessor(predecessor_root)
    predecessor_receipt, predecessor_receipt_sha = _load_json(root / "predecessor_authority.json", f"{SCHEMA}_predecessor_authority")
    _need(predecessor_receipt == predecessor, "predecessor authority receipt differs from live immutable authority")
    smoke, smoke_sha = _load_json(root / "smoke" / "terminal.json")
    _need(smoke.get("status") == STATUS_SMOKE and smoke.get("target_recordings_opened") == 0 and smoke.get("target_bytes_read") == 0, "GPU smoke terminal drift")
    rows: list[dict[str, Any]] = []
    total_recordings = 0
    total_bytes = 0
    for date in CONFIRMATORY_DATES:
        directory = root / "cells" / date
        terminal, terminal_sha = _load_json(directory / "terminal.json")
        _need(terminal.get("status") == STATUS_CELL and terminal.get("outer_date") == date, f"date terminal drift: {date}")
        _need(terminal.get("checkpoint_sha256") == PREDECESSOR[date]["checkpoint_sha256"], f"date checkpoint drift: {date}")
        _need(terminal.get("model_state_before_sha256") == terminal.get("model_state_after_sha256") == PREDECESSOR[date]["terminal_state_sha256"], f"date model state drift: {date}")
        _need(terminal.get("optimizer_steps") == 0 and terminal.get("backward_steps") == 0 and terminal.get("model_updates") == 0, f"date model update recorded: {date}")
        _need(terminal.get("selection_performed") is False, f"date target selection recorded: {date}")
        audit, audit_sha = _load_json(directory / "target_access.json")
        _need(audit_sha == terminal.get("target_access_sha256"), f"target access SHA drift: {date}")
        expected_sessions = target_sessions_for_date(date)
        _need(tuple(audit.get("target_sessions_opened", ())) == expected_sessions == tuple(audit.get("expected_sessions", ())), f"target date isolation drift: {date}")
        _need(audit.get("target_recordings_opened") == len(expected_sessions) and int(audit.get("target_bytes_read", 0)) > 0, f"target access count drift: {date}")
        _need(audit.get("optimizer_steps") == 0 and audit.get("backward_steps") == 0 and audit.get("target_driven_selection") is False, f"target audit adaptation drift: {date}")
        cache_manifest, cache_manifest_sha = _load_json(directory / "prediction_cache.json", f"{SCHEMA}_prediction_cache")
        _need(cache_manifest_sha == terminal.get("prediction_cache_manifest_sha256"), f"cache manifest SHA drift: {date}")
        arrays = _load_cache(directory, cache_manifest)
        recomputed = score_cache(arrays, expected_sessions)
        recomputed = {**recomputed, "outer_date": date, "sessions": list(expected_sessions)}
        metrics, metrics_sha = _load_json(directory / "metrics.json", f"{SCHEMA}_metrics")
        _need(metrics_sha == terminal.get("metrics_sha256") and metrics == recomputed == terminal.get("metrics"), f"metric recomputation drift: {date}")
        total_recordings += int(audit["target_recordings_opened"])
        total_bytes += int(audit["target_bytes_read"])
        rows.append(
            {
                "outer_date": date,
                "terminal_relative": str((directory / "terminal.json").relative_to(root)),
                "terminal_sha256": terminal_sha,
                "checkpoint_sha256": terminal["checkpoint_sha256"],
                "prediction_cache_sha256": terminal["prediction_cache_sha256"],
                "target_recordings_opened": audit["target_recordings_opened"],
                "target_bytes_read": audit["target_bytes_read"],
                "gpu": terminal["gpu"],
                "metrics": metrics,
            }
        )
    decision = transfer_decision(rows)
    status = STATUS_PASS if decision["verdict"] == "PASS_TRANSFER" else STATUS_NO_TRANSFER
    body = {
        "schema": SCHEMA,
        "status": status,
        "finished_at_utc": utc_now(),
        "date_order": list(CONFIRMATORY_DATES),
        "alpha_order": list(ALPHAS),
        "governing_alpha": GOVERNING_ALPHA,
        "decision": decision,
        "experiment_attempt_sha256": attempt_sha,
        "code_closure_sha256": attempt["code_closure_sha256"],
        "predecessor_authority_sha256": predecessor_receipt_sha,
        "predecessor_terminal_sha256": PREDECESSOR_TERMINAL_SHA256,
        "smoke_terminal_sha256": smoke_sha,
        "cells": rows,
        "target_recordings_opened": total_recordings,
        "target_bytes_read": total_bytes,
        "target_optimizer_steps": 0,
        "target_backward_steps": 0,
        "target_model_updates": 0,
        "target_driven_selection": False,
        "terminal_verifier_target_recordings_opened": 0,
        "claim": "fixed alpha=0.7 causal EMA transfer on five regenerated H-C checkpoints only",
    }
    terminal_sha = publish_json(root / "terminal.json", body)
    lines = [
        "# H1 five-date causal output EMA V1",
        "",
        f"- Status: `{status}`",
        f"- Governing verdict: `{decision['verdict']}`",
        f"- Equal-date mean delta R2 (alpha 0.7 - raw): `{decision['equal_date_mean_delta_r2']:.9f}`",
        f"- Positive dates: `{decision['positive_dates']}/5`",
        "- Model updates, optimizer steps, backward steps, and target-driven selection: `0`.",
        "",
        "| Outer date | Raw equal-recording R2 | Alpha 0.7 equal-recording R2 | Delta | GPU UUID |",
        "|---|---:|---:|---:|---|",
    ]
    key = _alpha_key(GOVERNING_ALPHA)
    for row in rows:
        raw = row["metrics"]["alpha_metrics"][_alpha_key(0.0)]["equal_recording_mean_r2"]
        filtered = row["metrics"]["alpha_metrics"][key]["equal_recording_mean_r2"]
        delta = row["metrics"]["alpha_metrics"][key]["delta_equal_recording_mean_r2_vs_raw"]
        lines.append(f"| {row['outer_date']} | {raw:.9f} | {filtered:.9f} | {delta:+.9f} | `{row['gpu']['uuid']}` |")
    lines.extend(["", f"Terminal SHA-256: `{terminal_sha}`", ""])
    publish_text(root / "EXPERIMENT_RECORD.md", "\n".join(lines))
    return body


__all__ = (
    "ALPHAS",
    "BATCH_SIZE",
    "EmaExperimentError",
    "GOVERNING_ALPHA",
    "MIN_EQUAL_DATE_DELTA",
    "MIN_POSITIVE_DATES",
    "PREDECESSOR",
    "PREDECESSOR_TERMINAL_SHA256",
    "SCHEMA",
    "STATUS_CELL",
    "STATUS_NO_TRANSFER",
    "STATUS_PASS",
    "STATUS_SMOKE",
    "causal_ema",
    "create_attempt",
    "dry_plan",
    "load_attempt",
    "publish_predecessor_authority",
    "run_evaluation_cell",
    "run_smoke_cell",
    "score_cache",
    "transfer_decision",
    "validate_predecessor",
    "verify_terminal",
)
