"""Pre-training metadata feasibility gate for H1 all-source held-out V1."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from src.data.h1_cal_aug_all_source_heldout_v1 import (
    H1_HELDOUT_SESSIONS,
    M4_MINIMUM_LEGAL_TRIALS,
    HeldoutTrialMetadata,
    audit_registered_heldout_metadata,
)
from src.h1_hc_date_lodo_regen_v1 import publish_json, publish_text, verify_sidecar
from src.h1_m4_cce_contract import canonical_sha256


SCHEMA = "h1_cal_aug_all_source_heldout_v1"
PASS_STATUS = "PASS_H1_ALL_SOURCE_HELDOUT_M4_METADATA_FEASIBILITY"
STOP_STATUS = "STOP_H1_ALL_SOURCE_HELDOUT_M4_PROTOCOL_INFEASIBLE"
PREDECESSOR_COMMIT = "c60052c9d8ccb8391d6ce53bde9ccfb4f2319884"
PREDECESSOR_TERMINAL_SHA256 = "dc9e7ab44954d3d193f67f9bf8936aafdaf2b05be9968d5e0091c0b0ecf092fd"


class AllSourceHeldoutError(RuntimeError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise AllSourceHeldoutError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path, schema: str | None = None) -> tuple[dict[str, Any], str]:
    digest = verify_sidecar(path)
    body = json.loads(path.read_text(encoding="utf-8"))
    if schema is not None:
        _need(body.get("schema") == schema, f"schema drift: {path}")
    return body, digest


def dry_plan() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "DRY_NO_WRITE_NO_NWB_NO_CUDA",
        "phase": "pre_training_heldout_metadata_feasibility_only",
        "registered_heldout_sessions": list(H1_HELDOUT_SESSIONS),
        "minimum_legal_trial_count": M4_MINIMUM_LEGAL_TRIALS,
        "permitted_nwb_fields": ["TrialNum", "eval_mask_or_Blacklist"],
        "neural_reads": 0,
        "behavior_reads": 0,
        "carrier_fits": 0,
        "model_inference": 0,
        "r2_calculations": 0,
        "gpu_training": 0,
    }


def create_attempt(result_root: Path, closure: Mapping[str, str], head: str) -> dict[str, Any]:
    root = result_root.resolve()
    _need(not root.exists(), f"canonical successor result root is not fresh: {root}")
    body = {
        "schema": SCHEMA,
        "artifact": "attempt",
        "status": "ATTEMPT_BEFORE_HELDOUT_METADATA_ACCESS_AND_GPU",
        "created_at_utc": utc_now(),
        "head": head,
        "closure": dict(closure),
        "code_closure_sha256": canonical_sha256(dict(closure)),
        "predecessor_commit": PREDECESSOR_COMMIT,
        "predecessor_terminal_sha256": PREDECESSOR_TERMINAL_SHA256,
        "registered_heldout_sessions": list(H1_HELDOUT_SESSIONS),
        "heldout_files_opened": 0,
        "heldout_neural_arrays_read": 0,
        "heldout_behavior_arrays_read": 0,
        "cuda_initialized": False,
    }
    publish_json(root / "attempt.json", body)
    return body


def load_attempt(result_root: Path) -> dict[str, Any]:
    body, _ = _load_json(result_root.resolve() / "attempt.json", SCHEMA)
    _need(body.get("status") == "ATTEMPT_BEFORE_HELDOUT_METADATA_ACCESS_AND_GPU", "attempt status drift")
    _need(body.get("heldout_files_opened") == 0, "attempt recorded pre-audit held-out access")
    _need(body.get("heldout_neural_arrays_read") == body.get("heldout_behavior_arrays_read") == 0,
          "attempt recorded forbidden held-out arrays")
    _need(body.get("cuda_initialized") is False, "attempt recorded CUDA initialization")
    return body


def validate_predecessor(predecessor_root: Path) -> dict[str, Any]:
    terminal_path = predecessor_root.resolve() / "terminal.json"
    _need(verify_sidecar(terminal_path) == PREDECESSOR_TERMINAL_SHA256, "Experiment-4 A1 terminal drift")
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    _need(terminal.get("status") == "PASS_H1_CAL_AUG_PREFIX_CYCLE_TRANSFER", "Experiment-4 predecessor did not pass")
    return {
        "schema": f"{SCHEMA}_predecessor_authority",
        "status": "PASS_H1_ALL_SOURCE_HELDOUT_V1_PREDECESSOR",
        "predecessor_commit": PREDECESSOR_COMMIT,
        "predecessor_terminal_sha256": PREDECESSOR_TERMINAL_SHA256,
        "heldout_metadata_accessed": False,
    }


def _row_payload(row: HeldoutTrialMetadata) -> dict[str, Any]:
    return {
        "session_name": row.session_name,
        "legal_trial_count": int(row.legal_trial_count),
        "m4_evaluable": bool(row.m4_evaluable),
    }


def run_metadata_feasibility_audit(data_root: Path, result_root: Path) -> dict[str, Any]:
    root = result_root.resolve()
    load_attempt(root)
    _load_json(root / "predecessor_authority.json", f"{SCHEMA}_predecessor_authority")
    rows = audit_registered_heldout_metadata(data_root)
    payload_rows = [_row_payload(row) for row in rows]
    evaluable = sum(int(row.m4_evaluable) for row in rows)
    passed = evaluable == len(H1_HELDOUT_SESSIONS) == 14
    status = PASS_STATUS if passed else STOP_STATUS
    validity_fields = {row.session_name: row.validity_field for row in rows}
    access = {
        "schema": f"{SCHEMA}_metadata_access",
        "status": "COMPLETE_METADATA_ONLY_ACCESS",
        "registered_sessions": list(H1_HELDOUT_SESSIONS),
        "heldout_files_opened": len(rows),
        "datasets_read_by_session": {
            session: ["TrialNum", field] for session, field in validity_fields.items()
        },
        "heldout_neural_arrays_read": 0,
        "heldout_behavior_arrays_read": 0,
        "heldout_velocity_arrays_read": 0,
        "model_inference_calls": 0,
        "prediction_rows": 0,
        "r2_calculations": 0,
        "loss_calculations": 0,
        "carrier_fits": 0,
        "plan_fits": 0,
        "normalizer_fits": 0,
        "gpu_training_steps": 0,
        "selection_performed": False,
        "trial_level_arrays_persisted": False,
    }
    access_sha = publish_json(root / "metadata_access.json", access)
    feasibility = {
        "schema": f"{SCHEMA}_metadata_feasibility",
        "status": status,
        "minimum_legal_trial_count": M4_MINIMUM_LEGAL_TRIALS,
        "registered_recordings": len(H1_HELDOUT_SESSIONS),
        "m4_evaluable_recordings": evaluable,
        "all_14_m4_evaluable": passed,
        "recordings": payload_rows,
        "metadata_access_sha256": access_sha,
        "continuation_to_training_allowed": passed,
        "automatic_m3_fallback": False,
        "model_or_budget_selection_performed": False,
        "finished_at_utc": utc_now(),
    }
    feasibility_sha = publish_json(root / "metadata_feasibility.json", feasibility)
    terminal = {
        "schema": f"{SCHEMA}_metadata_feasibility_terminal",
        "status": status,
        "metadata_feasibility_sha256": feasibility_sha,
        "metadata_access_sha256": access_sha,
        "registered_recordings": 14,
        "m4_evaluable_recordings": evaluable,
        "continuation_to_training_allowed": passed,
        "gpu_training_started": False,
        "prediction_performed": False,
        "r2_calculated": False,
        "carrier_fitted_on_heldout": False,
        "heldout_neural_arrays_read": 0,
        "heldout_behavior_arrays_read": 0,
        "automatic_budget_change": False,
        "next_action": "AWAIT_USER_AUTHORIZATION" if passed else "TERMINATE_CURRENT_M4_EXPERIMENT",
        "finished_at_utc": utc_now(),
    }
    terminal_sha = publish_json(root / "metadata_feasibility_terminal.json", terminal)
    lines = [
        "# H1 CAL-AUG All-Source Held-Out V1 — Metadata Feasibility Audit",
        "",
        f"- Status: `{status}`",
        f"- M4 evaluable: `{evaluable}/14`",
        "- Permitted datasets read: `TrialNum` and `eval_mask`/legacy `Blacklist` only.",
        "- Neural, behavior, prediction, R², carrier, plan, normalizer and GPU training activity: `0`.",
        "",
        "| Session | Legal TrialNum count | M4 evaluable |",
        "|---|---:|---|",
    ]
    for row in payload_rows:
        lines.append(f"| {row['session_name']} | {row['legal_trial_count']} | `{str(row['m4_evaluable']).lower()}` |")
    lines.extend(["", f"Metadata terminal SHA-256: `{terminal_sha}`", ""])
    publish_text(root / "METADATA_FEASIBILITY_RECORD.md", "\n".join(lines))
    if not passed:
        publish_json(root / "terminal.json", {
            **terminal,
            "schema": SCHEMA,
            "status": STOP_STATUS,
            "claim": "M4 protocol infeasible before training; no automatic M3 fallback",
        })
    return terminal


def verify_metadata_terminal(result_root: Path) -> dict[str, Any]:
    root = result_root.resolve()
    terminal, _ = _load_json(root / "metadata_feasibility_terminal.json", f"{SCHEMA}_metadata_feasibility_terminal")
    feasibility, feasibility_sha = _load_json(root / "metadata_feasibility.json", f"{SCHEMA}_metadata_feasibility")
    access, access_sha = _load_json(root / "metadata_access.json", f"{SCHEMA}_metadata_access")
    _need(terminal.get("metadata_feasibility_sha256") == feasibility_sha, "feasibility SHA binding drift")
    _need(terminal.get("metadata_access_sha256") == access_sha == feasibility.get("metadata_access_sha256"),
          "metadata access SHA binding drift")
    _need(tuple(row["session_name"] for row in feasibility["recordings"]) == H1_HELDOUT_SESSIONS,
          "metadata audit roster drift")
    evaluable = sum(int(row["m4_evaluable"] and row["legal_trial_count"] >= M4_MINIMUM_LEGAL_TRIALS)
                    for row in feasibility["recordings"])
    _need(evaluable == feasibility["m4_evaluable_recordings"], "metadata feasibility count drift")
    expected_status = PASS_STATUS if evaluable == 14 else STOP_STATUS
    _need(terminal.get("status") == feasibility.get("status") == expected_status, "metadata feasibility status drift")
    for field in (
        "heldout_neural_arrays_read", "heldout_behavior_arrays_read", "heldout_velocity_arrays_read",
        "model_inference_calls", "prediction_rows", "r2_calculations", "loss_calculations", "carrier_fits",
        "plan_fits", "normalizer_fits", "gpu_training_steps",
    ):
        _need(access.get(field) == 0, f"forbidden metadata-stage activity: {field}")
    _need(access.get("selection_performed") is False and access.get("trial_level_arrays_persisted") is False,
          "metadata audit selection/persistence drift")
    _need(terminal.get("gpu_training_started") is False and terminal.get("automatic_budget_change") is False,
          "metadata terminal training/budget drift")
    return terminal


__all__ = (
    "PASS_STATUS",
    "PREDECESSOR_COMMIT",
    "PREDECESSOR_TERMINAL_SHA256",
    "SCHEMA",
    "STOP_STATUS",
    "create_attempt",
    "dry_plan",
    "load_attempt",
    "run_metadata_feasibility_audit",
    "validate_predecessor",
    "verify_metadata_terminal",
)
