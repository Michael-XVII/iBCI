"""Build the corrected interpretation from the immutable V1 raw matrix."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
from typing import Any, Mapping

from m1_h1_activity_headroom_v1.m1 import write_once

from .plan import DATE_ORDER, V1_RELATIVE, V1_SHA256, corrected_decision


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    path = root / V1_RELATIVE
    sidecar = path.with_name(path.name + ".sha256")
    _need(path.is_file() and not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o444, "V1 body is not immutable")
    _need(sidecar.is_file() and not sidecar.is_symlink() and stat.S_IMODE(sidecar.stat().st_mode) == 0o444, "V1 sidecar is not immutable")
    _need(_sha_file(path) == V1_SHA256 and sidecar.read_text(encoding="ascii") == f"{V1_SHA256}  {path.name}\n", "V1 SHA binding drift")
    source = json.loads(path.read_text(encoding="utf-8"))
    _need(source.get("schema") == "h1_date_lodo_activity_system_compare_v1" and source.get("status") == "COMPLETE_FROZEN_WEIGHT_FIVE_DATE_HS_HC_ACTIVITY_COMPARISON", "V1 schema/status drift")
    _need(source.get("date_order") == list(DATE_ORDER) and source.get("target_optimizer_backward_update") == 0, "V1 date/update contract drift")
    rows = source.get("date_results")
    _need(isinstance(rows, list) and len(rows) == 5, "V1 date matrix drift")
    target_digests = []
    for row in rows:
        _need(isinstance(row, Mapping), "V1 date row drift")
        systems = row.get("systems")
        _need(isinstance(systems, Mapping), "V1 system matrix drift")
        cell_targets = [systems[system][arm]["target_sha256"] for system in ("H-S", "H-C") for arm in ("STATIC_SUPPORT", "CAUSAL_GROWING_CAP30")]
        _need(len(set(cell_targets)) == 1 and cell_targets[0] == row.get("target_sha256"), "V1 same-target proof drift")
        target_digests.append(cell_targets[0])
    decision = corrected_decision(rows)
    return {
        "schema": "h1_date_lodo_activity_system_compare_interpretation_v2",
        "status": "COMPLETE_CORRECTED_DIFFERENCE_IN_DIFFERENCES_INTERPRETATION",
        "v1": {"relative": V1_RELATIVE, "sha256": V1_SHA256, "raw_scores_unchanged": True},
        "date_order": list(DATE_ORDER),
        "target_sha256_by_date": dict(zip(DATE_ORDER, target_digests, strict=True)),
        "correction": {
            "supersedes_v1_descriptive_verdict_only": True,
            "reason": "carrier_by_activity_interaction_requires_difference_in_differences_not_final_growing_system_gap",
            "v1_raw_cells_remain_valid": True,
        },
        "decision": decision,
        "interaction_verdict": decision["interaction_verdict"],
        "recommended_frozen_system_basis": decision["recommended_frozen_system_basis"],
        "target_optimizer_backward_update": 0,
        "formal_selection_claim": False,
    }


__all__ = ("run", "write_once")
