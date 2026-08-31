"""Preregistered source attrition gate for masked dense-auxiliary V2."""
from __future__ import annotations

from typing import Any, Mapping


MIN_RECORDING_RETENTION = 0.25


def evaluate_attrition(training_audit: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if not training_audit:
        raise ValueError("attrition gate requires at least one source recording")
    fractions = {session: float(row["retention_fraction"])
                 for session, row in training_audit.items()}
    empty = [session for session, row in training_audit.items()
             if int(row["retained_windows"]) == 0]
    missing_trials = {session: list(row["trials_losing_all_windows"])
                      for session, row in training_audit.items()
                      if row["trials_losing_all_windows"]}
    final_bad = [session for session, row in training_audit.items()
                 if row["final_all_true"] is not True]
    passed = (not empty and not missing_trials and not final_bad
              and min(fractions.values()) >= MIN_RECORDING_RETENTION)
    return {
        "minimum_recording_retention_fraction": min(fractions.values()),
        "retention_fraction_by_recording": fractions,
        "empty_recordings": empty,
        "recordings_with_trials_losing_all_windows": missing_trials,
        "recordings_with_final_contract_failure": final_bad,
        "thresholds": {"minimum_recording_retention_fraction": MIN_RECORDING_RETENTION,
                       "every_original_trial_retains_a_window": True,
                       "every_retained_final_is_legal": True},
        "attrition_gate_passed": passed,
        "verdict": "PASS_V2_ATTRITION_GATE_AUTHORIZE_GPU_SMOKE" if passed
                   else "STOP_V2_ATTRITION_GATE_NO_GPU",
    }
