"""Contract and selection laws for bounded causal H1 activity memory."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from h1_date_lodo_activity_headroom_v1.plan import DATE_ORDER


PREDECESSOR_RELATIVE = "tfpd_exploration/results/h1_date_lodo_activity_headroom_v1.json"
PREDECESSOR_SHA256 = "65c9bb40ad45ab7b74740da88fd8081504b7656e807e76b6eb9db903450adb68"
SUPPORT = 4
CAP = 30
MIN_MEAN_IMPROVEMENT = 0.01
MIN_POSITIVE_DATES = 4
ARM_ORDER = ("CAUSAL_FIFO_CAP30", "CAUSAL_COVERAGE_CAP30", "CAUSAL_ALL_PAST")


def selection_for_arm(arm: str, *, output_trial_index: int) -> tuple[int, ...]:
    if arm not in ARM_ORDER or type(output_trial_index) is not int or output_trial_index < SUPPORT:
        raise ValueError("causal representative selection contract drift")
    support = tuple(range(SUPPORT))
    completed_query = tuple(range(SUPPORT, output_trial_index))
    if arm == "CAUSAL_ALL_PAST":
        return support + completed_query
    available = CAP - SUPPORT
    if len(completed_query) <= available:
        return support + completed_query
    if arm == "CAUSAL_FIFO_CAP30":
        return support + completed_query[-available:]
    offsets = np.rint(np.linspace(0, len(completed_query) - 1, available)).astype(np.int64)
    selected = tuple(completed_query[int(index)] for index in offsets)
    if len(selected) != available or len(set(selected)) != available:
        raise ValueError("coverage selector did not produce exact unique capacity")
    return support + selected


def decision(date_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if tuple(str(row.get("outer_date")) for row in date_rows) != DATE_ORDER:
        raise ValueError("date rows must follow exact order")
    deltas = []
    fifo_deltas = []
    all_past_deltas = []
    for row in date_rows:
        baseline = float(row["predecessor"]["CAUSAL_GROWING_CAP30"]["equal_recording_mean_r2"])
        arms = row.get("new_arms")
        if not isinstance(arms, Mapping) or tuple(arms) != ARM_ORDER:
            raise ValueError("new arm order drift")
        fifo_deltas.append(float(arms["CAUSAL_FIFO_CAP30"]["equal_recording_mean_r2"]) - baseline)
        deltas.append(float(arms["CAUSAL_COVERAGE_CAP30"]["equal_recording_mean_r2"]) - baseline)
        all_past_deltas.append(float(arms["CAUSAL_ALL_PAST"]["equal_recording_mean_r2"]) - baseline)
    mean_delta = sum(deltas) / len(deltas)
    positive = sum(value > 0.0 for value in deltas)
    passed = mean_delta >= MIN_MEAN_IMPROVEMENT and positive >= MIN_POSITIVE_DATES
    return {
        "coverage_minus_frozen_growing_equal_date_mean": mean_delta,
        "coverage_positive_dates": positive,
        "required_mean_improvement": MIN_MEAN_IMPROVEMENT,
        "required_positive_dates": MIN_POSITIVE_DATES,
        "per_date_coverage_minus_frozen_growing": dict(zip(DATE_ORDER, deltas, strict=True)),
        "fifo_minus_frozen_growing_equal_date_mean": sum(fifo_deltas) / len(fifo_deltas),
        "all_past_minus_frozen_growing_equal_date_mean": sum(all_past_deltas) / len(all_past_deltas),
        "pass": passed,
        "verdict": "PASS_CAUSAL_REPRESENTATIVE_ACTIVITY_CAP30" if passed else "STOP_CAUSAL_REPRESENTATIVE_ACTIVITY_CAP30",
    }


def dry_plan() -> dict[str, Any]:
    return {
        "schema": "h1_causal_representative_activity_plan_v1",
        "status": "DRY_NO_TARGET_NO_CHECKPOINT_NO_CUDA_NO_WRITE",
        "dates": list(DATE_ORDER),
        "predecessor": {"relative": PREDECESSOR_RELATIVE, "sha256": PREDECESSOR_SHA256},
        "new_arms": list(ARM_ORDER),
        "governing_arm": "CAUSAL_COVERAGE_CAP30",
        "support": SUPPORT,
        "capacity": CAP,
        "target_updates": 0,
    }
