"""Difference-in-differences interpretation of the immutable V1 2x2 matrix."""
from __future__ import annotations

import itertools
from typing import Any, Mapping, Sequence

import numpy as np

from h1_date_lodo_activity_system_compare_v1.plan import DATE_ORDER


V1_RELATIVE = "tfpd_exploration/results/h1_date_lodo_activity_system_compare_v1.json"
V1_SHA256 = "0dc4576b92b5df383252d164b29d869ca0888a431756338b4938d0269e29a862"
MATERIAL_EFFECT = 0.01


def _exact_date_bootstrap_interval(values: Sequence[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (5,) or not np.isfinite(array).all():
        raise ValueError("bootstrap requires exactly five finite date effects")
    means = np.asarray([array[np.asarray(index, dtype=np.int64)].mean(dtype=np.float64) for index in itertools.product(range(5), repeat=5)])
    return [float(value) for value in np.quantile(means, [0.025, 0.975], method="linear")]


def corrected_decision(date_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if tuple(str(row.get("outer_date")) for row in date_rows) != DATE_ORDER:
        raise ValueError("date rows must follow the exact five-date order")
    hs_gain: list[float] = []
    hc_gain: list[float] = []
    final_gap: list[float] = []
    for row in date_rows:
        systems = row.get("systems")
        if not isinstance(systems, Mapping) or set(systems) != {"H-S", "H-C"}:
            raise ValueError("date row lacks exact systems")
        values: dict[tuple[str, str], float] = {}
        for system in ("H-S", "H-C"):
            arms = systems[system]
            if not isinstance(arms, Mapping) or set(arms) != {"STATIC_SUPPORT", "CAUSAL_GROWING_CAP30"}:
                raise ValueError("system row lacks exact arms")
            for arm, evidence in arms.items():
                values[(system, arm)] = float(evidence["equal_recording_mean_r2"])
        hs_gain.append(values[("H-S", "CAUSAL_GROWING_CAP30")] - values[("H-S", "STATIC_SUPPORT")])
        hc_gain.append(values[("H-C", "CAUSAL_GROWING_CAP30")] - values[("H-C", "STATIC_SUPPORT")])
        final_gap.append(values[("H-C", "CAUSAL_GROWING_CAP30")] - values[("H-S", "CAUSAL_GROWING_CAP30")])
    interaction = [hc - hs for hc, hs in zip(hc_gain, hs_gain, strict=True)]
    mean_interaction = float(np.mean(interaction, dtype=np.float64))
    mean_final_gap = float(np.mean(final_gap, dtype=np.float64))
    if mean_interaction >= MATERIAL_EFFECT:
        interaction_verdict = "POSITIVE_CARRIER_BY_ACTIVITY_INTERACTION"
    elif mean_interaction <= -MATERIAL_EFFECT:
        interaction_verdict = "ACTIVITY_GAIN_MATERIALLY_LARGER_WITHOUT_CARRIER"
    else:
        interaction_verdict = "NO_MATERIAL_CARRIER_BY_ACTIVITY_INTERACTION"
    if mean_final_gap >= MATERIAL_EFFECT:
        system_basis = "H-C_CAUSAL_GROWING_RETAINS_HIGHER_FINAL_LEVEL"
    elif mean_final_gap <= -MATERIAL_EFFECT:
        system_basis = "H-S_CAUSAL_GROWING_HAS_HIGHER_FINAL_LEVEL"
    else:
        system_basis = "H-S_AND_H-C_CAUSAL_GROWING_ARE_MATERIALLY_TIED"
    return {
        "equal_date_mean": {
            "hs_growing_minus_static": float(np.mean(hs_gain, dtype=np.float64)),
            "hc_growing_minus_static": float(np.mean(hc_gain, dtype=np.float64)),
            "interaction_hc_gain_minus_hs_gain": mean_interaction,
            "final_growing_hc_minus_hs": mean_final_gap,
        },
        "exact_five_date_bootstrap_95": {
            "hs_growing_minus_static": _exact_date_bootstrap_interval(hs_gain),
            "hc_growing_minus_static": _exact_date_bootstrap_interval(hc_gain),
            "interaction_hc_gain_minus_hs_gain": _exact_date_bootstrap_interval(interaction),
            "final_growing_hc_minus_hs": _exact_date_bootstrap_interval(final_gap),
        },
        "per_date": {
            "hs_growing_minus_static": dict(zip(DATE_ORDER, hs_gain, strict=True)),
            "hc_growing_minus_static": dict(zip(DATE_ORDER, hc_gain, strict=True)),
            "interaction_hc_gain_minus_hs_gain": dict(zip(DATE_ORDER, interaction, strict=True)),
            "final_growing_hc_minus_hs": dict(zip(DATE_ORDER, final_gap, strict=True)),
        },
        "positive_hs_activity_dates": sum(value > 0.0 for value in hs_gain),
        "positive_hc_activity_dates": sum(value > 0.0 for value in hc_gain),
        "positive_final_hc_dates": sum(value > 0.0 for value in final_gap),
        "material_effect": MATERIAL_EFFECT,
        "interaction_verdict": interaction_verdict,
        "recommended_frozen_system_basis": system_basis,
        "formal_selection_claim": False,
    }


def dry_plan() -> dict[str, Any]:
    return {
        "schema": "h1_date_lodo_activity_system_compare_interpretation_plan_v2",
        "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_CUDA_NO_WRITE",
        "v1": {"relative": V1_RELATIVE, "sha256": V1_SHA256},
        "correction": "use_difference_in_differences_for_carrier_by_activity_interaction",
        "material_effect": MATERIAL_EFFECT,
        "target_updates": 0,
    }
