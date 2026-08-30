"""Immutable authorities and descriptive decision for the H1 2x2 comparison."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from h1_date_lodo_activity_headroom_v1.plan import AUTHORITIES as HC_AUTHORITIES
from h1_date_lodo_activity_headroom_v1.plan import DATE_ORDER


HC_PREDECESSOR_RELATIVE = "tfpd_exploration/results/h1_date_lodo_activity_headroom_v1.json"
HC_PREDECESSOR_SHA256 = "65c9bb40ad45ab7b74740da88fd8081504b7656e807e76b6eb9db903450adb68"
SYSTEM_EFFECT_THRESHOLD = 0.01


@dataclass(frozen=True)
class HsAuthority:
    date: str
    checkpoint_sha256: str
    config_sha256: str
    accepted_static_pooled_r2: float

    @property
    def checkpoint_relative(self) -> str:
        return f"tfpd_exploration/results/h1_date_lodo_checkpoint_cache_v1/{self.date}/hs_epoch_049.ckpt"

    @property
    def config_relative(self) -> str:
        return f"tfpd_exploration/results/h1_date_lodo_checkpoint_cache_v1/{self.date}/hs_config.yaml"


HS_AUTHORITIES = {
    "19250108": HsAuthority(
        "19250108", "0d2454d8c37deb9a7855190f6d09866eaad356387e044cfd2940df7cebbf9cf4",
        "6b179acac265e5694cfb490e22565c072f33afb01a7ddc896a5a1e9553b4c636", 0.4264575655391447,
    ),
    "19250113": HsAuthority(
        "19250113", "0d5c9482f41cfab421cd1474c8165b9d650485534059941bfc242b687a1c45e8",
        "aaf93ab486303d90f9587c4bed430330f5c2cf792dc3a97c44b0bcdc435a0936", 0.25665944932970264,
    ),
    "19250115": HsAuthority(
        "19250115", "1347a82ac5feac34ed787c09b187de623954a57464a666a0e1d65c02d77c06b8",
        "664de9d8e4c92c0059bff3bc2e63530207c652210da189e46d69ad49b1796523", 0.4418922977875869,
    ),
    "19250119": HsAuthority(
        "19250119", "506d655617a9feee594ec4845bfb6034ef1126005e4a7633858d4fbd3d2fcc1f",
        "8e947549b7383fe077c623bce45dfe8fab813296759ab5def09d77c8a2edf606", 0.2946034289841183,
    ),
    "19250120": HsAuthority(
        "19250120", "32bde896ac0ca1bb9ec5e222d17167e80dc1e4ad169058e63857c6204b9b39a5",
        "afec359e020f04b4a6d3dd4130d72fd1f7f2f6b6c1a8b5e2d757443bf9185891", 0.37544321142614345,
    ),
}


def comparison_decision(date_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if tuple(str(row.get("outer_date")) for row in date_rows) != DATE_ORDER:
        raise ValueError("date rows must follow the exact five-date order")
    deltas: dict[str, list[float]] = {
        "hs_growing_minus_static": [],
        "hc_growing_minus_static": [],
        "growing_hc_minus_hs": [],
        "static_hc_minus_hs": [],
    }
    for row in date_rows:
        systems = row.get("systems")
        if not isinstance(systems, Mapping) or set(systems) != {"H-S", "H-C"}:
            raise ValueError("date row lacks the exact H-S/H-C systems")
        values: dict[tuple[str, str], float] = {}
        for system in ("H-S", "H-C"):
            arms = systems[system]
            if not isinstance(arms, Mapping) or set(arms) != {"STATIC_SUPPORT", "CAUSAL_GROWING_CAP30"}:
                raise ValueError("system row lacks the exact static/growing arms")
            for arm, evidence in arms.items():
                values[(system, arm)] = float(evidence["equal_recording_mean_r2"])
        deltas["hs_growing_minus_static"].append(values[("H-S", "CAUSAL_GROWING_CAP30")] - values[("H-S", "STATIC_SUPPORT")])
        deltas["hc_growing_minus_static"].append(values[("H-C", "CAUSAL_GROWING_CAP30")] - values[("H-C", "STATIC_SUPPORT")])
        deltas["growing_hc_minus_hs"].append(values[("H-C", "CAUSAL_GROWING_CAP30")] - values[("H-S", "CAUSAL_GROWING_CAP30")])
        deltas["static_hc_minus_hs"].append(values[("H-C", "STATIC_SUPPORT")] - values[("H-S", "STATIC_SUPPORT")])
    means = {name: sum(values) / len(values) for name, values in deltas.items()}
    system_effect = means["growing_hc_minus_hs"]
    if system_effect >= SYSTEM_EFFECT_THRESHOLD:
        verdict = "HC_ACTIVITY_SYNERGY_RETAIN_CARRIER_SYSTEM"
    elif system_effect <= -SYSTEM_EFFECT_THRESHOLD:
        verdict = "HS_GROWING_OUTPERFORMS_HC"
    else:
        verdict = "ACTIVITY_MEMORY_DOMINANT_NO_MATERIAL_CARRIER_SYNERGY"
    return {
        "equal_date_mean_deltas": means,
        "per_date_deltas": {name: dict(zip(DATE_ORDER, values, strict=True)) for name, values in deltas.items()},
        "positive_hs_growing_dates": sum(value > 0.0 for value in deltas["hs_growing_minus_static"]),
        "positive_hc_growing_dates": sum(value > 0.0 for value in deltas["hc_growing_minus_static"]),
        "system_effect_threshold": SYSTEM_EFFECT_THRESHOLD,
        "verdict": verdict,
        "formal_selection_claim": False,
    }


def dry_plan() -> dict[str, Any]:
    return {
        "schema": "h1_date_lodo_activity_system_compare_plan_v1",
        "status": "DRY_NO_TARGET_NO_CHECKPOINT_LOAD_NO_CUDA_NO_WRITE",
        "dates": list(DATE_ORDER),
        "matrix": ["H-S:STATIC_SUPPORT", "H-S:CAUSAL_GROWING_CAP30", "H-C:STATIC_SUPPORT", "H-C:CAUSAL_GROWING_CAP30"],
        "hs_authorities": {date: asdict(HS_AUTHORITIES[date]) for date in DATE_ORDER},
        "hc_authorities": {date: asdict(HC_AUTHORITIES[date]) for date in DATE_ORDER},
        "hc_predecessor": {"relative": HC_PREDECESSOR_RELATIVE, "sha256": HC_PREDECESSOR_SHA256},
        "system_effect_threshold": SYSTEM_EFFECT_THRESHOLD,
        "target_updates": 0,
    }
