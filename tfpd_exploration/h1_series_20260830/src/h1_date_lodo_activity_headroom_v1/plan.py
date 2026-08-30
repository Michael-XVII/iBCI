"""Immutable authorities and decision rule for H1 date-LODO breadth."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


DATE_ORDER = ("19250108", "19250113", "19250115", "19250119", "19250120")
MIN_POSITIVE_GROWING_DATES = 4
MIN_EQUAL_DATE_GROWING_DELTA = 0.01


@dataclass(frozen=True)
class DateAuthority:
    date: str
    terminal_sha256: str
    checkpoint_sha256: str
    config_sha256: str
    source_manifest_sha256: str
    query_window_indices_sha256: str
    accepted_static_pooled_r2: float

    @property
    def terminal_relative(self) -> str:
        return (
            "SPINT-main/pilot_artifacts/h1_carrierid_date_lodo_phase2/"
            "terminal_evaluations/"
            f"H1_CARRIERID_DATE_LODO_PHASE2_{self.date}_HS_HC_TERMINAL_EVALUATION_v1.json"
        )

    @property
    def checkpoint_relative(self) -> str:
        return f"tfpd_exploration/results/h1_date_lodo_checkpoint_cache_v1/{self.date}/epoch_049.ckpt"

    @property
    def config_relative(self) -> str:
        return f"tfpd_exploration/results/h1_date_lodo_checkpoint_cache_v1/{self.date}/config.yaml"

    @property
    def source_manifest_relative(self) -> str:
        return (
            "SPINT-main/pilot_artifacts/h1_carrierid_date_lodo_phase1/"
            f"source_bundles_v1/{self.date}/shared_source_manifest.json"
        )


AUTHORITIES = {
    "19250108": DateAuthority(
        "19250108",
        "cd91ab7ddac7eed4bad715dbfc8cb82f432f11c889adc950d49832b9ab9c28a7",
        "cfe7a482b8de618c2aa069a3547827cb590b9f54400c554b6d5a08f82b6a787c",
        "d4af24f4a29023663823b7ca37b54fd60ba6fbd7db4a9439b3e29358f43ca73d",
        "647d9c91c11da0267b4781db180065c4038b2b8934ba2ce935a2396e30850cc5",
        "a0f9851d1a149cd62dbf4ee5561119861797664ec94bf132fd1380a23c702672",
        0.5457671023719811,
    ),
    "19250113": DateAuthority(
        "19250113",
        "30666d144326733e76615942d925731b0bf072549c453787b98b81d141532819",
        "acda8c21b5d9dc0e91b91bdab4c7c5626d189ab3b5436e57476b7a44bda42458",
        "0b5616dc4d1d79b81800ce977342fc0d398425d6a24b75906d2b16f6adbc1d99",
        "d910d088cbf222cde457b8e0ea833dec2e40fa4a65fd5dd33f28803dae61dec5",
        "0e1ee7f45cf8728fb494a4e30c55f615def4797866fc128640fcd79fbac57905",
        0.3393896881102032,
    ),
    "19250115": DateAuthority(
        "19250115",
        "3e2521ab190c36c33423abecd62c04457308cf0b84b3c2e0e0753c56615ce617",
        "4086f575755f54a731f88a7cf1d901dd0a40c45842deb380919c7b6392bf1174",
        "55e9e3b32ec40e895a5611e459a6bf7e81151e82996ff8f021bea411cf0d850d",
        "7611a4d4a6a9ddbf43ae5c16253c95a555faf7dc620b41d02f09c95a8b67fbe2",
        "81ab5ca22c6c317d2af36b8bccb0feea0256da04beba2fa32478824157ad80be",
        0.5237869559374768,
    ),
    "19250119": DateAuthority(
        "19250119",
        "af67a363e861f358bb541e03d000cfee15266809e7fbf50a7863fea30131906c",
        "90f57d86a780ebbd5a705288829dd6defc2121673137b89952cd2f702713d4b5",
        "0ac3117c1f2cd4df4f950189ccb5894f0c43ff59c0bdead6f32b5be82b030991",
        "792207cfa7599f89b4166a23c20f04f9ad8679afecc5afdf17a7782db0fab202",
        "6c339c8e2b0b4691c20396ff691c10397c9308520252b1cc9486d7ac3db6427e",
        0.31730403564738896,
    ),
    "19250120": DateAuthority(
        "19250120",
        "aac68a947a902d55a7b31221e1b446efeec1598acc99e69280ee41f750ff5dcd",
        "e9e38b884f5459f3f0c81e4301e2d505faa9528062a536d1a165c2891a89b1cd",
        "c4ead9f929bc543363176bd59b0c39e3d60f2af470c2c7dd1d0b757f7dab2a42",
        "df0ab72f191c3ccf7c8fdd32570ba9ced742cebef35ec2f16795868cfa2d2934",
        "6b6bd44d20fcdb5418104422a8acc4583f0405f8fa5cf6c9e9e293a1fe9fda60",
        0.35024086568273993,
    ),
}


def breadth_decision(date_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if tuple(str(row.get("outer_date")) for row in date_rows) != DATE_ORDER:
        raise ValueError("date rows must follow the exact five-date authority order")
    deltas: list[float] = []
    for row in date_rows:
        results = row.get("results")
        if not isinstance(results, Sequence):
            raise ValueError("date row lacks result sequence")
        by_arm = {str(item.get("arm")): item for item in results if isinstance(item, Mapping)}
        if set(by_arm) != {
            "STATIC_SUPPORT", "ROLLING_FIXED_M", "CAUSAL_GROWING_CAP30", "FULL_SESSION_ORACLE"
        }:
            raise ValueError("date row lacks exact four-arm result set")
        static = float(by_arm["STATIC_SUPPORT"]["equal_recording_mean_r2"])
        growing = float(by_arm["CAUSAL_GROWING_CAP30"]["equal_recording_mean_r2"])
        deltas.append(growing - static)
    positive = sum(delta > 0.0 for delta in deltas)
    mean_delta = sum(deltas) / len(deltas)
    breadth_pass = positive >= MIN_POSITIVE_GROWING_DATES and mean_delta >= MIN_EQUAL_DATE_GROWING_DELTA
    return {
        "positive_growing_dates": positive,
        "required_positive_growing_dates": MIN_POSITIVE_GROWING_DATES,
        "equal_date_mean_growing_delta": mean_delta,
        "required_equal_date_mean_growing_delta": MIN_EQUAL_DATE_GROWING_DELTA,
        "per_date_growing_delta": dict(zip(DATE_ORDER, deltas, strict=True)),
        "breadth_pass": breadth_pass,
        "verdict": (
            "PASS_H1_ACTIVITY_HEADROOM_BREADTH_FOR_NEW_STATE_DESIGN"
            if breadth_pass
            else "STOP_H1_ACTIVITY_HEADROOM_NOT_BROAD_ACROSS_DATES"
        ),
    }


def dry_plan() -> dict[str, Any]:
    return {
        "schema": "h1_date_lodo_activity_headroom_plan_v1",
        "status": "DRY_NO_TARGET_NO_CHECKPOINT_LOAD_NO_CUDA_NO_WRITE",
        "dates": list(DATE_ORDER),
        "arms": ["STATIC_SUPPORT", "ROLLING_FIXED_M", "CAUSAL_GROWING_CAP30", "FULL_SESSION_ORACLE"],
        "authorities": {date: asdict(AUTHORITIES[date]) for date in DATE_ORDER},
        "decision": {
            "min_positive_growing_dates": MIN_POSITIVE_GROWING_DATES,
            "min_equal_date_growing_delta": MIN_EQUAL_DATE_GROWING_DELTA,
        },
        "target_updates": 0,
    }
