"""CPU-only frozen contract for H1 CAL-AUG M3-Aware Dual-Selection V2."""
from __future__ import annotations

from collections import Counter
import hashlib
from typing import Any, Mapping, Sequence


SCHEMA = "h1_cal_aug_m3_aware_dual_selection_v2"
PREDECESSOR_COMMIT = "84c7aaecb656be812d046317fec07783fa6701a4"
V1_C1_CHECKPOINT_SHA256 = "0f406a8e69fdb57cf6a5480149f04ab3500e7fad849d36db38042edbadb2cd06"
V1_INITIAL_STATE_SHA256 = "bc6dc8a0543c760811f770206c7ee22ae35eaf970c6dad0ec259a84172e4d04b"
V1_DROPOUT_SHA256 = "c1dd24d682878f477050cb4e5886dd1f34aff3424b58bdcb158c12c11ba1d247"
V1_SOURCE_AUTHORITY_SHA256 = "8ea4bb1174c00ab713843cd7561562d43f81509eaaea6ea12ee80cd4eba95de7"
V1_SCHEDULE_SHA256 = "91ff1ac5d5d72dfe4833d9dc7dad726adfbb913afb0a75970d36bbb569e58bd4"
V1_BATCH_ORDER_SHA256 = "390a648f85da9b4ca558d84d752b5035fe0a2330e064962b5aea545ef2db4bc1"
V1_M7_SCHEDULE_SHA256 = "ed22dd1ae7680fb72110c85a1b890ceff230f10b0366363f26639897ec27eacf"
V1_SOURCE_TENSOR_SHA256 = "5fd41c76db209fdde035636bab0197911effb6d0a4929dee5b2602f657a95bd9"
V1_PLAN_SHA256 = "a92b57350f2dcb04027bb6d848e5582d84e1bef4bf507d6844963ccad3c87bd5"
V1_NORMALIZER_SHA256 = "9b35870244a38d0568c3e5f3fd5ab2f10bc2ede815dafe308a130a04e5db6732"
V1_CARRIER_CACHE_SHA256 = "9a3ecd8223bffa016ff64a4d682a441db96f6f11af5356263b4e82373abef98c"

C2_CYCLE = (7, 5, 4, 3)
EPOCHS = 50
BATCHES_PER_EPOCH = 4133
GLOBAL_STEPS = 206650
BATCH_SIZE = 32
WINDOW_SIZE = 700
PREDICTION_DIVISOR = 20.0
MODEL_PARAMETERS = 10_947_836


class V2ContractError(ValueError):
    pass


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise V2ContractError(message)


def prefix_schedule() -> tuple[tuple[int, ...], ...]:
    rows = []
    for epoch in range(EPOCHS):
        token = hashlib.sha256(f"{SCHEMA}|prefix|h1_all_source_13|{epoch}".encode()).digest()
        offset = int.from_bytes(token[:8], "big") % len(C2_CYCLE)
        row = tuple(C2_CYCLE[(index + offset) % len(C2_CYCLE)] for index in range(BATCHES_PER_EPOCH))
        counts = Counter(row)
        _need(set(row) == set(C2_CYCLE), "C2 cycle coverage drift")
        _need(max(counts.values()) - min(counts.values()) <= 1, "C2 cycle balance drift")
        rows.append(row)
    return tuple(rows)


def select_epoch(rows: Sequence[Mapping[str, Any]], metric: str) -> Mapping[str, Any]:
    _need(len(rows) == EPOCHS, "selection requires all 50 epochs")
    _need({int(row["epoch_zero_based"]) for row in rows} == set(range(EPOCHS)), "epoch roster drift")
    for row in rows:
        _need(metric in row and all(key in row for key in ("worst_session_r2", "session_std_population")), "selection metric incomplete")
    return max(
        rows,
        key=lambda row: (
            float(row[metric]),
            float(row["worst_session_r2"]),
            -float(row["session_std_population"]),
            -int(row["epoch_zero_based"]),
        ),
    )


def dry_plan() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "DRY_RUN_NO_WRITE_NO_DATA_NO_CUDA",
        "predecessor_commit": PREDECESSOR_COMMIT,
        "frozen_baseline": "V1 C1 epoch49",
        "new_training_arms": ["c2"],
        "c2_cycle": list(C2_CYCLE),
        "epochs": EPOCHS,
        "global_steps": GLOBAL_STEPS,
        "checkpoint_epochs": list(range(EPOCHS)),
        "training_validation_interleaved": False,
        "offline_selection_surfaces": ["HI-M3", "HO-M3"],
        "ho_surface_role": "development/model-selection; not untouched held-out generalization",
        "selection_tie_break": ["higher_mean", "higher_worst_session", "lower_population_std", "earlier_epoch"],
        "docker_builds": 0,
        "evalai_submissions": 0,
        "writes": 0,
        "nwb_files_opened": 0,
        "cuda_initialized": False,
    }


__all__ = (
    "BATCHES_PER_EPOCH", "BATCH_SIZE", "C2_CYCLE", "EPOCHS", "GLOBAL_STEPS",
    "MODEL_PARAMETERS", "PREDECESSOR_COMMIT", "SCHEMA", "V1_C1_CHECKPOINT_SHA256",
    "V1_DROPOUT_SHA256", "V1_INITIAL_STATE_SHA256", "V2ContractError", "dry_plan",
    "prefix_schedule", "select_epoch",
)
