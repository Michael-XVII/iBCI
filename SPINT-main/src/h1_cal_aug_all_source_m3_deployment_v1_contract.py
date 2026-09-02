"""Pure review-stage contract for H1 all-source M3 deployment finalization.

This module is deliberately stdlib-only.  It performs no filesystem access and
has no data, NWB, model, training, inference, packaging, or submission entry
point.
"""
from __future__ import annotations

from collections import Counter
import hashlib
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "h1_cal_aug_all_source_m3_deployment_v1"
ARMS = ("t0", "c1")
T0_PREFIX = 7
C1_CYCLE = (7, 5, 4)
TRAINING_CARRIER_TRIALS = 4
DEPLOYMENT_CALIBRATION_TRIALS = 3
WINDOW_SIZE = 700
HIDDEN_SIZE = 32
CARRIER_DIMS = 4
OUTPUT_DIMS = 7
SEED = 42
BATCH_SIZE = 32
LEARNING_RATE = 5.0e-5
WEIGHT_DECAY = 0.0
EPOCHS = 50
CHECKPOINT_EPOCH_ZERO_BASED = 49
PREDICTION_DIVISOR = 20.0

EXPERIMENT4_A1_COMMIT = "c60052c9d8ccb8391d6ce53bde9ccfb4f2319884"
EXPERIMENT4_A1_TERMINAL_SHA256 = "dc9e7ab44954d3d193f67f9bf8936aafdaf2b05be9968d5e0091c0b0ecf092fd"
M4_STOP_COMMIT = "0d0ab2f"
M4_STOP_TERMINAL_SHA256 = "3ff971dc576958b13ace990bcca8aea2e8b999e2af2ed50f418296d05f8d5cfc"
M3_SEAL_COMMIT = "36a9f58"
M3_TERMINAL_SHA256 = "199a2fec864d7ae40d33ec911e43cd32e8623e5687ed9d44c5c9ac946a964429"
M3_VERDICT = "STRONG_M3_PREFIX_EXTRAPOLATION"

HELDIN_SESSION_TO_FALCON_KEY: tuple[tuple[str, str], ...] = (
    ("ses-19250101T111740", "S0_set_1"),
    ("ses-19250101T112404", "S0_set_2"),
    ("ses-19250108T110520", "S1_set_1"),
    ("ses-19250108T111022", "S1_set_2"),
    ("ses-19250108T111455", "S1_set_3"),
    ("ses-19250113T120811", "S2_set_1"),
    ("ses-19250113T121303", "S2_set_2"),
    ("ses-19250115T110633", "S3_set_1"),
    ("ses-19250115T111328", "S3_set_2"),
    ("ses-19250119T113543", "S4_set_1"),
    ("ses-19250119T114045", "S4_set_2"),
    ("ses-19250120T115044", "S5_set_1"),
    ("ses-19250120T115537", "S5_set_2"),
)

HELDOUT_SESSION_TO_FALCON_KEY: tuple[tuple[str, str], ...] = (
    ("ses-19250126T113454", "S6_set_1"),
    ("ses-19250126T114029", "S6_set_2"),
    ("ses-19250127T120333", "S7_set_1"),
    ("ses-19250127T120826", "S7_set_2"),
    ("ses-19250129T112555", "S8_set_1"),
    ("ses-19250129T113059", "S8_set_2"),
    ("ses-19250202T113958", "S9_set_1"),
    ("ses-19250202T114452", "S9_set_2"),
    ("ses-19250203T113515", "S10_set_1"),
    ("ses-19250203T114018", "S10_set_2"),
    ("ses-19250206T112219", "S11_set_1"),
    ("ses-19250206T112712", "S11_set_2"),
    ("ses-19250209T111826", "S12_set_1"),
    ("ses-19250209T112327", "S12_set_2"),
)


class DeploymentContractError(ValueError):
    """A frozen finalization boundary was violated."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise DeploymentContractError(message)


def frozen_training_contract() -> dict[str, Any]:
    return {
        "model": "H1CarrierIdSpint",
        "hidden_size": HIDDEN_SIZE,
        "window_bins": WINDOW_SIZE,
        "carrier_dims": CARRIER_DIMS,
        "outputs": OUTPUT_DIMS,
        "arms": list(ARMS),
        "t0_identity_prefix": T0_PREFIX,
        "c1_identity_cycle": list(C1_CYCLE),
        "training_carrier_trials": TRAINING_CARRIER_TRIALS,
        "m3_in_training_cycle": False,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "precision": "fp32",
        "epochs": EPOCHS,
        "checkpoint_epoch_zero_based": CHECKPOINT_EPOCH_ZERO_BASED,
        "loss": "last_bin_mse",
        "prediction_divisor": PREDICTION_DIVISOR,
        "validation_checkpoint_selection": False,
        "early_stopping": False,
        "warm_start": False,
        "target_fitting": False,
        "budget_sweep": False,
        "hyperparameter_sweep": False,
    }


def deterministic_prefix_cycle(domain: str, batches: int) -> tuple[int, ...]:
    _need(bool(domain), "cycle domain must be nonempty")
    _need(int(batches) > 0, "cycle requires positive batch count")
    offset = int.from_bytes(hashlib.sha256(f"{SCHEMA}|prefix|{domain}".encode()).digest()[:8], "big") % 3
    return tuple(C1_CYCLE[(index + offset) % 3] for index in range(int(batches)))


def validate_balanced_cycle(values: Sequence[int]) -> None:
    _need(bool(values), "empty C1 cycle")
    _need(set(values) == set(C1_CYCLE), "C1 cycle must contain exactly M7/M5/M4")
    _need(DEPLOYMENT_CALIBRATION_TRIALS not in values, "M3 is forbidden from C1 training")
    counts = Counter(int(value) for value in values)
    _need(max(counts.values()) - min(counts.values()) <= 1, "C1 cycle is not balanced")


def session_mapping() -> dict[str, str]:
    rows = HELDIN_SESSION_TO_FALCON_KEY + HELDOUT_SESSION_TO_FALCON_KEY
    mapping = dict(rows)
    _need(len(mapping) == len(rows) == 27, "H1 session mapping is incomplete or duplicated")
    _need(len(set(mapping.values())) == 27, "FALCON keys are not one-to-one")
    return mapping


def classify_local_inventory(relative_paths: Iterable[str]) -> dict[str, Any]:
    paths = tuple(str(path) for path in relative_paths)
    counts = {
        "held_in_calib": sum(path.startswith("sub-HumanPitt-held-in-calib/") and path.endswith(".nwb") for path in paths),
        "held_in_minival": sum(path.startswith("sub-HumanPitt-held-in-minival/") and path.endswith(".nwb") for path in paths),
        "held_out_calib": sum(path.startswith("sub-HumanPitt-held-out-calib/") and path.endswith(".nwb") for path in paths),
        "held_out_eval_or_test": sum(
            path.endswith(".nwb") and ("held-out-minival" in path or "/eval/" in f"/{path}" or "/test/" in f"/{path}")
            for path in paths
        ),
    }
    complete_public_inventory = counts == {
        "held_in_calib": 13,
        "held_in_minival": 13,
        "held_out_calib": 14,
        "held_out_eval_or_test": 0,
    }
    return {
        "counts": counts,
        "complete_public_inventory": complete_public_inventory,
        "local_heldin_minival_scoring_available": counts["held_in_minival"] == 13,
        "local_heldout_postcalibration_scoring_available": counts["held_out_eval_or_test"] > 0,
        "official_heldout_score_requires_evalai": counts["held_out_eval_or_test"] == 0,
    }


def validate_m3_deployment(calibration_trial_ordinals: Sequence[int], scoring_stream: str) -> dict[str, Any]:
    trials = tuple(int(value) for value in calibration_trial_ordinals)
    _need(trials == (1, 2, 3), "M3 deployment requires earliest three chronological trials")
    _need(scoring_stream not in {"calibration", "held-out-calib", "same_three_trials"},
          "calibration trials cannot be used for governing R2")
    _need(scoring_stream in {"held-in-minival", "evalai-test"}, "unknown independent scoring stream")
    return {
        "identity_trials": [1, 2, 3],
        "carrier_trials": [1, 2, 3],
        "carrier_function": "fit_deployment_carrier",
        "scoring_stream": scoring_stream,
        "calibration_rows_scored": 0,
        "optimizer_steps": 0,
        "backward_steps": 0,
        "parameter_updates": 0,
    }


def validate_decoder_payload(payload: Mapping[str, Any], expected_keys: Sequence[str]) -> None:
    _need(payload.get("schema") == f"{SCHEMA}_decoder_payload", "decoder payload schema drift")
    _need(payload.get("window_bins") == WINDOW_SIZE, "decoder W drift")
    _need(payload.get("prediction_divisor") == PREDICTION_DIVISOR, "decoder scaling drift")
    sessions = payload.get("sessions")
    _need(isinstance(sessions, Mapping), "decoder session payload missing")
    _need(tuple(sessions) == tuple(expected_keys), "decoder session-key roster/order drift")
    for key in expected_keys:
        row = sessions[key]
        _need(row.get("identity_shape") == [3, 1024, 176], f"{key}: identity shape drift")
        _need(row.get("carrier_shape") == [176, 4], f"{key}: carrier shape drift")
        _need(row.get("calibration_trials") == [1, 2, 3], f"{key}: M3 calibration drift")


def dry_plan() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "DRY_REVIEW_ONLY_NO_WRITE_NO_FILE_ACCESS_NO_DATA_NO_CUDA",
        "authorized_stage": ["work_order", "cpu_contract", "cpu_tests", "dry_run"],
        "training_contract": frozen_training_contract(),
        "source_recordings": 13,
        "heldout_calibration_recordings": 14,
        "deployment_budget": 3,
        "local_independent_stream": "held-in-minival (S0-S5 sanity only)",
        "official_heldout_stream": "EvalAI remote test/eval (S6-S12)",
        "official_heldout_score_locally_available": False,
        "writes": 0,
        "filesystem_reads": 0,
        "nwb_files_opened": 0,
        "cuda_initialized": False,
        "training_started": False,
        "inference_started": False,
        "r2_calculated": False,
        "carrier_fitted": False,
        "decoder_packaged": False,
        "evalai_submissions": 0,
        "result_root_created": False,
    }


__all__ = (
    "ARMS", "C1_CYCLE", "DeploymentContractError", "HELDIN_SESSION_TO_FALCON_KEY",
    "HELDOUT_SESSION_TO_FALCON_KEY", "SCHEMA", "classify_local_inventory",
    "deterministic_prefix_cycle", "dry_plan", "frozen_training_contract", "session_mapping",
    "validate_balanced_cycle", "validate_decoder_payload", "validate_m3_deployment",
)
