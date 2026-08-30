"""Frozen plan for H1 variable activity exposure V1."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariableActivityTrainingPlan:
    seed: int = 42
    support_trials: int = 4
    batch_size: int = 128
    epochs: int = 5
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.0
    replay_period: int = 2
    window_size: int = 700
    max_trial_length: int = 1024
    units: int = 176
    outputs: int = 7
    behavior_scale: float = 20.0

    def __post_init__(self) -> None:
        if not (
            self.seed == 42
            and self.support_trials == 4
            and self.batch_size == 128
            and self.epochs == 5
            and self.learning_rate == 1.0e-5
            and self.weight_decay == 0.0
            and self.replay_period == 2
            and self.window_size == 700
            and self.max_trial_length == 1024
            and self.units == 176
            and self.outputs == 7
            and self.behavior_scale == 20.0
        ):
            raise ValueError("H1 variable-activity training plan drift")


TRAINING_PLAN = VariableActivityTrainingPlan()

SEALED_CHECKPOINT_RELATIVE = (
    "SPINT-main/pilot_artifacts/h1_carrierid/gpu_runs/h32_fold0_v1/full/"
    "checkpoints/fixed_epoch50/epoch_049.ckpt"
)
SEALED_CHECKPOINT_SHA256 = "f23e83c9ee8ca6c11d3c6b86410e856d906ccc8c37486aa13ae2e3a2af008fff"
SEALED_CONFIG_RELATIVE = (
    "SPINT-main/pilot_artifacts/h1_carrierid/gpu_runs/h32_fold0_v1/full/.hydra/config.yaml"
)
SEALED_CONFIG_SHA256 = "049751a0135ab707968fe9a91582a88bbf726834279893a758204b62fe3874df"
SOURCE_AUTHORITY_RELATIVE = "SPINT-main/pilot_artifacts/h1_carrierid_hu/source_authority_v1"
SOURCE_AUTHORITY_RECEIPT_SHA256 = "fcb0cf843f351715677f14e3cf80acdb613fc5a59fb96bfcf7efc23836b16fb8"
DATA_RELATIVE = "SPINT-main/data/000954"
RAW_RECEIPT_RELATIVE = (
    "sua_exploration/results/h1_m4_population_decoder_carrier_date_lodo_v1/"
    "H1_M4_POPULATION_DECODER_CARRIER_CPU_RECEIPT.json"
)
EB_RECEIPT_RELATIVE = (
    "sua_exploration/results/h1_m4_empirical_bayes_confidence_carrier_date_lodo_v1/"
    "H1_M4_EMPIRICAL_BAYES_CONFIDENCE_CARRIER_CPU_RECEIPT.json"
)


__all__ = (
    "DATA_RELATIVE",
    "EB_RECEIPT_RELATIVE",
    "RAW_RECEIPT_RELATIVE",
    "SEALED_CHECKPOINT_RELATIVE",
    "SEALED_CHECKPOINT_SHA256",
    "SEALED_CONFIG_RELATIVE",
    "SEALED_CONFIG_SHA256",
    "SOURCE_AUTHORITY_RECEIPT_SHA256",
    "SOURCE_AUTHORITY_RELATIVE",
    "TRAINING_PLAN",
    "VariableActivityTrainingPlan",
)
