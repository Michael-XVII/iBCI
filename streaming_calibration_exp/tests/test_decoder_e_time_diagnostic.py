from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def _load_diag_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_m2_decoder_e_time_structure.py"
    spec = importlib.util.spec_from_file_location("diagnose_m2_decoder_e_time_structure", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_identity_variants_mean_and_permutation_are_deterministic():
    diag = _load_diag_module()
    identity = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)

    variants_a = diag.identity_variants(identity, permutation_seed=7)
    variants_b = diag.identity_variants(identity, permutation_seed=7)

    assert torch.equal(variants_a["original"], identity)
    assert torch.allclose(
        variants_a["mean_time"],
        identity.mean(dim=-1, keepdim=True).expand_as(identity),
    )
    assert torch.equal(variants_a["permute_time"], variants_b["permute_time"])
    assert not torch.equal(variants_a["permute_time"], identity)
    assert torch.equal(
        torch.sort(variants_a["permute_time"], dim=-1).values,
        torch.sort(identity, dim=-1).values,
    )


def test_r2_accumulator_perfect_and_bad_prediction():
    diag = _load_diag_module()
    target = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]])

    perfect = diag.R2Accumulator()
    perfect.update(target, target)
    assert perfect.compute() == 1.0

    bad = diag.R2Accumulator()
    bad.update(torch.zeros_like(target), target)
    assert bad.compute() < 0.0


def test_t4_z4_side_group_returns_exact_zero_on_synthetic_m2_data():
    from src.data.falcon_datamodule import FalconDataset

    neural = np.arange(24, dtype=np.float32).reshape(6, 4)
    session = {
        "neural": neural,
        "covariates": np.zeros((6, 2), dtype=np.float32),
        "eval_mask": np.ones(6, dtype=bool),
        "trial_change": np.array([True, False, False, True, False, False]),
        "trial_target_angles": np.array([0.0, 90.0], dtype=np.float32),
    }
    dataset = FalconDataset(
        {"synthetic": session},
        {"synthetic": session},
        window_size=2,
        calibration_n_trials=2,
        random_calibration=False,
        smooth_calibration=False,
        max_trial_length=3,
        side_feature_group="t4_z4",
        side_feature_mean=np.zeros(4, dtype=np.float32),
        side_feature_std=np.ones(4, dtype=np.float32),
        task="m2",
    )

    *_, side_features = dataset[0]
    assert side_features.shape == (4, 4)
    assert np.array_equal(side_features, np.zeros((4, 4), dtype=np.float32))
