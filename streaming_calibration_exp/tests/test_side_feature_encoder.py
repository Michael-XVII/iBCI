"""Tests for B3S side-feature encoder and unit-side feature cache."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.components.streaming_encoders import (
    EarlyPoolEncoder,
    ElectrodeAnchorEarlyPoolEncoder,
    ElectrodeGateEarlyPoolEncoder,
    SameElectrodeRelationEarlyPoolEncoder,
    SideFeatureEarlyPoolEncoder,
    build_encoder,
)


@pytest.fixture
def shapes():
    return dict(batch=2, trials=5, trial_len=100, neurons=8, window=50, side_dim=3)


def _random_calib(shapes):
    return torch.randn(
        shapes["batch"],
        shapes["trials"],
        shapes["trial_len"],
        shapes["neurons"],
    )


def _random_side(shapes):
    return torch.randn(shapes["batch"], shapes["neurons"], shapes["side_dim"])


def _repo_sample_nwb() -> Path:
    path = (
        Path(__file__).resolve().parents[2]
        / "sua_exploration/data/dandi_000688/sub-C/sub-C_ses-CO-20151103_behavior+ecephys.nwb"
    )
    if not path.is_file():
        pytest.skip("DANDI 000688 sample session not available locally")
    return path


def test_b3s_side_dim_zero_matches_b3(shapes):
    calib = _random_calib(shapes)
    b3 = EarlyPoolEncoder(shapes["trial_len"], shapes["window"], 64)
    b3s = SideFeatureEarlyPoolEncoder(shapes["trial_len"], shapes["window"], 64, side_dim=0)
    b3s.load_state_dict(b3.state_dict(), strict=True)
    b3.eval()
    b3s.eval()
    with torch.no_grad():
        out_b3 = b3.forward_batch(calib)
        out_b3s = b3s.forward_batch(calib)
    assert out_b3.shape == out_b3s.shape == (shapes["batch"], shapes["neurons"], shapes["window"])
    assert torch.allclose(out_b3, out_b3s)


def test_b3s_zero_init_matches_b3_with_side_features(shapes):
    calib = _random_calib(shapes)
    side = _random_side(shapes)
    hidden_dim = 64
    b3 = EarlyPoolEncoder(shapes["trial_len"], shapes["window"], hidden_dim)
    b3s = SideFeatureEarlyPoolEncoder(
        shapes["trial_len"], shapes["window"], hidden_dim, side_dim=shapes["side_dim"]
    )
    b3s.pre_pool.load_state_dict(b3.pre_pool.state_dict())
    with torch.no_grad():
        b3s.post_pool[0].weight[:, :hidden_dim].copy_(b3.post_pool[0].weight)
        b3s.post_pool[0].bias.copy_(b3.post_pool[0].bias)
    for b3_layer, b3s_layer in zip(b3.post_pool[2:], b3s.post_pool[2:]):
        b3s_layer.load_state_dict(b3_layer.state_dict())
    b3.eval()
    b3s.eval()
    with torch.no_grad():
        out_b3 = b3.forward_batch(calib)
        out_b3s = b3s.forward_batch(calib, side_features=side)
    assert out_b3.shape == out_b3s.shape
    assert torch.allclose(out_b3, out_b3s)


def test_b3s_output_shape_independent_of_side_dim(shapes):
    calib = _random_calib(shapes)
    for side_dim in (0, 3, 6):
        encoder = SideFeatureEarlyPoolEncoder(
            shapes["trial_len"], shapes["window"], 64, side_dim=side_dim
        )
        encoder.eval()
        side = None if side_dim == 0 else torch.randn(shapes["batch"], shapes["neurons"], side_dim)
        with torch.no_grad():
            out = encoder.forward_batch(calib, side_features=side)
        assert out.shape == (shapes["batch"], shapes["neurons"], shapes["window"])


def test_b3s_rejects_mismatched_side_dim(shapes):
    calib = _random_calib(shapes)
    encoder = SideFeatureEarlyPoolEncoder(
        shapes["trial_len"], shapes["window"], 64, side_dim=shapes["side_dim"]
    )
    encoder.eval()
    bad_side = torch.randn(shapes["batch"], shapes["neurons"], shapes["side_dim"] + 1)
    with pytest.raises(ValueError, match="last dim"):
        encoder.forward_batch(calib, side_features=bad_side)


def test_b3s_permutation_invariance(shapes):
    calib = _random_calib(shapes)
    side = _random_side(shapes)
    encoder = SideFeatureEarlyPoolEncoder(
        shapes["trial_len"], shapes["window"], 64, side_dim=shapes["side_dim"]
    )
    encoder.eval()
    perm = torch.randperm(shapes["neurons"])
    calib_perm = calib[..., perm]
    side_perm = side[:, perm, :]
    with torch.no_grad():
        base = encoder.forward_batch(calib, side_features=side)
        permuted = encoder.forward_batch(calib_perm, side_features=side_perm)
    assert torch.allclose(base[:, perm, :], permuted, atol=1e-6)


def test_side_feature_cache_is_deterministic_and_keyed(tmp_path):
    pytest.importorskip("pynwb")
    from mc_maze.unit_side_features import (
        _side_feature_cache_path,
        compute_unit_side_features_uncached,
        load_unit_side_features,
    )

    repo_data = _repo_sample_nwb()
    nwb_path = tmp_path / repo_data.name
    import shutil

    shutil.copy(repo_data, nwb_path)

    raw_a, _ = compute_unit_side_features_uncached(nwb_path, feature_group="f1", pool_size=50)
    raw_b, _ = compute_unit_side_features_uncached(nwb_path, feature_group="f1", pool_size=50)
    assert np.allclose(raw_a, raw_b)

    mean = raw_a.mean(axis=0)
    std = raw_a.std(axis=0)
    std[std < 1e-8] = 1.0
    cache_dir = tmp_path / "cache"
    norm_a, _ = load_unit_side_features(
        nwb_path, feature_group="f1", pool_size=50, mean=mean, std=std, cache_dir=cache_dir
    )
    norm_b, _ = load_unit_side_features(
        nwb_path, feature_group="f1", pool_size=50, mean=mean, std=std, cache_dir=cache_dir
    )
    assert np.allclose(norm_a, norm_b)

    path_f1 = _side_feature_cache_path(
        cache_dir, nwb_path, feature_group="f1", pool_size=50, bin_size_ms=20, window_size=50, trial_result_filter="R"
    )
    path_f2 = _side_feature_cache_path(
        cache_dir, nwb_path, feature_group="f2", pool_size=50, bin_size_ms=20, window_size=50, trial_result_filter="R"
    )
    path_pool10 = _side_feature_cache_path(
        cache_dir, nwb_path, feature_group="f1", pool_size=10, bin_size_ms=20, window_size=50, trial_result_filter="R"
    )
    assert path_f1 != path_f2
    assert path_f1 != path_pool10
    assert path_f1.is_file()
    assert not path_f2.is_file()
    assert not path_pool10.is_file()


def test_side_feature_pool_size_changes_features():
    pytest.importorskip("pynwb")
    from mc_maze.unit_side_features import compute_unit_side_features_uncached

    nwb_path = _repo_sample_nwb()
    small, _ = compute_unit_side_features_uncached(nwb_path, feature_group="f1", pool_size=10)
    large, _ = compute_unit_side_features_uncached(nwb_path, feature_group="f1", pool_size=50)
    assert not np.allclose(small, large)


def test_side_feature_pool_end_matches_datamodule():
    pytest.importorskip("pynwb")
    from mc_maze.multisession_datamodule import calibration_pool_end_time
    from mc_maze.unit_side_features import compute_unit_side_features_uncached

    nwb_path = _repo_sample_nwb()
    pool_end = calibration_pool_end_time(
        nwb_path, pool_size=50, bin_size_ms=20, window_size=50, trial_result_filter="R"
    )
    features, _ = compute_unit_side_features_uncached(
        nwb_path,
        feature_group="f1",
        pool_size=50,
        pool_end_time=pool_end,
    )
    assert features.shape[1] == 3


def test_build_encoder_b3s_registered():
    enc = build_encoder("B3S", window_size=50, trial_length=100, hidden_dim=64, side_dim=3)
    assert enc.variant == "B3S"
    assert enc.side_dim == 3


def test_same_electrode_relation_zero_init_matches_plain_t4_and_is_permutation_equivariant():
    torch.manual_seed(101)
    calib = torch.randn(2, 4, 100, 6)
    side = torch.randn(2, 6, 4)
    memberships = torch.tensor([[3, 3, 8, 9, 9, 9], [2, 7, 7, 7, 1, 1]])
    t4 = SideFeatureEarlyPoolEncoder(100, 50, 64, side_dim=4)
    relation = SameElectrodeRelationEarlyPoolEncoder(100, 50, 64, side_dim=4)
    relation.load_state_dict(t4.state_dict(), strict=False)
    # New relation head is zero-init, while the shared T4 substrate is exact.
    relation.eval(); t4.eval()
    with torch.no_grad():
        expected = t4.forward_batch(calib, side_features=side)
        observed = relation.forward_batch(calib, side_features=side, electrode_ids=memberships)
    assert torch.equal(observed, expected)
    order = torch.tensor([5, 1, 3, 0, 4, 2])
    with torch.no_grad():
        permuted = relation.forward_batch(
            calib[..., order], side_features=side[:, order], electrode_ids=memberships[:, order]
        )
    assert torch.allclose(permuted, observed[:, order], atol=1e-6)


def test_same_electrode_relation_singleton_boundary_survives_optimizer_step():
    torch.manual_seed(102)
    encoder = SameElectrodeRelationEarlyPoolEncoder(100, 50, 32, side_dim=4)
    calib = torch.randn(1, 3, 100, 5)
    side = torch.randn(1, 5, 4)
    grouped = torch.tensor([[0, 0, 1, 1, 1]])
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-2)
    loss = encoder.forward_batch(calib, side_features=side, electrode_ids=grouped).square().mean()
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert not torch.equal(encoder.relation_output.weight, torch.zeros_like(encoder.relation_output.weight))
    singletons = torch.arange(5).view(1, 5)
    with torch.no_grad():
        relation = encoder.forward_batch(calib, side_features=side, electrode_ids=singletons)
        substrate = SideFeatureEarlyPoolEncoder(100, 50, 32, side_dim=4)
        substrate.load_state_dict(encoder.state_dict(), strict=False)
        expected = substrate.forward_batch(calib, side_features=side)
    assert torch.equal(relation, expected)


def test_same_electrode_no_group_is_parameter_matched_and_ignores_memberships():
    grouped = SameElectrodeRelationEarlyPoolEncoder(100, 50, 32, side_dim=4, use_group_relation=True)
    no_group = SameElectrodeRelationEarlyPoolEncoder(100, 50, 32, side_dim=4, use_group_relation=False)
    assert sum(p.numel() for p in grouped.parameters()) == sum(p.numel() for p in no_group.parameters())
    calib, side = torch.randn(1, 2, 100, 4), torch.randn(1, 4, 4)
    with torch.no_grad():
        output = no_group.forward_batch(calib, side_features=side)
    assert output.shape == (1, 4, 50)


# ------------------------------------------------------------------------------------
# FS1/FS2 dimension-matched shuffled controls (UNIT_SIDE_FEATURE_ABLATION.md section 6,
# revised 2026-07-25). The original single 6-dim "fs" compared a 3-dim F1 against a 6-dim
# control -- two different post_pool architectures (fan_in 67 vs 70) with RNG streams that
# diverge from the first layer on. FS1 must permute F1 (3 dims); FS2 must permute F2 (6
# dims); each feature group's content gate only ever compares against its own
# dimension-matched control.
# ------------------------------------------------------------------------------------
def test_side_feature_dims_has_dimension_matched_shuffled_controls():
    from mc_maze.unit_side_features import SIDE_FEATURE_DIMS

    assert "fs" not in SIDE_FEATURE_DIMS, (
        "the single 6-dim 'fs' control was a charter defect (F1 vs a 6-dim control is not "
        "a same-architecture comparison) and must not silently reappear"
    )
    assert SIDE_FEATURE_DIMS["fs1"] == SIDE_FEATURE_DIMS["f1"] == 3
    assert SIDE_FEATURE_DIMS["fs2"] == SIDE_FEATURE_DIMS["f2"] == 6


def test_is_shuffled_control():
    from mc_maze.unit_side_features import is_shuffled_control

    assert is_shuffled_control("fs1") is True
    assert is_shuffled_control("fs2") is True
    assert is_shuffled_control("fs3") is True
    assert is_shuffled_control("f1") is False
    assert is_shuffled_control("f2") is False
    assert is_shuffled_control("f3") is False
    assert is_shuffled_control("none") is False


def test_base_feature_group_resolves_shuffled_controls_to_their_matching_dimension():
    from mc_maze.unit_side_features import base_feature_group

    assert base_feature_group("fs1") == "f1"
    assert base_feature_group("fs2") == "f2"
    assert base_feature_group("f3") == "f2"
    assert base_feature_group("fs3") == "f2"
    # Identity for real feature groups -- a control's base group is never itself permuted
    # again, and a real group is computed exactly as named.
    assert base_feature_group("f1") == "f1"
    assert base_feature_group("f2") == "f2"


def test_shuffled_controls_permute_their_matching_dimension_feature_set():
    """FS1/FS2 must be a permutation (same rows, different order) of the normalized F1/F2
    values -- never a different distribution, and never a mix of the wrong dimension."""
    pytest.importorskip("pynwb")
    from mc_maze.unit_side_features import (
        SIDE_FEATURE_DIMS,
        base_feature_group,
        compute_unit_side_features_uncached,
        load_unit_side_features,
    )

    nwb_path = _repo_sample_nwb()
    for control in ("fs1", "fs2"):
        base = base_feature_group(control)
        raw, _ = compute_unit_side_features_uncached(nwb_path, feature_group=base, pool_size=50)
        assert raw.shape[1] == SIDE_FEATURE_DIMS[control]

        mean = raw.mean(axis=0)
        std = raw.std(axis=0)
        std[std < 1e-8] = 1.0

        unpermuted, _ = load_unit_side_features(
            nwb_path, feature_group=base, pool_size=50, mean=mean, std=std, permutation_seed=None,
        )
        permuted, _ = load_unit_side_features(
            nwb_path, feature_group=base, pool_size=50, mean=mean, std=std, permutation_seed=1234,
        )

        assert permuted.shape == unpermuted.shape == (raw.shape[0], SIDE_FEATURE_DIMS[control])
        # A real permutation (with >1 unit) changes the per-unit assignment...
        assert not np.allclose(permuted, unpermuted)
        # ...but leaves the multiset of per-unit feature rows unchanged (sorting each
        # column recovers the same values -- this is a shuffle, not a different feature
        # computation reached via the wrong base group).
        assert np.allclose(np.sort(permuted, axis=0), np.sort(unpermuted, axis=0))


def test_fs1_and_fs2_permute_independently_different_base_groups():
    """FS1 (based on F1, 3 dims) and FS2 (based on F2, 6 dims) must not collapse onto the
    same feature content merely because both are "shuffled controls" -- FS1's 3 columns
    must equal F1's first three raw feature columns (p2p, noise_std, snr) and FS2 must
    equal all 6 of F2's, dimension-matched to their own base group only."""
    pytest.importorskip("pynwb")
    from mc_maze.unit_side_features import FEATURE_GROUPS, compute_unit_side_features_uncached

    nwb_path = _repo_sample_nwb()
    raw_f1, _ = compute_unit_side_features_uncached(nwb_path, feature_group="f1", pool_size=50)
    raw_f2, _ = compute_unit_side_features_uncached(nwb_path, feature_group="f2", pool_size=50)

    assert raw_f1.shape[1] == 3
    assert raw_f2.shape[1] == 6
    assert FEATURE_GROUPS["f1"] == ("p2p", "noise_std", "snr")
    # F1's three columns are exactly F2's first three (both start from the same template
    # scalars) -- confirms FS1 and FS2 are genuinely dimension-matched to distinct content,
    # not two views of an unrelated computation.
    assert np.allclose(raw_f1, raw_f2[:, :3])


# ------------------------------------------------------------------------------------
# E3 directional tuning features: T4 (cosine-tuning fit) / T8 (per-direction mean rate),
# with dimension-matched shuffled controls TS4/TS8 (E3_E4_ENCODER_PROGRAM.md section 1,
# UNIT_SIDE_FEATURE_ABLATION.md section 6 for the general FSx pattern this extends).
# ------------------------------------------------------------------------------------

def _sua_scripts_path() -> Path:
    return Path(__file__).resolve().parents[2] / "sua_exploration" / "scripts"


# ---- registry / constants (no I/O) --------------------------------------------------

def test_side_feature_dims_includes_tuning_groups():
    from mc_maze.unit_side_features import SIDE_FEATURE_DIMS

    assert SIDE_FEATURE_DIMS["t4"] == SIDE_FEATURE_DIMS["ts4"] == 4
    assert SIDE_FEATURE_DIMS["t8"] == SIDE_FEATURE_DIMS["ts8"] == 8


def test_tuning_shuffled_control_registry():
    from mc_maze.unit_side_features import base_feature_group, is_shuffled_control

    assert is_shuffled_control("ts4") is True
    assert is_shuffled_control("ts8") is True
    assert is_shuffled_control("t4") is False
    assert is_shuffled_control("t8") is False
    assert base_feature_group("ts4") == "t4"
    assert base_feature_group("ts8") == "t8"
    assert base_feature_group("t4") == "t4"
    assert base_feature_group("t8") == "t8"


def test_known_feature_groups_covers_waveform_and_tuning_real_groups():
    """fit_side_feature_stats/load_unit_side_features gate on this combined registry (not
    FEATURE_GROUPS alone, which is deliberately waveform-only -- see the module comment next
    to TUNING_FEATURE_NAMES) so t4/t8 pass the same validity check f1/f2 already did."""
    from mc_maze.unit_side_features import (
        FEATURE_GROUPS,
        KNOWN_FEATURE_GROUPS,
        TEMPLATE_RIDGE_FEATURE_NAMES,
        TUNING_FEATURE_NAMES,
    )

    assert KNOWN_FEATURE_GROUPS == (
        frozenset(FEATURE_GROUPS)
        | frozenset(TUNING_FEATURE_NAMES)
        | frozenset(TEMPLATE_RIDGE_FEATURE_NAMES)
        | frozenset({"f3"})
    )
    assert {"f1", "f2", "f3", "t4", "t8"}.issubset(KNOWN_FEATURE_GROUPS)
    # Shuffled-control tokens are never "known" feature groups in their own right -- they
    # only ever reach compute_unit_side_features_uncached/fit_side_feature_stats after
    # base_feature_group() resolves them to a real group.
    assert "fs1" not in KNOWN_FEATURE_GROUPS
    assert "fs3" not in KNOWN_FEATURE_GROUPS
    assert "ts4" not in KNOWN_FEATURE_GROUPS


def test_canonical_directions_are_fixed_and_match_the_dataset():
    """8 directions, 45 degrees apart, ascending -- and must equal the exact target_dir
    values ROADMAP.md/E3_E4_ENCODER_PROGRAM.md report for this dataset's center-out task."""
    from mc_maze.unit_side_features import CANONICAL_DIRECTIONS_RAD, TUNING_NUM_DIRECTIONS

    assert len(CANONICAL_DIRECTIONS_RAD) == TUNING_NUM_DIRECTIONS == 8
    assert list(CANONICAL_DIRECTIONS_RAD) == sorted(CANONICAL_DIRECTIONS_RAD)
    diffs = np.diff(CANONICAL_DIRECTIONS_RAD)
    assert np.allclose(diffs, np.pi / 4.0, atol=1e-9)
    # Observed 2026-07-25 against sub-C ses-CO-20151103 (UNIT_SIDE_FEATURE_ABLATION.md /
    # E3_E4_ENCODER_PROGRAM.md): these are the exact 8 unique target_dir values.
    observed = sorted(
        [-2.356194490192345, -1.5707963267948966, -0.7853981633974483, 0.0,
         0.7853981633974483, 1.5707963267948966, 2.356194490192345, 3.141592653589793]
    )
    assert np.allclose(sorted(CANONICAL_DIRECTIONS_RAD), observed, atol=1e-9)


def test_nearest_canonical_direction_index_handles_wrap_and_noise():
    from mc_maze.unit_side_features import CANONICAL_DIRECTIONS_RAD, _nearest_canonical_direction_index

    for index, direction in enumerate(CANONICAL_DIRECTIONS_RAD):
        assert _nearest_canonical_direction_index(direction) == index
        # Small perturbation still snaps to the same bin.
        assert _nearest_canonical_direction_index(direction + 1e-4) == index
    # +pi and -pi are the same physical angle; -pi must snap to the pi bin (index 7), not
    # wrap around to something else, exercising the circular-distance handling.
    pi_index = CANONICAL_DIRECTIONS_RAD.index(max(CANONICAL_DIRECTIONS_RAD))
    assert _nearest_canonical_direction_index(-np.pi) == pi_index


# ---- pure math: cosine-tuning fit and per-unit classification (no I/O) --------------

def test_fit_cosine_tuning_recovers_known_curve_exactly():
    """Noiseless synthetic case: an exact 8-point cosine curve must be recovered to numerical
    precision, and the returned (a, c, m, b) must literally be
    (m*cos(phi), m*sin(phi), m, b) -- this is the T4 emission contract."""
    from mc_maze.unit_side_features import CANONICAL_DIRECTIONS_RAD, _fit_cosine_tuning

    true_b, true_m, true_phi = 5.0, 3.0, 0.6
    thetas = np.asarray(CANONICAL_DIRECTIONS_RAD)
    rates = true_b + true_m * np.cos(thetas - true_phi)
    a, c, m, b = _fit_cosine_tuning(thetas, rates)
    assert np.isclose(b, true_b, atol=1e-6)
    assert np.isclose(m, true_m, atol=1e-6)
    assert np.isclose(a, true_m * np.cos(true_phi), atol=1e-6)
    assert np.isclose(c, true_m * np.sin(true_phi), atol=1e-6)


def test_fit_cosine_tuning_flat_rate_gives_zero_modulation():
    from mc_maze.unit_side_features import CANONICAL_DIRECTIONS_RAD, _fit_cosine_tuning

    thetas = np.asarray(CANONICAL_DIRECTIONS_RAD)
    rates = np.full(8, 7.5)
    a, c, m, b = _fit_cosine_tuning(thetas, rates)
    assert np.isclose(m, 0.0, atol=1e-9)
    assert np.isclose(b, 7.5, atol=1e-9)


def test_fit_cosine_tuning_two_directions_stays_finite():
    """K=2 is the smallest input the orchestrator ever actually calls this with (the <2
    degeneracy gate lives in the caller, E3_E4_ENCODER_PROGRAM.md section 1.4); the
    underdetermined 2-equation/3-unknown system must still produce a finite, deterministic
    minimum-norm answer, never NaN/inf or an exception."""
    from mc_maze.unit_side_features import _fit_cosine_tuning

    thetas = np.asarray([0.0, np.pi / 2.0])
    rates = np.asarray([4.0, 6.0])
    a, c, m, b = _fit_cosine_tuning(thetas, rates)
    for value in (a, c, m, b):
        assert np.isfinite(value)
    # Deterministic: repeated calls agree exactly.
    a2, c2, m2, b2 = _fit_cosine_tuning(thetas, rates)
    assert (a, c, m, b) == (a2, c2, m2, b2)


def test_unit_tuning_features_zero_spike_gets_exact_zero_fill():
    from mc_maze.unit_side_features import _unit_tuning_features

    trial_rates = np.zeros(6)
    direction_indices = np.array([0, 1, 2, 3, 0, 1])
    t4, t8, is_zero_spike, is_zero_modulation = _unit_tuning_features(
        trial_rates, direction_indices, present_directions=[0, 1, 2, 3]
    )
    assert is_zero_spike is True
    assert is_zero_modulation is False
    assert np.array_equal(t4, np.zeros(4, dtype=np.float32))
    assert np.array_equal(t8, np.zeros(8, dtype=np.float32))


def test_unit_tuning_features_ignores_trials_with_unknown_direction():
    """direction_indices == -1 (no usable target_dir) must never be silently folded into
    direction bin 0 or any real bin."""
    from mc_maze.unit_side_features import _unit_tuning_features

    trial_rates = np.array([10.0, 10.0, 10.0, 10.0, 999.0])
    direction_indices = np.array([0, 1, 2, 3, -1])
    t4, t8, is_zero_spike, is_zero_modulation = _unit_tuning_features(
        trial_rates, direction_indices, present_directions=[0, 1, 2, 3]
    )
    assert is_zero_spike is False
    assert is_zero_modulation is True  # flat 10.0 rate across all 4 present directions
    assert np.isclose(t8[0], 10.0) and np.isclose(t8[1], 10.0)
    assert t8[4] == 0.0 and t8[5] == 0.0  # directions never present stay at the fixed fill
    assert np.isfinite(t4).all()


def test_unit_tuning_features_direction_specific_unit_recovers_preferred_direction():
    from mc_maze.unit_side_features import CANONICAL_DIRECTIONS_RAD, _unit_tuning_features

    present_directions = list(range(8))
    thetas = np.asarray(CANONICAL_DIRECTIONS_RAD)
    true_phi = thetas[3]
    trial_rates = 5.0 + 4.0 * np.cos(thetas - true_phi)
    direction_indices = np.arange(8)
    t4, t8, is_zero_spike, is_zero_modulation = _unit_tuning_features(
        trial_rates, direction_indices, present_directions
    )
    assert is_zero_spike is False
    assert is_zero_modulation is False
    recovered_phi = float(np.arctan2(t4[1], t4[0]))
    assert np.isclose(recovered_phi, true_phi, atol=1e-4)
    assert np.allclose(t8, trial_rates, atol=1e-5)


# ---- real-session behavioral tests (I/O; the mandatory leakage test lives here) -----

def test_tuning_feature_pool_size_changes_features():
    """The mandatory leakage-discipline test (analogous to
    test_side_feature_pool_size_changes_features above): computing with pool_size=10 vs
    pool_size=50 must produce different T4 features on real data. A test that only checks
    an assertion/helper function in isolation does not demonstrate this -- this test goes
    through the full compute_unit_side_features_uncached -> list_datamodule_rewarded_trials
    -> real NWB spike read path."""
    pytest.importorskip("pynwb")
    from mc_maze.unit_side_features import compute_unit_side_features_uncached

    nwb_path = _repo_sample_nwb()
    small, _ = compute_unit_side_features_uncached(nwb_path, feature_group="t4", pool_size=10)
    large, _ = compute_unit_side_features_uncached(nwb_path, feature_group="t4", pool_size=50)
    assert not np.allclose(small, large)
    small8, _ = compute_unit_side_features_uncached(nwb_path, feature_group="t8", pool_size=10)
    large8, _ = compute_unit_side_features_uncached(nwb_path, feature_group="t8", pool_size=50)
    assert not np.allclose(small8, large8)


def test_tuning_features_finite_shape_and_internal_consistency_on_real_session():
    pytest.importorskip("pynwb")
    from mc_maze.unit_side_features import compute_unit_side_features_uncached

    nwb_path = _repo_sample_nwb()
    t4, meta4 = compute_unit_side_features_uncached(nwb_path, feature_group="t4", pool_size=50)
    t8, meta8 = compute_unit_side_features_uncached(nwb_path, feature_group="t8", pool_size=50)
    assert t4.shape[1] == 4 and t8.shape[1] == 8
    # Rows are per-unit (one feature row per recorded unit), not per pool trial -- pool_size
    # only controls which trials are read, not the output row count.
    assert t4.shape[0] == t8.shape[0] > 0
    assert meta4.pool_size == meta8.pool_size == 50
    assert np.isfinite(t4).all() and np.isfinite(t8).all()
    # a = m*cos(phi), c = m*sin(phi) by construction -> m must equal hypot(a, c) for every unit.
    assert np.allclose(np.hypot(t4[:, 0], t4[:, 1]), t4[:, 2], atol=1e-3)
    # This session has full 8-direction coverage well within a 50-trial pool (ROADMAP.md), so
    # no degenerate units are expected here; the degenerate *paths* themselves are covered by
    # the pure _unit_tuning_features tests and the pool_size=1 test below.
    assert meta4.insufficient_direction_unit_count == 0


def test_tuning_features_pool_size_one_triggers_session_wide_insufficient_direction():
    """With only 1 pool trial there is at most 1 distinct direction, which is exactly the
    '<2 distinct directions' degeneracy from E3_E4_ENCODER_PROGRAM.md section 1.4: every
    unit must get the fixed all-zero fill and be counted, never NaN/inf and never a
    per-unit lstsq minimum-norm guess."""
    pytest.importorskip("pynwb")
    from mc_maze.unit_side_features import compute_unit_side_features_uncached

    nwb_path = _repo_sample_nwb()
    t4, meta = compute_unit_side_features_uncached(nwb_path, feature_group="t4", pool_size=1)
    assert meta.insufficient_direction_unit_count == t4.shape[0]
    assert meta.degenerate_unit_count == t4.shape[0]
    assert np.array_equal(t4, np.zeros_like(t4))
    assert np.isfinite(t4).all()


def test_ts4_ts8_permute_their_matching_dimension_tuning_features():
    """Same contract as test_shuffled_controls_permute_their_matching_dimension_feature_set
    above, extended to the tuning groups: TS4/TS8 must be a permutation of the normalized
    T4/T8 values, never a different distribution."""
    pytest.importorskip("pynwb")
    from mc_maze.unit_side_features import (
        SIDE_FEATURE_DIMS,
        base_feature_group,
        compute_unit_side_features_uncached,
        load_unit_side_features,
    )

    nwb_path = _repo_sample_nwb()
    for control in ("ts4", "ts8"):
        base = base_feature_group(control)
        raw, _ = compute_unit_side_features_uncached(nwb_path, feature_group=base, pool_size=50)
        assert raw.shape[1] == SIDE_FEATURE_DIMS[control]

        mean = raw.mean(axis=0)
        std = raw.std(axis=0)
        std[std < 1e-8] = 1.0

        unpermuted, _ = load_unit_side_features(
            nwb_path, feature_group=base, pool_size=50, mean=mean, std=std, permutation_seed=None,
        )
        permuted, _ = load_unit_side_features(
            nwb_path, feature_group=base, pool_size=50, mean=mean, std=std, permutation_seed=2024,
        )
        assert not np.allclose(permuted, unpermuted)
        assert np.allclose(np.sort(permuted, axis=0), np.sort(unpermuted, axis=0))


def test_tuning_feature_train_only_normalization_is_zero_mean_on_train_session():
    """fit_side_feature_stats/_fit_robust_stats (unmodified by this task) clip to the 1st/99th
    percentile before averaging, so even fitting and normalizing the same session's data does
    not zero the mean exactly -- a few unclipped tail points pull it slightly (this is a
    pre-existing property of the shared robust-stats helper, not specific to T4). atol=0.05 is
    loose enough to absorb that small clipping bias while still catching a real bug (e.g.
    normalization not applied at all, which would leave raw Hz-scale means of order 1-10)."""
    pytest.importorskip("pynwb")
    from mc_maze.unit_side_features import fit_side_feature_stats, load_unit_side_features

    nwb_path = _repo_sample_nwb()
    mean, std = fit_side_feature_stats([nwb_path], feature_group="t4", pool_size=50)
    normalized, _ = load_unit_side_features(
        nwb_path, feature_group="t4", pool_size=50, mean=mean, std=std
    )
    assert np.allclose(normalized.mean(axis=0), 0.0, atol=0.05)


def test_old_format_side_feature_cache_self_heals_instead_of_crashing(tmp_path):
    """Backward compatibility: caches written before zero_modulation_unit_count /
    insufficient_direction_unit_count existed (e.g. any already-computed f1/f2 cache on
    disk from side_feature_ablation_v2) must not crash load_unit_side_features when read
    back under the new SideFeatureMetadata schema -- the existing except (KeyError, ...)
    fallback must discard and recompute them exactly as it already does for any other
    unreadable/incompatible cache entry."""
    pytest.importorskip("pynwb")
    from mc_maze.unit_side_features import _side_feature_cache_path, load_unit_side_features
    from mc_maze.multisession_datamodule import _write_npz_atomically

    nwb_path = _repo_sample_nwb()
    cache_dir = tmp_path / "cache"
    cache_path = _side_feature_cache_path(
        cache_dir, nwb_path, feature_group="t4", pool_size=50,
        bin_size_ms=20, window_size=50, trial_result_filter="R",
    )
    _write_npz_atomically(
        cache_path,
        features=np.zeros((38, 4), dtype=np.float32),
        feature_group=np.asarray("t4"),
        feature_version=np.asarray(1),
        pool_size=np.asarray(50),
        cache_key=np.asarray("stale"),
        degenerate_unit_count=np.asarray(0),
        zero_spike_unit_count=np.asarray(0),
        single_spike_unit_count=np.asarray(0),
        zero_noise_std_unit_count=np.asarray(0),
        zero_template_max_unit_count=np.asarray(0),
        # zero_modulation_unit_count / insufficient_direction_unit_count deliberately absent.
    )
    mean = np.zeros(4, dtype=np.float32)
    std = np.ones(4, dtype=np.float32)
    normalized, metadata = load_unit_side_features(
        nwb_path, feature_group="t4", pool_size=50, mean=mean, std=std, cache_dir=cache_dir
    )
    assert normalized.shape == (38, 4)
    assert metadata.cache_key != "stale"  # proves this was recomputed, not the stale hit
    with np.load(cache_path, allow_pickle=False) as rewritten:
        assert "zero_modulation_unit_count" in rewritten.files
        assert "insufficient_direction_unit_count" in rewritten.files


# ---- B3S encoder wiring at T4/T8 widths (no NWB I/O) ---------------------------------

def test_b3s_forward_pass_at_tuning_feature_widths():
    """UNIT_SIDE_FEATURE_ABLATION.md section 5 / task instructions: side_dim is generic, so
    B3S must already work unchanged at side_dim=4 (T4) and side_dim=8 (T8) with no encoder
    changes -- this is a direct verification, not an assumption."""
    for side_dim in (4, 8):
        encoder = build_encoder("B3S", window_size=50, trial_length=100, hidden_dim=64, side_dim=side_dim)
        encoder.eval()
        calib = torch.randn(2, 5, 100, 6)
        side = torch.randn(2, 6, side_dim)
        with torch.no_grad():
            out = encoder.forward_batch(calib, side_features=side)
        assert out.shape == (2, 6, 50)


# ---- datamodule + eval-path integration (I/O; "verify, don't assume" for E3 wiring) --

def test_datamodule_wires_tuning_side_features_end_to_end(tmp_path):
    """Verifies (not assumes) that Dandi688MultiSessionDataModule -- unchanged by this task
    -- already produces correctly-shaped T4/TS8 side_features batches once
    mc_maze.unit_side_features knows about t4/t8/ts4/ts8, exactly as claimed for F1/F2.
    Two real sessions copied into an isolated temp data_dir. Both are sub-C/CO/27-6-6
    *validation* sessions per side_feature_ablation_v2's session_splits (never one of the 6
    held-out test sessions -- ses-CO-20151103 and ses-CO-20151104 are not in that split's
    session_splits["test"]; deliberately not ses-CO-20151119, which is), single dataloader
    worker, CPU only."""
    pytest.importorskip("pynwb")
    import shutil

    from mc_maze.multisession_datamodule import Dandi688MultiSessionDataModule

    src_dir = Path(__file__).resolve().parents[2] / "sua_exploration/data/dandi_000688/sub-C"
    files = [
        src_dir / "sub-C_ses-CO-20151103_behavior+ecephys.nwb",
        src_dir / "sub-C_ses-CO-20151104_behavior+ecephys.nwb",
    ]
    for f in files:
        if not f.is_file():
            pytest.skip("DANDI 000688 sample sessions not available locally")

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for f in files:
        shutil.copy(f, data_dir / f.name)

    dm = Dandi688MultiSessionDataModule(
        data_dir=str(data_dir), task="CO", split_counts=(1, 1, 0),
        batch_size=2, window_size=50, calibration_n_trials=10, max_trial_length=100,
        bin_size_ms=20, num_workers=0, cache_dir=str(tmp_path / "cache"),
        side_feature_group="t4", side_feature_pool_size=50,
    )
    dm.setup("fit")
    neural, behavior, calib, session_name, side = next(iter(dm.train_dataloader()))
    assert side.shape[0] == neural.shape[0]
    assert side.shape[-1] == 4
    assert side.shape[1] == calib.shape[-1]  # one side-feature row per neuron/unit
    assert torch.isfinite(side).all()

    dm_ts = Dandi688MultiSessionDataModule(
        data_dir=str(data_dir), task="CO", split_counts=(1, 1, 0),
        batch_size=2, window_size=50, calibration_n_trials=10, max_trial_length=100,
        bin_size_ms=20, num_workers=0, cache_dir=str(tmp_path / "cache"),
        side_feature_group="ts8", side_feature_pool_size=50, side_permutation_seed=42,
    )
    dm_ts.setup("fit")
    _, _, _, _, side_ts8 = next(iter(dm_ts.train_dataloader()))
    assert side_ts8.shape[-1] == 8
    assert torch.isfinite(side_ts8).all()


def test_eval_path_attaches_tuning_side_features_from_run_metadata(tmp_path):
    """Verifies (not assumes) the exact eval_adaptation_dandi688.py helpers
    select_gradient_free_protocol_dandi688.py / eval_epoch_window_dandi688.py reuse
    (load_side_feature_stats_for_run_metadata + attach_side_features) already work for
    t4/ts4 given only a run_metadata["side_features"]["group"] string -- no checkpoint,
    no GPU, no training, matching the hard constraint against touching
    eval_epoch_window_dandi688.py itself while still exercising the code path it calls."""
    pytest.importorskip("pynwb")
    pytest.importorskip("torchmetrics")
    import shutil

    scripts_path = str(_sua_scripts_path())
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    import eval_adaptation_dandi688 as eval_mod

    nwb_path = _repo_sample_nwb()
    session_copy = tmp_path / nwb_path.name
    shutil.copy(nwb_path, session_copy)
    cache_dir = tmp_path / "cache"

    behavior_mean = np.zeros(2, dtype=np.float32)
    behavior_std = np.ones(2, dtype=np.float32)
    rec = eval_mod.load_session_with_trials(
        session_copy, 20, eval_mod.WINDOW_SIZE, 10, eval_mod.TRIAL_LENGTH, eval_mod.PAD_VALUE,
        behavior_mean, behavior_std, cache_dir=cache_dir,
    )

    run_metadata = {"side_features": {"group": "t4", "pool_size": 50, "permutation_seed": None}}
    resolved = eval_mod.load_side_feature_stats_for_run_metadata(run_metadata, [session_copy], cache_dir)
    assert resolved is not None
    side_feature_group, waveform_feature_group, pool_size, permutation_seed, mean, std = resolved
    assert side_feature_group == "t4" and waveform_feature_group == "t4"
    assert pool_size == 50 and permutation_seed is None

    attached = eval_mod.attach_side_features(
        rec,
        session_copy,
        side_feature_group=side_feature_group,
        waveform_feature_group=waveform_feature_group,
        pool_size=pool_size,
        permutation_seed=permutation_seed,
        mean=mean,
        std=std,
        cache_dir=cache_dir,
    )
    assert attached["side_features"].shape == (rec["n_units"], 4)

    run_metadata_ts = {"side_features": {"group": "ts4", "pool_size": 50, "permutation_seed": 42}}
    resolved_ts = eval_mod.load_side_feature_stats_for_run_metadata(run_metadata_ts, [session_copy], cache_dir)
    side_feature_group_ts, waveform_feature_group_ts, pool_size_ts, permutation_seed_ts, mean_ts, std_ts = resolved_ts
    attached_ts = eval_mod.attach_side_features(
        rec,
        session_copy,
        side_feature_group=side_feature_group_ts,
        waveform_feature_group=waveform_feature_group_ts,
        pool_size=pool_size_ts,
        permutation_seed=permutation_seed_ts,
        mean=mean_ts,
        std=std_ts,
        cache_dir=cache_dir,
    )
    assert not np.allclose(attached_ts["side_features"], attached["side_features"])

    # run_metadata["side_features"]["group"] == "none" must remain a no-op (every existing
    # non-B3S / B3S-none checkpoint's eval path is unaffected by this task).
    assert eval_mod.load_side_feature_stats_for_run_metadata(
        {"side_features": {"group": "none"}}, [session_copy], cache_dir
    ) is None


def test_b3s_f3_zero_init_matches_b3_with_electrode_embedding(shapes):
    """F3's zero-init embed + zero-init post_pool side columns must match B3 at step 0."""
    calib = _random_calib(shapes)
    side_dim = 6
    embed_dim = 8
    num_electrodes = 12
    side = torch.randn(shapes["batch"], shapes["neurons"], side_dim)
    electrode_ids = torch.randint(0, num_electrodes, (shapes["batch"], shapes["neurons"]))
    hidden_dim = 64
    b3 = EarlyPoolEncoder(shapes["trial_len"], shapes["window"], hidden_dim)
    b3s = SideFeatureEarlyPoolEncoder(
        shapes["trial_len"],
        shapes["window"],
        hidden_dim,
        side_dim=side_dim,
        electrode_embed_dim=embed_dim,
        num_electrodes=num_electrodes,
    )
    b3s.pre_pool.load_state_dict(b3.pre_pool.state_dict())
    with torch.no_grad():
        b3s.post_pool[0].weight[:, :hidden_dim].copy_(b3.post_pool[0].weight)
        b3s.post_pool[0].bias.copy_(b3.post_pool[0].bias)
    for b3_layer, b3s_layer in zip(b3.post_pool[2:], b3s.post_pool[2:]):
        b3s_layer.load_state_dict(b3_layer.state_dict())
    b3.eval()
    b3s.eval()
    with torch.no_grad():
        out_b3 = b3.forward_batch(calib)
        out_b3s = b3s.forward_batch(calib, side_features=side, electrode_ids=electrode_ids)
    assert out_b3.shape == out_b3s.shape
    assert torch.allclose(out_b3, out_b3s)


def test_fs3_shuffles_electrode_ids_not_waveform_scalars(tmp_path):
    pytest.importorskip("pynwb")
    from mc_maze.unit_side_features import (
        load_session_electrode_ids,
        permute_electrode_ids,
    )

    nwb_path = _repo_sample_nwb()
    ids = load_session_electrode_ids(nwb_path)
    shuffled = permute_electrode_ids(ids, permutation_seed=42)
    assert ids.shape == shuffled.shape
    assert not np.array_equal(ids, shuffled)


# ------------------------------------------------------------------------------------
# T4-substrate electrode designs D (gate, B3SEG) and C (anchor, B3SEA) --
# docs/ELECTRODE_ANCHOR_DESIGNS.md. Design A (learned embedding concatenated alongside T4,
# "t4e") reuses B3S/SideFeatureEarlyPoolEncoder unchanged (already covered by the F3 tests
# above with side_dim=4 in place of F2's 6) and is exercised at the registry level below.
# ------------------------------------------------------------------------------------
_ELECTRODE_DESIGN_CLASSES = (
    (ElectrodeGateEarlyPoolEncoder, "B3SEG"),
    (ElectrodeAnchorEarlyPoolEncoder, "B3SEA"),
)


def _perturb_electrode_mechanism(encoder) -> None:
    """Move an electrode-design encoder's own mechanism away from its zero-init so tests can
    tell whether its output actually depends on electrode id assignment."""
    with torch.no_grad():
        if hasattr(encoder, "electrode_gate"):
            encoder.electrode_gate.normal_(generator=torch.Generator().manual_seed(0))
        if hasattr(encoder, "electrode_anchor"):
            encoder.electrode_anchor.weight.normal_(generator=torch.Generator().manual_seed(0))
            encoder.anchor_alpha.fill_(0.7)


# ---- registry (no I/O) ---------------------------------------------------------------

def test_side_feature_dims_includes_t4_electrode_designs():
    from mc_maze.unit_side_features import SIDE_FEATURE_DIMS

    for group in ("t4e", "t4e_shuffled", "t4gate", "t4gate_shuffled", "t4anchor", "t4anchor_shuffled"):
        assert SIDE_FEATURE_DIMS[group] == SIDE_FEATURE_DIMS["t4"] == 4


def test_t4_electrode_designs_base_feature_group_resolves_to_t4():
    """Every T4-substrate electrode design reuses T4's own cosine-tuning fit unchanged --
    none of them are, or read from, a waveform feature group (f1/f2), and design A/D/C's
    shuffled controls resolve the same way as their real counterpart."""
    from mc_maze.unit_side_features import base_feature_group

    for group in ("t4e", "t4e_shuffled", "t4gate", "t4gate_shuffled", "t4anchor", "t4anchor_shuffled"):
        assert base_feature_group(group) == "t4"


def test_t4_electrode_designs_shuffled_control_registry():
    from mc_maze.unit_side_features import (
        is_electrode_shuffle_control,
        is_feature_shuffle_control,
        is_shuffled_control,
    )

    for real, shuffled in (("t4e", "t4e_shuffled"), ("t4gate", "t4gate_shuffled"), ("t4anchor", "t4anchor_shuffled")):
        assert is_shuffled_control(real) is False
        assert is_shuffled_control(shuffled) is True
        # Electrode-shuffle (ids permuted), never feature-shuffle (T4 values permuted) --
        # the T4-substrate designs must never permute the tuning content itself.
        assert is_electrode_shuffle_control(shuffled) is True
        assert is_feature_shuffle_control(shuffled) is False


def test_uses_electrode_ids_is_a_superset_of_uses_electrode_embedding():
    """uses_electrode_embedding (narrow: concat mechanism) gates F3/FS3/t4e/t4e_shuffled only.
    uses_electrode_ids (broad: needs electrode ids attached at all) additionally covers
    design D/C (t4gate/t4anchor and their shuffled controls), whose own gate/anchor tables
    consume electrode ids without concatenating anything at the psi input."""
    from mc_maze.unit_side_features import uses_electrode_embedding, uses_electrode_ids

    concat_groups = {"f3", "fs3", "t4e", "t4e_shuffled"}
    gate_anchor_groups = {"t4gate", "t4gate_shuffled", "t4anchor", "t4anchor_shuffled"}
    for group in concat_groups:
        assert uses_electrode_embedding(group) is True
        assert uses_electrode_ids(group) is True
    for group in gate_anchor_groups:
        assert uses_electrode_embedding(group) is False
        assert uses_electrode_ids(group) is True
    for group in ("none", "f1", "f2", "t4", "t8", "fs1", "fs2", "ts4", "ts8"):
        assert uses_electrode_embedding(group) is False
        assert uses_electrode_ids(group) is False


def test_post_pool_side_dim_t4_electrode_designs():
    """t4e (design A) concatenates T4(4) + ELECTRODE_EMBED_DIM(8) = 12 at the psi input, same
    as F3. t4gate/t4anchor (designs D/C) do not concatenate an embedding at all -- their
    mechanism applies to psi's OUTPUT -- so post_pool_side_dim is just T4's own 4."""
    from mc_maze.unit_side_features import ELECTRODE_EMBED_DIM, post_pool_side_dim

    assert post_pool_side_dim("t4e") == post_pool_side_dim("t4e_shuffled") == 4 + ELECTRODE_EMBED_DIM
    assert post_pool_side_dim("t4gate") == post_pool_side_dim("t4gate_shuffled") == 4
    assert post_pool_side_dim("t4anchor") == post_pool_side_dim("t4anchor_shuffled") == 4


# ---- encoder construction / forward pass (no I/O, synthetic tensors) -----------------

def test_build_encoder_b3seg_and_b3sea_registered(shapes):
    for variant, cls in (("B3SEG", ElectrodeGateEarlyPoolEncoder), ("B3SEA", ElectrodeAnchorEarlyPoolEncoder)):
        enc = build_encoder(
            variant, window_size=shapes["window"], trial_length=shapes["trial_len"],
            hidden_dim=64, side_dim=shapes["side_dim"], num_electrodes=12,
        )
        assert enc.variant == variant
        assert isinstance(enc, cls)
        assert isinstance(enc, SideFeatureEarlyPoolEncoder)


@pytest.mark.parametrize("encoder_cls,variant", _ELECTRODE_DESIGN_CLASSES)
def test_build_encoder_rejects_electrode_embed_dim_for_gate_and_anchor(encoder_cls, variant, shapes):
    """Design D/C's mechanism is not the concat-embedding mechanism electrode_embed_dim
    controls on the B3S base class; passing it non-zero must raise, not silently ignore it."""
    with pytest.raises(ValueError, match="electrode_embed_dim"):
        build_encoder(
            variant, window_size=shapes["window"], trial_length=shapes["trial_len"],
            hidden_dim=64, side_dim=shapes["side_dim"], num_electrodes=12, electrode_embed_dim=8,
        )


@pytest.mark.parametrize("encoder_cls,variant", _ELECTRODE_DESIGN_CLASSES)
def test_electrode_design_requires_positive_num_electrodes(encoder_cls, variant, shapes):
    with pytest.raises(ValueError, match="num_electrodes"):
        encoder_cls(shapes["trial_len"], shapes["window"], 64, side_dim=shapes["side_dim"], num_electrodes=0)


@pytest.mark.parametrize("encoder_cls,variant", _ELECTRODE_DESIGN_CLASSES)
def test_electrode_design_zero_init_matches_plain_t4(encoder_cls, variant, shapes):
    """Both designs must be exactly equivalent to plain T4 (SideFeatureEarlyPoolEncoder with
    the same side_dim) at initialization -- the whole point of zero-initializing g/M/alpha is
    that the comparison starts from an identical function (coordinator spec)."""
    calib = _random_calib(shapes)
    side = _random_side(shapes)
    num_electrodes = 12
    electrode_ids = torch.randint(0, num_electrodes, (shapes["batch"], shapes["neurons"]))
    hidden_dim = 64

    t4 = SideFeatureEarlyPoolEncoder(shapes["trial_len"], shapes["window"], hidden_dim, side_dim=shapes["side_dim"])
    enc = encoder_cls(
        shapes["trial_len"], shapes["window"], hidden_dim,
        side_dim=shapes["side_dim"], num_electrodes=num_electrodes,
    )
    enc.load_state_dict(t4.state_dict(), strict=False)
    t4.eval()
    enc.eval()
    with torch.no_grad():
        out_t4 = t4.forward_batch(calib, side_features=side)
        out_enc = enc.forward_batch(calib, side_features=side, electrode_ids=electrode_ids)
    assert out_t4.shape == out_enc.shape == (shapes["batch"], shapes["neurons"], shapes["window"])
    assert torch.allclose(out_t4, out_enc)


@pytest.mark.parametrize("encoder_cls,variant", _ELECTRODE_DESIGN_CLASSES)
def test_electrode_design_output_shape_independent_of_n(encoder_cls, variant, shapes):
    hidden_dim = 64
    num_electrodes = 12
    enc = encoder_cls(
        shapes["trial_len"], shapes["window"], hidden_dim,
        side_dim=shapes["side_dim"], num_electrodes=num_electrodes,
    )
    _perturb_electrode_mechanism(enc)
    enc.eval()
    for n in (3, shapes["neurons"], 15):
        calib = torch.randn(shapes["batch"], shapes["trials"], shapes["trial_len"], n)
        side = torch.randn(shapes["batch"], n, shapes["side_dim"])
        electrode_ids = torch.randint(0, num_electrodes, (shapes["batch"], n))
        with torch.no_grad():
            out = enc.forward_batch(calib, side_features=side, electrode_ids=electrode_ids)
        assert out.shape == (shapes["batch"], n, shapes["window"])


@pytest.mark.parametrize("encoder_cls,variant", _ELECTRODE_DESIGN_CLASSES)
def test_electrode_design_permutation_invariance_with_electrode_ids_in_sync(encoder_cls, variant, shapes):
    """Permuting the unit axis of calib/side_features/electrode_ids together must permute the
    output identically -- units carry their own electrode id with them, exactly like F3."""
    calib = _random_calib(shapes)
    side = _random_side(shapes)
    num_electrodes = 12
    electrode_ids = torch.randint(0, num_electrodes, (shapes["batch"], shapes["neurons"]))
    enc = encoder_cls(
        shapes["trial_len"], shapes["window"], 64,
        side_dim=shapes["side_dim"], num_electrodes=num_electrodes,
    )
    _perturb_electrode_mechanism(enc)
    enc.eval()
    perm = torch.randperm(shapes["neurons"])
    with torch.no_grad():
        base = enc.forward_batch(calib, side_features=side, electrode_ids=electrode_ids)
        permuted = enc.forward_batch(
            calib[..., perm], side_features=side[:, perm, :], electrode_ids=electrode_ids[:, perm]
        )
    assert torch.allclose(base[:, perm, :], permuted, atol=1e-6)


@pytest.mark.parametrize("encoder_cls,variant", _ELECTRODE_DESIGN_CLASSES)
def test_electrode_design_output_depends_on_electrode_assignment(encoder_cls, variant, shapes):
    """With a non-degenerate (post zero-init) gate/anchor, permuting ONLY the electrode id
    assignment (unit order and T4 content held fixed) must change the output -- otherwise the
    mechanism would not actually be reading electrode identity."""
    calib = _random_calib(shapes)
    side = _random_side(shapes)
    num_electrodes = 12
    electrode_ids = torch.randint(0, num_electrodes, (shapes["batch"], shapes["neurons"]))
    enc = encoder_cls(
        shapes["trial_len"], shapes["window"], 64,
        side_dim=shapes["side_dim"], num_electrodes=num_electrodes,
    )
    _perturb_electrode_mechanism(enc)
    enc.eval()
    shuffled_ids = electrode_ids[:, torch.randperm(shapes["neurons"])]
    with torch.no_grad():
        base = enc.forward_batch(calib, side_features=side, electrode_ids=electrode_ids)
        shuffled = enc.forward_batch(calib, side_features=side, electrode_ids=shuffled_ids)
    assert not torch.allclose(base, shuffled, atol=1e-6)


@pytest.mark.parametrize("encoder_cls,variant", _ELECTRODE_DESIGN_CLASSES)
def test_electrode_design_requires_electrode_ids(encoder_cls, variant, shapes):
    calib = _random_calib(shapes)
    side = _random_side(shapes)
    enc = encoder_cls(shapes["trial_len"], shapes["window"], 64, side_dim=shapes["side_dim"], num_electrodes=12)
    enc.eval()
    with pytest.raises(ValueError, match="electrode_ids"):
        enc.forward_batch(calib, side_features=side)


@pytest.mark.parametrize("encoder_cls,variant", _ELECTRODE_DESIGN_CLASSES)
def test_electrode_design_out_of_range_electrode_id_raises_never_clamps(encoder_cls, variant, shapes):
    calib = _random_calib(shapes)
    side = _random_side(shapes)
    num_electrodes = 12
    enc = encoder_cls(shapes["trial_len"], shapes["window"], 64, side_dim=shapes["side_dim"], num_electrodes=num_electrodes)
    enc.eval()
    for bad_value in (-1, num_electrodes, num_electrodes + 50):
        electrode_ids = torch.randint(0, num_electrodes, (shapes["batch"], shapes["neurons"]))
        electrode_ids[0, 0] = bad_value
        with pytest.raises(ValueError, match="out of range"):
            enc.forward_batch(calib, side_features=side, electrode_ids=electrode_ids)


@pytest.mark.parametrize("encoder_cls,variant", _ELECTRODE_DESIGN_CLASSES)
def test_electrode_design_parameter_count(encoder_cls, variant, shapes):
    """g (design D) has exactly num_electrodes scalars; M+alpha (design C) has
    num_electrodes*window_size + 1 -- both on top of plain T4's own parameter count, and
    neither design adds anything to post_pool's input width (electrode_embed_dim=0 always)."""
    hidden_dim = 64
    num_electrodes = 12
    t4 = SideFeatureEarlyPoolEncoder(shapes["trial_len"], shapes["window"], hidden_dim, side_dim=shapes["side_dim"])
    enc = encoder_cls(
        shapes["trial_len"], shapes["window"], hidden_dim,
        side_dim=shapes["side_dim"], num_electrodes=num_electrodes,
    )
    t4_params = sum(p.numel() for p in t4.parameters())
    enc_params = sum(p.numel() for p in enc.parameters())
    if encoder_cls is ElectrodeGateEarlyPoolEncoder:
        assert enc_params - t4_params == num_electrodes
    else:
        assert enc_params - t4_params == num_electrodes * shapes["window"] + 1
    # electrode_embed_dim is always 0 for these designs -- post_pool's input width (and
    # therefore its own weight count) must be identical to plain T4's.
    assert enc.post_pool[0].weight.shape == t4.post_pool[0].weight.shape


# ---- data pipeline (requires the repo's sample NWB session) ---------------------------

def test_t4gate_and_t4anchor_shuffle_electrode_ids_not_tuning_features(tmp_path):
    """The electrode-shuffle controls for designs D/C must permute only the electrode id
    assignment -- the SAME T4 tuning features (base_feature_group resolves both real and
    shuffled tokens to "t4") stay correctly attached to their real unit."""
    pytest.importorskip("pynwb")
    from mc_maze.unit_side_features import (
        base_feature_group,
        compute_unit_side_features_uncached,
        load_session_electrode_ids,
        permute_electrode_ids,
    )

    nwb_path = _repo_sample_nwb()
    for real_token, shuffled_token in (("t4gate", "t4gate_shuffled"), ("t4anchor", "t4anchor_shuffled")):
        assert base_feature_group(real_token) == base_feature_group(shuffled_token) == "t4"

    raw_t4, _ = compute_unit_side_features_uncached(nwb_path, feature_group="t4", pool_size=50)
    ids = load_session_electrode_ids(nwb_path)
    shuffled_ids = permute_electrode_ids(ids, permutation_seed=42)
    assert ids.shape == shuffled_ids.shape == (raw_t4.shape[0],)
    assert not np.array_equal(ids, shuffled_ids)
    # A shuffle, not a different electrode assignment: same multiset of ids, different order.
    assert np.array_equal(np.sort(ids), np.sort(shuffled_ids))
