from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mc_maze import unit_side_features as usf


def test_weighted_dual_ridge_recovers_known_linear_map():
    design = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    beta_true = np.asarray(
        [[0.5, -1.0], [1.5, 0.25], [-0.75, 0.75], [2.0, -0.5]],
        dtype=np.float32,
    )
    target = design @ beta_true
    beta, condition, trace_hat = usf._weighted_dual_ridge(
        design, target, np.ones(design.shape[0], dtype=np.float32), ridge=1.0e-7
    )
    assert beta.shape == beta_true.shape
    assert np.allclose(beta, beta_true, atol=1.0e-5)
    assert math.isfinite(condition) and condition > 0.0
    assert math.isfinite(trace_hat) and trace_hat > 0.0


def test_template_ridge_reduction_emits_finite_n_by_4(monkeypatch):
    window_size = 3
    num_units = 2
    profile = np.asarray([1.0, 0.5, 0.25], dtype=np.float32)
    trials = [
        {
            "start_time": 0.06,
            "stop_time": 0.22,
            "start": 3,
            "stop": 11,
            "go_cue_time": 0.08,
            "target_on_time": 0.04,
            "target_dir": 0.0,
        },
        {
            "start_time": 0.26,
            "stop_time": 0.42,
            "start": 13,
            "stop": 21,
            "go_cue_time": 0.28,
            "target_on_time": 0.24,
            "target_dir": math.pi / 2.0,
        },
    ]
    binned = np.arange(60, dtype=np.float32).reshape(30, num_units) % 5
    bin_edges = np.arange(31, dtype=np.float64) * 0.02

    monkeypatch.setattr(usf, "list_datamodule_rewarded_trials", lambda *args, **kwargs: trials)
    monkeypatch.setattr(usf, "_binned_spike_matrix", lambda *args, **kwargs: (binned, bin_edges))
    monkeypatch.setattr(usf, "_source_fingerprint", lambda path: {"path": str(path), "size": 0, "mtime_ns": 0})

    features, metadata = usf._compute_template_ridge_features_uncached(
        Path("synthetic_behavior+ecephys.nwb"),
        feature_group="tr4",
        pool_size=2,
        template_profile=profile,
        bin_size_ms=20,
        window_size=window_size,
        trial_result_filter="R",
    )

    assert features.shape == (num_units, 4)
    assert np.isfinite(features).all()
    assert metadata.template_ridge_constructed_rows == 6
    assert metadata.template_ridge_feature_count == window_size * num_units
    assert metadata.template_ridge_profile_sha256 == usf._array_sha256(profile)
    assert metadata.template_ridge_alignment_event == "go_cue_time"


def test_trls4_label_shuffle_is_deterministic_and_changes_features(monkeypatch):
    window_size = 3
    num_units = 2
    profile = np.asarray([1.0, 0.6, 0.2], dtype=np.float32)
    trials = []
    for index, direction in enumerate([0.0, math.pi / 2.0, math.pi, -math.pi / 2.0]):
        start = 3 + index * 8
        trials.append({
            "start_time": start * 0.02,
            "stop_time": (start + 7) * 0.02,
            "start": start,
            "stop": start + 7,
            "go_cue_time": (start + 1) * 0.02,
            "target_on_time": start * 0.02,
            "target_dir": direction,
        })
    binned = (np.arange(100, dtype=np.float32).reshape(50, num_units) % 7) + 1.0
    bin_edges = np.arange(51, dtype=np.float64) * 0.02

    monkeypatch.setattr(usf, "list_datamodule_rewarded_trials", lambda *args, **kwargs: trials)
    monkeypatch.setattr(usf, "_binned_spike_matrix", lambda *args, **kwargs: (binned, bin_edges))
    monkeypatch.setattr(usf, "_source_fingerprint", lambda path: {"path": str(path), "size": 0, "mtime_ns": 0})

    kwargs = dict(
        nwb_path=Path("synthetic_behavior+ecephys.nwb"),
        pool_size=4,
        template_profile=profile,
        bin_size_ms=20,
        window_size=window_size,
        trial_result_filter="R",
    )
    real, _ = usf._compute_template_ridge_features_uncached(feature_group="tr4", **kwargs)
    shuffled_a, _ = usf._compute_template_ridge_features_uncached(feature_group="trls4", **kwargs)
    shuffled_b, _ = usf._compute_template_ridge_features_uncached(feature_group="trls4", **kwargs)

    assert np.allclose(shuffled_a, shuffled_b)
    assert not np.allclose(real[:, :2], shuffled_a[:, :2])


def test_template_ridge_registry_and_controls():
    assert usf.SIDE_FEATURE_DIMS["tr4"] == 4
    assert usf.SIDE_FEATURE_DIMS["trs4"] == 4
    assert usf.SIDE_FEATURE_DIMS["trls4"] == 4
    assert usf.SIDE_FEATURE_DIMS["trz4"] == 4
    assert usf.base_feature_group("trs4") == "tr4"
    assert usf.base_feature_group("trz4") == "tr4"
    assert usf.is_feature_shuffle_control("trs4") is True
    assert usf.is_template_ridge_zero_control("trz4") is True
    assert usf.side_features_use_behavior_labels("tr4") is True
    assert usf.side_features_use_behavior_labels("trls4") is True


def test_template_ridge_stats_are_fit_from_supplied_train_files_only(monkeypatch, tmp_path):
    seen = {"learn_files": None, "compute_files": []}
    profile = np.asarray([1.0, 0.5, 0.25], dtype=np.float32)

    def fake_learn(train_files, **kwargs):
        seen["learn_files"] = [Path(path).name for path in train_files]
        return {
            "profile": profile,
            "profile_sha256": usf._array_sha256(profile),
            "source_trial_count": 2,
            "raw_peak_speed": 1.0,
            "alignment_event": "go_cue_time",
            "alignment_event_counts": {"go_cue_time": 2},
            "source_sessions": seen["learn_files"],
        }

    def fake_compute(nwb_path, **kwargs):
        seen["compute_files"].append(Path(nwb_path).name)
        assert kwargs["template_profile"] is not None
        return np.ones((2, 4), dtype=np.float32), usf.SideFeatureMetadata(
            feature_group="tr4",
            feature_version=1,
            pool_size=2,
            cache_key="synthetic",
            degenerate_unit_count=0,
            zero_spike_unit_count=0,
            single_spike_unit_count=0,
            zero_noise_std_unit_count=0,
            zero_template_max_unit_count=0,
        )

    monkeypatch.setattr(usf, "learn_template_ridge_speed_profile", fake_learn)
    monkeypatch.setattr(usf, "compute_unit_side_features_uncached", fake_compute)
    mean, std, receipt = usf.fit_side_feature_stats(
        [tmp_path / "train-a.nwb", tmp_path / "train-b.nwb"],
        feature_group="tr4",
        pool_size=2,
        window_size=3,
        return_template_receipt=True,
    )

    assert seen["learn_files"] == ["train-a.nwb", "train-b.nwb"]
    assert seen["compute_files"] == ["train-a.nwb", "train-b.nwb"]
    assert receipt["source_sessions"] == ["train-a.nwb", "train-b.nwb"]
    assert mean.shape == (4,)
    assert std.shape == (4,)
