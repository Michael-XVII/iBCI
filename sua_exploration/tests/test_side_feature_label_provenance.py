from __future__ import annotations

import ast
from pathlib import Path

import pytest

from mc_maze.unit_side_features import side_features_use_behavior_labels


@pytest.mark.parametrize(
    "group",
    (
        # Every legacy generic-evaluator label-derived token.
        "t4", "t8", "ts4", "ts8",
        "t4e", "t4e_shuffled",
        "t4gate", "t4gate_shuffled",
        "t4anchor", "t4anchor_shuffled",
        "t4rel", "t4rel_membership_shuffled", "t4rel_nogroup",
        "t4cf", "t4cf_ts4", "t4cf_confidence_shuffled",
        # The prior omission plus every equivalent residual/shuffled substrate.
        "t4w3", "ts4w3", "t4cf_residual", "t4cf_residual_shuffled",
        # Template-Ridge descriptors also derive from target-direction labels.
        "tr4", "trs4", "trls4", "trz4",
    ),
)
def test_label_provenance_keeps_all_tuning_substrates_true(group):
    assert side_features_use_behavior_labels(group) is True


@pytest.mark.parametrize("group", ("f1", "f2", "f3", "fs1", "fs2", "fs3", "none"))
def test_label_provenance_keeps_waveform_and_none_false(group):
    assert side_features_use_behavior_labels(group) is False


def test_generic_evaluator_uses_the_helper_without_changing_forward_call_contract():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "eval_epoch_window_generic_dandi688.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    helper_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "side_features_use_behavior_labels"
    ]
    assert len(helper_calls) == 1
    assert isinstance(helper_calls[0].args[0], ast.Name)
    assert helper_calls[0].args[0].id == "side_feature_group"
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "evaluate_fixed_protocol_over_validation_sessions"
    ]
    assert len(calls) == 1
    assert [keyword.arg for keyword in calls[0].keywords] == [
        "ckpt_path",
        "teacher_ckpt",
        "variant",
        "data_dir",
        "task",
        "split_counts",
        "max_units_exclusive",
        "cache_dir",
        "pool_size",
        "selection_mode",
        "calibration_n",
        "signal_view",
        "train_val_manifest",
    ]
