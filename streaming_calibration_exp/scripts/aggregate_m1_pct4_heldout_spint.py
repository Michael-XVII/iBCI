#!/usr/bin/env python
"""Aggregate M1-PCT4-v1 SPINT-style held-out selected runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARM_EXPERIMENTS = {
    "baseline": "native_mua_f0_m1_heldout_spint",
    "t4": "native_mua_t4_m1_heldout_spint",
    "pct4": "native_mua_pct4_m1_heldout_spint",
    "pct4_z4": "native_mua_pct4_z4_m1_heldout_spint",
    "pct4_rs": "native_mua_pct4_rs_m1_heldout_spint",
    "pct4_ls": "native_mua_pct4_ls_m1_heldout_spint",
}
CONTRASTS = {
    "pct4_minus_z4": ("pct4", "pct4_z4"),
    "pct4_minus_rs": ("pct4", "pct4_rs"),
    "pct4_minus_ls": ("pct4", "pct4_ls"),
    "pct4_minus_t4": ("pct4", "t4"),
    "pct4_minus_baseline": ("pct4", "baseline"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split_metrics(metrics_path: Path) -> dict[str, object]:
    values: dict[str, object] = {"heldout_sessions": {}, "heldin_sessions": {}}
    with metrics_path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split = row.get("split")
            session = row.get("session", "")
            r2 = float(row["R2_variance_weighted"])
            if split == "test_heldout":
                if session == "mean":
                    values["test_heldout_r2_mean"] = r2
                else:
                    values["heldout_sessions"][session] = r2
            elif split == "test_heldin":
                if session == "mean":
                    values["test_heldin_r2_mean"] = r2
                else:
                    values["heldin_sessions"][session] = r2
    if "test_heldout_r2_mean" not in values:
        sessions = values["heldout_sessions"]
        if sessions:
            values["test_heldout_r2_mean"] = statistics.mean(sessions.values())
    if "test_heldin_r2_mean" not in values:
        sessions = values["heldin_sessions"]
        if sessions:
            values["test_heldin_r2_mean"] = statistics.mean(sessions.values())
    if "test_heldout_r2_mean" not in values:
        raise ValueError(f"No test_heldout R2 found in {metrics_path}")
    return values


def parse_run(run_dir: Path) -> dict[str, object] | None:
    metrics_path = run_dir / "metrics_per_session.csv"
    split_path = run_dir / "split_manifest.json"
    config_path = run_dir / "resolved_config.yaml"
    checkpoint_path = run_dir / "checkpoint_manifest.json"
    metadata_path = run_dir / "run_metadata.json"
    if not (metrics_path.is_file() and split_path.is_file() and config_path.is_file() and checkpoint_path.is_file()):
        return None
    split = json.loads(split_path.read_text())
    checkpoint = json.loads(checkpoint_path.read_text())
    if split.get("validation_protocol") != "minival":
        return None
    if split.get("heldout_evaluated_in_fit") is not True or split.get("heldout_evaluated_in_test") is not True:
        return None
    if checkpoint.get("selected_by_metric") != "val_heldout/r2_mean":
        return None
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    metrics = read_split_metrics(metrics_path)
    return {
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "split_manifest": split,
        "checkpoint_manifest": checkpoint,
        "run_metadata": metadata,
        "config_text": config_path.read_text(),
        "metrics": metrics,
        "metrics_sha256": sha256_file(metrics_path),
        "manifest_sha256": sha256_file(split_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def discover_cells(base_dir: Path, arms: dict[str, str], seeds: list[int]) -> dict[tuple[str, int], dict[str, object]]:
    run_dirs = sorted(base_dir.glob("*heldout_spint*")) if base_dir.is_dir() else []
    runs = [run for run in (parse_run(path) for path in run_dirs) if run is not None]
    cells: dict[tuple[str, int], dict[str, object]] = {}
    for arm, run_id in arms.items():
        for seed in seeds:
            matches = []
            for run in runs:
                split = run["split_manifest"]
                metadata = run["run_metadata"]
                text = run["config_text"]
                checkpoint = run["checkpoint_manifest"]
                if split.get("validation_protocol") != "minival":
                    continue
                if split.get("heldout_evaluated_in_fit") is not True or split.get("heldout_evaluated_in_test") is not True:
                    continue
                if checkpoint.get("selected_by_metric") != "val_heldout/r2_mean":
                    continue
                if f"run_id: {run_id}" not in text:
                    continue
                if int(metadata.get("seed", -1)) != seed and f"seed: {seed}" not in text:
                    continue
                norm = split.get("native_t4_normalization")
                if arm == "baseline":
                    if norm is not None:
                        continue
                else:
                    expected_group = arm
                    if arm == "t4":
                        expected_group = "t4"
                    if not isinstance(norm, dict) or norm.get("feature_group") != expected_group:
                        continue
                matches.append(run)
            if len(matches) != 1:
                raise SystemExit(f"Expected one heldout-spint cell for arm={arm} seed={seed}, got {len(matches)}")
            cells[(arm, seed)] = matches[0]
    return cells


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=Path("outputs/streaming_calibration"), type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/m1_pct4_v1"), type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    args = parser.parse_args()

    cells = discover_cells(args.base_dir, ARM_EXPERIMENTS, args.seeds)
    rows = []
    for seed in args.seeds:
        row = {"seed": seed}
        for arm in ARM_EXPERIMENTS:
            metrics = cells[(arm, seed)]["metrics"]
            row[f"{arm}_test_heldout_r2"] = metrics["test_heldout_r2_mean"]
            row[f"{arm}_test_heldin_r2"] = metrics.get("test_heldin_r2_mean", "")
            row[f"{arm}_selected_value"] = cells[(arm, seed)]["checkpoint_manifest"].get("selected_metric_value")
        for name, (left, right) in CONTRASTS.items():
            row[name] = row[f"{left}_test_heldout_r2"] - row[f"{right}_test_heldout_r2"]
        rows.append(row)

    means = {
        arm: statistics.mean(float(row[f"{arm}_test_heldout_r2"]) for row in rows)
        for arm in ARM_EXPERIMENTS
    }
    paired = {}
    for name in CONTRASTS:
        values = [float(row[name]) for row in rows]
        paired[name] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "positive_seeds": sum(value > 0 for value in values),
            "n_seeds": len(values),
            "by_seed": dict(zip(map(str, args.seeds), values)),
        }
    payload = {
        "protocol": "m1-pct4-v1-heldout-spint-style",
        "primary_metric": "test_heldout/r2_mean from the best checkpoint selected by val_heldout/r2_mean",
        "seeds": args.seeds,
        "arms": ARM_EXPERIMENTS,
        "heldout_selected": True,
        "means_test_heldout_r2": means,
        "paired_contrasts_test_heldout_r2": paired,
        "cells": {
            f"{arm}:seed{seed}": {
                "run_dir": cells[(arm, seed)]["run_dir"],
                "run_id": cells[(arm, seed)]["run_id"],
                "test_heldout_r2_mean": cells[(arm, seed)]["metrics"]["test_heldout_r2_mean"],
                "test_heldin_r2_mean": cells[(arm, seed)]["metrics"].get("test_heldin_r2_mean"),
                "heldout_sessions": cells[(arm, seed)]["metrics"]["heldout_sessions"],
                "selected_by_metric": cells[(arm, seed)]["checkpoint_manifest"].get("selected_by_metric"),
                "selected_metric_value": cells[(arm, seed)]["checkpoint_manifest"].get("selected_metric_value"),
                "metrics_sha256": cells[(arm, seed)]["metrics_sha256"],
                "manifest_sha256": cells[(arm, seed)]["manifest_sha256"],
                "config_sha256": cells[(arm, seed)]["config_sha256"],
            }
            for arm in ARM_EXPERIMENTS
            for seed in args.seeds
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "M1_PCT4_HELDOUT_SPINT_AGGREGATE.json"
    csv_path = args.output_dir / "M1_PCT4_HELDOUT_SPINT_PER_SEED.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(json.dumps({"means_test_heldout_r2": means, "paired": paired}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
