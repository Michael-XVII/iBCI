#!/usr/bin/env python
"""Aggregate M1-PCT4-v1 source-only LOSO runs."""
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
    "baseline": "native_mua_f0_m1_loso_internal",
    "t4": "native_mua_t4_m1_loso_internal",
    "pct4": "native_mua_pct4_m1_loso_internal",
    "pct4_z4": "native_mua_pct4_z4_m1_loso_internal",
    "pct4_rs": "native_mua_pct4_rs_m1_loso_internal",
    "pct4_ls": "native_mua_pct4_ls_m1_loso_internal",
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


def read_r2(metrics_path: Path) -> float:
    with metrics_path.open() as handle:
        reader = csv.DictReader(handle)
        values = []
        for row in reader:
            if row.get("split") == "test_heldin":
                values.append(float(row["R2_variance_weighted"]))
    if len(values) != 1:
        raise ValueError(f"Expected exactly one test_heldin row in {metrics_path}, got {len(values)}")
    return values[0]


def parse_run(run_dir: Path) -> dict[str, object] | None:
    metrics_path = run_dir / "metrics_per_session.csv"
    manifest_path = run_dir / "split_manifest.json"
    config_path = run_dir / "resolved_config.yaml"
    if not metrics_path.is_file() or not manifest_path.is_file() or not config_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    cfg_text = config_path.read_text()
    return {
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "r2": read_r2(metrics_path),
        "manifest": manifest,
        "config_text": cfg_text,
        "metrics_sha256": sha256_file(metrics_path),
        "manifest_sha256": sha256_file(manifest_path),
        "config_sha256": sha256_file(config_path),
    }


def discover_cells(base_dir: Path, arms: dict[str, str], seeds: list[int], folds: list[int]) -> dict[tuple[str, int, int], dict[str, object]]:
    if base_dir.is_dir():
        run_dirs = sorted({path.parent for path in base_dir.rglob("metrics_per_session.csv")})
    else:
        run_dirs = []
    runs = [run for run in (parse_run(path) for path in run_dirs) if run is not None]
    cells: dict[tuple[str, int, int], dict[str, object]] = {}
    for arm, run_id in arms.items():
        for seed in seeds:
            for fold in folds:
                matches = []
                for run in runs:
                    manifest = run["manifest"]
                    text = run["config_text"]
                    if manifest.get("validation_protocol") != "loso" or manifest.get("fold_id") != fold:
                        continue
                    if manifest.get("heldout_evaluated_in_fit") or manifest.get("heldout_evaluated_in_test"):
                        continue
                    if f"run_id: {run_id}" not in text:
                        continue
                    if f"seed: {seed}" not in text:
                        continue
                    if "calibration_n_trials: 10" not in text:
                        continue
                    if "random_calibration: false" not in text:
                        continue
                    if "smooth_calibration: false" not in text:
                        continue
                    if arm.startswith("pct4"):
                        norm = manifest.get("native_t4_normalization", {})
                        expected_group = arm
                        if arm == "pct4":
                            expected_group = "pct4"
                        if norm.get("feature_group") != expected_group:
                            continue
                    matches.append(run)
                if len(matches) != 1:
                    raise SystemExit(f"Expected one cell for arm={arm} seed={seed} fold={fold}, got {len(matches)}")
                cells[(arm, seed, fold)] = matches[0]
    return cells


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=Path("outputs/streaming_calibration"), type=Path)
    parser.add_argument("--output-dir", default=Path("outputs/m1_pct4_v1"), type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3])
    args = parser.parse_args()

    cells = discover_cells(args.base_dir, ARM_EXPERIMENTS, args.seeds, args.folds)
    per_fold_rows = []
    for seed in args.seeds:
        for fold in args.folds:
            row = {"seed": seed, "fold": fold}
            for arm in ARM_EXPERIMENTS:
                row[arm] = cells[(arm, seed, fold)]["r2"]
            for name, (left, right) in CONTRASTS.items():
                row[name] = row[left] - row[right]
            per_fold_rows.append(row)

    means = {arm: statistics.mean(row[arm] for row in per_fold_rows) for arm in ARM_EXPERIMENTS}
    paired = {}
    for name in CONTRASTS:
        values = [row[name] for row in per_fold_rows]
        by_seed = [statistics.mean(row[name] for row in per_fold_rows if row["seed"] == seed) for seed in args.seeds]
        by_fold = [statistics.mean(row[name] for row in per_fold_rows if row["fold"] == fold) for fold in args.folds]
        paired[name] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "positive_cells": sum(value > 0 for value in values),
            "n_cells": len(values),
            "seed_means": dict(zip(map(str, args.seeds), by_seed)),
            "fold_means": dict(zip(map(str, args.folds), by_fold)),
            "positive_seed_means": sum(value > 0 for value in by_seed),
            "positive_fold_means": sum(value > 0 for value in by_fold),
        }
    gate = {
        "primary": "pct4_minus_z4",
        "passes_seed42_screen": (
            paired["pct4_minus_z4"]["mean"] >= 0.03
            and paired["pct4_minus_z4"]["positive_seed_means"] == len(args.seeds)
            and paired["pct4_minus_z4"]["positive_fold_means"] == len(args.folds)
            and paired["pct4_minus_rs"]["mean"] > 0
            and paired["pct4_minus_ls"]["mean"] > 0
            and paired["pct4_minus_t4"]["mean"] > 0
        ),
        "phase_granularity_claim": paired["pct4_minus_t4"]["mean"] >= 0.03,
    }
    payload = {
        "estimator_version": "m1-pct4-v1-event-aligned-bin-end",
        "seeds": args.seeds,
        "folds": args.folds,
        "arms": ARM_EXPERIMENTS,
        "means": means,
        "paired_contrasts": paired,
        "gate": gate,
        "cells": {
            f"{arm}:seed{seed}:fold{fold}": {
                key: cells[(arm, seed, fold)][key]
                for key in ["run_dir", "run_id", "r2", "metrics_sha256", "manifest_sha256", "config_sha256"]
            }
            for arm in ARM_EXPERIMENTS
            for seed in args.seeds
            for fold in args.folds
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "M1_PCT4_LOSO_AGGREGATE.json"
    csv_path = args.output_dir / "M1_PCT4_LOSO_PER_FOLD.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_fold_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_fold_rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
