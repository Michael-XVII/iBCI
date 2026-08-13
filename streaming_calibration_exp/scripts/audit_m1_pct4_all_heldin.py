#!/usr/bin/env python
"""CPU constructibility audit for M1-PCT4-v1 calibration sessions."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from falcon_challenge.config import FalconTask
from falcon_challenge.dataloaders import load_nwb

from src.data.falcon_m1_pct4_features import (
    PCT4_ESTIMATOR_VERSION,
    calibration_m1_pct4_metadata,
    pct4_from_phase_sums,
    phase_window_trial_sums,
)


def audit_file(path: Path, support_trials: int) -> dict[str, object]:
    neural, _, _, eval_mask = load_nwb(path, FalconTask.m1)
    meta = calibration_m1_pct4_metadata(path, FalconTask.m1)
    reach_sums, reach_lengths = phase_window_trial_sums(
        neural,
        eval_mask,
        meta.bin_timestamps,
        meta.move_onset_times,
        meta.contact_times,
        source=f"{path.name}:reach",
    )
    post_sums, post_lengths = phase_window_trial_sums(
        neural,
        eval_mask,
        meta.bin_timestamps,
        meta.contact_times,
        meta.stop_times,
        source=f"{path.name}:post",
    )
    angles = meta.target_angles[:support_trials]
    design = np.stack([np.ones(angles.shape[0]), np.cos(angles), np.sin(angles)], axis=1)
    rank = int(np.linalg.matrix_rank(design))
    condition = float(np.linalg.cond(design)) if rank == 3 else float("inf")
    pct4 = pct4_from_phase_sums(
        reach_sums[:support_trials],
        reach_lengths[:support_trials],
        post_sums[:support_trials],
        post_lengths[:support_trials],
        angles,
        source=f"{path.name}[0:{support_trials}]",
    )
    return {
        "session": path.stem,
        "split": "heldout" if "held-out-calib" in path.name else "heldin",
        "path": str(path),
        "support_M": int(support_trials),
        "n_trials": int(meta.target_angles.shape[0]),
        "n_channels": int(neural.shape[1]),
        "direction_rank": rank,
        "direction_condition": condition,
        "min_reach_bins": int(reach_lengths[:support_trials].min()),
        "median_reach_bins": float(np.median(reach_lengths[:support_trials])),
        "max_reach_bins": int(reach_lengths[:support_trials].max()),
        "min_post_bins": int(post_lengths[:support_trials].min()),
        "median_post_bins": float(np.median(post_lengths[:support_trials])),
        "max_post_bins": int(post_lengths[:support_trials].max()),
        "n_nan": int(np.isnan(pct4).sum()),
        "pct4_abs_mean": float(np.abs(pct4).mean()),
        "pct4_abs_max": float(np.abs(pct4).max()),
        "first_two_rows": pct4[:2].astype(float).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--support-trials", default=10, type=int)
    parser.add_argument("--output-dir", default=Path("outputs/m1_pct4_v1"), type=Path)
    parser.add_argument("--include-heldout", action="store_true", help="Also audit official M1 held-out calibration NWBs.")
    args = parser.parse_args()

    files = sorted(args.data_dir.rglob("*held-in-calib*.nwb"))
    if args.include_heldout:
        files += sorted(args.data_dir.rglob("*held-out-calib*.nwb"))
    if not files:
        raise SystemExit(f"No calibration NWB files found under {args.data_dir}")
    rows = [audit_file(path, args.support_trials) for path in sorted(files)]
    for row in rows:
        if row["direction_rank"] != 3:
            raise SystemExit(f"Rank failure: {row['session']}")
        if row["min_reach_bins"] <= 0 or row["min_post_bins"] <= 0:
            raise SystemExit(f"Empty phase window: {row['session']}")
        if row["n_nan"] != 0:
            raise SystemExit(f"Non-finite PCT4: {row['session']}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "HELDIN_HELDOUT" if args.include_heldout else "HELDIN"
    csv_path = args.output_dir / f"M1_PCT4_CPU_AUDIT_{suffix}.csv"
    json_path = args.output_dir / f"M1_PCT4_CPU_AUDIT_{suffix}.json"
    fieldnames = [key for key in rows[0] if key != "first_two_rows"]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    payload = {"estimator_version": PCT4_ESTIMATOR_VERSION, "include_heldout": bool(args.include_heldout), "rows": rows}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
