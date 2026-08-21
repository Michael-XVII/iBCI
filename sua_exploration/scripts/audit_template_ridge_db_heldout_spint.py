"""CPU audit for Template-Ridge D-b side-feature construction on DANDI 000688."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mc_maze.multisession_datamodule import (
    chronological_session_split,
    discover_nwb_files,
    list_datamodule_rewarded_trials,
    session_name_from_path,
)
from mc_maze.unit_side_features import (
    TEMPLATE_RIDGE_RIDGE,
    learn_template_ridge_speed_profile,
)


def parse_split_counts(text: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in text.split(",")]
    if len(parts) != 3 or any(part < 0 for part in parts):
        raise ValueError("split_counts must be three comma-separated non-negative integers")
    return parts[0], parts[1], parts[2]


def audit_session(path: Path, *, split: str, pool_size: int, bin_size_ms: int, window_size: int) -> dict:
    trials = list_datamodule_rewarded_trials(
        path,
        bin_size_ms=bin_size_ms,
        window_size=window_size,
        trial_result_filter="R",
    )
    pool = trials[:pool_size]
    dt = bin_size_ms / 1000.0
    rows = 0
    labelled = 0
    events = {}
    directions = []
    for trial in pool:
        direction = trial.get("target_dir")
        if direction is None or not np.isfinite(direction):
            continue
        labelled += 1
        directions.append(float(direction))
        align = None
        event = None
        for key in ("go_cue_time", "target_on_time", "start_time"):
            value = trial.get(key)
            if value is not None and np.isfinite(value):
                align = float(value)
                event = key
                break
        if align is None:
            continue
        events[event] = events.get(event, 0) + 1
        for offset in range(window_size):
            if align + offset * dt < float(trial["stop_time"]):
                rows += 1
    return {
        "session": session_name_from_path(path),
        "split": split,
        "rewarded_trials": len(trials),
        "support_trials_requested": pool_size,
        "support_trials_available": len(pool),
        "labelled_support_trials": labelled,
        "unique_direction_count": len({round(direction, 6) for direction in directions}),
        "constructed_synthetic_rows": rows,
        "alignment_event_counts": events,
        "pass": len(pool) >= pool_size and labelled > 0 and rows >= 4,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/home/ial-mohd/dataset/ial-mohd/000688/sub-C"))
    parser.add_argument("--task", default="CO")
    parser.add_argument("--split-counts", default="37,8,8")
    parser.add_argument("--support-trials", type=int, default=50)
    parser.add_argument("--bin-size-ms", type=int, default=20)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--out", type=Path, default=Path("sua_exploration/results/template_ridge_db_heldout_spint_v1/TEMPLATE_RIDGE_DB_HELDOUT_SPINT_AUDIT.json"))
    args = parser.parse_args()

    files = discover_nwb_files(args.data_dir, task=args.task)
    train, val, test = chronological_session_split(files, parse_split_counts(args.split_counts))
    profile = learn_template_ridge_speed_profile(
        train,
        pool_size=args.support_trials,
        bin_size_ms=args.bin_size_ms,
        window_size=args.window_size,
        trial_result_filter="R",
    )
    rows = []
    for split, split_files in (("train", train), ("val_heldout_selection", val), ("test_heldout", test)):
        for path in split_files:
            rows.append(audit_session(
                path,
                split=split,
                pool_size=args.support_trials,
                bin_size_ms=args.bin_size_ms,
                window_size=args.window_size,
            ))
    payload = {
        "schema_version": 1,
        "data_dir": str(args.data_dir),
        "task": args.task,
        "split_counts": list(parse_split_counts(args.split_counts)),
        "support_trials": args.support_trials,
        "bin_size_ms": args.bin_size_ms,
        "window_size": args.window_size,
        "ridge_lambda": TEMPLATE_RIDGE_RIDGE,
        "template_profile_sha256": profile["profile_sha256"],
        "template_source_sessions": profile["source_sessions"],
        "template_source_trial_count": profile["source_trial_count"],
        "template_alignment_event": profile["alignment_event"],
        "template_alignment_event_counts": profile["alignment_event_counts"],
        "heldout_used_for_template_fit": False,
        "rows": rows,
        "all_sessions_pass": all(row["pass"] for row in rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
