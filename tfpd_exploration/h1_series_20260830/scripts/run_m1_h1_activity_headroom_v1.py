#!/usr/bin/env python3
"""Execute one frozen M1 or H1 activity-headroom matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("m1", "h1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({
            "status": "DRY_NO_DATA_NO_CHECKPOINT_NO_CUDA_NO_WRITE",
            "datasets": ["m1", "h1"],
            "arms": ["STATIC_SUPPORT", "ROLLING_FIXED_M", "CAUSAL_GROWING_CAP30", "FULL_SESSION_ORACLE"],
            "full_session_oracle": "LABEL_FREE_BUT_NONCAUSAL",
        }, sort_keys=True))
        return
    if args.dataset is None or args.output is None:
        parser.error("--execute requires --dataset and --output")
    if args.dataset == "m1":
        from tfpd_exploration.src.m1_h1_activity_headroom_v1.m1 import run, write_once
    else:
        from tfpd_exploration.src.m1_h1_activity_headroom_v1.h1 import run, write_once
    payload = run(ROOT, device=args.device)
    path, digest = write_once(args.output, payload)
    print(json.dumps({"status": payload["status"], "path": str(path), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
