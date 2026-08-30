#!/usr/bin/env python3
"""Run the H1 variable activity exposure smoke or full cell."""
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
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--checkpoint-output", type=Path)
    parser.add_argument("--receipt-output", type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({
            "status": "DRY_NO_SOURCE_NO_CHECKPOINT_NO_CUDA_NO_WRITE",
            "cell": "H1_HC_VARIABLE_ACTIVITY_EXPOSURE_V1",
            "batch_size": 128,
            "epochs": 5,
            "activity": "50_PERCENT_M4_REPLAY_PLUS_50_PERCENT_VARIABLE_PREFIX",
            "carrier": "FIXED_FIRST4_HC",
        }, sort_keys=True))
        return
    if args.receipt_output is None:
        parser.error("--execute requires --receipt-output")
    if args.smoke_steps is None and args.checkpoint_output is None:
        parser.error("full execution requires --checkpoint-output")
    from tfpd_exploration.src.h1_variable_activity_exposure_v1.train import run_training, write_json_once
    payload = run_training(
        ROOT,
        device=args.device,
        max_steps=args.smoke_steps,
        checkpoint_output=args.checkpoint_output,
    )
    path, digest = write_json_once(args.receipt_output, payload)
    print(json.dumps({"status": payload["status"], "path": str(path), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
