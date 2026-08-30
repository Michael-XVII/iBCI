#!/usr/bin/env python3
"""Run the matched H1 variable-activity successor score."""
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({
            "status": "DRY_NO_TARGET_NO_CHECKPOINT_NO_CUDA_NO_WRITE",
            "surface": "H1_FOLD0_HC_MATCHED_ACTIVITY_HEADROOM",
            "comparator": "IMMUTABLE_SEALED_HC_ACTIVITY_HEADROOM",
        }, sort_keys=True))
        return
    if args.output is None:
        parser.error("--execute requires --output")
    from tfpd_exploration.src.h1_variable_activity_exposure_v1.evaluate import run_evaluation
    from tfpd_exploration.src.h1_variable_activity_exposure_v1.train import write_json_once
    payload = run_evaluation(ROOT, device=args.device)
    path, digest = write_json_once(args.output, payload)
    print(json.dumps({
        "status": payload["status"],
        "verdict": payload["verdict"],
        "path": str(path),
        "sha256": digest,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
