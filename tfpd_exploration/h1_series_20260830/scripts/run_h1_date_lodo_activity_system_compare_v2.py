#!/usr/bin/env python3
"""Dry-by-default CLI for corrected H1 2x2 interpretation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "tfpd_exploration" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from h1_date_lodo_activity_system_compare_v2.plan import dry_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dry_plan(), sort_keys=True))
        return 0
    if not args.output:
        parser.error("--execute requires --output")
    from h1_date_lodo_activity_system_compare_v2.evaluate import run, write_once
    payload = run(ROOT)
    path, digest = write_once(Path(args.output), payload)
    print(json.dumps({"path": str(path), "sha256": digest, "interaction_verdict": payload["interaction_verdict"], "recommended_frozen_system_basis": payload["recommended_frozen_system_basis"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
