#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/ial-mohd/workspace/iBCI
PY=/home/ial-mohd/workspace/envs/spint/bin/python
DEV=${DEVICE:-cpu}
OUT=$ROOT/sua_exploration/results/e05_rotation_consistency_t4_v1
cd "$ROOT"
PYTHONPATH=streaming_calibration_exp:sua_exploration "$PY" sua_exploration/scripts/eval_e05_rotation_consistency.py --artifact sua_exploration/results/p3_template_ridge_db_heldout_spint_t4_s42_seed42.json --device "$DEV" --out "$OUT/E05_ROTATION_CONSISTENCY.json"
"$PY" sua_exploration/scripts/audit_e05_rotation_consistency.py --run "$OUT/E05_ROTATION_CONSISTENCY.json" --out "$OUT/E05_ROTATION_CONSISTENCY_AUDIT.json"
