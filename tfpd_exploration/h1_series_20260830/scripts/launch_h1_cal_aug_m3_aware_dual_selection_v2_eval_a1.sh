#!/usr/bin/env bash
set -euo pipefail
repo=/home/ial-mohd/workspace/iBCI
python=/home/ial-mohd/workspace/envs/spint/bin/python
runner="$repo/tfpd_exploration/h1_series_20260830/scripts/run_h1_cal_aug_m3_aware_dual_selection_v2_eval_a1.py"
logs="$repo/logs/h1_cal_aug_m3_aware_dual_selection_v2_eval_a1"
mkdir -p "$logs"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
"$python" "$runner" --detached-supervisor >"$logs/supervisor.log" 2>&1
