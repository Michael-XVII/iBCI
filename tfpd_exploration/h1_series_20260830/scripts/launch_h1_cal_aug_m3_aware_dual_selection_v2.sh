#!/usr/bin/env bash
set -euo pipefail
repo_root=/home/ial-mohd/workspace/iBCI
python_bin=/home/ial-mohd/workspace/envs/spint/bin/python
runner="$repo_root/tfpd_exploration/h1_series_20260830/scripts/run_h1_cal_aug_m3_aware_dual_selection_v2.py"
log_root="$repo_root/logs/h1_cal_aug_m3_aware_dual_selection_v2"
mkdir -p "$log_root"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TQDM_DISABLE=1
exec "$python_bin" "$runner" --detached-supervisor --physical-gpu 0 >"$log_root/supervisor.log" 2>&1
