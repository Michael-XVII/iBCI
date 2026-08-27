#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

mkdir -p logs sua_exploration/results
exec /home/ial-mohd/workspace/envs/spint/bin/python \
  sua_exploration/scripts/eval_e08_analytic_only_dandi688.py \
  --out_path sua_exploration/results/e08_analytic_only_t4_seed42.json
