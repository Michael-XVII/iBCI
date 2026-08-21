#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/ial-mohd/workspace/iBCI}
PY=${PY:-/home/ial-mohd/workspace/envs/spint/bin/python}
DATA_DIR=${DATA_DIR:-/home/ial-mohd/dataset/ial-mohd/000688/sub-C}
CACHE_DIR=${CACHE_DIR:-/tmp/ibci_template_ridge_db_cache}
TEACHER_CKPT=${TEACHER_CKPT:?Set TEACHER_CKPT to the DANDI688 teacher checkpoint}
GPU_LIST=${GPU_LIST:-0,1,2,3}
MAX_EPOCHS=${MAX_EPOCHS:-40}
PATIENCE=${PATIENCE:-10}
SEEDS=${SEEDS:-42,43,44}
ARMS=${ARMS:-baseline,t4,ts4,tr4,trs4,trls4,trz4}

cd "$ROOT"
IFS=, read -r -a gpus <<< "$GPU_LIST"
IFS=, read -r -a seeds <<< "$SEEDS"
IFS=, read -r -a arms <<< "$ARMS"

arm_to_side() {
  case "$1" in
    baseline) echo none ;;
    t4) echo t4 ;;
    ts4) echo ts4 ;;
    tr4) echo tr4 ;;
    trs4) echo trs4 ;;
    trls4) echo trls4 ;;
    trz4) echo trz4 ;;
    *) echo "unknown arm $1" >&2; return 2 ;;
  esac
}

idx=0
failed=0
pids=()
names=()

wait_for_oldest_slot() {
  local pid=${pids[0]}
  local name=${names[0]}
  if ! wait "$pid"; then
    echo "[fail] $name pid=$pid" >&2
    failed=1
  fi
  pids=("${pids[@]:1}")
  names=("${names[@]:1}")
}

for seed in "${seeds[@]}"; do
  for arm in "${arms[@]}"; do
    side=$(arm_to_side "$arm")
    gpu=${gpus[$((idx % ${#gpus[@]}))]}
    out_name=template_ridge_db_heldout_spint_${arm}_s${seed}
    echo "[run] arm=$arm seed=$seed gpu=$gpu out=$out_name"
    CUDA_VISIBLE_DEVICES=$gpu "$PY" sua_exploration/scripts/train_variant_dandi688.py \
      --teacher_ckpt "$TEACHER_CKPT" \
      --task CO \
      --data_dir "$DATA_DIR" \
      --variant B3S \
      --side_features "$side" \
      --side_feature_pool_size 50 \
      --calibration_n_trials 50 \
      --max_epochs "$MAX_EPOCHS" \
      --patience "$PATIENCE" \
      --heldout_spint_selection \
      --require_gpu \
      --accelerator gpu \
      --disable_progress_bar \
      --num_workers 4 \
      --batch_size 32 \
      --cache_dir "$CACHE_DIR" \
      --seed "$seed" \
      --out_name "$out_name" &
    pids+=("$!")
    names+=("$out_name")
    idx=$((idx + 1))
    if ((${#pids[@]} >= ${#gpus[@]})); then
      wait_for_oldest_slot
    fi
  done
done

while ((${#pids[@]} > 0)); do
  wait_for_oldest_slot
done

exit "$failed"
