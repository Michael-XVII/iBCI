#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/ial-mohd/workspace/iBCI
PY=/home/ial-mohd/workspace/envs/spint/bin/python
GPU_ID=${GPU_ID:?Set GPU_ID to one idle physical GPU index}
OUT_NAME=e06_minimal_so2_t4_s42_20260827
LOG=${ROOT}/logs/${OUT_NAME}.log
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  RESUME_ARGS+=(--resume_checkpoint "${RESUME_CHECKPOINT}")
fi

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH=streaming_calibration_exp:sua_exploration
"${PY}" sua_exploration/scripts/train_variant_dandi688.py \
  --variant B3S \
  --side_features t4 \
  --decoder_mode so2 \
  --data_dir /home/ial-mohd/dataset/ial-mohd/000688/sub-C \
  --task CO \
  --split_counts 37,8,8 \
  --teacher_ckpt sua_exploration/checkpoints/teacher_dandi688_co_heldout_spint_seed42/best-epoch=020-val_heldout/r2_mean=0.1045.ckpt \
  --cache_dir /tmp/ibci_template_ridge_db_cache \
  --side_feature_pool_size 50 \
  --calibration_n_trials 50 \
  --seed 42 \
  --max_epochs 40 \
  --patience 10 \
  --lr 1e-4 \
  --batch_size 32 \
  --num_workers 0 \
  --accelerator gpu \
  --require_gpu \
  --heldout_spint_selection \
  --disable_progress_bar \
  --out_name "${OUT_NAME}" \
  "${RESUME_ARGS[@]}" 2>&1 | tee -a "${LOG}"
