#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ial-mohd/workspace/iBCI}
PY=${PY:-/home/ial-mohd/workspace/envs/spint/bin/python}
GPU=${GPU:?Set GPU to an idle physical GPU index}
TEACHER_CKPT=${TEACHER_CKPT:-$ROOT/sua_exploration/checkpoints/teacher_dandi688_co_heldout_spint_seed42/best-epoch=020-val_heldout/r2_mean=0.1045.ckpt}
OUT=e02_posterior_mean_t4_s42
RESULT=$ROOT/sua_exploration/results/p3_${OUT}_seed42.json
EVIDENCE=$ROOT/sua_exploration/results/e02_posterior_mean_t4_v1
cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" sua_exploration/scripts/train_variant_dandi688.py --teacher_ckpt "$TEACHER_CKPT" --task CO --data_dir /home/ial-mohd/dataset/ial-mohd/000688/sub-C --variant B3S --side_features t4r --side_feature_pool_size 50 --calibration_n_trials 50 --max_epochs 40 --patience 10 --heldout_spint_selection --require_gpu --accelerator gpu --disable_progress_bar --num_workers 4 --batch_size 32 --cache_dir /tmp/ibci_e02_t4r_cache --seed 42 --out_name "$OUT"
"$PY" sua_exploration/scripts/audit_e02_posterior_mean_t4.py --run "$RESULT" --out "$EVIDENCE/E02_POSTERIOR_MEAN_T4_AUDIT.json"
"$PY" sua_exploration/scripts/aggregate_e02_posterior_mean_t4.py --baseline sua_exploration/results/p3_template_ridge_db_heldout_spint_t4_s42_seed42.json --e02 "$RESULT" --out "$EVIDENCE/E02_POSTERIOR_MEAN_T4_AGGREGATE.json"
