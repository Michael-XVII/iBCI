#!/usr/bin/env bash
set -euo pipefail
ROOT=${ROOT:-/home/ial-mohd/workspace/iBCI}
PY=${PY:-/home/ial-mohd/workspace/envs/spint/bin/python}
GPU=${GPU:?Set GPU to an idle physical GPU index}
TEACHER_CKPT=${TEACHER_CKPT:-$ROOT/sua_exploration/checkpoints/teacher_dandi688_co_heldout_spint_seed42/best-epoch=020-val_heldout/r2_mean=0.1045.ckpt}
OUT=e04_posterior_reliability_logit_t4_s42
RESULT=$ROOT/sua_exploration/results/p3_${OUT}_seed42.json
EVIDENCE=$ROOT/sua_exploration/results/e04_posterior_reliability_logit_t4_v1
E02=$ROOT/sua_exploration/results/p3_e02_posterior_mean_t4_s42_seed42.json
cd "$ROOT"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" sua_exploration/scripts/train_variant_dandi688.py --teacher_ckpt "$TEACHER_CKPT" --task CO --data_dir /home/ial-mohd/dataset/ial-mohd/000688/sub-C --variant B3S --side_features t4rql --side_feature_pool_size 50 --calibration_n_trials 50 --max_epochs 40 --patience 10 --heldout_spint_selection --require_gpu --accelerator gpu --disable_progress_bar --num_workers 4 --batch_size 32 --cache_dir /tmp/ibci_e04_t4rql_cache --seed 42 --out_name "$OUT"
"$PY" sua_exploration/scripts/audit_e04_posterior_reliability_logit.py --run "$RESULT" --out "$EVIDENCE/E04_POSTERIOR_RELIABILITY_LOGIT_AUDIT.json"
"$PY" sua_exploration/scripts/aggregate_e04_posterior_reliability_logit.py --e02 "$E02" --e04 "$RESULT" --out "$EVIDENCE/E04_POSTERIOR_RELIABILITY_LOGIT_AGGREGATE.json"
