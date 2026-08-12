# M1 PCT4-v1 Source-Only Experiment Record

**Date:** 2026-08-12
**Branch:** `exp/m1-pct4-v1-source-only`
**Base SHA at implementation time:** `799d30d971a71fa30095b03bcb6009ebfc2f6c5c`
**Docker/env:** container `docker-ial-mohd`, Python `/home/ial-mohd/workspace/envs/spint/bin/python`
**Dataset:** `/home/ial-mohd/dataset/ial-mohd/000941`

## Status

Implementation, CPU audit, unit tests, source-only teacher smoke, and PCT4 fold0 smoke passed. The formal source-only LOSO R2 matrix was not run in this session because no valid pre-existing M1 teacher checkpoint was available and all usable GPUs were occupied by other users' jobs when checked. Official M1 held-out was not opened.

Therefore PCT4-v1 is **indeterminate / not promoted** as of this record. No aggregate R2 JSON/CSV was generated because doing so without the full `test_heldin` matrix would be misleading.

## Handoff Assessment

The handoff document is scientifically reasonable. Direction+Object should not be used because official M1 10-trial calibration is object-degenerate. PCT4 is a valid new estimand because it changes the measured functional carrier from whole-trial direction tuning to phase-conditioned signed direction tuning. It does not reopen the old M1 negative boundary unless it first passes a source-only gate.

The implementation follows the handoff boundaries: M2 T4 remains unchanged, PCT4 is M1-only, phase windows are frozen as `move_onset -> contact` and `contact -> stop`, and raw official bin-end/eval-mask semantics are used.

## Implemented Files

- `streaming_calibration_exp/src/data/falcon_m1_pct4_features.py`
- `streaming_calibration_exp/src/data/falcon_datamodule.py`
- `streaming_calibration_exp/src/models/falcon_module.py`
- `streaming_calibration_exp/configs/model/streaming_b3s_pct4_m1.yaml`
- `streaming_calibration_exp/configs/model/falcon_m1_source_only_teacher.yaml`
- `streaming_calibration_exp/configs/experiment/b3s_pct4*_m1_loso_internal.yaml`
- `streaming_calibration_exp/configs/experiment/falcon_m1_source_only_teacher.yaml`
- `streaming_calibration_exp/scripts/audit_m1_pct4_all_heldin.py`
- `streaming_calibration_exp/scripts/aggregate_m1_pct4_loso.py`
- `streaming_calibration_exp/tests/test_falcon_m1_pct4_features.py`
- `streaming_calibration_exp/tests/test_training_infra.py`

`falcon_m1_source_only_teacher` was added because the visible local checkpoint named `m1_epoch50` was verified to be an M2 checkpoint and is invalid for M1 PCT4.

## CPU Audit

Command:

```bash
docker exec -w /home/ial-mohd/workspace/iBCI/streaming_calibration_exp docker-ial-mohd   /home/ial-mohd/workspace/envs/spint/bin/python scripts/audit_m1_pct4_all_heldin.py   --data-dir /home/ial-mohd/dataset/ial-mohd/000941 --support-trials 10
```

| session | trials | channels | rank | condition | reach bins min/med/max | post bins min/med/max | NaN | abs mean | abs max |
|---|---:|---:|---:|---:|---|---|---:|---:|---:|
| 20120924 | 414 | 64 | 3 | 5.474933 | 6/8.5/12 | 51/55/59 | 0 | 0.180260 | 1.801924 |
| 20120926 | 409 | 64 | 3 | 5.198521 | 7/8/13 | 52/55/82 | 0 | 0.362181 | 2.715236 |
| 20120927 | 376 | 64 | 3 | 5.428911 | 6/7/14 | 50/54/68 | 0 | 0.330376 | 2.606750 |
| 20120928 | 412 | 64 | 3 | 5.750384 | 7/10/14 | 52/54/58 | 0 | 0.223425 | 1.892864 |

Interpretation: PCT4 is constructible on all held-in sessions with rank-3 direction design, nonempty reach/post windows, and no NaNs. This supports running the source-only R2 matrix, but it is not evidence of decoder improvement.

## Smoke Runs

Teacher smoke:

```bash
python src/train.py experiment=falcon_m1_source_only_teacher   data.data_dir=/home/ial-mohd/dataset/ial-mohd/000941   data.num_workers=0 data.pin_memory=false trainer=default trainer.max_epochs=1   +trainer.limit_train_batches=1 +trainer.limit_val_batches=1 test=false
```

Result: passed on CPU. Log confirmed `Held-out dataset skipped for this stage.` Checkpoint shape is M1-compatible.

PCT4 student smoke:

```bash
python src/train.py experiment=b3s_pct4_m1_loso_internal   data.data_dir=/home/ial-mohd/dataset/ial-mohd/000941 data.loso_fold=0   data.num_workers=0 data.pin_memory=false trainer=default trainer.max_epochs=1   +trainer.limit_train_batches=1 +trainer.limit_val_batches=1 test=false   model.teacher_ckpt_path=<source-only teacher smoke checkpoint>   require_baseline_validation=false
```

Result: passed on CPU. This validates estimator integration, LOSO fold selection, source-only PCT4 normalization, side-feature tensor wiring, and teacher/student shape compatibility.

## GPU Record

Before formal training, GPUs were checked with `nvidia-smi`. GPU 1 was occupied and intentionally avoided. GPUs 0, 2, and 3 were also not available for clean PCT4 training at the final check: GPU0 and GPU3 had active compute load, while GPU2 had approximately 27 GB allocated by another process. No long-running PCT4 training was launched to avoid interfering with other jobs.

## Tests

Passed:

```bash
pytest -q tests/test_falcon_m1_pct4_features.py tests/test_falcon_t4_features.py   tests/test_side_feature_encoder.py tests/test_training_infra.py
# 72 passed, 17 skipped

python -m py_compile src/data/falcon_m1_pct4_features.py src/data/falcon_datamodule.py   src/models/falcon_module.py scripts/audit_m1_pct4_all_heldin.py   scripts/aggregate_m1_pct4_loso.py
```

Additional focused rerun after log-std cleanup:

```bash
pytest -q tests/test_training_infra.py tests/test_falcon_m1_pct4_features.py
# 19 passed
```

## Result Analysis

PCT4-v1 passes constructibility and integration checks. The CPU audit shows the first-10 support trials are direction-rank sufficient on all four held-in sessions, and the phase windows have adequate retained bins. The smoke runs show source-only data loading correctly skips official held-out and that `pct4` side features can train a B3S student with an M1-shaped teacher.

No per-fold/per-seed R2 matrix exists yet, so the required paired deltas cannot be computed:

- `PCT4-Z4`: unavailable
- `PCT4-RS`: unavailable
- `PCT4-LS`: unavailable
- `PCT4-T4`: unavailable
- `PCT4-baseline`: unavailable

Promotion gate: **not evaluated / fail to promote**. The correct next step is to train a clean source-only M1 teacher when GPU is available, then run the frozen seed-42 matrix. If that matrix is negative or indeterminate, stop PCT4-v1 and keep official held-out closed.

## Resume Commands

Train clean M1 teacher when a non-GPU1 card is genuinely idle:

```bash
CUDA_VISIBLE_DEVICES=<gpu> /home/ial-mohd/workspace/envs/spint/bin/python src/train.py   experiment=falcon_m1_source_only_teacher   data.data_dir=/home/ial-mohd/dataset/ial-mohd/000941   data.num_workers=0 data.pin_memory=false seed=42
```

Then run each source-only cell with the selected teacher checkpoint:

```bash
CUDA_VISIBLE_DEVICES=<gpu> /home/ial-mohd/workspace/envs/spint/bin/python src/train.py   experiment=<arm_config> data.data_dir=/home/ial-mohd/dataset/ial-mohd/000941   data.loso_fold=<0..3> seed=42 data.num_workers=0 data.pin_memory=false   model.teacher_ckpt_path=<source-only-M1-teacher.ckpt> require_baseline_validation=false
```

Aggregate only after all cells exist:

```bash
python scripts/aggregate_m1_pct4_loso.py --seeds 42 --folds 0 1 2 3
```
