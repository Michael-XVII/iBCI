# M1 PCT4 v1 Held-Out SPINT-Style Experiment

Date: 2026-08-13 Asia/Shanghai  
Branch: `exp/m1-pct4-v1-heldout-spint`  
Final submitted commit SHA: reported in the final run report after commit/push verification; a Git commit cannot contain its own final hash.  
Base/source SHA during runs: `698829c705346b9cf720a1dc0ba50e857ab038a1`  
Environment: Docker container `docker-ial-mohd`, Python `/home/ial-mohd/workspace/envs/spint/bin/python`  
Data path: `/home/ial-mohd/dataset/ial-mohd/000941`

## Protocol

This is a SPINT-style held-out-selected experiment, not a source-only or blind official held-out evaluation.

Training loss used held-in calibration data only. Official `held-out-calib` data was loaded during validation/test to compute `val_heldout/r2_mean` and `test_heldout/r2_mean`. Checkpoint selection and early stopping monitored `val_heldout/r2_mean`. Therefore the held-out set influenced model selection, although it did not directly contribute gradients.

The original plan requested seeds 42, 43, and 44. After seed43 was underway, the protocol was amended by user instruction: stop after seed43 and do not run seed44. The final aggregate therefore uses seeds `42,43`.

## Sessions

Held-in training/calibration sessions:

| Session | Trials in CPU audit |
|---|---:|
| `ses-20120924` | 414 |
| `ses-20120926` | 409 |
| `ses-20120927` | 376 |
| `ses-20120928` | 412 |

Held-out calibration sessions used for validation/test R2 and model selection:

| Session | Support trials in CPU audit |
|---|---:|
| `ses-20121004` | 10 |
| `ses-20121017` | 10 |
| `ses-20121024` | 10 |

## Calibration Semantics

The student models were trained on all held-in calibration sessions. With `include_heldout_in_fit=true`, `FalconDataModule.val_dataloader()` included held-out calibration data, but `train_dataset` remained held-in only. This matches the requested SPINT-style behavior: held-out R2 is visible for checkpointing/early stopping, but held-out samples are not used as loss batches.

PCT4 side-feature normalization was still fitted only from held-in train sessions. Held-out data did not fit PCT4 mean/std. The `pct4_z4` arm emitted an exact zero side tensor.

## PCT4 Estimator

Estimator version: `m1-pct4-v1-event-aligned-bin-end`

PCT4 is `[a_reach, c_reach, a_post, c_post]`, fitted from event-aligned linear features in two fixed phase windows:

| Phase | Window |
|---|---|
| reach | `move_onset_time -> contact_time` |
| post | `contact_time -> stop_time` |

The estimator uses official M1 bin-end semantics and the raw `eval_mask`. Controls are `pct4_rs`, `pct4_ls`, and `pct4_z4`.

## Implementation Notes

Key implementation changes:

| Area | Change |
|---|---|
| DataModule | Side features may now coexist with `include_heldout_in_fit=true` as validation/test only. |
| Training module | `StreamingCalibrationLitModule` now logs aggregate `val_heldout/r2_mean`, `val_heldout/r2_std`, and best mean. |
| Configs | Added held-out SPINT-style teacher/student experiment configs monitoring `val_heldout/r2_mean`. |
| Aggregation | Added `scripts/aggregate_m1_pct4_heldout_spint.py`, now scoped to held-out SPINT-style artifacts. |
| Tests | Added/updated tests for held-out validation without training leakage. |

Validation commands run before training:

```bash
/home/ial-mohd/workspace/envs/spint/bin/python -m pytest -q tests/test_falcon_m1_pct4_features.py tests/test_falcon_t4_features.py tests/test_side_feature_encoder.py
/home/ial-mohd/workspace/envs/spint/bin/python -m py_compile src/data/falcon_m1_pct4_features.py src/data/falcon_datamodule.py src/models/streaming_calibration_module.py scripts/audit_m1_pct4_all_heldin.py scripts/aggregate_m1_pct4_heldout_spint.py
```

Result: `63 passed, 17 skipped`; py_compile passed.

## CPU Audit

Command:

```bash
/home/ial-mohd/workspace/envs/spint/bin/python scripts/audit_m1_pct4_all_heldin.py --data-dir /home/ial-mohd/dataset/ial-mohd/000941 --support-trials 10 --include-heldout
```

Output files:

- `outputs/m1_pct4_v1/M1_PCT4_CPU_AUDIT_HELDIN_HELDOUT.csv`
- `outputs/m1_pct4_v1/M1_PCT4_CPU_AUDIT_HELDIN_HELDOUT.json`

Summary:

| Session | Split | Trials | Rank | Condition | Reach bins median | Post bins median | NaNs | PCT4 abs mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `ses-20120924` | heldin | 414 | 3 | 5.475 | 8.5 | 55.0 | 0 | 0.1803 |
| `ses-20120926` | heldin | 409 | 3 | 5.199 | 8.0 | 55.0 | 0 | 0.3622 |
| `ses-20120927` | heldin | 376 | 3 | 5.429 | 7.0 | 54.0 | 0 | 0.3304 |
| `ses-20120928` | heldin | 412 | 3 | 5.750 | 10.0 | 54.0 | 0 | 0.2234 |
| `ses-20121004` | heldout | 10 | 3 | 5.429 | 9.0 | 55.0 | 0 | 0.3028 |
| `ses-20121017` | heldout | 10 | 3 | 5.199 | 10.5 | 55.0 | 0 | 0.2678 |
| `ses-20121024` | heldout | 10 | 3 | 5.259 | 8.0 | 54.5 | 0 | 0.3539 |

## GPU Usage

Training used GPU 0 and GPU 1 for the final seed43 runs after GPU 2 and GPU 3 were observed to be occupied by unrelated high-memory jobs. Earlier seed42 runs used GPU 0/1/2/3, then conflicted queue jobs were stopped and re-run cleanly on available GPUs. No seed44 jobs were launched after the protocol amendment.

The 5090 devices emitted PyTorch architecture compatibility warnings (`sm_120` not in the installed PyTorch capability list), but training and evaluation completed.

## Teacher And Smoke

The held-out SPINT-style teacher was trained because the expected prior M1 teacher checkpoint path was unavailable. Best teacher checkpoint:

`logs/train/runs/2026-08-12-20-18-33-008327_rid-falcon_m1_heldout_spint_teacher_fNone_s42/checkpoints/best_ckpt/epoch_045.ckpt`

Teacher best-checkpoint test metrics:

| Metric | Value |
|---|---:|
| `test_heldin/r2_mean` | 0.811469 |
| `test_heldout/r2_mean` | 0.656681 |
| selected `val_heldout/r2_mean` | 0.649798 |

Student smoke run: PCT4, seed42, `trainer.max_epochs=1`; `test_heldout/r2_mean=0.640332`, `test_heldin/r2_mean=0.807940`.

## Results

Aggregate files:

- `outputs/m1_pct4_v1/M1_PCT4_HELDOUT_SPINT_AGGREGATE.json`
- `outputs/m1_pct4_v1/M1_PCT4_HELDOUT_SPINT_PER_SEED.csv`

Primary metric: `test_heldout/r2_mean` from the best checkpoint selected by `val_heldout/r2_mean`.

Per-seed held-out R2:

| Seed | Baseline | T4 | PCT4 | PCT4_Z4 | PCT4_RS | PCT4_LS |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.644663 | 0.641151 | 0.650966 | 0.634923 | 0.627978 | 0.626988 |
| 43 | 0.639145 | 0.648717 | 0.651045 | 0.642781 | 0.635484 | 0.644003 |
| Mean | 0.641904 | 0.644934 | 0.651006 | 0.638852 | 0.631731 | 0.635496 |

Per-seed held-in R2:

| Seed | Baseline | T4 | PCT4 | PCT4_Z4 | PCT4_RS | PCT4_LS |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.817290 | 0.819415 | 0.817415 | 0.819876 | 0.816332 | 0.820497 |
| 43 | 0.818321 | 0.822062 | 0.821119 | 0.821418 | 0.821164 | 0.817718 |

Held-out paired deltas:

| Contrast | Seed42 | Seed43 | Mean | Positive seeds |
|---|---:|---:|---:|---:|
| `PCT4 - Z4` | +0.016043 | +0.008265 | +0.012154 | 2/2 |
| `PCT4 - RS` | +0.022988 | +0.015562 | +0.019275 | 2/2 |
| `PCT4 - LS` | +0.023977 | +0.007042 | +0.015510 | 2/2 |
| `PCT4 - T4` | +0.009815 | +0.002329 | +0.006072 | 2/2 |
| `PCT4 - baseline` | +0.006303 | +0.011901 | +0.009102 | 2/2 |

## Conclusion

Under this held-out-selected SPINT-style protocol, PCT4 was the best arm on mean held-out R2 across the two completed seeds. It improved over baseline by `+0.009102` mean R2 and over T4 by `+0.006072` mean R2, with positive paired deltas on both seeds. The strongest control separation was against `PCT4_RS` (`+0.019275`) and `PCT4_LS` (`+0.015510`), suggesting the structured PCT4 feature carries useful signal beyond randomized or least-squares controls in this setup.

This result should not be reported as a blind official held-out result. The held-out calibration sessions were used for validation, checkpoint selection, and early stopping. It is therefore appropriate to describe the outcome as **PCT4-v1 positive under held-out-selected SPINT-style model selection**, with the caveat that only seeds 42 and 43 were run after the protocol was amended.
