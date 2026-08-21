# Template-Ridge D-b Held-Out SPINT-Style Experiment (2026-08-14; completed 2026-08-20)

## Status

Completed for the six-arm `seed=42` matrix. This is a negative result for Template-Ridge D-b v1: ordinary T4 reaches `test_heldout/r2_mean=0.613659`, while real Template-Ridge (`tr4`) reaches `-0.098714` and fails the frozen promotion gate.

The intended three-seed confirmation was not run. This document reports one seed and eight final held-out sessions; it is not a multi-seed efficacy claim.

## Branch And Environment

- Branch: `exp/template-ridge-db-heldout-spint`
- Base `HEAD` at analysis: `799d30d971a71fa30095b03bcb6009ebfc2f6c5c` (`docs: record 14-fold RT sparse checkpoint`). The Template-Ridge implementation and this record were still uncommitted in the working tree, so this SHA is a branch-lineage anchor rather than a complete source snapshot.
- Repository: `/home/ial-mohd/workspace/iBCI`
- Container: `docker-ial-mohd`
- Python: `/home/ial-mohd/workspace/envs/spint/bin/python`
- Data: `/home/ial-mohd/dataset/ial-mohd/000688/sub-C`
- GPU use: local teacher GPU 1; `t4`, `ts4`, `tr4`, `trls4`, and `trz4` GPU 3; `trs4` GPU 2.
- PyTorch emitted an RTX 5090 `sm_120` architecture warning. The runs completed CUDA training and testing without a kernel failure.

## Protocol

This is SPINT-style held-out-selected, not blind held-out. Held-in train sessions provide training loss. Validation/test held-out sessions are not used for backward gradients, but `val_heldout/r2_mean` is used for checkpoint selection and early stopping. The primary metric is `test_heldout/r2_mean` from the selected checkpoint.

The default MC_Maze teacher checkpoint documented elsewhere in the repository was not present on this machine, and the original `000128/sub-Jenkins` teacher-training data was also absent. The self-contained local DANDI688 teacher was trained with the same split and held-out selection. All six formal students use `teacher_dandi688_co_heldout_spint_seed42/best-epoch=020-val_heldout/r2_mean=0.1045.ckpt` (SHA256 `86ef4ef55eda8a0c07dad860aebe6f58f8481b59847033e144e35718fcb18adb`).

The current DANDI `test_dataloader()` deliberately passes the same official test loader through the `test_heldin` and `test_heldout` metric namespaces. Consequently, those two values are identical in these artifacts. Only `test_heldout/r2_mean` is interpreted here; `test_heldin/r2_mean` is not an independent held-in result.

## Session Split

- Split counts: 37 train, 8 validation-heldout-selection, 8 test-heldout.
- Train first/last: `sub-C_ses-CO-20131003` to `sub-C_ses-CO-20151119`.
- Validation held-out selection sessions: `sub-C_ses-CO-20151120`, `sub-C_ses-CO-20151201`, `sub-C_ses-CO-20160909`, `sub-C_ses-CO-20160912`, `sub-C_ses-CO-20160914`, `sub-C_ses-CO-20160915`, `sub-C_ses-CO-20160919`, `sub-C_ses-CO-20160921`.
- Test held-out sessions: `sub-C_ses-CO-20160923`, `sub-C_ses-CO-20160929`, `sub-C_ses-CO-20161005`, `sub-C_ses-CO-20161006`, `sub-C_ses-CO-20161007`, `sub-C_ses-CO-20161011`, `sub-C_ses-CO-20161013`, `sub-C_ses-CO-20161021`.

## Estimator

Template-Ridge D-b v1 learns a source-only go-cue aligned scalar speed template from train sessions only. For each target session, the first 50 chronological rewarded support trials provide direction labels, timestamps, and neural counts. The synthetic target is `Y_synth(t)=s(t-go_cue)[cos(theta), sin(theta)]`; weighted ridge uses `s(t-go_cue)^2` weights. The fitted `50N -> C` ridge coefficients are reduced to a per-unit `[N,4]` descriptor:

`[template_cos_weight, template_sin_weight, template_norm, support_rate]`.

The audit receipt shows the template used 37 train sessions and 1,701 source support trials. No held-out session contributed to template fitting or side-feature normalization.

Arms:

- `t4`: existing T4 descriptor.
- `ts4`: row-shuffled T4 control.
- `tr4`: real Template-Ridge D-b descriptor.
- `trs4`: dimension-matched Template-Ridge row-shuffle control.
- `trls4`: label/template direction shuffle null.
- `trz4`: exact-zero 4D floor reference.

The legacy `baseline` artifact is excluded from this matrix because it predates the final local teacher and does not carry a valid `test_heldout` metric.

## Completed Checks

- `py_compile` passed for modified Template-Ridge, datamodule, trainer, evaluator, teacher, audit, and aggregate scripts.
- Relevant pytest passed: `86 passed, 15 skipped` for `sua_exploration/tests/test_template_ridge_side_features.py`, `sua_exploration/tests/test_side_feature_label_provenance.py`, and `streaming_calibration_exp/tests/test_side_feature_encoder.py`.
- DANDI teacher smoke passed with small train/val/test batch limits and wrote checkpoint/test metadata.
- TR4 student smoke passed with `max_epochs=1`, `limit_train_batches=2`, `limit_val_batches=2`, and `limit_test_batches=2`. Smoke R2 is not interpreted because it used only two train batches and a provisional teacher checkpoint.

## CPU Audit

- Passed for all 53/53 sessions: 37 train, 8 held-out-selection, and 8 final held-out test sessions.
- Template SHA256: `1a8f5cc3bb73be6d8f272ee42a2056d878f0bacf3b1f7870ec21886321788b50`.
- Alignment event: `go_cue_time` for all 1,848 source-support trials.
- Each session had all eight reach directions. Constructed synthetic rows ranged from 2,230 to 2,500 (mean 2,483.0); labelled support trials ranged from 49 to 50.
- Held-out data were not used for template fitting. Ridge lambda was 1.0.

## Formal Seed42 Results

All six runs completed with `max_epochs=40`, `patience=10`, `calibration_n_trials=50`, and `val_heldout/r2_mean` checkpoint selection. The final test values below are official test-heldout R2 from the selected checkpoint.

| Arm | Best epoch | Best val held-out R2 | Test held-out R2 | Test minus val | Positive test sessions |
|---|---:|---:|---:|---:|---:|
| T4 | 14 | 0.655805 | **0.613659** | -0.042146 | **8/8** |
| TS4 | 10 | 0.203608 | -0.020305 | -0.223913 | 3/8 |
| TR4 | 20 | 0.131992 | -0.098714 | -0.230705 | 2/8 |
| TRS4 | 33 | 0.213182 | -0.130216 | -0.343398 | 2/8 |
| TRLS4 | 15 | 0.209326 | -0.111830 | -0.321155 | 1/8 |
| TRZ4 | 9 | 0.170778 | -0.061625 | -0.232403 | 2/8 |

### Per-Session Final Held-Out R2

| Test session | T4 | TS4 | TR4 | TRS4 | TRLS4 | TRZ4 |
|---|---:|---:|---:|---:|---:|---:|
| `sub-C_ses-CO-20160923` | 0.478565 | -0.243005 | -0.337183 | -0.195150 | -0.335067 | -0.169854 |
| `sub-C_ses-CO-20160929` | 0.704932 | 0.192966 | -0.091993 | -0.118132 | 0.067448 | -0.025811 |
| `sub-C_ses-CO-20161005` | 0.427968 | 0.073050 | 0.010326 | 0.004266 | -0.020552 | 0.016490 |
| `sub-C_ses-CO-20161006` | 0.597291 | -0.038184 | -0.196129 | -0.134143 | -0.167365 | -0.130956 |
| `sub-C_ses-CO-20161007` | 0.688241 | 0.096786 | 0.020575 | 0.109902 | -0.058664 | 0.015992 |
| `sub-C_ses-CO-20161011` | 0.612027 | -0.072751 | -0.095931 | -0.323908 | -0.120269 | -0.043072 |
| `sub-C_ses-CO-20161013` | 0.765325 | -0.072723 | -0.074672 | -0.238985 | -0.082216 | -0.005873 |
| `sub-C_ses-CO-20161021` | 0.634927 | -0.098581 | -0.024704 | -0.145581 | -0.177955 | -0.149918 |

### Paired Final Held-Out Deltas

| Comparison | Mean delta R2 | Median session delta | Positive sessions |
|---|---:|---:|---:|
| TR4 - T4 | **-0.712373** | -0.750688 | 0/8 |
| TR4 - TS4 | -0.078408 | -0.069467 | 1/8 |
| TR4 - TRS4 | +0.031503 | +0.016100 | 5/8 |
| TR4 - TRLS4 | +0.013116 | +0.015941 | 5/8 |
| TR4 - TRZ4 | -0.037088 | -0.059016 | 2/8 |

For context, T4 - TS4 is `+0.633965` and is positive in all 8/8 final test sessions. The ordinary functional descriptor is therefore strongly supported in this protocol; the Template-Ridge descriptor is not.

## Interpretation

- `tr4` is below T4 by `-0.712373` R2 and loses in every final held-out session. It also loses to the exact-zero four-dimensional reference by `-0.037088`.
- `tr4` is slightly above its row-shuffle and label-shuffle controls, but those controls are also negative and the direction signal is not sufficient to make the descriptor useful.
- All non-T4 arms have a much larger selected-validation to final-test drop (`-0.224` to `-0.343`) than T4 (`-0.042`). This is consistent with poor transfer of these alternatives from the held-out selection split to the final held-out split.
- This result does not show that all synthesized-trajectory approaches are impossible. It does show that this frozen Template-Ridge D-b v1 estimator, reduction, weighting, and B3S consumer interface should not be promoted or tuned further under this protocol.

## Promotion Gate

**Failed.** The primary requirement is `TR4-T4 >= +0.03`, all seed means positive, and all held-out session deltas positive. At seed42, `TR4-T4=-0.712373` with 0/8 positive sessions. The primary null comparison also fails the aggregate implementation's `+0.03` and all-session requirement: `TR4-TRLS4=+0.013116` with 5/8 positive sessions. Seeds 43 and 44 were not run, so the 3/3-seed criterion is also incomplete.

The frozen conclusion is **Template-Ridge D-b v1 negative/indeterminate; no promotion and no phase/template/feature retuning using these held-out-selected results.**

## Result Artifacts

- `sua_exploration/results/template_ridge_db_heldout_spint_v1/TEMPLATE_RIDGE_DB_HELDOUT_SPINT_AUDIT.json`
- `sua_exploration/results/template_ridge_db_heldout_spint_v1/TEMPLATE_RIDGE_DB_HELDOUT_SPINT_AGGREGATE.json`
- `sua_exploration/results/template_ridge_db_heldout_spint_v1/TEMPLATE_RIDGE_DB_HELDOUT_SPINT_PER_SEED_SESSION.csv`
- Per-arm result summaries: `sua_exploration/results/p3_template_ridge_db_heldout_spint_{t4,ts4,tr4,trs4,trls4,trz4}_s42_seed42.json`

The reported results are held-out-selected SPINT-style results: held-out validation selected checkpoints and early stopping, while held-out examples never contributed backward gradients. They are not blind official held-out claims.
