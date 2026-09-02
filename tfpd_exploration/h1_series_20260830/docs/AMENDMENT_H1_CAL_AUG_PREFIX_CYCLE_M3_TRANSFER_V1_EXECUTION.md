# Execution Amendment: H1 CAL-AUG Prefix-Cycle M3 Transfer V1

This additive amendment authorizes formal evaluation under the unchanged `H1_CAL_AUG_PREFIX_CYCLE_M3_TRANSFER_V1` work order and CPU-reviewed protocol. It authorizes no model training and no other experiment.

## Predecessor and model immutability

Only the ten sealed Experiment-4 epoch-49 checkpoints are eligible: the matched T0/C1 pair corresponding to each of `19250108/13/15/19/20`. Before target access, the execution must verify every checkpoint SHA, embedded provenance, terminal SHA, config SHA, epoch, state hash and paired-integrity receipt. A model from one date may not be used on another date, and no checkpoint may be chosen using performance.

Optimizer steps, backward steps, parameter updates, warm starts and checkpoint selections are fixed at zero. There is no seed, epoch, checkpoint, budget or hyperparameter sweep. T0/C1 model state hashes must be identical before and after evaluation.

## Calibration, causality and metrics

Each outer-date target recording uses its earliest three legal chronological trials for both identity and the existing `fit_deployment_carrier(..., 3 trials)` estimator. The q, lambda, PCA, EB prior and normalizer remain the corresponding outer-date source-only frozen authority; none is refit from target data.

Trials 1–3 are calibration-only. Trial 4 is the first query trial. Causal window starts are no earlier than the trial-4 boundary, every prediction uses a complete W700 history, only last-bin predictions divided by 20 are scored, and the eval mask is applied only at scoring. EMA, TTA and postprocessing are prohibited.

R² is computed independently per recording as float64 seven-output variance-weighted R². Bins from different recordings are never pooled. Date results are equal-recording means and the aggregate is an equal mean across five dates. Every recording and date reports T0, C1 and C1−T0.

## Pre-registered interpretation

- `STRONG_M3_PREFIX_EXTRAPOLATION`: equal-date mean delta is at least +0.01 and at least four of five date deltas are strictly positive.
- `SUPPORT_M3_PREFIX_EXTRAPOLATION`: equal-date mean delta is strictly positive and at least four of five date deltas are strictly positive, but the strong threshold is not met.
- Otherwise: `NO_CLEAR_M3_PREFIX_EXTRAPOLATION`.

These labels are descriptive secondary-diagnostic interpretations. They do not alter the sealed M4 governing result.

## Data and execution boundary

Only the registered held-in outer-date recordings may be opened. All 14 H1 held-out-calib recordings remain forbidden and must have an access count of zero. This diagnostic is not official held-out R² and cannot trigger M2/M3 all-source training or another successor.

The mandatory order is CPU gate, immutable attempt, ten-checkpoint predecessor authority, five M3 evaluation cells, and terminal recomputation. There is no automatic retry. Prediction caches, manifests, per-date access/metric receipts, aggregate terminal and experiment record are mandatory.
