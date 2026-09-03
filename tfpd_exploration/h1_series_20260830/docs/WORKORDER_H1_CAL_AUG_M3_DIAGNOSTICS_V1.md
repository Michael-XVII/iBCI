# H1 CAL-AUG M3 Diagnostics V1

Status: evaluation-only diagnostic successor of sealed H1 CAL-AUG V1.

This work order does not alter or supersede the frozen T0/C1 epoch-49
checkpoints, H-C authority, M3 deployment protocol, Docker images, or official
EvalAI results. Training, checkpoint selection, optimizer/backward/update,
Docker rebuild, and new EvalAI submission are forbidden.

## D1: official H1 grouping

Reuse the sealed local held-out-calib prediction/target cache. For each arm,
concatenate eval-valid predictions and targets from `Sx_set_1` and
`Sx_set_2` before computing one float64 seven-output variance-weighted R2 for
each of S6--S12. Report the arithmetic mean and population standard deviation
(`ddof=0`) across the seven sessions. The implementation must call and match
FALCON `FalconEvaluator.compute_metrics_regression()`.

## D2: calibration-target reuse optimism

Use the frozen packaged T0/C1 models and frozen M3 identity/carrier payloads.
For the 13 held-in-calib recordings, score only eval-valid bins whose TrialNum
belongs to the exact three trials registered in calibration authority. Group
sets into S0--S5 before R2. Compare with the independent held-in-minival stream
using its sealed strict complete-W700-history score mask. Also report the
held-in-minival official zero-prefix/eval-mask surface as a sensitivity check.
Optimism is `reuse R2 - independent R2`; no result may trigger training.

The result root is
`results/h1_cal_aug_m3_diagnostics_v1/`. Receipts are immutable and paired
with SHA-256 sidecars; binary caches remain local. The terminal may recommend
whether a separately preregistered M3-aware V2 is scientifically motivated,
but must not create or run that successor.
