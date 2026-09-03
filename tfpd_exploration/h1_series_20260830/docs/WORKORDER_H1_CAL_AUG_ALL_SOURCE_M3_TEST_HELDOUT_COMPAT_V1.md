# H1 CAL-AUG All-Source M3 Test-Heldout Compatibility V1

Status: authorized strict evaluation-only additive successor.

## Objective

Reproduce the local semantics of the original SPINT
`FalconLitModule.test_step()` and `on_test_epoch_end()` using only the frozen
all-source T0/C1 epoch-49 packages. This successor does not modify the frozen
checkpoints, Package-A1 root, or the sealed val-heldout-style diagnostic.

The result label is `local SPINT-style held-out-calib test-loop R²`. It is not
the official FALCON hidden-test R². The same public held-out-calib trials supply
the frozen M3 identity/carrier and scoring labels, so the result is not an
independent post-calibration deployment score.

## Frozen scoring contract

- exactly 14 registered H1 held-out-calib sessions (S6--S12);
- W=700 causal windows with W-1 zero-prefix padding;
- eval-valid last-bin targets and predictions divided by 20;
- chronological per-session batches of 32, dropping the incomplete final batch;
- frozen earliest-M3 identity and H-C carrier from Package-A1;
- float64 seven-output variance-weighted R² computed independently per session;
- `test_heldout/r2_mean` is the arithmetic mean of 14 session R² values;
- `test_heldout/r2_std` follows `torch.stack(values).std()`, i.e. sample standard
  deviation (`ddof=1`). Population std (`ddof=0`) is recorded only as an
  auxiliary comparison with the predecessor receipt.

## Independent evaluator and equivalence gate

The compatibility evaluator loads the frozen package state, constructs the same
W700 windows as `FalconDataset`, and performs batched model calls directly. It
does not reuse streaming predictions to produce its test-loop metrics. After
inference, it compares against the sealed val-style prediction cache:

- target bytes and legacy scoring masks must be exactly equal;
- scored predictions must be allclose with `rtol=2e-3`, `atol=2e-4`, matching
  the already sealed package CPU/GPU numerical-equivalence tolerance;
- every per-session R², each arm mean, and C1-T0 mean delta must differ by no
  more than `2e-4`;
- the scored-bin population must be identical for every session and arm.

Any mismatch fails closed and publishes an immutable failure receipt. No retry,
training, checkpoint selection, optimizer, backward, model update, parameter
fit, EvalAI access, upload, or submission is permitted.

The canonical result root is
`results/h1_cal_aug_all_source_m3_test_heldout_compat_v1/`. JSON/Markdown
receipts are mode 0444 with SHA sidecars; prediction caches remain Git-ignored.
