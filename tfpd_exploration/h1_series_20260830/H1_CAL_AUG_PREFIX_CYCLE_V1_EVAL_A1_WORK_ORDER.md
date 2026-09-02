# H1 CAL-AUG Prefix-Cycle V1 Evaluation Amendment A1

## Scope

This additive amendment completes only the target evaluation and terminal verification of the ten successfully trained V1 epoch-49 models. It does not retry training, alter the sealed V1 result root, or replace its immutable aggregate failure receipt.

## Fixed predecessor authority

- Sealed training commit: `a162018f5195fc30578a6d3e8de43a216b216937`.
- V1 attempt SHA-256: `bf5bf46e0a56780a17a038468c8526553a882048e744bc28dad86338e3476b02`.
- V1 aggregate failure SHA-256: `b2b83a6c4b85023d941833fa4f2fc531fe8f61b0e1ed919f82b26af8f663a4f1`.
- V1 source authority SHA-256: `4763533232a8676dd5bab2c991ebaf25982c34feee78b4e1730097cae9416c60`.
- Regeneration and Experiment 3 authorities remain those fixed by the V1 work order.

Before CUDA initialization or target loading, A1 must publish its own immutable attempt and revalidate all five V1 paired-integrity receipts and all ten checkpoint, config, terminal, embedded provenance, initial-state, terminal-state, step-count, dropout and epoch-49 bindings. The V1 `training_failure.json` must describe only the known duplicate immutable receipt publication and must record zero target access.

## Evaluation protocol

- Outer dates remain `19250108/13/15/19/20`; arms remain fresh matched `T0/C1` from the sealed V1 training run.
- Each target recording is evaluated independently at M4, M5 and M7.
- Identity uses the earliest M legal trials. The analytic H-C carrier always uses the earliest four trials.
- Strict causal raw-output inference begins at trial M+1 with complete W700 history.
- No EMA, optimizer, backward, TTA, model update, target-driven selection, checkpoint selection or retraining is permitted.
- R2 is float64 seven-output variance-weighted R2: per recording, equal-recording within date, equal-date over the five dates.
- Governing verdict is unchanged from V1:
  - `PASS_H1_CAL_AUG_PREFIX_CYCLE_TRANSFER` iff M4 equal-date delta is at least `+0.01`, at least four of five M4 date deltas are strictly positive, and M5/M7 equal-date deltas are each at least `-0.01`.
  - Otherwise `COMPLETE_H1_CAL_AUG_PREFIX_CYCLE_NO_TRANSFER`.

## Execution and artifacts

- Result root: `tfpd_exploration/h1_series_20260830/results/h1_cal_aug_prefix_cycle_v1_eval_a1/`.
- Log root: `logs/h1_cal_aug_prefix_cycle_v1_eval_a1/`.
- CLI defaults to dry-run and exposes `--prepare-authority`, `--evaluate`, `--verify-terminal`, `--detached-supervisor`, explicit roots, `--physical-gpus`, and `--allow-shared-gpus`.
- GPU0-3 may be shared. A detached `nohup + setsid` supervisor is required; Codex must not own a persistent PTY or tail logs.
- Progress bars are prohibited. Each date log contains only START, TERMINAL or ERROR summaries plus library warnings.
- Any cell failure is immutable and stops new work; no automatic retry occurs.
- JSON/Markdown receipts and SHA sidecars are mode 0444. Prediction NPZ and raw logs remain Git ignored and are represented by SHA sidecars.

## Acceptance

1. CPU/no-data gate passes with zero CUDA/NWB/target access.
2. Training authority revalidates all ten sealed checkpoints without writing to the V1 root.
3. Five date-isolated target evaluations terminate with immutable model states and zero optimizer/backward/update/selection activity.
4. Prediction caches, metrics and access audits pass terminal recomputation and SHA verification.
5. A1 publishes exactly one governing terminal verdict and experiment record.

The claim is limited to the five sealed H1 matched date-LODO T0/C1 pairs and the registered M4/M5/M7 raw-output surface.
