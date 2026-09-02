# H1 CAL-AUG Prefix-Cycle M3 Transfer V1 Work Order

Status: pre-registered, evaluation-only secondary diagnostic. The current authorized stage is limited to work-order review, CPU/synthetic tests, and a no-write dry-run. Real target access and M3 inference require a later explicit authorization.

## Objective and claim boundary

This additive successor evaluates the already sealed Experiment-4 matched T0/C1 epoch-49 checkpoints at a new M3 calibration budget on the same five held-in outer dates: `19250108/13/15/19/20`. It does not train, tune, or select a model. M3 is a secondary transfer diagnostic and cannot replace, reinterpret, reopen, or modify the sealed M4 governing result.

The diagnostic will report per-recording T0/C1 R², per-recording C1−T0, each date's equal-recording T0/C1 means and delta, the five-date equal-date mean delta, and the count of dates with strictly positive delta. No positive-transfer threshold or new governing PASS claim is introduced.

## Sealed authorities

- Experiment-4 A1 branch seal: commit `c60052c9d8ccb8391d6ce53bde9ccfb4f2319884`.
- Experiment-4 A1 terminal SHA-256: `dc9e7ab44954d3d193f67f9bf8936aafdaf2b05be9968d5e0091c0b0ecf092fd`.
- M4 local-heldout feasibility seal: commit `0d0ab2f`.
- M4 local-heldout terminal SHA-256: `3ff971dc576958b13ace990bcca8aea2e8b999e2af2ed50f418296d05f8d5cfc`.
- Metadata terminal SHA-256: `e692db3b744a7831610c2338ce34504ccedc4c8696e4d73333de899ed295b563`.
- The ten existing T0/C1 checkpoints, their paired-integrity receipts, source authority, configs, training steps, state hashes, and checkpoint SHA-256 values remain immutable.
- No file under `results/h1_cal_aug_all_source_heldout_v1/` or the sealed Experiment-4 result roots may be overwritten, reopened for execution, or amended.

The future canonical result root is `tfpd_exploration/h1_series_20260830/results/h1_cal_aug_prefix_cycle_m3_transfer_v1/`. It must be fresh. Future quiet logs belong under `logs/h1_cal_aug_prefix_cycle_m3_transfer_v1/`.

## Fixed M3 evaluation protocol

For each of the five original held-in outer dates, load only that date's registered outer recordings and its already frozen matched T0/C1 epoch-49 checkpoints. No H1 held-out-calib file is in scope.

- Calibration identity is formed from the earliest three legal chronological evaluation-valid trials.
- The H-C carrier must call the existing `fit_deployment_carrier(record, plan, earliest_three_trials)` implementation. No trial may be padded, duplicated, synthesized, or replaced.
- The first query trial is the fourth legal trial. Trials 1–3 are calibration-only and are never scored.
- Candidate causal windows begin no earlier than the first evaluation-valid bin of trial 4. Every prediction uses a complete contiguous `W=700` neural history and the model's last-bin output.
- Inference preserves the sealed Experiment-4 scale conversion `prediction / 20`.
- Predictions are generated continuously over the authorized causal surface; `eval_mask` is applied only when calculating the score.
- Each per-recording score is float64 seven-output variance-weighted R². A recording must provide more than one eval-valid output row.
- T0 and C1 must share byte-identical target arrays, score masks, output-bin indices, M3 identities, M3 carriers, plan, normalizer, and causal surfaces. Only checkpoint state differs by arm.
- Date results are equal-recording means. The cross-date diagnostic is the equal mean of the five date deltas; bins and recordings are never pooled across dates.

## Prohibited operations

- no retraining, optimizer construction/steps, backward, gradient, warm start, parameter update, TTA, checkpoint selection, or budget/hyperparameter selection;
- no checkpoint mutation and no writes into predecessor roots;
- no M4/M5/M7 rerun or target-driven comparison used to choose M3;
- no EMA or filtered outputs;
- no access to `sub-HumanPitt-held-out-calib`, its neural data, behavior labels, or metadata during this successor;
- no modification of the sealed M4 governing verdict or Experiment-4 A1 receipts.

Model state hashes must match before and after every future evaluation cell. Any mismatch is an execution failure, not a retry or selection opportunity.

## Why H1 held-out-calib cannot provide a clean M3/M4 local R²

The sealed metadata feasibility audit established that all 14 H1 held-out-calib recordings contain exactly three legal trials. An M3 carrier consumes all three behavior-labeled calibration trials, leaving no independent trial 4+ post-calibration scoring surface. It is therefore invalid to calculate governing R² on those same three carrier-fitting trials. M4 is also infeasible because a fourth calibration trial and fifth scoring trial do not exist.

Consequently, the current H-C method cannot obtain leakage-free M3 or M4 post-calibration local R² from these held-out-calib files themselves. A genuinely held-out evaluation requires an evaluation/test stream independent of the three calibration trials, or a separately pre-registered M2 carrier successor that leaves trial 3 for scoring. Neither alternative is authorized or selected by this work order.

## Future execution order

After explicit authorization, an additive execution implementation must:

1. publish an immutable attempt before CUDA, checkpoint loading, or outer-date target access;
2. validate the M4 STOP authority, Experiment-4 A1 terminal, ten checkpoint SHA/provenance records, and five paired-integrity receipts;
3. run a source/synthetic GPU smoke with zero target access;
4. evaluate exactly M3 on the five outer dates without altering model state;
5. publish prediction-cache authority, target-access audit, per-date metrics and terminals;
6. verify byte-matched T0/C1 evaluation surfaces, float64 metric recomputation, five-date completeness, state immutability, and zero optimizer/backward/update counts;
7. publish a neutral `COMPLETE_SECONDARY_M3_TRANSFER_DIAGNOSTIC` terminal regardless of delta sign.

Large prediction caches and raw logs remain Git ignored; receipts, experiment records, and SHA sidecars are mode 0444. No automatic retry is permitted.

## Current review-stage acceptance

- Default CLI is dry-run only: zero writes, zero data/NWB access, zero CUDA, zero checkpoint loads and zero model execution.
- CPU tests cover earliest-three support, fourth-trial boundary, complete W700 windows, eval-mask-only scoring, three-trial deployment-carrier invocation, `/20` contract, float64 seven-output R², exact five-date/recording aggregation, and absence of held-out or execution entry points.
- The canonical future result root must not be created during this review stage.
