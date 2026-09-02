# H1 CAL-AUG All-Source Local Held-Out R² V1 Work Order

Status: pre-registered additive successor with a mandatory pre-training metadata-feasibility gate. The first authorized execution stage ends after that gate and awaits explicit user authorization before GPU training.

## Frozen predecessor and boundaries

- Base branch authority: `exp/h1-cal-aug-prefix-cycle-eval-a1` at sealed commit `c60052c9d8ccb8391d6ce53bde9ccfb4f2319884`.
- Experiment-4 A1 terminal SHA-256: `dc9e7ab44954d3d193f67f9bf8936aafdaf2b05be9968d5e0091c0b0ecf092fd`.
- The existing Experiment-4 source, evaluator, work orders, results and receipts are frozen and must not be edited.
- New canonical result root: `tfpd_exploration/h1_series_20260830/results/h1_cal_aug_all_source_heldout_v1/`.
- This experiment reports local public H1 held-out-calibration R², not an EvalAI hidden-test score.

## Pre-training metadata-feasibility amendment

The binding amendment is `docs/AMENDMENT_H1_CAL_AUG_ALL_SOURCE_HELDOUT_V1_METADATA_FEASIBILITY.md`. Before any GPU training, prediction, R² calculation, carrier fitting, source-plan fitting or normalizer fitting, an immutable attempt and CPU gate must precede one metadata-only access to all 14 registered held-out-calib NWBs.

The audit may read only `TrialNum` and evaluation validity (`eval_mask`, or legacy `Blacklist`). It records only session name, legal trial count, and whether the count is at least five. It may not persist trial-level arrays. Full `load_nwb()` is forbidden in this stage.

- If all 14 recordings have at least five legal trials, publish `PASS_H1_ALL_SOURCE_HELDOUT_M4_METADATA_FEASIBILITY`, stop this execution stage, and await user authorization.
- If any recording has fewer than five, publish `STOP_H1_ALL_SOURCE_HELDOUT_M4_PROTOCOL_INFEASIBLE`, do not train, and do not fall back to M3.

## Registered data partitions

All-source training, if later authorized, uses exactly the 13 registered `sub-HumanPitt-held-in-calib` sessions. Source plan, q/lambda selection, carrier cache, normalizer, windows and schedules must use only those 13 recordings.

Held-out governing evaluation uses exactly the 14 registered sessions in `sub-HumanPitt-held-out-calib`, grouped into seven dates: `19250126/27/29` and `19250202/03/06/09`. Scoring data may be opened only after both epoch-49 checkpoints and pair integrity are frozen. The metadata-only feasibility access is the sole pre-training exception.

## Frozen matched training estimand

If continuation is later authorized:

- Fit a new all-source H-C plan from all 13 held-in sessions using the existing source-inner date-LODO q/lambda grid and tie-break; do not reuse a five-date outer plan.
- Fit the source RMS normalizer from held-in carriers only.
- T0 always uses a scheduled contiguous M7 identity prefix.
- C1 uses the deterministic balanced M7/M5/M4 identity-prefix cycle from the same scheduled M7 block.
- Both arms use the earliest M4 of that same M7 block for the analytic H-C carrier.
- Batch order, scheduled M7 block, query, target, carrier bytes, fresh initial state, optimizer and dynamic-dropout probability sequence must match exactly between arms.
- Model: `H1CarrierIdSpint`, h=32, W=700, 10,947,836 parameters.
- Seed 42, batch 32, Adam 5e-5, weight decay 0, FP32, 50 epochs, last-bin MSE, dynamic dropout U(0,1).
- No validation, early stopping, SWA, warm start or checkpoint selection; preserve epoch 49 only.
- GPU0 and GPU1 may train T0/C1 concurrently only if the implementation preserves exact paired random streams and byte-matched schedules; otherwise run them serially on separate fixed devices with pre-materialized matching authorities.

## Frozen held-out evaluation estimand

Only after pair integrity passes:

- Governing budget is M4 only; no M5/M7 sweep or post-target selection.
- Earliest four legal trials supply both identity and analytic H-C carrier.
- Trials 1–4 are calibration only and never scored.
- Scoring begins at trial 5, uses complete W700 causal history, raw outputs and eval-valid output bins only.
- No EMA, optimizer, backward, TTA, parameter update or target-driven selection.
- Metric is float64 seven-output variance-weighted R² per recording.
- Report all 14 recording scores, seven equal-recording date means, 14-session mean/std, C1 minus T0, and positive recording/date counts. Do not pool bins across recordings for the governing mean.

## Execution and artifacts

All new JSON/Markdown receipts are immutable mode 0444 with SHA-256 sidecars. Arrays, checkpoints and raw logs remain Git ignored with tracked sidecars. Attempt precedes each authorized data boundary. Every phase is terminal-or-failure and does not automatically retry or change the budget.

GPU execution, when separately authorized, must use a detached `nohup + setsid` supervisor independent of Codex, write quiet START/epoch/TERMINAL or ERROR summaries under `logs/h1_cal_aug_all_source_heldout_v1/`, and emit no progress bar.

Engineering success is defined by protocol completion, not whether held-out delta is positive. Scientific status is separately recorded as positive transfer, no transfer, or negative transfer.
