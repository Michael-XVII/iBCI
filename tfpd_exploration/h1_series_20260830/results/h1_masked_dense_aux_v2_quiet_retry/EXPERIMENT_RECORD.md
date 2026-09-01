# H1 Masked Dense-Auxiliary V2 — Experiment Record

- Status: `STOP_SOURCE_GATE_NO_TARGET_ACCESS`
- Branch: `exp/h1-masked-dense-aux-v2-final-legal-subset`
- Execution commit: `836bbe12763f8b0ce4065b8a88dbd7a12c49d1f0`
- Required V1 terminal commit: `21033795c0a7c175611ab6ff6c9bab48728bae4a`
- Interpreter: `/home/ial-mohd/workspace/envs/spint/bin/python`
- Result root: `/home/ial-mohd/workspace/iBCI/tfpd_exploration/h1_series_20260830/results/h1_masked_dense_aux_v2_quiet_retry`
- Per-cell logs: `/home/ial-mohd/workspace/iBCI/logs/h1_masked_dense_aux_v2_quiet_retry`

## Registered V2 change

Training uses the final-legal subset: final eval-mask true and non-still. T0 and all positive-lambda arms use identical filtered indices and samplers. Held-source validation and one-shot outer evaluation retain the original unfiltered four-field loader and last-bin variance-weighted R2 population.

The source attrition gate precedes CUDA: every recording must retain at least 25%, no recording may be empty, every represented trial must retain a window, and every retained final must satisfy the frozen contract.

For each 50-epoch source cell, held-source-date last-bin R2 is evaluated and recorded at every epoch 0 through 49; only epoch 49 governs selection.

## Source selection

```json
{
  "candidate_equal_date_mean_r2": {
    "0.1": 0.22135013341903687,
    "0.3": 0.24644045233726503,
    "1.0": 0.22620488703250885
  },
  "created_at_utc": "2026-08-31T22:35:15.859419+00:00",
  "every_epoch_held_source_r2_recorded": true,
  "mean_delta_r2": 0.025035257637500762,
  "paired_delta_r2_by_date": {
    "19250108": -0.010091066360473633,
    "19250113": 0.02958160638809204,
    "19250115": -0.013499468564987183,
    "19250119": 0.08000502735376358,
    "19250120": 0.03918018937110901
  },
  "positive_dates": 3,
  "schema": "h1_masked_dense_aux_v2_quiet_retry",
  "selected_lambda": 0.3,
  "source_gate_passed": false,
  "terminal_epoch_zero_based": 49,
  "thresholds": {
    "mean_delta_r2_min": 0.01,
    "positive_dates_min": 4,
    "worst_date_delta_r2_min": -0.02
  },
  "verdict": "STOP_SOURCE_GATE_NO_TARGET_ACCESS",
  "worst_date_delta_r2": -0.013499468564987183
}
```

## GPU authorization conclusion

Physical GPUs 0–3 are user-authorized, with at most two idle devices used. GPU smoke is authorized only after the CPU/no-data and source attrition gates. Target fold-0 remains closed until the source lambda gate passes; the fourteen formal held-out recordings are never opened.
