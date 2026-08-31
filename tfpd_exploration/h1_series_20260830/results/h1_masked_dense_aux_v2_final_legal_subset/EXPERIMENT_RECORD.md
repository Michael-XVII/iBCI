# H1 Masked Dense-Auxiliary V2 — Experiment Record

- Status: `FAIL_H1_MASKED_DENSE_AUX_V2_NO_RETRY`
- Branch: `exp/h1-masked-dense-aux-v2-final-legal-subset`
- Execution commit: `21033795c0a7c175611ab6ff6c9bab48728bae4a`
- Required V1 terminal commit: `b1dfc9d8b6516eefe41f9dbbe32525b5a72e10fa`
- Interpreter: `/home/ial-mohd/workspace/envs/spint/bin/python`
- Result root: `/home/ial-mohd/workspace/iBCI/tfpd_exploration/h1_series_20260830/results/h1_masked_dense_aux_v2_final_legal_subset`
- Per-cell logs: `/home/ial-mohd/workspace/iBCI/logs/h1_masked_dense_aux_v2_final_legal_subset`

## Registered V2 change

Training uses the final-legal subset: final eval-mask true and non-still. T0 and all positive-lambda arms use identical filtered indices and samplers. Held-source validation and one-shot outer evaluation retain the original unfiltered four-field loader and last-bin variance-weighted R2 population.

The source attrition gate precedes CUDA: every recording must retain at least 25%, no recording may be empty, every represented trial must retain a window, and every retained final must satisfy the frozen contract.

For each 50-epoch source cell, held-source-date last-bin R2 is evaluated and recorded at every epoch 0 through 49; only epoch 49 governs selection.

## Failure

`FileNotFoundError: [Errno 2] No such file or directory: '/home/ial-mohd/workspace/iBCI/tfpd_exploration/h1_series_20260830/results/h1_masked_dense_aux_v2_final_legal_subset/source_cells/source_19250108_lambda_0.1/terminal.json'`

## GPU authorization conclusion

Physical GPUs 0–3 are user-authorized, with at most two idle devices used. GPU smoke is authorized only after the CPU/no-data and source attrition gates. Target fold-0 remains closed until the source lambda gate passes; the fourteen formal held-out recordings are never opened.
