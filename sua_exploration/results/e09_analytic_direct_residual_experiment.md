# E09 Analytic + Direct Residual

## Status

- Branch: `exp/e09-analytic-direct-residual`
- Seed: 42
- Dataset/split: DANDI 000688 sub-C CO SUA, chronological 37/8/8, matched to E01 and E08
- State: implementation verified; formal training pending launch
- Formal log: `logs/e09_analytic_direct_residual_t4_s42.log`
- Formal result: `sua_exploration/results/p3_e09_analytic_direct_residual_t4_s42_seed42.json`

## Locked analytic branch

- Source: completed E08 B0-2 receipt `e08_analytic_only_t4_seed42.json`.
- Formula: ridge OLE with `W=I`, source-selected lambda 100, source-fixed gain 0.3090719545.
- Activity: mean firing rate over the causal 50-bin (1.0 s) window.
- T4: raw `[a,c,m,b]` reconstructed from source-only normalization statistics.
- Target session: first 50 rewarded trials construct T4; no target optimizer, gradient, embedding, or continuous-velocity fit.

## Residual branch and controls

- B-Base: E01 original T4 network, mean target R2 0.6137.
- B-Ana: E08 B0-2, mean target R2 0.2056.
- B-Res: matched B3S/T4 coupled network outputs `delta_v`; final output is `v_ana + delta_v`.
- Objective: `MSE(v_ana + delta_v, v)`, exactly equivalent to training against `v-v_ana`.
- Initialization: decoder output head is exact zero, so the initial model equals B-Ana.
- B-Res-Zero: force `delta_v=0`; exact analytic-only degeneration path.
- B-Res-Shuffle: row-permute only the analytic carrier while keeping the residual network's T4 aligned.
- Diagnostics: per-session R2 for B-Res/B-Ana/B-Res-Shuffle plus residual-to-target energy fraction.

## Decision gate

E09 is a bounded seed-42 hypothesis test. Continue to E10 only if the residual is meaningfully smaller/more stable than the full target signal and B-Res improves external or worst-session behavior beyond a mere reparameterization of E01. If the residual carries almost all target energy or B-Res does not beat the matched T4 baseline, close Variant B.

## Verification

- Focused and adjacent regression tests: 9 passed.
- GPU 1 smoke: completed one epoch with 8 train batches, checkpoint reload, and bounded validation/test.
- Smoke controls emitted: B-Res, B-Ana/B-Res-Zero, B-Res-Shuffle, and residual-energy fraction.
- Smoke scores are wiring-only and are not scientific results because only a tiny fraction of batches was evaluated.
