# E08 Analytic-only Decoder

## Status

- Branch: `exp/e08-analytic-only`
- Seed: 42 (the implementation is closed-form and deterministic)
- Dataset: DANDI 000688, sub-C, CO, SUA
- Split: chronological 37 source / 8 validation-unused / 8 target-test sessions, matched to E01
- State: complete
- Started: `2026-08-27T16:06:25+08:00`
- Completed runtime: 772.4 seconds
- Formal log: `logs/e08_analytic_only_seed42.log`
- Formal result: `sua_exploration/results/e08_analytic_only_t4_seed42.json`

- Formal result: B0-1 mean R2 = -0.0216; B0-2 mean/median/worst R2 = 0.2056 / 0.2302 / 0.1093, with 8/8 positive target sessions.
- E01 comparison: mean R2 = 0.6137; B0-2 mean paired delta = -0.4080 and was lower on every target session.
- Direction/speed diagnostic: B0-2 median angular error = 55.0 degrees, median speed ratio = 0.859, mean vector-scale slope = 0.183.
- Decision: partially supported. B0-2 is a non-trivial but weak anchor; proceed only to a minimal E09 direct-residual test. Do not use B0-1, and do not proceed to E10 unless E09 shows that the residual remains smaller/more stable than the full target signal.
## Matched protocol

- T4 carrier: raw `[a, c, m, b]` computed from the first 50 rewarded trials.
- Evaluation: only windows from rewarded trials `[50:]`; the calibration prefix is disjoint.
- Window activity: mean firing rate over the causal 50-bin (1.0 s) window; target is cursor velocity at the final bin, matching the neural decoder's last-timestep objective.
- B0-1: population-vector-like `sum_i((r_i-b_i) beta_i)`.
- B0-2: ridge OLE with `W=I` and a 2x2 pseudoinverse.
- Hyperparameters: B0-2 lambda is selected by leave-one-source-session-out R2; each decoder's single isotropic zero-intercept speed gain is fitted on source sessions only.
- Target session: no backpropagation, optimizer, new embedding, or continuous-velocity fit. Prefix endpoint `target_dir` labels are used only to construct T4. Continuous target velocity is read only after predictions are fixed, for offline scoring.
- Compute: CPU-only closed-form evaluation. GPU 1 is intentionally unused because this workload is data I/O plus 2x2 linear algebra, not training.

## Required outputs

The JSON record contains per-session and aggregate mean/median/worst R2, positive-session count, paired delta versus E01, direction cosine/angular error, speed-scale diagnostics, selected lambda/gains, target-side compute, parameter count, source/target session receipts, and a leakage self-audit.

## Smoke test

- Command scope: first 2 source sessions, first 1 test session, lambdas `{0, 1, 100}`.
- Tests: 3 focused unit tests passed.
- Result status: `dev_smoke_complete`.
- B0-1 R2 on smoke target: -0.7494.
- B0-2 R2 on smoke target: +0.1241.
- Interpretation: smoke values are an execution check only; they are not E08 conclusions because source selection used only two sessions and the lambda landed on the reduced grid boundary.
