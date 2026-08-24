# E02 Posterior-Mean T4 — CO seed 42

## Frozen protocol

E01 is the completed `template_ridge_db_heldout_spint_t4_s42` T4 artifact: 37/8/8 CO split, 50 chronological rewarded calibration trials, seed 42, B3S, and the shared teacher SHA `86ef4ef55eda8a0c07dad860aebe6f58f8481b59847033e144e35718fcb18adb`. Its final held-out mean R2 is `0.613659`, with 8/8 positive test sessions.

E02 changes only the carrier. For each unit, target calibration uses trial-level rates and `X=[1, cos(theta), sin(theta)]`; it estimates OLS residual variance `sigma_i^2` and emits the posterior mean under an isotropic source-only prior: `w_i=(X'X+diag(0,sigma_i^2/tau^2,sigma_i^2/tau^2))^-1X'y_i`. The resulting four-vector is `[a,c,hypot(a,c),b]`.

`tau^2` is an empirical-Bayes moment estimate from the 37 source-train sessions only. Target validation/test sessions never influence the prior, normalization, hyperparameters, gradients, optimizer, or decoder weights.

## Evidence

The runner writes all terminal output to `logs/e02_posterior_mean_t4_seed42.log`; ignored artifacts are written under `sua_exploration/results/e02_posterior_mean_t4_v1/`. The aggregate reports raw per-session R2, paired deltas, median, worst session, positive-session count, parameter/MAC delta, and target-side compute. This is seed-42 development evidence only.

## Stop rule

Only a positive held-out signal versus frozen E01 (mean, worst session, or stability) authorizes seeds 43/44. A negative result is retained without held-out-driven retuning.
