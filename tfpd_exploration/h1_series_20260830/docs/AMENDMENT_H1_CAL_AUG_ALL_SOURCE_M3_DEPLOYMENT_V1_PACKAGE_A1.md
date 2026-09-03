# Additive Amendment — H1 All-Source M3 Deployment Package A1

Status: authorized successor to the immutable packaging failure in `h1_cal_aug_all_source_m3_deployment_v1`.

The predecessor successfully sealed its all-source authority, two epoch-49 checkpoints and pair-integrity receipt, then stopped before publishing any calibration payload, package, minival prediction or aggregate terminal. Its `packages_failure.json` SHA-256 is `71a6915f7f1273d2dd78b71b47c5456957acaf020f26de6fc2c33c0c2511576a`.

The failure was caused by the successor-specific held-out loader calling the sealed M4 helper `_ordered_eval_trials`, which rejects fewer than five trials. This was an implementation mismatch: deployment is pre-registered as M3, and all 14 held-out-calib recordings have exactly three legal trials. A new local parser will require exactly/at least three finite chronological eval-valid TrialNum values without imposing an M4 query-trial requirement. It does not modify the sealed M4 helper or predecessor root.

This successor:

1. verifies and binds the predecessor failure, source authority, pair-integrity, both training terminals and both checkpoint SHA/state provenances before calibration access;
2. performs no training, optimizer construction/step, backward, model update, warm start, checkpoint selection or retry;
3. reconstructs the frozen all-source plan/prior/normalizer from predecessor authority;
4. materializes M3 identity/carrier payloads for 13 held-in and 14 held-out calibration sessions using earliest three trials and `fit_deployment_carrier(..., 3 trials)`;
5. packages the unchanged T0/C1 checkpoint states with the additive carrier-aware decoder;
6. runs local held-in-minival through serialized-then-reloaded packaged decoders, followed by CPU/GPU/batch/state rehearsal;
7. publishes a successor terminal and stops without EvalAI submission.

The fresh result root is `tfpd_exploration/h1_series_20260830/results/h1_cal_aug_all_source_m3_deployment_v1_package_a1/`; quiet logs are under `logs/h1_cal_aug_all_source_m3_deployment_v1_package_a1/`. The original failed root remains untouched.

All minival metrics retain the exact label `local held-in-minival deployment sanity R²`. They are not held-out R² and cannot trigger training or tuning. `evalai push`, remote test and official score access remain forbidden.
