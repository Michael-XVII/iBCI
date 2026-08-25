# E04 Posterior Mean + Reliability Logit — CO seed 42

E04 is paired to completed E02 (`test_heldout/r2_mean=0.628706`, versus frozen E01 `0.613659`). It preserves DANDI-000688 CO 37/8/8 sessions, 50 chronological rewarded support trials, the validated teacher, B3S, seed 42, 40 epochs, patience 10 and held-out-selected checkpointing.

`t4rql` derives exactly E03's source-only posterior carrier `[a,c,hypot(a,c),b,q_theta]`, including source-only normalization and the fail-closed `q_theta=-20` rule for near-zero modulation. B3S receives only the normalized first four posterior-T4 columns. The normalized q column is injected independently into each coupled decoder cross-attention layer before softmax: `logit_{l,h,c,i} += gamma_l q_i`. Each `gamma_l=softplus(raw_gamma_l)` is a shared non-negative scalar, initialized at `0.001` and trained only on the 37 source-train sessions.

Target sessions only compute the closed-form posterior/covariance, q_theta and static attention bias; target optimizer, backward and hyperparameter search are all false. E04 neither consumes E03 outputs nor uses a multiplicative activity gate. Its artifact audit and strict E02 pairing are written to `results/e04_posterior_reliability_logit_t4_v1/`; terminal output is retained in `logs/e04_posterior_reliability_logit_t4_seed42.log`.

Stop criterion: seed 42 is development evidence only. Add seeds 43/44 only if E04 improves E02 on external mean, worst session or session stability. Otherwise preserve the negative result without held-out-driven retuning.
