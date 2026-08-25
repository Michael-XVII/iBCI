# E03 Posterior Mean + Angular Reliability — CO seed 42

E03 extends E02's source-only isotropic posterior T4 carrier with one reliability scalar. It uses the same DANDI-000688 CO 37/8/8 split, 50 chronological rewarded support trials, B3S consumer, teacher SHA, seed 42, 40-epoch budget, and held-out-selected checkpointing.

For posterior direction mean `mu=[a,c]` and directional posterior covariance `Sigma_ac`, E03 emits `q_theta=-log((u_perp^T Sigma_ac u_perp)/(||mu||^2+eps)+eps)`, where `u_perp=[-c,a]/(||mu||+eps)`. Units with `||mu|| <= 1e-6` receive the fail-closed scalar `-20`. The E03 carrier is `[a,c,hypot(a,c),b,q_theta]`; normalisation and the posterior prior use source-train sessions only.

E03 runs concurrently with E02 on GPU 3. It does not consume E02 intermediate outputs. Its aggregate writes `PENDING_E02_BASELINE` if E02's final result is unavailable rather than polling or retrying. Terminal output is retained in `logs/e03_posterior_angular_reliability_t4_seed42.log`.
