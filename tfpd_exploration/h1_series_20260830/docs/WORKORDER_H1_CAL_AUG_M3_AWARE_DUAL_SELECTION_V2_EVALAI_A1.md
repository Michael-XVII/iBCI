# H1 CAL-AUG M3-Aware Dual-Selection V2 EvalAI Submission A1

## Scope

This is a deployment/evaluation-only successor of the sealed
`h1_cal_aug_m3_aware_dual_selection_v2_eval_a1`.  It freezes exactly three
candidates, in this order: C2-E49, C2-HI-E45, and C2-HO-E15.  Training,
finetuning, checkpoint reselection, carrier refitting, model updates, and an
automatic successor are forbidden.

The predecessor is Git commit
`ae14a232d1dfc84de0661916da34fdb9596753c2`, with evaluation terminal SHA-256
`d0735087091bd1681804e76c96f10f9f262efc3295fd32aa2f24498a51a4a31a`.
The HI and HO selection receipts govern epochs 45 and 15 respectively.  The
sealed C2 training terminal is used only to resolve the selected receipt's
canonical checkpoint path and model-state SHA; it cannot change the selected
epoch or checkpoint SHA.

## Frozen candidates and deployment surface

All candidates use the V1 C1 package as the immutable deployment template:
the architecture, 27-session M3 identity/carrier payloads, H-C source
authority, q/lambda/PCA/U/mu/tau2, source normalizer, W700 history, `/20`
scaling, session mapping, decoder, and FALCON interface are identical.  Only
the candidate state dict and its checkpoint/state provenance differ.

The package names and Docker tags are:

- `c2_e49.pt` -> `h1-m3aware-v2-c2-e49:a1`
- `c2_hi_e45.pt` -> `h1-m3aware-v2-c2-hi-e45:a1`
- `c2_ho_e15.pt` -> `h1-m3aware-v2-c2-ho-e15:a1`

Each package must pass exact repeated CPU prediction, exact repeated GPU
prediction, state immutability, batch-size 1/8, and CPU/GPU tolerance
`rtol=2e-3, atol=2e-4`.  One shared Docker recipe and entrypoint are used for
all candidates.  Host/container CPU predictions must be exact and GPU
predictions must satisfy the same tolerance.

## Submission governance

All three images are built and verified before submission.  One immutable
`submission/manifest.json` is committed before the first new score is
accessed.  The exact phase is private `few-shot-test-2319` (challenge 2319,
phase ID 4599).  Submission order is C2-E49, C2-HI-E45, C2-HO-E15.  Only IDs
and status may be recorded until all three submissions exist; results are
then retrieved together.

The primary endpoint is official Held Out R2 Mean.  Secondary endpoints are
official Held Out R2 Std., official Held In R2 Mean, and Normalized Latency.
The preregistered performance-oriented primary successor is C2-HO-E15.
Contrasts are frozen as C2-E49 minus V1-C1-E49, C2-HI-E45 minus C2-E49,
C2-HO-E15 minus C2-E49, C2-HO-E15 minus C2-HI-E45, and C2-HO-E15 minus
V1-C1-E49.  The V1 C1 official Held Out R2 Mean authority is
`0.28413945277226266` (reported rounded value `0.284139`).

If an infrastructure-only failure occurs, no successful-arm score may be used
to modify another candidate.  Any recovery requires an additive amendment
and proof of package semantics equivalence.  The successful terminal is
`COMPLETE_H1_M3_AWARE_DUAL_SELECTION_V2_EVALAI_OFFICIAL_A1` and must record
training=false, zero optimizer/backward/update/retraining counts, three Docker
images, and three EvalAI submissions.
