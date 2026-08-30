# H1 Frozen H-S/H-C Activity-System Comparison V1

## Question

Does causal growing activity memory help the ordinary H-S system as much as it helps H-C, or is there measurable carrier/activity synergy?

## Fixed experiment

- Dates: 19250108, 19250113, 19250115, 19250119, 19250120.
- Surface: the exact strict post-M4 query windows already used by the accepted date-LODO terminal evaluations.
- Systems and arms: H-S static, H-S causal growing to cap 30, H-C static, H-C causal growing to cap 30.
- H-C rows are reused from the immutable completed five-date result with SHA `65c9bb40ad45ab7b74740da88fd8081504b7656e807e76b6eb9db903450adb68`.
- H-S uses the original seed-42 epoch-49 checkpoints and ordinary B3S identity path. No carrier is supplied to the H-S identity computation.
- The growing identity for an output trial may use only the first four support trials plus completed earlier trials, capped at 30.
- Model weights, carriers, normalizers, query windows, and targets are unchanged. There are no target gradients, optimizer steps, or updates.
- Metric: last-bin seven-output variance-weighted R2, with both pooled and equal-recording summaries. The cross-date summary is equal-date weighted.

## Interpretation

The comparison is a descriptive mechanism screen, not a target-selected formal deployment claim. The raw four-cell matrix is authoritative. Its initial automatic labels below compared final system levels and must not be interpreted as a statistical interaction. `WORKORDER_H1_DATE_LODO_ACTIVITY_SYSTEM_COMPARE_V2_20260828.md` and its immutable V2 receipt supersede that label only, using the correct difference in differences.

- H-C growing minus H-S growing >= 0.01: `HC_ACTIVITY_SYNERGY_RETAIN_CARRIER_SYSTEM`.
- Absolute difference < 0.01: `ACTIVITY_MEMORY_DOMINANT_NO_MATERIAL_CARRIER_SYNERGY`.
- H-S growing minus H-C growing >= 0.01: `HS_GROWING_OUTPERFORMS_HC`.

The result decides which frozen system is the better basis for a future causal state implementation; it does not authorize training by itself.
