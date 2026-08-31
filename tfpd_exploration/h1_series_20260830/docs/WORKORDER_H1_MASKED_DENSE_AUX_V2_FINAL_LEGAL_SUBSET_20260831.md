# Work Order: H1 Masked Dense-Auxiliary V2 Final-Legal Subset

V1 remains a sealed failure. V2 changes only the training population: an additive index view removes windows whose governing final target is still-time. The established loader, experiment-1 contract, model architecture, and sealed receipts are not modified.

Training samples require final `eval_mask=true` and non-still behavior. Internal window positions use the frozen same-trial/non-still/eval mask. T0 and every positive-lambda arm share the identical filtered indices, sampler, seed, initialization, calibration schedule, and number of steps.

Held-source-date validation remains the unfiltered established four-field loader and computes last-bin variance-weighted R2 after every epoch. Outer fold-0 evaluation uses the same legacy metric population. Thus report metrics remain comparable to the original H1 last-bin domain even though matched training uses the final-legal subset.

Before GPU access, every source recording must retain at least 25 percent of its original windows, every original trial represented by an admitted window must retain at least one training window, every retained final position must be contract-legal, and no recording may be empty. The audit records original/retained/excluded counts, per-trial counts, and canonical digests. Failure stops without GPU.

After this new attrition gate, all V1 smoke, 5x4 source screen, lambda-selection thresholds, two final fits, and one-shot outer gates remain unchanged. At most two currently idle GPUs from the user-authorized physical set 0–3 may run concurrently. No failed cell is retried; target data stays closed until the source gate passes; formal held-out recordings are always forbidden.
