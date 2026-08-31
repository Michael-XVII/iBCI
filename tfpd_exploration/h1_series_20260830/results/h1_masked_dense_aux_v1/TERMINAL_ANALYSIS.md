# H1 Masked Dense-Auxiliary V1 — Terminal Analysis

## Verdict

`STOP_BEFORE_GPU_SOURCE_DATA_WINDOW_CONTRACT_CONTRADICTION`

Experiment 2 is not authorized to proceed to smoke or training. The source-only H1 data audit found 5,467 admitted windows in the first deterministically audited source recording whose final positions violate the experiment-1 law `eval_mask && non-still-time`. The immutable exception lists starts 15 through 12192 (non-contiguous). The established loader admits windows using final-position `eval_mask` alone, while `build_window_valid` correctly requires the final position also to be non-still. Automatically dropping these windows would change the preregistered training population and is therefore not an allowed repair inside this attempt.

## Execution evidence

- Branch: `exp/h1-masked-dense-aux-v1`
- Execution commit: `0e45e088a215f8e1ece6b32f794e0311d95a7feb`
- Final focused CPU/no-data gate: 29 tests passed; CUDA was not initialized and H1 data was not opened by that gate.
- The first supervisor invocation stopped before data because of a `Path` plumbing defect; the defect, immutable failure, repair gate, and user authorization for GPUs 2/3 are preserved in the result root.
- The repaired source audit opened only the eleven source recordings. It stopped during mask construction before GPU allocation, smoke, optimizer construction, or training.
- GPU training steps: 0.
- Held-source-date per-epoch R2 histories: not produced because no fit was authorized.
- Fold-0 target recordings opened: 0.
- Formal held-out recordings opened: 0.
- Governing failure receipt: `failure_v2.json` with status `FAIL_H1_MASKED_DENSE_AUX_V1_NO_RETRY`.
- Aggregate log: `logs/h1_masked_dense_aux_v1_repair_v2_20260831_2030.log`.

Any successor must have a new work order that explicitly chooses how to reconcile still-time final positions with the frozen window population. It must not reinterpret this attempt as a lambda result or silently filter windows.
