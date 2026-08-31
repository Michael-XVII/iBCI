# H1 Masked Dense-Auxiliary V1 — Experiment Record

- Status: `FAIL_H1_MASKED_DENSE_AUX_V1_NO_RETRY`
- Branch: `exp/h1-masked-dense-aux-v1`
- Execution commit: `1cd4d7712749bf5c2c1dca678ec373d8e925ba23`
- Required experiment-1 ancestor: `d1c66774a8b1e081972b72e9d4f5a89829b4c700`
- Interpreter: `/home/ial-mohd/workspace/envs/spint/bin/python`
- Result root: `/home/ial-mohd/workspace/iBCI/tfpd_exploration/h1_series_20260830/results/h1_masked_dense_aux_v1`
- Aggregate/per-cell logs: `/home/ial-mohd/workspace/iBCI/logs`

## Scope and protocol

The five source dates were screened by grouped leave-one-date-out. Every epoch 0–49 evaluated the held-source-date recordings and each cell receipt stores per-recording and equal-recording-mean last-bin variance-weighted R². Only epoch 49 governs lambda selection. The two 19250101 recordings are the one-shot outer fold; the fourteen formal held-out recordings were never opened.

The frozen recipe is W=700, two calibration trials, batch 32, Adam 5e-5, FP32, seed 42, 50 epochs, no early stopping, raw unsmoothed input, random calibration, active calibration segments, and cubic interpolation. Training loss is last-bin MSE plus lambda times contract-masked dense MSE; all validation and report metrics are last-bin only.

## Failure

`AttributeError: 'str' object has no attribute 'rglob'`

## GPU authorization conclusion

GPU training was authorized only after the experiment-1 contract gate, this experiment's CPU/no-data gate, source data audit, and two smoke cells passed. Physical GPUs were restricted to 0 and 1 and were checked idle before each cell; GPUs 2 and 3 were forbidden. A failed source gate does not authorize outer target access.
