# Work Order: H1 Masked Dense-Auxiliary V1

Date: 2026-08-31

Branch: `exp/h1-masked-dense-aux-v1`

Required ancestor: experiment-1 PASS commit `d1c66774a8b1e081972b72e9d4f5a89829b4c700`

## Question

Does contract-valid dense supervision improve the unchanged H1 last-bin decoder when its weight is selected using source dates only?

## Frozen scope

- Fold-0 outer target: the two recordings dated `19250101`, opened at most once and only after the source gate passes.
- Source: the eleven held-in recordings dated `19250108`, `19250113`, `19250115`, `19250119`, and `19250120`.
- The fourteen formal `held-out-calib` recordings are forbidden throughout.
- Five grouped leave-one-source-date-out folds; arms `T0`, `lambda=0.1`, `lambda=0.3`, and `lambda=1.0` share seed 42, initialization law, sampler law, and terminal-epoch rule.
- At most 22 fresh fits: 20 source-screen fits plus paired all-source fits only after the source gate.

## Loss and metrics

Training uses `last_bin_mse + lambda * masked_dense_aux_mse`. The mask is never a network input. Predictions are divided by the frozen H1 behavior scale 20 before either term is computed. Every epoch evaluates the held-source-date recordings and records per-recording and equal-recording-mean last-bin variance-weighted R2; selection remains fixed to epoch 49. `T0` is the exact `lambda=0` governing-loss path.

Positive lambda is chosen by the highest equal-date mean terminal R2; exact ties choose the smaller lambda. Continue only if its paired source deltas satisfy mean at least `+0.01`, at least four of five dates positive, and worst date at least `-0.02`. Outer success requires equal-recording mean delta at least `+0.01` and both recording deltas positive.

## Frozen recipe

W=700, calibration trials=2, batch=32, Adam `5e-5`, weight decay 0, seed 42, FP32, 50 epochs, fixed terminal epoch 49, no early stopping. The established direct H1 `SpintModel` is used with raw unsmoothed input, random calibration, active calibration segments, and cubic interpolation.

## Gates and persistence

1. CPU/no-data tests bind experiment-1 receipts and run both new tests and the existing sampler regression.
2. An attempt receipt is written before source NWB access. The source audit must show legal final positions, per-recording mask/dataset digests, and zero legal padding/still/intertrial positions.
3. T0 and lambda=1 smoke cells each run 20 GPU batches and must have finite loss/gradients and nonzero auxiliary loss.
4. The 20 source fits run without target access. A failed cell is recorded once and is never retried.
5. Only a passing source gate authorizes the paired all-source fits and single outer opening.

The user explicitly authorizes idle local GPUs and overrides historical companion-machine allocation. The supervisor may use only physical GPU 0 or 1 after a fresh idle check; GPU 2 and 3 must never be used or preempted. It runs in detached `tmux`, writes aggregate and per-cell logs below repository-root `logs/`, and does not depend on a Codex-managed terminal. Receipts, sidecars, and checkpoints are immutable mode `0444`; the supervisor never commits or pushes.
