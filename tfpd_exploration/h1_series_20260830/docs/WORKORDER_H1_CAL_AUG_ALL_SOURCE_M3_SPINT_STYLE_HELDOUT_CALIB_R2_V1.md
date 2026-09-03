# H1 CAL-AUG All-Source M3 SPINT-Style Held-Out-Calib R² V1

Status: authorized additive diagnostic. This work order does not modify the sealed
`H1_CAL_AUG_ALL_SOURCE_M3_DEPLOYMENT_V1` or Package-A1 roots.

## Objective and claim boundary

Use only the two frozen all-source epoch-49 T0/C1 checkpoints, through their
sealed Package-A1 carrier-aware decoder packages, to reproduce the *training-time*
SPINT `val_heldout/r2` scoring surface on the 14 public H1 `held-out-calib` NWBs.

This is deliberately labelled
`local original-SPINT-style held-out-calib validation R²`. It is not the FALCON
official held-out R², is not a leakage-free post-calibration deployment R², and
must not be used for checkpoint, budget, q/lambda, recipe, or submission selection.
The same three behavior-labelled trials provide the frozen M3 identity/carrier
payload and the validation targets, exactly exposing the calibration-reuse
limitation of this local diagnostic.

## Frozen predecessor

- Package-A1 terminal SHA-256:
  `4137495462a299e948beb58be578c739cc211330de4769992c03e743d7c7bf26`.
- T0 epoch-49 checkpoint SHA-256:
  `6d4d14226b706951274982438b588527beb442200aad2f50f9d18b68e54a9648`.
- C1 epoch-49 checkpoint SHA-256:
  `0f406a8e69fdb57cf6a5480149f04ab3500e7fad849d36db38042edbadb2cd06`.
- T0/C1 packages, source authority, M3 calibration authority, model-state hashes,
  and pair-integrity authority must verify before any held-out scoring access.
- Training, optimizer, backward, parameter update, warm start, checkpoint
  selection, and EvalAI submission counts are all fixed at zero.

## Original-SPINT-style local validation surface

- Open exactly the registered S6--S12 14-file `held-out-calib` roster.
- Use the serialized-then-reloaded `H1CarrierIdSpintDecoder`; do not evaluate a
  bare training model.
- `reset()` selects the already sealed earliest-M3 identity and H-C carrier for
  the corresponding FALCON session key. No fitting occurs in this successor.
- Stream the same NWB as the neural input and labelled validation surface.
- Use W=700 causal last-bin predictions, `/20` scaling, and the NWB eval mask.
- Match the original `FalconDataset` pre-history behavior: early windows use
  W-1 zero padding and remain scoreable when their last bin is eval-valid.
- Match the original `SessionBatchSampler(batch_size=32)`: preserve chronological
  eval-valid order and drop only the final incomplete per-session batch.
- Compute float64 seven-output variance-weighted R² independently per recording;
  report the equal-session mean and population standard deviation over 14
  recordings, plus paired C1-T0 values.
- Save prediction/target/eval-mask/legacy-score-mask cache arrays and SHA
  manifests. T0 and C1 target and scoring-mask bytes must match exactly.

The original generic SPINT H1 configuration used two calibration trials. That
identity interface is incompatible with `H1CarrierIdSpint`. This successor does
not change the frozen M3 deployment interface: it reproduces the original
SPINT *validation population and aggregation*, while explicitly recording the
successor-specific M3 identity+carrier calibration.

## Execution and terminal discipline

1. CPU/no-data/no-CUDA gate.
2. Immutable attempt before held-out NWB access or CUDA initialization.
3. Predecessor/package/checkpoint/state verification.
4. One evaluation pass for T0 and one for C1; no automatic retry.
5. Terminal recomputation from the immutable prediction cache.

The new canonical result root is
`results/h1_cal_aug_all_source_m3_spint_style_heldout_calib_r2_v1/`.
Receipts and Markdown are mode 0444 with SHA-256 sidecars. Large `.npz` caches
remain Git-ignored. No EvalAI credential access, upload, or remote test occurs.
