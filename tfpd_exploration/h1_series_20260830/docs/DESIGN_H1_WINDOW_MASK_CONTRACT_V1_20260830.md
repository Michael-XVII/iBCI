# H1 Window-Mask Contract V1 — design for the masked dense-auxiliary route

Date: 2026-08-30. CPU-side deliverable; no H1 data touched yet.
Authority: `EXECUTION_GUIDANCE_AFTER_AC3_AND_CONTINUITY_AUDIT_20260829.md` §10
("the remaining M1/H1 work is … a typed mask/data contract …, adversarial
tests, and a separate work order for each real W"). H1 W = 700.

## 1. Facts pinned to code (read-only)

| Fact | Location |
|---|---|
| loss reads only the last position (`pred[:, -1:, :]`) | `streaming_calibration_exp/src/models/streaming_calibration_module.py:479-489` (`_slice_last_timestep`, flag `decode_last_timestep_only=True` set at `:135`) |
| `pre_history = W-1` zero/False prepending | `streaming_calibration_exp/src/data/falcon_datamodule.py:213-234` |
| still-time rows = `all(abs(covariates) < 0.001, axis=1)` | `falcon_datamodule.py:228` |
| intertrial rows excludable only in calibration today (`use_calib_intertrialials=False`) | `falcon_datamodule.py:104-107` |
| batch carries NO per-position validity | `__getitem__` return tuple (audited) |

Consequence: an UNMASKED dense auxiliary loss would supervise zero-padded,
intertrial, and still-time rows — the audit's shrinkage-amplification risk.

## 2. The contract (additive; the frozen batch stays untouched by default)

New typed tensor in the batch, position-aligned with the window axis:

```text
window_valid[b, w] : bool   # True iff position w of window b is a LEGAL
                            # supervision target
construction (per window, from session-level arrays ALREADY in the loader):
  legal(t) = eval_mask[t]                       # FALCON-evaluable bin
           AND NOT still_time[t]                # |cov|<1e-3 row
           AND NOT in_prehistory_padding(t)     # t < pre_history start
           AND same_trial_as_last_position(t)   # trial_change boundary: the
                                                # window must not straddle a
                                                # trial change
window_valid[b, w] = legal(start_b + w)
```

Invariants (adversarially tested before any training):
1. `window_valid[:, -1] == True` for every admitted window (admission already
   checks the final position — the mask must never contradict admission);
2. any window with `trial_change` inside `start+0..start+W-2` has all earlier
   positions `False` from the boundary onward;
3. padded rows (pre-history region) are `False` everywhere they appear;
4. mask is a pure function of session-level arrays (deterministic digest per
   session; recomputable at eval);
5. adding the mask to the batch does NOT change any existing tensor
   (byte-digest equality of neural/covariates/calib with and without the
   contract, same index order).

## 3. The masked dense-auxiliary loss (the future H1 training cell)

```python
aux = ((pred - target) ** 2).sum(-1)          # [B, W]
loss = last_bin_mse                           # UNCHANGED governing loss
     + lam * (aux * window_valid).sum() / (window_valid.sum() * C)
```

- `lam` selected ONLY inside source folds (LOSO), candidates {0.1, 0.3, 1.0};
- checkpoint selection and every reported metric stay last-bin, unchanged;
- the mask never enters the model graph — inference is bit-identical with or
  without this training (same architecture, same readout);
- contract tests must PASS before any GPU cell is authorized (§10's own rule).

## 4. Adversarial tests (CPU, synthetic sessions; no H1 data)

- synthetic session with injected still-time rows / intertrial gaps / short
  first trial (padding) / trial change mid-window → mask bits must match the
  hand-computed expectation in all four cases;
- property: `window_valid` is invariant to window-sampling order;
- property: digest-determinism of the mask per session;
- regression: with the mask all-True the aux loss equals the naive dense loss;
  with the mask last-bin-only it equals the current governing loss (boundary
  cases of the λ-blend);
- the batch-compat test: collation with the new field leaves existing fields'
  bytes identical.

## 5. Data-side verification (CPU, once H1 data is opened on this machine)

Per H1 session: fraction of legal positions per window; the distribution of
`legal_count/W`; confirmation that padded/still rows are exactly where the
loader says. Receipt-gated; no GPU.

## 6. Authorization boundary

This document creates the CONTRACT only. The H1 masked-dense training cell
(W=700, its own loader policy and checkpoint rule, λ by source folds) is a
separate work order — and per the operator's 2026-08-30 split, its GPU work
runs on the companion machine; this machine's GPUs stay reserved for the
running C2/C3 series.
