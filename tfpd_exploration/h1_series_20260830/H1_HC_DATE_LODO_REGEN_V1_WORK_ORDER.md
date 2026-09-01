# H1 five-date H-C Regeneration Successor V1 work order

Status: pre-registered additive successor. This document authorizes only a new
source-only H-C checkpoint authority. It does not reproduce historical
checkpoint bytes, historical target scores, the causal output EMA experiment,
or any DANDI transfer decision.

## Fixed scope

- Outer dates, in authority order: `19250108`, `19250113`, `19250115`,
  `19250119`, `19250120`.
- Canonical result root:
  `tfpd_exploration/h1_series_20260830/results/h1_hc_date_lodo_regen_v1/`.
- Data root: `/data/ial-dataset/ial-mohd/000954`.
- Each cell may index the outer-date filenames. It must not open an outer-date
  NWB, read its bytes, run target inference, or use target gradients.
- Every receipt and its sidecar is publish-once, mode `0444`. The canonical
  result root must not exist before the experiment attempt is published.
- No sealed/frozen module may be edited by this work order.

## Source carrier authority

For one outer date, the exact source roster is every public held-in calibration
recording not on that date. Carrier hyperparameters are selected with a nested
source-only date LODO:

- candidates: `q in {4,8,12,16}` and
  `lambda in {1e-3,1e-2,1e-1,1,10}`;
- for an inner held-source date, standardization and PCA are fitted on the
  other source dates;
- for every held-source recording, the analytic ridge decoder is fitted on its
  earliest four legal trials and evaluated on all later legal trial blocks;
- the score is float64 seven-output variance-weighted R-squared;
- recordings are averaged equally within a date and dates are averaged equally;
- exact ties are resolved by higher worst-date R-squared, smaller `q`, then
  larger `lambda`.

After selection, standardization, PCA, raw rows, `U`, `mu`, and `tau2` are
refitted on the full outer-source roster. The existing analytic EB shrinkage
equations are then applied without modification. All legal contiguous M=4
source carriers are materialized. The authority binds candidate tables,
session roster, source NWB SHA-256, transform arrays, carrier cache, source RMS
normalizer, fixed batch order, and fixed 50-epoch calibration schedule.

## H-C training cell

- Model: existing `H1CarrierIdSpint`, `carrier_hidden_dim=32`,
  `carrier_dim=4`, `window_size=700`, seven outputs, dynamic dropout
  `Uniform(0,1)`.
- Expected parameter count: `10,947,836`.
- Fresh initialization for every date with seed `42`; warm start and checkpoint
  loading are forbidden.
- Batch size `32`, Adam learning rate `5e-5`, weight decay `0`, FP32.
- Exactly 50 epochs, no validation, early stopping, SWA, target selection, or
  automatic retry. The only checkpoint is zero-based epoch `49`.
- `deterministic=false`; the produced checkpoint SHA-256 is the authority.
- A cell binds one physical GPU index and UUID. At most two explicitly supplied
  idle GPUs may be used concurrently.

## Gates and failure semantics

The only authorized order is:

1. CPU/no-data gate PASS.
2. Five-date source authority PASS.
3. One-date limited-step GPU smoke PASS.
4. All five full training cells reach their fixed terminal epoch.
5. Terminal verifier PASS.

The experiment-level attempt must exist before any NWB read or CUDA query. Each
GPU cell publishes its own attempt before importing/initializing CUDA or
reloading source data. The smoke validates fresh state, identity/carrier batch
alignment, finite loss, gradients, and Adam state, plus zero target access.

A failure publishes an immutable failure receipt and stops new work. There is
no automatic retry. Any later retry needs an additive amendment and a fresh
canonical result root; existing artifacts are never resumed or overwritten.

## Terminal authority

For all five dates, the verifier must validate:

- complete and ordered date set;
- epoch-49 checkpoint and sidecar SHA-256;
- embedded fixed-epoch, no-warm-start, and zero-target-access provenance;
- resolved configuration SHA-256;
- source date authority and source NWB hashes;
- experiment attempt and code-closure SHA-256;
- identical fresh initial state hash, date-specific terminal state hash, and
  exact optimizer step count;
- `target_recordings_opened=0`, `target_bytes_read=0`, and
  `target_optimizer_steps=0` everywhere.

Large `.ckpt` files remain Git-ignored. Receipts, sidecars, checkpoint hashes,
and the experiment record are committed. Success permits a later branch for
the independently pre-registered EMA work order; it does not itself authorize
EMA, target evaluation, or target score reporting.

## Public command surface

The runner is
`tfpd_exploration/h1_series_20260830/scripts/run_h1_hc_date_lodo_regen_v1.py`.
With no phase flag it is a zero-write, zero-data, zero-CUDA dry run. Its public
phase flags are:

- `--prepare-source-authority`
- `--smoke --gpus <physical-index>`
- `--train --gpus <one-or-two-physical-indices>`
- `--verify-terminal`

All phases accept explicit `--data-root` and `--result-root`.
