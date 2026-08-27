# E06 Minimal SO(2)-Equivariant Consumer

## Status

Implementation and strict structural tests completed on 2026-08-27. A bounded
CPU wiring smoke completed a forward/backward, validation, checkpoint, and
metadata cycle. Its log is:

    logs/e06_so2_wiring_smoke_v2_20260827.log

The earlier unsuffixed smoke log records cache preparation and an intentional
interrupt after Lightning parsed 1.0 as 100% of batches. The matched seed-42
training command is:

    sua_exploration/scripts/run_e06_minimal_so2.sh

## Matched protocol

- E01 split: 37 train / 8 validation / 8 held-out-selected test sessions.
- Seed 42, T4 pool 50, activity calibration prefix 50, batch 32.
- Same E01 teacher checkpoint, optimizer, learning rate, 40-epoch cap,
  patience 10, and held-out-SPINT selection convention.
- No target labels, target optimizer, target backward, or target-side search.
- The legacy unrestricted decoder is retained only for checkpoint/API
  compatibility and frozen; it is absent from the E06 computation graph.

## Consumer

For each neuron, a shared scalar network sees only live activity, the
activity-calibration identity, modulation magnitude, and baseline. It produces
an attention logit and two scalar coefficients. The physical output is:

    z_t = sum_i alpha_i,t * (A_i,t * u_i + B_i,t * J * u_i)

T4 (a, c) is first reconstructed from source-only normalization statistics, so
the transformation law is defined in physical task coordinates. The
activity-calibration encoder sees only invariant (m, b). There is no learned
vector bias or unrestricted vector MLP after the equivariant sum.

## Structural verification

The test file
streaming_calibration_exp/tests/test_e06_minimal_so2_consumer.py verifies:

1. physical SO(2) equivariance after anisotropic source normalization;
2. neuron permutation invariance under aligned activity/identity/T4 shuffling;
3. gradient flow through the scalar network.

Focused E05/E06 result: 5 tests passed. The adjacent B3S/T4 regression suite
also passed: 54 passed and 15 skipped.

## Implementation receipts

The successful smoke records 67,208 optimizer-trainable parameters: 18,162 in
the invariant activity/calibration encoder and 49,046 in the SO(2) scalar
consumer. At batch 1, 64 units, and a 50-bin window, the scalar network records
3,112,960 MACs. The retained legacy teacher decoder has zero trainable
parameters and is excluded from the active-path count.

## Pending result fields

Formal training status, selected checkpoint, per-session R2, paired delta
against E01, worst session, positive-session ratio, and post-training
equivariance error will be filled from the run artifact after seed-42 training
completes.
