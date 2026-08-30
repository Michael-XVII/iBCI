# H1 Causal Representative Activity Memory V1

## Goal

Recover part of the remaining causal-growing-to-full-oracle H1 gap without training or target labels.

## Governing design

Use the frozen five-date H-C checkpoints. Keep the original first four support trials. For each output trial, select at most 26 additional completed query trials, evenly spaced over the causal history. The memory therefore has at most 30 trials, preserves the honest support, covers old and recent conditions, and never reads the current or a future trial.

## Controls

- `CAUSAL_FIFO_CAP30`: support four plus the most recent 26 completed query trials.
- `CAUSAL_ALL_PAST`: support plus every completed past trial; causal but unbounded and diagnostic.
- Immutable predecessor `CAUSAL_GROWING_CAP30`: support plus the first completed trials, then frozen at 30.
- Immutable noncausal full-session oracle is retained only as an upper-bound reference.

## Gate

The governing coverage-cap30 arm must improve equal-date R2 over frozen growing by at least `+0.01` and be positive on at least four of five dates. Otherwise stop this policy. All cells use identical targets, last-bin seven-output variance-weighted R2, frozen parameters/carriers/normalizers, and zero target optimization/backward/update.
