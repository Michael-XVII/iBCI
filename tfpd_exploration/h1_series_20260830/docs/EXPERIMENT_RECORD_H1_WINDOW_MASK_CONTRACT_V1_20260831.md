# Experiment Record — H1 Window-Mask Contract V1 CPU Gate

Date: 2026-08-31.  This record closes queue item 1 from the H1 series bundle.

## Identity and scope

- Baseline: `main@21d0881f50a8d88d78fb5d5b941a6bf470019e9f`.
- Experiment branch: `exp/h1-window-mask-contract-v1`.
- Attempt count: one; no retry or restart.
- Runtime: `/home/ial-mohd/workspace/envs/spint/bin/python3.10` with
  `PYTHONNOUSERSITE=1` and `CUDA_VISIBLE_DEVICES=""`.
- Synthetic sessions only.  H1 data roots and NWB files opened: zero.
- CUDA contexts, allocated GPUs, training and optimizer steps: zero.
- Existing Falcon loader/model/config/result files were not edited.

## Additive implementation

The contract implementation constructs a typed `torch.bool [B,W]` mask from
the padded session `eval_mask`, covariates, cumulative trial identity and
admitted window starts.  A legal position is evaluable, non-still, and in the
same trial as the last window position.  It also supplies canonical
sampling-order-independent digests, the masked dense auxiliary MSE law, and an
explicit opt-in tuple append that leaves the established batch unchanged by
default.

The focused suite covers hand-computed still-time, intertrial, short-first-trial
padding and mid-window-boundary cases; the real H1 width `W=700`; order and
digest determinism; all-True and last-bin-only loss boundaries; invalid-window
rejection; byte-identical old batch fields after opt-in collation; and the
existing Falcon sampler regression suite.

## Execution

The runner was launched once in detached tmux session
`h1_window_mask_contract_v1_20260831`.  The session did not depend on the Codex
PTY and exited normally after writing the terminal receipt.  Complete stdout
and stderr are retained at:

`logs/h1_window_mask_contract_v1_20260831.log`

Log SHA-256:
`ba1b447d56d8b12159750ab7cf46d2b9327c10d778e5be4876c717735f0e5dbe`.

Focused pytest result: **17 passed, 0 failed, 0 errors, 0 skipped in 2.732 s**.

## Immutable evidence

Result root:
`tfpd_exploration/h1_series_20260830/results/h1_window_mask_contract_v1/`.

| Artifact | SHA-256 | Mode |
|---|---|---|
| `attempt.json` | `3b0a9bc3146a0e78b5a44bf6a4663929960be5fdd8ec4c85604ff551b7e9ab5f` | `0444` |
| `pytest.xml` | `c9ed8399c774e0febdc4ca1b1d56ae472e7ec8332a4a3d1e2df27e41f30ac15f` | `0444` |
| `terminal.json` | `894a9c703b70098a3cb587d49ca9a802f4fe8c3c266af23878dacb6d81e0cce7` | `0444` |

All three sidecars passed `sha256sum -c` and are also mode `0444`.  The attempt
receipt binds the exact executed source/test/runner/design/work-order closure.

## Verdict

`PASS_WINDOW_MASK_CONTRACT_V1_CPU_GATE`

The adversarial CPU prerequisite for later H1 work is satisfied.  This receipt
does **not** by itself authorize a GPU cell: masked dense-auxiliary training
still requires its separate work order and explicit launch authorization.
