# Work Order — H1 Window-Mask Contract V1 CPU Gate

Date: 2026-08-31. Authority:
`DESIGN_H1_WINDOW_MASK_CONTRACT_V1_20260830.md` §4 and the H1 series queue item 1.

## Cell

Implement and run the additive window-validity contract against synthetic
sessions only.  The cell must cover still-time rows, intertrial gaps,
pre-history padding, mid-window trial boundaries, sampling-order invariance,
digest determinism, dense/last-bin loss boundaries, and batch compatibility.

## Frozen boundary

- Baseline: `main@21d0881f50a8d88d78fb5d5b941a6bf470019e9f`.
- Branch: `exp/h1-window-mask-contract-v1`.
- Existing loaders, models, configs, and sealed result bodies are read-only.
- H1 data is forbidden for this cell; target/source NWB files opened: zero.
- CUDA construction and GPU allocation are forbidden; training steps: zero.
- Python: `/home/ial-mohd/workspace/envs/spint/bin/python`, always with
  `PYTHONNOUSERSITE=1` and an empty `CUDA_VISIBLE_DEVICES`.

## Attempt and terminal law

The result root must be fresh.  Before pytest imports the new contract, the
runner publishes an immutable attempt receipt binding the executed closure and
exact command.  It then publishes exactly one immutable terminal or failure
receipt, with SHA-256 sidecars.  The experiment is not automatically retried.

The only accepting status is `PASS_WINDOW_MASK_CONTRACT_V1_CPU_GATE`, requiring
all focused contract tests and the established Falcon sampler regression tests
to pass in the same pytest process.  Any other outcome leaves all later GPU
cells unauthorized.
