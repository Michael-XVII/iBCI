# H1 H-C Regeneration V1 Amendment 1: detached execution

Status: additive operational amendment. The model, data, carrier selection,
source-only boundary, seeds, schedule, optimizer, fixed epoch, and terminal
acceptance criteria in the V1 work order are unchanged.

## Cause and predecessor disposition

The first canonical attempt at
`results/h1_hc_date_lodo_regen_v1/` passed CPU, source-authority, and GPU-smoke
gates. Two full cells were then launched from a Codex-managed persistent PTY.
At the operator's request, those cells were stopped before terminal epoch so
training could be independent of the Codex session. No checkpoint or target
evaluation was produced.

The predecessor is permanently failed and is not resumed. Its immutable
`training_failure.json` SHA-256 is
`0298aa841e215b216d5f8e4fa96a18d1700bf745d851591de3dbc3a290b08018`.

## Fresh successor authority

- New canonical result root:
  `tfpd_exploration/h1_series_20260830/results/h1_hc_date_lodo_regen_v1_detached_a1/`.
- Runtime log root:
  `logs/h1_hc_date_lodo_regen_v1_detached_a1/`.
- The new root starts from a new attempt and regenerates CPU gate, source
  authority, smoke, all five fresh model initializations, and terminal receipt.
- No artifact, partial model state, optimizer state, carrier cache, or schedule
  file from the failed root is copied or used as training input.

## Detached execution rule

The full-training supervisor is launched with `nohup` and `setsid`, stdin bound
to `/dev/null`, and stdout/stderr redirected under the runtime log root. It is
not attached to a Codex PTY and must continue if the Codex conversation or tool
session ends.

Each cell writes stdout/stderr directly to its own file in `logs/`; the
supervisor does not pipe, inspect, tail, or relay training output. Completed
cell logs are sealed mode `0444` with `.sha256`. The model training loop emits
no epoch progress line or progress bar. Cell outcome is communicated only by
immutable terminal/failure receipts.

The detached supervisor automatically runs terminal verification only after
all five cells succeed. If any cell fails, it stops launching further dates,
seals completed logs, publishes immutable failure authority, and does not
retry.

## Launch form

After the fresh CPU/source/smoke gates pass, the authorized detached launch is
equivalent to:

```bash
nohup setsid env PYTHONNOUSERSITE=1 \
  /home/ial-mohd/workspace/envs/spint/bin/python \
  tfpd_exploration/h1_series_20260830/scripts/run_h1_hc_date_lodo_regen_v1.py \
  --detached-supervisor --gpus 0,1 \
  --data-root /data/ial-dataset/ial-mohd/000954 \
  --result-root tfpd_exploration/h1_series_20260830/results/h1_hc_date_lodo_regen_v1_detached_a1 \
  --log-root logs/h1_hc_date_lodo_regen_v1_detached_a1 \
  </dev/null >logs/h1_hc_date_lodo_regen_v1_detached_a1/supervisor.log 2>&1 &
```

GPU indices remain subject to the V1 idle-at-launch and UUID-binding rules.
