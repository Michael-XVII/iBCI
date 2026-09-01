# H1 five-date causal output EMA V1 work order

Status: pre-registered additive target-evaluation successor. This work order
authorizes one fixed causal output-filter transfer test on the five regenerated
H-C checkpoints. It does not authorize training, checkpoint selection, target
adaptation, a DANDI artifact-reproduction claim, or changes to sealed/frozen
modules.

## Predecessor authority

- Predecessor root:
  `tfpd_exploration/h1_series_20260830/results/h1_hc_date_lodo_regen_v1_detached_a2/`.
- Terminal SHA-256:
  `470634334480c33fd8d4679baa470454984dadeb57a25b9dec8d05612c43b9ec`.
- Source authority SHA-256:
  `6cf656048af20174c3cf164406e25051137bfe687b0dda93d06ff1835c80500e`.
- Regeneration code-closure SHA-256:
  `0727459079dd401f896bc714e85da46c5e5d08820aea3932a507d04cc85a4a74`.
- Regeneration attempt SHA-256:
  `e0d39b534cb0800f1ca027ce1893c7f023648391380b94d723f531b1b9fa67b4`.
- Canonical initial model-state SHA-256:
  `bc6dc8a0543c760811f770206c7ee22ae35eaf970c6dad0ec259a84172e4d04b`.
- Shared resolved-config SHA-256:
  `d9fd74381bbc9769a116f47657f5762225ef56258714fe9de20f0498e0496f4f`.

| Outer date | Cell terminal SHA-256 | Checkpoint SHA-256 | Source authority SHA-256 | Steps | Terminal state SHA-256 |
|---|---|---|---|---:|---|
| `19250108` | `6ec102fd871b76a0c37978f7694d6a6c7b4bd55a6df611a8f8445538ebad5fb3` | `d08e9488757a5e9672c2bbafaac218bf2043e41f17b8ffe02ceae9c28b65d4de` | `c740fcf56980b4d308d2030ad0b606a7a8d282abee70aaca4f56e17eaf7edb93` | 167800 | `06ae7856c71a6586685b508d9bcc8abf084472f380aea3454cbedd9e012509f9` |
| `19250113` | `e509abc4ce54bd84673e35143607c76303e19a0e999c577e2b3bee28c3243a8a` | `76e275b4f79a90e8223d444e610a7780dec1204574787b096f524e1466a4910c` | `6e12eb63a93c4de9e548548d01d2ff85f26900bda7dee63f6fea4188680f2b7d` | 171100 | `03dd75975edabf8b064cd0496d872b401e20748c03de7c63862d19c98b3bafed` |
| `19250115` | `2e6fc4ead6351c708363c61395a475bcfb854941a72f32bcd363e1f1da74dfac` | `470fa3fc3e023fd98fb0ac52a304bb75e167b2e15caa5afae5e94e6111da0249` | `5949d3fb1c10880ca1db69a983f332afd0f1fc340531d5836ae22f566a3e27b5` | 172700 | `3cb305725b7ac7abe947eba1d1207f3285480071ed9368b6b69a0271ec6d7af3` |
| `19250119` | `4c78f0e7ba1027f1b4965ef53c885548fe3e15ae4d57d417ca6947fd5e445b75` | `3cc84b8f956fe6d7e2b581ec2fc73aa7099bf1aac4d20fd63ee2f73cc20ac637` | `fba7d5054104aded142875cacda1cf2636b918dc9c340edd099cf7b3cae6cde4` | 170200 | `446a8a21dc2a7b85c7afe01b61a87e0abe28cd97c65cb0cd65b613d9f3f4468e` |
| `19250120` | `49d536369ae892670f317fcd33ec7899007a44d63e50c8d0ea16bddb6e63044d` | `405ae97bf31513466f6be8583f07ac2ca354ba1fbf256e2bec97728675386af4` | `fc4d6a6ae756aea9247018bc1d9368a0cc28d4d062eaa224aadc16039fab06f8` | 170950 | `f8e455cfdf217ea34e37fc0ed309a44da500d142f148f198a2a8706bf7f8528c` |

Every predecessor receipt, sidecar, plan array, config, embedded checkpoint
metadata, checkpoint state, and table value above must pass before an
outer-date target recording is opened.

## Fixed causal filter and evaluation surface

- Alpha grid, in authority order: `0`, `0.3`, `0.5`, `0.7`, `0.9`.
- The recurrence is float64 `s[0]=prediction[0]` and
  `s[t]=alpha*s[t-1]+(1-alpha)*prediction[t]`.
- `alpha=0` must be numerically identical to the raw prediction. Only
  `alpha=0.7` is governing. Other alphas are descriptive and may not select a
  filter, checkpoint, date, recording, metric, or retry.
- Each outer-date recording uses exactly its earliest four chronological legal
  trials to reconstruct the frozen H-C identity and analytic EB carrier. No
  later label enters identity/carrier construction.
- The first output uses the first complete `W=700` neural history beginning at
  the first eval-valid bin of trial five. Every subsequent one-bin-shifted
  output updates the EMA, including outputs whose endpoints are outside the
  scoring mask. Scoring applies the existing eval mask only after filtering.
- EMA resets only at a recording boundary. It does not reset at trial changes
  or eval-mask gaps.
- The model is inference-only and immutable: no optimizer, backward, gradient,
  parameter/buffer update, warm start, TTA fit, or target-driven selection.

## Metrics and decision

For every alpha, compute float64 seven-output variance-weighted R-squared per
recording. Recordings are averaged equally inside an outer date; the five
outer dates are then averaged equally. Pooled scores are descriptive only.

The governing `alpha=0.7` transfer result is `PASS_TRANSFER` only if both:

1. equal-date mean `delta R-squared >= +0.01` versus alpha zero; and
2. at least four of five date-level deltas are strictly positive.

Otherwise the scientific terminal is `COMPLETE_NO_TRANSFER`. That is a valid
completed negative experiment, not an execution failure.

## Execution and artifacts

- Data root: `/data/ial-dataset/ial-mohd/000954`.
- Canonical result root:
  `tfpd_exploration/h1_series_20260830/results/h1_causal_output_ema_v1/`.
- Runtime logs: `logs/h1_causal_output_ema_v1/`.
- The canonical result root must be fresh. The experiment attempt is published
  mode `0444` with its sidecar before CUDA, target indexing, or target loading.
- CPU/no-data gate precedes a synthetic checkpoint GPU smoke. Smoke reads zero
  NWB files. Full evaluation queues are physical GPU0
  (`19250108,19250115,19250120`) and physical GPU1
  (`19250113,19250119`), with runtime UUID binding.
- The detached supervisor is independent of the launching terminal. It emits
  no progress bar and writes only start/terminal/error summaries to logs.
- There is no automatic retry. A cell failure stops new launches, preserves
  any already-running sibling, and publishes an immutable failure receipt.
- Per-date prediction caches are local ignored `.npz` artifacts with tracked
  SHA sidecars. JSON/Markdown receipts and all sidecars are publish-once mode
  `0444`. No target or model artifact is overwritten.

## Public command surface

The runner is
`tfpd_exploration/h1_series_20260830/scripts/run_h1_causal_output_ema_v1.py`.
With no phase flag it is a zero-write, zero-data, zero-CUDA dry run. Public
phases are `--cpu-gate`, `--smoke`, `--evaluate`, `--verify-terminal`, and
`--detached-supervisor`. All runtime phases accept explicit `--data-root`,
`--predecessor-root`, `--result-root`, `--log-root`, and
`--physical-gpus`.

Success or a complete negative result supports only the stated five-date H-C
transfer conclusion. It does not reproduce the external DANDI bytes or score,
and it does not generalize to another H1 checkpoint family or evaluation
surface.
