# H1 H-C Regeneration V1 Amendment 2: explicit GPU authorization

Status: additive operational amendment. All statistical, source-only, model,
schedule, optimizer, detached-execution, logging, failure, and terminal rules
from the V1 work order and Amendment 1 remain unchanged.

## A1 disposition

Detached A1 failed before any full-training cell attempt was created because
GPU 0 crossed the runner's conservative idle heuristic between smoke and
detached launch. Its immutable `detached_training_failure.json` SHA-256 is
`3e5db1a3474622536927480e750635030dc44c38099172c1648df502044133f1`.
It records zero target recordings opened and zero target bytes read. A1 is not
resumed or retried.

## Explicit authorization

On 2026-09-01 the operator explicitly authorized both physical GPU 0 and GPU 1
for this experiment. For Amendment 2 only, `--allow-authorized-busy-gpus`
replaces the local `<1024 MiB and <=5%` launch heuristic with that explicit
authorization. The runner must still query both GPUs, record each physical
index and UUID, bind `CUDA_VISIBLE_DEVICES` per cell, and limit concurrency to
two. It does not terminate, reconfigure, or otherwise manage any foreign GPU
process.

## Fresh A2 authority

- Canonical result root:
  `tfpd_exploration/h1_series_20260830/results/h1_hc_date_lodo_regen_v1_detached_a2/`.
- Runtime log root: `logs/h1_hc_date_lodo_regen_v1_detached_a2/`.
- A2 regenerates a fresh attempt, CPU gate, source authority, smoke, five fresh
  H-C cells, and terminal authority. No A1 model/optimizer/cache/schedule file
  is training input.
- Full training is launched only through the Amendment 1 `nohup` + `setsid`
  detached supervisor, with `--gpus 0,1 --allow-authorized-busy-gpus`.
- Cell output is written directly to sealed files under `logs/`; Codex does not
  pipe, tail, or monitor training output. The supervisor automatically runs
  terminal verification after all five epoch-49 cells succeed.
