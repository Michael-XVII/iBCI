# H1 Series Bundle — 2026-08-30

Entry point for the H1 (human, FALCON track, W=700) experiment series. Clone
the full repo `iBCI` and work from this folder; the repo contains everything
the bundle only references (see MANIFEST_OF_REPO_PATHS.md).

## 1. Environment

python: `/home/xinyuan/miniconda3/envs/spint/bin/python` equivalent (torch
2.5.1, lightning, hdmf/pynwb); ALWAYS `PYTHONNOUSERSITE=1`. H1 NWB data is NOT
in this repo — mount/copy the FALCON H1 data root and set the data-root env
vars per the scripts' requirements (see `SPINT-main/src/data/` loaders).

## 2. What is already established on H1 (sealed receipts in results/)

| Line | Verdict |
|---|---|
| trial-structure sparse context carrier | **works, mainline** (the only transfer mechanism valid on all 3 datasets) |
| date-LoDO activity system compare v1/v2 | sealed comparisons |
| variable activity exposure | sealed |
| causal representative activity | sealed |
| unsupervised/contrastive representations | do NOT transfer (cross-dataset law) |

## 3. The queue to run (in priority order)

1. **Window-mask contract adversarial tests** (CPU, no H1 data):
   implement per `docs/DESIGN_H1_WINDOW_MASK_CONTRACT_V1_20260830.md` §4.
   All tests must pass before any GPU cell.
2. **Masked dense-auxiliary training cell** (GPU, W=700): the contract's §3
   loss law, λ ∈ {0.1, 0.3, 1.0} selected inside source LOSO folds only;
   last-bin metrics and checkpoint rule unchanged. Separate work order first.
3. **Causal output filter α sweep on H1** (GPU inference only): transfer test
   of the DANDI result (α=0.7 causal EMA, +0.01–0.017 external there); H1
   eval is causal (FSU/TTA) so a causal EMA is contract-legal.
4. **CAL-AUG prefix-cycle on H1** (GPU, training pair): motivation =
   `tfpd_exploration/results/cal_aug_v1/` (prefix invariance, external M4
   +0.0265 / M10 +0.0350 on DANDI); needs an H1-specific budget structure and
   its own matched T0. Do NOT port DANDI numbers as claims (guidance §10).

## 4. Disciplines (binding)

* additive files only; never edit sealed/frozen modules; receipts 0444 +
  sidecars, attempt-before-data, terminal-or-failure;
* no target-session fitting; checkpoint selection never reads held sessions;
* pre-register gates before runs; report failures as failures.

## 5. Not included here (by size)

`results/h1_date_lodo_checkpoint_cache_v1` (1.6 GB) and `*.pt` artifacts —
regenerate via the sealed scripts if needed; see the manifest.
