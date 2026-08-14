# M2 Decoder E-Time-Structure Diagnostic Experiment - 2026-08-14

## Summary

This experiment follows `HANDOFF_DECODER_LANE_20260813.md` item 1: freeze the trained consumer/decoder, perturb the learned identity waveform `E` along its time axis, and measure whether last-bin decoding still works. Per the follow-up request, the primary run was repeated in the original SPINT-style held-out analysis mode: held-in data supplies the training loss, official `held-out-calib` is evaluated during validation/test, and `val_heldout/r2_mean` selects the checkpoint.

Primary held-out-selected result: T4 reaches `test_heldout/r2_mean=0.300307`, Z4 reaches `0.234755`, so T4 improves official held-out session-mean R2 by `+0.065552`. Destroying E time structure collapses both arms to near-zero held-out R2 and removes the T4-Z4 gap. The decoder therefore uses the temporal layout of `E`; the observed T4 lift is carried through a live waveform interface, but Z4 also depends on that interface, so this diagnostic is not by itself evidence of a T4-specific decoder mechanism.

## Provenance

- Branch: `exp/decoder-lane-e-time-structure`
- Base git SHA at run start: `2108eca05c0b0b6fc8f0313cbdb956e1e9c13161`
- Runtime: Docker container `docker-ial-mohd`
- Python: `/home/ial-mohd/workspace/envs/spint/bin/python`
- Dataset: `/home/ial-mohd/dataset/ial-mohd/000953`
- Teacher checkpoint: `/home/ial-mohd/workspace/SPINT/logs/train/runs/m2_ca_ffn512_50ep/checkpoints/best_ckpt/epoch_049.ckpt`
- GPU use: held-out-selected T4 on GPU 0, held-out-selected Z4 on GPU 1, diagnostic on GPU 0. PyTorch emitted the existing RTX 5090 `sm_120` compatibility warning, but CUDA execution proceeded.

## Implementation Notes

- Added `t4_z4` to native FALCON M2 side groups. It preserves the B3S/T4 architecture and parameter count but returns an exact zero `[N,4]` side tensor.
- Restored iBCI `SpintModel` compatibility with SPINT checkpoints that set `cross_attention_dim_feedforward=512`; the prior iBCI constructor implicitly rebuilt FFN as 2048 and could not load the SPINT M2 teacher.
- Added `b3s_t4_m2_decoder_lane_heldout_spint.yaml` and `b3s_t4_z4_m2_decoder_lane_heldout_spint.yaml` for all-held-in/minival training with official held-out validation/test.
- Added `scripts/diagnose_m2_decoder_e_time_structure.py`, which loads a trained checkpoint, computes `E`, and evaluates three frozen-consumer conditions: `original`, `mean_time = E.mean(dim=-1)` broadcast across 50 bins, and deterministic `permute_time` with seed `20260813`.
- Added `tests/test_decoder_e_time_diagnostic.py` for synthetic E perturbation checks, R2 accumulator sanity, and exact-zero `t4_z4` side features.

## Calibration And Selection

The primary run follows original SPINT-style held-out analysis:

- Train loader: held-in calibration sessions only. Held-out samples are not in `train_dataset` and do not contribute to the training loss.
- Validation/test loaders: include official `held-out-calib` when `include_heldout_in_fit=true` and `include_heldout_in_test=true`.
- Checkpoint selector: `val_heldout/r2_mean`, with early stopping enabled.
- Interpretation: this is held-out-selected analysis, not a blind/source-only official held-out result.

Held-in sessions used for all-held-in training/minival:

`ses-2020-10-19-Run1`, `ses-2020-10-19-Run2`, `ses-2020-10-20-Run1`, `ses-2020-10-20-Run2`, `ses-2020-10-27-Run1`, `ses-2020-10-27-Run2`, `ses-2020-10-28-Run1`.

Official held-out sessions evaluated:

`ses-2020-10-30-Run1`, `ses-2020-10-30-Run2`, `ses-2020-11-18-Run1`, `ses-2020-11-19-Run1`, `ses-2020-11-24-Run1`, `ses-2020-11-24-Run2`.

## Primary Runs

| Arm | Run directory | Best checkpoint | Selector | `test_heldin/r2_mean` | `test_heldout/r2_mean` |
|---|---|---|---:|---:|---:|
| T4 | `logs/train/runs/2026-08-14-12-19-51-095688_rid-decoder_lane_t4_m2_heldout_spint_fNone_s42` | `epoch_011.ckpt` | `val_heldout/r2_mean=0.300307` | 0.674606 | 0.300307 |
| Z4 | `logs/train/runs/2026-08-14-12-19-51-130100_rid-decoder_lane_t4_z4_m2_heldout_spint_fNone_s42` | `epoch_009.ckpt` | `val_heldout/r2_mean=0.234755` | 0.674042 | 0.234755 |

Primary paired held-out delta: `T4 - Z4 = +0.065552` session-mean R2.

## E-Time Perturbation Diagnostic

Diagnostic artifacts:

- `streaming_calibration_exp/outputs/decoder_lane/e_time_structure_heldout_spint_s42/M2_DECODER_E_TIME_STRUCTURE_DIAGNOSTIC.csv`
- `streaming_calibration_exp/outputs/decoder_lane/e_time_structure_heldout_spint_s42/M2_DECODER_E_TIME_STRUCTURE_DIAGNOSTIC.json`
- `streaming_calibration_exp/outputs/decoder_lane/e_time_structure_heldout_spint_s42/M2_DECODER_E_TIME_STRUCTURE_SUMMARY.csv`
- `streaming_calibration_exp/outputs/decoder_lane/e_time_structure_heldout_spint_s42/M2_DECODER_E_TIME_STRUCTURE_SUMMARY.json`

The table below uses the diagnostic script's pooled variance-weighted R2 over each split. The SPINT primary metric above remains session-mean R2.

| Arm | Split | Original R2 | Mean-time R2 | Delta | Permute-time R2 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| T4 | held-in all/minival | 0.696157 | 0.067361 | -0.628796 | -0.011849 | -0.708006 |
| Z4 | held-in all/minival | 0.696181 | 0.067332 | -0.628849 | -0.010381 | -0.706562 |
| T4 | official held-out | 0.317016 | -0.027959 | -0.344975 | -0.035738 | -0.352754 |
| Z4 | official held-out | 0.248924 | -0.028450 | -0.277375 | -0.033725 | -0.282649 |

Matched T4-Z4 deltas under the same pooled diagnostic metric:

| Split | Original T4-Z4 | Mean-time T4-Z4 | Permute-time T4-Z4 |
|---|---:|---:|---:|
| held-in all/minival | -0.000024 | +0.000029 | -0.001468 |
| official held-out | +0.068092 | +0.000492 | -0.002013 |

Official held-out session-mean diagnostic R2:

| E condition | T4 | Z4 | T4-Z4 |
|---|---:|---:|---:|
| original | 0.300307 | 0.234755 | +0.065552 |
| mean-time | -0.029068 | -0.029486 | +0.000418 |
| permute-time | -0.039664 | -0.037727 | -0.001937 |

Official held-out per-session original R2:

| Session | T4 | Z4 | T4-Z4 |
|---|---:|---:|---:|
| `ses-2020-10-30-Run1` | 0.451750 | 0.361820 | +0.089930 |
| `ses-2020-10-30-Run2` | 0.456965 | 0.388471 | +0.068495 |
| `ses-2020-11-18-Run1` | 0.316592 | 0.203901 | +0.112692 |
| `ses-2020-11-19-Run1` | 0.238800 | 0.169849 | +0.068951 |
| `ses-2020-11-24-Run1` | 0.254950 | 0.225400 | +0.029550 |
| `ses-2020-11-24-Run2` | 0.082784 | 0.059089 | +0.023695 |

## Interpretation

The handoff gate said: if both arms move by less than `0.01` after `mean_t(E)` broadcast or time-bin permutation, the waveform contract is unused. That gate clearly fails in the useful direction. On official held-out, T4 drops from `0.300307` session-mean R2 to `-0.029068` under mean-time and `-0.039664` under permutation. Z4 similarly drops from `0.234755` to `-0.029486` and `-0.037727`. The decoder is therefore sensitive to the temporal layout of `E` even though only the final behavioral bin is scored.

The T4 advantage is present under the original SPINT-style held-out-selected metric (`+0.065552`), but after destroying E time structure the advantage is essentially gone (`+0.000418` for mean-time, `-0.001937` for permutation). The conservative conclusion is that T4 improves the identity waveform delivered to an active decoder interface; it does not show that the decoder mechanism is uniquely T4-specific, because an activity-only Z4 identity also collapses when the waveform is damaged.

This keeps decoder-lane temporal/interface ideas alive, especially designs that alter how side information enters the decoder. It does not promote a new decoder design yet; the next item in the handoff, query-side T4, is the more direct load-bearing test because Z4 should not be able to reproduce the same carrier-built query signal.

## Internal LOSO Reference Only

Before the held-out-selected rerun, the same diagnostic was run on LOSO fold 0 with checkpoints selected by `val_heldin/r2_mean`. That run is not the primary answer to the held-out request, but it agreed qualitatively: official held-out T4 was `0.318303`, Z4 was `0.231740`, and both collapsed to near/under zero after E mean-time or permutation.

## Verification

Commands completed after the held-out-selected changes:

```bash
pytest -q tests/test_decoder_e_time_diagnostic.py tests/test_falcon_t4_features.py tests/test_side_feature_encoder.py
python -m py_compile src/data/falcon_datamodule.py src/models/components/spint.py src/models/streaming_calibration_module.py scripts/diagnose_m2_decoder_e_time_structure.py
pytest -q tests/test_decoder_e_time_diagnostic.py
python -m py_compile scripts/diagnose_m2_decoder_e_time_structure.py
```

Result: the broad side-feature regression passed with `56 passed, 17 skipped`; the focused diagnostic test passed with `3 passed`.
