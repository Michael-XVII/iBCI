# M1 PCT4-v1 Source-Only Protocol

**Date:** 2026-08-12
**Branch:** `exp/m1-pct4-v1-source-only`
**Scope:** native FALCON M1 held-in/source-only development only. Official M1 held-out must remain closed for this protocol.

## Rationale

M1 is already a negative boundary for the previously tested carrier family: Original/T4/D4 official held-out scores do not show positive carrier gain, and matched EMG-AFC4 content was negative. PCT4 is therefore authorized only as a new M1-specific estimand, not as post-hoc tuning of T4.

The scientific hypothesis is that whole-trial T4 averages across reach and post-contact/manipulation activity and may erase phase-dependent functional identity. Direction+Object is not a valid production carrier for the official M1 10-trial calibration because held-out calibration is structurally object-degenerate.

## Estimator

`PCT4 = [a_reach, c_reach, a_post, c_post]` per channel.

The two phase windows are fixed before any R2 result is inspected:

- Reach: `move_onset_time <= t < contact_time`
- Post-contact/manipulation-hold: `contact_time <= t < stop_time`

For a phase window `[start, stop)`, map events to official M1 bin-end timestamps with:

```python
lo = np.searchsorted(bin_timestamps, start, side="left")
hi = np.searchsorted(bin_timestamps, stop, side="left")
```

Then apply the raw FALCON `eval_mask` inside the window, sum raw binned neural counts, and divide by retained-bin count. OLS design is `[1, cos(theta), sin(theta)]`; rank must be exactly 3. The final carrier keeps signed coefficients only, no magnitude and no baseline.

Estimator version: `m1-pct4-v1-event-aligned-bin-end`.

## Controls

- `pct4`: real phase-conditioned signed coefficients.
- `pct4_z4`: exact zero side tensor at consumer input, not normalized zero raw values.
- `pct4_rs`: deterministic row shuffle by session name and seed; content real but channel attachment wrong.
- `pct4_ls`: deterministic label shuffle before OLS; reach and post use the same shuffled angle order.
- `t4`: existing whole-trial native M1 T4 path, unchanged.
- `baseline`: matched native MUA source-only no-side-feature baseline.

## Source-Only Rules

- Dataset: `/home/ial-mohd/dataset/ial-mohd/000941`.
- Held-in sessions: `ses-20120924`, `ses-20120926`, `ses-20120927`, `ses-20120928`.
- Validation: LOSO folds `0..3` over held-in sessions.
- Calibration: chronological first 10 trials, `random_calibration=false` for all side-feature arms.
- PCT4 requires `smooth_calibration=false` and `task=m1`.
- `include_heldout_in_fit=false` and `include_heldout_in_test=false` for every source-only run.
- PCT4 normalization is fitted only from LOSO train sessions.
- Official M1 held-out must not be loaded, evaluated, selected on, aggregated, or used for decisions.

## Teacher Lineage

A valid M1 teacher checkpoint is required for streaming student experiments. The legacy local path named `m1_epoch50` was found to contain an M2 checkpoint and must not be used. Any teacher used for this protocol must itself be M1 and selected without official held-out. This branch adds `falcon_m1_source_only_teacher` for that purpose; checkpoint selection monitors `val_heldin/r2_mean` only and the datamodule skips held-out when the held-out flags are false.

## Matrix

Seed-42 gate matrix:

```text
folds: 0,1,2,3
arms: native_mua_f0_m1_loso_internal
      b3s_t4_m1_loso_internal
      b3s_pct4_m1_loso_internal
      b3s_pct4_z4_m1_loso_internal
      b3s_pct4_rs_m1_loso_internal
      b3s_pct4_ls_m1_loso_internal
```

Every command must explicitly include:

```text
data.data_dir=/home/ial-mohd/dataset/ial-mohd/000941
seed=42
data.num_workers=0
data.pin_memory=false
require_baseline_validation=false
model.teacher_ckpt_path=<source-only-M1-teacher.ckpt>
```

If seed 42 passes the gate, repeat the same matrix for seeds `43` and `44`. If seed 42 fails or is indeterminate, stop PCT4-v1 and do not tune phase boundaries or features.

## Promotion Gate

Primary content gate: `PCT4 - Z4`.

PCT4-v1 can only be promoted if, on source-only held-in LOSO:

- `mean(PCT4-Z4) >= +0.03`
- `PCT4-Z4` positive for every seed mean and every fold mean
- `mean(PCT4-RS) > 0`
- `mean(PCT4-LS) > 0`
- `mean(PCT4-T4) > 0`

Temporal-granularity claim requires the stronger condition:

- `mean(PCT4-T4) >= +0.03`

If the gate is not satisfied, record PCT4-v1 as negative or indeterminate and keep official held-out closed.
