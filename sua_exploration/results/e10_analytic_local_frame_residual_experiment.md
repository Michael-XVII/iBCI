# E10 Analytic Local-Frame Residual

## Status

- Branch: `exp/e10-analytic-local-frame-residual`
- Seed: 42
- Dataset/split: DANDI 000688 sub-C CO SUA, chronological 37/8/8, matched to E01/E08/E09
- State: formal training running
- Started: `2026-08-28T22:21:01+08:00`
- Device: physical GPU 3 (`CUDA_VISIBLE_DEVICES=3`)
- User service: `ibci-e10-analytic-local-frame-residual.service`
- systemd invocation: `479d92079152429cb7b828210ef3a67f`
- Startup processes: service main PID 3580213; Python PID 3580215
- Formal log: `logs/e10_analytic_local_frame_residual_t4_s42.log`
- Formal result: `sua_exploration/results/p3_e10_analytic_local_frame_residual_t4_s42_seed42.json`

The formal process is launched through the user systemd manager, not a Codex
PTY or terminal session. Closing Codex therefore does not terminate training.
The one-time startup check found the service active/running, the Python trainer
and log `tee` inside the service cgroup, and the formal log initialized. No
continuing monitor is attached.

## E10 gate

- E01 test mean R2: 0.6136593819.
- E09 test mean R2: 0.6277234554, an absolute improvement of 0.0140640736.
- E09 improves 7/8 external sessions and has mean residual-energy fraction
  0.3710742, so the residual is materially smaller than the target rather than
  simply relearning the entire target velocity.
- These completed E09 results satisfy the document's bounded gate for testing
  whether the useful residual is better expressed relative to the analytic
  direction.

The training entrypoint verifies the completed E09 receipt, its seed, hash, and
the E09 > E01 mean-R2 gate before allocating the formal output directory.

## Locked analytic carrier

- The completed E08 B0-2 receipt remains authoritative.
- Ridge lambda: 100; fixed gain: 0.3090719545; W=I.
- Activity: mean firing rate over the causal 50-bin (1.0 s) window.
- T4: raw [a,c,m,b] reconstructed using source-only normalization statistics.
- Target sessions use the first 50 rewarded trials to construct T4; there is no
  target optimizer, target gradient, embedding, or continuous-velocity fit.

## Local-frame residual

For each prediction:

```text
u = v_ana / (||v_ana|| + 1e-6)
delta_v = delta_parallel * u + delta_perpendicular * J(u)
v_hat = v_ana + delta_v
```

The matched B3S/T4 coupled network still emits exactly two values and has the
same parameter count as E09. E10 changes only their semantics from Cartesian
`(delta_v_x, delta_v_y)` to the scalar corrections
`(delta_parallel, delta_perpendicular)`. The physical residual reconstructed
above is used for loss, prediction, and residual-energy diagnostics.

The decoder output head is initialized to exact zero. Thus the initial E10
prediction and B-Res-Zero control are exactly the analytic-only prediction.
At test time, B-Res-Shuffle row-permutes the analytic T4 carrier; the learned
scalars retain aligned T4, while their physical residual is reconstructed in
the shuffled carrier's local frame.

## Training protocol

- Fresh seed-42 fit; no E09 warm-start.
- Task-only end-to-end objective:
  `MSE(v_ana + delta_parallel*u + delta_perpendicular*J(u), v)`.
- 40 epoch cap, patience 10, learning rate 1e-4, batch size 32.
- Held-out SPINT checkpoint selection, deterministic training, no target-session
  backpropagation.
- Controls and formal test metrics match E09: full residual model, analytic-only,
  analytic-carrier shuffle, per-session R2, and physical residual-energy fraction.

## Verification

- Local-frame reconstruction, rotation equivariance, zero-anchor behavior,
  E08 analytic formula, and deterministic carrier shuffle: 9 focused tests pass.
- Python syntax and CLI construction checks pass.
