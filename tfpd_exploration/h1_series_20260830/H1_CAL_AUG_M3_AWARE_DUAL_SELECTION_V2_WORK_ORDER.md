# H1 CAL-AUG M3-Aware Dual-Selection V2 Work Order

## Authority and scope

- Additive successor branch: `exp/h1-cal-aug-m3-aware-dual-selection-v2`.
- Exact predecessor: `origin/exp/h1-cal-aug-all-source-m3-deployment-v1` at `84c7aaecb656be812d046317fec07783fa6701a4`.
- Frozen baseline: V1 C1 epoch 49, SHA-256 `0f406a8e69fdb57cf6a5480149f04ab3500e7fad849d36db38042edbadb2cd06`.
- The only new training arm is C2. T0 and C1 are not retrained or modified.
- No Docker build, EvalAI submission, remote score read, successor creation, or automatic retry is authorized.

## Frozen source and model contract

C2 reuses and SHA-binds the V1 all-source q/lambda/PCA/U/mu/tau2 plan, source carrier normalizer, batch order, M7 support starts, source tensor surface, and M4 carrier cache. The model remains `H1CarrierIdSpint`, h=32, W=700, 10,947,836 parameters. Training is seed 42, batch 32, Adam 5e-5, weight decay 0, FP32, 50 epochs, dynamic dropout, last-bin MSE after prediction `/20`.

C2 uses the deterministic balanced identity prefix cycle `(M7,M5,M4,M3)`. M7/M5/M4 batches use the frozen V1 M4 carrier. An M3 batch uses the earliest three trials from the same scheduled M7 block and calls the existing `fit_deployment_carrier()` with the frozen V1 plan. Its carrier is normalized only by frozen V1 `s_src`. No H-C estimator, plan, prior, normalizer, architecture, or carrier-aware decoder is changed.

The fresh initial-state SHA must equal V1. The completed run must have exactly 206650 global/optimizer steps and exactly 206650 dynamic-dropout draws. The dropout digest must be `c1dd24d682878f477050cb4e5886dd1f34aff3424b58bdcb158c12c11ba1d247`. Any mismatch fails closed.

All epoch-end checkpoints `epoch_000.ckpt` through `epoch_049.ckpt` are saved, SHA-recorded, and excluded from Git. Training is uninterrupted: no validation, R², ranking, selection, early stopping, checkpoint choice, or target access occurs between epochs.

## Offline selection after training integrity

Only after the complete training integrity receipt passes may the validator open evaluation data.

- HI-M3: held-in-calib earliest M3 supplies identity and H-C carrier; the independent held-in-minival stream supplies scoring. Set recordings are merged into S0–S5 before float64 seven-output variance-weighted R². The primary metric is `val_hi_m3_official/r2_mean`, equally averaged across six sessions.
- HO-M3: held-out-calib earliest M3 supplies identity and H-C carrier and the local SPINT-style stream supplies development scoring. Set recordings are merged into S6–S12 before R². The primary metric is `val_ho_m3_grouped/r2_mean`, equally averaged across seven sessions. This is explicitly a development/model-selection surface, not untouched held-out generalization. The 14-recording mean `val_ho_m3_spint14/r2_mean` is secondary and cannot select a checkpoint.

Both surfaces evaluate all C2 epochs 000–049. The same validator also evaluates frozen V1 C1 epoch49. HI and HO independently select by: higher primary mean, then higher worst-session R², then lower population session standard deviation, then earlier epoch. Immutable receipts are `selection/c2_hi.json` and `selection/c2_ho.json`.

The terminal report includes V1 C1 e49, C2 e49, C2-HI, C2-HO, both 50-epoch curves, per-session metrics, Pearson/Spearman correlation between curves, and all preregistered deltas. Evaluation performs zero optimizer, backward, or model-update steps and cannot change model state.

## Execution and stop boundary

Before data or CUDA access, publish an immutable attempt with Git/code closure. Run the CPU/no-data gate, then a single GPU/memory/disk fail-fast precheck. Training runs in a detached system tmux session with `TQDM_DISABLE=1`; stdout/stderr are redirected under `logs/h1_cal_aug_m3_aware_dual_selection_v2/`. There is no GPU resource polling and no automatic retry/restart.

After local dual-selection receipts and terminal verification are sealed, stop. Future C2-E49/C2-HI/C2-HO Docker images and any EvalAI manifest/submissions require separate human authorization.
