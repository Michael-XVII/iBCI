# Execution Amendment — H1 CAL-AUG All-Source M3 Deployment V1

Status: authorized local execution. This amendment adds source-authority, paired-training, deployment-decoder, local held-in-minival sanity, package-rehearsal and terminal layers to the sealed design at commit `a0ac939`. It does not modify that work order or any Experiment-4/M3 predecessor file.

## Authorized order

1. A CPU/no-data gate and immutable attempt precede all NWB and CUDA access.
2. Fit and seal one all-source authority from exactly the 13 registered `held-in-calib` recordings. This phase has zero held-out-calib, minival and EvalAI/test access.
3. Run a finite paired GPU smoke, then train exactly one fresh T0 and one fresh C1 checkpoint. T0 and C1 may run concurrently on physical GPU0/GPU1. No automatic retry is permitted.
4. Verify pair integrity before opening minival or held-out calibration content.
5. Build M3 calibration payloads, package both frozen arms with the additive carrier-aware decoder, and run local held-in-minival through the reloaded packaged decoder path.
6. Run CPU/GPU reload, batch compatibility, numerical-equivalence and state-immutability rehearsal.
7. Publish a terminal authority and stop locally.

## Fixed training and deployment contract

The sealed work order remains binding: all 13 source recordings; source-inner date-LODO analytic `q/lambda`; one refitted plan/prior/normalizer; T0 M7; C1 deterministic balanced M7/M5/M4; both carriers from the scheduled M7 block's earliest M4; `H1CarrierIdSpint`, h=32, W700, 10,947,836 parameters, seed 42, batch 32, Adam 5e-5 with weight decay zero, FP32, 50 epochs, last-bin MSE, `/20`, matched dynamic-dropout draws, no validation/selection/early-stop/warm-start. Only zero-based epoch 49 is retained.

Deployment is M3. The earliest three calibration trials build both identity and `fit_deployment_carrier(..., 3 trials)` carrier from the frozen all-source plan/prior/normalizer. Calibration trials are never scored. No target optimizer, backward, parameter update, fitting choice or model selection is permitted.

## Local diagnostic and package boundary

Local S0–S5 scoring must use the serialized-then-reloaded `H1CarrierIdSpintDecoder` and FALCON's streaming `predict_files` path with progress display disabled. Calibration comes from `held-in-calib`; labels/mask come only from the corresponding `held-in-minival`. Metrics are per-session float64 seven-output variance-weighted R², plus unweighted session mean, population standard deviation and C1−T0. Every label is `local held-in-minival deployment sanity R²`; it is not held-out R² and cannot change any artifact.

Packages include calibration payloads for the 13 held-in and 14 held-out FALCON keys so that a later, separately authorized hidden-test run does not require model modification. Package verification is not an EvalAI submission and does not access hidden data or official scores.

## Detached execution and STOP

Training output is quiet: START, one line per epoch, TERMINAL or ERROR only; no tqdm/progress bar. Logs are written under `logs/h1_cal_aug_all_source_m3_deployment_v1/`. The supervisor is started with `setsid`/`nohup`, has no Codex PTY/session handle, and owns all child processes. GPU resource waiting is not a retry. Any failed phase publishes an immutable failure receipt and stops without changing the protocol or result root.

The following are categorically absent: `evalai push`, remote FALCON test, official held-out score access, and post-minival tuning/retraining. A successful terminal status is `COMPLETE_LOCAL_H1_ALL_SOURCE_M3_DEPLOYMENT_READY_NO_EVALAI_SUBMISSION`.
