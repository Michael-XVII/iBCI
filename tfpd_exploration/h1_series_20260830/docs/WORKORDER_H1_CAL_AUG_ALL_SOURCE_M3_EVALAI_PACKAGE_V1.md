# H1 CAL-AUG All-Source M3 EvalAI Package V1

Status: authorized submission-readiness packaging only; EvalAI submission is forbidden.

## Frozen inputs

- T0/C1 epoch-49 checkpoints and Package-A1 terminal remain byte-frozen.
- All-source H-C plan/prior/normalizer and all 27 earliest-M3 calibration
  payloads remain frozen inside the verified packages.
- The additive `H1CarrierIdSpintDecoder` remains the deployment interface.
- No checkpoint selection, training, optimizer, backward, target fitting,
  model update, held-out scoring, remote evaluator, credential access, image
  push, or EvalAI submission is permitted.

## Artifacts

Prepare one Docker image per arm from the verified T0/C1 package. Each image:

- contains the exact arm package and its checkpoint/package SHA labels;
- uses H1, W=700, batch size 8, carrier-aware reset/predict, and `/20` output;
- defaults to the original FALCON remote evaluator contract with `phase=test`;
- contains no training entry point and does not require runtime calibration fitting.

The local readiness gate runs only `--evaluation smoke`; it never invokes the
remote or local scorer. CPU and GPU container smoke must load the package,
resolve a registered H1 key, reset, predict finite outputs, and preserve the
model-state SHA. Host and container predictions for the identical zero-input
sequence must be numerically equivalent. Docker image configuration and labels
must bind the correct arm, checkpoint, package SHA, H1 task, test phase and
batch size.

The canonical result root is
`results/h1_cal_aug_all_source_m3_evalai_package_v1/`. Docker images and staged
`.pt` files remain local/ignored; JSON/Markdown receipts and SHA sidecars are
mode 0444. A successful terminal means submission-ready only. It does not
authorize `docker push`, `evalai push`, or official hidden-test access.

