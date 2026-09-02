# H1 CAL-AUG All-Source M3 Deployment V1 Work Order

Status: pre-registered finalization design. The currently authorized stage is limited to this work order, a pure-CPU contract, CPU tests, and a no-write dry-run. It does not authorize source-plan fitting, NWB loading, CUDA initialization, model training, prediction, packaging, local scoring, or EvalAI submission.

## Objective and claim boundary

This additive successor will train a final all-source matched T0/C1 pair from all 13 registered H1 `held-in-calib` recordings and prepare C1 for M3 deployment. T0 remains the matched M7 control; C1 remains the already validated deterministic M7/M5/M4 prefix-cycle arm. M3 is a deployment calibration budget only and is not added to either training recipe.

The sealed Experiment-4 M4 governing result remains unchanged. The sealed `STRONG_M3_PREFIX_EXTRAPOLATION` result is secondary mechanism evidence supporting the pre-registration of M3 deployment; it is not a new training-budget or hyperparameter selection surface.

## Sealed lineage

- Experiment-4 A1 seal: commit `c60052c9d8ccb8391d6ce53bde9ccfb4f2319884`; terminal SHA-256 `dc9e7ab44954d3d193f67f9bf8936aafdaf2b05be9968d5e0091c0b0ecf092fd`.
- M4 local-heldout feasibility STOP seal: commit `0d0ab2f`; terminal SHA-256 `3ff971dc576958b13ace990bcca8aea2e8b999e2af2ed50f418296d05f8d5cfc`.
- M3 prefix-transfer execution seal: commit `36a9f58`; terminal SHA-256 `199a2fec864d7ae40d33ec911e43cd32e8623e5687ed9d44c5c9ac946a964429`; verdict `STRONG_M3_PREFIX_EXTRAPOLATION`.
- No file in any of those result roots or their sealed source modules may be modified.

The future canonical result root is `tfpd_exploration/h1_series_20260830/results/h1_cal_aug_all_source_m3_deployment_v1/`. Future quiet logs belong under `logs/h1_cal_aug_all_source_m3_deployment_v1/`. Neither root may be created during the current review stage.

## Local FALCON evaluation inventory and boundary

A filename-only inventory of Dandiset `000954` found exactly three local NWB surfaces:

| Surface | Files | Sessions | Local role |
|---|---:|---|---|
| `sub-HumanPitt-held-in-calib` | 13 | S0–S5 | all-source fitting/training and M3 calibration |
| `sub-HumanPitt-held-in-minival` | 13 | the same S0–S5 sessions | independent-file local held-in sanity scoring |
| `sub-HumanPitt-held-out-calib` | 14 | S6–S12 | M3 calibration and packaging only |

There is no local `eval`, held-out-minival, or test directory. The installed `FalconEvaluator.get_eval_handles(..., phase="test")` requires `<H1 data>/eval`, and its missing-data error states that test-phase data are only available on EvalAI remote.

Therefore:

1. **Locally possible:** later fit/freeze the all-source plan, prior, normalizer and carrier cache; train and integrity-check T0/C1; construct per-session M3 identity/carrier payloads for all 27 registered sessions; run an interface/package smoke; and optionally compute a local held-in-minival R² for S0–S5 using calibration from the corresponding held-in-calib file. Such a score is an engineering sanity result on known held-in sessions, not held-out generalization and not the governing official score.
2. **Requires FALCON official/EvalAI:** any leakage-free post-calibration R² for the 14 new S6–S12 recordings. Their local held-out-calib files contain exactly three legal trials and are calibration-only. Neural/behavior labels from an independent subsequent test stream are absent locally. Only the remote hidden `test/eval` stream can produce the governing held-out score.

No EvalAI call, credential access, upload, or submission is permitted by this work order's current stage. A later submission requires separate explicit authorization and a frozen submission policy.

## Registered data and session mapping

All-source fitting and training use exactly the 13 S0–S5 held-in-calib recordings already registered by `H1_HELDIN_SESSIONS`. The seven held-out dates contain exactly the 14 S6–S12 calibration recordings sealed by the feasibility audit. FALCON's canonical dataset hash mapping must be copied into the future package authority and verified one-to-one: each calibration filename and corresponding eval filename must resolve to the same `Sx_set_y` key.

No held-in-minival or held-out-calib neural/behavior data may enter source plan fitting, normalizer fitting, training, checkpoint selection, budget selection, or hyperparameter selection. Filename-only roster verification is permitted before execution; content access requires the appropriate future immutable attempt.

## All-source H-C authority

After a later explicit execution authorization, the source authority is fitted once from all 13 held-in-calib recordings:

1. Validate the exact session/NWB roster and immutable input hashes.
2. Reuse the regeneration source-inner date-LODO analytic selection semantics and its already registered candidate grid `q={4,8,12,16}`, `lambda={1e-3,1e-2,1e-1,1,10}`. This inherited source-only analytic procedure is part of the fixed recipe, not a new target-driven sweep.
3. Refit PCA, `U`, `mu`, `tau2`, the analytic EB prior and source RMS normalizer on the complete 13-recording source roster after selection.
4. Materialize every legal continuous earliest-M4 training carrier and seal the arrays, candidate table, roster/NWB hashes, plan/prior/normalizer authority and SHA-256 values.

The all-source authority must have zero held-in-minival and held-out access. It is shared byte-for-byte by T0 and C1.

## Matched all-source training contract

T0 and C1 are trained fresh from the same initial state. They differ only in the identity prefix presented for a scheduled M7 block.

- model: `H1CarrierIdSpint`, `h=32`, 4-D carrier, seven outputs, `W=700`;
- T0 identity prefix: fixed M7;
- C1 identity prefix: the existing deterministic balanced cycle M7/M5/M4;
- training H-C carrier for both arms: earliest M4 of that same scheduled M7 block;
- M3 is forbidden from the training cycle;
- seed 42, batch size 32, Adam `lr=5e-5`, weight decay 0, FP32;
- 50 epochs, last-bin MSE, prediction scale `/20`;
- dynamic dropout `U(0,1)` sequence must have the same digest in both arms;
- matched query windows, M7 starts, batch order, carrier bytes, target bytes, optimizer construction and step count;
- no validation/checkpoint selection, early stopping, SWA, warm start, target fitting, TTA, budget sweep or hyperparameter sweep;
- only the fixed zero-based epoch-49 checkpoint is retained.

Pair integrity must verify initial state, schedules, shared tensors, dropout digest, training steps, checkpoint epoch, embedded provenance and final checkpoint SHA before any scoring stream is opened. Target results cannot choose between epochs, seeds, budgets, or checkpoints.

## M3 deployment calibration

For every new session, use its earliest three legal chronological calibration trials:

- identity is the three interpolated trial identities;
- carrier is computed by the existing `fit_deployment_carrier(record, frozen_all_source_plan, earliest_three_trials)` path;
- the frozen all-source plan/prior/normalizer is used without refitting `q`, `lambda`, PCA, prior or normalizer;
- calibration behavior labels are used only by the analytic M3 carrier fit;
- optimizer, backward, parameter update, warm start and TTA counts are all zero.

Those three trials are calibration-only. They must never enter governing R², loss, prediction selection, or performance reporting. A governing prediction requires the subsequent stream, complete causal `W=700` history, last-bin output and `/20` scaling.

## Local held-in-minival diagnostic

If separately authorized after both epoch-49 checkpoints are frozen, a local evaluator may pair each S0–S5 `held-in-calib` M3 payload with the separate same-key `held-in-minival` file. It must audit that every scored eval-valid output has complete W700 history and compute float64 seven-output variance-weighted R² per recording before any aggregation. This surface is labeled `LOCAL_HELDIN_MINIVAL_SANITY_ONLY` and may not be called official held-out R² or used to modify training, packaging, or submission policy.

The present work order does not authorize this local scoring run.

## Official held-out deployment path

The future carrier-aware decoder package must map every S6–S12 hidden eval filename to the corresponding held-out-calib M3 identity/carrier via FALCON's `hash_dataset` key. The existing generic `third_party/falcon_challenge/spint_decoder.py` is not compatible with `H1CarrierIdSpint`: it passes identity only and does not pass the required H-C carrier. It remains untouched. A successor-specific adapter must later:

- load a frozen T0 or C1 epoch-49 checkpoint and immutable all-source authority;
- carry per-key identity `[3,1024,176]` and carrier `[176,4]` payloads;
- reset causal buffers per recording and never update model state;
- pass both identity and carrier into `H1CarrierIdSpint`;
- preserve W700 streaming, last-bin prediction and `/20` scaling;
- emit required interface outputs during warm-up while declaring only full-history, official eval-valid bins scientifically eligible;
- record package/checkpoint/calibration payload SHA-256 and verify model state before/after inference.

The official metric itself is computed against hidden labels by FALCON/EvalAI. This repository cannot locally recompute or independently audit that hidden held-out R². Whether a future authorized submission sends only the predesignated C1 deployment artifact or also a separately registered T0 control must be frozen before any upload; remote results may not be used to choose a checkpoint or alter the recipe.

## Planned additive files

Current review-stage additions:

- `H1_CAL_AUG_ALL_SOURCE_M3_DEPLOYMENT_V1_WORK_ORDER.md` — this binding design;
- `SPINT-main/src/h1_cal_aug_all_source_m3_deployment_v1_contract.py` — pure-CPU constants and fail-closed validators;
- `SPINT-main/tests/test_h1_cal_aug_all_source_m3_deployment_v1_contract.py` — CPU/no-data contract tests;
- `scripts/run_h1_cal_aug_all_source_m3_deployment_v1.py` — default/only dry-run review CLI.

Future files, not implemented or enabled now:

- `SPINT-main/src/h1_cal_aug_all_source_m3_deployment_v1.py` — attempt, all-source authority, paired training, integrity and terminal execution;
- `SPINT-main/third_party/falcon_challenge/h1_carrier_id_decoder_v1.py` — successor-specific carrier-aware streaming decoder;
- a detached supervisor under `scripts/` and additive execution amendment;
- immutable receipts under the fresh canonical result root and quiet logs under the registered log root.

Sealed Experiment-4, M3 diagnostic, generic decoder and data modules are never edited.

## CPU contract and test plan

The review-stage module is stdlib-only. Importing it must not import Torch, CUDA, pynwb, h5py, model code, or any NWB loader. Tests must establish:

1. exact 13 held-in and 14 held-out session rosters, unique FALCON keys and S0–S12 partition;
2. local inventory classification: held-in calib + held-in minival + held-out calib, with no local held-out eval/test stream;
3. exact model/training constants and that C1 is only `(7,5,4)`, balanced deterministically, and never contains M3;
4. matched-arm contract and zero validation/selection/warm-start/target-fitting fields;
5. M3 calibration uses exactly three trials, rejects scoring on calibration rows, and requires an independent later stream;
6. held-in-minival is classified as local held-in sanity only; S6–S12 governing score is classified as EvalAI-only;
7. carrier-aware package payload shapes and one-to-one FALCON session-key mapping;
8. dry-run reports zero writes, zero file/data/NWB access, zero CUDA, zero training, zero inference, zero scoring, zero packaging and zero submission;
9. the review CLI exposes no train/evaluate/package/submit/prepare/smoke entry point;
10. this work order records the no-leakage and official-score boundary.

## Future execution order (not yet authorized)

1. Add and review an execution amendment; commit code and freeze code closure.
2. Run the formal CPU/no-data gate.
3. Publish immutable attempt before source NWB or CUDA access.
4. Fit/seal the all-source authority with zero minival/held-out access.
5. Run paired GPU smoke and matched T0/C1 training; freeze epoch-49 checkpoints.
6. Verify pair integrity and state provenance.
7. Under a separate access attempt, materialize M3 calibration payloads and package the carrier-aware decoder; optionally run the clearly labeled local held-in-minival sanity evaluation.
8. Stop locally. Do not submit to EvalAI until a distinct explicit authorization freezes the submission artifact(s), phase and reporting policy.
