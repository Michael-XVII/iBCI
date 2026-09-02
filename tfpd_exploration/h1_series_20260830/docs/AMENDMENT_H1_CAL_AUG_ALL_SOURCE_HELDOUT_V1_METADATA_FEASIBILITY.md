# Amendment: H1 CAL-AUG All-Source Held-Out V1 Metadata Feasibility

## Purpose

This amendment resolves one pre-training feasibility uncertainty without inspecting any held-out performance information: whether every registered H1 held-out-calib recording contains at least five legal evaluation-valid `TrialNum` values, as required for earliest-M4 calibration and scoring from trial 5.

## Revised target-access boundary

Before training, exactly one metadata-only feasibility access is permitted after an immutable attempt and CPU/no-data gate. This exception does not authorize held-out neural or behavior access.

Permitted NWB datasets:

- `TrialNum`;
- `eval_mask`, or the legacy `Blacklist` fallback solely to derive evaluation validity.

Forbidden during this audit:

- units, spikes, binned neural data or any model input;
- velocity, kinematics, behavior labels or any target values;
- full `falcon_challenge.load_nwb()`;
- carrier, plan or normalizer fitting;
- GPU/CUDA, model construction, inference or prediction;
- loss, R² or any performance calculation;
- model, budget, hyperparameter, epoch or checkpoint selection;
- persistence of trial-level metadata arrays.

## Registered roster and output

The audit must cover exactly the 14 H1 held-out sessions already registered by the predecessor repository. For each session it publishes only:

- session name;
- legal evaluation-valid chronological `TrialNum` count;
- `m4_evaluable = legal_trial_count >= 5`.

The aggregate publishes `m4_evaluable_recordings` and `all_14_m4_evaluable`.

## Gate

- Exactly `14/14` evaluable permits the original all-source T0=M7, C1=M7/M5/M4-cycle and held-out M4 plan to continue only after a new explicit user authorization.
- Any value below `14/14` terminates this M4 experiment as `STOP_H1_ALL_SOURCE_HELDOUT_M4_PROTOCOL_INFEASIBLE`.
- M3 fallback, adaptive M3/M4, padding, trial duplication or another target inspection is prohibited within this experiment.

Regardless of outcome, this audit stage launches no GPU training. Scoring-capable held-out data remains closed until both future checkpoints and pair-integrity authority are frozen.
