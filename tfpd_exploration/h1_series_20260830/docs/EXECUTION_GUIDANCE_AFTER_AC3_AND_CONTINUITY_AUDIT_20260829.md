# Execution Guidance After the AC3 and Continuity Audit

Date: 2026-08-29

Status: execution guidance for independent review; no launch authorization

Primary scope: DANDI 000688 sub-C/sub-M center-out, using the sealed Cell-D producer

Out of scope: automatic M1 or H1 transfer, formal FALCON submission, new target updates

## 1. Decision summary

The earlier conversational queue proposed the following first GPU experiment:

> Add full-window supervision to Cell-D while keeping last-bin scoring.

That proposal must **not** be implemented for this DANDI Cell-D route. The
premise was wrong.

The sealed Cell-D training route already:

- predicts all `W=50` output bins;
- computes dense MSE over all valid output bins;
- draws windows that are fully contained inside rewarded trials;
- uses no left-padded or intertrial query windows in this DANDI route.

The review that motivated the proposed T1 cell inspected
`streaming_calibration_exp/src/models/falcon_module.py`. That is not the training
module used by the sealed DANDI Cell-D run. The real call chain is:

```text
run_pop_robust_cell.py
    -> run_admission_arm.train_epoch
    -> StreamingSpintModel / SpintModel
    -> dense valid-bin MSE over prediction [B,50,2]
```

The relevant data path is
`sua_exploration/mc_maze/multisession_datamodule.py`, not the generic FALCON
sliding-window dataset. It constructs `valid_starts` separately inside every
rewarded trial. A selected 50-bin window cannot cross a trial boundary.

Therefore:

1. cancel the proposed DANDI Cell-D `T1 = add full-window supervision` cell;
2. do not add a new mask tuple to this Cell-D batch for this purpose;
3. reinterpret the measured trajectory-alignment gain as ensembling of already
   supervised redundant positions, not recovery of 49 untrained positions;
4. keep matched retraining `T0` as mandatory for every genuinely new training
   intervention;
5. finish one bounded AC3 utility bridge using the zero-parameter group ensemble;
6. make calibration/B3S-only robustness the first genuinely new decoder-training
   candidate;
7. stop the line if that candidate and the AC3 utility bridge are null.

This document supersedes only the conversational T0--T4 queue. It does not edit,
reinterpret, or authorize mutation of any immutable result.

Terminology rule: this document does not use `M2` as shorthand for DANDI 000688.
`M2` is reserved for the distinct FALCON track. Any earlier conversational use
of `M2 Cell-D` below has been replaced by the exact DANDI center-out surface.

### 1.1 Exact mapping from the earlier T1--T4 queue

The earlier four ideas are not all discarded, but they are not equally
unaffected:

| Earlier item | DANDI Cell-D disposition after this audit | Next action |
|---|---|---|
| T1: add full-window supervision | **Cancel.** Sealed DANDI Cell-D already trains with dense valid-bin loss over all 50 output positions. | Do not implement or launch a DANDI Cell-D T1 successor. Reuse the completed preliminary loader audit when specifying M1/H1 separately. |
| T2: gauge/unit/calibration/T4 augmentation | **Keep only after splitting the bundled intervention.** A new query-unit dropout arm is redundant with Cell-D's existing dynamic `U(0,1)` whole-unit dropout. T4 perturbation has separate negative precedents. | Complete **CAL-AUG C1** first: perturb only the B3S calibration prefix; keep query activity, T4, loss, model, and optimizer identical to matched T0. |
| T3: session-affine nuisance regularization | **Keep as a conditional research hypothesis, not an immediate GPU cell.** | First prove source-only identifiability, gauge control, grouped-OOF dispersion, and deployment removal of the nuisance head. |
| T4: smooth-basis output head | **Keep as a conditional compression/parameter-sharing hypothesis.** Its old high-frequency-denoising rationale is invalid. | First run the frozen-prediction, source-only basis reconstruction audit. Train only if a small fixed basis is nearly lossless and improves source-held calibration dispersion. |

Therefore the next training design to complete is the matched **T0 versus
CAL-AUG C1** work order. AC3-U and SLOT-AUDIT should be completed in parallel
because they are zero-training measurements and do not consume the GPU slot.

## 2. Evidence checked for this guidance

### 2.1 Cell-D producer and training path

| Item | SHA-256 at review |
|---|---|
| `tfpd_exploration/scripts/run_pop_robust_cell.py` | `761cc5306e7d1c1e4684771edeae36331f535f13d8cb07cf56db711fa6b7190b` |
| `tfpd_exploration/scripts/run_admission_arm.py` | `a1073e6271771cb49322a4dac3ae5b6164117683b56069c6441b305c215fb5ad` |
| `sua_exploration/mc_maze/multisession_datamodule.py` | `674fb4c235ba8f9393a6d1614f1f6f4260177ed9751e88acb4c05c4396d81e2d` |
| `streaming_calibration_exp/src/models/components/spint.py` | `0542f06f3cf92cb2d562982d7244c36e7bc48dd4bcd3244ee93f9040082bcae7` |
| sealed Cell-D terminal | `b3431db41efee937e83245e010ffaba50e50c517b6679d1d5285c889edbb7442` |
| sealed Cell-D SWA | `626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd` |

The exact producer loss is equivalent to:

```python
prediction, _ = model(neural, calib_trials=calib, side_features=side)
valid = (behavior != -1.0).all(dim=-1)
diff2 = ((prediction - behavior) ** 2).sum(dim=-1)
loss = (diff2 * valid).sum() / (valid.sum() * behavior.shape[-1])
```

This uses every valid temporal position in the output tensor.

The DANDI loader constructs query windows as:

```python
for trial in rewarded_trials:
    for start in range(trial.start, trial.stop - window_size + 1):
        valid_starts.append(start)
```

Thus the generic FALCON concerns about padded prefixes and windows crossing
intertrial regions do not apply to this DANDI Cell-D producer.

### 2.2 Continuity evidence

| Item | SHA-256 at review |
|---|---|
| `results/continuity_probe_v1/continuity_probe_v1.json` | `8afaa9109f2dbeb1fb68e1044c4e7ae275771113c4cb3daf5c5565ea77c588b1` |
| `docs/DESIGN_COMMON_MODE_BOUNDED_CARRIER_SUCCESSOR_20260829.md` | `0d39b398cc9d4604c110061e4e0e37719e93b0ffb3b7af2b823aac5e8ee819bb` |

The strongest within-M10 trajectory-alignment row is approximately `+0.0418`
at `K=16`, with 15 bins of delay. Because all 50 output positions were already
supervised, this gain means:

> overlapping windows provide multiple imperfect, correlated estimates of one
> absolute behavior bin, and delayed averaging reduces part of their error.

It does **not** mean that dense training supervision is missing.

The phrase "zero-lag" must not be used as a latency claim. `trajalign(K)` needs
windows ending at `t ... t+K-1`, so it incurs `K-1` bins of output delay.

### 2.3 AC3-0 terminal

The completed AC3-0 graph is immutable and internally consistent:

| Artifact | SHA-256 |
|---|---|
| `attempt.json` | `ca9732bca0b1f39a43c14cd80b4df6577029887bba4d47488259324b28760118` |
| `materialize.json` | `7abd9e8ac128fe03361231151d27242140ad39a9039ef367db7da29d7ee3adde` |
| `selection.json` | `9c812366087ed3a2347958f7b03d25bc0d6377201a295ff88f836928a173ac9a` |
| `screen.json` | `963955b578cc8627c8208e0cc5ad34d0481fd8036f65b7a57ca315a1e3fa4235` |
| `terminal.json` | `bb4e2d0710da7c5e82bcd4f77bb28f3458900768dcdc041edabd70e8813ae407` |

All corresponding sidecars match the exact file bytes. The terminal status is
`TERMINAL`. Target optimizer, backward, and update counts are zero.

The final source grouped-OOF representation table is:

| Row | Method | Circular error (rad, lower is better) | Snap mismatch |
|---|---|---:|---:|
| R0 | raw single-view direction | 0.536895 | 47.30% |
| R0.5 | zero-parameter four-group circular ensemble | 0.492767 | 45.42% |
| R1 | smoothed pseudo direction | 0.537308 | 46.97% |
| R2 | supervised circular MLP | 0.489782 | 42.78% |
| R3 | temporal contrastive | 0.543403 | 49.05% |
| R4 | cross-session action contrastive | 0.507286 | 48.03% |
| R5 | hybrid contrastive | 0.499155 | 44.51% |
| RS | shuffled-action contrastive | 0.498842 | 46.41% |
| O2 | true-direction leakage oracle | 0.000000 | 0.00% |

Direct reading:

- R-GE improves circular error over R0 by `0.044128 rad`, not the required
  `0.10 rad`;
- R2 improves over R-GE by only `0.002984 rad`, inside the declared `0.02 rad`
  equivalence band;
- R4 and R5 do not beat R2 or R-GE;
- correct-action contrastive R4 is worse than shuffled RS by `0.008443 rad`;
- the representation gate and contrastive claim gate both fail;
- the coherent carrier-utility row was not run and remains pending.

The correct scientific disposition is:

> Keep R-GE as a zero-learning baseline. Keep R2 only as a comparator. End the
> contrastive-method claim. Do not authorize AC3-2 decoder training.

AC3-0 did not measure downstream carrier utility, so it cannot by itself prove
that R-GE has zero carrier value.

## 3. What is closed

The following cells must not be built or launched from this program:

1. DANDI Cell-D full-window-supervision T1 as previously described;
2. a `decode_last_timestep_only=false` Cell-D retrain;
3. a second contrastive AC3 screen with larger embeddings or more epochs;
4. AC3-2 joint decoder/contrastive GPU training;
5. longer causal output smoothers, GRU/TCN/Transformer filters, or more alpha/K
   sweeps;
6. output-coordinate InfoNCE;
7. target-session affine fitting presented as a deployable method;
8. variance matching of predictions to target behavior;
9. a new query-activity dropout cell presented as novel augmentation: Cell-D
   already samples whole-unit dropout probability from `U(0,1)` on every
   training forward;
10. a combined activity + calibration + T4 perturbation with no single-axis
    evidence.

## 4. Immediate experiment A: AC3-U M4-only zero-learning utility bridge

### 4.1 Purpose

Close the one M4 question that AC3-0 could not answer:

> Does the zero-parameter group direction ensemble improve coherent carrier
> utility even though it misses the representation threshold?

This is not a learned-representation continuation. It is one bounded utility
measurement.

The immutable AC3 materialization is strictly:

```text
budget = M4
surface = within
sessions = 6 sub-C sessions
completed trials = 1,206
```

Therefore AC3-U is an **M4-only bridge**. It cannot make an M10 or M30 claim.
This restriction matters because the prior matched opportunity table is not
budget-flat: `O2 - O1` is `+0.0922` at M4 but `+0.0320` at M10, while the M10
gate opportunity `O1 - A1` remains `+0.0602`. A null M4 bridge does not
mathematically close those M10 quantities; testing M10 would require a separate
predeclared M10 materialization and utility bridge.

### 4.2 Fixed rows

Run exactly these rows on the existing frozen AC3 trajectories and the frozen
P2-prime coherent replay machinery:

| Row | Direction input | Role |
|---|---|---|
| U0 | R0 raw single-view pseudo direction | reference |
| UGE | R0.5 four-group circular mean + group dispersion | primary candidate |
| U2 | R2 supervised-head direction | comparator only |

Do not include R3, R4, R5, RS, new learned encoders, or a target-selected arm.
R2 is retained only because the terminal pre-registered the E7-mirror outcome;
its `0.002984 rad` advantage over R-GE is not large enough to justify a method
claim.

### 4.3 Replay contract

- source sessions only;
- exact P2-prime horizon `H=5` coherent counterfactual;
- identical trial order and proposal opportunities for U0, UGE, and U2;
- no target data or target updates;
- no decoder training;
- no carrier threshold sweep;
- no output filter;
- report equal-session R2 first, direction error second;
- preserve per-session paired deltas and deterministic input/output digests.

One session is a predeclared high-error stratum, not an excludable outlier:

| Session | R0 circular error | R0 snap mismatch | A0 matrix R2 |
|---|---:|---:|---:|
| `sub-C_ses-CO-20151103` | `1.007954 rad` | `72.05%` | `0.178389` |

It must remain in the governing equal-session mean and in the denominator of
the `4/6` breadth gate. Report all six paired deltas. A five-session sensitivity
summary may be shown only as explicitly non-governing. In practical terms, if
this session is nonpositive, the `4/6` rule requires wins in four of the other
five sessions; that implication is declared before replay rather than explained
after seeing the result.

### 4.4 Decision

Advance a small **M4** carrier-policy study only if UGE satisfies both:

```text
UGE - U0 coherent utility >= +0.01 R2
positive sessions >= 4/6
```

U2 may support a supervised-direction comparator but cannot rescue a failed UGE
gate unless it exceeds UGE by at least `0.01 R2` under the same replay.

If UGE and U2 both fail, close the AC3 continuation on this frozen M4 materialized
surface. Do not add more M4 representation capacity, seeds, folds, or an AC3 GPU
training stage. Do not rewrite that conclusion as an all-budget AC3 closure;
M10 and M30 were not present. This scope caveat does not itself authorize a new
M10 run.

## 5. Immediate experiment B: output-slot audit, no training

### 5.1 Purpose

The continuity result remains real, but its mechanism must be measured under the
correct dense-supervision fact.

Use the sealed Cell-D full prediction bodies to report, for output slots 0--49:

- per-slot source and development R2;
- per-slot MSE and bias;
- residual variance and covariance across redundant estimates of the same
  absolute bin;
- pairwise correlation as a function of slot separation;
- trajectory-alignment gain for K in `{2,4,8,16}`;
- latency in bins and milliseconds;
- last-bin and full-window state/prediction digests.

The central question is now:

> Are some already-supervised slots systematically better calibrated, or is the
> gain only covariance reduction from averaging equivalent slots?

### 5.2 Interpretation

- If individual earlier slots are poor but their average helps, the method is
  redundancy ensembling.
- If a stable subset of slots is better than the last slot across source folds,
  a fixed source-selected slot ensemble may be considered.
- If the best slot/subset changes by session, do not target-select it.
- This audit cannot authorize a new training run by itself.

## 6. First genuinely new training candidate: CAL-AUG

### 6.1 Hypothesis

Cell-D already has strong query/unit-set augmentation through whole-unit dynamic
dropout. The remaining clean single-axis training hypothesis is calibration
identity robustness:

> Make the B3S activity identity less sensitive to the exact source calibration
> prefix, while leaving query activity, T4, decoder architecture, and output loss
> unchanged.

This is a calibration/B3S perturbation only. It is not a T4 perturbation and not a
new query-activity dropout method.

### 6.2 Matched arms

| Arm | B3S calibration activity | T4 | Query activity | Loss |
|---|---|---|---|---|
| T0 | ordinary M30 | ordinary M30 | existing dynamic U(0,1) unit dropout | existing dense valid-bin MSE |
| C1 | one predeclared deterministic prefix/subsample perturbation | unchanged ordinary M30 | identical to T0 | identical to T0 |

The first candidate must use one perturbation only. Recommended first operator:

```text
deterministic B3S prefix length cycle: M30, M10, M4
T4 remains the exact ordinary M30 source carrier
```

This isolates activity-identity robustness. It deliberately does not claim to
match the complete M4/M10 deployment condition, where T4 quality also changes.

Alternative calibration perturbations must be separate named cells. Do not mix
prefix truncation, trial resampling, rate scaling, and T4 noise in C1.

### 6.3 Mandatory T0 matching

T0 and C1 must share:

- seed 42;
- canonical initial tensor state;
- model graph and parameter count;
- strict-27 source roster;
- batch size 32;
- 33,925 optimizer steps per epoch;
- 48 epochs;
- Adam parameters and warmup/cosine schedule;
- final-four SWA rule;
- batch order;
- whole-unit dropout probability sequence;
- normalized ordinary M30 T4;
- dense valid-bin MSE;
- no target selection or target update.

The prefix schedule must use an isolated deterministic arithmetic/hash domain. It
must not consume Python, NumPy, or Torch global RNG and must not change the Cell-D
dynamic-dropout `p` sequence. T0 must execute through the same successor runner,
with its prefix operator disabled, rather than comparing C1 only against the old
sealed checkpoint.

### 6.4 Cost and device binding

The accepted sealed Cell-D predecessor provides a measured cost anchor rather
than a guessed estimate:

| Quantity | Immutable predecessor value |
|---|---:|
| Device | NVIDIA GeForce RTX 3090 |
| Steps per run | 1,628,400 |
| Start to terminal | 18,512 s = 5 h 08 m 32 s |
| Matched T0+C1 reference | 37,024 s = 10 h 17 m 04 s |
| Matched-pair reference GPU cost | approximately 10.28 GPU-hours |

The historical work order planned 5--6 hours for one run, consistent with the
receipt. This is a 3090 reference, not a promise for a 5070 Ti or another GPU.

Before full authorization, run the same fixed throughput probe for T0 and C1 on
the selected device and bind:

- selected GPU UUID/profile and software stack;
- measured steps per second after preparation;
- projected seconds for 1,628,400 steps per arm;
- projected pair GPU-hours;
- separate scoring time;
- a hard wall-clock timeout and failure policy.

The default planning ceiling is `12 GPU-hours` for the two training arms,
excluding separately measured scoring. Exceeding it requires a revised explicit
authorization. T0 and C1 should run serially on the same physical GPU/profile;
using spare memory or a second unlike GPU must not introduce a hardware confound
into the matched comparison.

### 6.5 Smoke

The smoke is a feasibility and no-harm gate only. It cannot select the method.

Required evidence:

- exact T0/C1 initial state equality;
- exact optimizer and schedule equality;
- exact first-N batch indices and session order;
- exact dynamic-dropout probability digest equality;
- C1 prefix lengths and support-row digests;
- finite loss, model, gradients, and Adam state;
- B3S consumes the declared variable prefix;
- T4 bytes are unchanged;
- query neural bytes before the inherited dropout are unchanged;
- no target path is resolved or opened.

### 6.6 Mechanism readout

Because C1 changes only the B3S-activity axis while leaving T4 at ordinary M30,
the mechanism must not be judged only by the total M4/M10 deployment R2 delta.
For each arm `X` and prefix `M` in `{M10, M4}`, compute on the fixed source
roster:

```text
prefix_degradation_X(M)
    = R2_X(B3S=M, T4=M30) - R2_X(B3S=M30, T4=M30)

prefix_robustness_recovery(M)
    = prefix_degradation_C1(M) - prefix_degradation_T0(M)
```

All forwards must use the same query activity, T4, output metric, and checkpoint
within an arm. Report equal-session means and every paired source-session delta.
Also report the B3S identity-vector distance between M30 and each shortened
prefix, but treat latent distance as descriptive because collapse can make a
distance look artificially small.

A source mechanism may be registered, without making a deployment claim, when:

```text
at least one of M4 or M10 prefix_robustness_recovery >= +0.01 R2
positive source sessions >= 18/27
C1 M30 source R2 delta versus T0 >= -0.01
```

This is source-roster mechanism evidence, not held-target generalization, and it
must not be used to hide a negative deployment result.

### 6.7 Selection and reporting

No target result selects a checkpoint or hyperparameter. If a choice is needed,
make it on grouped folds drawn only from the strict-27 source roster.

After a fixed source choice, score T0 and C1 once on the same M4/M10/M30 within
and external inputs. Report:

- governing last-bin variance-weighted R2;
- equal session weighting;
- paired per-session deltas;
- positive-session counts;
- fixed-seed session bootstrap intervals;
- M30 safety;
- full prediction and state digests;
- unchanged target optimizer/backward/update counts.

Use two distinct decision levels. The lower continuation gate acknowledges that
C1 addresses only the activity side of the deployed low-budget system:

```text
at least one of external M4 or M10: mean delta >= +0.015,
                                     >= 10/15 positive,
                                     fixed-bootstrap lower bound >= 0
the other of M4 or M10: mean delta >= 0
external M30: mean delta >= -0.02
within at every reported budget: mean delta >= -0.02
```

Passing this lower gate authorizes only the next predeclared single-axis study;
it is not yet a headline performance claim. A primary performance claim still
requires the original stronger condition:

```text
at least one of external M4 or M10: mean delta >= +0.03 and >= 10/15 positive
the other low budget: mean delta >= 0
all M30 and within safety conditions above hold
```

If the source mechanism gate passes but the external continuation gate does not,
record `MECHANISM_POSITIVE__DEPLOYMENT_INCONCLUSIVE` and stop C1 expansion. Do
not call the mechanism null, and do not call it a deployable improvement.

Do not average budgets to hide an M30 regression.

## 7. Budget-matched E02/E03 successor queue

The original generic rule above placed every T4 perturbation behind a positive
C1 deployment result.  That rule is superseded for one predeclared family only:
the budget-matched E02/E03 successor specified in
`WORKORDER_BUDGET_MATCHED_POSTERIOR_CAL_AUG_V1_20260829.md`.

This exception is narrow.  It exists because E02 and E03 already have positive
same-subject session-heldout-8 evidence at M50, while the missing scientific
question is whether that mechanism survives the real M4/M10 carrier budget and
the sub-C to sub-M external surface.  It does not authorize arbitrary T4 noise,
posterior sampling, a posterior consumer, target fitting, or an E09/E10 residual
head.

### 7.1 E02 maps to C2

C2 uses one matched budget on both calibration axes for every training forward:

```text
budget cycle = M30, M10, M4
B3S activity prefix = chronological first M trials
carrier fit = the same exact M trial IDs and order
carrier = float64 posterior mean, cast once at the Cell-D input
```

The source-only prior is fitted once from strict-source M30 evidence and reused
unchanged at every budget.  A posterior-specific source normalizer is fitted
with equal M4/M10/M30 row weight.  M30 carrier bytes may not be substituted at
M10 or M4.  C2 is compared primarily with the matched C1 checkpoint, while T0
remains the no-augmentation reference.

Before any GPU smoke, every strict-source M4/M10/M30 prefix must pass the exact
rank-3 and positive residual-degree-of-freedom gate.  In particular, M4 has only
one residual degree of freedom when it is valid.  Any rank or degree-of-freedom
failure stops this exact C2 estimator; no pseudoinverse, M30 variance borrowing,
or post-hoc trial reselection is allowed.

### 7.2 E03 maps to C3-Real and two controls

E03 adds only the posterior angular-reliability scalar `q` to C2.  Its
implementation is prepared at the same time as C2; it does not wait for a C2
external result.  The matched identification set is:

| Cell | Meaning | Training cost |
|---|---|---:|
| C2 | posterior-mean carrier, no reliability input | one run |
| C3-Const | same five-dimensional input width, constant `q` | one run |
| C3-Real | real posterior angular reliability | one run |
| C3-RowShuffle-Eval | deterministic unit-row shuffle of `q` using the frozen C3-Real checkpoint | no retraining |

This is the minimum set that can distinguish a posterior-mean benefit from a
wider input layer and from genuine unit-to-reliability binding.  E03 is promoted
only if C3-Real beats both C2 and C3-Const on M4 or M10, the row shuffle removes
the relevant gain, and M30/tail safety remains acceptable.

### 7.3 Execution timing

Code, receipt schemas, synthetic tests, and the source rank/DOF audit are prepared
now on independent paths.  They must not edit `cal_aug_v1` while T0/C1 is live.
The source audit may run only when its CPU and I/O load cannot interfere with the
active matched pair.

After T0/C1 naturally terminalize:

1. descriptor-audit their immutable graphs and confirm the current source
   closure did not drift;
2. run the strict-source posterior prior/normalizer/rank/DOF audit;
3. run one bounded C2 smoke;
4. if finite, deterministic, and source-grouped-OOF non-harmful, run the matched
   C2, C3-Const, and C3-Real jobs on available compatible GPUs;
5. score all cells once on identical within and external M4/M10/M30 inputs;
6. run C3-RowShuffle-Eval from the immutable C3-Real checkpoint.

C1 does not have to cross the original `+0.015` deployment continuation gate
for the E02/E03 source audit and bounded smoke to proceed.  Its measured result
still matters: C2-minus-C1 is the clean estimate of the carrier-axis increment,
and a strongly negative C1 result raises the no-harm burden for the combined
system.  No full job is launched merely because code exists; each stage retains
its independent root, closure, device, and launch review.

### 7.4 E09/E10 are parked

Do not implement or launch E09 or E10 in parallel with this queue.  Reconsider
them only after C2/C3 have complete M4/M10/M30 and external-15 receipts.  E09
then requires a residual-only control and a worst-session non-degradation rule.
E10 is considered only if E09 first passes; its local-frame output remains an
auxiliary source-training device, while governing output and scoring stay in
Cartesian coordinates.

## 8. Affine miscalibration diagnostics

### 8.1 Correct target

The target is not the distance between the mean affine correction and the
identity matrix. The relevant quantities are:

1. dispersion of session-specific corrections;
2. performance of a zero-target-label source-mean correction (`r=0`).

Represent each correction by the six-vector

```text
[A00, A01, A10, A11, b0, b1]
```

and report at least:

- covariance trace across source-held sessions;
- leading covariance eigenvalue;
- median distance to the source mean correction;
- first-PC explained fraction;
- paired change versus T0.

Fit the source-mean correction only on source folds. Apply it unchanged to the
held session. The deployable diagnostic is:

```text
r0_gain = R2(source_mean_affine(prediction)) - R2(raw_prediction)
```

The current exploratory reference is approximately `-0.0074 (4/6)`. The desired
direction is a stable positive value, not a prettier mean matrix.

### 8.2 Leakage rule

Any session-specific affine fitted with target labels is an oracle diagnostic and
must carry:

```text
target_label_leakage = true
checkpoint_selection_eligible = false
deployment_eligible = false
```

It must not select a checkpoint, loss weight, prefix operator, or experiment.
The reported full affine opportunity of approximately `+0.0482` is diagnostic
headroom only.

### 8.3 Nuisance-regularization training

Do not launch an affine-nuisance training cell yet. It first needs a source-only
identifiability test that proves:

- the base decoder and nuisance matrices cannot trade an arbitrary gauge;
- the zero-mean or identity-centered constraint is sufficient;
- held-source dispersion is computed out of fold;
- the nuisance head is absent at deployment;
- checkpoint selection never reads target affine diagnostics.

Only after that CPU gate may an affine-dispersion training work order be written.

## 9. Smooth-basis output head

The smooth-basis route is not justified by missing supervision; dense supervision
already exists. Its only remaining rationale is parameter sharing and restriction
of session-specific output miscalibration.

Before any GPU cell, run a source-only reconstruction audit:

- project existing full-window predictions and targets onto fixed bases of
  dimension K in a small declared set;
- report reconstruction error by output slot;
- report last-bin R2 change;
- report trajectory-alignment change;
- select K only inside source folds;
- prove the basis cannot use future target labels at inference.

If no small K is nearly lossless and does not improve source-held calibration
dispersion, close this route. Do not launch a basis-head model merely because the
residual is low frequency.

## 10. M1 and H1 are separate

This correction is specific to the DANDI 000688 sub-C/sub-M center-out Cell-D
producer. It says nothing directly about the distinct FALCON M2 track.

Some M1/H1 training routes do use `FalconLitModule` with
`decode_last_timestep_only=true`. For those routes, a dense auxiliary loss may be
genuinely new. However:

- M1 uses `W=100`, not 50;
- H1 uses `W=700`, not 50;
- their loaders, masks, checkpoint policies, and source folds differ;
- the DANDI trajectory-alignment `+0.0418` is not evidence for M1/H1;
- H1 dense supervision may be computationally expensive and may include invalid
  internal positions unless its own per-bin mask is proved.

The preliminary M1/H1 loader audit is already substantially complete and must be
reused rather than repeated from zero. It established the following concrete
implementation facts for the relevant FALCON path:

- `FalconLitModule` uses `decode_last_timestep_only=true` for the present recipe;
- the generic dataset prepends `pre_history = W - 1` rows;
- window admission checks the final position rather than carrying an explicit
  validity decision for every internal output position;
- internal positions can include padded, intertrial, or still-time rows;
- the current batch does not expose the required `[B,W]` auxiliary-loss mask;
- an unmasked dense auxiliary loss could strengthen the already observed output
  shrinkage rather than improve calibration;
- last-bin metrics and checkpoint selection must remain unchanged even if an
  auxiliary dense loss is added.

The remaining M1/H1 work is not another exploratory loader audit. It is to turn
these findings into a typed mask/data contract, source-only activity-oracle
decision, adversarial tests, and a separate work order for each real `W` (`100`
for M1 and `700` for H1). Nothing in this document authorizes direct transfer.

## 11. Final execution order

The next queue is:

1. **AC3-U**: run one M4-only source coherent utility bridge for R0, R-GE, and R2.
2. **SLOT-AUDIT**: characterize all 50 already-supervised output slots and the
   delayed trajectory-alignment covariance mechanism; no training.
3. **CAL-AUG smoke**: matched T0/C1, calibration/B3S prefix only.
4. **CAL-AUG full pair**: only after smoke integrity passes; run matched T0 and C1.
5. **Affine diagnostics**: compute dispersion and `r=0` on the frozen T0/C1
   predictions; never select from target oracle affine results.
6. **E02/C2 source audit and bounded smoke**: exact M4/M10/M30 posterior
   estimator, normalizer, rank/DOF, and budget-binding gates. Implementation and
   audit preparation proceed now; execution waits for T0/C1 to release resources.
7. **E03/C3 matched controls**: prepare C3-Const and C3-Real with C2, then run
   C3-RowShuffle-Eval without retraining. Full jobs require the C2 smoke gate.
8. **Common E02/E03 scoring**: identical within/external M4/M10/M30 records,
   with C2-minus-C1 and C3-Real-minus-{C2,C3-Const} as the primary contrasts.
9. **Conditional affine-nuisance or basis-head design**: only after their CPU
   identifiability/reconstruction gates.
10. **E09/E10 review only after E02/E03**: neither is implemented or launched
    until the full low-budget/external E02/E03 result is available.
11. **Stop and write** if AC3-U, C1, C2, and C3 are all null or unsafe; preserve
    bounded source-only mechanism findings without opening an indefinite
    architecture queue.

There is no DANDI Cell-D full-window-supervision GPU cell in this queue because
that training contract already exists.

## 12. Explicit stop-and-write node

If the M4 AC3-U bridge is null and CAL-AUG plus the predeclared E02/E03 successor
do not pass, stop adding DANDI Cell-D decoder-training variants in this queue.
Do not use that terminal node to retroactively authorize E09/E10.  The paper can still report a
coherent negative/positive mechanism boundary:

- Cell-D already uses dense all-bin supervision;
- delayed trajectory alignment gives a bounded gain from redundant estimates,
  not from untrained output slots;
- causal averaging of different time points is harmful at large K;
- disjoint group residuals have high pairwise correlation (`rho_group=0.9171`),
  but this must not be called `92% common-mode variance` without a hierarchical
  variance model;
- the zero-parameter direction ensemble improves circular error modestly;
- learned contrastive rows do not beat the zero-parameter/supervised controls;
- target-label affine correction exposes about `+0.0482` oracle headroom, but
  low-budget target fitting cannot recover it;
- remaining deployable performance must come from source-trained calibration
  invariance or a different cross-session mechanism, not more output filtering.

That is a valid terminal scientific result. It is not a reason to continue an
open-ended sequence of GPU architecture trials.

## 13. Launch boundary

This document is not a launch authorization.

Before any new process:

- create an additive work order for the exact cell;
- bind the current source closure and immutable predecessors;
- prove canonical result-root freshness;
- prove no conflict with active CPU/GPU jobs;
- run no-data/no-CUDA focused tests;
- obtain an independent root review;
- launch exactly once under a separate explicit authorization;
- publish attempt before source/model/data access;
- preserve atomic terminal-or-failure receipts.

### 13.1 Closure drift policy: do not rerun for proven non-executed changes

Future successors must not use one undifferentiated package-wide glob as the
sole scientific-validity decision.  They must publish two disjoint closures at
attempt time:

1. **executed closure**: every file that can affect data materialization, model
   construction, forward/loss, optimizer/schedule, RNG, checkpoint/SWA,
   scoring, or the receipt lifecycle for this run kind;
2. **downstream-only closure**: files used only by a later mechanism,
   deployment, report, or diagnostic process and not imported or dynamically
   loaded by the active process.

Any executed-closure drift remains a hard failure.  A downstream-only drift is
recorded as a warning and does not require retraining when an independent
pre-score review proves all of the following:

- the changed path was not imported, dynamically loaded, or executed by the
  active process;
- no launch-bound input, immutable predecessor, data authority, model state,
  optimizer state, epoch receipt, checkpoint body/state, or SWA body/state
  changed;
- all numerical, finite-state, step-count, RNG, and matched-arm invariants pass;
- the classification was made before consulting the new run's governing target
  scores;
- the review is published in a separate immutable closure-review receipt that
  names the changed paths and both byte hashes.

The original terminal is never overwritten.  If a legacy runner already
published a closure failure under an over-broad glob, a successor review may
mark its training artifacts `ACCEPTED_NONEXECUTED_CLOSURE_DRIFT` while retaining
the original failure receipt and full disclosure.  This is an artifact-use
decision, not a claim that the original terminal bytes changed status.

The 2026-08-29 CAL-AUG incident is the motivating concrete case:
`run_cal_aug_cell_v1.py` imports `plan`, `receipts`, `schedule`, and `hook`, but
does not import `cal_aug_v1.mechanism`; the package `__init__.py` does not import
it either.  `mechanism.py` is consumed only by the later
`run_cal_aug_mechanism_v1.py` process.  Therefore a `mechanism.py`-only drift is
eligible for the independent non-executed review above.  Future CAL-AUG runners
must encode this split directly so the condition yields a warning rather than a
five-hour retraining decision.

Do not edit a live runner to apply this rule.  Freeze the active process, let it
terminalize naturally, and apply the policy through an additive successor or a
predeclared next-run validator.  This prevents the attempted repair from itself
creating another closure drift.
