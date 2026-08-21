# Functional carrier for BP-free cross-session neural decoding — self-contained handoff

**Date:** 2026-08-13. **Audience:** an engineer or researcher joining this work, with no access to our other
internal documents. Every number needed to act is inlined here; nothing below requires reading another file.
**Authorizes no GPU run.**

---

## 1. The system in one page

**Base decoder (SPINT).** A cross-attention decoder over per-unit ("neuron") tokens, permutation-invariant across
units, so it tolerates a different number and identity of recorded units in every session. A learned global query
token `rep` of shape `[1, C, W]` is projected by a nonlinear read-in MLP `fc_in` (`Linear(W, model_dim) → ReLU →
Linear(model_dim, model_dim)`) and attends over the unit tokens, which serve as **both** keys and values. The
readout is `fc_out = Linear(model_dim, W)`.

Configuration, verified across all 30 model configs: `num_layers: 1` and `num_heads: 64` everywhere. `W`
(`window_size`) is `50` on the monkey centre-out and M2 datasets, `700` on H1, `100` on M1. `model_dim` is `512`
on the centre-out substrate and `1024` on H1/M1. Covariate count `C` is `2` (2-D cursor velocity), `7` on H1
(7-DoF human kinematics) and `16` on M1 (**16 EMG channels**, not kinematics — this matters in §6).

**The problem.** Because the decoder is permutation-invariant, it has no built-in notion of which unit is which.
Cross-session and cross-subject transfer therefore fails not because the attention mechanism breaks but because
the *identity assignment* fails. SPINT's original answer is a learned identity MLP with **5,965,500 parameters**
that infers identity from binned activity.

**Our contribution (T4, "the functional carrier").** Replace that learned encoder with an analytically fitted
per-unit descriptor. For each unit, fit

```
rate_i(theta) = b_i + a_i cos(theta) + c_i sin(theta)
```

by ordinary least squares over the target session's small labelled calibration set, and use
`T4_i = [a_i, c_i, m_i, b_i]` with `m_i = sqrt(a_i^2 + c_i^2)` as that unit's identity. The 4-vector is added to
the unit token stream before `fc_in`.

**The deployment contract, which is the point of the method.** On a new session there is **no optimizer, no loss,
and no backward pass**. Only forward passes through frozen source-trained parameters, one closed-form solve, and
caching. Target labels are **trial-level only**: one discrete reach direction per calibration trial, plus
task-event timestamps (`go_cue_time`, `target_on_time`, `start_time`, `stop_time`, `result`).

**A constraint that has already killed two proposals, so state it precisely.** The per-bin behavioural streams
(`cursor_vel`, `cursor_pos`, `cursor_acc`) are inadmissible at target-session calibration **in any role, including
as a mask**. One proposal died for copying an upstream pipeline's `|velocity| < epsilon` activity gate: using the
dense stream to *select* timesteps still consumes it and forfeits the label-efficiency claim. Source-domain
training is unconstrained and may read anything.

---

## 2. The central claim, with its numbers

Not "we improved accuracy on dataset X", and **not** "carrier value equals correspondence-problem severity" — that
second framing was tried and retracted, see §6.

The claim is **supervision efficiency**, and it is measured by holding features, calibration trials, query windows
and regularization fixed and changing **only the regression target**:

| endpoint | dense-label ridge | same ridge held to trial-level direction labels | T4 | T4 − sparse | T4 − dense |
|---|---:|---:|---:|---:|---:|
| centre-out, external subject, M50 | `0.417922` | **`−0.122027`** | `0.356828` | **`+0.478855`** (15/15) | `−0.061093` (3/15) |
| centre-out, external subject, M30 | — | **`−0.208580`** | `0.358154` | ≈ `+0.567` | — |
| pseudo-MUA view, M30 | — | — | — | `+0.468830` (14/15) | `−0.036153` (4/15) |
| pseudo-MUA view, M15 | — | **`−0.424391`** | `0.288234` | **`+0.712625`** (15/15) | **`+0.159127`** |

`dense − sparse` is `+0.539948` at centre-out M50 (15/15 sessions, hierarchical bootstrap
`[+0.4299, +0.6584]`) and `+0.553498` at pseudo-MUA M15 (14/15).

**Held to the same supervision, the classical baseline is negative.** At pseudo-MUA M15 our method beats even the
dense-label ridge. Across 15 sessions T4 consumes **750 direction scalars** against the dense ridge's **149,725
finite 2-D target rows** — a `199.633x` row ratio.

**The sentence for an abstract.** *A frozen cross-session decoder calibrated from roughly 750 trial-level
direction scalars — no optimizer, no loss, no backward pass on the new session, and no per-bin kinematics in any
role — comes within `0.061` R² (single-unit view) and `0.104` (pseudo-MUA view) of a per-session closed-form ridge
trained on `199.633x` more target supervision, and beats that same ridge by `+0.47` to `+0.71` R² on 14–15 of 15
sessions when the ridge is held to the same trial-level supervision.*

**A second claim, narrowed today and important to get right.** "Backprop-free" is **not** by itself a
differentiator, because the leading unsupervised-recalibration method in the field is also closed-form (§5). The
defensible claim is **backprop-free adaptation of a nonlinear attention decoder**. That competitor's closed form
exists *because its decoder is linear*; there is no analogue for a nonlinear cross-attention decoder.

**Efficiency figures, for completeness.** The identity path compresses `102.6x` (`58,140` parameters against
`5,965,500`). Whole-model compression is only `1.54x` (`10.95M` against `16.86M`) because the decoder dominates —
quote the identity-path number as an identity-path number. The full coupled decoder costs `57,970,688`
multiply-accumulates per window at `N = 64` units, decomposing exactly as source read-in `18,415,616`, query
read-in `575,488`, QKV `34,078,720`, attention output `524,288`, scores and weighted values `131,072`, FFN
`4,194,304`, readout `51,200`.

---

## 3. Where the remaining gap actually lives

A linear-decoder control on the same substrate and the same six validation sessions:

| decoder | R² | inverts a cross-unit Gram | has per-unit temporal taps |
|---|---:|---|---|
| population vector (cosine tuning — T4's own math) | `0.0870177` | no | no |
| ridge on pooled window rate (N-dimensional) | `0.0887223` | **yes** | no |
| ridge on the flattened `50 × N` window | `0.3077721` | yes | **yes** |
| SPINT with activity-only identity | `0.3139872` | — | — |
| SPINT with the functional carrier | `0.5667486` | — | — |

Two decompositions follow, and both are load-bearing:

- **Cross-unit Gram inversion is worth `+0.0017046`.** Inverting the population covariance, *with dense labels*,
  buys essentially nothing over the tuning-based population vector. This prices the entire population-statistic
  family, including whitened decoding rows, at zero.
- **Per-unit temporal taps are worth `+0.2190498`.** That is where the whole classical gain sits.

**But read the third and fourth rows together before drawing a plan from that `+0.219`.** Activity-only SPINT
(`0.3139872`) does **not exceed** the 50-tap linear ridge (`0.3077721`); the gap is `−0.006215` against a
two-sigma seed tolerance of `0.0213`. So SPINT's activity path already extracts what the 50-tap ridge extracts and
nothing nonlinear beyond it. `+0.2190498` is **50 taps versus one pooled scalar**, not 100 taps versus 50. Any
proposal to double temporal resolution must not cite it as support.

Seed sigmas on this protocol, for powering any new arm: T4 `0.010665`, activity-only `0.023207`.

---

## 4. Directions worth doing

### 4.1 Template-Ridge — the strongest remaining structural direction

**Why.** §2 shows the residual gap to the dense-label ridge is *entirely* supervision density: held to trial-level
labels that same ridge is **negative**. And §3 plus §6 show every route that tried to close the gap by
re-parameterizing the 4-vector is now closed and priced — angular resolution `+0.003979`, population whitening
`+0.0017046`, calibration budget `+0.001326`, response-window realignment closed by measurement, direction×time
kernel closed by a rank-1 collapse. **The gap is not in the descriptor. This is the only admissible route to the
supervision itself.**

**What.** Fit the ridge's *object* — a session-specific `50N → C` causal map — at the target session, against a
**synthesized** target

```
Y_synth(t) = s(t - go_cue) * [cos(theta), sin(theta)]
```

where `theta` is the trial's discrete direction label and `s` is a speed profile learned **on source sessions
only**. The target session never opens `cursor_vel`, `cursor_pos` or `cursor_acc`.

**Two versions. Choose in writing before implementing; they are not interchangeable.**

- **D-a, as a decoder.** The fitted map decodes directly, replacing SPINT at the target session. This **forfeits
  the `102.6x` identity-path compression claim** and the deployed-object accounting built on it, and it caches a
  `[50N, C]` map — `6400` floats at `N = 64` against the current `256`.
- **D-b, as a descriptor source.** Fit the map, then feed a fixed-width per-unit reduction of it into the identity
  path; deployment stays SPINT and both the compression figure and the BP-free contract survive. **D-b is
  recommended.**

**Add confidence weighting — borrowed, and it addresses the known failure mode directly.** The upstream method in
§5 refits with `decoder.fit(neural, inferredPosErr, maxProb**2)`: closed-form weighted least squares with
**squared confidence as sample weights**. Transplant that shape. `Y_synth`'s reliability is known a priori: during
the hold period `s ≈ 0` and the direction is undefined, while mid-reach it is most reliable. So weight each
constructed row by `s(t - go_cue)^2`.

This is not a decorative analogy. The reason a trial-constant sparse-direction target measures **`−0.122027`** is
precisely that it asserts a direction during the hold period when there is no movement. `s(t - go_cue)` already
attenuates those bins in amplitude, and weighting by `s^2` removes them from the fit. The two mechanisms compound.

**Costs, all four, because two are far larger than T4's and one is invisible if you only count labels.**

1. **Label cost — unchanged, and that is the point.** Per calibration trial: `target_dir` plus task-event
   timestamps. `go_cue_time` is finite on **99.89%** of rewarded trials (`13,929` of `13,945` across 53 centre-out
   sessions) and 87.1% of all trials (`14,882` of `17,092`). The claimable ratio stays `199.633x` fewer *measured*
   kinematic target rows.
2. **Design-size cost — as large as the dense comparator's, and it must be disclosed.** The solver sees thousands
   of **constructed** rows. A design of this shape is `rows 9744 / features 4600 / ratio 2.118` at centre-out M50,
   `5958 / 3100 / 1.922` at M30, and `3568 / 3100 / 1.151` at pseudo-MUA M15, with regularized Gram condition
   `63.98` and `trace_hat 1944.09` at M50. The honest sentence is "the same design size as the dense comparator,
   built from `199.633x` fewer *measured* target rows", **not** "the same supervision". Say it before a reviewer
   does.
3. **Calibration compute cost — four to five orders of magnitude above T4, and it scales quadratically.** T4 is one
   shared `[3, n_dir]` pseudo-inverse, microseconds. Template-Ridge at centre-out M50 forms a `4600 × 4600` Gram
   from a `9744 × 4600` design and factorizes it: roughly `2×10^11` flops to build the Gram and `3×10^10` to
   factorize, with a `359 MB` design matrix and a `169 MB` Gram in float64. Features scale as `O(N^2 W^2)`. On a
   workstation this is seconds; for an on-device story it is **not** free and must never be reported as "one
   closed-form solve" without its size.
4. **Cached-state cost.** D-b keeps state comparable to today's `[N, 4]` descriptor plus the `[N, 50]` identity.
   D-a is the `[50N, C]` map above.

**The failure mode is already measured, and it is not a flat.** A too-stereotyped template degenerates toward the
sparse-direction target, which measured **`−0.122027`** — a large negative, not a null. What decides this is how
much real velocity variance the template explains: a 0.6 s minimum-jerk template reaches affine-R² `0.371` and
correlation `0.609`; a source-empirical go-cue-aligned profile reaches `0.411` and `0.641`. **Freeze the template
family before running**, or the arm becomes a template search with the outcome as selector.

### 4.2 Activity-path dropout ("carrier forcing") — the training-method arm

**Why.** In a matched subject-shift experiment, within-subject carrier/zero-carrier arms score `0.574976` and
`0.326008`, while on an external subject they score `0.341367` and **`−0.143399`** — the activity path does not
merely degrade cross-subject, it goes negative. The interaction is `+0.235799` with hierarchical bootstrap
`[+0.100852, +0.371768]`. So the condition this method is paid for — decode when activity-derived identity is
unreliable — is exactly the condition source training never practises. Dropping the activity path during source
training with probability `p` forces the analytic carrier to be load-bearing.

**Three blockers, all of which must be fixed before it runs. The last two were found on 2026-08-13 and are not
optional; without them the result is uninterpretable.**

1. **Not wired on this substrate.** The centre-out training script contains **zero** occurrences of
   `activity_path_dropout`; the `0.0` visible in run metadata is a constructor default that no command-line flag
   reaches. Only the M2 path is wired. A launcher patch is required.
2. **The zero-carrier sibling is not a valid generic-regularization control.** With a zero carrier the identity is
   `post_pool([mean_feat, 0])`. Zeroing `mean_feat` leaves the carrier arm with an informative identity but leaves
   the zero-carrier arm with `post_pool([0, 0])` — a **constant**. A positive carrier-minus-zero interaction is
   therefore predicted by that asymmetry alone, with no carrier-reliance mechanism. Replace it with an
   information-matched control: dropout on the *concatenated* `[mean_feat, T4]`, or matched-magnitude noise on
   `mean_feat`.
3. **The mask is per window; the quantity is per session.** The implemented mask has shape `[B, 1]`, but
   `mean_feat` is pooled over calibration trials and is a *session* identity. Per-window dropout trains the
   consumer on an identity that flickers between windows of one recording — an unphysical high-frequency process,
   and the exact construction for which a sibling proposal was rejected. Use a session-epoch schedule.

**Gate.** Absolute external carrier-arm lift `≥ +0.03` **and** within-subject non-inferiority at `−0.03`. Twelve
source trainings (`{carrier, zero} × {p = 0, p > 0} × 3 seeds`) plus external scoring. No new teacher.

### 4.3 Two lower-priority arms, both blocked

- **Teacher-domain correction.** The decoder is warm-started from a teacher trained on a *different task and
  subject*, and `decoder.load_state_dict(teacher.state_dict(), strict=True)` runs regardless of loss mode, so
  disabling distillation does not remove the warm start. Fixing this means training a task-native teacher, **which
  does not exist yet** — that is the arm's real cost. To be explicit: we checked for data leakage and there is
  **none**. The teacher is DANDI 000128 sub-Jenkins (2009, Shenoy/Stanford, `hand_vel`); validation is DANDI
  000688 sub-C (2015, Miller/Northwestern, `cursor_vel`). No shared animal, lab, year or dandiset.
- **Source-session exposure reweighting.** Exposure is proportional to window count and nothing else reweights.
  Across the 27 source sessions there are `1,086,007` valid windows, a max/min ratio of `6.14`, a **top-5 share of
  47.4%**, and an effective session count `1/Σp² = 17.25` of 27 — and those top five sessions all fall inside a
  single 11-day span, so nearly half of source exposure comes from one fortnight with correlated unit populations.
  Equalizing raises effective distinct unit-descriptors from `1037.7` to `1613` of `1613`. **The strongest reason
  it may still be inert:** the identity path is permutation-invariant and learns a *function* of a 4-vector, so
  this is a sampling-density change, not a coverage change, and it never creates the cross-subject condition the
  method is paid for. Calibrate expectations against two comparable objective-distribution nudges already measured
  here, `+0.013748` and `+0.012833`, both under half the `+0.03` floor.

### 4.4 One arm built and ready to run

**H-U — bound the identity axis on H1 with a zero-parameter, label-free descriptor.** Width 4, computed from
neural activity alone with no behavioural input, using only rotation-invariant per-unit scalars:
`[mean_rate, fano, log_isi_cv, autocorr_tau_s]`. Principal components are deliberately **excluded**: per-session
PCs are defined only up to sign and rotation, so a PC-based descriptor would test the registration hypothesis with
a broken instrument.

Dry run on one source date: 175 of 176 units live (one channel is dead across all public recordings), the live
`175 × 4` matrix has **rank 4**, and the maximum absolute Pearson correlation against the functional carrier's
columns is `0.141` — non-degenerate and not a copy. 33 contract tests pass, including a bitwise-identity check
that the existing carrier features are unchanged. One GPU cell, about 3 hours.

**Its two readings are frozen in a receipt before results exist** (sha256
`6a94ee2ac96885d7462d14e47b8ea454dcf6414f07ddf8f29a43660ea602dd8c`), and the comparator discipline is part of the
freeze. On the H1 fold-0 protocol (single date, seed 42, epoch-49 fixed checkpoint, 8,965 strict post-support
windows):

| arm | pooled R² | init lineage |
|---|---:|---|
| carrier (H-C) | `0.5255108532950696` | `8208b6eb…` |
| zero carrier (H-C0) | `0.48661567465077105` | `8208b6eb…` |
| learned 5,965,500-parameter identity MLP (H-S) | `0.49683307249781583` | **`f4f876d5…`** |
| label-shuffled-then-refit carrier (H-LS) | `0.49989530489208744` | separately trained |

- **Only H-C and H-C0 are a controlled comparison** (shared initial state). The budget is `+0.038895`.
- If H-U lands at or near H-C0, the arm **fails**, and that must be reported as strengthening the baseline.
- If it lands materially above H-C0 toward H-C, report the fraction of `+0.038895` recovered.
- H-S has a **different initial state**, so any H-U-versus-H-S sentence is *system-level*, not controlled.

### 4.5 Open but not queued — do not mistake these for closed

Four items are alive and are deliberately *not* in the queue above. They are recorded here because a reader who
finds them mentioned only inside §6 would reasonably conclude they had been killed.

**Closed-form target readout adaptation — unmeasured, not foreclosed.** Fit a fresh `Linear(model_dim, C)` readout
on frozen hidden states at the target session by closed-form ridge, swapping only that head. An earlier contract
was written for this and would have foreclosed it had its screen returned "small headroom" — but that screen never
ran: its preflight appends a `SUPERSEDED` marker unconditionally and exits non-zero, and the scorer it names
contains no adaptation modes at all. **So the gate never fired and the idea is simply unmeasured.**

Its blocker is real, though: fitting a readout needs **per-window** targets, and one direction label per trial does
not provide them. A trial-constant target has no within-trial variance, so the fit collapses to a direction
classifier — which is exactly why the trial-level ridge in §2 measures **negative** rather than merely weak. There
is therefore no trial-level-only version. It exists only composed with a synthesized target, i.e. as a variant of
§4.1, and the one inference-based alternative source of labels is now closed (§5.2). Treat it as a possible §4.1
sub-variant, not an independent arm.

**Carrier-content corruption during source training — held.** A separate contracted arm owns the condition
"the descriptor's *content* is wrong", as distinct from §4.2's "activity identity is *absent*". It deliberately
excludes both Gaussian carrier noise and activity-path dropout so that one experiment tests one mechanism, so it
does not overlap §4.2 and cannot be folded into it. Held rather than killed.

**Estimator-noise-aware consumer training — held, and priced below the floor.** Show the consumer descriptors of
varying reliability during source training. Two facts bound it. The implementation is **dead-wired**: the setter
that installs the noise covariance factor is never called, so the config that appears to enable it produces a
bitwise-identical baseline, and wiring it is a prerequisite. And the headroom is capped by measurement: the whole
target-time budget span from M15 to M50 is `0.0187`, and M30 already sits at `+0.001326` relative to M50, so
"teach tolerance of a noisier descriptor" has at most about `0.019` to recover — under the `+0.03` floor before
starting. Note also that source-side *diversity* and target-side *noise tolerance* are genuinely different
mechanisms; rebalancing correct descriptors (§4.3) does not teach tolerance of a wrong one, so §4.3 does not
subsume this.

**Doubling temporal resolution and decoder depth — held, with one cheap step that would settle it.** §6 kills the
motivation that was offered for these, but not the family. Before spending a new teacher on either, run the
**identity time-structure diagnostic**, which is CPU-only and forward-only: on a frozen consumer, replace the
identity waveform `E` by its own time-average broadcast across the window, and separately by a permutation of its
time bins, with matched carrier and zero-carrier arms. If neither manipulation moves the score, the `W`-bin axis of
the identity is not being used, and no amount of finer binning will change that. If they do move it, the family is
worth costing. This diagnostic is the only inexpensive route to reopening the capacity/resolution axis, and it
requires no GPU and no new teacher.

**One cheap test of a retracted claim.** §6 retracts "carrier value equals correspondence-problem severity" partly
because the ordering is collinear with covariate count. That confound is directly testable without any new
supervision: rescore the `C=16` or `C=7` dataset on a **2-dimensional subset of its own covariates** and re-measure
the carrier delta. If the delta stays near zero at `C=2`, the correspondence account survives a real test for the
first time; if it jumps, the retraction was correct for a second, independent reason.

**And one non-accuracy contribution that is already in scope.** An evaluation standard for backprop-free
session adaptation — worked, immutable examples of what must be reported (comparator lineage, protocol scale,
label budget, which comparator family a number belongs to). Every misquoted number in §6 exists because no such
standard was written. This costs no GPU and is the one deliverable that improves whether or not any arm moves.

---

## 5. The external method you must engage with: PRI-T

**Reference.** G. H. Wilson et al., "Long-term unsupervised recalibration of cursor-based intracortical
brain–computer interfaces using a hidden Markov model", *Nature Biomedical Engineering*, 2025.
DOI `10.1038/s41551-025-01536-z` — <https://www.nature.com/articles/s41551-025-01536-z>
Compact reference implementation: <https://github.com/guyhwilson/PRI-T>
Full study code: <https://github.com/guyhwilson/nonstationarities>
Closed-loop datasets: <https://doi.org/10.5061/dryad.1jwstqk6g> (personal-use data withheld for privacy)

**What it does.** A hidden Markov model whose hidden state is *which discrete on-screen target the user is moving
toward*. Its observations are the **decoder's own outputs** — decoded cursor velocity and cursor position. Viterbi
yields a target sequence and the posterior marginal yields per-timestep confidence, which weights pseudo-labels in
a weighted-least-squares refit of the decoder. It therefore uses **zero** ground-truth labels. Its headline
finding is not pairwise superiority — PRI-T, factor-analysis stabilization and an adversarial alignment network are
statistically equivalent on single pairs of days, and all three fail above roughly 90° of subspace rotation. What
separates them is **iteration**: chained daily alignment accumulates compounding error it cannot undo, whereas
chained target-labelling drives accumulated error back down and matches supervised recalibration at 60 simulated
days. Validated with one month of closed-loop human control and 73 sessions over five years.

### 5.1 It is a threat to §2 and must be answered in the paper

A reviewer who knows this paper will ask why 750 trial-level scalars is impressive when PRI-T needs none. Three
legs of the answer hold, all stated by the paper itself:

1. **It needs a human in the closed loop.** The paper explains its own offline-versus-closed-loop gap by the
   user's *visual corrections*: driving a misaligned decoder while correcting is what makes the output informative
   about the true target. Our setting is offline and cross-subject, so those corrections do not exist.
2. **Its mechanism is a chain of daily sessions.** Single-pair recalibration fails above ~90° subspace rotation.
   Our setting is a single cold start on a new session or subject, with no chain available.
3. **It updates decoder weights per session**, where we keep the decoder frozen and cache an identity.

**Do not use a fourth leg that seems obvious.** "PRI-T needs a stereotyped discrete-target task" is pre-empted:
its state space is a screen discretization chosen explicitly to maintain generality, and it is validated on
radial-8, grids, **random** targets, and freeform personal use (email, web browsing) with a uniform prior.

**And it forces our own claim to narrow.** Its refit is `decoder.fit(neural_flattened, inferredPosErr,
maxProb**2)` using scikit-learn `LinearRegression` — closed-form weighted least squares, no optimizer, no backward
pass. **So "BP-free" alone is not a differentiator.** The surviving claim is BP-free adaptation of a *nonlinear*
decoder: their closed form exists because their decoder is linear.

### 5.2 What we measured when we tried to compose it with our method

The composition tested was: use PRI-T to infer the direction labels that T4 needs, making the pipeline label-free
as well as BP-free. We implemented the upstream HMM verbatim, validated the implementation on the upstream repo's
own bundled closed-loop demo with a deliberately misaligned decoder (median direction recovery `7.97°`), and ran
it on six validation sessions, 1,206 evaluation trials, 8 classes, chance `0.125`.

**Verdict: dead.** The operative condition is cold start, because T4's labels are the thing being eliminated:

- our real carrier checkpoint deployed with its identity zeroed: **`0.119`** — i.e. chance;
- a purpose-built zero-carrier checkpoint: `0.295` at the authors' recommended defaults, `0.438` as best-of-276
  configurations tuned on the evaluation sessions themselves (an optimistic ceiling);
- the encouraging `0.723` exists **only after true labels have already built the identity** — circular in the
  fatal direction, since offline inference quality is bounded by decoder quality and you need labels precisely when
  inference is worst.

The reverse order — T4 first, then PRI-T as a refinement loop on unlabelled trials — is also dead, and by **our
own** number: the target-time calibration-budget curve is M10 `−0.052565`, M15 `−0.018713`, M20 `−0.005061`, **M30
`+0.001326`**, M40 `−0.005180` against the M50 baseline `0.356828`. The descriptor saturates by 30 trials, so
adding pseudo-labelled trials beyond that buys approximately zero.

### 5.3 Three things worth borrowing anyway

1. **Confidence-squared sample weights in a closed-form refit.** Already folded into §4.1; this is the most
   valuable transfer and it directly addresses Template-Ridge's measured failure mode.
2. **Skip the HMM if you ever need offline label inference.** Snapping the mean decoded velocity to the nearest of
   8 directions scores `0.7075` against full PRI-T's `0.7232` — the entire state machinery is worth `+0.016` in an
   offline setting. Do not build a 400-state grid.
3. **Gate on the decoder's own speed.** PRI-T's emission model uses velocity *angle* and discards magnitude, which
   is harmless in closed loop and ruinous across centre-out hold periods; gating to the fastest 25% of timesteps
   moved our carrier arm from `0.383` to `0.723`. Because the gate reads the *decoded* output rather than the true
   stream, it is admissible under §1's constraint.

### 5.4 Two citations, one of which we initially got wrong

- **Cite its Fig. 1b for the estimand.** Its Methods give `x_t = b0 + b1 cos(theta_t) + b2 sin(theta_t)` fitted by
  standard least squares — identical parameterization and estimator to T4, which establishes per-unit cosine
  tuning as the field-standard characterization of this nonstationarity. One honest difference favours us: they
  regress against the *instantaneous* angle and therefore need dense kinematics, where T4 uses one scalar per
  trial. Say "same field-standard quantity, cheaper regressor".
- **Its Fig. 1f is population readout subspace drift, not per-unit tuning similarity.** It is defined over
  flattened ridge *decoder weight matrices*. The paper prints `0.75` (41°) within session and 71° at 1–2 weeks;
  it never prints `0.33`, which was our own `cos 71°` arithmetic. Quote instead the R² fall from `0.39` to `0.21`.
  Cite 1b for the estimand and 1f for population-level severity, and do not merge them.

**Reproduction judgment.** Do not reproduce the study. There is no closed-loop rig and no clinical participant
here; the comparator numbers are published; and the simulator is a synthetic-tuning model with hand-fitted drift
constants, so driving our decoder with it would replace the real drift our claim is about with synthetic drift.
Cloning the compact repository *was* necessary — the recommended defaults and the fact that speed is discarded
appear only in the source, not the paper — and its bundled `examples/exampledat.mat` provides ground-truth target
positions that served as a free positive control.

---

## 6. Closure ledger — analysed, priced, and not to be re-proposed

Everything here was closed with a number. Fourteen agents ran on 2026-08-13 and **net removed** candidates.

**Measured negative or flat.**

| Intervention | Result |
|---|---|
| Confidence-FiLM on the identity path | `+0.003399` |
| Live-activity multiplicative gain | `−0.002068` |
| Rank-8 carrier→attention-logit residual | `−0.003142`, 3/6 sessions |
| Electrode-identity gate | `−0.010817` |
| Same-electrode relational term | `−0.001440` |
| Identity interface width 32 → 64 | `−0.020130`, terminal |
| Decoupled keys/values | `−0.444658` — but this deleted the pretrained read-in, so it is a read-in ablation |
| Fixed slot router, K=32 | `−0.177935` |
| Cross-neuron encoder attention | gain explained by capacity alone: `+0.006354` against its parameter-matched control |
| Richer angular basis (8 raw per-direction means) | `+0.003979` |
| Label-free static per-unit descriptor | `+0.001588`, 3/6 |
| Fixed-K temporal prototypes | `−0.041963`, 0/4, lost to an order-invariant baseline |
| Wiener shrinkage at low budget | `+0.000753` |
| Query-fitted oracle carrier on a frozen consumer | `−0.002859` |
| Hidden-space carrier fusion (the tenth fusion arm) | `+0.012833`, 4/6, median `+0.0175`, against a `+0.03` gate |
| Carrier × distillation loss composition | `+0.059153 / +0.004307 / −0.022215`, mean `+0.013748`; spread `0.081` is six times the mean — a high-variance null |
| Cross-unit Gram / population whitening, incl. whitened decoding rows | `+0.0017046` |
| Calibration-budget increase beyond 30 trials | `+0.001326` (M30 → M50) |

**Closed by direct measurement made on 2026-08-13.**

- **Response-window realignment (anchoring the fit at the go cue).** Whole-trial versus go-cue-anchored pooling on
  three sessions gives `corr(a) = 0.9764 / 0.9432 / 0.9745` and a z-scored per-unit `[a, c]` cosine median of
  `0.9887`, with median `m_go/m_whole = 2.2993`. Although `0.7031` of the whole-trial window is pre-movement hold
  (mean trial duration `3.1461` s, mean go-cue offset `2.3996` s), the dilution is a nearly **uniform** scale
  factor, and the pipeline's per-column normalization removes uniform scale.
- **Direction × time tuning kernel.** Fitting a `[2, K]` per-unit kernel on the calibration tensor gives a rank-1
  fraction of `0.860 / 0.854 / 0.839` at M50 rising to `0.864 / 0.908 / 0.875` on all trials, and variance
  captured by the outer product of T4's direction with a single shared time shape of `0.793 / 0.763 / 0.778`
  rising to `0.816 / 0.862 / 0.832`. **Both rise with more trials**, so the residual was sampling noise. The
  descriptor is ~80% T4 times a shared shape. Closed before any GPU spend.
- **Un-sharing the readout row across covariates.** Motivated by only the last time bin being scored, so every
  covariate is read by the same row. But the dose-response in `C` is measured **inverted**: covariate
  participation ratios are `3.84–3.93` (C=16), `2.93–3.06` (C=7) and `1.82–1.91` (C=2) against `num_heads: 64`, and
  per-covariate independent dimensions run `0.95 > 0.43 > 0.24` in the *opposite* order to the prediction.

**Killed on reasoning, not measurement.**

- **Encoder–decoder co-design as a new design.** Run metadata for the mainline arm records
  `freeze_decoder = False` with the carrier active, and the flag is `store_true` which the launcher never passes.
  The mainline already jointly trains encoder and decoder with the carrier present. What is unrun is
  decoder-from-random-init, which would forfeit the transfer property rather than test co-design.
- **A session-conditioned residual on the query token.** It decomposes into a rank-≤`C` perturbation of the `C×N`
  score matrix — a *constrained special case* of the rank-8 logit residual already measured at `−0.003142` — plus
  a per-covariate output bias. Pre-norm makes the first leg stronger, not weaker.
- **Scoring all output bins, and localizing identity to the scored bin.** Windows slide, so every behavioural
  timepoint is already used once as some window's last bin; and the extra tasks predict bins from windows
  containing their own future.
- **Procrustes or CCA alignment of carrier clouds.** Requires paired rows that unordered unit populations do not
  provide, and would rotate `[a, c]` out of the shared task frame the decoder was trained in.
- **"Carrier value equals correspondence-problem severity" as the central claim.** The four-dataset ordering is
  perfectly collinear with `1/C`, and its low anchor is confounded: the `C=16` dataset's behaviour is 16 EMG
  channels while its carrier is a target-azimuth cosine, so that dataset's `−0.00652` is a descriptor–estimand
  mismatch rather than a correspondence datapoint.
- **Doubling temporal resolution to 10 ms with `W = 100`.** See §3: activity-only SPINT does not exceed the 50-tap
  ridge, so a path extracting nothing nonlinear from 50 taps is not fixed with 100. It additionally changes the
  scored target series, so its R² is not comparable to any existing number without a new common-target protocol,
  and it invalidates every sealed comparison because four weight shapes change and the strict state-dict copy
  fails.

**Numbers that keep being misquoted.**

- **`+0.296802` is a functional carrier versus an activity-only reference with a zero-width side path** — *not*
  velocity labels versus direction labels. On the same 15-fold nested cross-validation the direction-based carrier
  scores `0.448176` against dense velocity's `0.445189`, a difference of `+0.002987`, and the receipt explicitly
  declines any superiority, equivalence or non-inferiority claim.
- **`+0.2190498` is 50 taps versus one pooled scalar**, not 100 versus 50. See §3.
- **`0.693663` is a single-session, unfrozen-decoder number for an activity-only variant.** It is not a carrier
  ceiling.
- **`m2_joint_t4_upperbound`'s `+0.018389` is a contrast among independently joint-trained arms**, not a
  frozen-versus-unfrozen factorial.
- **A four-wide all-zero side port beats no port by `+0.089591`.** Port width alone has an effect, so any
  width-changing comparison needs a padded control. That contrast also mixes two different identity encoders, so
  it does not isolate a "dead port" effect.
- **The H1 four-arm set is fold-0, single date, seed 42, epoch-49** — not a five-date protocol, and the
  five-date `+0.056287` must not be combined with it. And the learned-MLP arm has a **different initial state**, so
  comparisons against it are system-level.
- **Two comparator families differ by three to five times, and the interventions measure different things.**
  Separately trained: carrier minus zero-carrier `+0.038895`, carrier minus label-shuffled `+0.025616`.
  Same-checkpoint forward-only interventions on the frozen carrier net: zeroing the carrier `+0.103872`, permuting
  its **unit rows** `+0.153079`, and rolling each support trial's velocity **in time** then refitting `+0.131571`.
  The separately-trained family lets the network re-optimize around missing content; the interventions lesion a
  network that already learned to rely on it. Both are valid answers to different questions, and **quoting only
  the conservative family understates what the trained decoder actually uses**. Decide which to report and
  disclose both.
- **A per-column normalization applied to `[a, c]` distorts an angle.** On 748 pooled units from 27 sessions,
  `mean_a = +0.03189` with `sd 1.07107` and `mean_c = +0.35127` with `sd 1.18728`, so `|mean_c| / sd_c = 0.2959`.
  The induced change in apparent preferred direction has median `15.96°`, 90th percentile `91.77°`, maximum
  `177.08°`, and an unmodulated unit acquires a spurious direction near `−96°`. This is **not** an accuracy bug —
  the downstream layer is trained and absorbs any invertible affine map, which is why the method works — but it is
  a disclosure, and it is a hard prerequisite for any rotation-based intervention. It applies to the centre-out and
  M2 substrates; H1 uses a single global RMS scale and is unaffected.
- **A band of `+0.012` to `+0.019` is not a property of the system.** Gates are *upper* thresholds and a dozen
  measured effects sit below the band, centred near zero with spread `≈±0.02`. The band members are that
  distribution's upper tail, selected on outcome, and grouping them merges a `p = 2^-12` effect with an unresolved
  null.

**Code paths that silently do nothing, or crash if used.** Each of these would waste a run.

- The setter that installs the carrier-noise covariance factor occurs **once** repo-wide — its own definition — so
  the field is always `None`, the guard never fires, and the config that appears to enable carrier noise produces a
  **bitwise-identical baseline**.
- A subset helper is called three times and defined zero times, so the consistent-unit-subset augmentation mode
  raises `NameError` if invoked.
- `activity_path_dropout` occurs **zero** times in the centre-out training script (see §4.2).
- One earlier contract's preflight appends a `SUPERSEDED` marker **unconditionally** and exits non-zero, and the
  scorer it names contains no adaptation modes at all. The gate that would have foreclosed closed-form last-layer
  adaptation therefore never fired, so that idea is **unmeasured, not foreclosed**.
- A sweep of the model tree found **no further** silent-failure paths beyond these.

---

## 7. Standing discipline

- **Primary null is label-shuffle-then-refit**, not a zeroed carrier. A zeroed carrier is a floor reference and is
  vacuous for any design where a zero input exactly reproduces the parent. Row permutations are likewise vacuous
  for any permutation-invariant set function of the carrier.
- **Every gate needs one synthetic input that must pass and one that must fail**, asserted in the aggregator test
  suite, before the gate is frozen. Five gates in this program turned out to be mathematically incapable of
  acting. Proving a gate computes correctly is not proving it can act.
- **Pre-register both readings before results exist**, and do not add a third afterwards. If the failure branch
  strengthens the baseline rather than the method, say so in those words.
- **Do not mix protocols.** Fold-0 single-date numbers, five-date numbers, within-subject numbers and
  external-subject numbers are four different scales.
- **Ablations, controls, diagnostics and evaluation standards are preconditions, not deliverables.** They travel
  with an arm; they do not get their own GPU cell.
- **Ideation now has negative expected value.** On 2026-08-13, fourteen agents net *removed* candidates, which is
  the signature of an exhausted search. The remaining value is in §4.1, §4.2, the reporting decision in §6 about
  the two comparator families, and writing the paper around §2.
