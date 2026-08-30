# M1/H1 activity-headroom experiment

## Question

Before training another model, test whether the frozen M1 and H1 decoders are
limited by their static calibration-activity identity.

This is an activity-only intervention.  It does not change a checkpoint,
optimizer, loss, output head, target, query window, or H1 carrier.

## Frozen systems

- M1: the completed fold-20120924 matched-ERM checkpoint, selected by source
  training loss only.  The accepted static-M10 held-in score is the exact
  numerical replay anchor.
- H1: the sealed fold-0 H-C epoch-49 checkpoint.  The ordinary normalized
  support carrier is held fixed for every arm.  The accepted static-M4 score
  is the numerical replay anchor.

## Arms

Every query window is assigned to the trial containing its output bin.
Updates may use only completed earlier trials.

1. `STATIC_SUPPORT`: the original first M trials for every window (M1 M=10,
   H1 M=4).
2. `ROLLING_FIXED_M`: before each query trial, use the most recent M completed
   trials.  This is causal and cardinality matched to training.
3. `CAUSAL_GROWING_CAP30`: before each query trial, use all completed trials up
   to a FIFO cap of 30.  The original support remains in force until M trials
   have completed.  This is causal but can be calibration-cardinality OOD.
4. `FULL_SESSION_ORACLE`: use every trial in the recording for every window.
   It is label-free but noncausal and is never a deployable result.

For M1 windows before the first ten trials have completed, all causal arms use
the existing first-ten support tensor.  H1 scoring is already strictly
post-first-four.

## Metric and weighting

- final-bin predictions only;
- M1 uses its accepted TorchMetrics variance-weighted multi-output R2 path on
  float32 tensors;
- H1 preserves its established float64 SSE/TSS accumulator (one pooled SSE
  divided by the sum of per-output centered TSS);
- M1 reports its one held-in fold directly;
- H1 reports per-recording R2, pooled R2, and equal-recording mean R2.

The static arm must reproduce the accepted score before any new arm can be
interpreted.  Model state must be identical before and after evaluation.

## Decision rule

- If `ROLLING_FIXED_M` improves consistently, prefer inference-time activity
  adaptation; no GPU training is needed.
- If only `CAUSAL_GROWING_CAP30` improves, authorize at most one matched model
  trained with variable activity-cardinality exposure.
- If only `FULL_SESSION_ORACLE` improves, the result is diagnostic headroom,
  not deployable evidence; a training cell is allowed only if the improvement
  is large enough to justify the cardinality/distribution change.
- If all activity arms are null or negative, do not spend GPU on CDM-style
  activity memory for that dataset.  Record an applicability boundary and
  redirect work to a different cross-session invariance mechanism.

No formal/EvalAI surface is opened by this experiment.

## M1 breadth extension

The initial matched-ERM fold-20120924 result triggered a low-cost breadth
check before any training authorization. The same four arms are therefore run
against the three available official source-only checkpoints:

- fold0 target `20120924`, fixed source-selected epoch 18;
- fold1 target `20120926`, fixed source-selected epoch 19;
- fold2 target `20120927`, fixed source-selected epoch 19.

Each fold descriptor-binds its own checkpoint, source-only manifest, and target
NWB. Each fold's direct eager static path is its internal numerical anchor and
must agree with the cached-identity path within maximum prediction error
`2e-6` and absolute R2 error `2e-7`. Comparisons are paired only within a
fold. The three official checkpoints are not pooled with the independently
trained matched-ERM checkpoint above.

The breadth disposition uses equal fold weighting:

- a causal arm must be positive on at least two of three folds and have a
  practically nontrivial positive equal-fold mean before M1 GPU training is
  considered;
- a positive noncausal oracle with null/negative causal arms records activity
  headroom but does not authorize training by itself.

Observed equal-fold deltas versus static are `-0.0178330541` for rolling M10,
`+0.0009793043` for causal growing M10-to-M30, and `+0.0117356380` for the
noncausal full-session oracle. The M1 training condition is therefore not met.
