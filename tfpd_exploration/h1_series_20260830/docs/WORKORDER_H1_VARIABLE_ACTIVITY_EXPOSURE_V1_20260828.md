# H1 variable activity exposure V1

## Objective

Develop the single H1 successor supported by the frozen-weight activity
headroom result. Keep the H-C carrier, decoder architecture, last-bin loss,
source-only boundary, and target scoring contract unchanged. Fine-tune only by
exposing the identity path to variable numbers of completed activity trials.

## Training cell

- warm start: sealed H-C fold0 epoch 49, seed 42;
- source data only: the eleven non-19250101 recordings from the immutable H1
  fold0 source authority;
- carrier: normalized analytic H-C computed from the first four trials of the
  selected source recording; it is unchanged as activity cardinality grows;
- activity identity: a deterministic 50/50 replay mixture of M4 and a
  variable prefix M5 through all available source trials;
- every batch is recording-homogeneous and uses one prefix cardinality;
- batch size 128, Adam, learning rate 1e-5, weight decay 0, five epochs;
- last-bin raw-output MSE, behavior scale 20, unchanged network;
- no target file, target label, target gradient, target optimizer step,
  minival, formal, held-out, or EvalAI access during training.

The variable-prefix arm is a performance-first fine-tuning successor, not a
clean from-scratch causal ablation. Its comparator remains the immutable
sealed H-C checkpoint.

## Gates

1. CPU tests prove deterministic cardinality scheduling, exact 50/50 replay
   structure, prefix-only activity access, fixed first-four carrier, and no
   target paths in the training provider.
2. A short GPU smoke must produce finite loss and gradients, exercise both M4
   and M>4, preserve the carrier digest across cardinalities, and write no
   target evidence.
3. The full cell launches once only after the smoke passes.
4. Evaluation reuses the exact four activity arms and H1 equal-recording
   governing metric from `WORKORDER_M1_H1_ACTIVITY_HEADROOM_20260828.md`.
5. A positive result requires causal-growing equal-recording R2 above the
   sealed checkpoint's causal-growing result while static M4 does not regress
   by more than 0.01. Full-session activity is diagnostic only.

## Completed execution

The route completed once on GPU1 without retry.

- smoke: 20/20 finite nonzero-gradient source steps, exactly ten M4 and ten
  variable-prefix steps, covering M5 through M15;
- full training: five epochs and 4,555/4,555 finite nonzero-gradient steps;
- prefix replay: 2,285 M4 steps and 2,270 variable-prefix steps;
- epoch mean loss: `2.729533e-6`, `2.441817e-6`, `2.378699e-6`,
  `2.354928e-6`, `2.346027e-6`;
- source-only boundary: no target session, target label, minival, formal, or
  EvalAI surface opened during training;
- training receipt SHA256:
  `5cd23406dff6781e0105b60ccc778f349d0e494ce7f3a9a7f791828df381059f`;
- checkpoint SHA256:
  `69f642f6aeba78a2c136316b78338dec4d8f31968a6071d860c1dc5935c6e77a`;
- checkpoint state SHA256:
  `fe6746208b54fd7dc03477a4154709f1f16930c9942141ab2d29a3110be6d60d`.

## Completed matched evaluation

The score used the same fold0 two-recording input authority, 8,965 windows,
four arms, and equal-recording governing metric as the immutable sealed H-C
comparator. No target backward, optimizer, or parameter update occurred, and
the successor state digest was identical before and after scoring.

| Arm | Sealed H-C | Successor | Successor minus sealed |
|---|---:|---:|---:|
| Static M4 | 0.4564882149 | 0.4556959065 | -0.0007923084 |
| Causal rolling M4 | 0.4706254846 | 0.4578327135 | -0.0127927711 |
| Causal growing | 0.4887798017 | 0.4833430372 | -0.0054367645 |
| Full-session oracle | 0.5116554512 | 0.5018032129 | -0.0098522384 |

The static safety gate passed, but the causal-growing improvement gate failed
on both recordings. Final verdict:
`STOP_VARIABLE_ACTIVITY_SUCCESSOR`. More epochs and additional seeds are not
authorized for this exact recipe.

Matched-score receipt SHA256:
`1a0f4357cf348acdd90a723d590bec4ac7cc766031f2a24e939613379261af70`.
