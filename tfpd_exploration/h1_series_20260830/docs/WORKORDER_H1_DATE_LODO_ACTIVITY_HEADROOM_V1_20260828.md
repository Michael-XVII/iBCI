# H1 five-date frozen activity-headroom breadth

## Question

Does the positive fold0 H1 activity-state signal generalize across the five
completed confirmatory date-LODO H-C checkpoints?

This is the required breadth experiment after the variable-cardinality
source-training successor failed. It performs no new training and does not
change the H-C carrier, decoder, checkpoint, target labels, or metric.

## Frozen inputs

- target dates, in order: 19250108, 19250113, 19250115, 19250119, 19250120;
- the original fresh-seed42, fixed epoch-49 H-C checkpoint for each date;
- each checkpoint/config SHA must match its immutable terminal-evaluation
  receipt;
- each Phase-1 source manifest and strict post-M4 target dataset must match
  the original terminal receipt exactly;
- the five checkpoints are recovered byte-for-byte from the original 5070Ti
  host; no training replay is permitted.

## Arms

For every date and recording, evaluate the same four frozen-weight arms:

1. static first-four support activity;
2. causal rolling activity with cardinality four;
3. causal growing activity from four through at most thirty completed trials;
4. full-session activity, label-free but noncausal and diagnostic only.

The analytic H-C carrier is fixed to the first four labelled support trials in
all arms. Target optimizer, backward, and parameter-update counts are zero.
The governing breadth summary is an equal-date mean of each date's
equal-recording variance-weighted last-bin R2.

## Predeclared decision

The frozen activity mechanism has sufficient breadth to justify one new H1
target-state design only if causal growing improves over static support on at
least four of five dates and the equal-date mean paired gain is at least
`+0.01` R2. Otherwise H1 activity headroom is not broad enough and this line
stops without another learned successor.

## Completed result

The experiment completed once on GPU1. Each recovered checkpoint and config
matched its original terminal receipt; every static replay reproduced the
accepted terminal pooled R2 inside the frozen tolerance. All model-state
digests were unchanged, and target backward/optimizer/update counts were zero.

| Target date | Static equal-recording R2 | Causal-growing R2 | Paired delta |
|---|---:|---:|---:|
| 19250108 | 0.545514 | 0.585001 | +0.039487 |
| 19250113 | 0.339962 | 0.375285 | +0.035324 |
| 19250115 | 0.521320 | 0.548556 | +0.027236 |
| 19250119 | 0.317487 | 0.363780 | +0.046293 |
| 19250120 | 0.349828 | 0.446173 | +0.096346 |
| Equal-date mean | 0.414822 | 0.463759 | **+0.048937** |

The causal-growing arm is positive on 5/5 dates and 11/11 recordings. The
exact bootstrap distribution over five date-resampling positions gives a 95%
interval `[+0.032665, +0.073602]`. The predeclared 4/5 and `+0.01` gates both
pass. Final verdict:
`PASS_H1_ACTIVITY_HEADROOM_BREADTH_FOR_NEW_STATE_DESIGN`.

Result SHA256:
`65c9bb40ad45ab7b74740da88fd8081504b7656e807e76b6eb9db903450adb68`.
