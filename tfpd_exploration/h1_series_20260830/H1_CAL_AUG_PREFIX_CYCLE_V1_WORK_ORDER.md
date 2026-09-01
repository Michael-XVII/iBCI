# H1 CAL-AUG Prefix-Cycle V1 work order

Status: pre-registered additive five-date matched-pair successor. This work
order authorizes fresh T0/C1 training and one fixed target evaluation. It does
not authorize editing sealed modules, warm starts, target-selected checkpoints,
EMA filtering, or importing DANDI numerical claims.

## Fixed scope and predecessor

- Branch: `exp/h1-cal-aug-prefix-cycle-v1` from sealed experiment-3 commit
  `c9ad06f96d811fdf6be94c391128f0735060f4c6`.
- Dates: `19250108`, `19250113`, `19250115`, `19250119`, `19250120`.
- Data root: `/data/ial-dataset/ial-mohd/000954`.
- Result root: `results/h1_cal_aug_prefix_cycle_v1`; logs are under
  `logs/h1_cal_aug_prefix_cycle_v1`.
- Experiment-3 terminal SHA-256:
  `69ca9ac9eedabc6328bd0e6afa6556b40b479ecb6ffcfa6e66294580fd37258f`.
- Regeneration A2 terminal/source authority are immutable predecessors. Their
  plan, normalizer, source roster, M4 carrier cache, schedule and NWB hashes
  must validate before use.

## H1-specific intervention

Every H1 recording has 8--15 legal trials. The maximum common causal budget
that leaves a later scoring trial is therefore M7.

- A training sample first selects a legal contiguous M7 prefix.
- T0 always presents all seven trials to the B3S identity path.
- C1 presents the first M7, M5, or M4 trials in a deterministic balanced cycle.
- The analytic carrier is identical across arms and always comes from the first
  four trials of that same M7 prefix.
- C1 scheduling uses an isolated integer/hash domain and consumes no Python,
  NumPy, or Torch global RNG.
- Batch order, M7 start, query bytes, target bytes, carrier bytes, initial
  state, optimizer, and dynamic-dropout probability sequence are matched.

## Training and evaluation

For every date, train fresh T0 then fresh C1 on the same physical GPU:
`H1CarrierIdSpint`, h=32, W=700, 10,947,836 parameters, seed 42, batch 32,
Adam 5e-5, weight decay 0, FP32, dynamic dropout U(0,1), 50 epochs, last-bin
MSE, no validation/early stopping/SWA/warm start. Only epoch 49 is retained.

After all ten checkpoints and pair-integrity receipts pass, score raw outputs
at M4/M5/M7. For budget M, identity uses the earliest M chronological trials,
the carrier uses the earliest four, and inference begins at trial M+1 with a
complete W700 history. There is no EMA, target optimizer, backward, update,
TTA, or target-driven selection.

The metric is float64 seven-output variance-weighted R2 per recording, equal
recording within date, and equal date across five dates. The result is
`PASS_H1_CAL_AUG_PREFIX_CYCLE_TRANSFER` only when M4 delta C1-T0 is at least
+0.01, at least four of five M4 date deltas are positive, and both M5 and M7
equal-date deltas are at least -0.01. Otherwise the scientific terminal is the
complete negative `COMPLETE_H1_CAL_AUG_PREFIX_CYCLE_NO_TRANSFER`.

## Execution and artifacts

Attempt precedes NWB and CUDA access. CPU gate precedes source authority,
paired GPU smoke, throughput gate, ten training cells, pair verification,
target evaluation, and terminal verification. Shared GPUs 0--3 are authorized;
different dates may run concurrently, but each T0/C1 pair stays serial on one
UUID. An arm has an eight-hour timeout and GPU resource waiting has a 24-hour
timeout. There is no automatic retry.

The supervisor is launched with `nohup` plus `setsid`, is independent of the
Codex PTY, emits no progress bar, and writes only start, epoch summaries,
terminal, and error lines to logs. JSON/Markdown receipts are publish-once mode
0444 with sidecars. Checkpoints, arrays, and raw logs remain Git-ignored; their
SHA sidecars and all receipts are tracked.
