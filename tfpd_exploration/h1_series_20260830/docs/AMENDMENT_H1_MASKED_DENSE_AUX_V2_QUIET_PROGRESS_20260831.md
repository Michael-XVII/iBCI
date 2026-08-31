# Amendment: quiet Lightning progress output and epoch-0 retry

After the first two source-screen cells had started, the user requested that
per-step progress bars not be written to per-cell logs. The running processes
had no intermediate checkpoints. They were first paused to preserve exact state;
the user then explicitly selected option 2: terminate them and restart from
epoch 0. The original attempt is sealed as a failure with no target access.

The additive quiet-retry attempt uses a new result root and binds the original
failure SHA-256. Every newly launched cell sets Lightning
`enable_progress_bar=false`; warnings, errors, terminal summaries, and immutable
epoch-by-epoch held-source R2 history remain recorded.

This amendment does not change data, window indices, samplers, initialization,
loss, optimizer, epoch count, validation, checkpoints, gates, or GPU assignment.
