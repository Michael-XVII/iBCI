# Amendment: H1 Masked Dense-Auxiliary V1 GPU and Path Repair

The first supervisor invocation failed before any NWB read or CUDA construction because its additive runner passed `data_dir` as `str` while the established DataModule requires a `Path` in saved hyperparameters. The immutable failure and experiment record remain as historical evidence. A second attempt is authorized only after a focused CPU/no-data closure gate verifies the repaired runner.

The user subsequently authorized physical GPUs 2 and 3. The effective allowlist is therefore GPU 0–3, with at most two concurrent cells, one cell per GPU, and an idle requirement of less than 1024 MiB allocated and less than 10 percent utilization immediately before launch. The supervisor dynamically chooses two idle GPUs; it never preempts a running process.

This amendment changes no dataset split, lambda, seed, epoch, gate, model, loss, outer-access rule, or formal-heldout prohibition.
