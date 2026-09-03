# Amendment A1 — Docker Daemon Proxy Reachability

The original packaging attempt failed before building T0 because Docker daemon
proxy `127.0.0.1:7897` was unavailable. The active host proxy is
`127.0.0.1:17897`; direct registry access through it was verified read-only.

This additive amendment uses a fresh result root ending in `_a1`. During the
detached supervisor only, `socat` forwards loopback TCP port 7897 to 17897. The
forwarder is terminated in a `finally` block. No Docker daemon configuration is
edited and Docker is not restarted, so unrelated running containers are not
interrupted.

No scientific or packaging contract changes: checkpoint/package bytes, H-C
authority, M3 calibration payloads, image definitions, CPU/GPU smoke, numerical
equivalence, no-data/no-score boundary and no-EvalAI-submission rule remain
identical. The failed V1 root remains immutable and no automatic retry occurs.

