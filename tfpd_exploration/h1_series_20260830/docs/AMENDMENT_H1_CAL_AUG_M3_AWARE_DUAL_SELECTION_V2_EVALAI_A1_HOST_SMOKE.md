# Host-smoke assertion amendment

The first host smoke stopped before GPU inference, Docker, or EvalAI because
the implementation incorrectly required batch-size 1 and batch-size 8 CPU
predictions to be bitwise identical. The frozen protocol requires exact
repeatability within each fixed batch size and numerical compatibility across
batch sizes; distinct GEMM batch shapes can differ by floating-point roundoff.

This additive amendment changes only that assertion: batch-size 1 versus 8
uses the already frozen `rtol=2e-3, atol=2e-4`, while repeated CPU predictions
at each batch size remain exact, repeated GPU predictions at each batch size
remain exact, and model-state hashes remain immutable. Packages, checkpoints,
calibration payloads, images, submission order, endpoints, and all scientific
selections are unchanged. No training or submission occurred before this
amendment.
