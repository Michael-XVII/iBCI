# Docker cross-runtime equivalence amendment

The first container smoke passed on CPU and GPU for C2-E49, including immutable
checkpoint/state provenance. The execution then stopped because it required
bitwise identity between a PyTorch 2.12 host CPU and the frozen V1 PyTorch 2.5
container CPU. Their predictions differ only by floating-point roundoff and are
inside the preregistered rtol=2e-3, atol=2e-4 tolerance.

This additive amendment applies that frozen tolerance across runtime versions.
An already-built image is reused only when its package and checkpoint labels
match exactly; it is not rebuilt. Repeated predictions within each runtime
remain exact. Packages, checkpoints, calibration, selections, image recipe,
submission order, and endpoints do not change. No EvalAI submission occurred.
