# Runtime-dependency amendment

The second host-smoke launch stopped before GPU inference, Docker, or EvalAI.
The RTX-5090-capable falcon runtime imported the submission module, which
unnecessarily imported a receipt helper through a training module and therefore
required Lightning. The deployment model and decoder do not require Lightning.

This additive amendment replaces only receipt publication and verification with
local immutable SHA-256 helpers. Package bytes, state dicts, calibration payloads,
model/decoder code, numerical tolerances, Docker recipe, selections, and
submission governance are unchanged. No training or submission occurred.
