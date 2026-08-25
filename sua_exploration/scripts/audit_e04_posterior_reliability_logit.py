"""Audit E04 source-only posterior reliability logit provenance."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads(args.run.read_text())
    side = run.get("side_features") or {}
    posterior = side.get("posterior_mean_t4")
    reliability = side.get("posterior_angular_reliability")
    logit = run.get("reliability_logit")
    if side.get("group") != "t4rql" or int(side.get("side_dim", -1)) != 4:
        raise ValueError("run is not an E04 t4rql / four-dimensional B3S artifact")
    if not isinstance(posterior, dict) or not isinstance(reliability, dict) or not isinstance(logit, dict):
        raise ValueError("E04 provenance receipts are incomplete")
    train = (run.get("session_splits") or {}).get("train", [])
    if posterior.get("source_sessions") != train:
        raise ValueError("posterior prior source sessions do not exactly match train split")
    if int(posterior.get("pool_size", -1)) != int(side.get("pool_size", -2)):
        raise ValueError("posterior pool budget mismatch")
    if posterior.get("target_sessions_used") is not False or posterior.get("target_optimizer") is not False or posterior.get("target_backward") is not False:
        raise ValueError("target-side posterior protocol violated")
    if reliability.get("formula_version") != "angular_posterior_variance_q3_v1" or reliability.get("consumer") != "B3S_attention_logit_bias":
        raise ValueError("unexpected reliability receipt")
    if logit.get("injection") != "cross_attention_softmax_pre_logit" or logit.get("target_optimizer") is not False or logit.get("target_backward") is not False:
        raise ValueError("E04 logit protocol violated")
    gamma = logit.get("final_gamma")
    if not isinstance(gamma, list) or len(gamma) != int(logit.get("layer_count", -1)) or any(float(value) < 0.0 for value in gamma):
        raise ValueError("invalid final non-negative gamma receipt")
    payload = {"schema_version": 1, "protocol": "e04_posterior_reliability_logit_v1", "pass": True, "run": str(args.run.resolve()), "prior_sha256": posterior["prior_sha256"], "source_sessions": train, "gamma": gamma, "target_iterative_optimization": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.out)

if __name__ == "__main__": main()
