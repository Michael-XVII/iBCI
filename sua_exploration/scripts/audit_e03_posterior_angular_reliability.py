"""Audit E03 posterior angular-reliability provenance."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--run", type=Path, required=True); p.add_argument("--out", type=Path, required=True); a=p.parse_args()
    run=json.loads(a.run.read_text()); side=run.get("side_features") or {}; posterior=side.get("posterior_mean_t4"); reliability=side.get("posterior_angular_reliability")
    if side.get("group") != "t4rq" or not isinstance(posterior, dict) or not isinstance(reliability, dict): raise ValueError("run is not E03 t4rq")
    train=run.get("session_splits", {}).get("train", [])
    if posterior.get("source_sessions") != train: raise ValueError("posterior prior sources differ from train split")
    if posterior.get("target_sessions_used") is not False or posterior.get("target_optimizer") is not False or posterior.get("target_backward") is not False: raise ValueError("target-side posterior protocol violated")
    if reliability.get("formula_version") != "angular_posterior_variance_q3_v1" or reliability.get("target_sessions_used") is not False: raise ValueError("reliability receipt invalid")
    payload={"schema_version":1,"protocol":"e03_posterior_angular_reliability_v1","pass":True,"run":str(a.run.resolve()),"prior_sha256":posterior["prior_sha256"],"source_sessions":train,"reliability":reliability,"target_iterative_optimization":False}
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n"); print(a.out)
if __name__ == "__main__": main()
