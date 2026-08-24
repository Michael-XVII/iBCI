"""Audit source-only prior provenance for E02 posterior-mean T4."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--run", type=Path, required=True); p.add_argument("--out", type=Path, required=True); a=p.parse_args()
    run=json.loads(a.run.read_text()); side=run.get("side_features", {}); receipt=side.get("posterior_mean_t4")
    if side.get("group") != "t4r" or not isinstance(receipt, dict): raise ValueError("run is not an E02 t4r artifact")
    train=run.get("session_splits", {}).get("train", [])
    if receipt.get("source_sessions") != train: raise ValueError("posterior prior source sessions do not exactly match train split")
    if receipt.get("target_sessions_used") is not False or receipt.get("target_optimizer") is not False or receipt.get("target_backward") is not False: raise ValueError("target-side posterior protocol violated")
    if int(receipt.get("pool_size", -1)) != int(side.get("pool_size", -2)): raise ValueError("posterior pool budget mismatch")
    payload={"schema_version":1,"protocol":"e02_posterior_mean_t4_v1","pass":True,"run":str(a.run.resolve()),"prior_sha256":receipt["prior_sha256"],"prior_variance":receipt["prior_variance"],"source_sessions":train,"target_sessions_used":False,"target_iterative_optimization":False}
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print(a.out)
if __name__ == "__main__": main()
