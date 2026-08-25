"""Strict E02/T4R versus E04/T4RQL paired aggregate."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

def metric(run: dict) -> dict:
    rows = run.get("test_metrics") or []
    if not rows: raise ValueError("missing test metrics")
    return rows[-1]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e02", type=Path, required=True)
    parser.add_argument("--e04", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    e02, e04 = json.loads(args.e02.read_text()), json.loads(args.e04.read_text())
    for key in ("seed", "task", "data_dir", "split_counts", "teacher_sha256", "variant"):
        if e02.get(key) != e04.get(key): raise ValueError(f"mismatched {key}")
    for run, group in ((e02, "t4r"), (e04, "t4rql")):
        if (run.get("side_features") or {}).get("group") != group: raise ValueError(f"expected side group {group}")
        if (run.get("training") or {}).get("calibration_n_trials") != 50: raise ValueError("wrong calibration budget")
        held = run.get("heldout_spint_selection") or {}
        if not held.get("heldout_selected") or held.get("heldout_backward_gradients") is not False: raise ValueError("held-out protocol mismatch")
    logit = e04.get("reliability_logit") or {}
    if logit.get("injection") != "cross_attention_softmax_pre_logit": raise ValueError("missing E04 pre-softmax logit receipt")
    bm, em = metric(e02), metric(e04)
    rows = []
    for session in e02["session_splits"]["test"]:
        key = f"test_heldout_{session}/r2"
        delta = float(em[key]) - float(bm[key])
        rows.append({"session": session, "e02_t4r_r2": float(bm[key]), "e04_t4rql_r2": float(em[key]), "delta_r2": delta})
    deltas = [row["delta_r2"] for row in rows]
    payload = {"schema_version": 1, "protocol": "e04_posterior_reliability_logit_v1", "e02": str(args.e02.resolve()), "e04": str(args.e04.resolve()), "seed": e04["seed"], "e02_mean_r2": float(bm["test_heldout/r2_mean"]), "e04_mean_r2": float(em["test_heldout/r2_mean"]), "mean_paired_delta_r2": statistics.mean(deltas), "median_paired_delta_r2": statistics.median(deltas), "worst_session_e02_r2": min(row["e02_t4r_r2"] for row in rows), "worst_session_e04_r2": min(row["e04_t4rql_r2"] for row in rows), "positive_session_count": sum(delta > 0.0 for delta in deltas), "session_count": len(deltas), "gamma": logit.get("final_gamma"), "parameter_count_delta": int(logit.get("layer_count", 0)), "decoder_query_mac_delta": 0, "target_side_compute": "per-unit 3x3 posterior solve plus 2x2 quadratic form and static pre-softmax attention bias; no optimizer/backward", "per_session": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(args.out)

if __name__ == "__main__": main()
