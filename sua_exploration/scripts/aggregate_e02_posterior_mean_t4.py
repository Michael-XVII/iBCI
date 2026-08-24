"""Strict paired E01/T4 versus E02/T4R aggregate."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

def metric(run):
    rows=run.get("test_metrics") or []
    if not rows: raise ValueError("missing test metrics")
    return rows[-1]

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--baseline", type=Path, required=True); p.add_argument("--e02", type=Path, required=True); p.add_argument("--out", type=Path, required=True); a=p.parse_args()
    base=json.loads(a.baseline.read_text()); e02=json.loads(a.e02.read_text())
    for key in ("seed","task","data_dir","split_counts","teacher_sha256","variant"):
        if base.get(key) != e02.get(key): raise ValueError(f"mismatched {key}")
    for run, group in ((base,"t4"),(e02,"t4r")):
        if (run.get("side_features") or {}).get("group") != group: raise ValueError(f"expected side group {group}")
        if (run.get("training") or {}).get("calibration_n_trials") != 50: raise ValueError("wrong calibration budget")
        held=run.get("heldout_spint_selection") or {}
        if not held.get("heldout_selected") or held.get("heldout_backward_gradients") is not False: raise ValueError("held-out protocol mismatch")
    bm, em=metric(base), metric(e02); sessions=base["session_splits"]["test"]
    deltas=[]; rows=[]
    for name in sessions:
        key=f"test_heldout_{name}/r2"
        delta=float(em[key])-float(bm[key]); deltas.append(delta); rows.append({"session":name,"e01_t4_r2":float(bm[key]),"e02_t4r_r2":float(em[key]),"delta_r2":delta})
    payload={"schema_version":1,"protocol":"e02_posterior_mean_t4_v1","seed":e02["seed"],"baseline":str(a.baseline.resolve()),"e02":str(a.e02.resolve()),"e01_mean_r2":float(bm["test_heldout/r2_mean"]),"e02_mean_r2":float(em["test_heldout/r2_mean"]),"mean_paired_delta_r2":statistics.mean(deltas),"median_paired_delta_r2":statistics.median(deltas),"worst_session_e01_r2":min(r["e01_t4_r2"] for r in rows),"worst_session_e02_r2":min(r["e02_t4r_r2"] for r in rows),"positive_session_count":sum(x>0 for x in deltas),"session_count":len(deltas),"per_session":rows,"parameter_count_delta":0,"decoder_query_mac_delta":0,"target_side_compute":"per-unit 3x3 closed-form solve; no optimizer/backward"}
    a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n"); print(a.out)
if __name__ == "__main__": main()
