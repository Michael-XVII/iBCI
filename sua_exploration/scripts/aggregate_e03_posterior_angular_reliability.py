"""Race-safe E02/T4R versus E03/T4RQ paired aggregate."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

def main() -> None:
 p=argparse.ArgumentParser(); p.add_argument("--e02", type=Path, required=True); p.add_argument("--e03", type=Path, required=True); p.add_argument("--out", type=Path, required=True); a=p.parse_args(); e03=json.loads(a.e03.read_text()); a.out.parent.mkdir(parents=True, exist_ok=True)
 if not a.e02.is_file():
  payload={"schema_version":1,"protocol":"e03_posterior_angular_reliability_v1","status":"PENDING_E02_BASELINE","e02_expected":str(a.e02),"e03":str(a.e03.resolve()),"reason":"E02 terminal artifact was absent when E03 completed; no polling or retry was performed."}
  a.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n"); print(a.out); return
 e02=json.loads(a.e02.read_text())
 for key in ("seed","task","data_dir","split_counts","teacher_sha256","variant"):
  if e02.get(key)!=e03.get(key): raise ValueError(f"mismatched {key}")
 for run,group in ((e02,"t4r"),(e03,"t4rq")):
  if (run.get("side_features") or {}).get("group")!=group: raise ValueError(f"expected {group}")
  if (run.get("training") or {}).get("calibration_n_trials")!=50: raise ValueError("calibration budget mismatch")
  if not (run.get("heldout_spint_selection") or {}).get("heldout_selected"): raise ValueError("held-out selection mismatch")
 bm=(e02.get("test_metrics") or [])[-1]; em=(e03.get("test_metrics") or [])[-1]; rows=[]
 for session in e02["session_splits"]["test"]:
  key=f"test_heldout_{session}/r2"; delta=float(em[key])-float(bm[key]); rows.append({"session":session,"e02_t4r_r2":float(bm[key]),"e03_t4rq_r2":float(em[key]),"delta_r2":delta})
 ds=[row["delta_r2"] for row in rows]; payload={"schema_version":1,"protocol":"e03_posterior_angular_reliability_v1","status":"COMPLETED","e02":str(a.e02.resolve()),"e03":str(a.e03.resolve()),"e02_mean_r2":float(bm["test_heldout/r2_mean"]),"e03_mean_r2":float(em["test_heldout/r2_mean"]),"mean_paired_delta_r2":statistics.mean(ds),"median_paired_delta_r2":statistics.median(ds),"worst_session_e02_r2":min(x["e02_t4r_r2"] for x in rows),"worst_session_e03_r2":min(x["e03_t4rq_r2"] for x in rows),"positive_session_count":sum(x>0 for x in ds),"session_count":len(ds),"parameter_count_delta":"B3S post_pool input +1 scalar; recorded per-run","decoder_query_mac_delta":0,"target_side_compute":"per-unit 3x3 posterior solve plus 2x2 quadratic form; no optimizer/backward","per_session":rows}
 a.out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n"); print(a.out)
if __name__=="__main__": main()
