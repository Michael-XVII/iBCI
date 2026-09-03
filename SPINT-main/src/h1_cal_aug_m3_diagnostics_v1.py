"""Evaluation-only diagnostics for sealed H1 CAL-AUG M3 artifacts."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import r2_score

from src.h1_cal_aug_all_source_m3_deployment_v1_contract import (
    ARMS, HELDIN_SESSION_TO_FALCON_KEY, HELDOUT_SESSION_TO_FALCON_KEY,
)
from src.h1_hc_date_lodo_regen_v1 import publish_json, publish_npz, publish_text, verify_sidecar
from src.h1_m4_cce_contract import sha256_file
from src.data.h1_m4_eb_pilot import array_sha256, index_heldin_calib, session_from_path

SCHEMA = "h1_cal_aug_m3_diagnostics_v1"
PACKAGE_TERMINAL_SHA = "4137495462a299e948beb58be578c739cc211330de4769992c03e743d7c7bf26"
OFFICIAL_TERMINAL_SHA = "189608c3c20a2485ba6eba3e08e590bf6f0a55b95dcb557657e8e8b836895eed"
HELDOUT_CACHE_SHA = "d67d5dfa016983f9c3a3dfbc8d202392aa3ee1139a11fbcfe620c65aa02a78b1"
MINIVAL_CACHE_SHA = "73227a61fadd58b98ae1520431abf26457cdb2fe07346a3e8f516a85380f88ed"

class DiagnosticError(RuntimeError): pass
def _need(x: bool, msg: str) -> None:
    if not x: raise DiagnosticError(msg)
def utc_now() -> str: return datetime.now(timezone.utc).isoformat()

def session_id(key: str) -> str:
    _need("_set_" in key, f"invalid FALCON key: {key}")
    return key.split("_set_", 1)[0]

def official_grouped_metrics(arrays: Mapping[str, np.ndarray], mapping: Sequence[tuple[str,str]], mask_suffix: str) -> dict[str, Any]:
    """Call and independently reproduce official FALCON session aggregation."""
    from falcon_challenge.evaluator import FalconEvaluator
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, (_recording, key) in enumerate(mapping): grouped[session_id(key)].append(i)
    sessions = sorted(grouped)
    out: dict[str, Any] = {}
    for arm in ARMS:
        preds_all=[]; targets_masked=[]; masks_all=[]; dset_lens={}; per_session={}; per_recording={}
        for sess in sessions:
            dset_lens[sess]=[]; sp=[]; st=[]
            for i in grouped[sess]:
                pred=np.asarray(arrays[f"{arm}_{i}_prediction"], np.float64)
                target=np.asarray(arrays[f"{arm}_{i}_target"], np.float64)
                mask=np.asarray(arrays[f"{arm}_{i}_{mask_suffix}"], bool).reshape(-1)
                _need(pred.shape==target.shape and pred.ndim==2 and pred.shape[1]==7 and len(mask)==len(pred), f"shape drift {arm}/{i}")
                _need(int(mask.sum())>1, f"empty score mask {arm}/{i}")
                preds_all.append(pred); targets_masked.append(target[mask]); masks_all.append(mask); dset_lens[sess].append(len(pred))
                sp.append(pred[mask]); st.append(target[mask])
                key=mapping[i][1]
                per_recording[key]=float(r2_score(target[mask],pred[mask],multioutput="variance_weighted"))
            sp=np.concatenate(sp); st=np.concatenate(st)
            per_session[sess]=float(r2_score(st,sp,multioutput="variance_weighted"))
        official=FalconEvaluator.compute_metrics_regression(np.concatenate(preds_all),np.concatenate(targets_masked),np.concatenate(masks_all),dset_lens,verbose=False)
        values=np.asarray([per_session[s] for s in sessions],np.float64)
        _need(math.isclose(float(official["R2 Mean"]),float(values.mean()),abs_tol=1e-12),"official mean mismatch")
        _need(math.isclose(float(official["R2 Std."]),float(values.std(ddof=0)),abs_tol=1e-12),"official std mismatch")
        out[arm]={"r2_mean":float(official["R2 Mean"]),"r2_std_population":float(official["R2 Std."]),"per_session_r2":per_session,"per_recording_r2":per_recording}
    return out

def load_cache(path: Path, expected_sha: str) -> dict[str,np.ndarray]:
    _need(verify_sidecar(path)==expected_sha, f"cache SHA drift: {path}")
    with np.load(path,allow_pickle=False) as z: return {k:np.asarray(z[k]) for k in z.files}

def align_trial_num_to_batched_timeline(trial_num: np.ndarray, timeline_length: int) -> np.ndarray:
    """Mirror FalconEvaluator's right padding; padded bins can never score."""
    trial=np.asarray(trial_num,np.float64).reshape(-1)
    _need(len(trial)<=int(timeline_length),"TrialNum longer than evaluator timeline")
    return np.pad(trial,(0,int(timeline_length)-len(trial)),constant_values=np.nan)

def create_attempt(root: Path, head: str) -> None:
    _need(not root.exists(),"result root exists")
    publish_json(root/"attempt.json",{"schema":SCHEMA,"status":"ATTEMPT_BEFORE_DATA_OR_CUDA","created_at_utc":utc_now(),"git_head":head,"training":False,"checkpoint_selection":False,"optimizer_steps":0,"backward_steps":0,"model_updates":0,"evalai_submissions":0})

def validate_predecessors(package_root: Path, official_root: Path, heldout_root: Path, minival_root: Path, result_root: Path) -> dict[str,Any]:
    checks=((package_root/"terminal.json",PACKAGE_TERMINAL_SHA),(official_root/"terminal.json",OFFICIAL_TERMINAL_SHA),(heldout_root/"evaluation/prediction_cache.npz",HELDOUT_CACHE_SHA),(minival_root/"minival/prediction_cache.npz",MINIVAL_CACHE_SHA))
    for p,d in checks: _need(verify_sidecar(p)==d,f"predecessor drift: {p}")
    package=json.loads((package_root/"terminal.json").read_text()); official=json.loads((official_root/"terminal.json").read_text())
    _need(package["status"]=="COMPLETE_LOCAL_H1_ALL_SOURCE_M3_DEPLOYMENT_READY_NO_EVALAI_SUBMISSION","package terminal drift")
    _need(official["status"]=="COMPLETE_H1_M3_EVALAI_OFFICIAL_C1_IMPROVES_T0","official terminal drift")
    body={"schema":f"{SCHEMA}_predecessor","status":"PASS_FROZEN_PREDECESSORS","package_terminal_sha256":PACKAGE_TERMINAL_SHA,"official_terminal_sha256":OFFICIAL_TERMINAL_SHA,"heldout_cache_sha256":HELDOUT_CACHE_SHA,"minival_cache_sha256":MINIVAL_CACHE_SHA,"training":False,"optimizer_steps":0,"backward_steps":0,"model_updates":0,"evalai_submissions":0}
    publish_json(result_root/"predecessor_authority.json",body); return body

def heldin_calib_inference(data_root: Path, package_root: Path, result_root: Path, device: str="cuda:0") -> dict[str,np.ndarray]:
    import torch
    import falcon_challenge.evaluator as evaluator_module
    from falcon_challenge.config import FalconConfig,FalconTask
    from falcon_challenge.evaluator import FalconEvaluator
    from pynwb import NWBHDF5IO
    from third_party.falcon_challenge.h1_carrier_id_spint_decoder import H1CarrierIdSpintDecoder
    package_auth=json.loads((package_root/"packages/packages.json").read_text())
    calib=json.loads((package_root/"packages/calibration_authority.json").read_text())
    supports={r["falcon_key"]:np.asarray(r["calibration_trials"],np.float64) for r in calib["sessions"] if r["scope"]=="held-in-calib"}
    indexed=index_heldin_calib(data_root); paths=[indexed[s].resolve() for s,_ in HELDIN_SESSION_TO_FALCON_KEY]
    arrays={}; states={}; old_tqdm=evaluator_module.tqdm; evaluator_module.tqdm=lambda x,*a,**k:x
    try:
      for row in package_auth["packages"]:
        arm=row["arm"]; path=package_root/row["relative"]; _need(verify_sidecar(path)==row["sha256"],f"package drift {arm}")
        decoder=H1CarrierIdSpintDecoder(FalconConfig(task=FalconTask.h1),path,batch_size=13,device=device); before=decoder.model_state_sha256()
        pred,tgt,masks,*_=FalconEvaluator(eval_remote=False,split="h1",verbose=False,dataloader_workers=0).predict_files(decoder,paths)
        after=decoder.model_state_sha256(); _need(before==after==row["model_state_sha256"],f"state mutation {arm}"); states[arm]=before
        for i,((session,key),nwb_path) in enumerate(zip(HELDIN_SESSION_TO_FALCON_KEY,paths,strict=True)):
          p=np.asarray(pred[key],np.float32); t=np.asarray(tgt[key],np.float32); m=np.asarray(masks[key],bool).reshape(-1)
          with NWBHDF5IO(str(nwb_path),"r") as io: trial=np.asarray(io.read().acquisition["TrialNum"].data[:],np.float64)
          trial=align_trial_num_to_batched_timeline(trial,len(p))
          reuse=m & np.isin(trial,supports[key]); _need(int(reuse.sum())>1 and len(trial)==len(p),f"reuse mask drift {key}")
          prefix=f"{arm}_{i}"; arrays[f"{prefix}_prediction"]=p; arrays[f"{prefix}_target"]=t; arrays[f"{prefix}_eval_mask"]=m; arrays[f"{prefix}_reuse_score_mask"]=reuse
    finally: evaluator_module.tqdm=old_tqdm
    for i in range(len(HELDIN_SESSION_TO_FALCON_KEY)):
      _need(np.array_equal(arrays[f"t0_{i}_target"],arrays[f"c1_{i}_target"]),"paired target drift")
      _need(np.array_equal(arrays[f"t0_{i}_reuse_score_mask"],arrays[f"c1_{i}_reuse_score_mask"]),"paired mask drift")
    digest=publish_npz(result_root/"heldin_calib/prediction_cache.npz",**arrays)
    publish_json(result_root/"heldin_calib/prediction_cache.json",{"schema":f"{SCHEMA}_heldin_calib_cache","arrays_file_sha256":digest,"array_sha256":{k:array_sha256(v) for k,v in arrays.items()},"model_state_sha256":states,"training":False,"optimizer_steps":0,"backward_steps":0,"model_updates":0})
    return arrays

def run_diagnostics(heldout_root:Path,minival_root:Path,calib_arrays:Mapping[str,np.ndarray],result_root:Path)->dict[str,Any]:
    heldout=load_cache(heldout_root/"evaluation/prediction_cache.npz",HELDOUT_CACHE_SHA)
    minival=load_cache(minival_root/"minival/prediction_cache.npz",MINIVAL_CACHE_SHA)
    d1=official_grouped_metrics(heldout,HELDOUT_SESSION_TO_FALCON_KEY,"eval_mask")
    reuse=official_grouped_metrics(calib_arrays,HELDIN_SESSION_TO_FALCON_KEY,"reuse_score_mask")
    independent=official_grouped_metrics(minival,HELDIN_SESSION_TO_FALCON_KEY,"score_mask")
    independent_zero=official_grouped_metrics(minival,HELDIN_SESSION_TO_FALCON_KEY,"eval_mask")
    d1_delta=d1["c1"]["r2_mean"]-d1["t0"]["r2_mean"]
    optimism={a:{"strict_post_m3":reuse[a]["r2_mean"]-independent[a]["r2_mean"],"official_zero_prefix":reuse[a]["r2_mean"]-independent_zero[a]["r2_mean"]} for a in ARMS}
    body={"schema":f"{SCHEMA}_metrics","status":"COMPLETE_EVALUATION_ONLY_DIAGNOSTICS","heldout_official_grouping":{"arms":d1,"delta_c1_minus_t0":d1_delta,"sessions":7},"heldin_reuse_optimism":{"same_m3_calibration_trials":reuse,"strict_post_m3_independent":independent,"official_zero_prefix_independent":independent_zero,"optimism_reuse_minus_independent":optimism},"training":False,"checkpoint_selection":False,"optimizer_steps":0,"backward_steps":0,"model_updates":0,"evalai_submissions":0}
    publish_json(result_root/"metrics.json",body); return body

def verify_terminal(result_root:Path)->dict[str,Any]:
    metrics=json.loads((result_root/"metrics.json").read_text()); msha=verify_sidecar(result_root/"metrics.json")
    d=metrics["heldout_official_grouping"]["delta_c1_minus_t0"]
    opt=metrics["heldin_reuse_optimism"]["optimism_reuse_minus_independent"]
    recommend=bool(abs(opt["t0"]["strict_post_m3"])>=0.01 or abs(opt["c1"]["strict_post_m3"])>=0.01)
    body={"schema":SCHEMA,"status":"COMPLETE_H1_CAL_AUG_M3_DIAGNOSTICS_V1","finished_at_utc":utc_now(),"metrics_sha256":msha,"heldout_grouped_delta_c1_minus_t0":d,"reuse_optimism_strict":{a:opt[a]["strict_post_m3"] for a in ARMS},"m3_aware_v2_scientifically_motivated":recommend,"v2_started":False,"training":False,"checkpoint_selection":False,"optimizer_steps":0,"backward_steps":0,"model_updates":0,"evalai_submissions":0}
    tsha=publish_json(result_root/"terminal.json",body)
    publish_text(result_root/"EXPERIMENT_RECORD.md",f"# H1 CAL-AUG M3 Diagnostics V1\n\n- Status: `{body['status']}`\n- Held-out grouped delta C1-T0: `{d:+.9f}`\n- Strict reuse optimism T0/C1: `{opt['t0']['strict_post_m3']:+.9f}` / `{opt['c1']['strict_post_m3']:+.9f}`\n- M3-aware V2 scientifically motivated: `{str(recommend).lower()}`; V2 started: `false`.\n- Training/model updates/EvalAI submissions: `false` / `0` / `0`.\n\nTerminal SHA-256: `{tsha}`\n")
    return body
