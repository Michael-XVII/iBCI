import numpy as np
from src.h1_cal_aug_m3_diagnostics_v1 import align_trial_num_to_batched_timeline, official_grouped_metrics, session_id

def test_session_id():
    assert session_id("S6_set_2")=="S6"

def test_sets_are_pooled_before_r2():
    mapping=(("a","S6_set_1"),("b","S6_set_2"),("c","S7_set_1"),("d","S7_set_2"))
    arrays={}
    target=np.arange(28,dtype=np.float64).reshape(4,7)
    for arm in ("t0","c1"):
      for i in range(4):
        arrays[f"{arm}_{i}_target"]=target+i
        arrays[f"{arm}_{i}_prediction"]=target+i+(0 if arm=="c1" else (i+1)*.1)
        arrays[f"{arm}_{i}_eval_mask"]=np.ones(4,bool)
    out=official_grouped_metrics(arrays,mapping,"eval_mask")
    assert set(out["t0"]["per_session_r2"])=={"S6","S7"}
    assert out["c1"]["r2_mean"]==1.0
    assert out["c1"]["r2_std_population"]==0.0

def test_trial_num_right_padding_is_never_a_support_trial():
    trial=align_trial_num_to_batched_timeline(np.array([1.,1.,2.]),5)
    assert np.array_equal(trial[:3],np.array([1.,1.,2.]))
    assert np.isnan(trial[3:]).all()
    assert not np.isin(trial[3:],np.array([1.,2.])).any()
