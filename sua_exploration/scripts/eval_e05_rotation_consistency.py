from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mc_maze.multisession_datamodule import Dandi688MultiSessionDataModule
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "streaming_calibration_exp"))
from src.models.streaming_calibration_module import StreamingCalibrationLitModule

def rotate_rows(x, angle):
 r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]],np.float32); return x @ r.T
def rotated_side(side, mean, std, angle, physical):
 out=side.copy()
 if physical:
  raw=side*std+mean; raw[:,:2]=rotate_rows(raw[:,:2],angle); out=(raw-mean)/std
 else: out[:,:2]=rotate_rows(out[:,:2],angle)
 if not np.isfinite(out).all(): raise ValueError('non-finite rotated carrier')
 return out
def sha(state):
 h=hashlib.sha256()
 for n,v in sorted(state.items()): h.update(n.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--artifact',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
 art=json.loads(a.artifact.read_text()); assert art['side_features']['group']=='t4'
 dm=Dandi688MultiSessionDataModule(data_dir=art['data_dir'],task='CO',split_counts=tuple(art['split_counts']),batch_size=32,calibration_n_trials=50,side_feature_group='t4',side_feature_pool_size=50,cache_dir='/tmp/ibci_e05_t4_cache',num_workers=0,seed=42);dm.setup('fit')
 allowed=dm.session_splits['train']+dm.session_splits['val']; assert not set(allowed)&set(dm.session_splits['test'])
 mean,std=dm._get_side_feature_stats(); bmean,bstd=dm._get_behavior_stats(); ckpt=Path(art['best_checkpoint'])
 model=StreamingCalibrationLitModule.load_from_checkpoint(str(ckpt),weights_only=False); model.eval(); student=model.student; assert student is not None; student.eval(); before=sha(student.state_dict()); dev=torch.device(a.device); student.to(dev)
 rng=np.random.RandomState(20260825); angles=np.r_[0.,rng.uniform(0,2*np.pi,32)].tolist(); rows=[]
 with torch.no_grad():
  for split in ('train','val'):
   for name,rec in getattr(dm,f'{split}_dataset').sessions.items():
    ids=np.unique(np.linspace(0,len(rec.valid_starts)-1,min(64,len(rec.valid_starts)),dtype=int)); starts=rec.valid_starts[ids]; neural=np.stack([rec.neural[s:s+50] for s in starts]); calib=np.repeat(rec.calib_trials[:50][None],len(starts),0); side=np.repeat(rec.side_features[None],len(starts),0)
    def pred(ss):
     z=[]
     for j in range(0,len(neural),32): z.append(student(torch.from_numpy(neural[j:j+32]).to(dev),calib_trials=torch.from_numpy(calib[j:j+32]).to(dev),side_features=torch.from_numpy(ss[j:j+32]).to(dev))[0].cpu().numpy()[:,-1]/5.)
     return np.concatenate(z)*bstd+bmean
    base=pred(side)
    for mode in ('physical_pipeline','normalized_internal'):
     for angle in angles:
      rot=np.stack([rotated_side(x,mean,std,angle,mode=='physical_pipeline') for x in side]); y=pred(rot); target=rotate_rows(base,angle); err=np.linalg.norm(y-target,axis=1); rows.append({'split':split,'session':name,'mode':mode,'angle':float(angle),'mean_epsilon':float(err.mean()),'rms_relative':float(np.sqrt(np.sum((y-target)**2)/max(np.sum(target**2),1e-12)))})
 after=sha(student.state_dict()); assert before==after
 payload={'schema_version':1,'protocol':'e05_rotation_consistency_t4_v1','artifact':str(a.artifact.resolve()),'checkpoint_sha256':art['best_checkpoint_sha256'],'source_normalization_sha256':art['side_features']['normalization_sha256'],'rotation_seed':20260825,'angles':angles,'window_budget_per_session':64,'allowed_sessions':allowed,'excluded_test_sessions':dm.session_splits['test'],'device':str(dev),'optimizer':False,'backward':False,'state_sha256_before':before,'state_sha256_after':after,'rows':rows}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(payload,indent=2)+'\n');print(a.out)
if __name__=='__main__':main()
