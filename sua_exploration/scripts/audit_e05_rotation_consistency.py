import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=json.loads(a.run.read_text());
if r['state_sha256_before']!=r['state_sha256_after']:raise ValueError('state changed')
if set(r['allowed_sessions'])&set(r['excluded_test_sessions']):raise ValueError('test session used')
if 0. not in r['angles']:raise ValueError('missing zero angle')
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps({'pass':True,'protocol':r['protocol'],'run':str(a.run.resolve())},indent=2)+'\n');print(a.out)
