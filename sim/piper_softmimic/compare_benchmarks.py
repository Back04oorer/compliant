"""Report measured speed and exact rollout/checkpoint equality for two local runs."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def compare(baseline,candidate):
    reports=[json.loads((path/'report.json').read_text()) for path in (baseline,candidate)]
    if any(r['status']!='complete' for r in reports):raise ValueError('Both benchmarks must be complete')
    if reports[0]['iterations']!=reports[1]['iterations']:raise ValueError('Update counts differ')
    arrays=[np.load(path/'rollout.npz') for path in (baseline,candidate)]
    rollout={}
    for name in arrays[0].files:
        a,b=arrays[0][name],arrays[1][name]
        rollout[name]={'equal':bool(np.array_equal(a,b)),
            'max_abs_diff':float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))}
    def checkpoint(path):
        return max(path.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
    states=[torch.load(checkpoint(path),map_location='cpu',weights_only=False) for path in (baseline,candidate)]
    differences=[]
    def check(a,b,name='checkpoint'):
        if isinstance(a,torch.Tensor):
            if not torch.equal(a,b):differences.append(name)
        elif isinstance(a,dict):
            if a.keys()!=b.keys():differences.append(name+'.keys')
            else:
                for key in a:check(a[key],b[key],name+'.'+str(key))
        elif isinstance(a,(tuple,list)):
            if len(a)!=len(b):differences.append(name+'.length')
            else:
                for i,(x,y) in enumerate(zip(a,b)):check(x,y,name+'.'+str(i))
        elif a!=b:differences.append(name)
    check(*states)
    seconds=[r['mean_collection_s']+r['mean_learning_s'] for r in reports]
    return {'baseline':str(baseline),'candidate':str(candidate),'ppo_updates':reports[0]['iterations'],
        'rollout':rollout,'entire_checkpoint_equal':not differences,'checkpoint_differences':differences[:20],
        'bitwise_equal':not differences and all(v['equal'] for v in rollout.values()),
        'baseline_update_s':seconds[0],'candidate_update_s':seconds[1],
        'time_reduction_pct':100*(1-seconds[1]/seconds[0]),'throughput_speedup':seconds[0]/seconds[1]}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline',type=Path);parser.add_argument('candidate',type=Path)
    args=parser.parse_args();report=compare(args.baseline,args.candidate)
    (args.candidate/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
