"""Generate fixed XYZ examples at five force levels, then export browser replays."""
import argparse
import json
from pathlib import Path

import numpy as np

from .config import OUTPUT
from .evaluate import DT, TIMING, VARIANTS, envelope
from .export_replay import extract, robot_description

FORCES = (.5, 1., 2., 3., 5.)
RUN = OUTPUT/'evaluations/force_examples_v1'
POLICIES = [('latest', 'BalancedPush · 20000', OUTPUT/'BalancedPush_v1/model_19999.pt', '#2563eb'),
            ('reference', 'BalancedPush · 8000', OUTPUT/'BalancedPush_v1/model_8000.pt', '#c77723'),
            ('first', 'FirstPPo · 24628', OUTPUT/'FirstPPo/model_24628.pt', '#9858a0')]


def prepare(path):
    from sim.piper_compliance.kinematics import PiperKinematics, Q_REF
    if path.exists():raise ValueError('Refusing to overwrite a frozen protocol')
    env=PiperKinematics()
    times=np.arange(round(TIMING['end_s']/DT)+1)*DT
    cases=[];forces=[];goals=[]
    for amplitude in FORCES:
        for axis in range(3):
            for sign in (-1,1):
                peak=np.eye(3)[axis]*amplitude*sign;cache={0.:Q_REF.copy()};errors=[]
                for fraction in np.arange(1,26)/25:
                    q=env.augmented_pose(peak*fraction,100,initial=cache[round(fraction-.04,8)])
                    cache[round(float(fraction),8)]=q
                    errors.append(float(np.linalg.norm(env.ik_data.site_xpos[env.tcp_id]-env.reference_pos-peak*fraction/100)))
                if max(errors)>1e-4:raise ValueError(f'Infeasible target: {amplitude} N, {sign} {axis}')
                cases.append(dict(axis='XYZ'[axis],sign=sign,force_N=amplitude,stiffness_N_per_m=100.,ik_max_position_error_m=max(errors)))
                forces.append(envelope(times)[:,None]*peak)
                goals.append(np.stack([cache[round(float(f),8)] for f in envelope(times)]))
    assert len(cases)==30
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,time=times,force=np.stack(forces,1),qgoal=np.stack(goals,1),reference_pos=env.reference_pos,reference_rot=env.reference_rot,qref=Q_REF)
    path.with_suffix('.json').write_text(json.dumps(dict(timing=TIMING,dt=DT,cases=cases,excluded_cases=[],variants=VARIANTS,
         training_dataset_used_for_test=False,force_mode='Prescribed world-frame TCP force, zero external torque'),indent=2)+'\n')


def evaluate_all():
    from .compare import run_evaluation, validate_pair
    baseline=None
    for mode in ('clean','noisy'):
        for key,_,checkpoint,_ in POLICIES:
            directory=RUN/(mode+'_'+key)
            if directory.exists():raise ValueError(f'Refusing to overwrite {directory}')
            report=run_evaluation(checkpoint,RUN/'protocol.npz',directory,mode=='noisy')
            if baseline and baseline['actor_observation_noise']==report['actor_observation_noise']:validate_pair(report,baseline)
            baseline=report
    export()


def export():
    output=OUTPUT/'replay/forces';output.mkdir(parents=True,exist_ok=True)
    manifest=dict(status='complete',forces_N=list(FORCES),stiffness_N_per_m=100,seconds=12,modes=['clean','noisy'],files={})
    for amplitude in FORCES:
        modes={}
        for mode in ('clean','noisy'):
            entries=[dict(label=label,color=color,**extract(RUN/(mode+'_'+key),'policy',amplitude)) for key,label,_,color in POLICIES]
            entries.append(dict(label='解析基线',color='#258369',**extract(RUN/(mode+'_latest'),'analytic',amplitude)))
            assert all(r['time']==entries[0]['time'] and r['protocol_sha256']==entries[0]['protocol_sha256'] for r in entries)
            modes[mode]=entries
        name=f'force_{amplitude:g}.json'
        payload=dict(force_N=amplitude,stiffness_N_per_m=100,robot=robot_description(OUTPUT/'piper_fixed.urdf'),modes=modes,
            note='Independently simulated recorded rollout; nominal plant; target displacement F/100; failures retained.')
        pending=output/(name+'.pending');pending.write_text(json.dumps(payload,separators=(',',':'),allow_nan=False));pending.replace(output/name)
        manifest['files'][f'{amplitude:g}']='/media/rollouts/'+name
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Force examples exported:',output,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare',action='store_true');parser.add_argument('--export',action='store_true');parser.add_argument('--accept-eula',action='store_true');args=parser.parse_args()
    if args.prepare:prepare(RUN/'protocol.npz')
    elif args.export:export()
    elif args.accept_eula:evaluate_all()
    else:parser.error('Choose --prepare, --export or --accept-eula to evaluate')


if __name__=='__main__':main()
