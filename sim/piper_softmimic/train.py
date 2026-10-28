"""Run the pinned SoftMimic PPO configuration from random weights on IsaacLab."""
import argparse
import copy
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

from .config import AGENT, ADAPTATIONS, DATASET, OUTPUT, UPSTREAM, UPSTREAM_COMMIT


class TrainingPaused(Exception):
    pass


def validate_resume(checkpoint,num_envs,dataset=None):
    if not checkpoint.is_file():raise ValueError('Resume checkpoint does not exist')
    manifest=json.loads((checkpoint.parent/'manifest.json').read_text())
    expected={'upstream_commit':UPSTREAM_COMMIT,'num_envs':num_envs,'physics_dt':.005,'control_dt':.02,
              'actor_observation_dim':117,'critic_observation_dim':793}
    if any(manifest.get(key)!=value for key,value in expected.items()):
        raise ValueError('Checkpoint robot/runtime observation or environment configuration does not match')
    dataset=dataset if dataset is not None else training_dataset(checkpoint)
    if manifest['dataset']['sha256']!=json.loads((dataset/'manifest.json').read_text())['sha256']:
        raise ValueError('Checkpoint augmentation dataset does not match')
    for section in ('policy','algorithm'):
        if any(manifest['agent'][section].get(key)!=value for key,value in AGENT[section].items() if key!='class_name'):
            raise ValueError('Checkpoint PPO/network configuration does not match')
    if manifest['agent']['num_steps_per_env']!=AGENT['num_steps_per_env']:
        raise ValueError('Checkpoint rollout length does not match')
    return manifest


def training_dataset(checkpoint=None,requested=None):
    if requested is not None:return requested.absolute()
    if checkpoint is not None:
        manifest=json.loads((checkpoint.parent/'manifest.json').read_text())
        return Path(manifest.get('dataset_path',DATASET)).absolute()
    return DATASET


def prepare_output(output,checkpoint=None,in_place=False):
    if not in_place:
        if output.exists():raise ValueError('Refusing to overwrite a training run; choose a fresh output directory')
        output.mkdir(parents=True)
        return None
    if checkpoint is None or checkpoint.resolve().parent!=output.resolve() or not checkpoint.is_file():
        raise ValueError('In-place resume requires a checkpoint in the output directory')
    manifest=json.loads((output/'manifest.json').read_text())
    if manifest['status'] not in ('paused','complete'):
        raise ValueError('In-place resume requires a paused or completed run')
    pid=manifest.get('pid')
    if pid:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise ValueError('Previous training process is still alive')
    latest=max(output.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
    if latest.resolve()!=checkpoint.resolve():
        raise ValueError('In-place resume requires the latest checkpoint to preserve later weights and logs')
    snapshot=output/'resume_history'/str(time.time_ns());snapshot.mkdir(parents=True)
    for name in ('manifest.json','progress.json'):
        shutil.copy2(output/name,snapshot/name)
    return snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-eula',action='store_true')
    parser.add_argument('--num-envs',type=int,default=4096)
    parser.add_argument('--iterations',type=int,default=AGENT['max_iterations'])
    parser.add_argument('--output',type=Path,default=OUTPUT/'train')
    parser.add_argument('--resume',type=Path,help='Restore a same-method PPO checkpoint and optimizer into a new run')
    parser.add_argument('--resume-in-place',action='store_true',help='Continue a paused/completed run from its latest checkpoint, retaining history and curves')
    parser.add_argument('--dataset',type=Path,help='Versioned CMA directory; resume defaults to the dataset recorded by that run')
    parser.add_argument('--headless',action='store_true',default=True)
    args=parser.parse_args()
    if not args.accept_eula:parser.error('Explicit --accept-eula is required for NVIDIA Omniverse licensing')
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
    os.environ['ACCEPT_EULA']='Y'
    dataset=training_dataset(args.resume,args.dataset)
    dataset_manifest=json.loads((dataset/'manifest.json').read_text())
    if dataset_manifest.get('status','complete')!='complete' or (dataset/'failure.json').exists():
        raise ValueError('Dataset generation is incomplete or failed')
    if hashlib.sha256((dataset/'train.npz').read_bytes()).hexdigest()!=dataset_manifest['sha256']:
        raise ValueError('Dataset contents do not match its manifest')
    reward_hash=dataset_manifest.get('reward_adapter_sha256')
    if reward_hash and reward_hash!=hashlib.sha256(Path(__file__).with_name('env.py').read_bytes()).hexdigest():
        raise ValueError('Reward/environment adapter changed since this dataset was generated')
    resume_manifest=validate_resume(args.resume,args.num_envs,dataset) if args.resume else None
    snapshot=prepare_output(args.output,args.resume,args.resume_in_place)
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=args.headless,device='cuda:0').app
    env=None
    try:
        import numpy as np
        import torch
        import rsl_rl
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from .upstream import assert_version
        from .asset import prepare_urdf
        from . import env as adapter
        adapter.DATASET=dataset
        PiperEnv,PiperCfg=adapter.PiperEnv,adapter.PiperCfg
        assert_version();model_path=prepare_urdf()
        if resume_manifest and resume_manifest['model_sha256']!=hashlib.sha256(model_path.read_bytes()).hexdigest():
            raise ValueError('Checkpoint Piper model does not match the current control URDF')
        torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=False
        cfg=PiperCfg();cfg.seed=0;cfg.scene.num_envs=args.num_envs
        env=PiperEnv(cfg)
        wrapped=RslRlVecEnvWrapper(env,clip_actions=None)
        agent=copy.deepcopy(AGENT)
        agent['obs_groups']={'policy':['policy'],'critic':['critic']}
        agent['max_iterations']=args.iterations
        adaptations=copy.deepcopy(ADAPTATIONS)
        if dataset!=DATASET:
            adaptations['dataset_sampling']=dataset_manifest.get('method','Explicit versioned CMA dataset')
            adaptations['motion_clock']=f"Original modulo mapping over {dataset_manifest['trajectories']} clips; random initial frame; retain clock across resets"
            if dataset_manifest.get('schema_version')==2:
                adaptations['ik']='Existing Piper Mink task weights and DAQP; cached F/K paths with balanced continuous force/release schedules instead of the upstream random event/rejection sampler'
                adaptations['feasibility']='Every generated frame checked for IK error, nominal static motor/command limits and reference joint speeds; generation aborts on failure'
        manifest={'status':'running','pid':os.getpid(),'upstream_commit':UPSTREAM_COMMIT,'upstream_url':'https://github.com/Improbable-AI/softmimic',
            'algorithm':'Official-release PPO from random weights; no teacher, BC, action imitation, replay, Z weighting or v1-v3 initialization',
            'adaptations':adaptations,'agent':agent,'num_envs':args.num_envs,'physics_dt':.005,'control_dt':.02,
            'actor_observation_dim':117,'critic_observation_dim':793,'dataset':dataset_manifest,'dataset_path':str(dataset),
            'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
            'model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
            'actuators':{'kp':env.robot.data.default_joint_stiffness[0].tolist(),
                'kd':env.robot.data.default_joint_damping[0].tolist(),
                'torque_limit_Nm':env.robot.data.joint_effort_limits[0].tolist(),
                'velocity_limit_rad_s':env.robot.data.joint_vel_limits[0].tolist()},
            'isaaclab_commit':subprocess.check_output(['git','-C',str(UPSTREAM.parent/'IsaacLab'),'rev-parse','HEAD'],text=True).strip(),
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
        if args.resume:
            manifest['algorithm']='Official-release PPO resumed with optimizer from a same-method checkpoint; no teacher/BC/replay or v1-v3 weights'
            manifest['resume_checkpoint']=str(args.resume.absolute())
            manifest['resume_checkpoint_sha256']=hashlib.sha256(args.resume.read_bytes()).hexdigest()
        if snapshot is not None:manifest['previous_session_metadata']=str(snapshot.absolute())
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        shutil.copy2(model_path,args.output/model_path.name)
        source=(snapshot if snapshot is not None else args.output)/'source';source.mkdir()
        for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,source/path.name)
        started=time.time();start_iteration=0;stop_requested=False
        def request_stop(*_):
            nonlocal stop_requested
            stop_requested=True
        signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
        class Runner(OnPolicyRunner):
            def log(self,locs,*a,**kw):
                super().log(locs,*a,**kw)
                iteration=int(locs.get('it',self.current_learning_iteration))+1
                elapsed=time.time()-started
                progress={'status':'running','iteration':iteration,'total_iterations':args.iterations,
                    'environment_steps':iteration*agent['num_steps_per_env']*args.num_envs,'elapsed_s':elapsed,'updated_at_unix':time.time(),
                    'estimated_remaining_s':elapsed/max(iteration-start_iteration,1)*max(0,args.iterations-iteration),
                    'metrics':{key:float(value) for key,value in env.extras.get('log',{}).items()}}
                pending=args.output/'progress.pending';pending.write_text(json.dumps(progress,indent=2,allow_nan=False)+'\n');pending.replace(args.output/'progress.json')
                if stop_requested:
                    checkpoint=args.output/f'model_{iteration-1}.pt'
                    pending=args.output/'checkpoint.pending.pt';self.save(str(pending));pending.replace(checkpoint)
                    raise TrainingPaused(str(checkpoint))
        runner=Runner(wrapped,agent,log_dir=str(args.output),device='cuda:0')
        if args.resume:
            runner.load(str(args.resume),load_optimizer=True)
            runner.alg.learning_rate=runner.alg.optimizer.param_groups[0]['lr']
            # RSL stores the index of the already-completed update in `iter`.
            runner.current_learning_iteration+=1
            start_iteration=runner.current_learning_iteration
            manifest['resumed_completed_updates']=start_iteration
            manifest['resumed_learning_rate']=runner.alg.learning_rate
            (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if start_iteration>=args.iterations:raise ValueError('Checkpoint has already reached the requested total update budget')
        runner.learn(num_learning_iterations=args.iterations-start_iteration,init_at_random_ep_len=True)
        runner.get_inference_policy(device='cpu')
        # RSL saves full PPO/optimizer checkpoints throughout training; export the actor independently.
        from isaaclab_rl.rsl_rl import export_policy_as_jit
        export_policy_as_jit(runner.alg.policy,normalizer=None,path=str(args.output),filename='actor.pt')
        manifest['status']='complete';manifest['elapsed_s']=time.time()-started
        (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        progress=json.loads((args.output/'progress.json').read_text());progress['status']='complete'
        (args.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    except TrainingPaused as paused:
        for name in ('manifest.json','progress.json'):
            path=args.output/name;record=json.loads(path.read_text())
            record.update(status='paused',preserved_checkpoint=str(paused))
            path.write_text(json.dumps(record,indent=2)+'\n')
        print(f'Training paused; policy and optimizer saved: {paused}',flush=True)
    except Exception as error:
        import traceback
        traceback.print_exc()
        (args.output/'failure.json').write_text(json.dumps({'error':repr(error)},indent=2)+'\n')
        for name in ('manifest.json','progress.json'):
            path=args.output/name
            if path.exists():
                record=json.loads(path.read_text());record.update(status='failed',error=repr(error));path.write_text(json.dumps(record,indent=2)+'\n')
        raise
    finally:
        if env is not None:env.close()
        app.close()


if __name__=='__main__':main()
