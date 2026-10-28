"""Compare collection speed and PPO outputs against the preserved pre-optimization adapter."""
import argparse
import copy
import importlib.util
import json
import os
import time
from pathlib import Path
from .config import AGENT, OUTPUT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--accept-eula',action='store_true')
    parser.add_argument('--variant',choices=['baseline','optimized'],required=True)
    parser.add_argument('--iterations',type=int,default=20)
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--adapter-file',type=Path)
    parser.add_argument('--experiment',choices=['none','buffer-once','compile-future','graph-future'],default='none')
    args=parser.parse_args()
    if not args.accept_eula:parser.error('--accept-eula is required')
    output=args.output or OUTPUT/'optimization'/f'{args.variant}{"_profile" if args.profile else ""}'
    output.mkdir(exist_ok=False,parents=True)
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,device='cuda:0').app
    env=None
    try:
        import numpy as np
        import torch
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from .asset import prepare_urdf
        prepare_urdf()
        if args.adapter_file or args.variant=='baseline':
            path=args.adapter_file or OUTPUT/'train/source/env.py'
            spec=importlib.util.spec_from_file_location('sim.piper_softmimic.baseline_env',path)
            adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
        else:
            from . import env as adapter
        torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=False
        cfg=adapter.PiperCfg();cfg.seed=0;cfg.scene.num_envs=4096
        env=adapter.PiperEnv(cfg)
        if args.experiment=='buffer-once':
            original_pre=env._pre_physics_step;original_apply=env._apply_action
            def apply_once(actions):original_pre(actions);original_apply()
            env._pre_physics_step=apply_once;env._apply_action=lambda:None
        elif args.experiment=='compile-future':
            env.future_q=torch.compile(env.future_q,fullgraph=True,dynamic=False,mode='reduce-overhead')
        elif args.experiment=='graph-future':
            original_future=env.future_q;stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream),torch.inference_mode():
                for _ in range(3):original_future()
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.inference_mode(),torch.cuda.graph(graph,stream=stream):future_output=original_future()
            def replay_future():graph.replay();return future_output
            env.future_q=replay_future
        if args.profile:
            scopes=[(env,name) for name in ['sample','_refresh','_get_observations','_get_rewards','_get_dones','_pre_physics_step','_apply_action','_reset_idx']]
            scopes.extend([(env.sim,'step'),(env.scene,'write_data_to_sim'),(env.scene,'update')])
            for owner,name in scopes:
                original=getattr(owner,name)
                def measured(*a,_fn=original,_name=name,**kw):
                    with torch.profiler.record_function('piper/'+_name):return _fn(*a,**kw)
                setattr(owner,name,measured)
        traces={name:[] for name in ['policy','critic','reward','done','q','force','torque']}
        original_step=env.step
        step_count=0
        def recording_step(actions):
            nonlocal step_count
            result=original_step(actions);obs,reward,done,timeout,_=result
            # GPU copies only; transfer after timing so measurement does not add device barriers.
            if step_count<48:
                for key,value in [('policy',obs['policy'][:32]),('critic',obs['critic'][:32]),('reward',reward),
                    ('done',done|timeout),('force',env.held_force),('torque',env.held_torque)]:traces[key].append(value.clone())
            if step_count%24==0:traces['q'].append(env.robot.data.joint_pos.clone())
            step_count+=1
            return result
        env.step=recording_step
        agent=copy.deepcopy(AGENT);agent['obs_groups']={'policy':['policy'],'critic':['critic']}
        timings=[];profiler=None
        class Runner(OnPolicyRunner):
            def log(self,locs,*a,**kw):
                started=time.perf_counter()
                super().log(locs,*a,**kw)
                timings.append({'collection_s':locs['collection_time'],'learning_s':locs['learn_time'],'logging_s':time.perf_counter()-started})
                if profiler is not None:profiler.step()
        runner=Runner(RslRlVecEnvWrapper(env,clip_actions=None),agent,log_dir=str(output),device='cuda:0')
        runner.load(str(OUTPUT/'train/model_1000.pt'));runner.current_learning_iteration=1001
        if args.profile:
            if args.iterations<3:raise ValueError('Profiling requires >= 3 iterations for warmup')
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(wait=1,warmup=1,active=1,repeat=1)) as profiler:
                runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=True)
            (output/'profile.txt').write_text(profiler.key_averages().table(sort_by='self_cpu_time_total',row_limit=45))
            scopes={item.key:{'cpu_total_ms':item.cpu_time_total/1000,'cpu_self_ms':item.self_cpu_time_total/1000,
                'gpu_total_ms':item.device_time_total/1000,'calls':item.count} for item in profiler.key_averages()
                if item.key.startswith('piper/') and item.device_type==torch.autograd.DeviceType.CPU}
            (output/'scopes.json').write_text(json.dumps(scopes,indent=2)+'\n')
            profiler.export_chrome_trace(str(output/'trace.json'))
        else:runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=True)
        np.savez_compressed(output/'rollout.npz',**{key:torch.stack(values).cpu().numpy() for key,values in traces.items()})
        measured=timings[5:] if len(timings)>5 else timings
        report={'status':'complete','variant':args.variant,'experiment':args.experiment,'num_envs':4096,'iterations':args.iterations,
                'timings':timings,'mean_collection_s':float(np.mean([t['collection_s'] for t in measured])),
                'mean_learning_s':float(np.mean([t['learning_s'] for t in measured])),
                'mean_logging_s':float(np.mean([t['logging_s'] for t in measured]))}
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    except Exception as error:
        import traceback
        traceback.print_exc();(output/'failure.json').write_text(json.dumps({'error':repr(error)}));raise
    finally:
        if env is not None:env.close()
        app.close()


if __name__=='__main__':main()
