"""Official-release settings; embodiment-only omissions are explicit."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = REPO/'sim/vendor/softmimic'
UPSTREAM_COMMIT = '823582557d5417c9382303ba749262afab625076'
OUTPUT = REPO/'outputs/piper_compliance/softmimic_official'
DATASET = OUTPUT/'dataset'
KEYPOINTS = ('link2', 'link3', 'link4', 'link5', 'tcp')
AGENT = {
    'seed': 0, 'num_steps_per_env': 24, 'max_iterations': 100000, 'save_interval': 500,
    'experiment_name': 'piper_softmimic_official', 'empirical_normalization': False,
    'policy': {'class_name': 'ActorCritic', 'init_noise_std': 1.,
               'actor_hidden_dims': [512,512,256,128], 'critic_hidden_dims': [512,512,512,512], 'activation':'elu'},
    'algorithm': {'class_name':'PPO','value_loss_coef':1.,'use_clipped_value_loss':True,
        'clip_param':.2,'entropy_coef':.002,'num_learning_epochs':5,'num_mini_batches':4,
        'learning_rate':1e-3,'schedule':'adaptive','gamma':.99,'lam':.95,'desired_kl':.01,'max_grad_norm':1.},
    'logger':'tensorboard', 'clip_actions': None,
}
ADAPTATIONS = {
    'robot': 'Fixed-base Piper, six arm joints; two fingers fixed at URDF zero; current control URDF joint frames/limits and vendor inertias',
    'reference': 'One fixed pose instead of a G1 motion clip; collision-emulator produces no moving-reference collisions at this pose',
    'unavailable_terms': ['stance-foot placement/sliding/contact schedule','CoP-aware CoM stabilization','floating-base orientation/velocity rewards','torso payload/CoM randomization'],
    'keypoints': list(KEYPOINTS),
    'actuators': 'Piper joint PD gains/torque limits; official action scale 0.5 rad; no ideal gravity/bias compensation',
    'ik': 'Official Mink task weights, DAQP, event/rejection/release logic; remove foot/CoM objectives and map G1 torso fallback to Piper tcp',
    'feasibility': 'Official bounds and shrink/reject logic; additionally enforce Piper joint command and static motor-torque limits',
    'simulation_compatibility': 'Isaac Sim 5.1.0.0 / IsaacLab 2.3.1 for RTX 5090, replacing upstream 4.5 / 2.1; RSL-RL 3.0.1 API compatibility',
    'quaternion_compatibility': 'Restore IsaacLab 2.1 quat_rotate keypoint broadcasting when calling unchanged upstream reward/termination functions',
    'forcefield_execution': 'Release force/torque equations evaluated with fixed-shape GPU masks instead of host branches/dynamic active-row indexing; original kernels retained as verification oracle; no force cache; zero-tolerance boundary comparison after TorchScript warmup and exact 60-update PPO checkpoint comparison',
    'observations': 'Term-major history 3; actor 117 channels without force/augmented reference; critic 793 with force, field parameters and 20 future frames out to 1.9 s; omit feet, retain constant fixed-base channels',
    'mass_randomization': 'Map upstream non-torso mass scaling to all Piper links; omit floating torso payload/CoM offset',
    'motion_clock': 'Official CSV mapping env id modulo 120 clips; random initial frame; retain motion clock across episode resets; loop same clip at its end; force event evaluated once per 50 Hz control step',
    'code_paper_differences': ['Release K_translation minimum 10 N/m, paper 40 N/m: use release 10',
        'Release data generation 30 Hz, control 50 Hz: retain both with interpolation',
        'Release actor original-reference history, future reference on critic: retain release layout',
        'Release retains joint-pose reward (weight 2, sigma 1): retain release'],
    'not_used': ['teacher/BC/DAgger','teacher action labels or action supervision','teacher replay','Z weighting/oversampling','v1-v3 checkpoints'],
}
