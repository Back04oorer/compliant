# Quick deployment

## 1. Load and audit

Use `stereo_patch.runtime.load_stereo_policy`. It registers all local policy
classes, reads the checkpoint config, overrides external foundation-model
paths, restores the correct head-specific processors, loads safetensors, and
returns a printable contract.

```python
from stereo_patch import load_stereo_policy

loaded = load_stereo_policy(
    "YOUR_ORG/YOUR_PRIVATE_MODEL",
    dino_model="/models/dinov3-vitb16",
    defm_root="/opt/defm",
    device="cuda",
)
print(loaded.contract)
```

## 2. Build one observation

The observation is a dictionary whose keys and shapes exactly match
`loaded.policy.config.input_features`. Arrays include a leading batch
dimension. RGB arrays are `[B,3,H,W]`; metric depth is `[B,H,W]`; state is
`[B,D]`. Use rectified RGB and depth computed from the same timestamped stereo
pair. Do not normalize state, action, or RGB manually; the checkpoint processor
does that.

```python
chunk = loaded.predict_chunk(observation)  # [horizon, action_dim], physical units
```

## 3. Connect a safe executor

The repository intentionally does not guess another robot's transport or
safety semantics. The adapter must:

1. snapshot synchronized cameras and measured joint state;
2. rectify/resize using the training calibration;
3. compute FastFS metric depth with the checkpoint parameters;
4. call the policy and timestamp every predicted action in its chunk;
5. validate finite values, joint limits, velocity/acceleration, and gripper
   convention before publishing;
6. publish at the robot control rate and clear policy/history on reset.

Keep these frequencies separate:

- **new-chunk frequency**: completed calls to `predict_action_chunk`;
- **control publish frequency**: commands accepted by the robot/mux;
- **FastFS frequency**: fresh metric-depth outputs.

A 30 Hz publisher does not imply 30 policy inferences per second. ACT temporal
ensembling can consume new chunks asynchronously; DP ensembling must align
overlapping chunks by intended execution time; Flow V2 RTC additionally needs
the previous unexecuted prefix and observed inference delay.

Start with command output disabled, record observations and action chunks, then
enable conservative guards on a clear robot workspace.
