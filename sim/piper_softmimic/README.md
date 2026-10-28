# Piper adaptation of the official SoftMimic release

This is a fixed-base, six-joint Piper adaptation of the released Hybrid deployable
training method. The earlier MuJoCo / Stable-Baselines3 PPO v1–v3 pipeline has
been removed; `sim/piper_compliance` retains only shared model/IK and dashboard utilities.
It starts PPO from random weights, without teacher actions, BC/DAgger, replay,
direction-specific rewards, force observations on the actor, or previous checkpoints.

## Pinned sources and runtime

- [SoftMimic](https://github.com/Improbable-AI/softmimic):
  `823582557d5417c9382303ba749262afab625076`, under `sim/vendor/softmimic`.
- IsaacLab v2.3.1: `5c2ec81cb17532d32f7922dd7fcaae40d123b71a`,
  under `sim/vendor/IsaacLab`.
- Isolated `.venv-softmimic`: Python 3.11.15, Isaac Sim 5.1.0.0,
  torch 2.7.0+cu128, RSL-RL 3.0.1, TensorDict 0.8.3, NumPy 1.26.0.
  The RTX 5090 runtime replaces upstream Isaac Sim 4.5 / IsaacLab 2.1 with the
  user's explicit authorization and NVIDIA Omniverse EULA acceptance.
- Offline augmentation uses the existing CPU MuJoCo environment with isolated
  `augmentation_deps` containing Mink 0.0.13, qpsolvers 4.8.1 and DAQP 0.7.2.

The complete installed package freeze is saved in
`outputs/piper_compliance/softmimic_official/environment.txt`. The runtime retains
Isaac Sim's Starlette 0.45.3 pin rather than IsaacLab's 0.49.1; optional dex/pink
retargeting and test extras are absent. These packages are not used by this task.
The managed Python executable backing this venv is under
`/tmp/piper-softmimic-python`; preserve that directory while the run is active.

## What is reused

`augmentation.py` invokes the pinned upstream force sampler, simulation runner,
event shrink/reject logic, single-step update and release trajectory logic.
Piper Mink tasks replace G1 task construction, retaining force-pose cost 5,
keypoint cost .01, posture cost 1e-4 and DAQP. Joint bounds and static motor/PD
command feasibility are additionally checked for Piper.

The generated dataset contains 40 forcefield clips, 40 collision-emulator clips
and 5 zero-wrench clips repeated 8 times, following the official Hybrid list.
It uses full six-dimensional force/torque targets and forcefield metadata,
translation/rotation stiffness variation, ramps, holds, pauses and release.
Data is sampled at the release's 30 Hz and interpolated for 50 Hz control.
The fixed reference is 20 seconds long; padding is excluded using valid lengths.

The static reference has no motion-induced collisions. Consequently collision
clips contain zero wrench. With the Piper feasibility checks, 5.825% of stored
frames carry force; this is recorded rather than changed through custom sampling.
Dataset SHA-256:
`44020fad1f22c53ffe451b52101eec2dec983958080a12e7b4e20fc70e91094c`.

`upstream.py` loads the original reward, termination and reactive force/torque
equations. Runtime force/torque helpers evaluate the same equations with fixed-shape
GPU masks; original kernels remain the oracle for equality checks. A compatibility
adapter restores IsaacLab 2.1 quaternion broadcasting.
`env.py` retains the original asymmetric observation structure: three-step,
term-major history, original references on the actor, privileged force and
augmented/current/future references on the critic. The Piper dimensions are
117 actor / 793 critic. CSV clips are assigned by environment index modulo clip
count; randomized initial frames and continuing clocks across episode resets
match the release. Clips loop without changing their assignment.

The official release's PPO/network settings are copied and checked against its
AST: 4096 environments × 24 steps, 100,000 updates (9,830,400,000 environment
steps), five epochs/four minibatches, adaptive KL .01, gamma .99, lambda .95,
entropy .002, actor [512,512,256,128], critic [512,512,512,512], ELU. Checkpoints
are saved every 500 updates. Five-update smoke runs are startup checks only.

## Embodiment and compatibility differences

- The current control URDF supplies Piper frames, limits and inertias. Fingers
  are fixed; a tiny fixed TCP body is added at the existing tool point.
- No stance-foot, CoM/CoP or floating-base objectives and randomization apply.
  Five arm/TCP keypoints replace G1 keypoints. Fixed-base observation channels
  remain constant; foot channels are omitted.
- Piper PD gains and torque limits replace G1 actuators. Action scale remains
  the release's 0.5 rad. No ideal gravity/bias torque compensation is added.
  Velocity limits retain the control URDF values [5,5,5,5,5,3] rad/s.
  PD gains and motor torque limits are simulation assumptions inherited from
  the Piper experiment; this run does not identify actual actuator dynamics.
- Isaac Sim / IsaacLab and RSL-RL API versions differ as recorded above. This is
  method alignment for Piper, not a bitwise reproduction of the G1 benchmark.
- Where paper and release differ, the pinned code wins: translation stiffness
  starts at 10 N/m, augmentation is 30 Hz, actor future references are removed,
  critic future indices extend to 1.9 s, and joint-pose reward remains enabled.

## Running and checking

The dashboard links to `/replay.html` for synchronized 3D evaluation playback.
Generate its frozen nominal 2 N / K=100 recordings with
`.venv-softmimic/bin/python -m sim.piper_softmimic.export_replay` after the 20000
versus 8000 and FirstPPo evaluation reports exist. The page uses recorded joint
angles, the control URDF and Piper STL meshes; it does not run a policy or physics
in the browser. Clean mode shows all three policies and the analytic baseline;
noisy mode omits FirstPPo because that matched evaluation is not available.
Three.js is served locally, with no CDN dependency. Tests cover transport boundaries,
record selection, playback controls and Three.js FK against recorded TCP positions.

`/soft-live.html` runs simultaneous interactive comparisons of BalancedPush
`model_19999.pt`, `model_8000.pt` and FirstPPo `model_24628.pt`. Start the worker with
`.venv-softmimic/bin/python -m sim.piper_softmimic.live --accept-eula` alongside the
dashboard. Its HTTP port 8766 binds only to localhost; dashboard routes
`/api/soft-live/{create,step,close}` proxy the fixed worker and retain same-origin
JSON checks. The nominal Isaac plant, 117D actor observations, 0.5-rad action scale,
50 Hz control and 200 Hz physics match independent evaluation. Each actor receives
its own state; a common world-frame TCP force (0–5 N) drives all three worlds.
K=100 N/m and Kr=1 remain fixed. Changing observation noise resets all worlds.

Drag a colored TCP marker to choose the force direction, with magnitude set by the
number field/slider; release the pointer or press Escape to remove force. Axis
constraints and press-and-hold ±XYZ buttons provide repeatable pushes. The service
uses one owner token, caps each request at five physics-control ticks, and pauses
without catching up when requests are absent for over 0.5 seconds. Idle force is
cleared; the owner expires after 10 seconds. Reconnect/reset starts at Q_REF. A
nonfinite state/action or 30-cm deviation stops all three worlds until reset.
It is an interactive comparison, not a replacement for the frozen evaluation
protocol. No reward, checkpoint, training dataset or robot driver is modified.

The replay force selector offers 0.5/1/2/3/5 N at fixed K=100 N/m (targets
5/10/20/30/50 mm) for ±XYZ, both clean and noisy observations. Prepare the
30-case protocol with `/tmp/piper-compliance-venv/bin/python -m
sim.piper_softmimic.rollout_examples --prepare`, then run
`.venv-softmimic/bin/python -m sim.piper_softmimic.rollout_examples --accept-eula`.
This evaluates all three frozen policies independently and exports the nominal
plant plus analytic trajectories to `replay/forces/`. Each rollout holds 3 s,
pushes during 3–6 s with 0.5-s ramps, and recovers for 6 s. Complete reports retain
light/heavy variants and failures under `evaluations/force_examples_v1/`.
The new batch changes per-world noise samples relative to previous evaluations;
it must not be presented as a repeat of the old noisy 2 N trace. To regenerate
browser JSON from completed reports without rerunning physics, use `--export`.

Run from the repository root after the pinned vendor checkouts and runtime have
been installed. Training requires explicit EULA opt-in; by default it refuses to
overwrite an existing run. Use a fresh `--output` for another experiment.

```bash
PYTHONPATH=outputs/piper_compliance/softmimic_official/augmentation_deps \
  /tmp/piper-compliance-venv/bin/python -m sim.piper_softmimic.augmentation
.venv-softmimic/bin/python -m sim.piper_softmimic.verify --accept-eula
.venv-softmimic/bin/python -m sim.piper_softmimic.train --accept-eula
```

The terminal menu now exposes this experiment as `f. Piper SoftMimic training
(simulation)`: status, resume, new or evaluate. Run `python3 scripts/teleop_tui.py --execute`
on the GPU host; `--softmimic-python` selects the isolated Isaac interpreter.
Preview mode displays the command without launching simulation.
`new` asks for a versioned CMA dataset, defaulting to the most recently completed
`dataset*` directory. Failed/rejected datasets are excluded. Training records its
absolute dataset path and SHA256; resume selects that path (legacy runs default
to `dataset`) and rejects a changed dataset.
Named formal runs such as `FirstPPo` are discovered as well as `train_*`;
smoke/preflight/TUI validation directories are excluded from TUI defaults.

For resumption, pass `--resume <run>/model_1000.pt --output <fresh-directory>` to
the trainer. `--iterations` is the total update budget including completed updates.
The trainer checks model/data/observation/PPO compatibility, restores the optimizer
and adaptive learning rate and starts after the checkpoint's completed update.
Ctrl-C/SIGTERM requests a pause at the next completed PPO update, atomically saves
its full RSL checkpoint, marks the run paused and closes Isaac. Simulator/RNG state
is initialized anew on resume.
To retain the same dashboard run and its curves, add `--resume-in-place` and use
that checkpoint's parent as `--output`. This requires the latest checkpoint of a
paused/completed run with no live trainer. Earlier metadata is saved under
`resume_history/`; existing checkpoints, source snapshots and event files remain.

Artifacts are in `outputs/piper_compliance/softmimic_official/`: dataset manifest,
GPU verification, full training `train/manifest.json`, atomic `train/progress.json`,
source snapshot/hashes, standard RSL checkpoints, final `actor.pt`, TensorBoard
events, `train.log` and process `launch.json`. The earlier adapter run discovered
  during audits is explicitly archived as `preflight_alignment_audit` and `preflight_model_audit`;
its weights are not used by formal training.

The dashboard's home page and `/softmimic.html` show training progress and curves
read directly from existing TensorBoard events: episodic return, rollout-mean
position/force errors, PPO value/surrogate losses and throughput. All six metrics
are rendered as PNG images with raw values and EMA (alpha 0.1) shown together;
click a chart to view or save it. The page checks for updates every 15 seconds,
reuses unchanged images, and keeps explanatory text collapsed. PNG snapshots
are cached in memory and Matplotlib rendering is serialized. Above 4,000 points,
the page samples evenly after smoothing;
full-resolution events remain on disk. `/api/training-curves` incrementally reads
events. The browser selector shows only `BalancedPush_v1` and `FirstPPo`, defaults
to `BalancedPush_v1`, and labels both as formal training. Smoke/preflight/legacy
records remain on disk and available to diagnostic tools. Selecting a run pins
both progress and PNG requests via
`?run=<id>`. The page saves selection as `?training_run=<id>` for sharing/reloading.
Run IDs must match discovered direct child directories; arbitrary paths are rejected.
Each curve belongs to the
displayed run; resumed runs retain global update numbers and start at their own
first recorded update. No simulator or policy is loaded for visualization.

Run the dashboard with the existing TensorBoard-equipped environment:
` .venv-softmimic/bin/python -m sim.piper_compliance.dashboard --host 100.94.104.102 --port 8765`.
Old PPO/CMA weights and their live comparison were removed; historical evaluation
files remain. The new policy still needs separate held-out force/torque and recovery
evaluation. Training curves are not evaluation scores or proof of convergence.

## Balanced fixed-pose data

`dataset_balanced_v1` changes only the force/reference distribution. It replaces
the ineffective static-pose collision clips with a stratified schedule and uses
the existing Piper Mink tasks, F/K targets and reactive-field equations. Original
`dataset` and FirstPPo checkpoints remain unchanged. This is an explicit Piper
data adaptation; it is not the unchanged upstream random event sampler.

Each 24 s clip contains 6 s hold, 12 s push, and 6 s release/recovery. Smooth force
ramp-down belongs to recovery; subsequent zero-force frames train stable return.
The three profiles use a sustained push, two peaks, or periodic modulation.
The dataset has 252 clips / 181,440 frames at 30 Hz, with 42 clips per signed axis.
Target peaks are 0.5/1/2/3 N at K=60/100/160 N/m and 5 N at K=100/160 N/m. The
83 mm 5 N / K=60 target is omitted uniformly across all axes. Every included
force/K/direction combination has all three profiles and fields of 10/100/500 N/m.
Torque targets and rotational field stiffness are zero; commanded Kr remains 1.
The gripper and nominal reference posture are unchanged.

The stage labels describe time allocation, not distinct networks or new rewards.
Measured nonzero-force occupancy is about 52.19%, including ramp-down. No empty
collision clips are included. Constant field stiffness and reference-relative
origins also preserve field equilibrium during 30-to-50 Hz metadata interpolation.
The target force peak is 5 N; a reactive field can exert more force when resisted.
Generation checks full-frame FK/orientation fidelity, joint speeds, static torque
and command limits, and 50 Hz interpolated field equilibrium. These checks use the
nominal model and do not prove feasibility under every randomized model.

```bash
PYTHONPATH=outputs/piper_compliance/softmimic_official/augmentation_deps \
  /tmp/piper-compliance-venv/bin/python -m sim.piper_softmimic.balanced_augmentation \
  --output outputs/piper_compliance/softmimic_official/dataset_balanced_new
.venv-softmimic/bin/python -m sim.piper_softmimic.verify --accept-eula \
  --dataset outputs/piper_compliance/softmimic_official/dataset_balanced_v1 \
  --output outputs/piper_compliance/softmimic_official/dataset_balanced_v1/gpu_verification.json
```

Choose `f → new → dataset_balanced_v1` in the TUI for the next experiment. The CLI
equivalent is `train --accept-eula --dataset <dataset-directory> --output <new-run>`.
Do not resume the original optimizer on changed data: resume checks the dataset
hash and retains its own data version. Data generation and GPU preflight do not
start PPO training. Reward code and PPO settings remain unchanged.

## Independent push/recovery evaluation

In the TUI, choose **f → evaluate**. Select the checkpoint (defaults to the latest
saved checkpoint of the most recently updated formal run), reference checkpoint
(defaults to `model_5500.pt` in the same run when available), the existing frozen
XYZ protocol, noise mode (`both` by default), and a fresh evaluation directory.
The two checkpoints are pinned by path and SHA256 at launch. Independent Isaac
processes run sequentially; they never restore an optimizer or stop live training.
Ctrl-C cancels this evaluation and its child process. Existing output directories
are never overwritten. Failed/interrupted outputs retain logs and `failure.json`.

The equivalent batch command is:

```bash
.venv-softmimic/bin/python -m sim.piper_softmimic.compare --accept-eula \
  --checkpoint outputs/piper_compliance/softmimic_official/FirstPPo/model_12500.pt \
  --reference-checkpoint outputs/piper_compliance/softmimic_official/FirstPPo/model_5500.pt \
  --noise both \
  --output outputs/piper_compliance/softmimic_official/evaluations/FirstPPo_comparison_new
```

Each checkpoint is re-evaluated against the analytic controller using the same
protocol, noise seed and paired plants. Comparison rejects mismatched protocols,
model/evaluator hashes, case sets or paired properties. The summary retains all
failures and missing settling times; `null` is never counted as zero time.
The top-level `report.json` contains summaries for each noise mode and all case
metrics; `summary.csv` contains all three controllers. Top-level positive/negative
XYZ PNGs compare all three controllers with clean observations when `both` is
selected (otherwise the selected mode). Per-mode subdirectories retain full
reports and traces. The analytic repeat is checked within 0.1 mm across checkpoint
runs; the maximum observed difference is recorded for the plotted mode.

Prepare analytic F/K targets with the existing MuJoCo Python; both controllers'
physical rollouts then run together in Isaac:

```bash
/tmp/piper-compliance-venv/bin/python -m sim.piper_softmimic.evaluate \
  --prepare-protocol outputs/piper_compliance/softmimic_official/evaluations/xyz_push_v1/protocol.npz
.venv-softmimic/bin/python -m sim.piper_softmimic.evaluate --accept-eula \
  --protocol outputs/piper_compliance/softmimic_official/evaluations/xyz_push_v1/protocol.npz \
  --checkpoint outputs/piper_compliance/softmimic_official/FirstPPo/model_5500.pt \
  --output outputs/piper_compliance/softmimic_official/evaluations/FirstPPo_model5500_xyz_v1
```

Add `--observation-noise` and a fresh output directory to include the training
sensor-noise model. Actor inference always uses the mean, without action sampling.
Inputs are prescribed world-frame TCP forces, not reactive training fields:
+/-XYZ, 2/5 N, K=60/100/160 N/m, zero external torque, rotational stiffness command
1 Nm/rad. The arm settles 3 s, force ramps up 0.5 s, holds 2 s, ramps down 0.5 s,
then recovers 6 s. Unreachable IK targets are documented and excluded for both
controllers. Test inputs and target positions do not sample training clips.

Each case runs in paired Isaac worlds with identical mass/inertia, damping,
armature, friction and physical limits (nominal and two model variants). The
privileged analytic baseline tracks IK(F/K) using the same joint PD interface,
with a position-command offset `(model gravity - J.T * known force) / KP`.
It observes exact external force and model state; PPO does not observe the force.
This is an informed model-based reference with more information than the actor.
Neither plant receives an ideal bias compensation force.

`report.json`, `summary.csv` and `traces.npz` retain every included case, including
failures. Settling means entering a 2/5/10 mm 3D band about the commanded reference
after full release and remaining inside through the end, for at least 0.5 s.
`null` means not observed within 6 s, not zero seconds. Return to the *pre-push*
position is reported separately so initial pose bias is not hidden. Estimated
torque-limit fractions use sampled PD torques, not hardware torque measurements.
Positive/negative XYZ PNGs show nominal 2 N, 100 N/m cases; CSVs include all loads,
stiffnesses and model variants. The dashboard exposes report/CSV/PNG files at
`/media/evaluations/<evaluation-directory>/<filename>`, keeping weights/NPZs private.

To diagnose force-field transfer and policy feedback without changing training:

```bash
.venv-softmimic/bin/python -m sim.piper_softmimic.diagnose_compliance --accept-eula \
  --checkpoints outputs/piper_compliance/softmimic_official/FirstPPo/model_5500.pt \
                outputs/piper_compliance/softmimic_official/FirstPPo/model_12500.pt \
  --output outputs/piper_compliance/softmimic_official/evaluations/compliance_diagnosis_new
```

This compares prescribed 2 N pushes, frozen commands with paired no-force controls,
and reactive fields at 30/100/500 N/m, for +/-XYZ and commanded K=60/100/160 N/m.
Nominal and midpoint joint properties are tested with observation noise off.
Frozen-command response is reported after subtracting its no-force twin's drift;
raw frozen deflection alone can mislead. Reactive fields share the desired F/K
equilibrium, but their actual forces increase when the arm resists; these are not
equal-force tests. Reports include dataset occupancy, bias-corrected errors,
unchanged upstream reward-kernel counterfactuals and positive/negative PNGs.
Counterfactual reward values exclude velocity, action-rate, limits and alive terms;
they establish target consistency and reward sensitivity, not retraining causality.

For zero-force pose-bias diagnosis, run `sim.piper_softmimic.diagnose_bias` with
`--accept-eula --checkpoints <model.pt> ... --output <fresh-directory>`. It compares
bare PD, the deterministic actor, model-only gravity offsets and a second gravity
offset added to the actor. Gravity-cancelled controls use body-COM forces only in
the diagnostic process. Matched nominal/training-midpoint/randomized properties,
20 s holds and bitwise observer parity separate model/configuration, settling and
checkpoint effects. K=100 N/m, rotational stiffness=1 Nm/rad, sensor noise off.
These interventions never modify a running training policy or its control path.

## Collection performance optimization

At the user's request, the initial formal run is paused. `model_1000.pt` retains
the policy and PPO optimizer; its SHA is in `optimization/pause.json`. The
checkpoint does not include simulator or random-generator state, so subsequent
resumption cannot reproduce every step of an uninterrupted run.

The retained optimization only caches reference interpolation until the motion
clock advances, batches the 20 future lookups, gathers term-major history with
precomputed indices and avoids copying keypoint IDs from GPU every step. Physics,
forces, rewards, observations/noise, data and PPO parameters remain unchanged.
Forcefield caching was tested and rejected because it changed rollouts.

The same checkpoint/seed and 4096 environments were run for 20 PPO updates with
both adapters. Recorded rollout values and the entire final checkpoint (including
optimizer state) were bitwise equal. After discarding the first five timing
samples, collection fell from .546 to .470 s/update and total PPO update time from
.650 to .575 s (11.5% less time, 1.13× throughput). This is a short controlled
comparison, not a guarantee of whole-run determinism or final convergence.

`benchmark.py` loads the original adapter from `train/source/env.py` for its
baseline and the current adapter for its optimized variant. It refuses existing
benchmark directories. Reports, rollout samples, rejected variants and profiler
trace are retained under `outputs/piper_compliance/softmimic_official/optimization/`.
The main training process stays stopped during these checks.

The second optimization round used a warmed-up profiler and independent 30-update
tests from the same checkpoint/seed, followed by 60-update confirmation runs.
The current adapter retains only dense force/torque calculation from this round:
it removes host `.any()` branches and dynamic active-row selection while retaining
the release's setpoint, plane, quaternion and active-mask formulas. No force values
are cached. GPU preflight compares mixed/all/zero active masks, plane and setpoint
modes against the original kernels with zero tolerance. Torque checks warm both
TorchScript specializations first: cold execution of even the unchanged oracle
showed rounding differences up to 4.8e-7 Nm. The failed cold comparisons are retained;
the passing warm check does not relax the numerical tolerance.

| Second-round candidate | Time per PPO update | Decision |
| --- | ---: | --- |
| Previous adapter, 30-update control | .566 s | Control |
| Fill held control buffers once | .566 s | No clear improvement; not retained |
| Reuse fixed tensors | .576 s | No clear improvement; not retained |
| CUDA Graph for future references | .574 s | No clear improvement; not retained |
| `torch.compile` for future references | .574 s | Numerical differences; rejected |
| Fixed-shape history reset | .574 s | No clear improvement; not retained |
| Fixed-shape force + torque calculation | .539 s | Retained; exact training and warmed boundary comparisons |

The longer paired runs measured .574 → .554 s/update (3.5% less time), with recorded
rollouts and the entire final PPO/optimizer checkpoint exactly equal after 60 updates.
A translation-only variant was also tested (.564 s, 1.8% less time) and not selected.
Full equivalence details and tested source hashes are recorded in
`optimization_round2/report.json`. This is an additional gain after the first
optimization round. Individual 1–2% timing changes
are treated as inconclusive rather than advertised as speedups. Reports, source
snapshots, rollout samples and profiler trace are under
`outputs/piper_compliance/softmimic_official/optimization_round2/`. `benchmark.py`
supports fresh `--output` paths, `--adapter-file` snapshots and isolated experimental
hooks; `compare_benchmarks.py` checks both rollouts and all checkpoint fields.
Formal training remains paused; none of these benchmark updates replace its weights.


## 30,000-update continuation comparison

The existing `continue_balanced` supervisor retains the 20,000-update baseline,
trains to 25,000 and 30,000 updates, and evaluates all three checkpoints on the
same frozen 30-case nominal protocol. Each checkpoint uses clean observations
and the same three noise seeds (68123–68125).

An independent CPU-only report task can wait for that supervisor:

```bash
.venv-softmimic/bin/python -m sim.piper_softmimic.summarize_continuation --wait
```

It writes `comparison.txt`, `comparison.csv`, and `comparison.json` under
`outputs/piper_compliance/softmimic_official/BalancedPush_v1/continuation_30000/`.
The comparison isolates ±Z at 0.5/1 N, reports achieved versus expected deflection,
tracking and recovery errors, sustained 2/5-mm recovery pass rates, and noise-seed
variation. Negative error changes mean improvement over 20,000 updates. It never
starts or changes training; an interrupted/failed supervisor produces
`comparison_failure.json` instead of a completed comparison.
