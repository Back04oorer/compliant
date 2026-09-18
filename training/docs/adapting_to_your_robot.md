# Adapting Stereo-ACT to another robot and camera

The repository reproduces the **Stereo-ACT method and training pipeline**, not
the original task checkpoint. You should collect your own demonstrations and
replace the hardware-dependent values below. The model-side defaults can
usually remain unchanged for the first experiment.

## What must be adapted

### 1. Robot state and action definition

Determine exactly what the policy observes and predicts:

- joint order and units;
- whether the gripper is position, width, or normalized command;
- which topic is measured robot state;
- which topic is the actual action label;
- whether effort is available and meaningful;
- single active arm or bimanual control.

The reference contracts are:

| Mode | State | Action | Wrist inputs |
|---|---|---|---|
| Single | position 7 + raw effort 7 = 14 | joints/gripper 7 | one canonical wrist |
| Bimanual | left position/effort + right position/effort = 28 | left 7 + right 7 = 14 | left + right wrists |

If your arm has a different number of joints, this is not just a command-line
topic change: update the dataset validator and single-arm canonicalization
logic. ACT itself accepts different feature dimensions, but the public helper
scripts intentionally enforce the tested 7-DoF-per-arm contract.

Choose action labels carefully. `--action-source mux` uses the command after
your command multiplexer; `--action-source teleop` uses the operator/master
command. Use the stream that best represents the motion the robot actually
executed. Do not mix action sources within one dataset.

### 2. ROS topics and message timing

Replace these converter options with topics from your bags:

- `--robot-left-topic`, `--robot-right-topic`;
- `--teleop-left-topic`, `--teleop-right-topic`;
- `--mux-left-topic`, `--mux-right-topic`;
- `--top-camera-topic`;
- `--left-wrist-topic`, `--right-wrist-topic`.

All reference streams are sampled on a common timeline. Start with 30 Hz and
inspect the converter's synchronization statistics. Tune:

- `--fps`: normally your lowest reliable camera/control rate;
- `--max-sync-lag-sec`: should be strict enough to reject misaligned frames;
- `--image-use-bag-time`: use only when camera header stamps are invalid or in
  a different clock domain.

Visually verify approach, contact, gripper closure, and release against joint
labels. A model cannot repair a systematic observation/action time offset.

### 3. Stereo calibration and crop

Supply calibration for **your own physical stereo pair**:

- per-eye image width and height;
- `K_left`, `D_left`, `K_right`, `D_right`;
- stereo rectification rotations `R1`, `R2`;
- baseline/extrinsics;
- compatible rectified projection matrices `P1`, `P2`.

`--actfs-processing-metadata` defines the rectified output size, crop,
rotation, and `P1/P2`. Choose a crop that covers the whole task workspace and
does not change between data collection and deployment. The reference system
used a `1240×620` rectified crop, then stored top RGB and depth at `640×320`.
Another aspect ratio is possible, but keep RGB and depth spatially aligned.

Never copy the reference calibration file for another camera. Recalibrate
after the stereo rig moves internally. If the whole fixed camera moves relative
to the workspace, collect new demonstrations or restore the original pose.

### 4. Fast-FoundationStereo settings

Benchmark FastFS on your crop and GPU, then use identical values in conversion,
training checkpoint metadata, and online deployment:

- `--actfs-valid-iters` / `--fastfs-valid-iters`;
- `--actfs-scale` / `--fastfs-scale`;
- `--actfs-max-disp` / `--fastfs-max-disp`;
- `--actfs-max-depth-m` / `--fastfs-max-depth-m`;
- scaled FastFS width/height passed to `train.py`.

The reference values are `iters=8`, `scale=0.5161290322580645`, input
`640×320`, `max_disp=64`, and `max_depth=2.5 m`. They are deployment-speed
choices, not universal camera constants. Increase maximum disparity for closer
objects if the stereo model and GPU budget allow it. Set maximum depth beyond
the farthest task-relevant surface, while rejecting irrelevant far geometry.

Always inspect metric-depth previews. Check that the table is approximately
planar, object edges align with RGB, depth is finite in the manipulation area,
and nearer objects have physically smaller depth in meters.

### 5. Image and token resolution

The tested model defaults are:

- stored image/depth: `640×320`;
- DINO/DeFM aligned grid: `40×20` (800 top tokens);
- wrist grid: `8×8`;
- DINO patch size 16; DeFM patch size 14.

Keep these defaults first. Increasing the top grid changes model memory length,
attention cost, checkpoint shape, training speed, and GPU usage. A `60×30` grid
has 2.25× as many top tokens as `40×20`; attention matrices grow approximately
with the square of token count. Treat this as a new architecture experiment,
not a camera calibration adjustment.

The source camera resolution itself does not have to be `1240×620`. What
matters is that rectification/cropping is stable and the stored image has the
same aspect ratio as the intended model input so objects are not distorted.

### 6. Task duration and ACT chunk

The reference policy predicts 50 action steps at 30 Hz, about 1.67 seconds.
Choose chunk length in relation to the temporal scale of your task:

```text
chunk seconds = chunk_size / dataset FPS
```

Start with `chunk_size=50` at 30 Hz. Shorter chunks react faster but may amplify
replanning jitter or repeated-pose traps; longer chunks reduce replanning but
can react late to visual changes. Changing chunk size changes the output head
and requires retraining.

Training-time `n_action_steps` in this package equals `chunk_size`. Runtime
queue length, temporal ensembling, smoothing, and safety limits are deployment
choices and are not part of this training repository.

### 7. Data volume and training length

The reference starting point is batch 16, 60k optimizer steps, checkpoints at
40k/60k, learning rate `1e-5`, AMP, and seed 1000. Keep those values for a
first comparable run. Then tune based on dataset size:

- compare progress by samples seen: `steps × global_batch_size`;
- doubling global batch roughly doubles samples per step;
- save intermediate checkpoints and evaluate them; the final checkpoint is not
  automatically the best one;
- balance task configurations and avoid long accidental holds dominating the
  training distribution.

More repeated demonstrations do not replace coverage of object poses, grasp
heights, backgrounds, and failure-relevant transitions.

## What should normally stay fixed

To reproduce the same Stereo-ACT architecture, initially keep:

- DINOv3 ViT-B/16 and DeFM ViT-S/14;
- both foundation encoders frozen;
- Fused mode `cross_relbias`;
- DINO as Query and DeFM as Key/Value;
- two fusion layers, eight heads, FFN dimension 2048;
- ACT dimension 512, encoder 4 layers, decoder 7 layers;
- ResNet18 wrist encoder;
- mean/std state and action normalization;
- L1 base action loss with auxiliary velocity/acceleration/DCT losses disabled.

Changing these values is valid research, but it produces a different model
variant rather than a hardware adaptation.

## Recommended bring-up sequence

1. Record one short bag and list its topics and rates.
2. Convert one episode without committing to a full dataset.
3. Replay top/wrist videos and inspect synchronized state/action samples.
4. Inspect rectified left/right images and metric depth.
5. Convert all demonstrations and materialize the depth mmap.
6. Run `validate_dataset.py`.
7. Run one real `train.py --smoke` optimizer step.
8. Train DINO-only and DeFM-only short baselines before the full Fused run.
9. Train Fused with standard defaults and compare checkpoints.
10. Build deployment mapping and safety behavior for your robot separately.

## Adaptation checklist

- [ ] Joint and gripper order/units documented.
- [ ] State and action topics selected and stable.
- [ ] Camera topics, rates, timestamp clocks, and compression confirmed.
- [ ] Stereo calibration belongs to the current physical camera pair.
- [ ] Rectified crop covers the complete workspace.
- [ ] FastFS parameters benchmarked and depth visually validated.
- [ ] RGB/depth input aspect ratios match.
- [ ] Single/bimanual dataset contract passes validation.
- [ ] One-step Fused smoke test passes on the target machine.
- [ ] Runtime uses the same preprocessing and FastFS contract as training.
