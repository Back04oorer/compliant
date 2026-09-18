# Stereo-Patch

Stereo-Patch combines one shared DINO + DeFM stereo observation encoder with
three interchangeable action heads:

| Head | Policy type | Standard horizon | Runtime mode |
|---|---|---:|---|
| Stereo-ACT | `stereo_act` | 50 | chunk queue or per-frame replan + temporal ensemble |
| Stereo-DP | `stereo_diffusion` | 32 | DDIM, configurable execute length; temporal ensemble belongs in the executor |
| Stereo Flow Matching V2 | `stereo_flow_v2` | 32 | conditional flow, queue mode or asynchronous RTC |

The same code supports a canonical single active arm (`state=14`, `action=7`,
one wrist camera) and bimanual control (`state=28`, `action=14`, two wrist
cameras). Fused, DINO-only, and DeFM-only variants share all non-ablated
components.

This repository contains model, training, data-conversion, checkpoint-loading,
and robot-adapter interfaces. Datasets, ROS bags, pretrained weights, camera
calibration, Fast-FoundationStereo, DINO, and DeFM assets are intentionally not
stored in Git.

## 15-minute quick start

```bash
git clone git@github.com:pairlab-usyd/Stereo-patch.git
cd Stereo-patch

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Select the CUDA wheel compatible with your driver.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .

python scripts/check_environment.py \
  --dino-model /path/to/dinov3-vitb16 \
  --defm-root /path/to/defm \
  --fastfs-root /path/to/Fast-FoundationStereo \
  --fastfs-checkpoint /path/to/fastfs_checkpoint.pth \
  --calibration-json /path/to/calibration.json \
  --processing-metadata /path/to/processing_metadata.json
```

Download a private checkpoint (first run `huggingface-cli login`):

```bash
stereo-patch-download \
  --repo-id YOUR_ORG/YOUR_MODEL \
  --output-dir checkpoints/task_model
```

Verify its contract and load all weights without starting a robot:

```bash
stereo-patch-infer \
  --checkpoint checkpoints/task_model \
  --dino-model /path/to/dinov3-vitb16 \
  --defm-root /path/to/defm
```

To run one real observation, save the exact checkpoint input keys as arrays in
an NPZ file, with batch dimension included, then add
`--observation-npz observation.npz --output-npy action_chunk.npy`.

Read [quick_deployment.md](docs/quick_deployment.md) before connecting robot
commands. The loader is robot-agnostic; your adapter remains responsible for
camera rectification, FastFS depth, state ordering, safety guards, command rate,
chunk scheduling, temporal ensembling, and RTC timing.

## Architecture

The top RGB is represented by DINO patch tokens and metric stereo depth by
DeFM tokens. In fused mode each branch first performs spatial self-attention;
DINO queries then attend to DeFM keys/values with learned factorized 2-D
relative-position bias. The fused top memory, wrist RGB memory, and joint state
condition one of the three action heads above. DINO and DeFM are frozen in the
standard configuration; fusion, projections, wrist encoders, and the selected
action head train end to end.

See [policy_heads.md](docs/policy_heads.md),
[architecture_zh.md](docs/architecture_zh.md), and
[fusion_attention.md](docs/fusion_attention.md).

## Prepare data

Convert ROS bags to a LeRobot v3 dataset and materialize metric depth. A full
bimanual example is in [prepare_bimanual.sh](examples/prepare_bimanual.sh).

```bash
python scripts/convert_rosbag_to_lerobot.py \
  --session-dir /path/to/session \
  --output-root /path/to/lerobot_data \
  --dataset-id task_depthmetric --fps 30 \
  --action-source mux --convert-to-v30 \
  --obs-indices all --state-indices 1-14,22-35 --action-indices all \
  --image-width 640 --image-height 320 \
  --actfs --actfs-depth-output-format defm_metric \
  --actfs-calibration-json /path/to/calibration.json \
  --actfs-processing-metadata /path/to/processing_metadata.json \
  --actfs-model-dir /path/to/fastfs_checkpoint.pth \
  --actfs-fast-foundationstereo-root /path/to/Fast-FoundationStereo \
  --actfs-valid-iters 8 --actfs-scale 0.5161290322580645 \
  --actfs-max-disp 64 --actfs-max-depth-m 2.5

python scripts/materialize_depth_mmap.py \
  --source /path/to/lerobot_data/task_depthmetric \
  --output /path/to/lerobot_data/task_depthmmap --dtype float16
```

Single-arm training uses
`scripts/build_active_arm_lerobot_subset.py` to map either physical arm into a
canonical active-arm dataset. The inactive arm is not predicted.

## Train any action head

Always run `--smoke` first; it executes a real dataloader, forward, backward,
and optimizer step.

```bash
python scripts/train_policy.py \
  --action-head act \
  --dataset-root /path/to/task_depthmmap \
  --output-dir /path/to/outputs/task_act \
  --arm-mode bimanual --variant fused \
  --dino-model /path/to/dinov3-vitb16 --defm-root /path/to/defm \
  --smoke
```

Replace `act` with `dp` or `flow-v2`. A normal run adds, for example:

```bash
--steps 80000 --checkpoint-steps 60000,80000 --batch-size 16
```

Relevant head overrides are:

```text
ACT:      --chunk-size 50
DP:       --horizon 32 --execute 8  --ddim-steps 16
Flow V2:  --horizon 32 --execute 20 --flow-steps 10
```

`execute`, DDIM steps, RTC, and temporal ensembling are evaluation/runtime
choices where supported; do not confuse them with the 30 Hz robot command
publisher. `scripts/train.py` remains as the legacy Stereo-ACT-only launcher.

## What another lab must change

At minimum, replace:

- ROS/topic names and state/action extraction;
- state and action ordering, units, limits, and gripper convention;
- stereo calibration, rectification, image resolution, and RGB/depth alignment;
- FastFS resolution, iterations, disparity, depth clipping, and checkpoint;
- DINO/DeFM local paths and licenses;
- wrist camera keys and single/bimanual mapping;
- control-loop safety, reset, temporal ensemble/RTC, and publishing frequency.

Do not silently change the checkpoint's token grids, feature shapes, fusion
mode, or FastFS contract. Details are in
[robot_camera_adapter.md](docs/robot_camera_adapter.md) and
[checkpoint_contract.md](docs/checkpoint_contract.md).

## Reproducibility boundary

Cloning this repository is sufficient to recreate the architecture and train
it on another lab's compatible dataset. It does not guarantee our task success
rate without our demonstrations, calibration, robot dynamics, and foundation
weights. Store checkpoints in a private Hugging Face model repository and use
`configs/checkpoint_manifest.example.yaml` as a non-secret manifest template.

## License

Apache-2.0. See [NOTICE](NOTICE). External dependencies and weights retain
their upstream licenses.
