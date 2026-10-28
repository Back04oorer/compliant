# RGB ACT predictions without robot movement

This is a predictions-only diagnostic adapter for the current three-camera,
bimanual Piper RGB ACT contract: state28 and action14. It has no actuator
publisher, robot service client, CAN access, mux switching, or execution flag.
It does not start arm drivers, cameras, recording, or training.

The separate [supervised execution adapter](08_rgb_act_execution.md) adds an
explicit, locally armed command path. These prediction-only scripts remain
unable to execute actions.

Both processes now run inside `piper-control`, with ROS Python 3.8 isolated from
`/opt/piper-inference/bin/python` (Python 3.11). See [Docker inference setup](09_docker_inference.md).

- `scripts/rgb_act_ros_observer.py`: ROS subscribers and a loopback-only HTTP server.
- `scripts/predict_rgb_act.py`: standard LeRobot ACT checkpoint loading and predictions.

Prediction defaults are CUDA, one PyTorch CPU thread, 1 prediction/second, and 10
predictions. These are diagnostic settings, not a robot control rate. Use
`--device cpu` for CPU inference. A checkpoint must already have
finished saving; this adapter does not create or reload checkpoints.

## Offline check first

The checkpoint argument is its `pretrained_model` directory, containing
`config.json`, `model.safetensors`, `policy_preprocessor.json`,
`policy_postprocessor.json`, and their referenced processor state files.
Saved normalization, image resizing, and action denormalization are restored.
For the current materialized 256px dataset, the adapter also reads
`resize_provenance.json` from the dataset root in `train_config.json` and
applies its area resize before the saved processor. Preserve this provenance
when moving the checkpoint/dataset. Repository-relative `data/` paths are relocated
using the original `output_dir` in `train_config.json`, so host paths work with
the `/opt/piper/data` mount. Missing referenced datasets fail explicitly.
Without a provenance file, only the saved checkpoint preprocessing applies.

An offline observation NPZ contains these arrays, without a batch dimension:

- `observation.state`: float32 `[28]`, in physical/raw units before normalization.
- `observation.image`: uint8 RGB `[H,W,3]` (Top).
- `observation.left_wrist_image`: uint8 RGB `[H,W,3]`.
- `observation.right_wrist_image`: uint8 RGB `[H,W,3]`.

For example, inside the container, from `/opt/piper`:

```bash
env -u PYTHONPATH -u PYTHONHOME /opt/piper-inference/bin/python scripts/predict_rgb_act.py \
  --checkpoint outputs/experiment1_rgb_25_256_rgb_act_bf16/checkpoints/080000/pretrained_model \
  --observation-npz outputs/diagnostics/rgb_act_prediction_verification_20261021/recorded_observation.npz \
  --output-dir outputs/predictions/rgb_act_offline_080000
```

The example NPZ is an ignored local verification artifact, not a bundled
dataset. To create your own from a converted dataset, use the training Python:

```python
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("experiment1_rgb_25", root="data/lerobot/experiment1_rgb_25",
                         video_backend="torchcodec")
sample = dataset[1000]
observation = {"observation.state": sample["observation.state"].numpy()}
for key in ("observation.image", "observation.left_wrist_image", "observation.right_wrist_image"):
    observation[key] = (sample[key].numpy().transpose(1, 2, 0) * 255).round().astype(np.uint8)
np.savez_compressed("observation.npz", **observation)
```

## Live observations, still predictions only

The container TUI **m. Test trained RGB ACT** asks for a checkpoint
(Enter accepts the latest complete checkpoint), mode and device. Choose
**predictions**, then **cuda** (default) or **cpu**, to run 10 predictions.
It starts and stops its own
read-only observer automatically. The managed `m` entry also starts missing
camera feeds from `outputs/tui/inference_cameras.json`, reuses existing publishers,
and stops only its own camera processes on exit. Configure the three roles with
the standalone camera command below if there is no saved mapping or complete known live camera setup. The
standalone `predict_rgb_act.py` and observer remain subscriber-only and require
externally supplied camera streams. The manual observer instructions below remain
available for standalone prediction commands.

For initial camera setup, run this command inside the ROS container, adjusting
devices to match the actual wrist/top views (use TUI **p** to identify them first):

```bash
cd /opt/piper
source workspaces/scripts/use_robot.sh
python3 scripts/start_rgb_cameras.py --camera left=/dev/video2 --camera right=/dev/video0 --camera top=/dev/video4
```

This saves the role mapping and starts the standard camera topics at the
recording settings (680x480, 30 FPS), with no recorder or arm driver. Keep its
terminal open; Ctrl-C stops only these camera processes. Close any earlier
standalone camera/selection window first so its devices are free. The **p**
preview can attach to these existing streams from another TUI terminal.

Existing ROS camera and measured-state streams must already be available.
The observer does not create them or enable motors. Use the shared follower
drivers (**2 → 1**) for state feedback and keep their terminal running. Starting
those drivers enables the followers. The predictions mode sends no robot
commands, but requires fresh measured feedback from both followers.

Missing live data is reported as HTTP 503 instead of being replaced by zeros or
stale samples. Do not replay a whole rosbag into the robot ROS master: it also
contains command topics.

For the managed path, run `python3 scripts/teleop_tui.py --execute` **inside the
container** and select **m**, confirm the checkpoint, then choose **predictions**
and **cuda** or **cpu**. No host model environment or separately started observer is needed.

For standalone diagnostics, start the observer in one container terminal:

```bash
cd /opt/piper
source workspaces/scripts/use_robot.sh
/usr/bin/python3 scripts/rgb_act_ros_observer.py
```

In another terminal in the **same container**:

```bash
cd /opt/piper
curl -sS http://127.0.0.1:8766/health
env -u PYTHONPATH -u PYTHONHOME /opt/piper-inference/bin/python scripts/predict_rgb_act.py \
  --checkpoint outputs/experiment1_rgb_25_256_rgb_act_bf16/checkpoints/080000/pretrained_model \
  --observer-url http://127.0.0.1:8766/observation \
  --output-dir outputs/predictions/rgb_act_live_080000 \
  --rate 1 --max-predictions 10
```

Use a new output directory each time. `--max-predictions 0` runs until Ctrl-C;
CUDA is the default and shares GPU resources with any active training.
Ctrl-C stops the observer/predictor only; neither process controls arm drivers.

The observer's defaults are:

| Input | ROS topic |
|---|---|
| Top | `/top_camera/image_raw/compressed` |
| Left wrist | `/fisheye_left/usb_cam_node/image_raw/compressed` |
| Right wrist | `/fisheye_right/usb_cam_node/image_raw/compressed` |
| Left measured state | `/robot/arm_left/joint_states_single` |
| Right measured state | `/robot/arm_right/joint_states_single` |

It checks JPEG data, exact joint names/order, finite positions/efforts, and
positive increasing header stamps. It selects matching samples from bounded
histories, rejecting inter-stream skew over 50ms, source/receive ages over
500ms, or inconsistent clocks. Invalid messages invalidate that stream until
valid data arrives. Topic and timing overrides are listed by `--help`.

## Inspect results

Each new output directory contains:

- `contract.json`: checkpoint, device, image/state/action shapes, ordered names,
  and any dataset area-resize step.
- `prediction_000000.npy`, etc.: finite, denormalized float32 `[50,14]` action
  chunks for the current model. Action order is left joint1..joint6/gripper,
  then right joint1..joint6/gripper; units follow the recorded action labels.
- `predictions.jsonl`: observation timestamps, input state, observation/prediction time,
  first predicted action, and first-action differences from measured positions.

State order is left position7 + left raw effort7 + right position7 + right
raw effort7, exactly matching the RGB converter. The API and offline checks
do not clip or hide anomalous predictions. Predicted chunks are saved to disk
only: they are not certified safe commands, and this adapter is not a robot
execution or safety controller.
