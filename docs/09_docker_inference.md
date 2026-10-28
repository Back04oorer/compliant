# RGB ACT inference inside Docker

All RGB ACT runtime processes run in the same `piper-control` container:

| Process | Interpreter |
|---|---|
| TUI, ROS observer, ROS execution bridge | `/usr/bin/python3` (ROS Noetic, Python 3.8) |
| ACT model and test launcher | `/opt/piper-inference/bin/python` (Python 3.11) |

The launcher starts local child processes. It no longer calls `docker exec`,
copies bridge scripts into temporary directories, or loads a model on the host.
Loopback HTTP remains between the two Python environments in this container;
no host Python server or host training environment is required. A host-network
container shares loopback with the host, so these endpoints are not network-isolated
from other host processes. Execution still requires the private token and local ARM.

## Run

From the host, enter the container:

```bash
docker exec -it -w /opt/piper piper-control bash
```

Inside it, start a **new** TUI process (an already open TUI keeps its old code):

```bash
python3 scripts/teleop_tui.py --execute
```

Choose **m. Test trained RGB ACT**, press Enter to accept the latest complete
checkpoint, then choose **predictions** and **cuda**. This runs 10 CUDA predictions
and returns to the menu.
Camera feeds start automatically from the saved role mapping, reusing existing
streams. Follower state must already be running: use the shared follower drivers
(`2` → `1`) for both predictions and execution. Starting follower drivers enables the followers.
The same menu also offers `dry-run`, `robot`, and CPU inference.
The device defaults to CUDA in the TUI and all inference scripts. Press Enter
at the device prompt to use the GPU, or explicitly choose `cpu` for CPU inference.
The TUI always uses `/opt/piper-inference/bin/python` for this action;
`--training-python` applies only to data conversion and training.
The observer and camera supervisor start and stop with inference. Only cameras
started for this run are stopped on exit; reused streams remain running. Camera
roles are remembered in `outputs/tui/inference_cameras.json`; configure all three once
with the [standalone camera setup command](07_rgb_act_predictions.md#live-observations-still-predictions-only)
if no mapping or complete known live setup exists. Robot/dry-run menu
tests position to dataset episode 0 frame 0 measured pose after local ARM, then
run continuous stepwise ACT with temporal ensemble;
local ARM selects policy on the mux and stop/exit restores teleop;
see [shared mux ACT routing and recording MIT J2/J3 rules](08_rgb_act_execution.md).

A direct prediction-only test inside the container is:

```bash
cd /opt/piper
env -u PYTHONPATH -u PYTHONHOME /opt/piper-inference/bin/python scripts/test_rgb_act.py \
  --checkpoint outputs/experiment1_rgb_25_256_rgb_act_bf16/checkpoints/080000/pretrained_model \
  --mode predictions --max-predictions 10 \
  --output-dir outputs/model_tests/docker_predictions_01
```

Use a new output directory for every run. The inference venv must not replace the
system `python3` or be used to run ROS nodes. The launcher explicitly rebuilds
the ROS environment for its system-Python children.

## Install in an existing container

This installs into `/opt/piper-python` and `/opt/piper-inference` and starts no
robot processes. It does not require a container restart. Run from the repository
on the host:

```bash
docker exec piper-control /usr/bin/python3 -m pip install --target /opt/piper-bootstrap uv==0.10.9
docker cp requirements/inference.txt piper-control:/opt/piper/requirements/inference.txt
docker cp scripts/setup_inference.sh piper-control:/opt/piper/scripts/setup_inference.sh
docker exec -e PATH=/opt/piper-bootstrap/bin:/usr/local/bin:/usr/bin:/bin \
  piper-control bash /opt/piper/scripts/setup_inference.sh
for script in piper teleop_tui tui_inputs select_recording_cameras test_rgb_act predict_rgb_act execute_rgb_act rgb_act_execution rgb_act_ros_observer rgb_act_ros_executor live_act recorded_replay replay_rosbag rosbag_trajectory; do
  docker cp "scripts/$script.py" "piper-control:/opt/piper/scripts/$script.py"
done
```

Also copy the updated launch file when migrating an older container:

```bash
docker cp src/zeno-wholebody-intervation/zeno-wholebody-robot/common/piper_ctrl/launch/piper_dual_robot.launch \
  piper-control:/opt/piper/src/zeno-wholebody-intervation/zeno-wholebody-robot/common/piper_ctrl/launch/piper_dual_robot.launch
```

Reopen the TUI after updating. If old direct drivers are still running, stop them
with Ctrl-C and start `2` → `1` once; subsequent teleop/ACT/replay changes reuse
that shared launch. No workspace rebuild is needed for these Python/launch edits.

Keep `config/`, `data/`, and `outputs/` mounted at `/opt/piper/config`,
`/opt/piper/data`, and `/opt/piper/outputs` as in the installation guide.
The existing container's `config` bind mount supplies `rgb_act_execution.json`.
Installing into a container persists across stop/start; recreating it requires
the inference image below or repeating installation.

The standard checkpoint records the original host checkout in `output_dir` and
`dataset.root`. The loader maps its repository-relative `data/...` path into the
current checkout and uses that same dataset for both resize provenance and
execution-contract checks. It fails if the referenced dataset cannot be found.
For a dataset outside the original checkout, mount it at its recorded path.
The saved checkpoint files are not rewritten.

## Rebuildable image

Build the base ROS image using `docker/Dockerfile.ros`, then:

```bash
docker build -f docker/Dockerfile.inference \
  --build-arg ROS_IMAGE=piper-ros-noetic:local \
  -t piper-ros-noetic:inference .
```

Use `piper-ros-noetic:inference` with `--gpus all` in the
[documented container creation command](00_installation.md).
The extension installs Python 3.11.15, Torch 2.10.0, Torchvision 0.25.0 and LeRobot
0.4.4 in an isolated environment. The default wheels include CUDA 12.8; rebuilding
requires no CUDA build argument. It does not include the separate stereo training package.

The host needs an NVIDIA driver compatible with CUDA 12.8 and NVIDIA Container
Toolkit. An image cannot grant GPU access: keep `--gpus all` when recreating the
container. Check a fresh container without starting ROS or connecting hardware:

```bash
docker run --rm --network none --gpus all piper-ros-noetic:inference \
  /opt/piper-inference/bin/python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda").item())'
```

For an optional CPU-only image, build with
`--build-arg INFERENCE_TORCH_INDEX=https://download.pytorch.org/whl/cpu`
and use a separate tag such as `piper-ros-noetic:inference-cpu`. Explicitly select
`cpu` in the TUI or pass `--device cpu` with that image.

An existing CPU-only container cannot gain GPU devices merely by choosing `cuda`
in the TUI. GPU setup and container recreation must be done while the robot stack
is stopped; the migration installer does not restart it.

For an existing inference environment in a container already created with GPU
access, switch the installed wheels explicitly:

```bash
docker exec -e INFERENCE_TORCH_INDEX=https://download.pytorch.org/whl/cu128 \
  piper-control bash /opt/piper/scripts/setup_inference.sh
docker exec piper-control /opt/piper-inference/bin/python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print(torch.ones(1, device="cuda").item())'
```

The installer reinstalls Torch and Torchvision from the selected index even when
the CPU wheels have the same version number. Selecting `cuda` now fails explicitly
if CUDA is unavailable; it does not silently select CPU.

Inside the container, choose **m**, confirm the checkpoint, select **predictions**
and **cuda**, or run:

```bash
cd /opt/piper
env -u PYTHONPATH -u PYTHONHOME /opt/piper-inference/bin/python scripts/test_rgb_act.py \
  --checkpoint outputs/experiment1_rgb_25_256_rgb_act_bf16/checkpoints/080000/pretrained_model \
  --mode predictions --device cuda --max-predictions 3 \
  --output-dir outputs/model_tests/docker_cuda_01
```

CPU inference remains available with `--device cpu` in the CUDA environment.

Neither installing nor building the image starts a model, a ROS master, drivers,
or motion. Motion remains an explicit `robot` trial followed by local `ARM`.

## Deployment verified on 2026-10-21

`piper-control` now has GPU access and Torch `2.10.0+cu128` (CUDA 12.8) on the
RTX 5090. ROS master, all three cameras and the passive CAN state reader were
restored after the approved container switch. Three live CUDA predictions passed
after the switch; no robot commands were sent. That historical check used a
passive reader that has since been removed; current live inference uses follower
drivers for state feedback. Reopen the TUI using the command
above because the previous container's interactive sessions ended.

The recorded-observation benchmark measured a warmed GPU median of 7.4 ms
versus 182.5 ms with one CPU thread; maximum CPU/CUDA output difference was
`6.8e-5`. Full results are in
`outputs/diagnostics/docker_cuda_20261021/verification.json`.

The stopped original container is retained as
`piper-control-cpu-backup-20261021`. The running container was migrated before
CUDA became the default in the image recipe. Building a new image does not
replace this running container; future containers should use
`piper-ros-noetic:inference` and the GPU runtime flag documented above.
