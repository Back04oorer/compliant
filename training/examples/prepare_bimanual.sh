#!/usr/bin/env bash
set -Eeuo pipefail

# 1. Convert timestamp-synchronized ROS1 bags and compute metric depth.
python scripts/convert_rosbag_to_lerobot.py \
  --session-dir /path/to/rosbag_session \
  --output-root /path/to/lerobot_data \
  --dataset-id my_task_depthmetric \
  --task "my bimanual task" \
  --fps 30 --action-source mux --convert-to-v30 \
  --obs-indices all \
  --state-indices 1-14,22-35 \
  --action-indices all \
  --image-width 640 --image-height 320 \
  --actfs --actfs-depth-output-format defm_metric \
  --actfs-calibration-json /path/to/calibration.json \
  --actfs-processing-metadata /path/to/processing_metadata.json \
  --actfs-model-dir /path/to/Fast-FoundationStereo/checkpoint.pth \
  --actfs-fast-foundationstereo-root /path/to/Fast-FoundationStereo \
  --actfs-valid-iters 8 --actfs-scale 0.5161290322580645 \
  --actfs-max-disp 64 --actfs-max-depth-m 2.5

# 2. Move depth out of Parquet into a frame-indexed mmap sidecar.
python scripts/materialize_depth_mmap.py \
  --source /path/to/lerobot_data/my_task_depthmetric \
  --output /path/to/lerobot_data/my_task_depthmmap \
  --dtype float16

# 3. Fail before training if state/action/camera/depth contracts disagree.
python scripts/validate_dataset.py \
  --dataset-root /path/to/lerobot_data/my_task_depthmmap \
  --arm-mode bimanual --variant fused
