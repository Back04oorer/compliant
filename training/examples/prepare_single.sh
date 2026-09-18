#!/usr/bin/env bash
set -Eeuo pipefail

# Start from the bimanual v3 depth-mmap dataset produced by prepare_bimanual.sh.
# The output always uses the canonical right-arm/right-wrist policy slot.
python scripts/build_active_arm_lerobot_subset.py \
  --source-dataset-dir /path/to/lerobot_data/my_task_depthmmap \
  --output-root /path/to/lerobot_data \
  --dataset-id my_task_single_left \
  --episode-indices '[0,1,2,3]' \
  --active-arm left

python scripts/validate_dataset.py \
  --dataset-root /path/to/lerobot_data/my_task_single_left \
  --arm-mode single --active-arm left --variant fused
