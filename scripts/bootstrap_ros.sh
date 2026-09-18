#!/usr/bin/env bash
# Creates only a new build workspace in THIS checkout; no ROS nodes or hardware.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/workspaces/ws_robot/src"
ZENO="${ROOT}/src/zeno-wholebody-intervation/zeno-wholebody-robot"
mkdir -p "${DEST}"
for rel in common/piper_ctrl teleop_side/teleop_setup robot_side/robot_setup bridge robot_side/cam_calibration \
           common/dm_drivers/dm_common common/dm_drivers/dm_hw common/dm_drivers/dm_controllers; do
  name="${rel##*/}"
  source_path="${ZENO}/${rel}"
  if [[ -e "${DEST}/${name}" || -L "${DEST}/${name}" ]]; then
    [[ "$(readlink -f "${DEST}/${name}")" == "${source_path}" ]] || { echo "Refusing to replace ${DEST}/${name}" >&2; exit 1; }
  else
    ln -s "${source_path}" "${DEST}/${name}"
  fi
done
for name in piper_description piper_msgs; do
  source_path="${ROOT}/src/piper_ros/src/${name}"
  if [[ -e "${DEST}/${name}" || -L "${DEST}/${name}" ]]; then
    [[ "$(readlink -f "${DEST}/${name}")" == "${source_path}" ]] || exit 1
  else
    ln -s "${source_path}" "${DEST}/${name}"
  fi
done
echo "Workspace prepared: ${DEST}"
echo "In a ROS Noetic environment: bash workspaces/scripts/build_robot_ws.sh"
