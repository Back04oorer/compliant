#!/usr/bin/env bash
set -euo pipefail

SCRIPT_TAG="build_robot_ws"

source_ros_setup() {
  export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
  if [[ -n "${ROS_SETUP_BASH:-}" ]] && [[ -f "${ROS_SETUP_BASH}" ]]; then
    source "${ROS_SETUP_BASH}"
    return 0
  fi
  if [[ -f /opt/ros/noetic/setup.bash ]]; then
    source /opt/ros/noetic/setup.bash
    return 0
  fi
  if [[ -f /opt/ros/melodic/setup.bash ]]; then
    source /opt/ros/melodic/setup.bash
    return 0
  fi
  echo "[${SCRIPT_TAG}] ROS setup not found under /opt/ros/{noetic,melodic}." >&2
  return 1
}

strip_workspace_overlay() {
  local root="$1"
  local var value entry filtered
  for var in CMAKE_PREFIX_PATH ROS_PACKAGE_PATH; do
    value="${!var:-}"
    filtered=""
    IFS=':' read -r -a _entries <<< "${value}"
    for entry in "${_entries[@]}"; do
      [[ -z "${entry}" ]] && continue
      case "${entry}" in
        "${root}"|${root}/*) ;;
        *) filtered="${filtered:+${filtered}:}${entry}" ;;
      esac
    done
    export "${var}=${filtered}"
  done
  unset _CATKIN_SETUP_DIR CATKIN_ENV_HOOK_WORKSPACE CATKIN_SHELL
}

workspace_generated_files_are_stale() {
  local ws_dir="$1"
  local build_cache="${ws_dir}/build/CMakeCache.txt"
  local setup_sh="${ws_dir}/devel/setup.sh"

  if [[ -f "${build_cache}" ]] && ! grep -Fq "${ws_dir}/build" "${build_cache}"; then
    return 0
  fi
  if [[ -f "${setup_sh}" ]] && ! grep -Fq "${ws_dir}/devel" "${setup_sh}"; then
    return 0
  fi
  return 1
}

reset_stale_workspace_if_needed() {
  local ws_dir="$1"
  if workspace_generated_files_are_stale "${ws_dir}"; then
    echo "[${SCRIPT_TAG}] Workspace path changed; removing stale build/devel under ${ws_dir}."
    rm -rf "${ws_dir}/build" "${ws_dir}/devel"
  fi
}

workspace_has_package() {
  local pkg="$1"
  local src_dir="$2"
  if command -v rg >/dev/null 2>&1; then
    rg -n --fixed-strings "<name>${pkg}</name>" "${src_dir}" -g package.xml >/dev/null 2>&1
  else
    grep -R -n --include=package.xml "<name>${pkg}</name>" "${src_dir}" >/dev/null 2>&1
  fi
}

try_install_ros_package() {
  local deb_pkg="$1"
  if ! command -v apt-get >/dev/null 2>&1; then
    return 1
  fi
  if [[ "$(id -u)" -eq 0 ]]; then
    apt-get update
    apt-get install -y "${deb_pkg}"
    return 0
  fi
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y "${deb_pkg}"
    return 0
  fi
  return 1
}

require_ros_package() {
  local pkg="$1"
  local ws_src_dir="$2"
  local ros_distro="${ROS_DISTRO:-noetic}"
  local deb_pkg="ros-${ros_distro}-${pkg//_/-}"

  if rospack find "${pkg}" >/dev/null 2>&1; then
    return 0
  fi
  if workspace_has_package "${pkg}" "${ws_src_dir}"; then
    return 0
  fi

  if [[ "${AUTO_INSTALL_MISSING_ROS_DEPS:-0}" == "1" ]]; then
    echo "[${SCRIPT_TAG}] Missing ROS package '${pkg}', attempting apt install (${deb_pkg})."
    if try_install_ros_package "${deb_pkg}"; then
      rospack profile >/dev/null 2>&1 || true
      if rospack find "${pkg}" >/dev/null 2>&1; then
        return 0
      fi
    else
      echo "[${SCRIPT_TAG}] Auto-install requested but apt/sudo is unavailable." >&2
    fi
  fi

  cat >&2 <<EOF
[${SCRIPT_TAG}] Missing required ROS package '${pkg}'.
[${SCRIPT_TAG}] Install it in the base ROS environment, then rerun:

  sudo apt-get update
  sudo apt-get install -y ${deb_pkg}

[${SCRIPT_TAG}] If you use the Docker alias 'ros1', add '${deb_pkg}' to the
[${SCRIPT_TAG}] ros1-piper Dockerfile and rebuild the image.
[${SCRIPT_TAG}] Tip: set AUTO_INSTALL_MISSING_ROS_DEPS=1 to auto-install.
EOF
  return 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)/ws_robot"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

strip_workspace_overlay "${ROOT_DIR}"
source_ros_setup
require_ros_package serial "${WS_DIR}/src"
reset_stale_workspace_if_needed "${WS_DIR}"
cd "${WS_DIR}"

if [[ -f "${WS_DIR}/build/CMakeCache.txt" ]] && command -v catkin_make >/dev/null 2>&1; then
  # Avoid mixing catkin tools with an existing catkin_make build tree.
  catkin_make
elif command -v catkin_make >/dev/null 2>&1; then
  catkin_make
elif command -v catkin >/dev/null 2>&1; then
  catkin build
else
  echo "[${SCRIPT_TAG}] Neither catkin nor catkin_make is available in PATH." >&2
  exit 1
fi

echo "[${SCRIPT_TAG}] Build complete: ${WS_DIR}"
