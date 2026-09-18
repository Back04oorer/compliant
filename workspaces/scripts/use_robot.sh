#!/usr/bin/env bash

if [[ -n "${ZSH_VERSION:-}" ]]; then
  # shellcheck disable=SC2296
  _THIS_FILE="${(%):-%N}"
elif [[ -n "${BASH_VERSION:-}" ]]; then
  _THIS_FILE="${BASH_SOURCE[0]}"
else
  _THIS_FILE="$0"
fi

if command -v readlink >/dev/null 2>&1; then
  _RESOLVED_THIS_FILE="$(readlink -f "${_THIS_FILE}" 2>/dev/null || true)"
  if [[ -n "${_RESOLVED_THIS_FILE}" ]]; then
    _THIS_FILE="${_RESOLVED_THIS_FILE}"
  fi
  unset _RESOLVED_THIS_FILE
fi

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

strip_path_entries() {
  local var_name="$1"
  local root="$2"
  local value filtered entry changed=0

  value="${!var_name:-}"
  [[ -z "${value}" ]] && return 0

  filtered=""
  IFS=':' read -r -a _entries <<< "${value}"
  for entry in "${_entries[@]}"; do
    [[ -z "${entry}" ]] && continue
    case "${entry}" in
      "${root}"|${root}/*)
        changed=1
        ;;
      *)
        filtered="${filtered:+${filtered}:}${entry}"
        ;;
    esac
  done

  if [[ "${changed}" -eq 1 ]]; then
    export "${var_name}=${filtered}"
  fi
}

unset_var_under_root() {
  local var_name="$1"
  local root="$2"
  local value="${!var_name:-}"

  [[ -z "${value}" ]] && return 0
  case "${value}" in
    "${root}"|${root}/*) unset "${var_name}" ;;
  esac
}

strip_conda_runtime() {
  local conda_root="" candidate var

  if [[ -n "${CONDA_EXE:-}" ]]; then
    conda_root="$(cd "$(dirname "${CONDA_EXE}")/.." 2>/dev/null && pwd -P)"
  elif [[ -n "${CONDA_PYTHON_EXE:-}" ]]; then
    conda_root="$(cd "$(dirname "${CONDA_PYTHON_EXE}")/.." 2>/dev/null && pwd -P)"
  fi

  if [[ -z "${conda_root}" ]]; then
    for candidate in "${HOME}/miniconda3" "${HOME}/anaconda3"; do
      if [[ -d "${candidate}" ]]; then
        conda_root="${candidate}"
        break
      fi
    done
  fi

  [[ -z "${conda_root}" ]] && return 0

  # ROS GUIs need the system GL/X11 stack, not Conda's runtime libraries.
  for var in PATH LD_LIBRARY_PATH PYTHONPATH; do
    strip_path_entries "${var}" "${conda_root}"
  done
  for var in PYTHONHOME QT_PLUGIN_PATH QML2_IMPORT_PATH QT_QPA_PLATFORM_PLUGIN_PATH; do
    unset_var_under_root "${var}" "${conda_root}"
  done
}

_source_ros_setup() {
  export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
  if [[ -n "${ROS_SETUP_BASH:-}" ]] && [[ -f "${ROS_SETUP_BASH}" ]]; then
    source "${ROS_SETUP_BASH}"
    _ensure_ros_python_site_packages
    return 0
  fi
  if [[ -f /opt/ros/noetic/setup.bash ]]; then
    source /opt/ros/noetic/setup.bash
    _ensure_ros_python_site_packages
    return 0
  fi
  if [[ -f /opt/ros/melodic/setup.bash ]]; then
    source /opt/ros/melodic/setup.bash
    _ensure_ros_python_site_packages
    return 0
  fi
  echo "[use_robot] ROS setup not found under /opt/ros/{noetic,melodic}." >&2
  return 1
}

_ensure_ros_python_site_packages() {
  local ros_distro="${ROS_DISTRO:-}"
  local py_ver site_path
  [[ -n "${ros_distro}" ]] || return 0
  py_ver="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
  [[ -n "${py_ver}" ]] || return 0
  site_path="/opt/ros/${ros_distro}/lib/python${py_ver}/site-packages"
  [[ -d "${site_path}" ]] || return 0
  case ":${PYTHONPATH:-}:" in
    *:"${site_path}":*) ;;
    *) export PYTHONPATH="${site_path}${PYTHONPATH:+:${PYTHONPATH}}" ;;
  esac
}

_workspace_setup_matches_current_path() {
  local ws_dir="$1"
  local setup_sh="${ws_dir}/devel/setup.sh"
  [[ -f "${setup_sh}" ]] || return 1
  grep -Fq "${ws_dir}/devel" "${setup_sh}"
}

_SCRIPT_DIR="$(cd "$(dirname "${_THIS_FILE}")" && pwd)"
_WS_DIR="$(cd "${_SCRIPT_DIR}/.." && pwd)/ws_robot"
_ROOT_DIR="$(cd "${_SCRIPT_DIR}/../.." && pwd)"

strip_workspace_overlay "${_ROOT_DIR}"
strip_conda_runtime
_source_ros_setup || { return 1 2>/dev/null || exit 1; }

if [[ -f "${_WS_DIR}/devel/setup.bash" ]]; then
  if ! _workspace_setup_matches_current_path "${_WS_DIR}"; then
    echo "[use_robot] Workspace appears to have moved since it was last built: ${_WS_DIR}" >&2
    echo "[use_robot] Detected stale Catkin setup files under ${_WS_DIR}/devel" >&2
    echo "[use_robot] Run: ${_SCRIPT_DIR}/build_robot_ws.sh" >&2
    { return 1 2>/dev/null || exit 1; }
  fi
  source "${_WS_DIR}/devel/setup.bash"
  _ensure_ros_python_site_packages
else
  echo "[use_robot] Workspace not built yet: ${_WS_DIR}" >&2
  echo "[use_robot] Run: ${_SCRIPT_DIR}/build_robot_ws.sh" >&2
  { return 1 2>/dev/null || exit 1; }
fi

export ZENO_ROLE_WORKSPACE=robot
echo "[use_robot] Active workspace: ${_WS_DIR}"
