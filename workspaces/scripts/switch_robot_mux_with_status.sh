#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
USE_ROBOT_SH="${ROOT_DIR}/workspaces/scripts/use_robot.sh"
FOLLOW_FLAG_TOPIC="/conrft_robot/slave_follow_flag"
MODE="vla"
LEFT_TOPIC=""
RIGHT_TOPIC=""
WAIT_SEC="0.30"
SKIP_LEFT="false"
SKIP_RIGHT="false"
GLOBAL_SELECTED_TMP=""
LEFT_HOLD_CFG="<unknown>"
RIGHT_HOLD_CFG="<unknown>"

usage() {
  cat <<'EOF'
用法:
  bash workspaces/scripts/switch_robot_mux_with_status.sh
  bash workspaces/scripts/switch_robot_mux_with_status.sh vla
  bash workspaces/scripts/switch_robot_mux_with_status.sh teleop
  bash workspaces/scripts/switch_robot_mux_with_status.sh --left-topic /foo --right-topic /bar
  bash workspaces/scripts/switch_robot_mux_with_status.sh --skip-left --right-topic /robot/arm_right/robot_cmd

默认模式:
  vla

说明:
  当前脚本处理 left / right 两个 mux；默认双臂，也支持 --skip-left/--skip-right 单臂跳过。
  不会处理 opp 第三臂；如果是 tri-arm，请继续手动切对应的 opp mux。

内置模式:
  vla
    left  -> /robot/arm_left/vla_joint_cmd
    right -> /robot/arm_right/vla_joint_cmd
  teleop
    left  -> /teleop/arm_left/joint_states_single
    right -> /teleop/arm_right/joint_states_single

可选参数:
  --left-topic TOPIC     左臂 mux 目标 topic
  --right-topic TOPIC    右臂 mux 目标 topic
  --skip-left            不切左臂 mux，适合右臂单臂 policy
  --skip-right           不切右臂 mux，适合左臂单臂 policy
  --wait-sec SEC         切换后等待 selected/follow_flag 回传的秒数，默认 0.30
  -h, --help             显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    vla|teleop)
      MODE="$1"
      shift
      ;;
    --left-topic)
      LEFT_TOPIC="$2"
      shift 2
      ;;
    --right-topic)
      RIGHT_TOPIC="$2"
      shift 2
      ;;
    --wait-sec)
      WAIT_SEC="$2"
      shift 2
      ;;
    --skip-left)
      SKIP_LEFT="true"
      shift
      ;;
    --skip-right)
      SKIP_RIGHT="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[mux_switch] Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$SKIP_LEFT" != "true" && "$SKIP_RIGHT" != "true" && ( -z "$LEFT_TOPIC" || -z "$RIGHT_TOPIC" ) ]]; then
  case "$MODE" in
    vla)
      LEFT_TOPIC="${LEFT_TOPIC:-/robot/arm_left/vla_joint_cmd}"
      RIGHT_TOPIC="${RIGHT_TOPIC:-/robot/arm_right/vla_joint_cmd}"
      ;;
    teleop)
      LEFT_TOPIC="${LEFT_TOPIC:-/teleop/arm_left/joint_states_single}"
      RIGHT_TOPIC="${RIGHT_TOPIC:-/teleop/arm_right/joint_states_single}"
      ;;
    *)
      echo "[mux_switch] Unsupported mode: $MODE" >&2
      exit 2
      ;;
  esac
fi

if [[ "$SKIP_LEFT" == "true" && -z "$RIGHT_TOPIC" ]]; then
  case "$MODE" in
    vla) RIGHT_TOPIC="/robot/arm_right/vla_joint_cmd" ;;
    teleop) RIGHT_TOPIC="/teleop/arm_right/joint_states_single" ;;
  esac
fi
if [[ "$SKIP_RIGHT" == "true" && -z "$LEFT_TOPIC" ]]; then
  case "$MODE" in
    vla) LEFT_TOPIC="/robot/arm_left/vla_joint_cmd" ;;
    teleop) LEFT_TOPIC="/teleop/arm_left/joint_states_single" ;;
  esac
fi

if [[ -f "$USE_ROBOT_SH" ]]; then
  # shellcheck disable=SC1090
  source "$USE_ROBOT_SH"
fi

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "[mux_switch] Missing required command: $cmd" >&2
    exit 127
  fi
}

require_cmd rosservice
require_cmd rostopic
require_cmd timeout

pick_mux_service() {
  local arm="$1"
  local new_service="/robot/arm_${arm}/joint_cmd_mux/select"
  local old_service="/robot/arm_${arm}/joint_cmd_mux_select"
  if rosservice info "$new_service" >/dev/null 2>&1; then
    echo "$new_service"
    return
  fi
  if rosservice info "$old_service" >/dev/null 2>&1; then
    echo "$old_service"
    return
  fi
  echo ""
}

read_string_topic_once() {
  local topic="$1"
  timeout 2s rostopic echo -n 1 "$topic" 2>/dev/null | sed -n "s/^data: ['\"]\\{0,1\\}\\(.*\\)['\"]\\{0,1\\}$/\\1/p" | head -n 1
}

read_bool_topic_once() {
  local topic="$1"
  timeout 2s rostopic echo -n 1 "$topic" 2>/dev/null | sed -n "s/^data: \\(True\\|False\\|true\\|false\\)$/\\1/p" | head -n 1
}

read_hold_cfg() {
  local param_name="$1"
  rosparam get "$param_name" 2>/dev/null | tr -d '\r' | head -n 1
}

cleanup() {
  if [[ -n "$GLOBAL_SELECTED_TMP" && -f "$GLOBAL_SELECTED_TMP" ]]; then
    rm -f "$GLOBAL_SELECTED_TMP"
  fi
}

trap cleanup EXIT

start_global_selected_capture() {
  GLOBAL_SELECTED_TMP="$(mktemp)"
  (
    timeout 3s rostopic echo /mux/selected >"$GLOBAL_SELECTED_TMP" 2>/dev/null
  ) &
}

read_global_selected_for_arm() {
  local arm="$1"
  local arm_token="/arm_${arm}/"
  if [[ -z "$GLOBAL_SELECTED_TMP" || ! -f "$GLOBAL_SELECTED_TMP" ]]; then
    return 0
  fi
  python3 - "$GLOBAL_SELECTED_TMP" "$arm_token" <<'PY'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
arm_token = sys.argv[2]
pattern = re.compile(r"""^data:\s*['"]?(.*?)['"]?\s*$""")
last_match = ""

try:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        value = match.group(1)
        if arm_token in value:
            last_match = value
except FileNotFoundError:
    pass

if last_match:
    print(last_match)
PY
}

switch_arm() {
  local arm="$1"
  local topic="$2"
  local service
  local response
  local prev_topic=""
  service="$(pick_mux_service "$arm")"
  if [[ -z "$service" ]]; then
    echo "[mux_switch] 未找到 ${arm} 臂 mux 切换服务。" >&2
    exit 1
  fi
  echo "[mux_switch] ${arm} 臂准备切换 mux:"
  echo "[mux_switch]   service = ${service}"
  echo "[mux_switch]   target  = ${topic}"
  response="$(rosservice call "$service" "$topic" 2>&1)" || {
    echo "[mux_switch] ${arm} 臂切换失败:"
    echo "$response"
    exit 1
  }
  echo "[mux_switch] ${arm} 臂 service 返回:"
  echo "$response"
  prev_topic="$(printf '%s\n' "$response" | sed -n 's/^prev_topic: "\(.*\)"$/\1/p' | head -n 1)"
  if [[ -n "$prev_topic" ]]; then
    if [[ "$prev_topic" == "$topic" ]]; then
      echo "[mux_switch] ${arm} 臂本次没有发生实际切换：之前就已经是 ${topic}"
    else
      echo "[mux_switch] ${arm} 臂已从 ${prev_topic} 切到 ${topic}"
    fi
  fi
}

classify_mode_text() {
  local left_selected="$1"
  local right_selected="$2"
  local follow_flag="$3"
  local left_hold_cfg="$4"
  local right_hold_cfg="$5"
  if [[ "$left_selected" == "/robot/arm_left/vla_joint_cmd" && "$right_selected" == "/robot/arm_right/vla_joint_cmd" ]]; then
    if [[ "$follow_flag" == "False" || "$follow_flag" == "false" ]]; then
      if [[ "$left_hold_cfg" == "true" || "$right_hold_cfg" == "true" ]]; then
        echo "robot 已切到 VLA，teleop 已与 robot live feedback 解耦；当前 hold 配置仍可能让主手保持在当前位置，但这不是物理硬锁。"
      else
        echo "robot 已切到 VLA，teleop 已与 robot live feedback 解耦；当前默认不会额外冻结主手，也不会物理锁住 teleop 主手。"
      fi
      return
    fi
    if [[ "$follow_flag" == "True" || "$follow_flag" == "true" ]]; then
      echo "robot 已切到 VLA，但 teleop 当前还是跟随锁定；主手会朝对齐目标移动，不会自由。"
      return
    fi
    echo "robot 已切到 VLA；teleop 是否冻结还要看 slave_follow_flag 当前值。"
    return
  fi
  if [[ "$left_selected" == "/teleop/arm_left/joint_states_single" && "$right_selected" == "/teleop/arm_right/joint_states_single" ]]; then
    if [[ "$follow_flag" == "False" || "$follow_flag" == "false" ]]; then
      echo "robot 已切回 teleop 原始输入，teleop 预期是自由模式。"
      return
    fi
    if [[ "$follow_flag" == "True" || "$follow_flag" == "true" ]]; then
      echo "robot 已切回 teleop 原始输入，但 teleop 仍在跟随锁定。"
      return
    fi
    echo "robot 已切回 teleop 原始输入；teleop 是否自由还要看 slave_follow_flag。"
    return
  fi
  echo "左右臂 mux 当前不是同一种输入，建议继续检查 selected 话题和 follow_flag。"
}

start_global_selected_capture

if [[ "$SKIP_LEFT" == "true" ]]; then
  echo "[mux_switch] left 臂跳过切换。"
else
  switch_arm left "$LEFT_TOPIC"
fi
if [[ "$SKIP_RIGHT" == "true" ]]; then
  echo "[mux_switch] right 臂跳过切换。"
else
  switch_arm right "$RIGHT_TOPIC"
fi

sleep "$WAIT_SEC"

LEFT_SELECTED="$(read_string_topic_once /robot/arm_left/joint_cmd_mux/selected || true)"
RIGHT_SELECTED="$(read_string_topic_once /robot/arm_right/joint_cmd_mux/selected || true)"
FOLLOW_FLAG="$(read_bool_topic_once "$FOLLOW_FLAG_TOPIC" || true)"
LEFT_HOLD_CFG="$(read_hold_cfg /piper_teleop_left_node/robot_feedback_gate/hold_position_when_inactive || true)"
RIGHT_HOLD_CFG="$(read_hold_cfg /piper_teleop_right_node/robot_feedback_gate/hold_position_when_inactive || true)"

if [[ -z "$LEFT_SELECTED" ]]; then
  LEFT_SELECTED="$(read_global_selected_for_arm left || true)"
fi
if [[ -z "$RIGHT_SELECTED" ]]; then
  RIGHT_SELECTED="$(read_global_selected_for_arm right || true)"
fi

LEFT_SELECTED="${LEFT_SELECTED:-<no-message>}"
RIGHT_SELECTED="${RIGHT_SELECTED:-<no-message>}"
if [[ "$SKIP_LEFT" == "true" ]]; then
  LEFT_SELECTED="<skipped>"
fi
if [[ "$SKIP_RIGHT" == "true" ]]; then
  RIGHT_SELECTED="<skipped>"
fi
FOLLOW_FLAG="${FOLLOW_FLAG:-<unknown>}"
LEFT_HOLD_CFG="${LEFT_HOLD_CFG:-<unknown>}"
RIGHT_HOLD_CFG="${RIGHT_HOLD_CFG:-<unknown>}"

echo
echo "[mux_switch] 当前脚本支持 2arm 或单臂跳过：left / right。"
echo "[mux_switch] 切换后状态摘要"
echo "[mux_switch]   left selected  = ${LEFT_SELECTED}"
echo "[mux_switch]   right selected = ${RIGHT_SELECTED}"
echo "[mux_switch]   slave_follow_flag = ${FOLLOW_FLAG}"
echo "[mux_switch]   hold cfg left/right = ${LEFT_HOLD_CFG} / ${RIGHT_HOLD_CFG}"
echo "[mux_switch]   结论: $(classify_mode_text "$LEFT_SELECTED" "$RIGHT_SELECTED" "$FOLLOW_FLAG" "$LEFT_HOLD_CFG" "$RIGHT_HOLD_CFG")"
echo "[mux_switch]   提醒: 这里的“解耦”指不再把 robot live feedback 直接喂回 teleop 手感；不等于物理硬锁。"
echo
echo "[mux_switch] 如果这里还是 <no-message>，通常说明 selected 话题这次没有成功回传。"
echo "[mux_switch] 建议优先确认 teleop 端是否已重启，并且使用的是 /joint_cmd_mux/select 新服务。"
