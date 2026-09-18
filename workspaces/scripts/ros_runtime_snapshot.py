#!/usr/bin/env python3
"""Capture ROS runtime snapshot for reproducibility and post-failure diffing."""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import re
import subprocess
import xmlrpc.client
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_KEY_TOPICS: List[str] = [
    "/robot/arm_left/joint_states_single",
    "/robot/arm_right/joint_states_single",
    "/robot/arm_opp/joint_states_single",
    "/teleop/arm_left/joint_states_single",
    "/teleop/arm_right/joint_states_single",
    "/teleop/arm_opp/joint_states_single",
    "/robot/arm_left/vla_joint_cmd",
    "/robot/arm_right/vla_joint_cmd",
    "/robot/arm_opp/vla_joint_cmd",
    "/robot/arm_left/joint_cmd_mux",
    "/robot/arm_right/joint_cmd_mux",
    "/robot/arm_opp/joint_cmd_mux",
    "/robot/arm_left/joint_cmd_mux/selected",
    "/robot/arm_right/joint_cmd_mux/selected",
    "/robot/arm_opp/joint_cmd_mux/selected",
]

DEFAULT_SELECTED_TOPICS: List[str] = [
    "/robot/arm_left/joint_cmd_mux/selected",
    "/robot/arm_right/joint_cmd_mux/selected",
    "/robot/arm_opp/joint_cmd_mux/selected",
]

DEFAULT_KEY_SERVICES: List[str] = [
    "/robot/arm_left/joint_cmd_mux_select",
    "/robot/arm_right/joint_cmd_mux_select",
    "/robot/arm_opp/joint_cmd_mux_select",
    "/robot/arm_left/joint_cmd_mux/select",
    "/robot/arm_right/joint_cmd_mux/select",
    "/robot/arm_opp/joint_cmd_mux/select",
    "/robot/arm_left/enable_srv",
    "/robot/arm_right/enable_srv",
    "/robot/arm_opp/enable_srv",
    "/robot/arm_left/reset_srv",
    "/robot/arm_right/reset_srv",
    "/robot/arm_opp/reset_srv",
]

DEFAULT_KEY_ROSPARAMS: List[str] = [
    "/piper_ctrl_left_node/can_port",
    "/piper_ctrl_right_node/can_port",
    "/piper_ctrl_opp_node/can_port",
    "/piper_ctrl_left_node/ctrl_mode",
    "/piper_ctrl_right_node/ctrl_mode",
    "/piper_ctrl_opp_node/ctrl_mode",
    "/piper_ctrl_left_node/auto_enable",
    "/piper_ctrl_right_node/auto_enable",
    "/piper_ctrl_opp_node/auto_enable",
    "/piper_ctrl_left_node/gripper_reverse",
    "/piper_ctrl_right_node/gripper_reverse",
    "/piper_ctrl_opp_node/gripper_reverse",
    "/piper_teleop_left_node/can_port",
    "/piper_teleop_right_node/can_port",
    "/piper_teleop_opp_node/can_port",
    "/piper_teleop_left_node/gripper_reverse",
    "/piper_teleop_right_node/gripper_reverse",
    "/piper_teleop_opp_node/gripper_reverse",
    "/piper_teleop_left_node/mit/enable_tor",
    "/piper_teleop_right_node/mit/enable_tor",
    "/piper_teleop_opp_node/mit/enable_tor",
    "/piper_teleop_left_node/mit/gravity_mix_mode",
    "/piper_teleop_right_node/mit/gravity_mix_mode",
    "/piper_teleop_opp_node/mit/gravity_mix_mode",
    "/piper_teleop_left_node/mit/torque_scale",
    "/piper_teleop_right_node/mit/torque_scale",
    "/piper_teleop_opp_node/mit/torque_scale",
    "/piper_teleop_left_node/mit/torque_feedback_sign",
    "/piper_teleop_right_node/mit/torque_feedback_sign",
    "/piper_teleop_opp_node/mit/torque_feedback_sign",
    "/piper_teleop_left_node/master_slave/enable",
    "/piper_teleop_right_node/master_slave/enable",
    "/piper_teleop_opp_node/master_slave/enable",
]

DEFAULT_PARAM_NAMESPACES: List[str] = [
    "/piper_ctrl_left_node",
    "/piper_ctrl_right_node",
    "/piper_ctrl_opp_node",
    "/piper_teleop_left_node",
    "/piper_teleop_right_node",
    "/piper_teleop_opp_node",
    "/startup_zero_via_topic",
    "/realsense_left/realsense2_camera",
    "/realsense_right/realsense2_camera",
    "/realsense_top/realsense2_camera",
    "/realsense_preflight_check",
]

IMPORTANT_NODE_RE = re.compile(r"(piper_ctrl|piper_teleop|mux_arm|replay|intervention|run_act|run_gripper|hold)")


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def dedup_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _run_cmd(cmd: List[str], timeout_sec: float = 2.5) -> Dict[str, object]:
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=float(timeout_sec),
            check=False,
        )
        return {
            "ok": result.returncode == 0,
            "returncode": int(result.returncode),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
        }
    except Exception as exc:  # pragma: no cover
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
            "timed_out": False,
        }


def _ros_master_status() -> Dict[str, object]:
    uri = str(os.environ.get("ROS_MASTER_URI", "") or "").strip()
    if not uri:
        return {"ok": False, "uri": "", "message": "ROS_MASTER_URI is not set"}
    try:
        proxy = xmlrpc.client.ServerProxy(uri)
        code, msg, _ = proxy.getUri("/ros_runtime_snapshot")
        return {
            "ok": bool(code == 1),
            "uri": uri,
            "message": str(msg),
        }
    except Exception as exc:
        return {"ok": False, "uri": uri, "message": str(exc)}


def _parse_rostopic_info(text: str) -> Dict[str, object]:
    topic_type = ""
    publishers: List[str] = []
    subscribers: List[str] = []
    section = ""
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Type:"):
            topic_type = line.split(":", 1)[1].strip()
            continue
        if line.startswith("Publishers:"):
            section = "publishers"
            continue
        if line.startswith("Subscribers:"):
            section = "subscribers"
            continue
        if line.startswith("*"):
            node_text = line[1:].strip()
            node_name = node_text.split(" ", 1)[0].strip()
            if section == "publishers":
                publishers.append(node_name)
            elif section == "subscribers":
                subscribers.append(node_name)
    return {
        "type": topic_type,
        "publishers": dedup_keep_order(publishers),
        "subscribers": dedup_keep_order(subscribers),
    }


def _collect_topic_info(topic: str, timeout_sec: float) -> Dict[str, object]:
    res = _run_cmd(["rostopic", "info", topic], timeout_sec=timeout_sec)
    if not bool(res["ok"]):
        stderr = str(res.get("stderr", "")).strip()
        stdout = str(res.get("stdout", "")).strip()
        timed_out = bool(res.get("timed_out", False))
        if timed_out:
            status = "timeout"
        elif "Unknown topic" in stderr or "Unknown topic" in stdout:
            status = "missing"
        else:
            status = "error"
        return {
            "status": status,
            "type": "",
            "publishers": [],
            "subscribers": [],
            "publisher_count": 0,
            "subscriber_count": 0,
            "error": stderr or stdout,
        }

    parsed = _parse_rostopic_info(str(res.get("stdout", "")))
    publishers = list(parsed.get("publishers", []))
    subscribers = list(parsed.get("subscribers", []))
    return {
        "status": "ok",
        "type": str(parsed.get("type", "")),
        "publishers": publishers,
        "subscribers": subscribers,
        "publisher_count": int(len(publishers)),
        "subscriber_count": int(len(subscribers)),
        "error": "",
    }


def _parse_scalar_or_yaml(text: str) -> object:
    raw = str(text).strip()
    if not raw:
        return ""
    if yaml is None:
        return raw
    try:
        return yaml.safe_load(raw)
    except Exception:
        return raw


def _collect_selected_value(topic: str, timeout_sec: float) -> Dict[str, object]:
    res = _run_cmd(["rostopic", "echo", "-n", "1", topic], timeout_sec=timeout_sec)
    if not bool(res["ok"]):
        timed_out = bool(res.get("timed_out", False))
        stderr = str(res.get("stderr", "")).strip()
        stdout = str(res.get("stdout", "")).strip()
        if timed_out:
            status = "timeout"
        elif "does not appear to be published yet" in stderr:
            status = "no_messages"
        elif "Unknown topic" in stderr or "Unknown topic" in stdout:
            status = "missing"
        else:
            status = "error"
        return {
            "status": status,
            "value": None,
            "raw": stdout,
            "error": stderr,
        }

    raw = str(res.get("stdout", "")).strip()
    parsed = _parse_scalar_or_yaml(raw)
    return {
        "status": "ok",
        "value": parsed,
        "raw": raw,
        "error": "",
    }


def _collect_service_type(service_name: str, timeout_sec: float) -> Dict[str, object]:
    res = _run_cmd(["rosservice", "type", service_name], timeout_sec=timeout_sec)
    if not bool(res["ok"]):
        stderr = str(res.get("stderr", "")).strip()
        stdout = str(res.get("stdout", "")).strip()
        timed_out = bool(res.get("timed_out", False))
        if timed_out:
            status = "timeout"
        elif "Unknown service" in stderr or "Unknown service" in stdout:
            status = "missing"
        else:
            status = "error"
        return {
            "status": status,
            "type": "",
            "error": stderr or stdout,
        }
    return {
        "status": "ok",
        "type": str(res.get("stdout", "")).strip(),
        "error": "",
    }


def _collect_rosparam(key: str, timeout_sec: float) -> Dict[str, object]:
    res = _run_cmd(["rosparam", "get", key], timeout_sec=timeout_sec)
    if not bool(res["ok"]):
        stderr = str(res.get("stderr", "")).strip()
        stdout = str(res.get("stdout", "")).strip()
        timed_out = bool(res.get("timed_out", False))
        if timed_out:
            status = "timeout"
        elif "ERROR" in stderr.upper() and "not set" in stderr.lower():
            status = "missing"
        else:
            status = "error"
        return {
            "status": status,
            "value": None,
            "error": stderr or stdout,
        }
    return {
        "status": "ok",
        "value": _parse_scalar_or_yaml(str(res.get("stdout", ""))),
        "error": "",
    }


def _collect_rosparam_namespace(namespace: str, timeout_sec: float) -> Dict[str, object]:
    res = _run_cmd(["rosparam", "get", namespace], timeout_sec=timeout_sec)
    if not bool(res["ok"]):
        stderr = str(res.get("stderr", "")).strip()
        stdout = str(res.get("stdout", "")).strip()
        timed_out = bool(res.get("timed_out", False))
        if timed_out:
            status = "timeout"
        elif "ERROR" in stderr.upper() and "not set" in stderr.lower():
            status = "missing"
        else:
            status = "error"
        return {
            "status": status,
            "value": None,
            "error": stderr or stdout,
        }
    return {
        "status": "ok",
        "value": _parse_scalar_or_yaml(str(res.get("stdout", ""))),
        "error": "",
    }


def _collect_git_info(timeout_sec: float) -> Dict[str, object]:
    root_res = _run_cmd(["git", "-C", str(REPO_ROOT), "rev-parse", "--show-toplevel"], timeout_sec=timeout_sec)
    if not bool(root_res["ok"]):
        return {
            "available": False,
            "repo_root": str(REPO_ROOT),
            "branch": "",
            "commit": "",
            "dirty": None,
        }

    repo_root = str(root_res.get("stdout", "")).strip()
    branch = str(_run_cmd(["git", "-C", repo_root, "rev-parse", "--abbrev-ref", "HEAD"], timeout_sec=timeout_sec).get("stdout", "")).strip()
    commit = str(_run_cmd(["git", "-C", repo_root, "rev-parse", "HEAD"], timeout_sec=timeout_sec).get("stdout", "")).strip()
    status_res = _run_cmd(["git", "-C", repo_root, "status", "--porcelain"], timeout_sec=timeout_sec)
    dirty = None
    if status_res.get("ok"):
        dirty = bool(str(status_res.get("stdout", "")).strip())

    return {
        "available": True,
        "repo_root": repo_root,
        "branch": branch,
        "commit": commit,
        "dirty": dirty,
    }


def _collect_node_list(timeout_sec: float) -> Dict[str, object]:
    res = _run_cmd(["rosnode", "list"], timeout_sec=timeout_sec)
    if not bool(res["ok"]):
        return {
            "status": "error" if not res.get("timed_out") else "timeout",
            "nodes": [],
            "important_nodes": [],
            "error": str(res.get("stderr", "")).strip() or str(res.get("stdout", "")).strip(),
        }
    nodes = dedup_keep_order(line.strip() for line in str(res.get("stdout", "")).splitlines())
    important = [node for node in nodes if IMPORTANT_NODE_RE.search(node)]
    return {
        "status": "ok",
        "nodes": nodes,
        "important_nodes": important,
        "error": "",
    }


def _parse_rosnode_info(text: str) -> Dict[str, object]:
    section = ""
    publications: List[str] = []
    subscriptions: List[str] = []
    services: List[str] = []
    pid = None
    uri = ""
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("Node [") and line.endswith("]"):
            section = ""
            continue
        if line.startswith("Publications:"):
            section = "publications"
            continue
        if line.startswith("Subscriptions:"):
            section = "subscriptions"
            continue
        if line.startswith("Services:"):
            section = "services"
            continue
        if line.startswith("Pid:"):
            try:
                pid = int(line.split(":", 1)[1].strip())
            except Exception:
                pid = None
            continue
        if line.startswith("URI:"):
            uri = line.split(":", 1)[1].strip()
            continue
        if line.startswith("*"):
            entry = line[1:].strip()
            token = entry.split(" ", 1)[0].strip()
            if section == "publications":
                publications.append(token)
            elif section == "subscriptions":
                subscriptions.append(token)
            elif section == "services":
                services.append(token)
    return {
        "pid": pid,
        "uri": uri,
        "publications": dedup_keep_order(publications),
        "subscriptions": dedup_keep_order(subscriptions),
        "services": dedup_keep_order(services),
    }


def _collect_important_node_details(nodes: Iterable[str], timeout_sec: float) -> Dict[str, object]:
    details: Dict[str, object] = {}
    for node_name in dedup_keep_order(nodes):
        res = _run_cmd(["rosnode", "info", node_name], timeout_sec=timeout_sec)
        if not bool(res.get("ok")):
            details[node_name] = {
                "status": "timeout" if bool(res.get("timed_out", False)) else "error",
                "pid": None,
                "uri": "",
                "publications": [],
                "subscriptions": [],
                "services": [],
                "error": str(res.get("stderr", "")).strip() or str(res.get("stdout", "")).strip(),
            }
            continue
        parsed = _parse_rosnode_info(str(res.get("stdout", "")))
        details[node_name] = {
            "status": "ok",
            "pid": parsed.get("pid"),
            "uri": parsed.get("uri"),
            "publications": parsed.get("publications", []),
            "subscriptions": parsed.get("subscriptions", []),
            "services": parsed.get("services", []),
            "error": "",
        }
    return details


def _collect_can_links(timeout_sec: float) -> List[str]:
    res = _run_cmd(["ip", "-brief", "link"], timeout_sec=timeout_sec)
    if not bool(res["ok"]):
        return []
    lines = [line.strip() for line in str(res.get("stdout", "")).splitlines() if line.strip()]
    return [line for line in lines if "can_" in line or line.startswith("can")]


def _collect_roslaunch_processes(timeout_sec: float) -> Dict[str, object]:
    res = _run_cmd(["ps", "-eo", "pid,args"], timeout_sec=timeout_sec)
    if not bool(res.get("ok")):
        return {
            "status": "timeout" if bool(res.get("timed_out", False)) else "error",
            "start_robot_all": [],
            "all_roslaunch": [],
            "error": str(res.get("stderr", "")).strip() or str(res.get("stdout", "")).strip(),
        }

    start_robot_all: List[Dict[str, object]] = []
    all_roslaunch: List[Dict[str, object]] = []
    for raw in str(res.get("stdout", "")).splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("PID ") or line.startswith("PID"):
            continue
        parts = line.split(None, 1)
        if len(parts) < 2:
            continue
        pid_raw, cmdline = parts[0], parts[1].strip()
        try:
            pid = int(pid_raw)
        except Exception:
            pid = None
        if "roslaunch" not in cmdline:
            continue
        item = {"pid": pid, "cmdline": cmdline}
        all_roslaunch.append(item)
        if "start_robot_all.launch" in cmdline and "robot_setup" in cmdline:
            start_robot_all.append(item)
    return {
        "status": "ok",
        "start_robot_all": start_robot_all,
        "all_roslaunch": all_roslaunch,
        "error": "",
    }


def capture_runtime_snapshot(
    *,
    key_topics: Optional[Iterable[str]] = None,
    selected_topics: Optional[Iterable[str]] = None,
    key_services: Optional[Iterable[str]] = None,
    key_rosparams: Optional[Iterable[str]] = None,
    key_param_namespaces: Optional[Iterable[str]] = None,
    command_timeout_sec: float = 2.5,
) -> Dict[str, object]:
    topics = dedup_keep_order(key_topics if key_topics is not None else DEFAULT_KEY_TOPICS)
    selected = dedup_keep_order(selected_topics if selected_topics is not None else DEFAULT_SELECTED_TOPICS)
    services = dedup_keep_order(key_services if key_services is not None else DEFAULT_KEY_SERVICES)
    rosparams = dedup_keep_order(key_rosparams if key_rosparams is not None else DEFAULT_KEY_ROSPARAMS)
    param_namespaces = dedup_keep_order(
        key_param_namespaces if key_param_namespaces is not None else DEFAULT_PARAM_NAMESPACES
    )

    topic_list_res = _run_cmd(["rostopic", "list"], timeout_sec=command_timeout_sec)
    service_list_res = _run_cmd(["rosservice", "list"], timeout_sec=command_timeout_sec)
    topic_count = None
    service_count = None
    if topic_list_res.get("ok"):
        topic_count = len([line for line in str(topic_list_res.get("stdout", "")).splitlines() if line.strip()])
    if service_list_res.get("ok"):
        service_count = len([line for line in str(service_list_res.get("stdout", "")).splitlines() if line.strip()])

    nodes_info = _collect_node_list(timeout_sec=command_timeout_sec)
    important_node_details = {}
    if str(nodes_info.get("status", "")) == "ok":
        important_node_details = _collect_important_node_details(
            nodes_info.get("important_nodes", []),
            timeout_sec=command_timeout_sec,
        )

    snapshot = {
        "snapshot_schema_version": 1,
        "captured_at_utc": utc_now_iso(),
        "host": platform.node(),
        "user": os.environ.get("USER"),
        "cwd": str(Path.cwd()),
        "python": {
            "executable": os.environ.get("PYTHON_EXECUTABLE") or os.sys.executable,
            "version": os.sys.version.split()[0],
        },
        "system": {
            "platform": platform.platform(),
            "uname": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
        },
        "git": _collect_git_info(timeout_sec=command_timeout_sec),
        "ros": {
            "env": {
                "ROS_MASTER_URI": os.environ.get("ROS_MASTER_URI"),
                "ROS_IP": os.environ.get("ROS_IP"),
                "ROS_HOSTNAME": os.environ.get("ROS_HOSTNAME"),
                "ROS_DISTRO": os.environ.get("ROS_DISTRO"),
            },
            "master": _ros_master_status(),
            "topic_count": topic_count,
            "service_count": service_count,
            "targets": {
                "topics": topics,
                "selected_topics": selected,
                "services": services,
                "rosparams": rosparams,
                "param_namespaces": param_namespaces,
            },
            "key_topics": {},
            "selected_values": {},
            "key_services": {},
            "key_rosparams": {},
            "key_param_namespaces": {},
            "nodes": nodes_info,
            "important_node_details": important_node_details,
        },
        "net": {
            "can_links": _collect_can_links(timeout_sec=command_timeout_sec),
        },
        "processes": {
            "roslaunch": _collect_roslaunch_processes(timeout_sec=command_timeout_sec),
        },
    }

    for topic in topics:
        snapshot["ros"]["key_topics"][topic] = _collect_topic_info(topic, timeout_sec=command_timeout_sec)

    for topic in selected:
        snapshot["ros"]["selected_values"][topic] = _collect_selected_value(topic, timeout_sec=command_timeout_sec)

    for service_name in services:
        snapshot["ros"]["key_services"][service_name] = _collect_service_type(
            service_name,
            timeout_sec=command_timeout_sec,
        )

    for key in rosparams:
        snapshot["ros"]["key_rosparams"][key] = _collect_rosparam(key, timeout_sec=command_timeout_sec)

    for namespace in param_namespaces:
        snapshot["ros"]["key_param_namespaces"][namespace] = _collect_rosparam_namespace(
            namespace,
            timeout_sec=command_timeout_sec,
        )

    return snapshot


def load_reference_snapshot(path: Path) -> Dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "snapshot_schema_version" in raw:
        return raw

    if isinstance(raw, dict) and "repro_snapshot" in raw and isinstance(raw["repro_snapshot"], dict):
        return dict(raw["repro_snapshot"])

    repro_file = None
    if isinstance(raw, dict):
        repro_file = raw.get("repro_snapshot_file")
        if isinstance(repro_file, str) and repro_file.strip():
            snapshot_path = Path(repro_file).expanduser()
            if not snapshot_path.is_absolute():
                snapshot_path = (path.parent / snapshot_path).resolve()
            if snapshot_path.is_file():
                return json.loads(snapshot_path.read_text(encoding="utf-8"))

        # Backward compatibility: old metadata.json without repro_snapshot.
        # In this mode we only compare ROS params that were already stored.
        env = raw.get("environment")
        if isinstance(env, dict):
            key_rosparams_raw = env.get("key_rosparams", {})
            if isinstance(key_rosparams_raw, dict):
                key_rosparams = {
                    str(key): {
                        "status": "ok" if value is not None else "missing",
                        "value": value,
                        "error": "",
                    }
                    for key, value in key_rosparams_raw.items()
                }
                return {
                    "snapshot_schema_version": 1,
                    "captured_at_utc": str(raw.get("ended_at_utc") or raw.get("started_at_utc") or ""),
                    "capture_status": "legacy_metadata_only",
                    "ros": {
                        "targets": {
                            "topics": [],
                            "selected_topics": [],
                            "services": [],
                            "rosparams": list(key_rosparams.keys()),
                            "param_namespaces": [],
                        },
                        "key_topics": {},
                        "selected_values": {},
                        "key_services": {},
                        "key_rosparams": key_rosparams,
                        "key_param_namespaces": {},
                        "nodes": {
                            "status": "legacy_metadata_only",
                            "nodes": [],
                            "important_nodes": [],
                            "error": "",
                        },
                        "important_node_details": {},
                    },
                    "net": {"can_links": []},
                }

    raise RuntimeError(f"No runtime snapshot found in {path}")
