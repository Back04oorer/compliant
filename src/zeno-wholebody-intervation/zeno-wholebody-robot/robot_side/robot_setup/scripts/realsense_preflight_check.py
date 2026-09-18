#!/usr/bin/env python3
"""Preflight checks before launching RealSense camera nodes."""

from __future__ import annotations

import os
import re
import signal
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import rospy


def _as_bool(name: str, default: bool) -> bool:
    value = rospy.get_param(name, default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in ("1", "true", "yes", "on")


def _csv_param(name: str, default: str = "") -> List[str]:
    raw = str(rospy.get_param(name, default))
    return [part.strip() for part in raw.split(",") if part.strip()]


def _run(cmd: Sequence[str]) -> Tuple[int, str]:
    try:
        completed = subprocess.run(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except Exception as exc:
        return 127, str(exc)
    return completed.returncode, completed.stdout or ""


def _check_realsense_viewer() -> List[str]:
    if shutil.which("pgrep") is None:
        return []
    code, out = _run(["pgrep", "-af", "realsense-viewer"])
    if code not in (0, 1):
        return [f"failed to run pgrep for realsense-viewer: {out.strip()}"]
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if not lines:
        return []
    return [f"realsense-viewer is running: {line}" for line in lines]


def _check_existing_realsense_ros_nodes(expected_camera_names: Sequence[str]) -> List[str]:
    if shutil.which("rosnode") is None:
        return []
    code, out = _run(["rosnode", "list"])
    if code != 0:
        return [f"failed to run rosnode list: {out.strip()}"]

    node_names = [line.strip() for line in out.splitlines() if line.strip()]
    conflicts: List[str] = []
    pattern = re.compile(r"^/([^/]+)/realsense2_camera(?:_manager)?$")
    expected = set(expected_camera_names)
    for node in node_names:
        matched = pattern.match(node)
        if not matched:
            continue
        cam_name = matched.group(1)
        if cam_name in expected:
            conflicts.append(f"existing RealSense node for target camera '{cam_name}': {node}")
        else:
            conflicts.append(f"existing RealSense node: {node}")
    return conflicts


def _read_ppid(pid: int) -> Optional[int]:
    try:
        with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as handle:
            text = handle.read().strip()
    except Exception:
        return None
    parts = text.split()
    if len(parts) < 4:
        return None
    try:
        return int(parts[3])
    except Exception:
        return None


def _read_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            data = handle.read()
    except Exception:
        return ""
    return data.replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()


def _find_roslaunch_parent_pid() -> Optional[int]:
    current = os.getpid()
    for _ in range(32):
        parent = _read_ppid(current)
        if parent is None or parent <= 1:
            return None
        cmdline = _read_cmdline(parent)
        if "roslaunch" in cmdline:
            return parent
        current = parent
    return None


def _rosnode_pid(node_name: str) -> Optional[int]:
    code, out = _run(["rosnode", "info", node_name])
    if code != 0:
        return None
    matched = re.search(r"^\s*Pid:\s*(\d+)\s*$", out, flags=re.MULTILINE)
    if not matched:
        return None
    try:
        return int(matched.group(1))
    except Exception:
        return None


def _extract_node_name(conflict_line: str) -> Optional[str]:
    matched = re.search(r"(:\s*)(/[A-Za-z0-9_/]+)\s*$", conflict_line)
    if not matched:
        return None
    return matched.group(2)


def _filter_conflicts_from_same_roslaunch(conflicts: Sequence[str]) -> List[str]:
    roslaunch_pid = _find_roslaunch_parent_pid()
    if roslaunch_pid is None:
        return list(conflicts)

    kept: List[str] = []
    node_pid_cache: Dict[str, Optional[int]] = {}
    for line in conflicts:
        if not line.startswith("existing RealSense node"):
            kept.append(line)
            continue

        node_name = _extract_node_name(line)
        if not node_name:
            kept.append(line)
            continue

        if node_name not in node_pid_cache:
            node_pid_cache[node_name] = _rosnode_pid(node_name)
        node_pid = node_pid_cache[node_name]
        if node_pid is None:
            kept.append(line)
            continue

        if _is_descendant_of(node_pid, roslaunch_pid):
            rospy.logwarn(
                "Ignoring node conflict from same roslaunch ancestry: node=%s pid=%s roslaunch_pid=%s",
                node_name,
                node_pid,
                roslaunch_pid,
            )
            continue
        kept.append(line)
    return kept


def _is_descendant_of(pid: int, ancestor_pid: int, max_hops: int = 64) -> bool:
    current = pid
    seen = set()
    for _ in range(max_hops):
        if current <= 1 or current in seen:
            return False
        if current == ancestor_pid:
            return True
        seen.add(current)
        parent = _read_ppid(current)
        if parent is None:
            return False
        current = parent
    return False


def _check_video_device_occupancy() -> List[str]:
    if shutil.which("lsof") is None:
        return []
    code, out = _run(["bash", "-lc", "lsof /dev/video* 2>/dev/null"])
    if code not in (0, 1):
        return [f"failed to run lsof on /dev/video*: {out.strip()}"]
    lines = [line for line in out.splitlines() if line.strip()]
    if len(lines) <= 1:
        return []

    # Skip header line and de-duplicate.
    entries = sorted({line.strip() for line in lines[1:] if line.strip()})
    return [f"/dev/video* is already occupied: {entry}" for entry in entries]


def _check_duplicate_serials(serials: Sequence[str]) -> List[str]:
    serial_list = [s.strip() for s in serials if s.strip()]
    seen = set()
    duplicates = set()
    for serial in serial_list:
        if serial in seen:
            duplicates.add(serial)
        else:
            seen.add(serial)
    if not duplicates:
        return []
    return [f"duplicate RealSense serial configured: {serial}" for serial in sorted(duplicates)]


def _collect_conflicts(
    camera_names: Sequence[str],
    serials: Sequence[str],
    check_video: bool,
) -> List[str]:
    conflicts: List[str] = []
    conflicts.extend(_check_duplicate_serials(serials))
    conflicts.extend(_check_realsense_viewer())
    conflicts.extend(_check_existing_realsense_ros_nodes(camera_names))
    if check_video:
        conflicts.extend(_check_video_device_occupancy())
    return _filter_conflicts_from_same_roslaunch(conflicts)


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _terminate_pid(pid: int, timeout_sec: float = 2.0) -> bool:
    if not _pid_exists(pid):
        return True
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return not _pid_exists(pid)

    deadline = time.time() + max(timeout_sec, 0.0)
    while time.time() < deadline:
        if not _pid_exists(pid):
            return True
        time.sleep(0.1)

    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        return not _pid_exists(pid)

    time.sleep(0.1)
    return not _pid_exists(pid)


def _cleanup_realsense_conflicts(conflicts: Sequence[str]) -> List[str]:
    cleanup_errors: List[str] = []
    candidate_nodes_set = set()
    for line in conflicts:
        if not line.startswith("existing RealSense node"):
            continue
        node_name = _extract_node_name(line)
        if node_name:
            candidate_nodes_set.add(node_name)
    candidate_nodes = sorted(candidate_nodes_set)
    for node_name in candidate_nodes:
        node_pid = _rosnode_pid(node_name)
        rospy.logwarn("Auto-cleanup: trying rosnode kill for %s", node_name)
        code, out = _run(["rosnode", "kill", node_name])
        if code != 0:
            text = out.strip()
            unknown = "Unknown node" in text or "Unknown node(s)" in text
            if not unknown:
                cleanup_errors.append(f"rosnode kill failed for {node_name}: {text}")
        if node_pid is not None and _pid_exists(node_pid):
            if not _terminate_pid(node_pid):
                cleanup_errors.append(f"failed to terminate PID {node_pid} for node {node_name}")

    viewer_pid_pattern = re.compile(r"realsense-viewer is running:\s*(\d+)\b")
    for line in conflicts:
        matched = viewer_pid_pattern.search(line)
        if not matched:
            continue
        pid = int(matched.group(1))
        rospy.logwarn("Auto-cleanup: terminating realsense-viewer pid=%s", pid)
        if not _terminate_pid(pid):
            cleanup_errors.append(f"failed to terminate realsense-viewer pid {pid}")

    return cleanup_errors


def main() -> int:
    rospy.init_node("realsense_preflight_check", anonymous=False)

    strict = _as_bool("~strict", True)
    hold_node = _as_bool("~hold_node", True)
    check_video = _as_bool("~check_video_device_occupancy", True)
    auto_cleanup = _as_bool("~auto_cleanup_conflicts", False)
    auto_cleanup_wait_sec = float(rospy.get_param("~auto_cleanup_wait_sec", 1.0))
    camera_names = _csv_param("~camera_names_csv", "realsense_left,realsense_right,realsense_top")
    serials = _csv_param("~camera_serials_csv", "")

    conflicts = _collect_conflicts(camera_names, serials, check_video)

    if conflicts and auto_cleanup:
        rospy.logwarn(
            "RealSense preflight found %d conflict(s); auto cleanup enabled, trying to recover.",
            len(conflicts),
        )
        cleanup_errors = _cleanup_realsense_conflicts(conflicts)
        if cleanup_errors:
            for err in cleanup_errors:
                rospy.logwarn("Auto-cleanup issue: %s", err)
        if auto_cleanup_wait_sec > 0.0:
            time.sleep(auto_cleanup_wait_sec)
        conflicts = _collect_conflicts(camera_names, serials, check_video)

    if conflicts:
        rospy.logerr("RealSense preflight check found %d conflict(s).", len(conflicts))
        for item in conflicts:
            rospy.logerr("  - %s", item)
        if strict:
            rospy.logerr("Strict mode enabled; refusing to launch camera nodes.")
            return 2
        rospy.logwarn("Strict mode disabled; continue launch despite conflicts.")
    else:
        rospy.loginfo("RealSense preflight check passed (strict=%s).", strict)

    if hold_node:
        rospy.loginfo("RealSense preflight node is holding to enforce required lifecycle.")
        while not rospy.is_shutdown():
            time.sleep(0.5)

    return 0


if __name__ == "__main__":
    sys.exit(main())
