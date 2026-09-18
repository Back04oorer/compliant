#!/usr/bin/env python3
"""Resolve a Linux video device path from a USB port marker (e.g. '3-7').

If the requested port cannot be resolved, exit non-zero instead of silently
falling back to another device.
"""

import argparse
import glob
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


USB_PORT_RE = re.compile(r"([0-9]+-[0-9.]+)(?::[0-9.]+)?")


def _extract_usb_port(sys_path: str) -> Optional[str]:
    matches = USB_PORT_RE.findall(sys_path)
    if not matches:
        return None
    return matches[-1]


def _udev_sys_path(dev_path: str) -> str:
    try:
        out = subprocess.check_output(
            ["udevadm", "info", "--query=path", "--name", dev_path],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return ""


def _sysfs_path_from_device(dev_path: str) -> str:
    dev_name = os.path.basename(dev_path)
    candidate = os.path.join("/sys/class/video4linux", dev_name)
    if os.path.exists(candidate):
        try:
            return os.path.realpath(candidate)
        except OSError:
            return ""
    return ""


def _device_usb_port(dev_path: str) -> Optional[str]:
    sys_path = _udev_sys_path(dev_path)
    if not sys_path:
        sys_path = _sysfs_path_from_device(dev_path)
    if not sys_path:
        return None
    return _extract_usb_port(sys_path)


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _interface_driver_name(iface_path: str) -> str:
    driver_link = os.path.join(iface_path, "driver")
    if not os.path.exists(driver_link):
        return "unbound"
    try:
        return os.path.basename(os.path.realpath(driver_link)) or "unbound"
    except OSError:
        return "unbound"


def _usb_port_diagnostics(port: str) -> Optional[Dict[str, object]]:
    root = os.path.join("/sys/bus/usb/devices", port)
    if not os.path.exists(root):
        return None

    interfaces: List[Dict[str, str]] = []
    for iface_path in sorted(glob.glob(f"/sys/bus/usb/devices/{port}:*")):
        if not os.path.isdir(iface_path):
            continue
        interfaces.append(
            {
                "name": os.path.basename(iface_path),
                "class": _read_text(os.path.join(iface_path, "bInterfaceClass")),
                "subclass": _read_text(os.path.join(iface_path, "bInterfaceSubClass")),
                "protocol": _read_text(os.path.join(iface_path, "bInterfaceProtocol")),
                "driver": _interface_driver_name(iface_path),
            }
        )

    return {
        "port": port,
        "manufacturer": _read_text(os.path.join(root, "manufacturer")),
        "product": _read_text(os.path.join(root, "product")),
        "vendor_id": _read_text(os.path.join(root, "idVendor")),
        "product_id": _read_text(os.path.join(root, "idProduct")),
        "interfaces": interfaces,
    }


def _format_unresolved_port_error(port: str) -> str:
    base = f"No video device matched USB port: {port}"
    diag = _usb_port_diagnostics(port)
    if not diag:
        return base

    details: List[str] = [base]
    label_parts = [str(diag.get("manufacturer") or "").strip(), str(diag.get("product") or "").strip()]
    label = " ".join(part for part in label_parts if part).strip()
    vid = str(diag.get("vendor_id") or "").strip()
    pid = str(diag.get("product_id") or "").strip()
    if label or (vid and pid):
        suffix = []
        if label:
            suffix.append(label)
        if vid and pid:
            suffix.append(f"{vid}:{pid}")
        details.append(f"USB device present ({', '.join(suffix)})")

    interfaces = diag.get("interfaces") or []
    if interfaces:
        iface_summary = ", ".join(
            f"{item['name']} class={item['class']}/{item['subclass']} driver={item['driver']}"
            for item in interfaces
        )
        details.append(f"interfaces: {iface_summary}")

        video_ifaces = [item for item in interfaces if item.get("class", "").lower() == "0e"]
        if video_ifaces and all(item.get("driver") == "unbound" for item in video_ifaces):
            details.append(
                "USB video interfaces are unbound to uvcvideo; run on host: "
                f"bash workspaces/scripts/bind_uvc_camera_by_port.sh {port}"
            )

    return ". ".join(details)


def _score_link(link_path: str) -> int:
    # Prefer the canonical capture node and keep deterministic ordering.
    score = 0
    name = os.path.basename(link_path).lower()
    if name.endswith("video-index0"):
        score += 20
    if "/by-path/" in link_path:
        score += 10
    if "046d" in name or "c920" in name:
        score += 5
    return score


def _supports_capture(dev_path: str) -> bool:
    real_dev = os.path.realpath(dev_path)
    if not os.path.exists(real_dev):
        return False
    try:
        proc = subprocess.run(
            ["v4l2-ctl", "-d", real_dev, "--list-formats-ext"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=0.8,
        )
        return proc.returncode == 0
    except FileNotFoundError:
        # If v4l2-ctl is unavailable, do not block resolution.
        return True
    except Exception:
        return False


def _candidates_by_path(port: str) -> List[Tuple[int, str]]:
    candidates: List[Tuple[int, str]] = []
    for link_path in sorted(glob.glob("/dev/v4l/by-path/*")):
        if not os.path.islink(link_path):
            continue
        try:
            target = os.path.realpath(link_path)
        except OSError:
            continue
        if _device_usb_port(target) != port:
            continue
        score = _score_link(link_path)
        if _supports_capture(target):
            score += 100
        candidates.append((score, link_path))
    return candidates


def _candidates_by_id(port: str) -> List[Tuple[int, str]]:
    candidates: List[Tuple[int, str]] = []
    for link_path in sorted(glob.glob("/dev/v4l/by-id/*")):
        if not os.path.islink(link_path):
            continue
        try:
            target = os.path.realpath(link_path)
        except OSError:
            continue
        if _device_usb_port(target) != port:
            continue
        score = _score_link(link_path)
        if _supports_capture(target):
            score += 100
        candidates.append((score, target))
    return candidates


def _candidates_by_devnode(port: str) -> List[str]:
    matches: List[str] = []
    for sys_entry in sorted(glob.glob("/sys/class/video4linux/video*")):
        dev_name = os.path.basename(sys_entry)
        dev_path = f"/dev/{dev_name}"
        if _device_usb_port(dev_path) == port and _supports_capture(dev_path):
            matches.append(dev_path)
    return matches


def resolve_video_device(port_or_path: str) -> str:
    request = (port_or_path or "").strip()
    if not request:
        raise RuntimeError("USB port marker is empty")
    if request.startswith("/dev/"):
        if not os.path.exists(request):
            raise RuntimeError(f"Requested video device does not exist: {request}")
        return request

    by_path = _candidates_by_path(request)
    if by_path:
        by_path.sort(key=lambda item: (-item[0], item[1]))
        return by_path[0][1]

    by_id = _candidates_by_id(request)
    if by_id:
        by_id.sort(key=lambda item: (-item[0], item[1]))
        return by_id[0][1]

    by_dev = _candidates_by_devnode(request)
    if by_dev:
        return by_dev[0]

    raise RuntimeError(_format_unresolved_port_error(request))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--usb-port", required=True, help="USB port marker, e.g. 3-7")
    args = parser.parse_args()

    try:
        device = resolve_video_device(args.usb_port)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2

    sys.stdout.write(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
