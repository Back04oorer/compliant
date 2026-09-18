#!/usr/bin/env python3
"""Launch RViz with image panels that match the enabled camera set."""

import argparse
import os
import subprocess
import sys
import tempfile

import yaml


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _image_display(name: str, topic: str, transport_hint: str) -> dict:
    return {
        "Class": "rviz/Image",
        "Enabled": True,
        "Image Topic": topic,
        "Max Value": 1,
        "Median window": 5,
        "Min Value": 0,
        "Name": name,
        "Normalize Range": True,
        "Queue Size": 2,
        "Transport Hint": transport_hint,
        "Unreliable": False,
        "Value": True,
    }


def _camera_displays(args: argparse.Namespace) -> list:
    displays = []
    transport_hint = str(args.rviz_image_transport).strip() or "compressed"
    if _parse_bool(args.enable_cameras):
        displays.extend(
            [
                _image_display("Wrist_Left", f"/{args.camera_left_name}/usb_cam_node/image_raw", transport_hint),
                _image_display("Wrist_Right", f"/{args.camera_right_name}/usb_cam_node/image_raw", transport_hint),
                _image_display("Overhead", f"/{args.camera_top_name}/color/image_raw", transport_hint),
            ]
        )
        if _parse_bool(args.camera_extra_1_enable):
            displays.append(
                _image_display(
                    args.camera_extra_1_name,
                    f"/{args.camera_extra_1_name}/color/image_raw",
                    transport_hint,
                )
            )
    return displays


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_config(path: str, config: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def _update_config(config: dict, displays: list) -> dict:
    viz = config.setdefault("Visualization Manager", {})
    viz["Displays"] = displays

    geometry = config.setdefault("Window Geometry", {})
    for key in list(geometry.keys()):
        if key in {"Wrist_Left", "Wrist_Right", "Overhead"}:
            geometry.pop(key, None)
    for display in displays:
        geometry[display["Name"]] = {"collapsed": False}
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--enable-cameras", default="true")
    parser.add_argument("--camera-left-name", default="realsense_left")
    parser.add_argument("--camera-right-name", default="realsense_right")
    parser.add_argument("--camera-top-name", default="realsense_top")
    parser.add_argument("--camera-extra-1-enable", default="false")
    parser.add_argument("--camera-extra-1-name", default="realsense_extra_1")
    parser.add_argument("--rviz-image-transport", default="compressed")
    parser.add_argument("--output-config", default="")
    args, passthrough = parser.parse_known_args()

    config = _load_config(args.base_config)
    config = _update_config(config, _camera_displays(args))

    output_config = args.output_config or os.path.join(
        tempfile.gettempdir(), "robot_setup_dynamic_rviz.rviz"
    )
    _write_config(output_config, config)

    if args.output_config:
        return 0

    os.execvp("rviz", ["rviz", "-d", output_config] + passthrough)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
