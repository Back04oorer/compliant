#!/usr/bin/env python3
"""Piper ROS1 workflow. Preview by default; --execute explicitly starts a command."""
import argparse
import json
import math
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def build_command(action, cfg, session=None, cameras=None, image_width=680, image_height=480):
    if action == "core":
        return ["roscore"]
    if action == "robot":
        cmd = ["roslaunch", "robot_setup", "start_robot_all.launch",
               "left_can_port:=" + cfg["robot_left_can"], "right_can_port:=" + cfg["robot_right_can"],
               "robot_control_mode:=teleop", "piper_ctrl_mode:=mit", "auto_enable:=true",
               "enable_auto_zero_on_start:=false", "enable_follower_gravity:=true",
               "enable_handeye_tf:=false", "enable_opp_slave:=false",
               "enable_cameras:=" + str(cfg["enable_cameras"]).lower(),
               "enable_rviz:=" + str(cfg["enable_rviz"]).lower()]
        if cfg["enable_cameras"]:
            for key in ("left_wrist_usb_port", "right_wrist_usb_port", "top_usb_port"):
                if not cfg[key]:
                    raise ValueError("Camera USB ports must be set before enabling cameras")
            cmd += [
                "camera_left_backend:=usb", "camera_left_name:=fisheye_left",
                "camera_left_usb_port:=" + cfg["left_wrist_usb_port"],
                "camera_left_width:=640", "camera_left_height:=480",
                "camera_right_backend:=usb", "camera_right_name:=fisheye_right",
                "camera_right_usb_port:=" + cfg["right_wrist_usb_port"],
                "camera_right_width:=640", "camera_right_height:=480",
                "camera_top_backend:=usb", "camera_top_name:=top_stereo",
                "camera_top_usb_port:=" + cfg["top_usb_port"],
                "camera_top_width:=2560", "camera_top_height:=720", "top_usb_fps:=30",
                "top_usb_pixel_format:=mjpeg", "top_usb_driver_impl:=mjpg_direct",
                "rviz_image_transport:=compressed",
            ]
        return cmd
    if action in {"teleop", "gravity-teleop"}:
        scale = cfg["gravity_joint_scale"]
        if (not isinstance(scale, list) or len(scale) != 6
                or any(type(v) not in (float, int) or not math.isfinite(v) for v in scale)):
            raise ValueError("gravity_joint_scale must contain six finite numbers")
    if action == "teleop":
        return ["roslaunch", "teleop_setup", "start_teleop_all.launch",
                "left_can_port:=" + cfg["teleop_left_can"], "right_can_port:=" + cfg["teleop_right_can"],
                "auto_enable:=true", "enable_paddle:=false", "enable_gravity_compensation:=true",
                "gravity_urdf_package:=piper_description",
                "gravity_urdf_relpath:=urdf/piper_description.urdf",
                "gravity_joint_scale:=" + json.dumps(scale)]
    if action == "gravity-teleop":
        return ["rosrun", "piper_ctrl", "piper_gravity_compensation_node.py",
                "_urdf_package:=piper_description", "_urdf_relpath:=urdf/piper_description.urdf",
                "_gravity_joint_scale:=" + json.dumps(scale)]
    if action in {"mux-teleop", "mux-vla"}:
        return ["bash", str(ROOT / "workspaces/scripts/switch_robot_mux_with_status.sh"), action.split('-', 1)[1]]
    if action.startswith("enable-"):
        _, role, side = action.split("-")
        return ["rosservice", "call", f"/{role}/arm_{side}/enable_srv", "enable_request: true"]
    if action == "record":
        if not session or Path(session).name != session or session in (".", ".."):
            raise ValueError("record requires a simple --session name, not a path")
        if cameras:
            if image_width <= 0 or image_height <= 0:
                raise ValueError("Recording image dimensions must be positive")
            command = ["python3", str(ROOT / "scripts/record_session.py"), "--session", session,
                       "--image-width", str(image_width), "--image-height", str(image_height)]
            for camera in cameras:
                command += ["--camera", camera]
            return command
        return ["python3", str(ROOT / "workspaces/scripts/record_act_keyboard.py"),
                "--profile", str(ROOT / "workspaces/config/rosbag_profiles/lfv_teleop_top_stereo_raw_profile.yaml"),
                "--session-root", str(ROOT / "rosbag_data"), "--session-name", session,
                "--expected-camera-fps", "30", "--save-workers", "1", "--wait-for-save",
                "--save-progress-mode", "verbose", "--dropframe-warn-ratio", "0.90"]
    raise ValueError(action)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["core", "robot", "teleop", "gravity-teleop", "mux-teleop", "mux-vla", "record"] +
                        [f"enable-{r}-{s}" for r in ("robot", "teleop") for s in ("left", "right")])
    parser.add_argument("--config", type=Path, default=ROOT / "config/hardware.local.json")
    parser.add_argument("--session")
    parser.add_argument("--camera", action="append", help="Recording view: left=/dev/video0, right=/dev/video2, top=/dev/videoN")
    parser.add_argument("--image-width", type=int, default=680)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    path = args.config if args.config.exists() else ROOT / "config/hardware.example.json"
    cfg = json.loads(path.read_text())
    cmd = build_command(args.action, cfg, args.session, args.camera, args.image_width, args.image_height)
    print(shlex.join(cmd), flush=True)
    if not args.execute:
        print("PREVIEW ONLY. Add --execute after reviewing hardware and workspace safety.")
        return 0
    if args.action != "core":
        if cfg.get("hardware_reviewed") is not True:
            parser.error("Set hardware_reviewed=true in your reviewed hardware.local.json first")
        ports = [cfg[k] for k in ("robot_left_can", "robot_right_can", "teleop_left_can", "teleop_right_can")]
        if not all(ports) or len(set(ports)) != 4:
            parser.error("All four CAN roles must have unique nonempty interface names")
    # Source in the execution child only; do not modify the operator shell.
    setup = ROOT / "workspaces/scripts/use_robot.sh"
    if args.action == "core":
        script = 'source /opt/ros/noetic/setup.bash; exec "$@"'
        return subprocess.call(["bash", "-c", script, "piper-core", *cmd], cwd=ROOT)
    if not (ROOT / "workspaces/ws_robot/devel/setup.bash").exists():
        parser.error(
            f"ROS workspace is not built in this checkout: {ROOT}. "
            "For Docker, enter the built workspace with: docker exec -it -w /opt/piper piper-control bash; "
            "then run python3 scripts/teleop_tui.py --execute. "
            "For native ROS Noetic, run bash scripts/bootstrap_ros.sh and "
            "bash workspaces/scripts/build_robot_ws.sh; see docs/01_startup.md")
    script = 'source "$1" >/dev/null || exit; shift; exec "$@"'
    command = ["bash", "-c", script, "piper-workflow", str(setup), *cmd]
    if args.action in {"robot", "teleop", "record"}:
        process = subprocess.Popen(command, cwd=ROOT)
        while True:
            try:
                return process.wait()
            except KeyboardInterrupt:
                continue
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
