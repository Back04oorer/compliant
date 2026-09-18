#!/usr/bin/env python3
"""Print installation commands by default. Never starts ROS, CAN or training."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def ros_commands(root=False):
    sudo = [] if root else ["sudo"]
    packages = [line.strip() for line in (ROOT / "requirements/ros-apt.txt").read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")]
    return [sudo + ["apt-get", "update"],
            sudo + ["env", "DEBIAN_FRONTEND=noninteractive", "apt-get", "install", "-y", "--no-install-recommends", *packages],
            sudo + ["/usr/bin/python3", "-m", "pip", "install", "-r", str(ROOT / "requirements/ros-python.txt")]]


def training_commands(python, venv, cuda_index):
    pip = [str(venv / "bin/python"), "-m", "pip"]
    return [[python, "-m", "venv", str(venv)],
            pip + ["install", "--upgrade", "pip"],
            pip + ["install", "torch==2.10.0", "torchvision==0.25.0", "--index-url", cuda_index],
            pip + ["install", "-r", str(ROOT / "requirements/training.txt")],
            pip + ["install", "-e", str(ROOT / "training")],
            pip + ["check"],
            [str(venv / "bin/python"), str(ROOT / "scripts/check_training_env.py")]]


def check_ros_host():
    release = dict(line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines() if "=" in line)
    if release.get("ID", "").strip('"') != "ubuntu" or release.get("VERSION_ID", "").strip('"') != "20.04":
        raise ValueError("ROS installer requires Ubuntu 20.04. On newer hosts use docker/Dockerfile.ros.")
    if not Path("/opt/ros/noetic/setup.bash").is_file():
        raise ValueError("ROS Noetic base/repository missing. Use the official ROS image via docker/Dockerfile.ros first.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=["ros", "training"])
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--python", default="python3.10", help="Training only: Python 3.10-3.12 with venv support")
    parser.add_argument("--venv", type=Path, default=ROOT / ".venv-training")
    parser.add_argument("--torch-index", default="https://download.pytorch.org/whl/cu128")
    args = parser.parse_args()
    venv = args.venv.expanduser().resolve()
    commands = ros_commands(os.geteuid() == 0) if args.target == "ros" else training_commands(args.python, venv, args.torch_index)
    for command in commands:
        print(shlex.join(command), flush=True)
    if not args.execute:
        print("PREVIEW ONLY. Add --execute to install. No CAN/robot commands are executed.")
        return 0
    try:
        if args.target == "ros":
            check_ros_host()
        else:
            if venv.exists():
                raise ValueError("Refusing to modify an existing training environment; select a new --venv path.")
            version = json.loads(subprocess.check_output([args.python, "-c", "import json,sys; print(json.dumps(list(sys.version_info[:2])))"], text=True))
            if version not in ([3, 10], [3, 11], [3, 12]):
                raise ValueError("Training requires Python 3.10-3.12, isolated from ROS Python 3.8.")
        for command in commands:
            subprocess.run(command, check=True, cwd=ROOT)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Installation stopped: {exc}\nNo robot was started. Review errors before retrying.\n")
    print("Environment installation/check complete. Hardware and model smoke tests are separate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
