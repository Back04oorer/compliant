#!/usr/bin/env python3
"""Launch the EyeTrax CSV->ROS bridge from a ROS package script."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def find_workspace_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "EyeTrax" / "gaze_teleop_point_csv_ros.py"
        if candidate.is_file():
            return parent
    raise RuntimeError("Cannot locate workspace root containing EyeTrax/gaze_teleop_point_csv_ros.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="")
    parser.add_argument("--python", default="python3")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root_dir = find_workspace_root()
    script_path = root_dir / "EyeTrax" / "gaze_teleop_point_csv_ros.py"

    env = os.environ.copy()
    if str(args.config).strip():
        env["GAZE_TELEOP_POINT_CSV_ROS_CONFIG"] = str(Path(args.config).expanduser().resolve())

    os.execvpe(str(args.python), [str(args.python), str(script_path)], env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
