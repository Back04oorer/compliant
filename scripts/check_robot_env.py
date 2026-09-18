#!/usr/bin/env python3
"""Offline imports and URDF gravity calculation only; never instantiate robot nodes."""
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    # Noetic Pinocchio uses site-packages, whereas standard ROS uses dist-packages.
    site = Path(f"/opt/ros/noetic/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
    if site.is_dir():
        sys.path.insert(0, str(site))
    drivers = ROOT / "src/zeno-wholebody-intervation/zeno-wholebody-robot/common/piper_ctrl/scripts"
    sys.path.insert(0, str(drivers))
    for name in ("rospy", "can", "piper_sdk", "piper_msgs.msg", "piper_msgs.srv", "cv_bridge",
                 "piper_ctrl_follower_node", "piper_gravity_compensation_node"):
        importlib.import_module(name)
        print(f"OK import {name}")
    import numpy as np
    import pinocchio as pin
    model = pin.buildModelFromUrdf(str(ROOT / "src/piper_ros/src/piper_description/urdf/piper_description.urdf"))
    gravity = pin.computeGeneralizedGravity(model, model.createData(), pin.neutral(model))
    if not np.isfinite(gravity).all():
        raise RuntimeError("URDF gravity calculation is not finite")
    print(json.dumps({"piper_sdk": importlib.metadata.version("piper_sdk"),
                      "python-can": importlib.metadata.version("python-can"),
                      "pinocchio": pin.__version__, "urdf_nq": model.nq, "urdf_nv": model.nv}))
    print("PASS: imports and offline URDF dynamics; no CAN, ROS master or hardware connection")


if __name__ == "__main__":
    main()
