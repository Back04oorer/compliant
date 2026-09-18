#!/usr/bin/env python3
"""Train the full-token Stereo Flow-Matching V2 policy."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for local_path in (REPO_ROOT, SCRIPT_DIR):
    if str(local_path) not in sys.path:
        sys.path.insert(0, str(local_path))

import lerobot_policy_stereo_flow_v2  # noqa: F401,E402
from lerobot_train_stereo_act import _rewrite_alias_args, lerobot_train_main  # noqa: E402


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *_rewrite_alias_args(sys.argv[1:])]
    lerobot_train_main()
