#!/usr/bin/env bash
# Install only the container's model runtime; starts no ROS or robot process.
set -euo pipefail
cd "$(dirname "$0")/.."
# BuildKit RUN steps need not expose /.dockerenv; require the deployed ROS checkout.
[[ "$PWD" == /opt/piper && -f /opt/ros/noetic/setup.bash ]] || {
    echo 'Run this installer in the ROS container checkout at /opt/piper.' >&2; exit 1;
}
unset PYTHONPATH PYTHONHOME
export UV_PYTHON_INSTALL_DIR=/opt/piper-python
uv python install 3.11.15
if [[ ! -x /opt/piper-inference/bin/python ]]; then
    uv venv --python /opt/piper-python/cpython-3.11.15-linux-x86_64-gnu/bin/python3.11 /opt/piper-inference
fi
uv pip install --no-cache --python /opt/piper-inference/bin/python \
    --reinstall-package torch --reinstall-package torchvision \
    --index-url "${INFERENCE_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}" \
    torch==2.10.0 torchvision==0.25.0
uv pip install --no-cache --python /opt/piper-inference/bin/python -r requirements/inference.txt
uv pip check --python /opt/piper-inference/bin/python
/opt/piper-inference/bin/python -c 'import torch, cv2; from lerobot.policies.act.modeling_act import ACTPolicy; from lerobot.policies.factory import make_pre_post_processors; print("Container RGB ACT runtime ready:", torch.__version__)'
