#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/workspaces/scripts/use_robot.sh" >/dev/null
if [[ -f /.dockerenv ]]; then
  library_dir="$(mktemp -d /tmp/piper-camera-window.XXXXXX)"
  trap 'rm -rf -- "${library_dir}"' EXIT
  cc -shared -fPIC -Wall -Werror "${ROOT}/scripts/x11_no_mitshm.c" \
    -Wl,--no-as-needed -lX11 -lxcb -ldl -o "${library_dir}/no_mitshm.so"
  export LD_PRELOAD="${library_dir}/no_mitshm.so${LD_PRELOAD:+:${LD_PRELOAD}}"
fi
python3 "${ROOT}/scripts/select_recording_cameras.py" "$@"
