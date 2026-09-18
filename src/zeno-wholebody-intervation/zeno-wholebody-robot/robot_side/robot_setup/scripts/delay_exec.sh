#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "[delay_exec] usage: delay_exec.sh <seconds> <command> [args...]" >&2
  exit 2
fi

delay_sec="$1"
shift

sleep "$delay_sec"
exec "$@"
