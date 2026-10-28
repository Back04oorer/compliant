#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPOSITORY="${ROOT}/external/fisheye_calibration"
REVISION=6ea4206ab2b1279171ee04a27990178b9992db57
PATCHES=("${ROOT}/patches/fisheye-bottom-ab.patch"
         "${ROOT}/patches/fisheye-charuco-7x5-38.patch")
UV="${UV_BIN:-uv}"
if ! command -v "${UV}" >/dev/null 2>&1; then
  echo "uv is required. Install uv or set UV_BIN to its executable path." >&2
  exit 1
fi
if [[ ! -e "${REPOSITORY}" ]]; then
  mkdir -p "${ROOT}/external"
  git clone --no-checkout https://github.com/Back04oorer/fisheye_calibration.git "${REPOSITORY}"
  git -C "${REPOSITORY}" checkout --detach "${REVISION}"
fi
if [[ "$(git -C "${REPOSITORY}" rev-parse HEAD)" != "${REVISION}" ]]; then
  echo "Existing calibration checkout differs from ${REVISION}; inspect it before setup. Nothing reset." >&2
  exit 1
fi
# Reconstruct only already-applied bundled patches before touching the checkout.
patch_state="$(mktemp -d)"
trap 'rm -rf "${patch_state}"' EXIT
export GIT_INDEX_FILE="${patch_state}/index"
git -C "${REPOSITORY}" read-tree HEAD
for patch in "${PATCHES[@]}"; do
  if git -C "${REPOSITORY}" apply --reverse --check "${patch}" 2>/dev/null; then
    git -C "${REPOSITORY}" apply --cached "${patch}"
  fi
done
if ! git -C "${REPOSITORY}" diff --no-ext-diff --no-color --binary HEAD | cmp -s - <(
  git -C "${REPOSITORY}" diff --cached --no-ext-diff --no-color --binary HEAD
); then
  echo "Calibration checkout has changes beyond the bundled patches; inspect it before setup. Nothing reset." >&2
  exit 1
fi
unset GIT_INDEX_FILE
for patch in "${PATCHES[@]}"; do
  if ! git -C "${REPOSITORY}" apply --reverse --check "${patch}" 2>/dev/null; then
    git -C "${REPOSITORY}" apply --check "${patch}"
    git -C "${REPOSITORY}" apply "${patch}"
  fi
done
cd "${REPOSITORY}/camera_calib_charuco"
"${UV}" sync --locked --python 3.12
.venv/bin/python "${ROOT}/scripts/calibrate_top_camera.py" --check
