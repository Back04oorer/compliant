#!/usr/bin/env bash
set -Eeuo pipefail

DATASET_ROOT="${DATASET_ROOT:?set DATASET_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT}"
DINO_MODEL="${DINO_MODEL:?set DINO_MODEL}"
DEFM_ROOT="${DEFM_ROOT:?set DEFM_ROOT}"
ARM_MODE="${ARM_MODE:-bimanual}"
ACTIVE_ARM="${ACTIVE_ARM:-right}"

for variant in fused dino-only defm-only; do
  extra=()
  [[ "${variant}" == defm-only ]] || extra+=(--dino-model "${DINO_MODEL}")
  [[ "${variant}" == dino-only ]] || extra+=(--defm-root "${DEFM_ROOT}")
  python scripts/train.py \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${OUTPUT_ROOT}/${ARM_MODE}_${variant}" \
    --arm-mode "${ARM_MODE}" --active-arm "${ACTIVE_ARM}" \
    --variant "${variant}" "${extra[@]}"
done
