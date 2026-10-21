#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${1:-outputs_hifsa_reproduce}"
SEED="${SEED:-666}"
DATASETS=(BTMRI BUSI Covid19 Kvasir)
PERCENTAGES=(10 25 50 100)

for DATASET in "${DATASETS[@]}"; do
  for DATA_PERCENTAGE in "${PERCENTAGES[@]}"; do
    bash scripts/efficiency.sh \
      "${OUTPUT_ROOT}" \
      "${DATASET}" \
      "${DATA_PERCENTAGE}" \
      "${SEED}"
  done
done
