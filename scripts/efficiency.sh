#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="$1"
DATASET="$2"
DATA_PERCENTAGE="$3"
SEED="$4"
CONFIG="configs/${DATASET}_MODULAR.yaml"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Unsupported dataset or missing config: ${CONFIG}" >&2
  exit 1
fi

python train.py \
  --config-file "${CONFIG}" \
  --output-dir "${OUTPUT_ROOT}" \
  --data_percentage "${DATA_PERCENTAGE}" \
  --seed "${SEED}"

python test.py \
  --config-file "${CONFIG}" \
  --output-dir "${OUTPUT_ROOT}" \
  --source_dataset "${DATASET}" \
  --data_percentage "${DATA_PERCENTAGE}" \
  --seed "${SEED}"

python utils/eval.py \
  --config-file "${CONFIG}" \
  --output-dir "${OUTPUT_ROOT}" \
  --data_percentage "${DATA_PERCENTAGE}" \
  --seed "${SEED}"
