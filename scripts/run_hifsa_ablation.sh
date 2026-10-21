#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${1:-output_hifsa_ablation}"
PERCENTAGE="${PERCENTAGE:-10}"
SEED="${SEED:-666}"
DATASETS=(BTMRI BUSI Covid19 Kvasir)
COMBINATIONS=(
  "0 0 0"
  "1 0 0"
  "0 1 0"
  "0 0 1"
  "1 1 0"
  "1 0 1"
  "0 1 1"
  "1 1 1"
)

as_bool() {
  if [[ "$1" == "1" ]]; then
    printf 'True'
  else
    printf 'False'
  fi
}

for DATASET in "${DATASETS[@]}"; do
  CONFIG="configs/${DATASET}_MODULAR.yaml"
  if [[ "${PERCENTAGE}" == "100" ]]; then
    DATASET_DIR="${DATASET}"
  else
    DATASET_DIR="${DATASET}_${PERCENTAGE}"
  fi
  if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing config: ${CONFIG}" >&2
    exit 1
  fi

  for COMBINATION in "${COMBINATIONS[@]}"; do
    read -r AFPI_BIT BFCI_BIT LCFSA_BIT <<<"${COMBINATION}"
    AFPI="$(as_bool "${AFPI_BIT}")"
    BFCI="$(as_bool "${BFCI_BIT}")"
    LCFSA="$(as_bool "${LCFSA_BIT}")"
    COMBO="afpi${AFPI_BIT}_bfci${BFCI_BIT}_lcfsa${LCFSA_BIT}"
    RUN_OUT="${OUTPUT_ROOT}/${COMBO}"
    OPTS=(
      MODEL.USE_AFPI "${AFPI}"
      MODEL.USE_BFCI "${BFCI}"
      MODEL.USE_LCFSA "${LCFSA}"
    )

    MODULES=()
    [[ "${AFPI_BIT}" == "1" ]] && MODULES+=(AFPI)
    [[ "${BFCI_BIT}" == "1" ]] && MODULES+=(BFCI)
    [[ "${LCFSA_BIT}" == "1" ]] && MODULES+=(LCFSA)
    if [[ "${#MODULES[@]}" -eq 0 ]]; then
      METHOD="MedCLIPSeg"
    else
      METHOD="HiFSA_$(IFS=_; echo "${MODULES[*]}")"
    fi
    CHECKPOINT="${RUN_OUT}/${DATASET_DIR}/trained_models/seed${SEED}/${METHOD}_unimedclip_ViT-B-16_latest.pth"

    echo "Dataset=${DATASET} Combo=${COMBO} AFPI=${AFPI} BFCI=${BFCI} LCFSA=${LCFSA}"

    python train.py \
      --config-file "${CONFIG}" \
      --output-dir "${RUN_OUT}" \
      --data_percentage "${PERCENTAGE}" \
      --seed "${SEED}" \
      "${OPTS[@]}"

    python test.py \
      --config-file "${CONFIG}" \
      --output-dir "${RUN_OUT}" \
      --source_dataset "${DATASET}" \
      --data_percentage "${PERCENTAGE}" \
      --seed "${SEED}" \
      --prompt_design original \
      --checkpoint-file "${CHECKPOINT}"

    python utils/eval.py \
      --config-file "${CONFIG}" \
      --output-dir "${RUN_OUT}" \
      --data_percentage "${PERCENTAGE}" \
      --seed "${SEED}" \
      --prompt_design original \
      --results-name "${METHOD}_unimedclip_ViT-B-16"
  done
done
