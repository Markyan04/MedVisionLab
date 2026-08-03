#!/usr/bin/env bash
set -Eeuo pipefail

# Sequential single-GPU launcher for the controlled APTOS ablation.
# Rerunning this file is safe: completed seeds are skipped and an interrupted
# seed resumes from its last completed epoch.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/aptos_10seed_ablation.py"

if [[ -n "${APTOS_DATA_ROOT:-}" ]]; then
    DATA_ROOT="${APTOS_DATA_ROOT}"
elif [[ -d "${REPO_ROOT}/APTOS-2019" ]]; then
    DATA_ROOT="${REPO_ROOT}/APTOS-2019"
elif [[ -d "${REPO_ROOT}/APTOS2019" ]]; then
    DATA_ROOT="${REPO_ROOT}/APTOS2019"
elif [[ -d "/root/autodl-tmp/APTOS2019" ]]; then
    DATA_ROOT="/root/autodl-tmp/APTOS2019"
elif [[ -d "/root/autodl-tmp/APTOS-2019" ]]; then
    DATA_ROOT="/root/autodl-tmp/APTOS-2019"
else
    echo "APTOS data not found. Set APTOS_DATA_ROOT to the dataset directory." >&2
    exit 2
fi

if [[ -n "${APTOS_OUTPUT_ROOT:-}" ]]; then
    OUTPUT_ROOT="${APTOS_OUTPUT_ROOT}"
elif [[ -d "/root/autodl-tmp" ]]; then
    OUTPUT_ROOT="/root/autodl-tmp/aptos_10seed_ablation"
else
    OUTPUT_ROOT="${SCRIPT_DIR}/ablation_outputs"
fi

read -r -a SEEDS <<< "${APTOS_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
read -r -a EXPERIMENTS <<< "${APTOS_EXPERIMENTS:-resnet_baseline resnet_layer3_mesc resnet_dast resnet_layer3_mesc_dast}"

mkdir -p "${OUTPUT_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

exec > >(tee -a "${OUTPUT_ROOT}/launcher.log") 2>&1

echo "APTOS 10-seed ablation"
echo "Repository: ${REPO_ROOT}"
echo "Data:       ${DATA_ROOT}"
echo "Output:     ${OUTPUT_ROOT}"
echo "Seeds:      ${SEEDS[*]}"
echo "Experiments:${EXPERIMENTS[*]}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
date
df -h "${OUTPUT_ROOT}" || true
nvidia-smi || true

for seed in "${SEEDS[@]}"; do
    for experiment in "${EXPERIMENTS[@]}"; do
        echo
        echo "======================================================================"
        echo "Running ${experiment}, seed=${seed}"
        echo "======================================================================"
        PYTHONHASHSEED="${seed}" python -u "${TRAIN_SCRIPT}" train \
            --experiment "${experiment}" \
            --seed "${seed}" \
            --data-root "${DATA_ROOT}" \
            --output-root "${OUTPUT_ROOT}" \
            --resume \
            "$@"
    done
done

python -u "${TRAIN_SCRIPT}" summarize \
    --output-root "${OUTPUT_ROOT}" \
    --expected-seeds "${#SEEDS[@]}"

echo
echo "All requested runs completed."
date
