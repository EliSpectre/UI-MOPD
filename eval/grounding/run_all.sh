#!/bin/bash
# Run 3 benchmarks (ScreenSpot-Pro, ScreenSpotV2, OSWorld-G) for multiple models.
# All use custom system prompt with tool-call format.
# Parameters: enable_thinking, temperature=0.7, top_p=0.8, top_k=20, max_new_tokens=8192
# 8 GPUs, batch_size=4

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ============ Model Paths (MODIFY THESE) ============
MODEL_1="/path/to/model_1"
MODEL_2="/path/to/model_2"
MODEL_3="/path/to/model_3"

# ============ Data Paths (MODIFY THESE) ============
SCREENSPOT_PRO_DIR="/path/to/ScreenSpot-Pro/"
SCREENSPOT_PRO_IMGS="${SCREENSPOT_PRO_DIR}images"
SCREENSPOT_V2_DIR="/path/to/ScreenSpot-v2/"
SCREENSPOT_V2_IMGS="${SCREENSPOT_V2_DIR}screenspotv2_image/"
OSWORLD_G_DIR="/path/to/OSWorld-G/"
OSWORLD_G_CLASSIFICATION="/path/to/OSWorld-G/benchmark/classification_result.json"

# ============ Common Parameters ============
NUM_GPUS=8
BATCH_SIZE=4
NUM_WORKERS=4
MAX_NEW_TOKENS=8192
TEMPERATURE=0.7
TOP_P=0.8
TOP_K=20

COMMON_ARGS="--batch-size ${BATCH_SIZE} --num-workers ${NUM_WORKERS} --max-new-tokens ${MAX_NEW_TOKENS} --enable-thinking --temperature ${TEMPERATURE} --top-p ${TOP_P} --top-k ${TOP_K}"

# ============ Run Function ============
run_eval() {
    local model_path=$1
    local model_name=$2
    local benchmark=$3
    local script=$4
    local extra_args=$5
    local log_file="${LOG_DIR}/${model_name}_${benchmark}.log"

    echo "========================================"
    echo "Model: ${model_name}"
    echo "Benchmark: ${benchmark}"
    echo "Log: ${log_file}"
    echo "========================================"

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=${NUM_GPUS} \
        "${SCRIPT_DIR}/${script}" \
        --model-path "${model_path}" \
        ${COMMON_ARGS} \
        ${extra_args} \
        2>&1 | tee "${log_file}"

    echo ""
}

# ============ Run All ============
MODELS=(
    "${MODEL_1}|model_1"
    "${MODEL_2}|model_2"
    "${MODEL_3}|model_3"
)

for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r model_path model_name <<< "${model_entry}"

    # ScreenSpot-Pro
    run_eval "${model_path}" "${model_name}" "screenspot_pro" "screenspot_pro_official.py" \
        "--screenspot-imgs ${SCREENSPOT_PRO_IMGS} --screenspot-test ${SCREENSPOT_PRO_DIR}"

    # ScreenSpotV2
    run_eval "${model_path}" "${model_name}" "screenspot_v2" "screenspot_v2_official.py" \
        "--screenspot-imgs ${SCREENSPOT_V2_IMGS} --screenspot-test ${SCREENSPOT_V2_DIR}"

    # OSWorld-G
    run_eval "${model_path}" "${model_name}" "osworld_g" "osworld_g_official.py" \
        "--data-dir ${OSWORLD_G_DIR} --classification-path ${OSWORLD_G_CLASSIFICATION}"
done

echo "All evaluations complete. Logs saved to: ${LOG_DIR}/"
