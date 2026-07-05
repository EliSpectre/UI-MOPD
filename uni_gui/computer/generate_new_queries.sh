#!/usr/bin/env bash
# Generate new task queries for the OSWorld/computer trajectories.
#
# This wraps generate_new_queries.py and runs both generation strategies:
#   - dimension : 5 minor + 10 major variants per task
#   - freeform  : N free-divergence queries per task (default 10)
#
# Edit the placeholder paths and model settings below before running.
set -euo pipefail

# -------- Model API config (export as environment variables) --------
# Fill in your own vision-language model endpoint (OpenAI/Gemini-compatible).
export MODEL_URL="https://your-model-endpoint/v1/chat/completions"
export MODEL_NAME="your-model-name"
export MODEL_PROVIDER_ID="your-provider-id"
export GEMINI_API_KEY="YOUR_API_KEY_HERE"

# -------- Paths (edit to your environment) --------
# Root containing per-task folders (each with task.json + screenshot_step*.png).
DATASET_DIR="/path/to/dataset/best"
# Output directory for generated query CSVs and the cache file.
OUTPUT_DIR="/path/to/output/query"

WORKERS=50

mkdir -p "${OUTPUT_DIR}"

# ===================== dimension mode =====================
# Input CSV needs columns: task_id, domain, instruction (or original_query).
DIM_CSV_INPUT="/path/to/input/dimension_tasks.csv"
# Optional CSV providing precondition info (task_id -> needs_precondition/...).
DIM_PRECOND_CSV="/path/to/input/collected_to_improve.csv"

python -u generate_new_queries.py \
    --mode dimension \
    --workers "${WORKERS}" \
    --csv-input "${DIM_CSV_INPUT}" \
    --csv-output "${OUTPUT_DIR}/new_queries_dimension.csv" \
    --cache-file "${OUTPUT_DIR}/generate_cache_dimension.json" \
    --dataset-dir "${DATASET_DIR}" \
    --precond-csv "${DIM_PRECOND_CSV}"

# ===================== freeform mode =====================
FREE_CSV_INPUT="/path/to/input/collected_to_improve.csv"

python -u generate_new_queries.py \
    --mode freeform \
    --queries-per-task 10 \
    --workers "${WORKERS}" \
    --csv-input "${FREE_CSV_INPUT}" \
    --csv-output "${OUTPUT_DIR}/new_queries_freeform.csv" \
    --cache-file "${OUTPUT_DIR}/generate_cache_freeform.json" \
    --dataset-dir "${DATASET_DIR}" \
    --prev-csv "${OUTPUT_DIR}/new_queries_dimension.csv"
