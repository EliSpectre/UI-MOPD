#!/usr/bin/env bash
# Generate a new batch of query variants for MobileWorld tasks (15 per task).
#
# This wraps generate_new_queries.py:
#   - 5 minor variants + 10 major variants per task
#   - anti-duplication against a previous-batch CSV
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
# Per-task trajectory folders (each with traj.json + screenshots/).
TRAJ_DIR="/path/to/dataset/trajectories"
# CSV listing the task_name values to process.
TASK_LIST_CSV="/path/to/dataset/successful_tasks.csv"
# Previous-batch CSV used for anti-duplication (leave as a non-existent path to skip).
PREV_CSV="/path/to/dataset/query_variants_v1.csv"
# Output directory for the generated CSV and cache file.
OUTPUT_DIR="/path/to/output/query"

WORKERS=50

mkdir -p "${OUTPUT_DIR}"

python -u generate_new_queries.py \
    --workers "${WORKERS}" \
    --traj-dir "${TRAJ_DIR}" \
    --task-list-csv "${TASK_LIST_CSV}" \
    --prev-csv "${PREV_CSV}" \
    --csv-output "${OUTPUT_DIR}/mobileworld_query_variants_v2.csv" \
    --cache-file "${OUTPUT_DIR}/generate_cache_mw_v2.json"

# ---- Useful variants ----
# Dry run (test only 2 tasks):
#   python -u generate_new_queries.py --dry-run --traj-dir "${TRAJ_DIR}" --task-list-csv "${TASK_LIST_CSV}" \
#       --prev-csv "${PREV_CSV}" --csv-output "${OUTPUT_DIR}/mobileworld_query_variants_v2.csv" \
#       --cache-file "${OUTPUT_DIR}/generate_cache_mw_v2.json"
