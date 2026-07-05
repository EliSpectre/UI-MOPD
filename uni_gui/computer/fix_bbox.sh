#!/usr/bin/env bash
# Generate grounding bboxes for OSWorld task.json steps, then retry failures.
#
# This wraps fix_bbox.py and runs the full pipeline by default:
#   PHASE A (generate): fill missing bbox via the model, set coordinate to bbox center
#   PHASE B (retry)   : re-run steps that failed in PHASE A (parallel, multi-round)
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
# Target directories to scan. Each may contain trajectory folders directly
# (with task.json) or a two-level app/episode_id structure.
INPUT_DIRS=(
    "/path/to/dataset/dir1"
)
# Base dir for relative IDs and progress/error logs.
BASE_DIR="/path/to/dataset"

WORKERS=50

python -u fix_bbox.py \
    --workers "${WORKERS}" \
    --base-dir "${BASE_DIR}" \
    --input-dir "${INPUT_DIRS[@]}"

# ---- Useful variants ----
# Resume an interrupted run:
#   python -u fix_bbox.py --resume --workers "${WORKERS}" --base-dir "${BASE_DIR}" --input-dir "${INPUT_DIRS[@]}"
# Overwrite existing bboxes:
#   python -u fix_bbox.py --overwrite --workers "${WORKERS}" --base-dir "${BASE_DIR}" --input-dir "${INPUT_DIRS[@]}"
# Generate only, skip the retry phase:
#   python -u fix_bbox.py --skip-retry --workers "${WORKERS}" --base-dir "${BASE_DIR}" --input-dir "${INPUT_DIRS[@]}"
# Dry run (1 trajectory, no files written):
#   python -u fix_bbox.py --dry-run --base-dir "${BASE_DIR}" --input-dir "${INPUT_DIRS[@]}"
