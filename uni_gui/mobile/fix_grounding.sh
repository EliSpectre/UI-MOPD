#!/usr/bin/env bash
# Generate grounding bboxes for MobileWorld task.json steps, then retry failures.
#
# This wraps fix_grounding.py and runs the full pipeline by default:
#   PHASE A (generate): fill missing bbox via the model, set coordinate to bbox center
#                       (swipe also gets bbox2 for the second point)
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
# Base dir; the script scans "<BASE_DIR>/dataset" for trajectory folders and
# writes progress/error logs under BASE_DIR.
BASE_DIR="/path/to/dataset"

WORKERS=50

python -u fix_grounding.py \
    --workers "${WORKERS}" \
    --base-dir "${BASE_DIR}"

# ---- Useful variants ----
# Resume an interrupted run:
#   python -u fix_grounding.py --resume --workers "${WORKERS}" --base-dir "${BASE_DIR}"
# Overwrite existing bboxes:
#   python -u fix_grounding.py --overwrite --workers "${WORKERS}" --base-dir "${BASE_DIR}"
# Generate only, skip the retry phase:
#   python -u fix_grounding.py --skip-retry --workers "${WORKERS}" --base-dir "${BASE_DIR}"
# Dry run (1 trajectory, no files written):
#   python -u fix_grounding.py --dry-run --base-dir "${BASE_DIR}"
