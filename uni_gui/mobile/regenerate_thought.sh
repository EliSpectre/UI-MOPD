#!/usr/bin/env bash
# Condense verbose model reasoning into concise 5-section thoughts (MobileWorld).
#
# This wraps regenerate_thought.py. It rewrites the `thought` and `action`
# fields of every step in each task.json (moving the original into `raw_thought`).
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
# Target directories; each contains per-episode folders with task.json.
INPUT_DIRS=(
    "/path/to/dataset/dir1"
)

WORKERS=100

python -u regenerate_thought.py \
    --workers "${WORKERS}" \
    --input-dir "${INPUT_DIRS[@]}"

# ---- Useful variants ----
# Resume (skip steps whose raw_thought is already filled):
#   python -u regenerate_thought.py --resume --workers "${WORKERS}" --input-dir "${INPUT_DIRS[@]}"
# Dry run (test 1 step only):
#   python -u regenerate_thought.py --dry-run --input-dir "${INPUT_DIRS[@]}"
