#!/usr/bin/env bash
# Clean newly collected OSWorld trajectories.
#
# This wraps clean_trajectories.py and runs the full pipeline by default:
#   PHASE A (clean) : raw traj.jsonl -> 4-stage cleaning -> task.json
#   PHASE B (dedup) : mark retry-loop / meaningless-repeat steps is_use=false
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
# Input roots and output roots correspond one-to-one.
# Each input root contains per-app folders; each app folder contains per-episode
# folders with traj.jsonl + instruction.txt + step screenshots.
INPUT_DIRS=(
    "/path/to/input/best"
)
OUTPUT_DIRS=(
    "/path/to/output/best"
)

WORKERS=50
THRESHOLD=70

python -u clean_trajectories.py \
    --workers "${WORKERS}" \
    --threshold "${THRESHOLD}" \
    --input-dir "${INPUT_DIRS[@]}" \
    --output-dir "${OUTPUT_DIRS[@]}"

# ---- Useful variants ----
# Dry run (statistics only, no files written):
#   python -u clean_trajectories.py --dry-run --input-dir "${INPUT_DIRS[@]}" --output-dir "${OUTPUT_DIRS[@]}"
# Basic clean + convert only, no Gemini (also skips dedup):
#   python -u clean_trajectories.py --skip-gemini --input-dir "${INPUT_DIRS[@]}" --output-dir "${OUTPUT_DIRS[@]}"
# Run the clean phase only (no dedup):
#   python -u clean_trajectories.py --skip-dedup --input-dir "${INPUT_DIRS[@]}" --output-dir "${OUTPUT_DIRS[@]}"
