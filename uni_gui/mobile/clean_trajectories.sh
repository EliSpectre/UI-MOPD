#!/usr/bin/env bash
# Clean MobileWorld non-rephrase variant trajectories.
#
# This wraps clean_trajectories.py and runs the full 4-stage pipeline by default:
#   Stage 1 basic clean -> Stage 2 precondition check (Gemini)
#   -> Stage 3 sub-task completion eval (Gemini) -> Stage 4 convert to task.json
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
# Input root contains <TaskName_vN_suffix[_backup_TS]> folders, each with
# traj.json + result.txt + screenshots/. Output writes <suffix>/<episode_id>/.
INPUT_DIR="/path/to/mobileworld/variants_traj_logs"
OUTPUT_DIR="/path/to/output/dataset/variants"

WORKERS=100
THRESHOLD=70

python -u clean_trajectories.py \
    --workers "${WORKERS}" \
    --threshold "${THRESHOLD}" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}"

# ---- Useful variants ----
# Dry run (Stage 1 statistics only, no gemini, no files written):
#   python -u clean_trajectories.py --dry-run --input-dir "${INPUT_DIR}" --output-dir "${OUTPUT_DIR}"
# Sample N trajectories through the full Stage 1+2+3 (no disk writes):
#   python -u clean_trajectories.py --dry-up 20 --input-dir "${INPUT_DIR}" --output-dir "${OUTPUT_DIR}"
# Basic clean + convert only, no Gemini (Stage 1+4):
#   python -u clean_trajectories.py --skip-gemini --input-dir "${INPUT_DIR}" --output-dir "${OUTPUT_DIR}"
# Process a single suffix only:
#   python -u clean_trajectories.py --only change_action --input-dir "${INPUT_DIR}" --output-dir "${OUTPUT_DIR}"
