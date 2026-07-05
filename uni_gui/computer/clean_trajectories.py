#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cleaning pipeline for newly collected OSWorld trajectories.

This tool runs a full sequential pipeline by default:

  PHASE A - clean (raw traj.jsonl -> task.json):
    Stage 1  Basic clean: step continuity, last step DONE, action-space
             validity, complete screenshots, <= 40 steps.
    Stage 2  Gemini precondition check: first 3 screenshots + query decide
             whether the task's preconditions are satisfied.
    Stage 3  Gemini task-completion evaluation: decompose into sub-tasks and
             score each one (overall = min of sub-task scores, threshold 70).
    Stage 4  Convert the kept trajectory into task.json format.

  PHASE B - dedup (runs over the task.json produced by PHASE A):
    Use a Gemini VLM to judge whether each step is a retry loop or a
    meaningless repeated click. Keep the first occurrence; mark subsequent
    repeats with is_use=false. Scrolls are never marked as duplicates.

Model API config is read from environment variables (see the bash wrapper):
    MODEL_URL, MODEL_NAME, MODEL_PROVIDER_ID, GEMINI_API_KEY

Usage:
    python clean_trajectories.py --dry-run
    python clean_trajectories.py --workers 50
    python clean_trajectories.py --skip-gemini
    python clean_trajectories.py --threshold 85
    python clean_trajectories.py --skip-dedup        # run clean only

    # Multi-directory mode: --input-dir and --output-dir correspond one-to-one.
    python clean_trajectories.py \
        --input-dir /path/to/input1 /path/to/input2 /path/to/input3 \
        --output-dir /path/to/output1 /path/to/output2 /path/to/output3
"""

import os
import json
import re
import sys
import time
import base64
import shutil
import argparse
import glob as glob_mod
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ======================== Config ========================
# Input/output directories. Edit these defaults or pass --input-dir/--output-dir.
# Each per-task folder under an input root contains traj.jsonl + instruction.txt
# + step screenshots. Outputs are written one folder per episode.
INPUT_ROOTS = [
    "/path/to/input/best",
]
OUTPUT_ROOTS = [
    "/path/to/output/best",
]
SCREEN_RESOLUTION = [1920, 1080]

# ---- Model API config (read from environment, with placeholder defaults) ----
MODEL_URL = os.environ.get("MODEL_URL", "https://your-model-endpoint/v1/chat/completions")
MODEL_NAME = os.environ.get("MODEL_NAME", "your-model-name")
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
MODEL_PROVIDER_ID = os.environ.get("MODEL_PROVIDER_ID", "your-provider-id")
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "X-Model-Provider-Id": MODEL_PROVIDER_ID,
    "Content-Type": "application/json",
    "X-Model-Request-Id": "clean-001",
}

PASS_THRESHOLD = 70
MAX_STEP_IMAGES = 3

W, H = SCREEN_RESOLUTION


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean OSWorld trajectories (clean -> dedup full pipeline)."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Statistics only, do not write files")
    parser.add_argument("--workers", type=int, default=100,
                        help="Number of parallel workers")
    parser.add_argument("--skip-gemini", action="store_true",
                        help="Skip Gemini evaluation (basic clean + convert only)")
    parser.add_argument("--skip-dedup", action="store_true",
                        help="Skip the dedup phase (run the clean phase only)")
    parser.add_argument("--threshold", type=int, default=70,
                        help="Task-completion evaluation threshold")
    parser.add_argument("--num-sample", type=int, default=-1,
                        help="Random sample count, -1 for all")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each dedup decision")
    parser.add_argument("--input-dir", type=str, nargs='+', default=INPUT_ROOTS,
                        help="Input directories; multiple allowed, paired with --output-dir")
    parser.add_argument("--output-dir", type=str, nargs='+', default=OUTPUT_ROOTS,
                        help="Output directories; multiple allowed, paired with --input-dir")
    return parser.parse_args()


# ======================== LLM call ========================

def call_gemini(messages, max_retries=15, max_tokens=8192, temperature=1):
    last_error = ""
    for attempt in range(max_retries):
        try:
            payload = {
                "model": MODEL_NAME,
                "messages": messages,
                "stream": False,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            resp = requests.post(MODEL_URL, headers=HEADERS, json=payload, timeout=300)
            if resp.status_code == 200:
                data = resp.json()
                msg = data["choices"][0]["message"]
                content = msg.get("content", "").strip()
                if content:
                    return content
                reasoning = msg.get("reasoning_content", "").strip()
                if reasoning:
                    return reasoning
                return None
            elif resp.status_code == 429 or "overloaded" in resp.text.lower():
                last_error = f"[429/overloaded] {resp.text[:200]}"
                time.sleep(min(5 * (attempt + 1), 60))
            else:
                last_error = f"[{resp.status_code}] {resp.text[:200]}"
                time.sleep(min(3 * (attempt + 1), 30))
        except Exception as e:
            last_error = f"[exception] {str(e)[:200]}"
            time.sleep(min(3 * (attempt + 1), 30))
    if last_error:
        print(f"    [call_gemini] FAILED after {max_retries} retries: {last_error}", flush=True)
    return None


def parse_json_response(text):
    if not text:
        return None
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        # Fallback: try to pull a small is_duplicate object out of the text.
        match = re.search(r'\{[^{}]*"is_duplicate"[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except (json.JSONDecodeError, ValueError):
                pass
        return None


def encode_image(image_path):
    if not os.path.isfile(image_path):
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ======================== Action validation ========================

def is_known_action_line(line):
    line = line.strip()
    if not line:
        return True
    if line.startswith("#"):
        return True
    known_prefixes = [
        "pyautogui.click(",
        "pyautogui.doubleClick(",
        "pyautogui.rightClick(",
        "pyautogui.tripleClick(",
        "pyautogui.middleClick(",
        "pyautogui.moveTo(",
        "pyautogui.scroll(",
        "pyautogui.hscroll(",
        "pyautogui.typewrite(",
        "pyautogui.write(",
        "pyautogui.sleep(",
        "pyautogui.hotkey(",
        "pyautogui.press(",
        "pyautogui.keyDown(",
        "pyautogui.keyUp(",
        "pyautogui.dragTo(",
        "pyautogui.drag(",
        "pyautogui.mouseDown(",
        "pyautogui.mouseUp(",
        "time.sleep(",
    ]
    for prefix in known_prefixes:
        if line.startswith(prefix):
            return True
    return False


def validate_action_space(action_str):
    action_str = action_str.strip()
    if not action_str:
        return True, []
    if action_str in ("DONE", "FAIL", "WAIT"):
        return True, []
    lines = [l.strip() for l in action_str.split("\n") if l.strip()]
    unmappable = []
    for line in lines:
        if not is_known_action_line(line):
            unmappable.append(line)
    return len(unmappable) == 0, unmappable


# ======================== Action conversion ========================

def to_permille(x, y):
    return [round(x * 999 / W), round(y * 999 / H)]


def extract_coords(s):
    match = re.search(r'\(([^)]+)\)', s)
    if not match:
        return None, None
    parts = match.group(1).split(',')
    try:
        x = int(parts[0].strip())
        y = int(parts[1].strip())
        return x, y
    except (ValueError, IndexError):
        return None, None


def extract_scroll_params(s):
    match = re.search(r'\(([^)]+)\)', s)
    if not match:
        return None, None, None
    parts = [p.strip() for p in match.group(1).split(',')]
    try:
        amount = int(parts[0])
    except ValueError:
        try:
            amount = int(float(parts[0]))
        except ValueError:
            return None, None, None
    if len(parts) >= 3:
        try:
            x = int(parts[1])
            y = int(parts[2])
            return amount, x, y
        except ValueError:
            pass
    return amount, None, None


def extract_typewrite_text(s):
    match = re.search(r'pyautogui\.typewrite\(\s*"""(.*?)"""', s, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"pyautogui\.typewrite\(\s*'''(.*?)'''", s, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r'pyautogui\.typewrite\(\s*"(.*?)"', s)
    if match:
        return match.group(1)
    match = re.search(r"pyautogui\.typewrite\(\s*'(.*?)'", s)
    if match:
        return match.group(1)
    return ""


def extract_write_text(s):
    match = re.search(r"message\s*=\s*'''(.*?)'''", s, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r'message\s*=\s*"""(.*?)"""', s, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"message\s*=\s*'(.*?)'", s)
    if match:
        return match.group(1)
    match = re.search(r'message\s*=\s*"(.*?)"', s)
    if match:
        return match.group(1)
    match = re.search(r'pyautogui\.write\(\s*"""(.*?)"""', s, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"pyautogui\.write\(\s*'''(.*?)'''", s, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r'pyautogui\.write\(\s*"(.*?)"', s)
    if match:
        return match.group(1)
    match = re.search(r"pyautogui\.write\(\s*'(.*?)'", s)
    if match:
        return match.group(1)
    return ""


def extract_key_name(s):
    match = re.search(r"keyDown\(\s*['\"](.+?)['\"]\s*\)", s)
    if match:
        return match.group(1)
    return ""


def extract_sleep_time(s):
    match = re.search(r'sleep\(\s*([\d.]+)\s*\)', s)
    if match:
        return float(match.group(1))
    return 0.5


def parse_action_to_plan(action_str, result_score=1.0):
    action_str = action_str.strip()
    if action_str == "DONE":
        return {"name": "computer_use", "arguments": {"action": "terminate", "status": "success"}}, action_str
    if action_str == "FAIL":
        return {"name": "computer_use", "arguments": {"action": "terminate", "status": "failure"}}, action_str
    if action_str == "WAIT":
        return {"name": "computer_use", "arguments": {"action": "wait", "time": 5}}, action_str
    lines = [l.strip() for l in action_str.split("\n") if l.strip()]
    if len(lines) == 1:
        return parse_single_action(lines[0])
    return parse_combo_action(lines)


def parse_single_action(line):
    if line.startswith("pyautogui.click("):
        x, y = extract_coords(line)
        if x is not None:
            return {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": to_permille(x, y)}}, line
        return {"name": "computer_use", "arguments": {"action": "left_click"}}, line
    if line.startswith("pyautogui.doubleClick("):
        x, y = extract_coords(line)
        if x is not None:
            return {"name": "computer_use", "arguments": {"action": "double_click", "coordinate": to_permille(x, y)}}, line
        return {"name": "computer_use", "arguments": {"action": "double_click"}}, line
    if line.startswith("pyautogui.rightClick("):
        x, y = extract_coords(line)
        if x is not None:
            return {"name": "computer_use", "arguments": {"action": "right_click", "coordinate": to_permille(x, y)}}, line
        return {"name": "computer_use", "arguments": {"action": "right_click"}}, line
    if line.startswith("pyautogui.tripleClick("):
        x, y = extract_coords(line)
        if x is not None:
            return {"name": "computer_use", "arguments": {"action": "triple_click", "coordinate": to_permille(x, y)}}, line
        return {"name": "computer_use", "arguments": {"action": "triple_click"}}, line
    if line.startswith("pyautogui.middleClick("):
        x, y = extract_coords(line)
        if x is not None:
            return {"name": "computer_use", "arguments": {"action": "middle_click", "coordinate": to_permille(x, y)}}, line
        return {"name": "computer_use", "arguments": {"action": "middle_click"}}, line
    if line.startswith("pyautogui.moveTo("):
        x, y = extract_coords(line)
        if x is not None:
            return {"name": "computer_use", "arguments": {"action": "mouse_move", "coordinate": to_permille(x, y)}}, line
        return {"name": "computer_use", "arguments": {"action": "mouse_move"}}, line
    if line.startswith("pyautogui.scroll("):
        amount, x, y = extract_scroll_params(line)
        args = {"action": "scroll"}
        if amount is not None:
            args["pixels"] = amount
        if x is not None and y is not None:
            args["coordinate"] = to_permille(x, y)
        return {"name": "computer_use", "arguments": args}, line
    if line.startswith("pyautogui.hscroll("):
        amount, x, y = extract_scroll_params(line)
        args = {"action": "hscroll"}
        if amount is not None:
            args["pixels"] = amount
        if x is not None and y is not None:
            args["coordinate"] = to_permille(x, y)
        return {"name": "computer_use", "arguments": args}, line
    if line.startswith("pyautogui.typewrite("):
        text = extract_typewrite_text(line)
        return {"name": "computer_use", "arguments": {"action": "type", "text": text}}, line
    if line.startswith("pyautogui.write("):
        text = extract_write_text(line)
        return {"name": "computer_use", "arguments": {"action": "type", "text": text}}, line
    if line.startswith("pyautogui.sleep(") or line.startswith("time.sleep("):
        t = extract_sleep_time(line)
        return {"name": "computer_use", "arguments": {"action": "wait", "time": t}}, line
    if line.startswith("pyautogui.hotkey("):
        match = re.search(r'\((.+)\)', line)
        if match:
            keys = [k.strip().strip("'\"") for k in match.group(1).split(",")]
            return {"name": "computer_use", "arguments": {"action": "key", "keys": keys}}, line
    if line.startswith("pyautogui.press("):
        match = re.search(r"['\"](.+?)['\"]", line)
        if match:
            return {"name": "computer_use", "arguments": {"action": "key", "keys": [match.group(1)]}}, line
    if line.startswith("pyautogui.dragTo(") or line.startswith("pyautogui.drag("):
        x, y = extract_coords(line)
        if x is not None:
            return {"name": "computer_use", "arguments": {"action": "left_click_drag", "coordinate": to_permille(x, y)}}, line
    if line.startswith("pyautogui.mouseDown("):
        x, y = extract_coords(line)
        args = {"action": "left_click_drag"}
        if x is not None:
            args["coordinate"] = to_permille(x, y)
        return {"name": "computer_use", "arguments": args}, line
    if line.startswith("pyautogui.mouseUp("):
        x, y = extract_coords(line)
        args = {"action": "left_click_drag"}
        if x is not None:
            args["coordinate"] = to_permille(x, y)
        return {"name": "computer_use", "arguments": args}, line
    return {"name": "computer_use", "arguments": {"action": "left_click"}}, line


def parse_combo_action(lines):
    if len(lines) == 2 and lines[0].startswith("pyautogui.moveTo(") and lines[1].startswith("pyautogui.scroll("):
        x, y = extract_coords(lines[0])
        amount, _, _ = extract_scroll_params(lines[1])
        args = {"action": "scroll"}
        if amount is not None:
            args["pixels"] = amount
        if x is not None and y is not None:
            args["coordinate"] = to_permille(x, y)
        return {"name": "computer_use", "arguments": args}, "\n".join(lines)
    if len(lines) == 2 and lines[0].startswith("pyautogui.moveTo(") and lines[1].startswith("pyautogui.dragTo("):
        x1, y1 = extract_coords(lines[0])
        x2, y2 = extract_coords(lines[1])
        args = {"action": "left_click_drag"}
        if x2 is not None and y2 is not None:
            args["coordinate"] = to_permille(x2, y2)
        if x1 is not None and y1 is not None:
            args["start_coordinate"] = to_permille(x1, y1)
        return {"name": "computer_use", "arguments": args}, "\n".join(lines)
    if (len(lines) >= 3 and lines[0].startswith("pyautogui.click(") and
        lines[1].startswith("pyautogui.typewrite(") and lines[2].startswith("pyautogui.keyDown(")):
        text = extract_typewrite_text(lines[1])
        return {"name": "computer_use", "arguments": {"action": "type", "text": text}}, "\n".join(lines)
    if (len(lines) == 3 and lines[0].startswith("pyautogui.typewrite(") and
        lines[1].startswith("pyautogui.keyDown(") and lines[2].startswith("pyautogui.keyUp(")):
        text = extract_typewrite_text(lines[0])
        return {"name": "computer_use", "arguments": {"action": "type", "text": text}}, lines[0]
    key_downs = []
    all_keys = True
    for line in lines:
        if line.startswith("pyautogui.keyDown("):
            key_downs.append(extract_key_name(line))
        elif line.startswith("pyautogui.keyUp("):
            pass
        else:
            all_keys = False
            break
    if all_keys and key_downs:
        return {"name": "computer_use", "arguments": {"action": "key", "keys": key_downs}}, "\n".join(lines)
    for line in lines:
        if line.startswith("pyautogui.") and not line.startswith("pyautogui.sleep("):
            plan, _ = parse_single_action(line)
            return plan, "\n".join(lines)
    return {"name": "computer_use", "arguments": {"action": "wait", "time": 1}}, "\n".join(lines)


# ======================== Stage 1: basic clean ========================

def basic_clean(traj_dir):
    """Basic clean. Returns (entries, query, reason); a non-empty reason means discard."""
    traj_path = os.path.join(traj_dir, "traj.jsonl")
    instruction_path = os.path.join(traj_dir, "instruction.txt")

    if not os.path.isfile(traj_path):
        return None, None, "no traj.jsonl"
    if not os.path.isfile(instruction_path):
        return None, None, "no instruction.txt"

    with open(instruction_path, "r", encoding="utf-8") as f:
        query = f.read().strip()

    entries = []
    with open(traj_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not entries:
        return None, None, "empty traj.jsonl"

    if len(entries) > 40:
        return None, None, f"too many steps: {len(entries)} > 40"

    # step_num must increase sequentially
    for idx, entry in enumerate(entries):
        expected = idx + 1
        actual = entry.get("step_num")
        if actual != expected:
            return None, None, f"step_num not sequential: expected {expected}, got {actual}"

    # last step must be DONE
    last_action = entries[-1].get("action", "").strip()
    if last_action != "DONE":
        return None, None, f"last step not DONE: '{last_action[:30]}'"

    # action-space validation
    for entry in entries:
        action_str = entry.get("action", "").strip()
        is_valid, unmappable = validate_action_space(action_str)
        if not is_valid:
            return None, None, f"unmappable actions: {unmappable[:3]}"

    # screenshot validation
    step0_matches = glob_mod.glob(os.path.join(traj_dir, "step_0_*.png"))
    if not step0_matches:
        return None, None, "step_0 screenshot missing"

    for entry in entries:
        screenshot_file = entry.get("screenshot_file", "")
        if not screenshot_file:
            return None, None, "entry missing screenshot_file"
        if not os.path.isfile(os.path.join(traj_dir, screenshot_file)):
            return None, None, f"screenshot missing: {screenshot_file}"

    return entries, query, None


# ======================== Stage 2: precondition check ========================

PROMPT_PRECONDITION = """You are evaluating whether a computer task's preconditions are met based on the initial screenshots.

## Task instruction:
"{query}"

## What to check:
Look at the provided screenshots (the initial state of the computer before the agent starts working). Determine if the environment has the necessary preconditions for this task:
- If the task mentions operating on a specific file/document, is that file visible or accessible?
- If the task requires a specific application to be open, is it open?
- If the task references specific data/content that should already exist, does it appear to exist?
- If the task requires a specific website/page to be loaded, is it loaded?

## Important:
- If the task is about changing a SETTING or navigating to something (no precondition needed), answer true.
- Only answer false if there is CLEAR evidence that a required precondition is NOT met (e.g., file not found error, wrong application, empty document when content is expected).
- When in doubt, answer true (give benefit of the doubt).

Output ONLY JSON:
{{
  "precondition_met": true/false,
  "reason": "Brief explanation"
}}"""


def check_precondition(traj_dir, entries, query):
    """Use Gemini to check the first 3 screenshots; returns (passed, reason)."""
    # Collect first 3 screenshots: step_0 + step_1 + step_2
    screenshot_paths = []
    step0_matches = glob_mod.glob(os.path.join(traj_dir, "step_0_*.png"))
    if step0_matches:
        screenshot_paths.append(step0_matches[0])
    for entry in entries[:2]:
        sf = entry.get("screenshot_file", "")
        if sf:
            path = os.path.join(traj_dir, sf)
            if os.path.isfile(path):
                screenshot_paths.append(path)

    prompt_text = PROMPT_PRECONDITION.format(query=query)
    user_content = [{"type": "text", "text": prompt_text}]

    for img_path in screenshot_paths:
        img_b64 = encode_image(img_path)
        if img_b64:
            user_content.append({"type": "text", "text": f"[{os.path.basename(img_path)}]"})
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})

    messages = [{"role": "user", "content": user_content}]
    resp = call_gemini(messages, max_tokens=4096)
    result = parse_json_response(resp)

    if result is None:
        return None, "API failure"
    return result.get("precondition_met", True), result.get("reason", "")


# ======================== Stage 3: task-completion evaluation ========================

PROMPT_DECOMPOSE = """You are a task decomposition expert. Break down the following user task into sequential, verifiable sub-tasks.

User task: "{query}"

Requirements:
1. Sub-tasks must be in logical execution order
2. Each sub-task should be completable in 1-5 GUI steps
3. Include ALL specific parameters/values from the query in the sub-task description
4. Do NOT merge multiple distinct operations into one sub-task
5. If the query mentions a specific target value (e.g., "set to 120", "change to Dark"), that value MUST appear in the sub-task

Output ONLY JSON:
{{
  "sub_tasks": ["sub-task 1 with specific values", "sub-task 2 with specific values", ...]
}}"""

PROMPT_EVALUATE_STEP = """You are a UI trajectory evaluator for training data quality. Evaluate whether this sub-task was completed.

## Context
Full user instruction: "{query}"
Sub-task being evaluated: "{sub_task}"

## Previous sub-tasks (already evaluated):
{prev_subtasks_status}

## Agent's actions (full trajectory):
{segment_actions}

## Evaluation Rules:

1. **Action history is primary evidence.** If the action sequence clearly shows the agent performed the correct operations for this sub-task (clicked the right buttons, typed the right values, navigated to the right pages), consider it completed.

2. **Screenshots are supplementary.** Use screenshots to confirm or deny, but do NOT require screenshot proof if the action history already demonstrates completion clearly.

3. **Give credit for logical action sequences.** If the agent navigated to the correct settings page and toggled the right option, that's completion — you don't need to see the toggle state in a screenshot.

4. **Focus on the core goal.** Minor inefficiencies or extra steps don't matter. What matters is whether the sub-task's goal was achieved.

## FAIL conditions (score below 50):
- Agent clearly did something UNRELATED to this sub-task
- Agent got stuck in a loop (same action 4+ times)
- Agent's actions show it went to a completely WRONG location
- The sub-task requires a specific value but the agent never typed/selected that value

## Scoring:
- 90-100: Action history clearly shows completion AND screenshot confirms
- 70-89: Action history clearly shows the right operations were performed (completion very likely)
- 50-69: Some relevant actions but unclear if sub-task was fully completed
- 0-49: Agent clearly did NOT complete this sub-task

Output ONLY JSON:
{{
  "sub_task_completed": true/false,
  "score": <0-100>,
  "evidence": "Brief explanation citing specific actions or screenshot elements",
  "issues": "Any problems found (empty string if none)"
}}"""


def decompose_query(query):
    prompt = PROMPT_DECOMPOSE.format(query=query)
    messages = [{"role": "user", "content": prompt}]
    resp = call_gemini(messages, max_tokens=4096)
    data = parse_json_response(resp)
    if data and "sub_tasks" in data and isinstance(data["sub_tasks"], list) and len(data["sub_tasks"]) > 0:
        return data["sub_tasks"]
    return [query]


def evaluate_task_completion(traj_dir, entries, query):
    """Two-stage completion evaluation (decompose sub-tasks + score each).
    Returns (passed, score, details)."""
    # Stage 1: decompose sub-tasks
    sub_tasks = decompose_query(query)

    # Build action history
    all_actions = []
    for i, entry in enumerate(entries):
        nl_action = entry.get("natural_language_action", "")
        action_str = entry.get("action", "")
        all_actions.append(f"Step {i+1}: {nl_action} [{action_str}]")

    # Collect screenshots: first + last N
    screenshot_paths = []
    step0_matches = glob_mod.glob(os.path.join(traj_dir, "step_0_*.png"))
    if step0_matches:
        screenshot_paths.append(step0_matches[0])
    img_start = max(0, len(entries) - MAX_STEP_IMAGES)
    for entry in entries[img_start:]:
        sf = entry.get("screenshot_file", "")
        if sf:
            path = os.path.join(traj_dir, sf)
            if os.path.isfile(path) and path not in screenshot_paths:
                screenshot_paths.append(path)

    # Stage 2: score each sub-task
    subtask_results = []
    for st_idx, sub_task in enumerate(sub_tasks):
        if subtask_results:
            prev_text = "\n".join([
                f"  {i+1}. {st['sub_task']} -> {'COMPLETED' if st['completed'] else 'NOT COMPLETED'}"
                for i, st in enumerate(subtask_results)
            ])
        else:
            prev_text = "  (This is the first sub-task)"

        segment_text = "\n".join(all_actions) if all_actions else "(No actions)"

        eval_text = PROMPT_EVALUATE_STEP.format(
            query=query,
            sub_task=sub_task,
            prev_subtasks_status=prev_text,
            segment_actions=segment_text,
        )

        user_content = [{"type": "text", "text": eval_text}]
        for img_path in screenshot_paths:
            img_b64 = encode_image(img_path)
            if img_b64:
                user_content.append({"type": "text", "text": f"[{os.path.basename(img_path)}]"})
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})

        messages = [{"role": "user", "content": user_content}]
        resp = call_gemini(messages, max_tokens=4096)
        result = parse_json_response(resp)

        if result:
            subtask_results.append({
                "sub_task": sub_task,
                "completed": result.get("sub_task_completed", False),
                "score": result.get("score", 0),
            })
        else:
            subtask_results.append({
                "sub_task": sub_task,
                "completed": False,
                "score": -1,
            })

    # Compute overall score
    valid_scores = [r["score"] for r in subtask_results if r["score"] >= 0]
    if not valid_scores:
        return None, -1, "All evaluations failed (API)"

    has_api_failure = any(r["score"] == -1 for r in subtask_results)
    if has_api_failure:
        return None, -1, "Some evaluations failed (API)"

    overall_score = min(valid_scores)
    passed = overall_score >= PASS_THRESHOLD
    return passed, overall_score, subtask_results


# ======================== Stage 4: convert to task.json ========================

def extract_action_description(response):
    if not response:
        return ""
    if isinstance(response, dict):
        content = response.get("content", "")
    elif isinstance(response, str):
        content = response
    else:
        return ""
    match = re.search(r"##\s*Action:\s*\n?(.+?)(?:\n##|\Z)", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"Action:\s*(.+?)(?:\n\n|<tool_call>|\Z)", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def convert_to_taskjson(traj_dir, entries, query, app_name, output_dir, dry_run=False):
    """Convert and save in task.json format."""
    episode_id = os.path.basename(traj_dir)
    step0_matches = glob_mod.glob(os.path.join(traj_dir, "step_0_*.png"))
    step0_path = step0_matches[0] if step0_matches else None

    data_steps = []
    screenshots_to_copy = []

    if step0_path:
        screenshots_to_copy.append((step0_path, "screenshot_step0.png"))

    for idx, entry in enumerate(entries):
        action_str = entry.get("action", "")
        response = entry.get("response", {})
        nl_action = entry.get("natural_language_action", "")
        if not nl_action:
            nl_action = extract_action_description(response)

        thought = ""
        if isinstance(response, dict):
            thought = response.get("reasoning_content", "") or ""

        plan, code = parse_action_to_plan(action_str)
        screenshot_name = f"screenshot_step{idx}.png"

        step_record = {
            "step": idx + 1,
            "query": query,
            "thought": thought,
            "action": nl_action,
            "pixel": SCREEN_RESOLUTION,
            "plan": plan,
            "bbox": [],
            "screenshot": screenshot_name,
            "code": code,
            "is_use": True,
            "is_reviewed": False,
            "is_delete": False,
            "train_test": "test",
            "raw_thought": "",
        }
        data_steps.append(step_record)

        src_screenshot = os.path.join(traj_dir, entry.get("screenshot_file", ""))
        screenshots_to_copy.append((src_screenshot, f"screenshot_step{idx + 1}.png"))

    task_json = {
        "task": "OSWorld",
        "app": app_name,
        "screen_resolution": SCREEN_RESOLUTION,
        "query": query,
        "episode_id": episode_id,
        "is_delete": False,
        "is_mock": False,
        "device": "computer",
        "verified": True,
        "task_completed": True,
        "data": data_steps,
    }

    if dry_run:
        return task_json

    out_dir = os.path.join(output_dir, episode_id)
    os.makedirs(out_dir, exist_ok=True)

    for src_path, dst_name in screenshots_to_copy:
        dst_path = os.path.join(out_dir, dst_name)
        if src_path and os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)

    task_json_path = os.path.join(out_dir, "task.json")
    with open(task_json_path, "w", encoding="utf-8") as f:
        json.dump(task_json, f, ensure_ascii=False, indent=4)

    return task_json


# ======================== Full processing of one trajectory ========================

def process_one_trajectory(traj_dir, app_name, output_dir, dry_run, skip_gemini):
    """Process one trajectory; returns (episode_id, stage_failed, reason).
    stage_failed: None=success, 1=basic clean, 2=precondition, 3=task completion."""
    episode_id = os.path.basename(traj_dir)

    # Stage 1
    entries, query, reason = basic_clean(traj_dir)
    if reason:
        return episode_id, 1, reason

    if skip_gemini:
        convert_to_taskjson(traj_dir, entries, query, app_name, output_dir, dry_run)
        return episode_id, None, "OK (skip gemini)"

    # Stage 2
    precondition_met, reason = check_precondition(traj_dir, entries, query)
    if precondition_met is None:
        return episode_id, 2, f"API error: {reason}"
    if not precondition_met:
        return episode_id, 2, reason

    # Stage 3
    passed, score, details = evaluate_task_completion(traj_dir, entries, query)
    if passed is None:
        return episode_id, 3, f"API error: {details}"
    if not passed:
        return episode_id, 3, f"score={score} < {PASS_THRESHOLD}"

    # Stage 4
    convert_to_taskjson(traj_dir, entries, query, app_name, output_dir, dry_run)
    return episode_id, None, f"OK (score={score})"


# ======================== Dedup phase (operates on task.json) ========================

EVALUATION_PROMPT = """You are a GUI operation-trajectory data cleaning expert. Decide whether the current step (Step {current_step}) is an **erroneous retry loop** or a **meaningless repeated click**.

## Task instruction
{query}

## Current step info (the step to judge)
- Step {current_step}
- Action: {current_action}
- Code: {current_code}

## Context steps (surrounding steps)
{context_steps}

## Decision rules

### Mark as duplicate (is_duplicate: true) ONLY in these two cases:
1. **Retry loop**: the operation clearly failed and the same operation is repeated. For example:
   - Entering a password fails, then the same or a different password is entered again
   - A button click has no effect, then the same button is clicked repeatedly
   - A command fails, then the same command is executed again
   - Several consecutive sleep/wait steps (meaningless idle waiting)

2. **Meaningless repeated click**: the same UI element is clicked multiple times in a row with no change in the interface and no effect from the click.

### The following MUST be kept; they are NOT duplicates (is_duplicate: false):
1. **Consecutive scrolling**: browsing a long page requires multiple scrolls; this is normal and must be kept no matter how many scrolls there are
2. **Consecutive operations on different targets**: same action type but different targets (e.g., clicking different buttons, selecting different files)
3. **Reasonable consecutive operations**: operations that must be repeated to accomplish the goal (e.g., selecting files row by row, deleting one by one)
4. **Similar operations but the interface changed substantively**: the screenshots show the page content actually changed after each operation

### Key rules:
- If the current step is the **first** attempt in a retry loop, set is_first_occurrence: true
- If the current step is a **subsequent retry** (not the first) in a retry loop, set is_first_occurrence: false
- **Consecutive scrolling is NEVER marked as a duplicate!**

## Based on the screenshots and context, output JSON:
```json
{{
  "is_duplicate": true/false,
  "is_first_occurrence": true/false,
  "reason": "Brief explanation of the decision"
}}
```

Note:
- Output ONLY the JSON, nothing else
- is_first_occurrence only matters when is_duplicate is true
- Use the interface changes in the screenshots to help decide
- Prefer under-marking to mis-marking; only mark clear retry loops and meaningless repeated clicks"""


def build_step_context(data_steps, current_idx, window=5):
    """Build textual context around the current step."""
    lines = []
    start = max(0, current_idx - window)
    end = min(len(data_steps), current_idx + window + 1)

    for i in range(start, end):
        step = data_steps[i]
        marker = " <<< CURRENT STEP" if i == current_idx else ""
        thought_snippet = (step.get("thought", "") or "")[:150]
        lines.append(
            f"Step {step['step']}{marker}:\n"
            f"  Action: {step.get('action', '')}\n"
            f"  Code: {step.get('code', '')}\n"
            f"  Thought: {thought_snippet}"
        )
    return "\n\n".join(lines)


def evaluate_step(traj_dir, data_steps, current_idx, query):
    """Evaluate whether a single step is a duplicate operation."""
    current_step = data_steps[current_idx]

    # Build textual context
    context_text = build_step_context(data_steps, current_idx, window=5)

    prompt_text = EVALUATION_PROMPT.format(
        current_step=current_step["step"],
        current_action=current_step.get("action", ""),
        current_code=current_step.get("code", ""),
        context_steps=context_text,
        query=query,
    )

    # Build a multi-image message
    user_content = [{"type": "text", "text": prompt_text}]

    # Collect 5 screenshots: 2 before + current + 2 after
    # screenshot naming: data[i]["screenshot"]
    screenshot_indices = []
    for offset in range(-2, 3):  # -2, -1, 0, +1, +2
        idx = current_idx + offset
        if 0 <= idx < len(data_steps):
            screenshot_indices.append(idx)

    for idx in screenshot_indices:
        screenshot_name = data_steps[idx].get("screenshot", "")
        if screenshot_name:
            img_path = os.path.join(traj_dir, screenshot_name)
            img_b64 = encode_image(img_path)
            if img_b64:
                marker = " (current step screenshot)" if idx == current_idx else ""
                user_content.append({
                    "type": "text",
                    "text": f"[Step {data_steps[idx]['step']}{marker} - {screenshot_name}]"
                })
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                })

    messages = [{"role": "user", "content": user_content}]
    resp = call_gemini(messages, max_tokens=2048, temperature=0.2)
    result = parse_json_response(resp)

    if result is None:
        return {"is_duplicate": False, "is_first_occurrence": False, "reason": "API parse failure"}

    return result


def dedup_trajectory(traj_dir, dry_run=False, verbose=False):
    """Dedup one trajectory in place; returns (episode_id, num_marked, total_steps, details)."""
    episode_id = os.path.basename(traj_dir)
    task_json_path = os.path.join(traj_dir, "task.json")

    if not os.path.isfile(task_json_path):
        return episode_id, 0, 0, "task.json not found"

    with open(task_json_path, "r", encoding="utf-8") as f:
        task_data = json.load(f)

    data_steps = task_data.get("data", [])
    query = task_data.get("query", "")
    total_steps = len(data_steps)

    if total_steps <= 2:
        return episode_id, 0, total_steps, "too few steps"

    marked_steps = []
    details = []

    # Iterate over each step (skip the first and the last)
    for idx in range(1, total_steps - 1):
        step = data_steps[idx]

        # Skip if already marked is_use: false
        if not step.get("is_use", True):
            continue

        result = evaluate_step(traj_dir, data_steps, idx, query)

        is_dup = result.get("is_duplicate", False)
        is_first = result.get("is_first_occurrence", False)
        reason = result.get("reason", "")

        # Mark false only when "duplicate AND not the first occurrence"
        if is_dup and not is_first:
            marked_steps.append(idx)
            data_steps[idx]["is_use"] = False
            details.append(f"  Step {step['step']}: MARKED (reason: {reason})")
            if verbose:
                print(f"    [{episode_id}] Step {step['step']}: DUPLICATE -> is_use=false | {reason}", flush=True)
        else:
            if verbose and is_dup and is_first:
                print(f"    [{episode_id}] Step {step['step']}: FIRST_OCCURRENCE (kept) | {reason}", flush=True)

    # Save changes
    num_marked = len(marked_steps)
    if num_marked > 0 and not dry_run:
        with open(task_json_path, "w", encoding="utf-8") as f:
            json.dump(task_data, f, ensure_ascii=False, indent=4)

    detail_text = f"marked {num_marked}/{total_steps} steps"
    if details:
        detail_text += "\n" + "\n".join(details)

    return episode_id, num_marked, total_steps, detail_text


# ======================== Phase A: clean ========================

def run_clean_phase(args, input_dirs, output_dirs):
    """Run the clean phase (raw traj -> task.json). Returns the list of output roots used."""
    import random

    if args.dry_run:
        print("=== DRY RUN ===\n", flush=True)

    for input_root in input_dirs:
        if not os.path.isdir(input_root):
            print(f"[ERROR] input directory does not exist: {input_root}", flush=True)
            sys.exit(1)

    if not args.dry_run:
        for output_root in output_dirs:
            os.makedirs(output_root, exist_ok=True)

    # Collect all trajectories: (traj_dir, app_name, output_root)
    all_trajs = []
    for input_root, output_root in zip(input_dirs, output_dirs):
        print(f">>> Scanning input directory: {input_root} -> {output_root}", flush=True)
        for app_name in sorted(os.listdir(input_root)):
            app_dir = os.path.join(input_root, app_name)
            if not os.path.isdir(app_dir):
                continue
            for episode_id in sorted(os.listdir(app_dir)):
                traj_dir = os.path.join(app_dir, episode_id)
                if not os.path.isdir(traj_dir):
                    continue
                all_trajs.append((traj_dir, app_name, output_root))

    print(f">>> Total trajectories found: {len(all_trajs)}", flush=True)

    if args.num_sample > 0 and args.num_sample < len(all_trajs):
        random.seed(42)
        all_trajs = random.sample(all_trajs, args.num_sample)
        print(f">>> Randomly sampled {args.num_sample} trajectories", flush=True)

    total = len(all_trajs)
    print(f">>> Processing this run: {total}", flush=True)

    if not all_trajs:
        print(">>> No trajectories to process.", flush=True)
        return

    # Statistics
    stage1_fail = {"step_not_sequential": 0, "last_not_done": 0, "action_invalid": 0,
                   "screenshot_missing": 0, "too_many_steps": 0, "other": 0}
    stage2_fail = 0
    stage2_error = 0
    stage3_fail = 0
    stage3_error = 0
    success = 0

    def categorize_stage1(reason):
        if "too many steps" in reason:
            stage1_fail["too_many_steps"] += 1
        elif "step_num not sequential" in reason:
            stage1_fail["step_not_sequential"] += 1
        elif "last step not DONE" in reason:
            stage1_fail["last_not_done"] += 1
        elif "unmappable actions" in reason:
            stage1_fail["action_invalid"] += 1
        elif "screenshot" in reason.lower():
            stage1_fail["screenshot_missing"] += 1
        else:
            stage1_fail["other"] += 1

    print(f">>> Processing with {args.workers} workers...", flush=True)
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one_trajectory, traj_dir, app_name, out_root, args.dry_run, args.skip_gemini): (traj_dir, app_name)
            for traj_dir, app_name, out_root in all_trajs
        }
        for future in as_completed(futures):
            done += 1
            try:
                episode_id, stage_failed, reason = future.result()
                if stage_failed is None:
                    success += 1
                    status = "OK"
                elif stage_failed == 1:
                    categorize_stage1(reason)
                    status = "FAIL_S1"
                elif stage_failed == 2:
                    if "API error" in reason:
                        stage2_error += 1
                        status = "ERR_S2"
                    else:
                        stage2_fail += 1
                        status = "FAIL_S2"
                elif stage_failed == 3:
                    if "API error" in reason:
                        stage3_error += 1
                        status = "ERR_S3"
                    else:
                        stage3_fail += 1
                        status = "FAIL_S3"
                else:
                    status = "???"

                if done % 20 == 0 or done == total:
                    print(f"  [{done}/{total}] {status} {episode_id[:12]}... | {reason[:60]}", flush=True)

            except Exception as e:
                print(f"  [{done}/{total}] EXCEPTION {str(e)[:80]}", flush=True)

    # ============ Statistics output ============
    stage1_total_fail = sum(stage1_fail.values())
    stage1_pass = total - stage1_total_fail
    stage2_pass = stage1_pass - stage2_fail - stage2_error if not args.skip_gemini else stage1_pass
    stage3_pass = stage2_pass - stage3_fail - stage3_error if not args.skip_gemini else stage2_pass

    print(f"\n{'=' * 60}", flush=True)
    print(f"{'Stage 1: Basic clean':^56}", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Total trajectories:      {total}", flush=True)
    print(f"  Passed:                  {stage1_pass} ({stage1_pass/total*100:.1f}%)" if total > 0 else "", flush=True)
    print(f"  Dropped - >40 steps:     {stage1_fail['too_many_steps']}", flush=True)
    print(f"  Dropped - non-seq steps: {stage1_fail['step_not_sequential']}", flush=True)
    print(f"  Dropped - last not DONE: {stage1_fail['last_not_done']}", flush=True)
    print(f"  Dropped - bad action:    {stage1_fail['action_invalid']}", flush=True)
    print(f"  Dropped - missing shot:  {stage1_fail['screenshot_missing']}", flush=True)
    print(f"  Dropped - other:         {stage1_fail['other']}", flush=True)

    if not args.skip_gemini:
        print(f"\n{'=' * 60}", flush=True)
        print(f"{'Stage 2: Precondition check (Gemini)':^56}", flush=True)
        print(f"{'=' * 60}", flush=True)
        print(f"  To check:                {stage1_pass}", flush=True)
        print(f"  Passed:                  {stage2_pass} ({stage2_pass/stage1_pass*100:.1f}%)" if stage1_pass > 0 else "", flush=True)
        print(f"  Dropped - precond unmet: {stage2_fail}", flush=True)
        print(f"  Error - API failure:     {stage2_error}", flush=True)

        print(f"\n{'=' * 60}", flush=True)
        print(f"{'Stage 3: Task-completion eval (Gemini)':^56}", flush=True)
        print(f"{'=' * 60}", flush=True)
        print(f"  To evaluate:             {stage2_pass}", flush=True)
        print(f"  Passed(>={PASS_THRESHOLD}):           {success} ({success/stage2_pass*100:.1f}%)" if stage2_pass > 0 else "", flush=True)
        print(f"  Dropped - incomplete:    {stage3_fail}", flush=True)
        print(f"  Error - API failure:     {stage3_error}", flush=True)

    print(f"\n{'=' * 60}", flush=True)
    print(f"{'Clean phase result':^56}", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Original total:          {total}", flush=True)
    print(f"  Finally saved:           {success} ({success/total*100:.1f}%)" if total > 0 else "", flush=True)
    if not args.dry_run:
        print(f"  Output directories:      {', '.join(output_dirs)}", flush=True)


# ======================== Phase B: dedup ========================

def run_dedup_phase(args, output_dirs):
    """Run the dedup phase over the task.json folders produced by the clean phase."""
    print(f"\n{'#' * 60}", flush=True)
    print(f"{'PHASE B: Dedup':^60}", flush=True)
    print(f"{'#' * 60}", flush=True)

    if args.dry_run:
        print("=== DRY RUN (no files modified) ===\n", flush=True)

    # Collect all trajectory dirs (each containing task.json) across output roots
    all_trajs = []
    for output_root in output_dirs:
        if not os.path.isdir(output_root):
            print(f"  [skip] output directory does not exist: {output_root}", flush=True)
            continue
        for name in sorted(os.listdir(output_root)):
            traj_path = os.path.join(output_root, name)
            if os.path.isdir(traj_path) and os.path.isfile(os.path.join(traj_path, "task.json")):
                all_trajs.append(traj_path)

    total_trajs = len(all_trajs)
    print(f">>> Trajectories found: {total_trajs}", flush=True)
    if total_trajs == 0:
        print(">>> Nothing to dedup.", flush=True)
        return
    print(f">>> Processing with {args.workers} workers...\n", flush=True)

    # Statistics
    total_marked = 0
    total_steps_all = 0
    trajs_with_marks = 0
    results_summary = []

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(dedup_trajectory, traj_dir, args.dry_run, args.verbose): traj_dir
            for traj_dir in all_trajs
        }
        for future in as_completed(futures):
            done += 1
            try:
                episode_id, num_marked, total_steps, detail = future.result()
                total_marked += num_marked
                total_steps_all += total_steps
                if num_marked > 0:
                    trajs_with_marks += 1
                    results_summary.append((episode_id, num_marked, total_steps, detail))

                if done % 10 == 0 or done == total_trajs:
                    print(f"  [{done}/{total_trajs}] {episode_id[:12]}... | {num_marked} marked / {total_steps} steps", flush=True)

            except Exception as e:
                print(f"  [{done}/{total_trajs}] EXCEPTION: {str(e)[:100]}", flush=True)

    # ============ Final statistics ============
    print(f"\n{'=' * 60}", flush=True)
    print(f"{'Dedup result':^56}", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Total trajectories:      {total_trajs}", flush=True)
    print(f"  Total steps:             {total_steps_all}", flush=True)
    print(f"  Marked is_use=false:     {total_marked} steps ({total_marked/total_steps_all*100:.1f}%)" if total_steps_all > 0 else "", flush=True)
    print(f"  Affected trajectories:   {trajs_with_marks} ({trajs_with_marks/total_trajs*100:.1f}%)" if total_trajs > 0 else "", flush=True)

    if results_summary:
        print(f"\n{'=' * 60}", flush=True)
        print(f"{'Affected trajectory details':^56}", flush=True)
        print(f"{'=' * 60}", flush=True)
        results_summary.sort(key=lambda x: x[1], reverse=True)
        for episode_id, num_marked, total_steps, detail in results_summary:
            print(f"\n  [{episode_id}] {num_marked}/{total_steps} steps marked", flush=True)
            for line in detail.split("\n"):
                if line.strip().startswith("Step"):
                    print(f"    {line.strip()}", flush=True)

    if args.dry_run:
        print(f"\n>>> DRY RUN complete, no files modified.", flush=True)
    else:
        print(f"\n>>> Dedup complete, task.json files updated.", flush=True)


# ======================== Main ========================

def main():
    args = parse_args()
    input_dirs = args.input_dir
    output_dirs = args.output_dir

    if len(input_dirs) != len(output_dirs):
        print(f"[ERROR] --input-dir and --output-dir counts must match: "
              f"{len(input_dirs)} inputs, {len(output_dirs)} outputs", flush=True)
        sys.exit(1)

    global PASS_THRESHOLD
    PASS_THRESHOLD = args.threshold

    # Phase A: clean (raw traj -> task.json)
    run_clean_phase(args, input_dirs, output_dirs)

    # Phase B: dedup (operates on the produced task.json), unless skipped
    if args.skip_dedup:
        print("\n>>> --skip-dedup set, skipping dedup phase.", flush=True)
    elif args.skip_gemini:
        # Dedup itself relies on Gemini; skip it when Gemini is disabled.
        print("\n>>> --skip-gemini set, skipping dedup phase (it requires Gemini).", flush=True)
    else:
        run_dedup_phase(args, output_dirs)


if __name__ == "__main__":
    main()
