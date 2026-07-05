#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extraction + cleaning + conversion pipeline for MobileWorld non-rephrase variant trajectories.

Input:  {input_root}/<TaskName_vN_suffix[_backup_TS]>/  (traj.json + result.txt + screenshots/)
Output: {output_base}/<suffix>/<episode_id>/

4 stages:
  1. Basic clean (local)
  2. Gemini precondition check
  3. Gemini sub-task completion evaluation (left-to-right, three states: completed/skipped/failed)
  4. Convert to qwen3-vl task.json

Model API config is read from environment variables (see the bash wrapper):
    MODEL_URL, MODEL_NAME, MODEL_PROVIDER_ID, GEMINI_API_KEY

Usage:
    python clean_trajectories.py --dry-run --input-dir /path/to/in --output-dir /path/to/out
    python clean_trajectories.py --dry-up 20 --input-dir /path/to/in --output-dir /path/to/out
    python clean_trajectories.py --skip-gemini --input-dir /path/to/in --output-dir /path/to/out
    python clean_trajectories.py --workers 100 --input-dir /path/to/in --output-dir /path/to/out
    python clean_trajectories.py --threshold 70 --input-dir /path/to/in --output-dir /path/to/out
    python clean_trajectories.py --only change_action --input-dir /path/to/in --output-dir /path/to/out
"""

import argparse
import base64
import json
import os
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from github.uni_gui.mobile.app_map import app_for_folder
from github.uni_gui.mobile.gemini_scroll_resolver import resolve_scroll, RETRIES as SCROLL_RETRIES

# ======================== Config ========================
# Input/output roots are set from --input-dir / --output-dir at runtime.
INPUT_ROOT = "/path/to/mobileworld/variants_traj_logs"
OUTPUT_BASE = "/path/to/output/dataset/variants"
SCREEN_RESOLUTION = [1080, 2400]
REPHRASE_SUFFIXES = ("_rephrase_1", "_rephrase_2", "_rephrase_3")

# ---- Model API config (read from environment, with placeholder defaults) ----
MODEL_URL = os.environ.get("MODEL_URL", "https://your-model-endpoint/v1/chat/completions")
MODEL_NAME = os.environ.get("MODEL_NAME", "your-model-name")
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
MODEL_PROVIDER_ID = os.environ.get("MODEL_PROVIDER_ID", "your-provider-id")
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "X-Model-Provider-Id": MODEL_PROVIDER_ID,
    "Content-Type": "application/json",
}

DEFAULT_WORKERS = 100
DEFAULT_RETRIES = 50
PASS_THRESHOLD = 70
MAX_STEP_IMAGES = 3

VALID_ACTION_TYPES = {
    "click", "long_press", "input_text", "scroll", "drag",
    "navigate_back", "navigate_home", "keyboard_enter",
    "wait", "status", "answer", "ask_user",
}

ACTION_RE = re.compile(r"Action:\s*(\{.*\})\s*$", re.DOTALL)
THOUGHT_RE = re.compile(r"Thought:\s*(.*?)\s*Action:", re.DOTALL)


# ======================== Utility functions ========================

def encode_image(image_path):
    if not os.path.isfile(image_path):
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_gemini(messages, max_retries=DEFAULT_RETRIES, max_tokens=4096):
    last_error = ""
    for attempt in range(max_retries):
        try:
            payload = {
                "model": MODEL_NAME,
                "messages": messages,
                "stream": False,
                "temperature": 1,
                "max_tokens": max_tokens,
            }
            headers = dict(HEADERS)
            headers["X-Model-Request-Id"] = f"clean-variant-{time.time()}"
            resp = requests.post(MODEL_URL, headers=headers, json=payload, timeout=300)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"].get("content", "").strip()
                if content:
                    return content
                reasoning = data["choices"][0]["message"].get("reasoning_content", "").strip()
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
    return None


def parse_json_response(text):
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass
    m = re.search(r'\{[^{}]*"', cleaned, re.DOTALL)
    if m:
        start = m.start()
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == '{':
                depth += 1
            elif cleaned[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(cleaned[start:i+1])
                    except (json.JSONDecodeError, ValueError):
                        pass
                    break
    return None


# ======================== Folder enumeration + backup de-dup ========================

def strip_backup(name):
    return re.sub(r'_backup_\d+_\d+$', '', name)


def backup_ts(name):
    m = re.search(r'_backup_(\d+_\d+)$', name)
    return m.group(1) if m else ""


def extract_suffix(folder_name):
    """Extract the suffix from TaskName_vN_<suffix> or TaskName_vN_<suffix>_backup_TS."""
    base = strip_backup(folder_name)
    m = re.match(r'^.+?_v\d+_(.+)$', base)
    if m:
        return m.group(1)
    return None


def enumerate_folders():
    """Return the de-duplicated [(folder_name, suffix, episode_id)] list."""
    all_dirs = sorted(os.listdir(INPUT_ROOT))

    non_rephrase = []
    for d in all_dirs:
        if any(d.endswith(s) or (strip_backup(d)).endswith(s) for s in REPHRASE_SUFFIXES):
            continue
        full = os.path.join(INPUT_ROOT, d)
        if os.path.isdir(full):
            non_rephrase.append(d)

    groups = defaultdict(list)
    for d in non_rephrase:
        base = strip_backup(d)
        groups[base].append(d)

    selected = []
    for base, members in groups.items():
        backups = [m for m in members if backup_ts(m)]
        if backups:
            chosen = max(backups, key=backup_ts)
        else:
            chosen = members[0]

        suffix = extract_suffix(chosen)
        if suffix is None:
            continue
        episode_id = base
        selected.append((chosen, suffix, episode_id))

    return selected


# ======================== prediction parse + action mapping ========================

def parse_prediction(pred):
    thought = ""
    mt = THOUGHT_RE.search(pred)
    if mt:
        thought = mt.group(1).strip()
    ma = ACTION_RE.search(pred)
    if not ma:
        ma = re.search(r"(\{.*\})", pred, re.DOTALL)
        if not ma:
            return thought, None, ""
    raw = ma.group(1).strip()
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return thought, None, raw
    return thought, obj, raw


def map_action(act):
    at = act.get("action_type")
    if at == "click":
        return {"action": "click", "coordinate": act.get("coordinate")}, False, None
    if at == "long_press":
        return {"action": "long_press", "coordinate": act.get("coordinate"), "time": 1}, False, None
    if at == "input_text":
        return {"action": "type", "text": act.get("text", "")}, False, None
    if at == "answer":
        return {"action": "answer", "text": act.get("text", "")}, False, None
    if at == "wait":
        return {"action": "wait", "time": 1}, False, None
    if at == "navigate_back":
        return {"action": "system_button", "button": "Back"}, False, None
    if at == "navigate_home":
        return {"action": "system_button", "button": "Home"}, False, None
    if at == "keyboard_enter":
        return {"action": "system_button", "button": "Enter"}, False, None
    if at == "ask_user":
        return {"action": "ask_user", "text": act.get("text", "")}, False, None
    if at == "status":
        gs = act.get("goal_status")
        status = "success" if gs == "complete" else "failure"
        return {"action": "terminate", "status": status}, False, None
    if at == "drag":
        return {"action": "swipe",
                "coordinate": act.get("start_coordinate"),
                "coordinate2": act.get("end_coordinate")}, False, None
    if at == "scroll":
        return {"action": "swipe"}, True, (act.get("direction") or "down").lower()
    return None, False, None


def generate_action_desc(act_obj):
    at = act_obj.get("action_type", "")
    coord = act_obj.get("coordinate")
    text = act_obj.get("text", "")
    direction = act_obj.get("direction", "")
    if at == "click" and coord:
        return f"Click at ({coord[0]}, {coord[1]})."
    if at == "long_press" and coord:
        return f"Long press at ({coord[0]}, {coord[1]})."
    if at == "scroll":
        return f"Scroll {direction}."
    if at == "input_text":
        return f"Type: {text[:50]}"
    if at == "navigate_back":
        return "Navigate back."
    if at == "navigate_home":
        return "Navigate to home screen."
    if at == "keyboard_enter":
        return "Press enter."
    if at == "wait":
        return "Wait for screen to update."
    if at == "status":
        return f"Task {act_obj.get('goal_status', 'complete')}."
    if at == "answer":
        return f"Answer: {text[:50]}"
    if at == "drag":
        return f"Drag from {act_obj.get('start_coordinate')} to {act_obj.get('end_coordinate')}."
    if at == "ask_user":
        return f"Ask user: {text[:50]}"
    return f"Perform {at}."


def list_screenshots(traj_dir):
    sdir = os.path.join(traj_dir, "screenshots")
    out = {}
    if not os.path.isdir(sdir):
        return out
    for fn in os.listdir(sdir):
        m = re.search(r"-(\d+)-(\d+)\.png$", fn)
        if m:
            out[int(m.group(2))] = fn
    return out


# ======================== Stage 1: basic clean ========================

def basic_clean(folder_name, episode_id):
    """Return (steps, query, shots_dict, reason). A non-empty reason means discard."""
    traj_dir = os.path.join(INPUT_ROOT, folder_name)

    tp = os.path.join(traj_dir, "traj.json")
    if not os.path.isfile(tp):
        return None, None, None, "no_traj_json"
    try:
        data = json.load(open(tp, encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, None, None, "traj_json_parse_error"

    steps = []
    for _, v in data.items():
        steps = v.get("traj", [])
        break
    if not steps:
        return None, None, None, "empty_traj"

    if len(steps) > 40:
        return None, None, None, "too_many_steps"

    ids = [st.get("step") for st in steps]
    if ids != list(range(1, len(steps) + 1)):
        return None, None, None, "step_not_sequential"

    # The last step must be a status action
    last_pred = steps[-1].get("prediction", "")
    _, last_act, _ = parse_prediction(last_pred)
    if last_act is None or last_act.get("action_type") != "status":
        return None, None, None, "last_not_status"

    # Every step's prediction must be parseable + have a valid action_type
    for st in steps:
        _, act_obj, _ = parse_prediction(st.get("prediction", ""))
        if act_obj is None:
            return None, None, None, "prediction_parse_failed"
        at = act_obj.get("action_type", "")
        if at not in VALID_ACTION_TYPES:
            return None, None, None, f"action_invalid:{at}"

    # Screenshot integrity
    shots = list_screenshots(traj_dir)
    shot_ids = sorted(shots.keys())
    if shot_ids != list(range(1, len(steps) + 1)):
        return None, None, None, "screenshot_missing"

    query = steps[0].get("task_goal", "")
    return steps, query, shots, None


# ======================== Stage 2: precondition check ========================

PROMPT_PRECONDITION = """You are evaluating whether a mobile task's preconditions are met based on the initial screenshots.

## Task instruction:
"{query}"

## What to check:
Look at the provided screenshots (the initial state of the mobile device before the agent starts working). Determine if the environment has the necessary preconditions for this task:
- If the task mentions modifying a specific setting/contact/event/message, does that item exist?
- If the task requires a specific app or page to be accessible, is it accessible?
- If the task references specific data/content that should already exist, does it appear to exist?

## Important:
- If the task is about changing a SETTING or navigating to something (no precondition needed), answer true.
- Only answer false if there is CLEAR evidence that a required precondition is NOT met.
- If the task asks to modify a variable/item that does NOT exist in the visible environment (e.g. "increase contact Alice's phone" but Alice not present, "modify event X" but X doesn't exist), answer false.
- When in doubt, answer true (give benefit of the doubt).

Output ONLY JSON:
{{
  "precondition_met": true/false,
  "reason": "Brief explanation"
}}"""


def check_precondition(folder_name, steps, query, shots):
    traj_dir = os.path.join(INPUT_ROOT, folder_name)
    sdir = os.path.join(traj_dir, "screenshots")

    screenshot_paths = []
    for k in range(1, min(4, len(steps) + 1)):
        if k in shots:
            path = os.path.join(sdir, shots[k])
            if os.path.isfile(path):
                screenshot_paths.append(path)

    prompt_text = PROMPT_PRECONDITION.format(query=query)
    user_content = [{"type": "text", "text": prompt_text}]
    for img_path in screenshot_paths:
        img_b64 = encode_image(img_path)
        if img_b64:
            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})

    messages = [{"role": "user", "content": user_content}]
    resp = call_gemini(messages)
    result = parse_json_response(resp)
    if result is None:
        return None, "API failure"
    return result.get("precondition_met", True), result.get("reason", "")


# ======================== Stage 3: sub-task evaluation ========================

PROMPT_DECOMPOSE = """You are a task decomposition expert. Break down the following mobile task into sequential, verifiable sub-tasks.

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

PROMPT_EVALUATE_SUBTASK = """You are a mobile UI trajectory evaluator. Evaluate whether this sub-task was achieved by the agent's trajectory.

## Context
Full user instruction: "{query}"
All sub-tasks: {all_subtasks_json}
Current sub-task being evaluated (index {subtask_idx}): "{sub_task}"

## Previous sub-tasks evaluation results:
{prev_results_text}

## Agent's action history (full trajectory):
{actions_text}

## Evaluation Rules:

1. **Three possible statuses:**
   - `completed`: The sub-task was clearly achieved by the trajectory (score >= {threshold}).
   - `skipped`: The sub-task is genuinely unnecessary in this trajectory's context. E.g. the initial decomposition was too fine-grained, or a previous step already achieved this goal indirectly, or the task can be completed without this step. Do NOT mark as skipped if the sub-task SHOULD have been done but wasn't.
   - `failed`: The sub-task SHOULD have been completed but was NOT achieved by the trajectory.

2. **Action history is primary evidence.** If actions clearly show the agent performed the correct operations, consider it completed.

3. **Screenshots are supplementary.** Use screenshots to confirm or deny.

4. **Focus on the core goal.** Minor inefficiencies don't matter.

## FAIL conditions (score below 50):
- Agent clearly did something UNRELATED to this sub-task when it should have done it
- Agent got stuck in a loop (same action 4+ times)
- The sub-task requires a specific value but the agent never typed/selected that value

## Scoring (only meaningful for completed/failed):
- 90-100: Clearly completed with evidence
- 70-89: Very likely completed based on actions
- 50-69: Unclear
- 0-49: Clearly NOT completed

Output ONLY JSON:
{{
  "score": <0-100>,
  "status": "completed|skipped|failed",
  "evidence": "Brief explanation citing specific actions or screenshots",
  "reason": "Why this status (especially important for skipped/failed)"
}}"""


def decompose_query(query):
    prompt = PROMPT_DECOMPOSE.format(query=query)
    messages = [{"role": "user", "content": prompt}]
    resp = call_gemini(messages)
    data = parse_json_response(resp)
    if data and "sub_tasks" in data and isinstance(data["sub_tasks"], list) and len(data["sub_tasks"]) > 0:
        return data["sub_tasks"]
    return [query]


def evaluate_task_completion(folder_name, steps, query, shots, threshold):
    """Evaluate sub-tasks left-to-right. Return (passed, details_list, fail_reason)."""
    traj_dir = os.path.join(INPUT_ROOT, folder_name)
    sdir = os.path.join(traj_dir, "screenshots")

    sub_tasks = decompose_query(query)
    if sub_tasks is None:
        return None, [], "decompose_api_failure"

    # action history
    all_actions = []
    for st in steps:
        thought, act_obj, _ = parse_prediction(st.get("prediction", ""))
        desc = generate_action_desc(act_obj) if act_obj else "unknown"
        all_actions.append(f"Step {st['step']}: {desc}")
    actions_text = "\n".join(all_actions)

    # screenshots: first + last MAX_STEP_IMAGES
    screenshot_paths = []
    if 1 in shots:
        screenshot_paths.append(os.path.join(sdir, shots[1]))
    img_start = max(1, len(steps) - MAX_STEP_IMAGES + 1)
    for k in range(img_start, len(steps) + 1):
        if k in shots:
            path = os.path.join(sdir, shots[k])
            if path not in screenshot_paths:
                screenshot_paths.append(path)

    prior_results = []
    for st_idx, sub_task in enumerate(sub_tasks):
        if prior_results:
            prev_text = "\n".join([
                f"  {i+1}. \"{r['sub_task']}\" -> {r['status'].upper()}"
                + (f" (evidence: {r['evidence'][:80]})" if r.get('evidence') else "")
                for i, r in enumerate(prior_results)
            ])
        else:
            prev_text = "  (This is the first sub-task)"

        eval_text = PROMPT_EVALUATE_SUBTASK.format(
            query=query,
            all_subtasks_json=json.dumps(sub_tasks, ensure_ascii=False),
            subtask_idx=st_idx + 1,
            sub_task=sub_task,
            prev_results_text=prev_text,
            actions_text=actions_text,
            threshold=threshold,
        )

        user_content = [{"type": "text", "text": eval_text}]
        for img_path in screenshot_paths:
            img_b64 = encode_image(img_path)
            if img_b64:
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})

        messages = [{"role": "user", "content": user_content}]
        resp = call_gemini(messages)
        result = parse_json_response(resp)

        if result is None:
            return None, prior_results, f"subtask_{st_idx+1}_api_failure"

        status = result.get("status", "failed")
        if status not in ("completed", "skipped", "failed"):
            status = "failed" if result.get("score", 0) < threshold else "completed"

        prior_results.append({
            "sub_task": sub_task,
            "status": status,
            "score": result.get("score", 0),
            "evidence": result.get("evidence", ""),
            "reason": result.get("reason", ""),
        })

        if status == "failed":
            return False, prior_results, f"subtask_{st_idx+1}_failed"

    return True, prior_results, None


# ======================== Stage 4: convert to task.json ========================

def convert_to_taskjson(folder_name, episode_id, suffix, steps, query, shots, dry_run=False):
    """Convert and write to disk. Return (task_json, fallback_steps)."""
    traj_dir = os.path.join(INPUT_ROOT, folder_name)
    sdir = os.path.join(traj_dir, "screenshots")
    fallback_steps = []

    app = app_for_folder(episode_id)
    if app is None:
        app = app_for_folder(folder_name)
    if app is None:
        app = "unknown"

    parsed = []
    for st in steps:
        k = st["step"]
        thought, act_obj, act_raw = parse_prediction(st.get("prediction", ""))
        if act_obj is None:
            return None, fallback_steps
        args, need_scroll, direction = map_action(act_obj)
        if args is None and not need_scroll:
            return None, fallback_steps
        parsed.append((k, thought, act_obj, act_raw, args, need_scroll, direction))

    data_steps = []
    for (k, thought, act_obj, act_raw, args, need_scroll, direction) in parsed:
        is_reviewed = False
        if need_scroll:
            if dry_run:
                from github.uni_gui.mobile.gemini_scroll_resolver import GEOMETRIC_FALLBACK
                c1, c2 = GEOMETRIC_FALLBACK.get(direction, GEOMETRIC_FALLBACK["down"])
                used_fallback = False
            else:
                img_path = os.path.join(sdir, shots[k])
                c1, c2, used_fallback = resolve_scroll(img_path, direction, retries=SCROLL_RETRIES)
            args = {"action": "swipe", "coordinate": c1, "coordinate2": c2}
            if used_fallback:
                is_reviewed = True
                fallback_steps.append((episode_id, k))

        step_thought = thought if thought else generate_action_desc(act_obj)
        data_steps.append({
            "step": k,
            "query": query,
            "thought": step_thought,
            "action": step_thought,
            "pixel": SCREEN_RESOLUTION,
            "plan": {"name": "mobile_use", "arguments": args},
            "bbox": [],
            "screenshot": f"screenshot_step{k - 1}.png",
            "code": act_raw,
            "is_use": True,
            "is_reviewed": is_reviewed,
            "is_delete": False,
            "train_test": "test",
            "raw_thought": "",
        })

    task_json = {
        "task": "MobileWorld",
        "app": app,
        "screen_resolution": SCREEN_RESOLUTION,
        "query": query,
        "episode_id": episode_id,
        "is_delete": False,
        "is_mock": False,
        "device": "mobile",
        "verified": False,
        "task_completed": True,
        "data": data_steps,
    }

    if dry_run:
        return task_json, fallback_steps

    out_dir = os.path.join(OUTPUT_BASE, suffix, episode_id)
    os.makedirs(out_dir, exist_ok=True)
    for k in range(1, len(steps) + 1):
        src = os.path.join(sdir, shots[k])
        dst = os.path.join(out_dir, f"screenshot_step{k - 1}.png")
        if os.path.isfile(src):
            shutil.copy2(src, dst)
    with open(os.path.join(out_dir, "task.json"), "w", encoding="utf-8") as f:
        json.dump(task_json, f, ensure_ascii=False, indent=4)

    return task_json, fallback_steps


# ======================== Single-trajectory processing ========================

def process_one(folder_name, suffix, episode_id, dry_run, skip_gemini, threshold):
    """Return (episode_id, stage_failed, reason, details)."""
    # Stage 1
    steps, query, shots, reason = basic_clean(folder_name, episode_id)
    if reason:
        return episode_id, 1, reason, None

    if skip_gemini:
        task_json, fallback = convert_to_taskjson(folder_name, episode_id, suffix, steps, query, shots, dry_run)
        return episode_id, None, f"OK(skip_gemini, fallback={len(fallback)})", None

    # Stage 2
    precondition_met, reason = check_precondition(folder_name, steps, query, shots)
    if precondition_met is None:
        return episode_id, 2, f"API_error: {reason}", None
    if not precondition_met:
        return episode_id, 2, reason, None

    # Stage 3
    passed, details, fail_reason = evaluate_task_completion(folder_name, steps, query, shots, threshold)
    if passed is None:
        return episode_id, 3, f"API_error: {fail_reason}", details
    if not passed:
        return episode_id, 3, fail_reason, details

    # Stage 4
    task_json, fallback = convert_to_taskjson(folder_name, episode_id, suffix, steps, query, shots, dry_run)
    if task_json is None:
        return episode_id, 4, "convert_failed", details
    return episode_id, None, f"OK(fallback={len(fallback)})", details


# ======================== dry-up mode ========================

def run_dry_up(folders, num_sample, threshold, workers):
    """Randomly sample N trajectories, run the full Stage 1+2+3 (no disk writes), print pass rate."""
    sample_size = min(num_sample, len(folders))
    rng = random.Random(42)
    sampled_folders = rng.sample(folders, sample_size)

    print(f"\n{'='*60}")
    print(f"  [DRY-UP] Randomly sampled {sample_size} of {len(folders)} (seed=42)")
    print(f"{'='*60}\n")

    # Stage 1
    stage1_pass = []
    stage1_fail_counts = defaultdict(int)
    for folder_name, suffix, episode_id in sampled_folders:
        steps, query, shots, reason = basic_clean(folder_name, episode_id)
        if reason:
            stage1_fail_counts[reason.split(":")[0]] += 1
        else:
            stage1_pass.append((folder_name, suffix, episode_id, steps, query, shots))

    print(f"  Stage 1 passed: {len(stage1_pass)}/{sample_size}")
    for r, c in sorted(stage1_fail_counts.items(), key=lambda x: -x[1]):
        print(f"    discarded - {r}: {c}")

    if len(stage1_pass) == 0:
        print("  No trajectory passed Stage 1; cannot continue.")
        return

    sampled = stage1_pass
    print(f"\n  Running Stage 2+3 on {len(sampled)} Stage-1-passed trajectories...")

    results = []

    def _run_one(item):
        folder_name, suffix, episode_id, steps, query, shots = item
        # Stage 2
        precondition_met, reason = check_precondition(folder_name, steps, query, shots)
        if precondition_met is None:
            return (episode_id, "stage2_error", reason, None)
        if not precondition_met:
            return (episode_id, "stage2_fail", reason, None)
        # Stage 3
        passed, details, fail_reason = evaluate_task_completion(folder_name, steps, query, shots, threshold)
        if passed is None:
            return (episode_id, "stage3_error", fail_reason, details)
        if not passed:
            return (episode_id, "stage3_fail", fail_reason, details)
        return (episode_id, "PASS", None, details)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_one, item): item for item in sampled}
        for f in as_completed(futures):
            results.append(f.result())

    # Statistics
    n_evaluated = len(results)
    pass_count = sum(1 for r in results if r[1] == "PASS")
    stage2_fail = sum(1 for r in results if r[1] == "stage2_fail")
    stage2_error = sum(1 for r in results if r[1] == "stage2_error")
    stage3_fail = sum(1 for r in results if r[1] == "stage3_fail")
    stage3_error = sum(1 for r in results if r[1] == "stage3_error")

    print(f"\n{'='*60}")
    print(f"  [DRY-UP] Result (sampled {sample_size}, Stage 1 passed {n_evaluated}, PASS_THRESHOLD={threshold})")
    print(f"{'='*60}")
    print(f"  Stage 1 passed: {n_evaluated}/{sample_size}")
    print(f"  Stage 2 passed: {n_evaluated - stage2_fail - stage2_error}/{n_evaluated}")
    print(f"  Stage 3 passed: {pass_count}/{n_evaluated}")
    print(f"  End-to-end pass rate (of total sampled): {pass_count}/{sample_size} = {pass_count/sample_size*100:.1f}%")
    print(f"\n  Failure breakdown:")
    print(f"    stage2_fail (precondition unmet): {stage2_fail}")
    print(f"    stage2_error (API failure):       {stage2_error}")
    print(f"    stage3_fail (subtask failed):     {stage3_fail}")
    print(f"    stage3_error (API failure):       {stage3_error}")

    print(f"\n  Per-trajectory:")
    for episode_id, status, reason, details in sorted(results, key=lambda x: x[1]):
        if status == "PASS":
            detail_str = ""
            if details:
                completed = sum(1 for d in details if d["status"] == "completed")
                skipped = sum(1 for d in details if d["status"] == "skipped")
                detail_str = f" (subtasks: {completed} completed, {skipped} skipped)"
            print(f"    {episode_id[:50]:50s} PASS{detail_str}")
        else:
            reason_short = (reason or "")[:60]
            print(f"    {episode_id[:50]:50s} {status} - {reason_short}")
    print(f"{'='*60}\n")


# ======================== Main ========================

def main():
    global INPUT_ROOT, OUTPUT_BASE

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, default=INPUT_ROOT,
                        help="Input root containing <TaskName_vN_suffix[_backup_TS]> folders")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_BASE,
                        help="Output base; writes <suffix>/<episode_id>/")
    parser.add_argument("--dry-run", action="store_true", help="Stage 1 only, no gemini calls, no disk writes")
    parser.add_argument("--dry-up", type=int, default=0, help="Sample N trajectories through the full Stage 1+2+3 (no disk writes)")
    parser.add_argument("--skip-gemini", action="store_true", help="Skip Stage 2/3, run Stage 1+4 only")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--threshold", type=int, default=PASS_THRESHOLD)
    parser.add_argument("--only", type=str, default=None, help="Process only this suffix")
    parser.add_argument("--num-sample", type=int, default=-1, help="Random sample size")
    args = parser.parse_args()

    INPUT_ROOT = args.input_dir
    OUTPUT_BASE = args.output_dir

    if not os.path.isdir(INPUT_ROOT):
        print(f"[ERROR] Input directory does not exist: {INPUT_ROOT}")
        sys.exit(1)

    folders = enumerate_folders()
    print(f"Total trajectories after de-dup: {len(folders)}")

    if args.only:
        folders = [(f, s, e) for f, s, e in folders if s == args.only]
        print(f"Filtered --only={args.only}: {len(folders)}")

    if not folders:
        print("No trajectories to process.")
        sys.exit(0)

    if args.num_sample > 0 and args.num_sample < len(folders):
        rng = random.Random(42)
        folders = rng.sample(folders, args.num_sample)
        print(f"Random sample of {args.num_sample}")

    # dry-up mode
    if args.dry_up > 0:
        run_dry_up(folders, args.dry_up, args.threshold, args.workers)
        return

    # dry-run mode: Stage 1 only
    if args.dry_run:
        print(f"\n=== DRY RUN (Stage 1 only) ===\n")
        stage1_stats = defaultdict(int)
        pass_count = 0
        for folder_name, suffix, episode_id in folders:
            steps, query, shots, reason = basic_clean(folder_name, episode_id)
            if reason:
                stage1_stats[reason.split(":")[0]] += 1
            else:
                pass_count += 1

        total = len(folders)
        print(f"{'='*60}")
        print(f"{'Stage 1: basic clean':^56}")
        print(f"{'='*60}")
        print(f"  Total trajectories: {total}")
        print(f"  Passed:             {pass_count} ({pass_count/total*100:.1f}%)")
        for r, c in sorted(stage1_stats.items(), key=lambda x: -x[1]):
            print(f"  discarded - {r:25s}: {c}")
        print(f"{'='*60}")

        # suffix distribution
        suffix_counts = defaultdict(int)
        for _, s, _ in folders:
            suffix_counts[s] += 1
        print(f"\n  Suffix distribution:")
        for s, c in sorted(suffix_counts.items()):
            print(f"    {s}: {c}")
        return

    # Full run
    print(f"\n=== FULL RUN (workers={args.workers}, threshold={args.threshold}) ===\n")
    total = len(folders)

    stage1_stats = defaultdict(int)
    stage2_fail = 0
    stage2_error = 0
    stage3_fail = 0
    stage3_error = 0
    stage4_fail = 0
    success = 0
    done = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, f, s, e, False, args.skip_gemini, args.threshold): (f, s, e)
            for f, s, e in folders
        }
        for future in as_completed(futures):
            done += 1
            try:
                episode_id, stage_failed, reason, details = future.result()
                if stage_failed is None:
                    success += 1
                elif stage_failed == 1:
                    stage1_stats[reason.split(":")[0]] += 1
                elif stage_failed == 2:
                    if "API_error" in reason:
                        stage2_error += 1
                    else:
                        stage2_fail += 1
                elif stage_failed == 3:
                    if "API_error" in reason:
                        stage3_error += 1
                    else:
                        stage3_fail += 1
                elif stage_failed == 4:
                    stage4_fail += 1
            except Exception:
                stage1_stats["exception"] += 1

            if done % 10 == 0 or done == total:
                print(f"  [{done}/{total}] success={success} | "
                      f"S1drop={sum(stage1_stats.values())} S2drop={stage2_fail}+{stage2_error} "
                      f"S3drop={stage3_fail}+{stage3_error}", flush=True)

    # Statistics output
    stage1_total_fail = sum(stage1_stats.values())
    stage1_pass = total - stage1_total_fail

    print(f"\n{'='*60}")
    print(f"{'Stage 1: basic clean':^56}")
    print(f"{'='*60}")
    print(f"  Total trajectories: {total}")
    print(f"  Passed:             {stage1_pass} ({stage1_pass/total*100:.1f}%)" if total > 0 else "")
    for r, c in sorted(stage1_stats.items(), key=lambda x: -x[1]):
        print(f"  discarded - {r:25s}: {c}")

    if not args.skip_gemini:
        stage2_pass = stage1_pass - stage2_fail - stage2_error
        print(f"\n{'='*60}")
        print(f"{'Stage 2: precondition check (Gemini)':^56}")
        print(f"{'='*60}")
        print(f"  To check:           {stage1_pass}")
        print(f"  Passed:             {stage2_pass} ({stage2_pass/stage1_pass*100:.1f}%)" if stage1_pass > 0 else "")
        print(f"  discarded - precondition unmet: {stage2_fail}")
        print(f"  error - API failure:            {stage2_error}")

        print(f"\n{'='*60}")
        print(f"{'Stage 3: task completion eval (Gemini)':^56}")
        print(f"{'='*60}")
        print(f"  To evaluate:        {stage2_pass}")
        print(f"  Passed:             {success} ({success/stage2_pass*100:.1f}%)" if stage2_pass > 0 else "")
        print(f"  discarded - subtask incomplete: {stage3_fail}")
        print(f"  error - API failure:            {stage3_error}")

    print(f"\n{'='*60}")
    print(f"{'Final result':^56}")
    print(f"{'='*60}")
    print(f"  Original total:     {total}")
    print(f"  Finally saved:      {success} ({success/total*100:.1f}%)" if total > 0 else "")
    print(f"  Output directory:   {OUTPUT_BASE}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
