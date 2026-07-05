#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a new batch of query variants (15 per task) for MobileWorld tasks.

Uses the same rewriting strategy as the OSWorld generator:
  - 5 minor edits (change target, change action, add complexity, change params, rephrase)
  - 10 major edits (5 dimensions x 2 variants: major target/action/complexity/perspective, batch op)

All variants avoid duplicating queries already produced in a previous batch
(the CSV passed via --prev-csv).

Model API config is read from environment variables (see the bash wrapper):
    MODEL_URL, MODEL_NAME, MODEL_PROVIDER_ID, GEMINI_API_KEY

Usage:
    python -u generate_new_queries.py --traj-dir /path/to/traj --task-list-csv /path/to/tasks.csv \
        --csv-output /path/to/out.csv --cache-file /path/to/cache.json
    python -u generate_new_queries.py --workers 10 ...
    python -u generate_new_queries.py --dry-run ...    # test 2 tasks
"""

import argparse
import base64
import csv
import glob as _glob
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

# ======================== Config ========================
# ---- Model API config (read from environment, with placeholder defaults) ----
MODEL_URL = os.environ.get("MODEL_URL", "https://your-model-endpoint/v1/chat/completions")
MODEL_NAME = os.environ.get("MODEL_NAME", "your-model-name")
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
MODEL_PROVIDER_ID = os.environ.get("MODEL_PROVIDER_ID", "your-provider-id")

# Paths are set from CLI args at runtime.
TRAJ_DIR = "/path/to/dataset/trajectories"        # per-task folders (traj.json + screenshots/)
TASK_LIST_CSV = "/path/to/dataset/successful_tasks.csv"
PREV_CSV = "/path/to/dataset/query_variants_v1.csv"
CSV_OUTPUT = "/path/to/output/query_variants_v2.csv"
CACHE_FILE = "/path/to/output/generate_cache_v2.json"

MAX_RETRIES = 50
MAX_SCREENSHOTS = 5
# ========================================================

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "X-Model-Provider-Id": MODEL_PROVIDER_ID,
    "Content-Type": "application/json",
}

# ==================== Minor edit dimensions ====================
DIMENSIONS_MINOR = [
    {
        "name": "change_target",
        "instruction": """Change the TARGET OBJECT of the operation while keeping the action type the same.
For example: if original operates on "Daniel's email", the new query could operate on "the email from Amazon" or "the latest unread email".
The new query MUST use a completely different sentence structure and wording — do NOT just swap the object name in the original text.
ONLY reference targets that are actually VISIBLE in the provided screenshots or plausible in the app environment.""",
    },
    {
        "name": "change_action",
        "instruction": """Change the ACTION/OPERATION while keeping the target object similar.
For example: if original "replies to an email", the new query could "forward the email" or "star the email" or "delete the email".
The new query MUST be written in a different style — as if a different person wrote it from scratch.
The action must be something the application shown in the screenshots actually supports.""",
    },
    {
        "name": "adjust_complexity",
        "instruction": """Make the task slightly MORE complex by adding one additional sub-step or requirement.
For example: if original just sends a reply, the new query could send the reply AND then archive the conversation.
Write it naturally as a single coherent request — not as a list of steps. Use a fresh writing style.
All referenced elements must exist in the current app environment as shown in the screenshots.""",
    },
    {
        "name": "change_params",
        "instruction": """Keep the same type of operation but change the specific PARAMETERS or VALUES.
For example: if original sets "alarm at 7:00 AM", the new one could set "alarm at 6:30 AM". If original sends "I'll be there at 10", new could send "Count me in for the 3 PM session".
IMPORTANT: Do NOT just find-and-replace the value in the original text. Rewrite the entire query from scratch in a completely different style.
The parameters you choose must be valid for the app shown in the screenshots.""",
    },
    {
        "name": "rephrase_and_tweak",
        "instruction": """Rewrite with a different TONE and make a small semantic tweak (slightly different but related goal).
For example: if original formally asks to "reply to Daniel's email", the new one could casually say "Hey, shoot Daniel a quick message back confirming Thursday morning works for me."
The semantic meaning should be close but not identical — like a different user asking for a similar thing in their own words.
The task must still be achievable in the app environment shown in the screenshots.""",
    },
]

# ==================== Major edit dimensions ====================
DIMENSIONS_MAJOR = [
    {
        "name": "change_target_major",
        "instruction": """Change the TARGET to a COMPLETELY DIFFERENT type of element or entity in the same app.
Do NOT just switch to a similar target (e.g., Daniel → Alice is too similar).
Instead, switch to a fundamentally different kind of target within the same app.
For example: in Gmail, switch from replying to an email → managing a label, or from a person's email → a notification setting.
In Settings, switch from brightness → WiFi, or from display → sound.
ONLY reference targets that are actually available in the app shown in the screenshots.
The resulting task should feel like a different use case of the same app.""",
    },
    {
        "name": "change_action_major",
        "instruction": """Replace the operation with a FUNDAMENTALLY DIFFERENT action from a different feature area of the same app.
Do NOT pick a closely related action (e.g., reply → forward is too similar).
Instead, pick an action from a completely different feature category.
For example: replying to email → composing a new email with attachment, reading email → managing filters/labels, setting alarm → using stopwatch/timer.
The action must be something the app shown in the screenshots actually supports.
The new task should exercise a completely different part of the app's functionality.""",
    },
    {
        "name": "adjust_complexity_major",
        "instruction": """Create a SIGNIFICANTLY more complex multi-step task that combines 2-3 different operations from different feature areas.
Combine operations from different parts of the app or even across apps visible on the device.
For example: "reply to email" → "find Daniel's email, reply confirming Thursday, then create a calendar event for Thursday 10 AM and set a reminder 30 minutes before".
All referenced apps and features must be available on the device as shown in the screenshots.
Write as a natural, coherent request — not as a numbered list of steps.""",
    },
    {
        "name": "change_params_major",
        "instruction": """Keep the general category of operation but DRASTICALLY reframe it from a completely different user perspective or use case.
Instead of just changing parameter values, reimagine WHY the user would need this operation and describe it from that new perspective.
For example: "reply to Daniel's email saying I'll be there" → "I just got an email from my interviewer. I need to send a professional response confirming my attendance at the scheduled time, making sure I sound enthusiastic but not overly casual."
The underlying operation should still be achievable in the app shown in the screenshots, but the framing should be dramatically different.""",
    },
    {
        "name": "batch_operation",
        "instruction": """Expand the single-item operation into a BATCH or conditional operation over multiple items.
For example: "reply to Daniel's email" → "reply to all unread emails from today with a quick acknowledgment"
"set alarm for 7 AM" → "set alarms for every weekday at 7 AM"
"delete this contact" → "find and delete all duplicate contacts"
The scope changes from one specific target to multiple targets sharing a property or condition.
IMPORTANT: The batch operation must be feasible within the app's capabilities as shown in the screenshots.
Write naturally as a single coherent request.""",
    },
]

# ==================== Prompt Templates ====================
PROMPT_TEMPLATE = """You are a mobile GUI task designer creating training data for a phone agent.

Original task query: {query}
Application/Task context: {task_name}

Action steps the agent took to complete the original task:
{steps_summary}

I've provided {num_screenshots} screenshots showing the phone's app environment at various stages.
These screenshots show you what apps, UI elements, and data are available on this device.

The device has apps like: Settings, Gmail, Calendar, Chrome, Contacts, Messages (SMS), Gallery, Files, Clock/Alarm,
Mastodon, Mattermost, TaoDian (shopping), Google Maps, and standard Android system features.

YOUR TASK: Generate ONE new query following this specific dimension:
{dimension_instruction}

CRITICAL CONSTRAINTS:
- The new query MUST be executable on the same device with the same apps and data
- Only reference apps, contacts, emails, files, or settings that are VISIBLE in the screenshots or standard on this Android device
- Do NOT invent data that doesn't exist (e.g., don't reference contacts or emails not shown)
- {modification_level}
- Write in English (or Chinese if the original query is in Chinese)
- Output ONLY the new query text, nothing else (no quotes, no explanation)"""

MODIFICATION_LEVEL_MINOR = "The modification should be a small but meaningful change — similar difficulty to the original"
MODIFICATION_LEVEL_MAJOR = "The modification from the original should be SIGNIFICANT — not a minor tweak but a substantially different task"

# ========================================================

_cache_lock = threading.Lock()


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with _cache_lock:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(dict(cache), f, ensure_ascii=False, indent=2)


def load_previous_queries():
    """Load previously generated queries from the previous batch to avoid duplicates."""
    prev_queries = {}
    if not os.path.exists(PREV_CSV):
        print(f"WARNING: Previous CSV not found at {PREV_CSV}")
        return prev_queries

    with open(PREV_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row["task_name"]
            if task_name not in prev_queries:
                prev_queries[task_name] = []
            for col_name, val in row.items():
                if col_name not in ("task_name", "original_query") and val:
                    prev_queries[task_name].append(val)

    total = sum(len(v) for v in prev_queries.values())
    print(f"Loaded {total} previous queries for {len(prev_queries)} tasks (anti-duplication)")
    return prev_queries


def encode_image(path):
    img = Image.open(path)
    if img.width > 1024:
        ratio = 1024 / img.width
        img = img.resize((1024, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_screenshots(task_name):
    """Get up to MAX_SCREENSHOTS screenshots, evenly sampled across the trajectory."""
    screenshots_dir = os.path.join(TRAJ_DIR, task_name, "screenshots")
    all_shots = sorted(
        _glob.glob(os.path.join(screenshots_dir, f"{task_name}-0-*.png")),
        key=lambda p: int(re.search(r'-(\d+)\.png$', p).group(1))
    )
    if not all_shots:
        return []
    if len(all_shots) <= MAX_SCREENSHOTS:
        return all_shots
    indices = [0]
    step = (len(all_shots) - 1) / (MAX_SCREENSHOTS - 1)
    for i in range(1, MAX_SCREENSHOTS - 1):
        indices.append(int(i * step))
    indices.append(len(all_shots) - 1)
    return [all_shots[i] for i in indices]


def get_action_summary(traj_data):
    """Extract the action history."""
    traj = traj_data.get("0", {}).get("traj", [])
    actions = []
    for step in traj:
        prediction = step.get("prediction", "")
        # Extract the action description (first line usually)
        lines = prediction.split("\n")
        if lines:
            action_line = lines[0].replace('Action: ', '').strip('"')
            if action_line:
                actions.append(f"Step {step.get('step', 0)}: {action_line}")
    return "\n".join(actions[:15])


def get_num_steps(traj_data):
    """Get the number of trajectory steps."""
    traj = traj_data.get("0", {}).get("traj", [])
    return len(traj)


def call_gemini(messages):
    for attempt in range(MAX_RETRIES):
        try:
            payload = {
                "model": MODEL_NAME,
                "messages": messages,
                "stream": False,
                "temperature": 0.9,
                "max_tokens": 512,
            }
            headers = dict(HEADERS)
            headers["X-Model-Request-Id"] = f"mw-v2-{time.time()}"

            resp = requests.post(MODEL_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            elif resp.status_code == 429:
                wait = min(5 * (attempt + 1), 60)
                time.sleep(wait)
            else:
                wait = min(3 * (attempt + 1), 30)
                time.sleep(wait)
        except Exception:
            wait = min(3 * (attempt + 1), 30)
            time.sleep(wait)
    return None


def generate_one(task_name, query, cache_key, dimension, is_major, cache, prev_queries,
                 screenshot_paths, steps_summary):
    """Generate one query variant."""
    if cache_key in cache:
        return cache_key, cache[cache_key], None

    modification_level = MODIFICATION_LEVEL_MAJOR if is_major else MODIFICATION_LEVEL_MINOR

    prompt_text = PROMPT_TEMPLATE.format(
        query=query,
        task_name=task_name,
        steps_summary=steps_summary,
        dimension_instruction=dimension["instruction"],
        num_screenshots=len(screenshot_paths),
        modification_level=modification_level,
    )

    # Anti-duplication
    avoid_list = []
    task_prev = prev_queries.get(task_name, [])
    avoid_list.extend(task_prev)

    # From current batch
    for ck, val in cache.items():
        if ck.startswith(f"{task_name}_") and ck != cache_key and val:
            avoid_list.append(val)

    if avoid_list:
        prompt_text += "\n\nIMPORTANT: The following queries have ALREADY been generated for this task. Your new query MUST be substantially different from ALL of them:\n"
        for aq in avoid_list:
            prompt_text += f"- {aq}\n"
        prompt_text += "\nGenerate something clearly distinct from all the above."

    # Build message with screenshots
    user_content = []
    for sp in screenshot_paths:
        try:
            img_b64 = encode_image(sp)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
        except Exception:
            pass
    user_content.append({"type": "text", "text": prompt_text})

    messages = [{"role": "user", "content": user_content}]

    result = call_gemini(messages)
    if not result:
        return cache_key, None, "API call failed"

    result = result.strip().strip('"').strip("'")

    with _cache_lock:
        cache[cache_key] = result

    return cache_key, result, None


def generate_task_queries(task_name, query, cache, prev_queries):
    """Generate all 15 queries for one task (5 minor + 10 major)."""
    screenshot_paths = get_screenshots(task_name)

    # Load trajectory for action summary
    traj_path = os.path.join(TRAJ_DIR, task_name, "traj.json")
    steps_summary = "Not available"
    if os.path.isfile(traj_path):
        with open(traj_path, "r", encoding="utf-8") as f:
            traj_data = json.load(f)
        steps_summary = get_action_summary(traj_data)

    results = []

    # 5 minor queries
    for dim_idx in range(5):
        cache_key = f"{task_name}_minor_v{dim_idx}"
        ck, result, err = generate_one(
            task_name, query, cache_key, DIMENSIONS_MINOR[dim_idx],
            is_major=False, cache=cache, prev_queries=prev_queries,
            screenshot_paths=screenshot_paths, steps_summary=steps_summary,
        )
        results.append((ck, result, err))

    # 10 major queries (a then b per dimension)
    for dim_idx in range(5):
        cache_key_a = f"{task_name}_major_v{dim_idx}_a"
        ck_a, result_a, err_a = generate_one(
            task_name, query, cache_key_a, DIMENSIONS_MAJOR[dim_idx],
            is_major=True, cache=cache, prev_queries=prev_queries,
            screenshot_paths=screenshot_paths, steps_summary=steps_summary,
        )
        results.append((ck_a, result_a, err_a))

        cache_key_b = f"{task_name}_major_v{dim_idx}_b"
        ck_b, result_b, err_b = generate_one(
            task_name, query, cache_key_b, DIMENSIONS_MAJOR[dim_idx],
            is_major=True, cache=cache, prev_queries=prev_queries,
            screenshot_paths=screenshot_paths, steps_summary=steps_summary,
        )
        results.append((ck_b, result_b, err_b))

    return results


def load_tasks():
    """Load tasks from the task-list CSV that have trajectory data."""
    target_tasks = set()
    with open(TASK_LIST_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target_tasks.add(row["task_name"])

    tasks = []
    for task_name in sorted(target_tasks):
        traj_path = os.path.join(TRAJ_DIR, task_name, "traj.json")
        if os.path.exists(traj_path):
            with open(traj_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            traj = data.get("0", {}).get("traj", [])
            if traj:
                goal = traj[0].get("task_goal", "")
                tasks.append((task_name, goal))
    return tasks


def main():
    global TRAJ_DIR, TASK_LIST_CSV, PREV_CSV, CSV_OUTPUT, CACHE_FILE

    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dir", type=str, default=TRAJ_DIR,
                        help="Root of per-task trajectory folders (traj.json + screenshots/)")
    parser.add_argument("--task-list-csv", type=str, default=TASK_LIST_CSV,
                        help="CSV listing task_name values to process")
    parser.add_argument("--prev-csv", type=str, default=PREV_CSV,
                        help="Previous-batch CSV used for anti-duplication")
    parser.add_argument("--csv-output", type=str, default=CSV_OUTPUT,
                        help="Output CSV path")
    parser.add_argument("--cache-file", type=str, default=CACHE_FILE,
                        help="Cache JSON path (resume-friendly)")
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    TRAJ_DIR = args.traj_dir
    TASK_LIST_CSV = args.task_list_csv
    PREV_CSV = args.prev_csv
    CSV_OUTPUT = args.csv_output
    CACHE_FILE = args.cache_file

    tasks = load_tasks()
    print(f"Loaded {len(tasks)} tasks")

    cache = load_cache()
    print(f"Cache has {len(cache)} entries")

    prev_queries = load_previous_queries()

    if args.dry_run:
        tasks = tasks[:2]
        print("DRY RUN: processing only 2 tasks")

    # Check which tasks need work
    jobs = []
    for task_name, query in tasks:
        needed = 0
        for dim_idx in range(5):
            if f"{task_name}_minor_v{dim_idx}" not in cache:
                needed += 1
        for dim_idx in range(5):
            for v in ["a", "b"]:
                if f"{task_name}_major_v{dim_idx}_{v}" not in cache:
                    needed += 1
        if needed > 0:
            jobs.append((task_name, query))

    total_queries = len(tasks) * 15
    cached_count = sum(
        1 for task_name, _ in tasks
        for k in [f"{task_name}_minor_v{d}" for d in range(5)] +
                 [f"{task_name}_major_v{d}_{v}" for d in range(5) for v in ["a", "b"]]
        if k in cache
    )
    print(f"Tasks needing work: {len(jobs)}/{len(tasks)}")
    print(f"Total query slots: {total_queries}, already cached: {cached_count}")

    if not jobs:
        print("All tasks already cached, writing CSV...")
    else:
        success = 0
        fail = 0
        failures = []

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(generate_task_queries, task_name, query, cache, prev_queries): task_name
                for task_name, query in jobs
            }
            done_count = 0
            for future in as_completed(futures):
                task_name = futures[future]
                done_count += 1
                try:
                    task_results = future.result()
                    for ck, result, error in task_results:
                        if result:
                            success += 1
                        else:
                            fail += 1
                            failures.append((ck, error))
                except Exception as e:
                    fail += 15
                    failures.append((task_name, str(e)))
                    print(f"  [FAIL] task {task_name}: {e}")

                if done_count % 10 == 0 or done_count == len(jobs):
                    print(f"  Tasks done: {done_count}/{len(jobs)} (queries ok={success}, fail={fail})")
                save_cache(cache)

        save_cache(cache)
        print(f"\nAPI calls done: success={success}, fail={fail}")

        if failures:
            print(f"\nFailed ({len(failures)}):")
            for item in failures[:20]:
                print(f"  {item}")

    # Write output CSV
    os.makedirs(os.path.dirname(CSV_OUTPUT), exist_ok=True)
    with open(CSV_OUTPUT, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_name", "original_query",
            "new_query_1_change_target", "new_query_2_change_action",
            "new_query_3_adjust_complexity", "new_query_4_change_params",
            "new_query_5_rephrase_tweak",
            "new_query_6_change_target_major_a", "new_query_7_change_target_major_b",
            "new_query_8_change_action_major_a", "new_query_9_change_action_major_b",
            "new_query_10_adjust_complexity_major_a", "new_query_11_adjust_complexity_major_b",
            "new_query_12_change_params_major_a", "new_query_13_change_params_major_b",
            "new_query_14_batch_operation_a", "new_query_15_batch_operation_b",
        ])
        for task_name, query in tasks:
            queries = []
            # 5 minor
            for dim_idx in range(5):
                cache_key = f"{task_name}_minor_v{dim_idx}"
                queries.append(cache.get(cache_key, ""))
            # 10 major
            for dim_idx in range(5):
                for variant in ["a", "b"]:
                    cache_key = f"{task_name}_major_v{dim_idx}_{variant}"
                    queries.append(cache.get(cache_key, ""))
            writer.writerow([task_name, query] + queries)

    cached_count = sum(
        1 for task_name, _ in tasks
        for k in [f"{task_name}_minor_v{d}" for d in range(5)] +
                 [f"{task_name}_major_v{d}_{v}" for d in range(5) for v in ["a", "b"]]
        if k in cache
    )
    print(f"\nOutput written to: {CSV_OUTPUT}")
    print(f"Total cells filled: {cached_count}/{total_queries}")

    if args.dry_run and tasks:
        print(f"\n{'='*60}")
        for task_name, query in tasks:
            print(f"\nTask: {task_name}")
            print(f"Original: {query}")
            print("  --- Minor ---")
            for i, dim in enumerate(DIMENSIONS_MINOR):
                q = cache.get(f"{task_name}_minor_v{i}", "N/A")
                print(f"  [{dim['name']}]: {q}")
            print("  --- Major ---")
            for i, dim in enumerate(DIMENSIONS_MAJOR):
                qa = cache.get(f"{task_name}_major_v{i}_a", "N/A")
                qb = cache.get(f"{task_name}_major_v{i}_b", "N/A")
                print(f"  [{dim['name']}_a]: {qa}")
                print(f"  [{dim['name']}_b]: {qb}")


if __name__ == "__main__":
    main()
