#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate the thought and action fields of every step in a MobileWorld dataset.

thought: 5-section structured CoT (aligned with the OSWorld/computer script)
action:  one short imperative sentence (no coordinates)

Model API config is read from environment variables (see the bash wrapper):
    MODEL_URL, MODEL_NAME, MODEL_PROVIDER_ID, GEMINI_API_KEY

Usage:
    python regenerate_thought.py --input-dir /path/to/dataset       # 100 workers
    python regenerate_thought.py --workers 50 --input-dir /path/to/dataset
    python regenerate_thought.py --dry-run --input-dir /path/to/dataset    # test 1 step
    python regenerate_thought.py --resume --input-dir /path/to/dataset     # skip already done
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ======================== Config ========================
# Default target directories; each contains per-episode folders with task.json.
TARGET_DIRS = [
    "/path/to/dataset/dir1",
]

# ---- Model API config (read from environment, with placeholder defaults) ----
MODEL_URL = os.environ.get("MODEL_URL", "https://your-model-endpoint/v1/chat/completions")
MODEL_NAME = os.environ.get("MODEL_NAME", "your-model-name")
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
MODEL_PROVIDER_ID = os.environ.get("MODEL_PROVIDER_ID", "your-provider-id")

DEFAULT_WORKERS = 100
DEFAULT_RETRIES = 100
# ========================================================

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "X-Model-Provider-Id": MODEL_PROVIDER_ID,
    "Content-Type": "application/json",
}

STYLE_EXAMPLES = """## Example 1 (Settings):
The home screen is currently displayed with various app icons visible. No prior actions have been taken. To reduce the text and icon size to the minimum, I need to navigate through Settings to the Display section where font/icon scaling options are located.

Plan:
1) Open the Settings app.
2) Navigate to Display settings.
3) Locate font/display size options and set them to the smallest value.

Possible next actions:
- Tap the Settings app icon on the home screen.
- Swipe up to open the app drawer and find Settings there.
- Pull down the notification shade and tap the gear icon.

Tapping the Settings icon directly on the home screen is the fastest path since it is already visible, avoiding extra navigation steps.

Expected consequence:
- The Settings app will open, showing the main settings menu with categories like Network, Display, Sound, etc.

Action: Tap the Settings app icon on the home screen to open device settings.

## Example 2 (Maps / Navigation):
The previous step opened the Maps application, and the map view is now showing the current location. The task requires searching for a specific restaurant. The search bar is visible at the top of the screen.

Possible next actions:
- Tap the search bar at the top to activate text input.
- Tap the microphone icon to use voice search.
- Tap the "Explore" tab to browse nearby places.

Tapping the search bar is the most direct way to enter a specific destination name, as voice search is less reliable for proper nouns and browsing would require extra filtering steps.

Expected consequence:
- The search bar will become active with a text cursor, the keyboard will appear, and recent/suggested searches may be displayed below.

Action: Tap the search bar at the top of the screen to begin entering the restaurant name."""

SYSTEM_PROMPT = f"""You are a thought process generator for a mobile GUI agent. Given a screenshot, task instruction, previous actions, and the CORRECT next tool_call, produce a structured thought AND a short action description in the target format.

## Thought output style (STRICTLY follow this structure):

1. **Brief context** (1-2 sentences): What is the current state? What has been done so far?
2. **Plan** (optional, only for multi-step reasoning): Numbered steps for the overall approach.
3. **Possible next actions** (2-3 bullet points): List plausible alternatives.
4. **Best choice reasoning** (1-2 sentences): Why the chosen action is optimal over alternatives.
5. **Expected consequence** (1 sentence): What should happen after executing this action.

## Action output style:
ONE short imperative sentence describing the concrete next move in natural language (no numeric coordinates), e.g.
  "Tap the Settings app icon to open device settings."
  "Type 'San Francisco' into the destination field."
  "Swipe up on the list to reveal more items."

## Style Rules:
- Be CONCISE. No verbose descriptions of the screen. No "I can see..." monologues.
- Focus on REASONING and DECISION-MAKING, not observation.
- Use professional, direct language.
- Do NOT repeat the task instruction verbatim.
- Do NOT describe UI elements in exhaustive detail.
- Do NOT include numeric coordinates in thought or action.
- REMOVE all repetition, self-correction, and meta-commentary.

## Output format (strict JSON, no markdown fences, no prefixes):
{{"thought": "<5-section structured thought>", "action": "<one imperative sentence>"}}

## Reference Examples:
{STYLE_EXAMPLES}"""


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_user_message(query, history_actions, plan_args, image_path):
    parts = []

    if os.path.isfile(image_path):
        b64 = encode_image(image_path)
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })

    history_text = "None"
    if history_actions:
        history_text = "\n".join(
            f"Step {i+1}: {a}" for i, a in enumerate(history_actions)
        )

    plan_json = json.dumps(plan_args, ensure_ascii=False)

    text = (
        f"Task instruction: {query}\n\n"
        f"Previous actions:\n{history_text}\n\n"
        f"CORRECT tool_call: {plan_json}\n\n"
        f"Output ONLY the JSON object with keys \"thought\" and \"action\"."
    )
    parts.append({"type": "text", "text": text})

    return parts


def call_gemini(messages, max_retries=100, step_label=""):
    for attempt in range(max_retries):
        try:
            payload = {
                "model": MODEL_NAME,
                "messages": messages,
                "stream": False,
                "temperature": 0.7,
                "max_tokens": 1024,
            }
            headers = dict(HEADERS)
            headers["X-Model-Request-Id"] = f"thought-mw-{time.time()}-{step_label}"

            resp = requests.post(
                MODEL_URL, headers=headers, json=payload, timeout=120
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"].strip()
                parsed = parse_response(content)
                if parsed:
                    return parsed
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


def parse_response(content):
    if not content:
        return None
    clean = content.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)

    try:
        obj = json.loads(clean)
        if obj.get("thought") and obj.get("action"):
            return obj
    except (ValueError, TypeError):
        pass

    m = re.search(r'\{[^{}]*"thought"\s*:', content, re.DOTALL)
    if m:
        start = m.start()
        depth = 0
        for i in range(start, len(content)):
            if content[i] == '{':
                depth += 1
            elif content[i] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(content[start:i+1])
                        if obj.get("thought") and obj.get("action"):
                            return obj
                    except (ValueError, TypeError):
                        pass
                    break
    return None


def generate_one_step(args_tuple):
    task_dir, step_idx, query, history_actions, plan_args, image_path, max_retries = args_tuple
    user_content = build_user_message(query, history_actions, plan_args, image_path)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    result = call_gemini(messages, max_retries=max_retries, step_label=f"{step_idx}")
    return task_dir, step_idx, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, nargs='+', default=TARGET_DIRS,
                        help="Target directories (one or more)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Concurrent workers (default 100)")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="Max retries per step (default 100)")
    parser.add_argument("--dry-run", action="store_true", help="Test 1 step only")
    parser.add_argument("--resume", action="store_true", help="Skip steps where raw_thought is already filled")
    args = parser.parse_args()

    dataset_dirs = args.input_dir

    task_dirs = []
    for target_dir in dataset_dirs:
        if not os.path.isdir(target_dir):
            print(f"Warning: directory does not exist, skipping: {target_dir}")
            continue
        for d in sorted(os.listdir(target_dir)):
            full_path = os.path.join(target_dir, d)
            if os.path.isdir(full_path) and os.path.isfile(os.path.join(full_path, "task.json")):
                task_dirs.append(full_path)
    print(f"Total task folders: {len(task_dirs)}")

    if not task_dirs:
        print("No task folders found.")
        sys.exit(1)

    all_tasks = {}
    step_jobs = []
    task_step_counts = {}

    for task_dir in task_dirs:
        task_json_path = os.path.join(task_dir, "task.json")
        with open(task_json_path, "r", encoding="utf-8") as f:
            task = json.load(f)
        all_tasks[task_dir] = task

        query = task["query"]
        data_steps = task["data"]
        task_step_counts[task_dir] = 0

        for idx, step in enumerate(data_steps):
            if args.resume and step.get("raw_thought", "").strip():
                continue

            plan_args = step.get("plan", {}).get("arguments", {})
            screenshot = step.get("screenshot", "")
            image_path = os.path.join(task_dir, screenshot)
            history_actions = [data_steps[i]["action"] for i in range(idx)]

            step_jobs.append((task_dir, idx, query, history_actions, plan_args, image_path, args.retries))
            task_step_counts[task_dir] = task_step_counts.get(task_dir, 0) + 1

    total_steps = len(step_jobs)
    total_trajs = len([d for d in task_step_counts if task_step_counts[d] > 0])
    print(f"Steps to process: {total_steps} (across {total_trajs} trajectories)")
    if args.resume:
        already = sum(len(t["data"]) for t in all_tasks.values()) - total_steps
        print(f"Already done (skipped): {already}")

    if total_steps == 0:
        print("Nothing to do.")
        return

    if args.dry_run:
        job = step_jobs[0]
        task_dir, step_idx, query, history_actions, plan_args, image_path, retries = job
        task = all_tasks[task_dir]
        step = task["data"][step_idx]
        print(f"\n=== DRY RUN: {task['episode_id']} step {step['step']} ===")
        print(f"Query: {query}")
        print(f"Plan args: {json.dumps(plan_args, ensure_ascii=False)}")
        print(f"Screenshot: {image_path}")
        print(f"Original thought (first 200): {step.get('thought', '')[:200]}...")
        print(f"\nCalling model (retries={retries})...")
        _, _, result = generate_one_step(job)
        if result:
            print(f"\n--- Generated thought ---\n{result['thought']}")
            print(f"\n--- Generated action ---\n{result['action']}")
        else:
            print("\nFAILED after all retries.")
        return

    success = 0
    failed_list = []
    done_steps = 0
    done_trajs = 0
    traj_pending = dict(task_step_counts)
    last_print_traj = 0
    saved_trajs = set()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(generate_one_step, job): job for job in step_jobs}

        for f in as_completed(futures):
            done_steps += 1
            job = futures[f]
            try:
                task_dir, step_idx, result = f.result()
                task = all_tasks[task_dir]
                step = task["data"][step_idx]

                if result:
                    step["raw_thought"] = step.get("thought", "")
                    step["thought"] = result["thought"]
                    step["action"] = result["action"]
                    success += 1
                else:
                    failed_list.append({
                        "episode_id": task["episode_id"],
                        "task_dir": task_dir,
                        "step": step["step"],
                        "step_idx": step_idx,
                    })
            except Exception as e:
                failed_list.append({
                    "episode_id": os.path.basename(job[0]),
                    "task_dir": job[0],
                    "step_idx": job[1],
                    "error": str(e),
                })

            traj_pending[job[0]] = traj_pending.get(job[0], 1) - 1
            if traj_pending[job[0]] == 0:
                done_trajs += 1

            should_print = (
                (done_trajs - last_print_traj) >= 10
                or done_steps == total_steps
            )
            if should_print:
                rate = success / done_steps * 100 if done_steps else 0
                print(f"  [Progress] steps={done_steps}/{total_steps}  "
                      f"trajs_done={done_trajs}/{total_trajs}  "
                      f"ok={success}  fail={len(failed_list)}  "
                      f"rate={rate:.1f}%")
                last_print_traj = done_trajs

                # On each progress print, flush completed trajectories to disk
                # (incremental save, so a network interruption doesn't lose progress)
                for td in list(traj_pending.keys()):
                    if traj_pending[td] == 0 and td not in saved_trajs:
                        tp = os.path.join(td, "task.json")
                        with open(tp, "w", encoding="utf-8") as fw:
                            json.dump(all_tasks[td], fw, ensure_ascii=False, indent=4)
                        saved_trajs.add(td)

    # Final pass: ensure every trajectory is written to disk
    for task_dir, task in all_tasks.items():
        if task_dir not in saved_trajs:
            task_json_path = os.path.join(task_dir, "task.json")
            with open(task_json_path, "w", encoding="utf-8") as f:
                json.dump(task, f, ensure_ascii=False, indent=4)
    print("\nAll task.json files saved.")

    print(f"\n{'='*56}")
    print(f"Done. {success}/{total_steps} steps regenerated successfully.")
    print(f"Failed: {len(failed_list)}")

    if failed_list:
        report_path = os.path.join(os.path.dirname(dataset_dirs[0]), "thought_action_gen_failures.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(failed_list, f, ensure_ascii=False, indent=2)
        print(f"Failure report saved to: {report_path}")
        print(f"\nFailed steps (first 20):")
        for item in failed_list[:20]:
            print(f"  {item.get('episode_id','?')} step={item.get('step', item.get('step_idx','?'))}")
        if len(failed_list) > 20:
            print(f"  ... and {len(failed_list) - 20} more")
        print(f"\nTo retry failed ones: python regenerate_thought.py --resume --workers {args.workers}")
    print(f"{'='*56}\n")


if __name__ == "__main__":
    main()
