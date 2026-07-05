#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate grounding bboxes for OSWorld task.json steps, with an automatic retry pass.

This tool runs two phases by default:

  PHASE A - generate:
    1. Walk every trajectory under the target directories.
    2. Select steps that need grounding but have an empty bbox.
    3. Call Gemini to get a bbox ([x1,y1,x2,y2] in a 1000x1000 system).
    4. Write bbox = [[x1,y1],[x2,y2]].
    5. Update coordinate = [(x1+x2)//2, (y1+y2)//2] (bbox center).
    Failures are recorded to an error log.

  PHASE B - retry:
    For each step logged as failed in PHASE A, fire CONCURRENT_PER_STEP
    parallel Gemini calls per step, up to --max-rounds rounds; the first
    success wins. Remaining failures are written to *_still_failed.json.

Model API config is read from environment variables (see the bash wrapper):
    MODEL_URL, MODEL_NAME, MODEL_PROVIDER_ID, GEMINI_API_KEY

Usage:
    python fix_bbox.py --workers 50
    python fix_bbox.py --workers 50 --resume
    python fix_bbox.py --overwrite
    python fix_bbox.py --dry-run
    python fix_bbox.py --skip-retry          # generate only, no retry phase

    # Multiple target dirs:
    python fix_bbox.py --input-dir /path/to/dir1 /path/to/dir2 --workers 50
"""

import os
import sys
import json
import time
import random
import base64
import argparse
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ============ Config ============
# Target directories to scan. Each may contain trajectory folders directly
# (with a task.json) or a two-level app/episode_id structure.
TRAJ_DIRS = [
    "/path/to/dataset/dir1",
]

# Base directory used to compute relative IDs and to store progress/error logs.
BASE_DIR = "/path/to/dataset"

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

SYSTEM_PROMPT = """You are a GUI element locator. The screen coordinate system is 1000x1000.

You will be given a screenshot and an action description. Your job is to find the target UI element described in the action and output its bounding box.

Rules:
- Output the bounding box as [x1, y1, x2, y2] in the 1000x1000 coordinate system
- (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner
- The bbox should tightly enclose the target element
- Be precise: match the visual boundaries of the element

Respond in EXACTLY this JSON format (no other text):
{"target_element": "brief description of the element you identified", "bbox": [x1, y1, x2, y2]}"""

NO_GROUNDING_ACTIONS = {"wait", "terminate", "key", "type", "answer", "hotkey", "scroll"}

# Retry-phase tuning
CONCURRENT_PER_STEP = 10
MAX_ROUNDS = 10


def parse_args():
    parser = argparse.ArgumentParser(description="OSWorld bbox generation + coordinate fix")
    parser.add_argument("--input-dir", type=str, nargs='+', default=TRAJ_DIRS,
                        help="Target directories to process (one or more)")
    parser.add_argument("--base-dir", type=str, default=BASE_DIR,
                        help="Base dir for relative IDs and progress/error logs")
    parser.add_argument("--workers", type=int, default=50, help="Number of parallel workers")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing non-empty bbox")
    parser.add_argument("--dry-run", action="store_true", help="Process only 1 trajectory, do not write files")
    parser.add_argument("--resume", action="store_true", help="Resume from last interrupted run")
    parser.add_argument("--skip-retry", action="store_true", help="Skip the retry phase (generate only)")
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS, help="Max retry rounds in the retry phase")
    return parser.parse_args()


# ============ Shared helpers ============

def encode_image(image_path):
    if not os.path.isfile(image_path):
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_bbox_response(text):
    if not text:
        return None

    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        data = json.loads(cleaned)
        if "bbox" in data and isinstance(data["bbox"], list) and len(data["bbox"]) == 4:
            return [int(v) for v in data["bbox"]]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    try:
        bbox_match = re.search(
            r'"bbox"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', text
        )
        if bbox_match:
            return [int(bbox_match.group(i)) for i in range(1, 5)]
    except Exception:
        pass

    return None


def build_messages(screenshot_path, action_desc):
    img_b64 = encode_image(screenshot_path)
    if img_b64 is None:
        return None

    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        {"type": "text", "text": (
            f"Action: {action_desc}\n\n"
            f"Find the target UI element for this action in the screenshot "
            f"and output its bounding box in 1000x1000 coordinates."
        )}
    ]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]


def collect_traj_dirs(root):
    """Collect all trajectory dirs containing task.json
    (supports the app/episode_id two-level structure)."""
    dirs = []
    for entry in os.listdir(root):
        path = os.path.join(root, entry)
        if not os.path.isdir(path):
            continue
        # If there's a task.json directly, treat as a trajectory dir.
        if os.path.isfile(os.path.join(path, "task.json")):
            dirs.append(path)
        else:
            # Two-level directory (app/episode_id)
            for sub_entry in os.listdir(path):
                sub_path = os.path.join(path, sub_entry)
                if os.path.isdir(sub_path) and os.path.isfile(os.path.join(sub_path, "task.json")):
                    dirs.append(sub_path)
    return dirs


def needs_bbox(step, overwrite):
    plan = step.get("plan", {})
    arguments = plan.get("arguments", {})
    action_type = arguments.get("action", "")

    if action_type in NO_GROUNDING_ACTIONS:
        return False
    if "coordinate" not in arguments:
        return False
    if step.get("is_use") is False:
        return False

    existing_bbox = step.get("bbox", [])
    if existing_bbox and not overwrite:
        return False

    return True


# ============ Phase A: generate ============

def call_gemini(messages, max_retries=10):
    for attempt in range(max_retries):
        try:
            headers = dict(HEADERS)
            headers["X-Model-Request-Id"] = f"fix-bbox-{int(time.time())}-{random.randint(0, 9999)}"

            payload = {
                "model": MODEL_NAME,
                "messages": messages,
                "stream": False,
                "temperature": 0.1,
                "max_tokens": 256,
            }

            resp = requests.post(MODEL_URL, headers=headers, json=payload, timeout=180)

            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"].strip()
            elif resp.status_code == 429:
                wait_time = min(5 * (attempt + 1), 60)
                time.sleep(wait_time)
            else:
                wait_time = min(3 * (attempt + 1), 30)
                time.sleep(wait_time)
        except Exception:
            wait_time = min(3 * (attempt + 1), 30)
            time.sleep(wait_time)
    return None


def get_bbox_for_step(traj_dir, step_data):
    step_index = step_data["_step_index"]
    action_desc = step_data.get("action", "")
    screenshot_file = step_data.get("screenshot", "")
    screenshot_path = os.path.join(traj_dir, screenshot_file)

    messages = build_messages(screenshot_path, action_desc)
    if messages is None:
        return (step_index, None, f"screenshot not found: {screenshot_path}")

    response_text = call_gemini(messages)
    bbox = parse_bbox_response(response_text)

    if bbox is None:
        return (step_index, None, f"parse failed: {response_text[:100] if response_text else 'no response'}")

    return (step_index, bbox, None)


def load_progress(progress_path):
    if os.path.isfile(progress_path):
        try:
            with open(progress_path, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()


def save_progress(progress_path, done_ids):
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(list(done_ids), f)


def process_one_trajectory(traj_dir, args, on_step_done=None):
    task_json_path = os.path.join(traj_dir, "task.json")

    try:
        with open(task_json_path, "r", encoding="utf-8") as f:
            task_data = json.load(f)
    except Exception as e:
        return {"status": "error", "reason": f"failed to read task.json: {e}", "steps_fixed": 0}

    steps = task_data.get("data", [])
    steps_to_process = []

    for i, step in enumerate(steps):
        if needs_bbox(step, args.overwrite):
            step["_step_index"] = i
            steps_to_process.append(step)

    if not steps_to_process:
        return {"status": "ok", "steps_fixed": 0, "reason": "no steps to process"}

    steps_fixed = 0
    step_errors = []
    task_id = task_data.get("episode_id", os.path.basename(traj_dir))[:8]

    with ThreadPoolExecutor(max_workers=min(len(steps_to_process), 20)) as step_executor:
        futures = {
            step_executor.submit(get_bbox_for_step, traj_dir, step): step
            for step in steps_to_process
        }
        for future in as_completed(futures):
            try:
                step_index, bbox, error_msg = future.result()
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    # write bbox
                    steps[step_index]["bbox"] = [[x1, y1], [x2, y2]]
                    # update coordinate with bbox center
                    new_x = (x1 + x2) // 2
                    new_y = (y1 + y2) // 2
                    steps[step_index]["plan"]["arguments"]["coordinate"] = [new_x, new_y]
                    steps_fixed += 1
                else:
                    step_errors.append({"step_index": step_index, "error": error_msg})
            except Exception as e:
                step_errors.append({"step_index": -1, "error": str(e)})

            if on_step_done:
                on_step_done(task_id, steps_fixed, len(step_errors), len(steps_to_process))

    # clean up temporary field
    for step in steps:
        step.pop("_step_index", None)

    # write back
    if steps_fixed > 0 and not args.dry_run:
        try:
            with open(task_json_path, "w", encoding="utf-8") as f:
                json.dump(task_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            return {"status": "error", "reason": f"failed to write task.json: {e}", "steps_fixed": steps_fixed}

    return {
        "status": "ok",
        "steps_fixed": steps_fixed,
        "steps_attempted": len(steps_to_process),
        "step_errors": step_errors,
    }


def run_generate_phase(args, error_log_path):
    """Run the bbox generation phase. Returns the list of collected error records."""
    target_roots = args.input_dir
    base_dir = args.base_dir
    progress_file = os.path.join(base_dir, "fix_bbox_progress.json")

    print(f">>> OSWorld bbox generation + coordinate fix")
    print(f">>> Target directories: {target_roots}")
    print(f">>> Workers: {args.workers}, overwrite: {args.overwrite}")

    all_dirs = []
    for i, root in enumerate(target_roots):
        print(f">>> [{i+1}/{len(target_roots)}] Scanning: {os.path.basename(root)} ...", flush=True)
        if not os.path.isdir(root):
            print(f"    Warning: directory does not exist, skipping")
            continue
        dirs = collect_traj_dirs(root)
        print(f"    Found {len(dirs)} trajectories", flush=True)
        all_dirs.extend(dirs)

    if not all_dirs:
        print("Error: no trajectory directories found")
        sys.exit(1)

    print(f">>> Total trajectories found: {len(all_dirs)}")

    # resume
    done_ids = set()
    if args.resume:
        done_ids = load_progress(progress_file)
        all_dirs = [d for d in all_dirs if os.path.relpath(d, base_dir) not in done_ids]
        print(f">>> Resume mode: {len(done_ids)} done, {len(all_dirs)} remaining")

    if args.dry_run:
        all_dirs = all_dirs[:1]
        print(">>> [DRY RUN] Processing only 1 trajectory, no files written")

    # pre-scan
    print(f">>> Pre-scanning...", flush=True)
    total_pending_steps = 0
    total_all_steps = 0
    total_no_grounding = 0
    total_already_has_bbox = 0
    dirs_with_work = []
    for scan_idx, traj_dir in enumerate(all_dirs):
        if (scan_idx + 1) % 200 == 0:
            print(f"    Scanned {scan_idx + 1}/{len(all_dirs)} trajectories...", flush=True)
        task_json_path = os.path.join(traj_dir, "task.json")
        try:
            with open(task_json_path, "r", encoding="utf-8") as f:
                task_data = json.load(f)
        except Exception:
            continue

        for step in task_data.get("data", []):
            total_all_steps += 1
            plan = step.get("plan", {})
            arguments = plan.get("arguments", {})
            action_type = arguments.get("action", "")

            if action_type in NO_GROUNDING_ACTIONS or "coordinate" not in arguments or step.get("is_use") is False:
                total_no_grounding += 1
            elif step.get("bbox", []) and not args.overwrite:
                total_already_has_bbox += 1
            else:
                total_pending_steps += 1

        pending = sum(1 for step in task_data.get("data", []) if needs_bbox(step, args.overwrite))
        if pending > 0:
            dirs_with_work.append((traj_dir, pending))

    print(f">>> Pre-scan complete:")
    print(f"    Total steps: {total_all_steps}")
    print(f"    No grounding needed (key/type/wait/scroll/etc): {total_no_grounding}")
    print(f"    Already have bbox (skipped): {total_already_has_bbox}")
    print(f"    Pending (missing bbox): {total_pending_steps}")
    print(f"    Trajectories involved: {len(dirs_with_work)}")

    if total_pending_steps == 0:
        print(">>> All steps already processed, nothing to do")
        return []

    all_dirs = [d for d, _ in dirs_with_work]

    # parallel processing
    total_trajs = len(all_dirs)
    total_steps_fixed = 0
    total_step_errors = 0
    global_steps_done = 0
    all_errors = []
    start_time = time.time()
    done_count = 0
    progress_lock = threading.Lock()

    def on_step_done(task_id, fixed, errors, total_in_traj):
        nonlocal global_steps_done
        with progress_lock:
            global_steps_done += 1
            if global_steps_done % 50 == 0:
                elapsed = time.time() - start_time
                print(
                    f"    step progress: {global_steps_done}/{total_pending_steps} | "
                    f"elapsed: {elapsed:.0f}s",
                    flush=True,
                )

    def make_traj_id(traj_dir):
        """Use the path relative to base_dir as a unique ID to avoid
        episode_id collisions across different folders."""
        return os.path.relpath(traj_dir, base_dir)

    def process_and_track(traj_dir):
        traj_id = make_traj_id(traj_dir)
        result = process_one_trajectory(traj_dir, args, on_step_done)
        return traj_id, result

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_and_track, traj_dir): traj_dir
            for traj_dir in all_dirs
        }

        try:
            for future in as_completed(futures):
                try:
                    traj_id, result = future.result()
                except Exception as e:
                    traj_id = os.path.relpath(futures[future], base_dir)
                    result = {"status": "error", "reason": str(e), "steps_fixed": 0}

                with progress_lock:
                    done_count += 1
                    steps_fixed = result.get("steps_fixed", 0)
                    total_steps_fixed += steps_fixed

                    step_errors = result.get("step_errors", [])
                    if step_errors:
                        total_step_errors += len(step_errors)
                        for err in step_errors:
                            all_errors.append({"task_id": traj_id, **err})

                    if result["status"] == "error":
                        all_errors.append({"task_id": traj_id, "error": result.get("reason", "")})

                    done_ids.add(traj_id)

                    if done_count % 10 == 0:
                        save_progress(progress_file, done_ids)

                    if done_count % 20 == 0 or done_count == total_trajs:
                        elapsed = time.time() - start_time
                        print(
                            f"  [{done_count}/{total_trajs}] "
                            f"steps fixed: {total_steps_fixed}, "
                            f"errors: {total_step_errors}, "
                            f"elapsed: {elapsed:.0f}s",
                            flush=True,
                        )
        except KeyboardInterrupt:
            print("\n>>> Interrupted! Saving progress...")
            executor.shutdown(wait=False, cancel_futures=True)

    save_progress(progress_file, done_ids)

    elapsed_total = time.time() - start_time

    print("\n" + "=" * 60)
    print("Generate phase done: bbox generation + coordinate fix")
    print("=" * 60)
    print(f"Trajectories processed: {done_count}")
    print(f"Steps fixed: {total_steps_fixed} (bbox + coordinate)")
    print(f"Errors: {total_step_errors}")
    print(f"Elapsed: {elapsed_total:.1f}s")
    print("=" * 60)

    if all_errors:
        print(f"\n--- Error summary ({len(all_errors)} total) ---")
        for i, err in enumerate(all_errors[:50]):
            print(f"  [{i+1}] task={err['task_id'][:8]}... step={err.get('step_index', '?')} | {err.get('error', '')[:80]}")
        if len(all_errors) > 50:
            print(f"  ... and {len(all_errors) - 50} more errors")

        with open(error_log_path, "w", encoding="utf-8") as f:
            json.dump(all_errors, f, ensure_ascii=False, indent=2)
        print(f"\n>>> Error log: {error_log_path}")

    print(f">>> Progress file: {progress_file}")
    return all_errors


# ============ Phase B: retry ============

def call_gemini_once(messages):
    """Single call, no retry."""
    try:
        headers = dict(HEADERS)
        headers["X-Model-Request-Id"] = f"retry-bbox-{int(time.time())}-{random.randint(0, 99999)}"

        payload = {
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False,
            "temperature": 0.2,
            "max_tokens": 256,
        }

        resp = requests.post(MODEL_URL, headers=headers, json=payload, timeout=60)

        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        else:
            return None
    except Exception:
        return None


def retry_one_step(traj_dir, step_index, task_data):
    """Fire CONCURRENT_PER_STEP parallel calls for one step;
    return the first successful bbox or None."""
    steps = task_data.get("data", [])
    if step_index >= len(steps):
        return None, "step_index out of range"

    step = steps[step_index]
    action_desc = step.get("action", "")
    screenshot_file = step.get("screenshot", "")
    screenshot_path = os.path.join(traj_dir, screenshot_file)

    messages = build_messages(screenshot_path, action_desc)
    if messages is None:
        return None, f"screenshot not found: {screenshot_path}"

    # CONCURRENT_PER_STEP parallel calls
    with ThreadPoolExecutor(max_workers=CONCURRENT_PER_STEP) as executor:
        futures = [executor.submit(call_gemini_once, messages) for _ in range(CONCURRENT_PER_STEP)]

        for future in as_completed(futures):
            response_text = future.result()
            bbox = parse_bbox_response(response_text)
            if bbox is not None:
                # cancel the rest
                for f in futures:
                    f.cancel()
                return bbox, None

    return None, "all attempts failed"


def run_retry_phase(args, errors, error_log_path):
    """Retry the steps recorded as failed in the generate phase."""
    base_dir = args.base_dir

    # Keep only step-level errors (exclude task.json read failures etc.)
    step_errors = [e for e in errors if isinstance(e.get("step_index"), int) and e["step_index"] >= 0]

    print(f"\n{'#' * 60}")
    print(f"{'PHASE B: Retry failed bboxes':^60}")
    print(f"{'#' * 60}")

    if not step_errors:
        print(">>> No step-level errors to retry.")
        return

    print(f">>> Error log: {error_log_path}")
    print(f">>> To retry: {len(step_errors)} steps")
    print(f">>> {CONCURRENT_PER_STEP} parallel calls per step, up to {args.max_rounds} rounds")
    print()

    still_failed = []

    for i, err in enumerate(step_errors):
        task_id = err["task_id"]
        step_index = err["step_index"]
        traj_dir = os.path.join(base_dir, task_id)

        if not os.path.isdir(traj_dir):
            print(f"  [{i+1}/{len(step_errors)}] {task_id} step{step_index}: dir not found, skipping")
            still_failed.append(err)
            continue

        task_json_path = os.path.join(traj_dir, "task.json")
        try:
            with open(task_json_path, "r", encoding="utf-8") as f:
                task_data = json.load(f)
        except Exception as e:
            print(f"  [{i+1}/{len(step_errors)}] {task_id} step{step_index}: read failed {e}")
            still_failed.append(err)
            continue

        # multi-round retry
        success = False
        for round_num in range(1, args.max_rounds + 1):
            print(f"  [{i+1}/{len(step_errors)}] {task_id} step{step_index}: "
                  f"round {round_num} ({CONCURRENT_PER_STEP} parallel)...", end="", flush=True)

            bbox, error_msg = retry_one_step(traj_dir, step_index, task_data)

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                steps = task_data["data"]
                steps[step_index]["bbox"] = [[x1, y1], [x2, y2]]
                steps[step_index]["plan"]["arguments"]["coordinate"] = [(x1 + x2) // 2, (y1 + y2) // 2]

                if not args.dry_run:
                    with open(task_json_path, "w", encoding="utf-8") as f:
                        json.dump(task_data, f, ensure_ascii=False, indent=4)

                print(f" success! bbox={bbox}")
                success = True
                break
            else:
                print(f" failed ({error_msg})")
                time.sleep(2)

        if not success:
            print(f"  [{i+1}/{len(step_errors)}] {task_id} step{step_index}: all {args.max_rounds} rounds failed!")
            still_failed.append(err)

    # summary
    print()
    print("=" * 60)
    fixed = len(step_errors) - len(still_failed)
    print(f"Retry done: success {fixed}/{len(step_errors)}, still failed {len(still_failed)}")
    print("=" * 60)

    if still_failed:
        out_path = error_log_path.replace(".json", "_still_failed.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(still_failed, f, ensure_ascii=False, indent=2)
        print(f"Still-failed records: {out_path}")


# ============ Main ============

def main():
    args = parse_args()
    error_log_path = os.path.join(args.base_dir, "fix_bbox_errors.json")

    # Phase A: generate bboxes
    errors = run_generate_phase(args, error_log_path)

    # Phase B: retry failed steps, unless skipped
    if args.skip_retry:
        print("\n>>> --skip-retry set, skipping retry phase.")
    elif args.dry_run:
        print("\n>>> --dry-run set, skipping retry phase.")
    elif not errors:
        print("\n>>> No failures recorded, skipping retry phase.")
    else:
        run_retry_phase(args, errors, error_log_path)


if __name__ == "__main__":
    main()
