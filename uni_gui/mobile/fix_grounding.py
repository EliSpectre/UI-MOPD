# -*- coding: utf-8 -*-
"""
Generate grounding bboxes for MobileWorld task.json steps, then retry failures.

Runs the full pipeline by default:
  PHASE A (generate): for click/long_press/swipe steps missing a bbox, call the model
                      to produce a bbox (999x999), set coordinate to the bbox center.
                      swipe also produces bbox (start) and bbox2 (end).
  PHASE B (retry)   : re-scan for steps still missing a bbox; for each, fire
                      CONCURRENCY_PER_STEP parallel calls per round, up to MAX_ROUNDS
                      rounds, first success wins; write back immediately.

Coordinate system: 999x999 (matching the mobile_world prompt).

Model API config is read from environment variables (see the bash wrapper):
    MODEL_URL, MODEL_NAME, MODEL_PROVIDER_ID, GEMINI_API_KEY

Usage:
    python fix_grounding.py --base-dir /path/to/mobile_world --workers 50
    python fix_grounding.py --base-dir /path/to/mobile_world --resume
    python fix_grounding.py --base-dir /path/to/mobile_world --overwrite
    python fix_grounding.py --base-dir /path/to/mobile_world --skip-retry
    python fix_grounding.py --base-dir /path/to/mobile_world --dry-run
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
# Base dir; the dataset/ subdirectory contains the trajectory folders.
# Edit to your environment, or pass --base-dir.
BASE_DIR = "/path/to/mobile_world"

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

NO_GROUNDING_ACTIONS = {"wait", "terminate", "type", "answer", "system_button", "ask_user"}
COORD_MAX = 999

# PHASE B retry knobs
CONCURRENCY_PER_STEP = 10
MAX_ROUNDS = 100

# ============ Prompts ============
CLICK_SYSTEM_PROMPT = """You are a GUI element locator. The screen coordinate system is 999x999.

You will be given a screenshot, a thought process, and an action description.
Your job is to find the target UI element described in the action and output its bounding box.

Context:
- Thought: {thought}
- Action: {action_desc}
- Current coordinate: {coordinate}

Rules:
- Output the bounding box as [x1, y1, x2, y2] in the 999x999 coordinate system
- (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner
- The bbox should tightly enclose the target element
- Be precise: match the visual boundaries of the element

Respond in EXACTLY this JSON format (no other text):
{{"target_element": "brief description of the element you identified", "bbox": [x1, y1, x2, y2]}}"""

SWIPE_SYSTEM_PROMPT = """You are a GUI swipe action locator. The screen coordinate system is 999x999.

You will be given a screenshot, a thought process, and a swipe action description.
The swipe goes from a starting point to an ending point.

Your job is to determine two bounding boxes:
1. "start_bbox": the region where the swipe starts
2. "end_bbox": the region where the swipe ends

Context:
- Thought: {thought}
- Action: {action_desc}
- Current start coordinate: {coordinate}
- Current end coordinate: {coordinate2}

Rules:
- Output both bounding boxes as [x1, y1, x2, y2] in the 999x999 coordinate system
- (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner
- The start_bbox should enclose the area where the swipe begins
- The end_bbox should enclose the area where the swipe ends
- For scrolling gestures, the bbox should represent reasonable start/end regions on the scrollable area
- Be precise: match the visual context of where the gesture makes sense

Respond in EXACTLY this JSON format (no other text):
{{"start_bbox": [x1, y1, x2, y2], "end_bbox": [x1, y1, x2, y2]}}"""


def parse_args():
    parser = argparse.ArgumentParser(description="MobileWorld grounding bbox generation + retry")
    parser.add_argument("--base-dir", type=str, default=BASE_DIR,
                        help="Base dir; the dataset/ subdir holds the trajectory folders")
    parser.add_argument("--workers", type=int, default=50, help="Trajectory-level concurrent workers")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing non-empty bboxes")
    parser.add_argument("--dry-run", action="store_true", help="Process 1 trajectory only, write nothing")
    parser.add_argument("--resume", action="store_true", help="Resume PHASE A from the last interruption")
    parser.add_argument("--skip-retry", action="store_true", help="Run PHASE A only, skip the retry phase")
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS, help="Max retry rounds per step (PHASE B)")
    return parser.parse_args()


# ============ Shared helpers ============

def encode_image(image_path):
    if not os.path.isfile(image_path):
        return None
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_mime(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    elif ext == ".png":
        return "image/png"
    return "image/png"


def clamp_bbox(bbox):
    """Clamp bbox values to [0, COORD_MAX] - the model sometimes outputs 1000 on a 999x999 grid."""
    return [max(0, min(v, COORD_MAX)) for v in bbox]


def call_gemini(messages, max_retries=10):
    for attempt in range(max_retries):
        try:
            headers = dict(HEADERS)
            headers["X-Model-Request-Id"] = f"fix-mw-{int(time.time())}-{random.randint(0, 9999)}"

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


def call_gemini_single(messages):
    """One request, no retry. Returns text on success, None on failure."""
    try:
        headers = dict(HEADERS)
        headers["X-Model-Request-Id"] = f"retry-mw-{int(time.time())}-{random.randint(0, 99999)}"

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
            time.sleep(random.uniform(2, 8))
        return None
    except Exception:
        return None


def parse_click_bbox_response(text):
    if not text:
        return None
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        data = json.loads(cleaned)
        if "bbox" in data and isinstance(data["bbox"], list) and len(data["bbox"]) == 4:
            bbox = [int(v) for v in data["bbox"]]
            if all(0 <= v <= 1000 for v in bbox):
                return clamp_bbox(bbox)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    try:
        bbox_match = re.search(
            r'"bbox"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', text
        )
        if bbox_match:
            bbox = [int(bbox_match.group(i)) for i in range(1, 5)]
            if all(0 <= v <= 1000 for v in bbox):
                return clamp_bbox(bbox)
    except Exception:
        pass

    return None


def parse_swipe_bbox_response(text):
    if not text:
        return None, None
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        data = json.loads(cleaned)
        start_bbox = data.get("start_bbox")
        end_bbox = data.get("end_bbox")
        if (isinstance(start_bbox, list) and len(start_bbox) == 4 and
                isinstance(end_bbox, list) and len(end_bbox) == 4):
            start_bbox = [int(v) for v in start_bbox]
            end_bbox = [int(v) for v in end_bbox]
            if all(0 <= v <= 1000 for v in start_bbox) and all(0 <= v <= 1000 for v in end_bbox):
                return clamp_bbox(start_bbox), clamp_bbox(end_bbox)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    try:
        start_match = re.search(
            r'"start_bbox"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', text
        )
        end_match = re.search(
            r'"end_bbox"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', text
        )
        if start_match and end_match:
            start_bbox = [int(start_match.group(i)) for i in range(1, 5)]
            end_bbox = [int(end_match.group(i)) for i in range(1, 5)]
            if all(0 <= v <= 1000 for v in start_bbox) and all(0 <= v <= 1000 for v in end_bbox):
                return clamp_bbox(start_bbox), clamp_bbox(end_bbox)
    except Exception:
        pass

    return None, None


def convert_dict_bbox(bbox_dict, pixel):
    """Convert a {x_min, x_max, y_min, y_max} pixel-format bbox to [[x1,y1],[x2,y2]] (999x999)."""
    width, height = pixel[0], pixel[1]
    x_min = int(bbox_dict["x_min"] / width * COORD_MAX)
    y_min = int(bbox_dict["y_min"] / height * COORD_MAX)
    x_max = int(bbox_dict["x_max"] / width * COORD_MAX)
    y_max = int(bbox_dict["y_max"] / height * COORD_MAX)
    return [[x_min, y_min], [x_max, y_max]]


def build_click_messages(traj_dir, step_data):
    thought = step_data.get("thought", "")
    action_desc = step_data.get("action", "")
    screenshot_file = step_data.get("screenshot", "")
    screenshot_path = os.path.join(traj_dir, screenshot_file)
    coordinate = step_data.get("plan", {}).get("arguments", {}).get("coordinate", [])

    img_b64 = encode_image(screenshot_path)
    if img_b64 is None:
        return None

    mime_type = get_image_mime(screenshot_path)
    system_prompt = CLICK_SYSTEM_PROMPT.format(
        thought=thought, action_desc=action_desc, coordinate=coordinate
    )
    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_b64}"}},
        {"type": "text", "text": "Find the target UI element for this action in the screenshot and output its bounding box in 999x999 coordinates."}
    ]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]


def build_swipe_messages(traj_dir, step_data):
    thought = step_data.get("thought", "")
    action_desc = step_data.get("action", "")
    screenshot_file = step_data.get("screenshot", "")
    screenshot_path = os.path.join(traj_dir, screenshot_file)
    arguments = step_data.get("plan", {}).get("arguments", {})
    coordinate = arguments.get("coordinate", [])
    coordinate2 = arguments.get("coordinate2", [])

    img_b64 = encode_image(screenshot_path)
    if img_b64 is None:
        return None

    mime_type = get_image_mime(screenshot_path)
    system_prompt = SWIPE_SYSTEM_PROMPT.format(
        thought=thought, action_desc=action_desc, coordinate=coordinate, coordinate2=coordinate2
    )
    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_b64}"}},
        {"type": "text", "text": "Identify the start and end regions for this swipe action and output both bounding boxes in 999x999 coordinates."}
    ]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]


def collect_trajectories(dataset_dir):
    """Collect all trajectory directories under dataset_dir that contain a task.json."""
    all_dirs = []

    if not os.path.isdir(dataset_dir):
        print(f"[ERROR] Dataset directory does not exist: {dataset_dir}")
        sys.exit(1)

    for current_dir, dirnames, filenames in os.walk(dataset_dir):
        dirnames.sort()
        if "task.json" in filenames:
            all_dirs.append(current_dir)
            dirnames[:] = []

    return all_dirs


# ============ PHASE A: generate ============

def get_bbox_for_click_step(traj_dir, step_data):
    """Call the model to obtain a bbox for a click/long_press step."""
    step_index = step_data["_step_index"]
    messages = build_click_messages(traj_dir, step_data)
    if messages is None:
        screenshot_path = os.path.join(traj_dir, step_data.get("screenshot", ""))
        return (step_index, None, f"screenshot not found: {screenshot_path}")

    response_text = call_gemini(messages)
    bbox = parse_click_bbox_response(response_text)

    if bbox is None:
        return (step_index, None, f"parse failed: {response_text[:100] if response_text else 'no response'}")

    return (step_index, bbox, None)


def get_bbox_for_swipe_step(traj_dir, step_data):
    """Call the model to obtain bbox + bbox2 for a swipe step."""
    step_index = step_data["_step_index"]
    messages = build_swipe_messages(traj_dir, step_data)
    if messages is None:
        screenshot_path = os.path.join(traj_dir, step_data.get("screenshot", ""))
        return (step_index, None, None, f"screenshot not found: {screenshot_path}")

    response_text = call_gemini(messages)
    start_bbox, end_bbox = parse_swipe_bbox_response(response_text)

    if start_bbox is None or end_bbox is None:
        return (step_index, None, None, f"parse failed: {response_text[:100] if response_text else 'no response'}")

    return (step_index, start_bbox, end_bbox, None)


def process_one_trajectory(traj_dir, args, on_step_done=None):
    """Process one trajectory in PHASE A."""
    task_json_path = os.path.join(traj_dir, "task.json")

    try:
        with open(task_json_path, "r", encoding="utf-8") as f:
            task_data = json.load(f)
    except Exception as e:
        return {"status": "error", "reason": f"failed to read task.json: {e}", "steps_fixed": 0}

    steps = task_data.get("data", [])
    pixel = task_data.get("screen_resolution", [1080, 2400])
    steps_to_process_click = []
    steps_to_process_swipe = []
    steps_fixed = 0
    task_id = task_data.get("episode_id", os.path.basename(traj_dir))[:8]

    for i, step in enumerate(steps):
        if step.get("is_use") is False:
            continue
        if step.get("is_delete", False):
            continue

        plan = step.get("plan", {})
        if not plan:
            continue
        arguments = plan.get("arguments", {})
        action_type = arguments.get("action", "unknown")
        coordinate = arguments.get("coordinate", None)

        if action_type in NO_GROUNDING_ACTIONS:
            continue
        if coordinate is None:
            continue

        # Handle an existing dict-format bbox (convert directly, no API call)
        existing_bbox = step.get("bbox", [])
        if isinstance(existing_bbox, dict) and "x_min" in existing_bbox:
            step_pixel = step.get("pixel", pixel)
            normalized_bbox = convert_dict_bbox(existing_bbox, step_pixel)
            step["bbox"] = normalized_bbox
            center_x = (normalized_bbox[0][0] + normalized_bbox[1][0]) // 2
            center_y = (normalized_bbox[0][1] + normalized_bbox[1][1]) // 2
            arguments["coordinate"] = [center_x, center_y]
            steps_fixed += 1
            continue

        # Skip steps that already have a valid bbox (unless --overwrite)
        if existing_bbox and isinstance(existing_bbox, list) and len(existing_bbox) == 2 and not args.overwrite:
            if action_type == "swipe" and not step.get("bbox2"):
                pass
            else:
                continue

        step["_step_index"] = i

        if action_type == "swipe":
            steps_to_process_swipe.append(step)
        else:
            steps_to_process_click.append(step)

    total_to_process = len(steps_to_process_click) + len(steps_to_process_swipe)
    if total_to_process == 0 and steps_fixed == 0:
        return {"status": "ok", "steps_fixed": 0, "reason": "no steps to process"}

    # Call the model in parallel
    step_errors = []

    with ThreadPoolExecutor(max_workers=min(total_to_process + 1, 20)) as step_executor:
        futures = {}

        for step in steps_to_process_click:
            future = step_executor.submit(get_bbox_for_click_step, traj_dir, step)
            futures[future] = ("click", step)

        for step in steps_to_process_swipe:
            future = step_executor.submit(get_bbox_for_swipe_step, traj_dir, step)
            futures[future] = ("swipe", step)

        for future in as_completed(futures):
            action_type, step = futures[future]
            try:
                if action_type == "click":
                    step_index, bbox, error_msg = future.result()
                    if bbox is not None:
                        steps[step_index]["bbox"] = [[bbox[0], bbox[1]], [bbox[2], bbox[3]]]
                        center_x = (bbox[0] + bbox[2]) // 2
                        center_y = (bbox[1] + bbox[3]) // 2
                        steps[step_index]["plan"]["arguments"]["coordinate"] = [center_x, center_y]
                        steps_fixed += 1
                    else:
                        step_errors.append({"step_index": step_index, "error": error_msg})
                else:
                    step_index, start_bbox, end_bbox, error_msg = future.result()
                    if start_bbox is not None and end_bbox is not None:
                        steps[step_index]["bbox"] = [[start_bbox[0], start_bbox[1]], [start_bbox[2], start_bbox[3]]]
                        steps[step_index]["bbox2"] = [[end_bbox[0], end_bbox[1]], [end_bbox[2], end_bbox[3]]]
                        start_cx = (start_bbox[0] + start_bbox[2]) // 2
                        start_cy = (start_bbox[1] + start_bbox[3]) // 2
                        steps[step_index]["plan"]["arguments"]["coordinate"] = [start_cx, start_cy]
                        end_cx = (end_bbox[0] + end_bbox[2]) // 2
                        end_cy = (end_bbox[1] + end_bbox[3]) // 2
                        steps[step_index]["plan"]["arguments"]["coordinate2"] = [end_cx, end_cy]
                        steps_fixed += 1
                    else:
                        step_errors.append({"step_index": step_index, "error": error_msg})
            except Exception as e:
                step_errors.append({"step_index": -1, "error": str(e)})

            if on_step_done:
                on_step_done(task_id, steps_fixed, len(step_errors), total_to_process)

    # Clean up the temporary field
    for step in steps:
        step.pop("_step_index", None)

    # Write back task.json
    if steps_fixed > 0 and not args.dry_run:
        try:
            with open(task_json_path, "w", encoding="utf-8") as f:
                json.dump(task_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            return {"status": "error", "reason": f"failed to write task.json: {e}", "steps_fixed": steps_fixed}

    return {
        "status": "ok",
        "steps_fixed": steps_fixed,
        "steps_attempted": total_to_process,
        "step_errors": step_errors,
    }


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


def run_generate_phase(args, dataset_dir, progress_file):
    print(f">>> [PHASE A] Grounding bbox generation (mobile_world)")
    print(f">>> Coordinate system: {COORD_MAX}x{COORD_MAX}")
    print(f">>> Workers: {args.workers}, overwrite: {args.overwrite}")

    all_dirs = collect_trajectories(dataset_dir)
    print(f">>> Found {len(all_dirs)} trajectories")

    if not all_dirs:
        print("Error: no trajectory directories found")
        sys.exit(1)

    # Resume
    done_ids = set()
    if args.resume:
        done_ids = load_progress(progress_file)
        all_dirs = [d for d in all_dirs if os.path.basename(d) not in done_ids]
        print(f">>> Resume mode: {len(done_ids)} done, {len(all_dirs)} remaining")

    if args.dry_run:
        all_dirs = all_dirs[:1]
        print(">>> [DRY RUN] Processing 1 trajectory only, no files written")

    # Pre-scan: count pending steps
    total_pending_steps = 0
    dirs_with_work = []
    for traj_dir in all_dirs:
        task_json_path = os.path.join(traj_dir, "task.json")
        try:
            with open(task_json_path, "r", encoding="utf-8") as f:
                task_data = json.load(f)
        except Exception:
            continue

        pending = 0
        for step in task_data.get("data", []):
            if step.get("is_use") is False:
                continue
            if step.get("is_delete", False):
                continue

            plan = step.get("plan", {})
            if not plan:
                continue
            arguments = plan.get("arguments", {})
            action_type = arguments.get("action", "unknown")
            coordinate = arguments.get("coordinate", None)

            if action_type in NO_GROUNDING_ACTIONS or coordinate is None:
                continue

            existing_bbox = step.get("bbox", [])
            if isinstance(existing_bbox, dict) and "x_min" in existing_bbox:
                pending += 1
                continue
            if existing_bbox and isinstance(existing_bbox, list) and len(existing_bbox) == 2 and not args.overwrite:
                continue
            pending += 1

        if pending > 0:
            total_pending_steps += pending
            dirs_with_work.append((traj_dir, pending))

    print(f">>> Pre-scan done: {len(dirs_with_work)} trajectories have pending steps, {total_pending_steps} steps total")

    if total_pending_steps == 0:
        print(">>> All steps already processed in PHASE A")
        return

    all_dirs = [d for d, _ in dirs_with_work]

    # Parallel processing
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
            if global_steps_done % 10 == 0:
                print(
                    f"    Global step progress: {global_steps_done}/{total_pending_steps} | "
                    f"[{task_id}] this traj: {fixed + errors}/{total_in_traj}"
                )

    def process_and_track(traj_dir):
        traj_id = os.path.basename(traj_dir)
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
                    traj_id = os.path.basename(futures[future])
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

                    if done_count % 5 == 0 or done_count == total_trajs:
                        elapsed = time.time() - start_time
                        print(
                            f"  [{done_count}/{total_trajs}] "
                            f"steps fixed: {total_steps_fixed}, "
                            f"errors: {total_step_errors}, "
                            f"elapsed: {elapsed:.0f}s"
                        )
        except KeyboardInterrupt:
            print("\n>>> Interrupted! Saving progress...")
            executor.shutdown(wait=False, cancel_futures=True)

    save_progress(progress_file, done_ids)

    elapsed_total = time.time() - start_time

    print("\n" + "=" * 60)
    print("PHASE A done (mobile_world)")
    print("=" * 60)
    print(f"Trajectories processed: {done_count}")
    print(f"Steps fixed: {total_steps_fixed}")
    print(f"Errors: {total_step_errors}")
    print(f"Elapsed: {elapsed_total:.1f}s")
    print("=" * 60)

    if all_errors:
        print(f"\n--- Error summary ({len(all_errors)}) ---")
        for i, err in enumerate(all_errors[:50]):
            print(f"  [{i+1}] task={err['task_id'][:8]}... step={err.get('step_index', '?')} | {err.get('error', '')[:80]}")
        if len(all_errors) > 50:
            print(f"  ... and {len(all_errors) - 50} more errors")

        log_path = os.path.join(args.base_dir, "fix_grounding_errors.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(all_errors, f, ensure_ascii=False, indent=2)
        print(f"\n>>> Error log: {log_path}")

    print(f">>> Progress file: {progress_file}")


# ============ PHASE B: retry ============

# Per-file write lock: the same task.json must not be written concurrently
file_locks = {}
file_locks_lock = threading.Lock()


def get_file_lock(path):
    with file_locks_lock:
        if path not in file_locks:
            file_locks[path] = threading.Lock()
        return file_locks[path]


def retry_one_step(step_info, max_rounds):
    """Retry a single step: each round fires CONCURRENCY_PER_STEP calls, up to max_rounds, first success wins."""
    traj_dir = step_info["traj_dir"]
    action_type = step_info["action_type"]
    step_data = step_info["step_data"]

    if action_type == "swipe":
        messages = build_swipe_messages(traj_dir, step_data)
    else:
        messages = build_click_messages(traj_dir, step_data)

    if messages is None:
        return {"success": False, "step_info": step_info, "error": "screenshot not found"}

    for round_num in range(max_rounds):
        with ThreadPoolExecutor(max_workers=CONCURRENCY_PER_STEP) as ex:
            futures = [ex.submit(call_gemini_single, messages) for _ in range(CONCURRENCY_PER_STEP)]
            for future in as_completed(futures):
                response_text = future.result()
                if response_text is None:
                    continue

                if action_type == "swipe":
                    start_bbox, end_bbox = parse_swipe_bbox_response(response_text)
                    if start_bbox is not None and end_bbox is not None:
                        return {
                            "success": True,
                            "step_info": step_info,
                            "start_bbox": start_bbox,
                            "end_bbox": end_bbox,
                            "rounds": round_num + 1,
                        }
                else:
                    bbox = parse_click_bbox_response(response_text)
                    if bbox is not None:
                        return {
                            "success": True,
                            "step_info": step_info,
                            "bbox": bbox,
                            "rounds": round_num + 1,
                        }

        time.sleep(random.uniform(1, 3))

    return {"success": False, "step_info": step_info, "error": f"failed after {max_rounds} rounds"}


def collect_missing_steps(dataset_dir):
    """Scan all task.json and collect steps still missing a bbox."""
    missing = []
    traj_scanned = 0
    steps_scanned = 0

    for current_dir, dirnames, filenames in os.walk(dataset_dir):
        dirnames.sort()
        if "task.json" not in filenames:
            continue
        dirnames[:] = []

        task_json_path = os.path.join(current_dir, "task.json")
        try:
            with open(task_json_path, "r", encoding="utf-8") as f:
                task_data = json.load(f)
        except Exception:
            continue

        traj_scanned += 1
        if traj_scanned % 50 == 0:
            print(f"    [scanning] checked {traj_scanned} trajectories, {steps_scanned} steps, {len(missing)} missing")

        for i, step in enumerate(task_data.get("data", [])):
            steps_scanned += 1

            if step.get("is_use") is False:
                continue
            if step.get("is_delete", False):
                continue

            plan = step.get("plan", {})
            if not plan:
                continue
            arguments = plan.get("arguments", {})
            action_type = arguments.get("action", "unknown")
            coordinate = arguments.get("coordinate", None)

            if action_type in NO_GROUNDING_ACTIONS:
                continue
            if coordinate is None:
                continue

            existing_bbox = step.get("bbox", [])
            if existing_bbox and isinstance(existing_bbox, list) and len(existing_bbox) == 2:
                continue

            missing.append({
                "traj_dir": current_dir,
                "task_json_path": task_json_path,
                "step_index": i,
                "action_type": action_type,
                "step_data": step,
                "episode_id": task_data.get("episode_id", ""),
            })

    print(f"    [scan done] {traj_scanned} trajectories, {steps_scanned} steps")
    return missing


def apply_result(result):
    """Write a successful retry result back to task.json."""
    if not result["success"]:
        return False

    step_info = result["step_info"]
    task_json_path = step_info["task_json_path"]
    step_index = step_info["step_index"]
    action_type = step_info["action_type"]

    lock = get_file_lock(task_json_path)
    with lock:
        try:
            with open(task_json_path, "r", encoding="utf-8") as f:
                task_data = json.load(f)
        except Exception:
            return False

        steps = task_data.get("data", [])
        if step_index >= len(steps):
            return False

        step = steps[step_index]

        if action_type == "swipe":
            start_bbox = result["start_bbox"]
            end_bbox = result["end_bbox"]
            step["bbox"] = [[start_bbox[0], start_bbox[1]], [start_bbox[2], start_bbox[3]]]
            step["bbox2"] = [[end_bbox[0], end_bbox[1]], [end_bbox[2], end_bbox[3]]]
            start_cx = (start_bbox[0] + start_bbox[2]) // 2
            start_cy = (start_bbox[1] + start_bbox[3]) // 2
            step["plan"]["arguments"]["coordinate"] = [start_cx, start_cy]
            end_cx = (end_bbox[0] + end_bbox[2]) // 2
            end_cy = (end_bbox[1] + end_bbox[3]) // 2
            step["plan"]["arguments"]["coordinate2"] = [end_cx, end_cy]
        else:
            bbox = result["bbox"]
            step["bbox"] = [[bbox[0], bbox[1]], [bbox[2], bbox[3]]]
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            step["plan"]["arguments"]["coordinate"] = [center_x, center_y]

        try:
            with open(task_json_path, "w", encoding="utf-8") as f:
                json.dump(task_data, f, ensure_ascii=False, indent=4)
        except Exception:
            return False

    return True


def run_retry_phase(args, dataset_dir):
    print(f"\n>>> [PHASE B] Grounding bbox retry (mobile_world)")
    print(f">>> {CONCURRENCY_PER_STEP} concurrent calls per step, up to {args.max_rounds} rounds")
    print(f">>> Concurrent steps: {args.workers}")

    missing = collect_missing_steps(dataset_dir)
    print(f">>> Found {len(missing)} steps still missing a bbox")

    if not missing:
        print(">>> All steps already have a bbox, nothing to do")
        return

    # Per-trajectory stats
    traj_counts = {}
    for s in missing:
        eid = s["episode_id"]
        traj_counts[eid] = traj_counts.get(eid, 0) + 1
    print(f">>> Spanning {len(traj_counts)} trajectories")
    for eid, cnt in sorted(traj_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {eid}: {cnt} steps")

    success_count = 0
    fail_count = 0
    start_time = time.time()
    progress_lock = threading.Lock()

    def process_and_save(step_info):
        nonlocal success_count, fail_count
        result = retry_one_step(step_info, args.max_rounds)
        if result["success"]:
            saved = apply_result(result)
            with progress_lock:
                if saved:
                    success_count += 1
                else:
                    fail_count += 1
                total_done = success_count + fail_count
                elapsed = time.time() - start_time
                print(
                    f"  [{total_done}/{len(missing)}] "
                    f"ok: {success_count}, fail: {fail_count}, "
                    f"elapsed: {elapsed:.0f}s "
                    f"| {step_info['episode_id'][:12]} step{step_info['step_index']} "
                    f"(rounds: {result.get('rounds', '?')})"
                )
        else:
            with progress_lock:
                fail_count += 1
                total_done = success_count + fail_count
                elapsed = time.time() - start_time
                print(
                    f"  [{total_done}/{len(missing)}] "
                    f"ok: {success_count}, fail: {fail_count}, "
                    f"elapsed: {elapsed:.0f}s "
                    f"| {step_info['episode_id'][:12]} step{step_info['step_index']} "
                    f"FAILED: {result.get('error', '')[:60]}"
                )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_and_save, s) for s in missing]
        try:
            for future in as_completed(futures):
                future.result()
        except KeyboardInterrupt:
            print("\n>>> Interrupted!")
            executor.shutdown(wait=False, cancel_futures=True)

    elapsed_total = time.time() - start_time
    print("\n" + "=" * 60)
    print("PHASE B done")
    print("=" * 60)
    print(f"Total missing steps: {len(missing)}")
    print(f"Fixed: {success_count}")
    print(f"Still failed: {fail_count}")
    print(f"Elapsed: {elapsed_total:.1f}s")
    print("=" * 60)


def main():
    args = parse_args()
    dataset_dir = os.path.join(args.base_dir, "dataset")
    progress_file = os.path.join(args.base_dir, "fix_grounding_progress.json")

    run_generate_phase(args, dataset_dir, progress_file)

    if args.dry_run:
        print("\n>>> [DRY RUN] Skipping PHASE B (retry).")
        return

    if args.skip_retry:
        print("\n>>> --skip-retry set, skipping PHASE B.")
        return

    run_retry_phase(args, dataset_dir)


if __name__ == "__main__":
    main()
