# -*- coding: utf-8 -*-
"""
Convert OSWorld rollout trajectories into the standard task.json format.

Input layout:  {input_root}/{app}/{episode_id}/   (traj.jsonl + instruction.txt + result.txt + step screenshots)
Output layout: {output_root}/{episode_id}/         (task.json + screenshot_step*.png)

Filtering rules:
  - result.txt != 1.0          -> skip
  - step_num not sequential (1,2,3,...) -> discard
  - screenshots missing or step_0 absent -> discard
  - action contains unmappable operations -> discard

Usage:
    python convert_osworld_to_taskjson.py
    python convert_osworld_to_taskjson.py --dry-run
    python convert_osworld_to_taskjson.py --input-dir /path/to/in1 /path/to/in2 --output-dir /path/to/out1 /path/to/out2
"""

import json
import os
import re
import shutil
import sys
import argparse
import glob as glob_mod

# ======================== Config ========================
# Input roots and output roots correspond one-to-one. Edit to your environment.
INPUT_ROOTS = [
    "/path/to/input/rewrite_1",
    "/path/to/input/rewrite_2",
    "/path/to/input/rewrite_3",
]
OUTPUT_ROOTS = [
    "/path/to/output/rewrite_1",
    "/path/to/output/rewrite_2",
    "/path/to/output/rewrite_3",
]
SCREEN_RESOLUTION = [1920, 1080]
# ========================================================

W, H = SCREEN_RESOLUTION


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Statistics only, do not write files")
    parser.add_argument("--input-dir", type=str, nargs='+', default=INPUT_ROOTS,
                        help="Input directories (one or more), paired one-to-one with --output-dir")
    parser.add_argument("--output-dir", type=str, nargs='+', default=OUTPUT_ROOTS,
                        help="Output directories (one or more), paired one-to-one with --input-dir")
    return parser.parse_args()


# ======================== Coordinate conversion ========================

def to_permille(x, y):
    """Convert pixel coordinates to permille coordinates (0-999)."""
    return [round(x * 999 / W), round(y * 999 / H)]


# ======================== Argument extraction ========================

def extract_coords(s):
    """Extract the first two numeric arguments from pyautogui.xxx(x, y, ...)."""
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
    """Extract params from scroll(amount, x, y) or scroll(amount)."""
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
    """Extract text from pyautogui.typewrite(\"\"\"text\"\"\", interval=...)."""
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
    """Extract text from pyautogui.write(message='''text''')."""
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
    """Extract the key name from keyDown('xxx')."""
    match = re.search(r"keyDown\(\s*['\"](.+?)['\"]\s*\)", s)
    if match:
        return match.group(1)
    return ""


def extract_sleep_time(s):
    """Extract the duration from pyautogui.sleep(t)."""
    match = re.search(r'sleep\(\s*([\d.]+)\s*\)', s)
    if match:
        return float(match.group(1))
    return 0.5


# ======================== Action validation ========================

def is_known_action_line(line):
    """Check whether a single action line belongs to the mappable action space."""
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
    """
    Validate that every line in the action string is mappable.
    Returns (is_valid, unmappable_lines).
    """
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

def parse_action_to_plan(action_str, result_score):
    """Parse the action string into the plan format and return the code-field string too."""
    action_str = action_str.strip()

    if action_str == "DONE":
        status = "success" if result_score >= 1.0 else "failure"
        return {"name": "computer_use", "arguments": {"action": "terminate", "status": status}}, action_str

    if action_str == "FAIL":
        return {"name": "computer_use", "arguments": {"action": "terminate", "status": "failure"}}, action_str

    if action_str == "WAIT":
        return {"name": "computer_use", "arguments": {"action": "wait", "time": 5}}, action_str

    lines = [l.strip() for l in action_str.split("\n") if l.strip()]

    if len(lines) == 1:
        return parse_single_action(lines[0], result_score)

    return parse_combo_action(lines, result_score)


def parse_single_action(line, result_score):
    """Parse a single-line action."""
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


def parse_combo_action(lines, result_score):
    """Parse a multi-line combined action."""

    # moveTo + scroll -> scroll with coordinate
    if len(lines) == 2 and lines[0].startswith("pyautogui.moveTo(") and lines[1].startswith("pyautogui.scroll("):
        x, y = extract_coords(lines[0])
        amount, _, _ = extract_scroll_params(lines[1])
        args = {"action": "scroll"}
        if amount is not None:
            args["pixels"] = amount
        if x is not None and y is not None:
            args["coordinate"] = to_permille(x, y)
        code = lines[0] + "\n" + lines[1]
        return {"name": "computer_use", "arguments": args}, code

    # moveTo + dragTo -> left_click_drag with start_coordinate
    if len(lines) == 2 and lines[0].startswith("pyautogui.moveTo(") and lines[1].startswith("pyautogui.dragTo("):
        x1, y1 = extract_coords(lines[0])
        x2, y2 = extract_coords(lines[1])
        args = {"action": "left_click_drag"}
        if x2 is not None and y2 is not None:
            args["coordinate"] = to_permille(x2, y2)
        if x1 is not None and y1 is not None:
            args["start_coordinate"] = to_permille(x1, y1)
        code = lines[0] + "\n" + lines[1]
        return {"name": "computer_use", "arguments": args}, code

    # click + typewrite + keyDown + keyUp -> type
    if (len(lines) >= 3 and
        lines[0].startswith("pyautogui.click(") and
        lines[1].startswith("pyautogui.typewrite(") and
        lines[2].startswith("pyautogui.keyDown(")):
        text = extract_typewrite_text(lines[1])
        code = "\n".join(lines)
        return {"name": "computer_use", "arguments": {"action": "type", "text": text}}, code

    # typewrite + keyDown + keyUp -> type
    if (len(lines) == 3 and
        lines[0].startswith("pyautogui.typewrite(") and
        lines[1].startswith("pyautogui.keyDown(") and
        lines[2].startswith("pyautogui.keyUp(")):
        text = extract_typewrite_text(lines[0])
        code = lines[0]
        return {"name": "computer_use", "arguments": {"action": "type", "text": text}}, code

    # Pure keyDown/keyUp combination -> key
    key_downs = []
    key_ups = []
    all_keys = True
    for line in lines:
        if line.startswith("pyautogui.keyDown("):
            key_downs.append(extract_key_name(line))
        elif line.startswith("pyautogui.keyUp("):
            key_ups.append(extract_key_name(line))
        else:
            all_keys = False
            break

    if all_keys and key_downs and len(key_downs) == len(key_ups):
        code = "\n".join(lines)
        return {"name": "computer_use", "arguments": {"action": "key", "keys": key_downs}}, code

    # Fall back to the first valid non-sleep action among the lines
    for line in lines:
        if line.startswith("pyautogui.") and not line.startswith("pyautogui.sleep("):
            plan, _ = parse_single_action(line, result_score)
            return plan, "\n".join(lines)

    return {"name": "computer_use", "arguments": {"action": "wait", "time": 1}}, "\n".join(lines)


# ======================== Response parsing ========================

def extract_action_description(response):
    """
    Extract the natural-language action description from a response.
    response is a dict: {"content": "...", "role": "assistant", "reasoning_content": "..."}
    Extract the text after '## Action:' from content.
    """
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


# ======================== Validation functions ========================

def validate_step_nums(entries):
    """Validate that step_num increases sequentially as 1,2,3,..."""
    for idx, entry in enumerate(entries):
        expected = idx + 1
        actual = entry.get("step_num")
        if actual != expected:
            return False, f"expected step_num {expected}, got {actual} at index {idx}"
    return True, ""


def validate_screenshots(traj_dir, entries):
    """
    Validate screenshot integrity:
    1. The step_0 screenshot must exist.
    2. Each entry's screenshot_file must exist.
    Returns (is_valid, reason, step0_path).
    """
    step0_matches = glob_mod.glob(os.path.join(traj_dir, "step_0_*.png"))
    if not step0_matches:
        return False, "step_0 screenshot missing", None
    step0_path = step0_matches[0]

    for idx, entry in enumerate(entries):
        screenshot_file = entry.get("screenshot_file", "")
        if not screenshot_file:
            return False, f"entry {idx} has no screenshot_file field", None
        full_path = os.path.join(traj_dir, screenshot_file)
        if not os.path.isfile(full_path):
            return False, f"screenshot missing: {screenshot_file}", None

    return True, "", step0_path


# ======================== Trajectory conversion ========================

def convert_trajectory(traj_dir, app_name, output_dir, dry_run=False):
    """
    Convert a single trajectory.
    Returns (task_json, error_msg, discard_reason).
    """
    traj_path = os.path.join(traj_dir, "traj.jsonl")
    instruction_path = os.path.join(traj_dir, "instruction.txt")
    result_path = os.path.join(traj_dir, "result.txt")

    if not os.path.isfile(traj_path):
        return None, "no traj.jsonl", None
    if not os.path.isfile(instruction_path):
        return None, "no instruction.txt", None

    with open(instruction_path, "r", encoding="utf-8") as f:
        query = f.read().strip()

    result_score = 0.0
    if os.path.isfile(result_path):
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                result_score = float(f.read().strip())
        except (ValueError, TypeError):
            pass

    if result_score < 1.0:
        return None, None, "result != 1.0"

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
        return None, "empty traj.jsonl", None

    # ====== step_num continuity check ======
    is_valid, reason = validate_step_nums(entries)
    if not is_valid:
        return None, None, f"step_num not sequential: {reason}"

    # ====== Action space check ======
    for entry in entries:
        action_str = entry.get("action", "").strip()
        is_valid, unmappable_lines = validate_action_space(action_str)
        if not is_valid:
            reason = f"unmappable actions: {unmappable_lines[:3]}"
            return None, None, reason

    # ====== Screenshot check ======
    is_valid, reason, step0_path = validate_screenshots(traj_dir, entries)
    if not is_valid:
        return None, None, f"screenshot issue: {reason}"

    # ====== Conversion ======
    episode_id = os.path.basename(traj_dir)

    data_steps = []
    screenshots_to_copy = []

    # screenshot_step0 <- step_0 initial screenshot
    screenshots_to_copy.append((step0_path, "screenshot_step0.png"))

    for idx, entry in enumerate(entries):
        action_str = entry.get("action", "")
        response = entry.get("response", {})
        # Prefer the natural_language_action field, otherwise extract from response
        nl_action = entry.get("natural_language_action", "")
        if not nl_action:
            nl_action = extract_action_description(response)

        # thought comes from response.reasoning_content
        thought = ""
        if isinstance(response, dict):
            thought = response.get("reasoning_content", "") or ""

        plan, code = parse_action_to_plan(action_str, result_score)

        # step N uses screenshot_step{N-1} as input
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

        # Copy this entry's screenshot as screenshot_step{idx+1}
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
        "verified": False,
        "task_completed": True,
        "data": data_steps,
    }

    if dry_run:
        return task_json, None, None

    out_dir = os.path.join(output_dir, episode_id)
    os.makedirs(out_dir, exist_ok=True)

    for src_path, dst_name in screenshots_to_copy:
        dst_path = os.path.join(out_dir, dst_name)
        if src_path and os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)

    task_json_path = os.path.join(out_dir, "task.json")
    with open(task_json_path, "w", encoding="utf-8") as f:
        json.dump(task_json, f, ensure_ascii=False, indent=4)

    return task_json, None, None


# ======================== Main ========================

def main():
    args = parse_args()
    input_dirs = args.input_dir
    output_dirs = args.output_dir

    if len(input_dirs) != len(output_dirs):
        print(f"[ERROR] --input-dir and --output-dir must have the same count: "
              f"{len(input_dirs)} inputs, {len(output_dirs)} outputs")
        sys.exit(1)

    if args.dry_run:
        print("=== DRY RUN ===\n")

    for input_root in input_dirs:
        if not os.path.isdir(input_root):
            print(f"[ERROR] Input directory does not exist: {input_root}")
            sys.exit(1)

    if not args.dry_run:
        for output_root in output_dirs:
            os.makedirs(output_root, exist_ok=True)

    total = 0
    success = 0
    skip_result = 0
    discarded = 0
    errors_list = []
    discarded_list = []
    app_stats = {}

    # Pre-scan to count total trajectories
    all_trajectories = []
    for input_root, output_root in zip(input_dirs, output_dirs):
        print(f"[INFO] Scanning input directory: {input_root} -> {output_root}")
        for app_name in sorted(os.listdir(input_root)):
            app_dir = os.path.join(input_root, app_name)
            if not os.path.isdir(app_dir):
                continue
            for episode_id in sorted(os.listdir(app_dir)):
                traj_dir = os.path.join(app_dir, episode_id)
                if not os.path.isdir(traj_dir):
                    continue
                all_trajectories.append((app_name, episode_id, traj_dir, output_root))

    total_count = len(all_trajectories)
    print(f"[INFO] Found {total_count} trajectories, starting processing...\n")

    for idx, (app_name, episode_id, traj_dir, output_root) in enumerate(all_trajectories):
        total += 1

        if (idx + 1) % 50 == 0 or (idx + 1) == total_count:
            print(f"[Progress] {idx + 1}/{total_count} ({(idx + 1) / total_count * 100:.1f}%) | OK: {success} | Skipped: {skip_result} | Discarded: {discarded}")

        result, err, discard_reason = convert_trajectory(
            traj_dir, app_name, output_root, dry_run=args.dry_run
        )

        if discard_reason:
            if "result != 1.0" in discard_reason:
                skip_result += 1
            else:
                discarded += 1
                discarded_list.append(f"{app_name}/{episode_id}: {discard_reason}")
        elif err:
            errors_list.append(f"{app_name}/{episode_id}: {err}")
        elif result:
            success += 1
            n_steps = len(result["data"])
            app_stats[app_name] = app_stats.get(app_name, 0) + n_steps

    print()

    print(f"{'=' * 60}")
    print(f"{'Statistics':^56}")
    print(f"{'=' * 60}")
    print(f"  Total trajectories:        {total}")
    print(f"  Converted successfully:    {success}")
    print(f"  Skipped (result != 1.0):   {skip_result}")
    print(f"  Discarded (other reasons): {discarded}")
    print(f"  Other errors:              {len(errors_list)}")
    if total > 0:
        print(f"  Conversion rate (ok/total): {success / total * 100:.1f}%")

    print(f"\n{'=' * 60}")
    print(f"{'Steps per app':^56}")
    print(f"{'=' * 60}")
    for app, count in sorted(app_stats.items(), key=lambda x: -x[1]):
        print(f"  {app}: {count} steps")

    total_steps = sum(app_stats.values())
    print(f"\n  Total steps: {total_steps}")

    if discarded_list:
        print(f"\n{'=' * 60}")
        print(f"{'Discarded trajectories (first 30)':^56}")
        print(f"{'=' * 60}")
        for d in discarded_list[:30]:
            print(f"  {d}")
        if len(discarded_list) > 30:
            print(f"  ... and {len(discarded_list) - 30} more")

    if errors_list:
        print(f"\n{'=' * 60}")
        print(f"{'Other errors (first 20)':^56}")
        print(f"{'=' * 60}")
        for e in errors_list[:20]:
            print(f"  {e}")

    if not args.dry_run:
        print(f"\nOutput directory: {output_root}")


if __name__ == "__main__":
    main()
