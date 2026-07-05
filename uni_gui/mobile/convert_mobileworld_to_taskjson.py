# -*- coding: utf-8 -*-
"""Convert MobileWorld (rephrase) trajectories into the standard task.json format (qwen3-vl).

Input layout:  {input_root}/<name>_rephrase_{1,2,3}/   (traj.json + result.txt + screenshots/)
Output layout: {output_base}/rephrase_{1,2,3}/<episode_id>/

Filtering rules (any hit discards the whole trajectory):
  - result.txt score != 1.0
  - step ids not sequential / duplicated / missing, or images don't match step count
  - contains double_tap (no qwen equivalent)
  - parse failure / action outside the mapping table

Coordinate source: the prediction field (the model's real output, already 0-1000
normalized), not the post-processed action field.
scroll action: call the teacher model to re-synthesize swipe coordinates
(gemini_scroll_resolver); on failure, fall back geometrically and mark is_reviewed=True.

Model API config is read from environment variables (see the bash wrapper):
    MODEL_URL, MODEL_NAME, MODEL_PROVIDER_ID, GEMINI_API_KEY

Usage:
    python convert_mobileworld_to_taskjson.py --dry-run
    python convert_mobileworld_to_taskjson.py --only AcceptMeetingTask_v1_rephrase_2
    python convert_mobileworld_to_taskjson.py --input-dir /path/to/in --output-dir /path/to/out
"""
import argparse
import json
import os
import re
import shutil
import sys

from github.uni_gui.mobile.app_map import app_for_folder
from github.uni_gui.mobile.gemini_scroll_resolver import resolve_scroll, RETRIES

# ======================== Config ========================
# Edit to your environment, or pass --input-dir / --output-dir.
INPUT_ROOT = "/path/to/mobileworld/traj_logs"
OUTPUT_BASE = "/path/to/output/dataset"
SCREEN_RESOLUTION = [1080, 2400]   # verified device resolution (coords are still 0-1000)
REPHRASE_SUFFIXES = ["_rephrase_1", "_rephrase_2", "_rephrase_3"]
# ========================================================

# Parse prediction: Thought optional, Action required
ACTION_RE = re.compile(r"Action:\s*(\{.*\})\s*$", re.DOTALL)
THOUGHT_RE = re.compile(r"Thought:\s*(.*?)\s*Action:", re.DOTALL)


def generate_action_desc(act_obj):
    """When the model emits no Thought, generate a short description from the action JSON."""
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


def parse_prediction(pred):
    """Return (thought:str, action_obj:dict|None, action_raw:str)."""
    thought = ""
    mt = THOUGHT_RE.search(pred)
    if mt:
        thought = mt.group(1).strip()
    ma = ACTION_RE.search(pred)
    if not ma:
        # Degraded: grab the first {...}
        ma = re.search(r"(\{.*\})", pred, re.DOTALL)
        if not ma:
            return thought, None, ""
    raw = ma.group(1).strip()
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return thought, None, raw
    return thought, obj, raw


# ======================== Action mapping ========================
# Return convention: (plan_arguments|None, needs_scroll_resolve:bool, scroll_direction|None)
#   plan_arguments=None and needs_scroll_resolve=False  => unmappable, discard the whole trajectory

def map_action(act):
    """gemini action_obj -> qwen3-vl mobile_use arguments."""
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

    # double_tap or any unknown action -> unmappable
    return None, False, None


# ======================== Screenshots ========================

def list_screenshots(traj_dir):
    """Return {step_index(int): filename}, named <...>-0-<k>.png."""
    sdir = os.path.join(traj_dir, "screenshots")
    out = {}
    if not os.path.isdir(sdir):
        return out
    for fn in os.listdir(sdir):
        m = re.search(r"-(\d+)-(\d+)\.png$", fn)
        if m:
            out[int(m.group(2))] = fn
    return out


# ======================== Single-trajectory conversion ========================

def read_score(traj_dir):
    rp = os.path.join(traj_dir, "result.txt")
    if not os.path.isfile(rp):
        return None
    m = re.search(r"score:\s*([\d.]+)", open(rp, encoding="utf-8").read())
    return float(m.group(1)) if m else None


def convert_trajectory(folder, suffix, input_root, output_base, dry_run=False):
    """Convert a single trajectory. Returns (task_json|None, discard_reason|None, fallback_steps:list).

    fallback_steps: list of (episode_id, step) that used the geometric fallback.
    """
    traj_dir = os.path.join(input_root, folder)
    fallback_steps = []

    # ---- score filter ----
    score = read_score(traj_dir)
    if score is None or score < 1.0:
        return None, "score != 1.0", fallback_steps

    # ---- read traj.json ----
    tp = os.path.join(traj_dir, "traj.json")
    if not os.path.isfile(tp):
        return None, "no traj.json", fallback_steps
    data = json.load(open(tp, encoding="utf-8"))
    # Take the first (and only) episode
    steps = []
    for _, v in data.items():
        steps = v.get("traj", [])
        break
    if not steps:
        return None, "empty traj", fallback_steps

    # ---- step continuity ----
    ids = [st.get("step") for st in steps]
    if ids != list(range(1, len(steps) + 1)):
        return None, f"step ids not sequential: {ids[:8]}", fallback_steps

    # ---- screenshot integrity ----
    shots = list_screenshots(traj_dir)
    shot_ids = sorted(shots.keys())
    if shot_ids != list(range(1, len(steps) + 1)):
        return None, f"screenshot ids mismatch (imgs={shot_ids[:8]}, steps={len(steps)})", fallback_steps

    # ---- app ----
    app = app_for_folder(folder)
    if app is None:
        return None, "app not found", fallback_steps

    query = steps[0].get("task_goal", "")
    episode_id = folder[: -len(suffix)]

    # ---- parse + map step by step (validate all mappable first, then call the model) ----
    parsed = []
    for st in steps:
        k = st["step"]
        thought, act_obj, act_raw = parse_prediction(st.get("prediction", ""))
        if act_obj is None:
            return None, f"step {k}: prediction parse failed", fallback_steps
        args, need_scroll, direction = map_action(act_obj)
        if args is None and not need_scroll:
            return None, f"step {k}: unmappable action {act_obj.get('action_type')}", fallback_steps
        parsed.append((k, thought, act_obj, act_raw, args, need_scroll, direction))

    # ---- build data steps (only call the model here, so we only spend on kept trajectories) ----
    data_steps = []
    for (k, thought, act_obj, act_raw, args, need_scroll, direction) in parsed:
        is_reviewed = False
        if need_scroll:
            if dry_run:
                # dry-run does not hit the network; use the fallback as a placeholder
                from github.uni_gui.mobile.gemini_scroll_resolver import GEOMETRIC_FALLBACK
                c1, c2 = GEOMETRIC_FALLBACK.get(direction, GEOMETRIC_FALLBACK["down"])
                used_fallback = False
            else:
                img_path = os.path.join(traj_dir, "screenshots", shots[k])
                c1, c2, used_fallback = resolve_scroll(img_path, direction, retries=RETRIES)
            args = {"action": "swipe", "coordinate": c1, "coordinate2": c2}
            if used_fallback:
                is_reviewed = True
                fallback_steps.append((episode_id, k))

        # If the model emitted no Thought, generate a short description from the action itself
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
            "code": act_raw,             # the prediction's raw Action JSON
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
        return task_json, None, fallback_steps

    # ---- write to disk ----
    rephrase_dir = "rephrase_" + suffix.split("_")[-1]
    out_dir = os.path.join(output_base, rephrase_dir, episode_id)
    os.makedirs(out_dir, exist_ok=True)
    for k in range(1, len(steps) + 1):
        src = os.path.join(traj_dir, "screenshots", shots[k])
        dst = os.path.join(out_dir, f"screenshot_step{k - 1}.png")
        shutil.copy2(src, dst)
    with open(os.path.join(out_dir, "task.json"), "w", encoding="utf-8") as f:
        json.dump(task_json, f, ensure_ascii=False, indent=4)

    return task_json, None, fallback_steps


# ======================== Main ========================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="No disk writes, no network (scroll uses the fallback placeholder)")
    ap.add_argument("--only", type=str, default=None, help="Process only the named folder (single-trajectory check)")
    ap.add_argument("--input-dir", type=str, default=INPUT_ROOT, help="Input root containing <name>_rephrase_{1,2,3} folders")
    ap.add_argument("--output-dir", type=str, default=OUTPUT_BASE, help="Output base; writes rephrase_{1,2,3}/<episode_id>/")
    args = ap.parse_args()

    input_root = args.input_dir
    output_base = args.output_dir

    if not os.path.isdir(input_root):
        print(f"[ERROR] Input directory does not exist: {input_root}")
        sys.exit(1)

    # Collect target folders
    folders = []
    for d in sorted(os.listdir(input_root)):
        for suf in REPHRASE_SUFFIXES:
            if d.endswith(suf):
                folders.append((d, suf))
                break
    if args.only:
        folders = [(d, s) for (d, s) in folders if d == args.only]
        if not folders:
            print(f"[ERROR] --only matched nothing: {args.only}")
            sys.exit(1)

    print(f"=== {'DRY RUN' if args.dry_run else 'RUN'} === {len(folders)} rephrase folders\n")

    total = success = 0
    discard_reasons = {}
    discard_list = []
    all_fallback = []

    def bucket(reason):
        """Collapse step-specific reasons into readable categories."""
        if "unmappable action" in reason:
            return "unmappable action (e.g. double_tap)"
        if "parse failed" in reason:
            return "prediction parse failed"
        if "step ids not sequential" in reason:
            return "step ids not sequential"
        if "screenshot ids mismatch" in reason:
            return "screenshot ids mismatch"
        return reason

    for idx, (folder, suf) in enumerate(folders):
        total += 1
        task_json, reason, fallback = convert_trajectory(folder, suf, input_root, output_base, dry_run=args.dry_run)
        all_fallback.extend(fallback)
        if reason:
            b = bucket(reason)
            discard_reasons[b] = discard_reasons.get(b, 0) + 1
            if b != "score != 1.0":
                discard_list.append(f"{folder}: {reason}")
        elif task_json:
            success += 1
        if (idx + 1) % 20 == 0 or (idx + 1) == len(folders):
            print(f"[Progress] {idx+1}/{len(folders)} | ok {success} | discarded {total-success}")

    print(f"\n{'='*56}")
    print(f"{'Statistics':^52}")
    print(f"{'='*56}")
    print(f"  Target folders:    {total}")
    print(f"  Converted:         {success}")
    print(f"  Discarded:         {total - success}")
    print(f"  -- Discard reasons --")
    for r, c in sorted(discard_reasons.items(), key=lambda x: -x[1]):
        print(f"     {r}: {c}")
    if discard_list:
        print(f"  -- Non-score discard list --")
        for d in discard_list:
            print(f"     {d}")

    print(f"\n{'='*56}")
    print(f"  scroll geometric-fallback steps (model still failed after all retries): {len(all_fallback)}")
    if all_fallback:
        print(f"  These steps were set is_reviewed=True:")
        for ep, k in all_fallback:
            print(f"     {ep}  step {k}")
    print(f"{'='*56}")
    if not args.dry_run:
        print(f"\nOutput directory: {output_base}/rephrase_{{1,2,3}}/")


if __name__ == "__main__":
    main()
