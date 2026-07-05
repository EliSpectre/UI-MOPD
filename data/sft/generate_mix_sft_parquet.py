"""
Generate mixed (mobile + desktop) SFT training and test Parquet files.

Usage:
    python generate_mix_sft_parquet.py \
        --mobile_dir /path/to/mobile/data \
        --desktop_dir /path/to/desktop/data \
        --output_dir ./output
Output:
    <output_dir>/mix_sft_train.parquet
    <output_dir>/mix_sft_test.parquet
"""

import argparse
import json
import os
import random
import sys

from datasets import Dataset

# ======================== Config ========================

RANDOM_SEED = 42
TEST_TRAJ_COUNT = 30

MOBILE_MIN_PIXELS = 40768
MOBILE_MAX_PIXELS = 548800
DESKTOP_MIN_PIXELS = 3136
DESKTOP_MAX_PIXELS = 13107200

# ======================== System Prompts ========================

MOBILE_SYSTEM_PROMPT = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n* The screen's resolution is 999x999.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `click`: Click the point on the screen with coordinate (x, y).\\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n* `type`: Input the specified text into the activated input box.\\n* `answer`: Output the answer.\\n* `system_button`: Press the system button.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.\\n* `ask_user`: Ask user for clarification.", "enum": ["click", "long_press", "swipe", "type", "answer", "system_button", "wait", "ask_user", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=type`, `action=ask_user` and `action=answer`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) Thought: one concise sentence explaining the next move (no multi-step reasoning).
2) Action: a short imperative describing what to do.
3) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Thought, Action, <tool_call>.
- Be brief: one sentence for Thought, one for Action.
- Do not output anything else outside those three parts.
- If finishing, use mobile_use with action=terminate in the tool call."""

DESKTOP_SYSTEM_PROMPT = (
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    '{"type": "function", "function": {"name_for_human": "computer_use", "name": "computer_use", '
    '"description": "Use a mouse and keyboard to interact with a computer, and take screenshots.\\n'
    "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. "
    "You must click on desktop icons to start applications.\\n"
    "* Some applications may take time to start or process actions, so you may need to wait and take "
    "successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window "
    "doesn't open, try wait and taking another screenshot.\\n"
    "* The screen's resolution is 1000x1000.\\n"
    "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a "
    "screenshot to determine the coordinates of the element before moving the cursor.\\n"
    "* If you tried clicking on a program or link but it failed to load even after waiting, try adjusting "
    "your cursor position so that the tip of the cursor visually falls on the element that you want to click.\\n"
    "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. "
    'Don\'t click boxes on their edges unless asked.", '
    '"parameters": {"properties": {"action": {"description": "\\n'
    "* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\\n"
    "* `type`: Type a string of text on the keyboard.\\n"
    "* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\\n"
    "* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.\\n"
    "* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\\n"
    "* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.\\n"
    "* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.\\n"
    "* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.\\n"
    "* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen "
    "(simulated as double-click since it's the closest action).\\n"
    "* `scroll`: Performs a scroll of the mouse scroll wheel.\\n"
    "* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).\\n"
    "* `wait`: Wait specified seconds for the change to happen.\\n"
    "* `terminate`: Terminate the current task and report its completion status.\\n"
    '* `answer`: Answer a question.\\n        ", '
    '"enum": ["key", "type", "mouse_move", "left_click", "left_click_drag", "right_click", "middle_click", '
    '"double_click", "triple_click", "scroll", "wait", "terminate"], "type": "string"}, '
    '"keys": {"description": "Required only by `action=key`.", "type": "array"}, '
    '"text": {"description": "Required only by `action=type`.", "type": "string"}, '
    '"coordinate": {"description": "The x,y coordinates for mouse actions.", "type": "array"}, '
    '"pixels": {"description": "The amount of scrolling.", "type": "number"}, '
    '"time": {"description": "The seconds to wait.", "type": "number"}, '
    '"status": {"description": "The status of the task.", "type": "string", "enum": ["success", "failure"]}'
    '}, "required": ["action"], "type": "object"}, '
    '"args_format": "Format the arguments as a JSON object."}}\n'
    "</tools>\n\n"
    "For each function call, return a json object with function name and arguments within "
    "<tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>\n\n"
    "# Response format\n\n"
    "Response format for every step:\n"
    "1) Action: a short imperative describing what to do in the UI.\n"
    "2) A single <tool_call>...</tool_call> block containing only the JSON: "
    '{"name": <function-name>, "arguments": <args-json-object>}.\n\n'
    "Rules:\n"
    "- Output exactly in the order: Action, <tool_call>.\n"
    "- Be brief: one sentence for Action.\n"
    "- Do not output anything else outside those parts.\n"
    "- If finishing, use action=terminate in the tool call."
)

# ======================== User Prompt Templates ========================

MOBILE_USER_TEMPLATE = "The user query: {query}.\nTask progress (You have done the following operation on the current device): {history}.\n<image>"

DESKTOP_USER_TEMPLATE = (
    "<image>\n\n"
    "Please generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
    "Instruction: {query}\n\n"
    "Previous actions:\n{history}"
)


# ======================== Helper Functions ========================


def build_ground_truth(thought, action, plan):
    plan_json = json.dumps(plan, ensure_ascii=False)
    return f"<think>\n{thought}\n</think>\nAction: {action}\n<tool_call>\n{plan_json}\n</tool_call>"


def build_mobile_user_content(query, data_steps, current_step):
    history_parts = []
    for s in data_steps:
        if s["step"] >= current_step:
            break
        history_parts.append(f"Step {s['step']}: {s['action']};")
    history = " ".join(history_parts) if history_parts else ""
    return MOBILE_USER_TEMPLATE.format(query=query, history=history)


def build_desktop_user_content(query, data_steps, current_step):
    history_parts = []
    for s in data_steps:
        if s["step"] >= current_step:
            break
        history_parts.append(f"Step {s['step'] + 1}: {s['action']}")
    history = "\n".join(history_parts) if history_parts else "None"
    return DESKTOP_USER_TEMPLATE.format(query=query, history=history)


def iter_trajectory_dirs(root_dir):
    if not os.path.isdir(root_dir):
        print(f"  [WARN] Directory does not exist, skipping: {root_dir}")
        return

    for current_dir, dirnames, filenames in os.walk(root_dir):
        dirnames.sort()
        if "task.json" not in filenames:
            continue
        yield os.path.join(current_dir, "task.json"), current_dir
        dirnames[:] = []


def process_mobile_task(task_json_path, traj_dir):
    with open(task_json_path, "r", encoding="utf-8") as f:
        task = json.load(f)

    app_name = task.get("app", "unknown")
    episode_id = task["episode_id"]
    query = task["query"]
    data_steps = task["data"]

    rows = []
    for step_info in data_steps:
        if not step_info.get("is_use", True):
            continue
        if step_info.get("is_delete", False):
            continue

        plan = step_info.get("plan", {})
        if not plan:
            continue

        step_num = step_info["step"]
        thought = step_info.get("thought", "")
        action = step_info.get("action", "")
        screenshot = step_info.get("screenshot", "")
        if not screenshot:
            continue

        current_image_path = os.path.join(traj_dir, screenshot)
        user_content = build_mobile_user_content(query, data_steps, step_num)

        prompt = [
            {"role": "system", "content": MOBILE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        images = [
            {"image": current_image_path, "min_pixels": MOBILE_MIN_PIXELS, "max_pixels": MOBILE_MAX_PIXELS}
        ]

        ground_truth = build_ground_truth(thought, action, plan)

        rows.append({
            "data_source": "mobile",
            "prompt": prompt,
            "images": images,
            "reward_model": {"ground_truth": ground_truth},
            "ability": "gui_agent",
            "extra_info": {
                "id": f"{episode_id}_step{step_num}",
                "episode_id": episode_id,
                "step_id": step_num,
                "image": current_image_path,
                "app": app_name,
            },
        })

    return rows


def process_desktop_task(task_json_path, traj_dir):
    with open(task_json_path, "r", encoding="utf-8") as f:
        task = json.load(f)

    app_name = task.get("app", "")
    episode_id = task["episode_id"]
    query = task["query"]
    data_steps = task["data"]

    rows = []
    for step_info in data_steps:
        if not step_info.get("is_use", True):
            continue
        if step_info.get("is_delete", False):
            continue

        plan = step_info.get("plan", {})
        if not plan:
            continue

        step_num = step_info["step"]
        thought = step_info.get("thought", "")
        action = step_info.get("action", "")
        screenshot = step_info.get("screenshot", "")
        if not screenshot:
            continue

        current_image_path = os.path.join(traj_dir, screenshot)
        user_content = build_desktop_user_content(query, data_steps, step_num)

        prompt = [
            {"role": "system", "content": DESKTOP_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        images = [
            {"image": current_image_path, "min_pixels": DESKTOP_MIN_PIXELS, "max_pixels": DESKTOP_MAX_PIXELS}
        ]

        ground_truth = build_ground_truth(thought, action, plan)

        rows.append({
            "data_source": "desktop",
            "prompt": prompt,
            "images": images,
            "reward_model": {"ground_truth": ground_truth},
            "ability": "gui_agent",
            "extra_info": {
                "id": f"{episode_id}_step{step_num}",
                "episode_id": episode_id,
                "step_id": step_num,
                "image": current_image_path,
                "app": app_name,
            },
        })

    return rows


# ======================== Main ========================


def collect_trajectories(data_dir, process_fn, side_name):
    traj_rows = {}
    total_steps = 0

    print(f"\n{'=' * 60}")
    print(f"=== {side_name} ===")
    print(f"{'=' * 60}")
    sys.stdout.flush()

    for task_json_path, traj_dir in iter_trajectory_dirs(data_dir):
        rows = process_fn(task_json_path, traj_dir)
        if rows:
            traj_rows[traj_dir] = rows
            total_steps += len(rows)

    print(f"  {side_name}: {len(traj_rows)} trajectories, {total_steps} steps")
    sys.stdout.flush()
    return traj_rows


def main():
    parser = argparse.ArgumentParser(description="Generate mixed SFT parquet from mobile + desktop data.")
    parser.add_argument("--mobile_dir", type=str, required=True, help="Root directory for mobile trajectory data")
    parser.add_argument("--desktop_dir", type=str, required=True, help="Root directory for desktop trajectory data")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for parquet files")
    parser.add_argument("--test_traj_count", type=int, default=TEST_TRAJ_COUNT, help="Number of trajectories for test set")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for train/test split")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Collect mobile trajectories
    mobile_traj_rows = collect_trajectories(args.mobile_dir, process_mobile_task, "Mobile")

    # Collect desktop trajectories
    desktop_traj_rows = collect_trajectories(args.desktop_dir, process_desktop_task, "Desktop")

    # Train/Test split (randomly select N trajectories for test)
    print(f"\n{'=' * 60}")
    print("=== Train/Test Split ===")
    print(f"{'=' * 60}")

    random.seed(args.seed)

    all_traj_keys = list(mobile_traj_rows.keys()) + list(desktop_traj_rows.keys())
    random.shuffle(all_traj_keys)

    test_keys = set(all_traj_keys[:args.test_traj_count])
    train_keys = set(all_traj_keys[args.test_traj_count:])

    train_rows = []
    test_rows = []

    all_traj_rows = {**mobile_traj_rows, **desktop_traj_rows}
    for key in train_keys:
        train_rows.extend(all_traj_rows[key])
    for key in test_keys:
        test_rows.extend(all_traj_rows[key])

    random.shuffle(train_rows)
    random.shuffle(test_rows)

    train_mobile = sum(1 for r in train_rows if r["data_source"] == "mobile")
    train_desktop = sum(1 for r in train_rows if r["data_source"] == "desktop")
    test_mobile = sum(1 for r in test_rows if r["data_source"] == "mobile")
    test_desktop = sum(1 for r in test_rows if r["data_source"] == "desktop")

    print(f"Test:  {len(test_keys)} trajectories, {len(test_rows)} steps (mobile {test_mobile}, desktop {test_desktop})")
    print(f"Train: {len(train_keys)} trajectories, {len(train_rows)} steps (mobile {train_mobile}, desktop {train_desktop})")

    if not train_rows:
        print("[ERROR] No training data generated")
        sys.exit(1)

    # Output Parquet
    print(f"\n{'=' * 60}")
    print("=== Output Parquet ===")
    print(f"{'=' * 60}")

    ds_train = Dataset.from_list(train_rows)
    train_path = os.path.join(args.output_dir, "mix_sft_train.parquet")
    ds_train.to_parquet(train_path)
    print(f"Train: {train_path}")
    print(f"  Rows: {len(ds_train)}, Size: {os.path.getsize(train_path) / 1024 / 1024:.1f} MB")

    if test_rows:
        ds_test = Dataset.from_list(test_rows)
        test_path = os.path.join(args.output_dir, "mix_sft_test.parquet")
        ds_test.to_parquet(test_path)
        print(f"Test:  {test_path}")
        print(f"  Rows: {len(ds_test)}, Size: {os.path.getsize(test_path) / 1024 / 1024:.1f} MB")

    # Verification
    print(f"\n{'=' * 60}")
    print("=== Verification ===")
    print(f"{'=' * 60}")

    ds_check = Dataset.from_parquet(train_path)
    print(f"Train rows: {len(ds_check)}")
    print(f"Columns: {ds_check.column_names}")

    row0 = ds_check[0]
    print(f"\n--- Sample 0 ---")
    print(f"data_source: {row0['data_source']}")
    print(f"prompt roles: {[m['role'] for m in row0['prompt']]}")
    print(f"images count: {len(row0['images'])}")
    print(f"image path: {row0['images'][0]['image']}")
    print(f"ground_truth (first 200 chars): {row0['reward_model']['ground_truth'][:200]}")
    print(f"ability: {row0['ability']}")
    print(f"extra_info: {row0['extra_info']}")


if __name__ == "__main__":
    main()
