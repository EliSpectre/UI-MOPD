"""
Generate SFT training and test Parquet files from desktop (OSWorld) trajectories.

Usage:
    python generate_desktop_sft_parquet.py --data_dir /path/to/desktop/data
Output:
    <output_dir>/desktop_sft_train.parquet
    <output_dir>/desktop_sft_test.parquet
"""

import argparse
import json
import os
import sys

from datasets import Dataset

# ======================== Config ========================
MIN_PIXELS = 3136
MAX_PIXELS = 13107200

SYSTEM_PROMPT = (
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

USER_PROMPT_TEMPLATE = (
    "<image>\n\n"
    "Please generate the next move according to the UI screenshot, instruction and previous actions.\n\n"
    "Instruction: {query}\n\n"
    "Previous actions:\n{history}"
)


def build_ground_truth(thought, action, plan):
    plan_json = json.dumps(plan, ensure_ascii=False)
    return f"<think>\n{thought}\n</think>\nAction: {action}\n<tool_call>\n{plan_json}\n</tool_call>"


def build_user_content(query, data_steps, current_step):
    history_parts = []
    for s in data_steps:
        if s["step"] >= current_step:
            break
        history_parts.append(f"Step {s['step'] + 1}: {s['action']}")
    history = "\n".join(history_parts) if history_parts else "None"
    return USER_PROMPT_TEMPLATE.format(query=query, history=history)


def iter_trajectory_dirs(root_dir):
    if not os.path.isdir(root_dir):
        print(f"[ERROR] Directory does not exist: {root_dir}")
        sys.exit(1)

    for current_dir, dirnames, filenames in os.walk(root_dir):
        dirnames.sort()
        if "task.json" not in filenames:
            continue
        yield os.path.join(current_dir, "task.json"), current_dir
        dirnames[:] = []


def process_task_json(task_json_path, traj_dir):
    with open(task_json_path, "r", encoding="utf-8") as f:
        task = json.load(f)

    app_name = task.get("app", "")
    episode_id = task["episode_id"]
    query = task["query"]
    data_steps = task["data"]

    train_rows = []
    test_rows = []

    for step_info in data_steps:
        if not step_info.get("is_use", True):
            continue
        if step_info.get("is_delete", False):
            continue

        step_num = step_info["step"]
        thought = step_info.get("thought", "")
        action = step_info.get("action", "")
        plan = step_info.get("plan", {})
        screenshot = step_info.get("screenshot", "")
        split = step_info.get("train_test", "train")

        if not plan:
            continue

        current_image_path = os.path.join(traj_dir, screenshot)
        images = [
            {"image": current_image_path, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}
        ]

        ground_truth = build_ground_truth(thought, action, plan)
        user_content = build_user_content(query, data_steps, step_num)

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        row = {
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
                "mode": split,
            },
        }

        if split == "test":
            test_rows.append(row)
        else:
            train_rows.append(row)

    return train_rows, test_rows


def main():
    parser = argparse.ArgumentParser(description="Generate desktop SFT parquet from trajectory data.")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory containing trajectory folders")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for parquet files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_tasks = list(iter_trajectory_dirs(args.data_dir))
    print(f"Found {len(all_tasks)} trajectories in {args.data_dir}")

    all_train_rows = []
    all_test_rows = []
    app_stats = {}

    for task_json_path, traj_dir in all_tasks:
        train_rows, test_rows = process_task_json(task_json_path, traj_dir)
        all_train_rows.extend(train_rows)
        all_test_rows.extend(test_rows)
        for row in train_rows + test_rows:
            app = row["extra_info"]["app"]
            app_stats[app] = app_stats.get(app, 0) + 1

    print(f"\n=== App statistics ===")
    for app, count in sorted(app_stats.items(), key=lambda x: -x[1]):
        print(f"  {app}: {count} steps")

    print(f"\nTotal: {len(all_tasks)} trajectories, "
          f"{len(all_train_rows)} train steps, {len(all_test_rows)} test steps")

    if all_train_rows:
        ds_train = Dataset.from_list(all_train_rows)
        train_path = os.path.join(args.output_dir, "desktop_sft_train.parquet")
        ds_train.to_parquet(train_path)
        print(f"\nTrain output: {train_path}")
        print(f"  Rows: {len(ds_train)}, Size: {os.path.getsize(train_path) / 1024 / 1024:.1f} MB")

    if all_test_rows:
        ds_test = Dataset.from_list(all_test_rows)
        test_path = os.path.join(args.output_dir, "desktop_sft_test.parquet")
        ds_test.to_parquet(test_path)
        print(f"\nTest output: {test_path}")
        print(f"  Rows: {len(ds_test)}, Size: {os.path.getsize(test_path) / 1024 / 1024:.1f} MB")

    if not all_train_rows and not all_test_rows:
        print("[ERROR] No records generated")
        return

    # Verification
    print("\n=== Verification ===")
    sample = (all_train_rows or all_test_rows)[0]
    print(f"data_source: {sample['data_source']}")
    print(f"prompt roles: {[m['role'] for m in sample['prompt']]}")
    print(f"images count: {len(sample['images'])}")
    print(f"ground_truth (first 300 chars): {sample['reward_model']['ground_truth'][:300]}...")
    print(f"extra_info: {sample['extra_info']}")


if __name__ == "__main__":
    main()
