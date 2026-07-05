"""
Generate SFT training and test Parquet files from mobile (MobileWorld) trajectories.

Usage:
    python generate_mobile_sft_parquet.py --data_dir /path/to/mobile/data
Output:
    <output_dir>/mobile_sft_train.parquet
    <output_dir>/mobile_sft_test.parquet
"""

import argparse
import json
import os
import sys

from datasets import Dataset

# ======================== Config ========================
MIN_PIXELS = 40768
MAX_PIXELS = 548800

SYSTEM_PROMPT = """# Tools

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

USER_PROMPT_TEMPLATE = "The user query: {query}.\nTask progress (You have done the following operation on the current device): {history}.\n<image>"


def build_ground_truth(thought: str, action: str, plan: dict) -> str:
    plan_json = json.dumps(plan, ensure_ascii=False)
    return f"<think>{thought}</think>\nAction: {action}\n<tool_call>\n{plan_json}\n</tool_call>"


def build_user_content(query: str, step_data: list, current_step: int) -> str:
    history_parts = []
    for s in step_data:
        if s["step"] >= current_step:
            break
        history_parts.append(f"Step {s['step']}: {s['action']};")
    history = " ".join(history_parts) if history_parts else ""
    return USER_PROMPT_TEMPLATE.format(query=query, history=history)


def process_task_json(task_json_path: str, traj_dir: str):
    with open(task_json_path, "r", encoding="utf-8") as f:
        task = json.load(f)

    app_name = task.get("app", "unknown")
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

        image_path = os.path.join(traj_dir, screenshot)
        images = [
            {"image": image_path, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}
        ]

        ground_truth = build_ground_truth(thought, action, plan)
        user_content = build_user_content(query, data_steps, step_num)

        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        row = {
            "data_source": "mobile",
            "prompt": prompt,
            "images": images,
            "reward_model": {"ground_truth": ground_truth},
            "ability": "gui_agent",
            "extra_info": {
                "id": f"{episode_id}_step{step_num}",
                "episode_id": episode_id,
                "step_id": step_num,
                "image": image_path,
                "app": app_name,
                "mode": split,
            },
        }

        if split == "test":
            test_rows.append(row)
        else:
            train_rows.append(row)

    return train_rows, test_rows


def collect_from_dir(data_dir):
    train_rows = []
    test_rows = []

    if not os.path.isdir(data_dir):
        print(f"[SKIP] Directory does not exist: {data_dir}")
        return train_rows, test_rows

    dir_name = os.path.basename(data_dir)
    entries = sorted(os.listdir(data_dir))

    episode_count = 0
    step_count = 0

    for entry in entries:
        entry_path = os.path.join(data_dir, entry)
        if not os.path.isdir(entry_path):
            continue

        task_json_path = os.path.join(entry_path, "task.json")
        if os.path.isfile(task_json_path):
            tr, te = process_task_json(task_json_path, entry_path)
            train_rows.extend(tr)
            test_rows.extend(te)
            episode_count += 1
            step_count += len(tr) + len(te)

    print(f"  [{dir_name}] {episode_count} trajectories, {step_count} steps")
    return train_rows, test_rows


def main():
    parser = argparse.ArgumentParser(description="Generate mobile SFT parquet from trajectory data.")
    parser.add_argument("--data_dir", type=str, required=True, help="Root directory containing trajectory folders")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for parquet files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    all_train_rows = []
    all_test_rows = []

    # Walk through subdirectories
    subdirs = sorted([
        os.path.join(args.data_dir, d)
        for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    ])

    if not subdirs:
        subdirs = [args.data_dir]

    print(f"=== Processing mobile trajectories ===")
    for subdir in subdirs:
        tr, te = collect_from_dir(subdir)
        all_train_rows.extend(tr)
        all_test_rows.extend(te)

    print(f"\nTotal: {len(all_train_rows)} train steps, {len(all_test_rows)} test steps")

    if not all_train_rows and not all_test_rows:
        print("[ERROR] No records generated")
        sys.exit(1)

    if all_train_rows:
        ds_train = Dataset.from_list(all_train_rows)
        train_path = os.path.join(args.output_dir, "mobile_sft_train.parquet")
        ds_train.to_parquet(train_path)
        print(f"\nTrain output: {train_path}")
        print(f"  Rows: {len(ds_train)}, Size: {os.path.getsize(train_path) / 1024 / 1024:.1f} MB")

    if all_test_rows:
        ds_test = Dataset.from_list(all_test_rows)
        test_path = os.path.join(args.output_dir, "mobile_sft_test.parquet")
        ds_test.to_parquet(test_path)
        print(f"\nTest output: {test_path}")
        print(f"  Rows: {len(ds_test)}, Size: {os.path.getsize(test_path) / 1024 / 1024:.1f} MB")

    # Verification
    print("\n=== Verification ===")
    sample = (all_train_rows or all_test_rows)[0]
    print(f"data_source: {sample['data_source']}")
    print(f"prompt roles: {[m['role'] for m in sample['prompt']]}")
    print(f"images count: {len(sample['images'])}")
    print(f"ground_truth (first 200 chars): {sample['reward_model']['ground_truth'][:200]}...")
    print(f"extra_info: {sample['extra_info']}")


if __name__ == "__main__":
    main()
