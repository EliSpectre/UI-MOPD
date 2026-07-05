#!/usr/bin/env python3
"""
Condense verbose model reasoning into a concise qwen3-vl-style thought.

Approach: take the original (often long) chain-of-thought as reference input and
ask the model to distill it into a compact, structured thought. The original
thought is preserved into the raw_thought field; the new thought overwrites the
thought field.

Operates in place on each task.json under the target directories.

Model API config is read from environment variables (see the bash wrapper):
    MODEL_URL, MODEL_NAME, MODEL_PROVIDER_ID, GEMINI_API_KEY

Usage:
    python regenerate_thought.py                       # 100 workers
    python regenerate_thought.py --workers 50
    python regenerate_thought.py --dry-run             # test 1 step
    python regenerate_thought.py --resume              # skip steps already done

    # Multiple target dirs:
    python regenerate_thought.py --input-dir /path/to/dir1 /path/to/dir2
"""

import argparse
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ======================== Config ========================
# Target directories; each contains per-episode folders with task.json.
TARGET_DIRS = [
    "/path/to/dataset/dir1",
]

# ---- Model API config (read from environment, with placeholder defaults) ----
MODEL_URL = os.environ.get("MODEL_URL", "https://your-model-endpoint/v1/chat/completions")
MODEL_NAME = os.environ.get("MODEL_NAME", "your-model-name")
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
MODEL_PROVIDER_ID = os.environ.get("MODEL_PROVIDER_ID", "your-provider-id")
# ========================================================

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "X-Model-Provider-Id": MODEL_PROVIDER_ID,
    "Content-Type": "application/json",
}

STYLE_EXAMPLES = """## Example 1:
With no prior actions taken, the document remains open in Writer with the caret in the first paragraph. To make the first three words left-aligned and the rest right-aligned using tab stops, I need to set appropriate tab stops for all the paragraphs where this layout is desired. In Writer, tab stop changes via the ruler or paragraph settings apply to the selection; if nothing is selected, they affect only the current paragraph.

Plan:
1) Select all the content so that tab stop settings will apply to every line uniformly.
2) Add a right-aligned tab stop near the right margin using the ruler or Paragraph > Tabs.
3) For each line, insert a tab after the third word so the following text aligns to the right tab stop.

Possible next actions:
- Open the Paragraph dialog and adjust tab stops.
- Click on the ruler to set a right tab stop.
- Select all text to ensure subsequent tab stop changes apply to the entire document.

The most logical immediate step is to select all text. This ensures that when I set tab stops next, they affect all lines at once rather than just the current paragraph. After selecting all, I'll be able to set a single right tab stop and then insert tabs at the correct positions in each line.

Expected consequence:
- All the text in the document becomes highlighted, and the status bar will show the number of selected words/characters. Subsequent tab stop modifications will apply to every paragraph in the selection.

## Example 2:
No previous actions have been taken; the browser is already on the Drugs.com homepage. The goal is to browse the natural products database. From experience with this site layout, the "Drugs A-Z" hub typically provides category links, including "Natural Products." Alternatives include using the search bar to type "Natural Products" or opening "More…" to look for it, but the most straightforward path is likely under "Drugs A-Z."

Possible next actions:
- Click "Drugs A-Z" in the top navigation to reach the comprehensive drug and topic index where "Natural Products" is usually listed.
- Use the search field to search for "Natural Products."
- Click "More…" to see additional sections.

Clicking "Drugs A-Z" is the best next step because it directly leads to an index page that includes a "Natural Products" link, minimizing steps to reach the database. After clicking, I expect to be taken to the Drugs & Medications A to Z page where I can then select "Natural Products" from the list of categories."""

SYSTEM_PROMPT = f"""You are a thought process condenser for a GUI agent. Given a screenshot, task instruction, previous actions, the CORRECT next action, and the agent's ORIGINAL verbose reasoning, condense the reasoning into a concise thought in the target format.

## Output Style (STRICTLY follow this structure):

1. **Brief context** (1-2 sentences): What is the current state? What has been done so far?
2. **Plan** (optional, only for multi-step reasoning): Numbered steps for the overall approach.
3. **Possible next actions** (2-3 bullet points): List plausible alternatives.
4. **Best choice reasoning** (1-2 sentences): Why the chosen action is optimal over alternatives.
5. **Expected consequence** (1 sentence): What should happen after executing this action.

## Style Rules:
- Be CONCISE. No verbose descriptions of the screen. No "I can see..." monologues.
- Focus on REASONING and DECISION-MAKING, not observation.
- Use professional, direct language.
- Do NOT repeat the instruction verbatim.
- Do NOT describe UI elements in exhaustive detail.
- Output ONLY the thought text, no Action or tool_call.
- PRESERVE the key reasoning logic from the original thinking, just make it concise.
- REMOVE all repetition, self-correction, and meta-commentary from the original.

## Reference Examples:
{STYLE_EXAMPLES}"""


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_user_message(query, history_actions, action, plan, image_path, original_thought):
    """Build the user message with image + context + original reasoning."""
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

    plan_json = json.dumps(plan, ensure_ascii=False)

    # Truncate an over-long original chain-of-thought (keep the first 2000 chars)
    truncated_thought = original_thought[:2000] if len(original_thought) > 2000 else original_thought

    text = (
        f"Task instruction: {query}\n\n"
        f"Previous actions:\n{history_text}\n\n"
        f"CORRECT next action: {action}\n"
        f"CORRECT tool_call: {plan_json}\n\n"
        f"--- Original verbose reasoning (condense this) ---\n"
        f"{truncated_thought}\n"
        f"--- End of original reasoning ---\n\n"
        f"Condense the above reasoning into the target format. "
        f"Keep the key logic and decision-making, remove verbose observations and repetition. "
        f"Output ONLY the condensed thought text."
    )
    parts.append({"type": "text", "text": text})

    return parts


def call_gemini(messages, max_retries=10):
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
            headers["X-Model-Request-Id"] = f"thought-{time.time()}"

            resp = requests.post(
                MODEL_URL, headers=headers, json=payload, timeout=120
            )
            if resp.status_code == 200:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()
            elif resp.status_code == 429:
                wait = min(5 * (attempt + 1), 60)
                time.sleep(wait)
            else:
                wait = min(3 * (attempt + 1), 30)
                print(f"    [Retry {attempt+1}/{max_retries}] API error {resp.status_code}, waiting {wait}s...")
                time.sleep(wait)
        except Exception as e:
            wait = min(3 * (attempt + 1), 30)
            print(f"    [Retry {attempt+1}/{max_retries}] {e}, waiting {wait}s...")
            time.sleep(wait)
    return None


def generate_one_step(args_tuple):
    """Process a single step."""
    task_dir, step_idx, query, history_actions, action, plan, image_path, original_thought = args_tuple
    user_content = build_user_message(query, history_actions, action, plan, image_path, original_thought)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    new_thought = call_gemini(messages)
    return task_dir, step_idx, new_thought


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, nargs='+', default=TARGET_DIRS,
                        help="Target directories (one or more)")
    parser.add_argument("--workers", type=int, default=100, help="Concurrent workers (default 100)")
    parser.add_argument("--dry-run", action="store_true", help="Test 1 step only")
    parser.add_argument("--resume", action="store_true", help="Skip steps where raw_thought is already filled")
    args = parser.parse_args()

    target_dirs = args.input_dir

    task_dirs = []
    for target_dir in target_dirs:
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

    for task_dir in task_dirs:
        task_json_path = os.path.join(task_dir, "task.json")
        with open(task_json_path, "r", encoding="utf-8") as f:
            task = json.load(f)
        all_tasks[task_dir] = task

        query = task["query"]
        data_steps = task["data"]
        for idx, step in enumerate(data_steps):
            # resume mode: a filled raw_thought means this step was already processed
            if args.resume and step.get("raw_thought", "").strip():
                continue

            action = step.get("action", "")
            plan = step.get("plan", {})
            screenshot = step.get("screenshot", "")
            image_path = os.path.join(task_dir, screenshot)
            history_actions = [data_steps[i]["action"] for i in range(idx)]
            original_thought = step.get("thought", "")

            step_jobs.append((task_dir, idx, query, history_actions, action, plan, image_path, original_thought))

    total_steps = len(step_jobs)
    print(f"Steps to process: {total_steps}")
    if args.resume:
        already = sum(len(t["data"]) for t in all_tasks.values()) - total_steps
        print(f"Already done (skipped): {already}")

    if total_steps == 0:
        print("Nothing to do.")
        return

    if args.dry_run:
        job = step_jobs[0]
        task_dir, step_idx, query, history_actions, action, plan, image_path, original_thought = job
        task = all_tasks[task_dir]
        step = task["data"][step_idx]
        print(f"\n=== DRY RUN: {task['episode_id']} step {step['step']} ===")
        print(f"Query: {query}")
        print(f"Action: {action}")
        print(f"Screenshot: {image_path}")
        print(f"Original thought (first 200): {original_thought[:200]}...")
        print(f"\nCalling model...")
        _, _, new_thought = generate_one_step(job)
        print(f"\nGenerated thought:\n{new_thought if new_thought else 'FAILED'}")
        return

    success = 0
    failed_list = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(generate_one_step, job): job for job in step_jobs}
        done_count = 0

        for f in as_completed(futures):
            done_count += 1
            try:
                task_dir, step_idx, new_thought = f.result()
                task = all_tasks[task_dir]
                step = task["data"][step_idx]

                if new_thought:
                    step["raw_thought"] = step.get("thought", "")
                    step["thought"] = new_thought
                    success += 1
                else:
                    failed_list.append({
                        "task_id": task["episode_id"],
                        "task_dir": task_dir,
                        "step": step["step"],
                        "step_idx": step_idx,
                    })
            except Exception as e:
                job = futures[f]
                failed_list.append({
                    "task_id": os.path.basename(job[0]),
                    "task_dir": job[0],
                    "step_idx": job[1],
                    "error": str(e),
                })

            if done_count % 100 == 0 or done_count == total_steps:
                print(f"  Progress: {done_count}/{total_steps} steps done, {success} OK, {len(failed_list)} failed")

    print("\nSaving task.json files...")
    for task_dir, task in all_tasks.items():
        task_json_path = os.path.join(task_dir, "task.json")
        with open(task_json_path, "w", encoding="utf-8") as f:
            json.dump(task, f, ensure_ascii=False, indent=4)

    print(f"\n{'='*50}")
    print(f"Done. {success}/{total_steps} steps regenerated successfully.")
    print(f"Failed: {len(failed_list)}")

    if failed_list:
        report_path = os.path.join(os.path.dirname(target_dirs[0]), "thought_gen_failures.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(failed_list, f, ensure_ascii=False, indent=2)
        print(f"Failure report saved to: {report_path}")
        print(f"\nFailed steps (first 20):")
        for item in failed_list[:20]:
            print(f"  {item.get('task_dir','?')} step={item.get('step', item.get('step_idx','?'))}")
        if len(failed_list) > 20:
            print(f"  ... and {len(failed_list) - 20} more")
        print(f"\nTo retry failed ones, run: python regenerate_thought.py --resume --workers {args.workers}")


if __name__ == "__main__":
    main()
