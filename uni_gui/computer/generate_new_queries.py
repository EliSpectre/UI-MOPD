"""
Generate new task queries with a vision-language model (Gemini-compatible API),
based on the screenshots and trajectory of an original query.

Two generation strategies are supported via --mode:

  --mode dimension   (originally "v2")
      For each task, generate 15 queries:
        - 5 MINOR variants (one per fixed dimension, similar difficulty)
        - 10 MAJOR variants (5 dimensions x 2 variants, substantially different)

  --mode freeform    (originally "v3")
      For each task, generate N (default 10) queries by freely diverging across
      the functionality points visible in the trajectory. Each new query keeps
      1-2 functionality points in common with the original but differs in the
      concrete operation / target / parameters, at a similar difficulty.

Both modes:
  - Send up to MAX_SCREENSHOTS representative screenshots (step0 + uniform sampling)
  - Enforce that the new query only references elements visible in the screenshots
    or mentioned in the action steps
  - De-duplicate against previously generated batches and against the queries
    already produced for the same task

Usage:
    python -u generate_new_queries.py --mode dimension
    python -u generate_new_queries.py --mode freeform --workers 50
    python -u generate_new_queries.py --mode freeform --dry-run   # test 2 tasks
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

# ======================== Model API config ========================
# NOTE: Fill in your own vision-language model endpoint compatible with the
# OpenAI/Gemini chat-completions request format. The API key is read from the
# environment variable GEMINI_API_KEY (falls back to the placeholder below).
MODEL_URL = os.environ.get("MODEL_URL", "https://your-model-endpoint/v1/chat/completions")
MODEL_NAME = os.environ.get("MODEL_NAME", "your-model-name")
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
# Provider id header value expected by your gateway (leave as-is if not needed).
MODEL_PROVIDER_ID = os.environ.get("MODEL_PROVIDER_ID", "your-provider-id")

MAX_RETRIES = 50
MAX_SCREENSHOTS = 5
# ==================================================================

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "X-Model-Provider-Id": MODEL_PROVIDER_ID,
    "Content-Type": "application/json",
}

# ==================== Dimension mode: MINOR dimensions ====================
DIMENSIONS_MINOR = [
    {
        "name": "change_target",
        "instruction": """Change the TARGET OBJECT of the operation while keeping the action type the same.
For example: if original operates on "paragraph 1", the new query could operate on "paragraph 3" or "the title" or "the last section".
The new query MUST use a completely different sentence structure and wording — do NOT just swap the object name in the original text.
ONLY reference targets that are actually VISIBLE in the provided screenshots or mentioned in the action steps.""",
    },
    {
        "name": "change_action",
        "instruction": """Change the ACTION/OPERATION while keeping the target object similar.
For example: if original "changes line spacing", the new query could "change font size" or "add bold formatting" on the same element.
The new query MUST be written in a different style — as if a different person wrote it from scratch.
The action must be something the application shown in the screenshots actually supports.""",
    },
    {
        "name": "adjust_complexity",
        "instruction": """Make the task slightly MORE complex by adding one additional sub-step or requirement.
For example: if original just changes one setting, the new query could change that setting AND another related setting.
Write it naturally as a single coherent request — not as a list of steps. Use a fresh writing style.
All referenced elements must exist in the current environment as shown in the screenshots.""",
    },
    {
        "name": "change_params",
        "instruction": """Keep the same type of operation but change the specific PARAMETERS or VALUES.
For example: if original sets "font size 14", the new one could set "font size 18". If original uses color "red", new could use "blue".
IMPORTANT: Do NOT just find-and-replace the value in the original text. Rewrite the entire query from scratch in a completely different style.
The parameters you choose must be valid for the application shown in the screenshots.""",
    },
    {
        "name": "rephrase_and_tweak",
        "instruction": """Rewrite with a different TONE and make a small semantic tweak (slightly different but related goal).
For example: if original formally asks to "enable Do Not Track", the new one could casually say "Hey, I'm worried about tracking cookies. Can you make Chrome stop letting websites track me?"
The semantic meaning should be close but not identical — like a different user asking for a similar thing in their own words.
The task must still be achievable in the environment shown in the screenshots.""",
    },
]

# ==================== Dimension mode: MAJOR dimensions ====================
DIMENSIONS_MAJOR = [
    {
        "name": "change_target_major",
        "instruction": """Change the TARGET to a COMPLETELY DIFFERENT type of element visible in the screenshots.
Do NOT just switch to a similar object (e.g., paragraph 1 → paragraph 3).
Instead, switch to a fundamentally different kind of target that you can see in the environment.
For example: in a spreadsheet, switch from operating on a cell → operating on a chart, or from one column → the sheet tabs.
In a document, switch from a paragraph → a table, or from text → an image.
ONLY reference targets that are actually VISIBLE in the provided screenshots or mentioned in the action steps.
The resulting task should feel like a different use case of the same application, not a minor variant.""",
    },
    {
        "name": "change_action_major",
        "instruction": """Replace the operation with a FUNDAMENTALLY DIFFERENT action from a different feature area of the same application.
Do NOT pick a closely related action (e.g., bold → italic is too similar).
Instead, pick an action from a completely different feature category.
For example: text formatting → inserting objects, page layout → find/replace, cell editing → chart creation, font changes → page numbering.
The action must be something the application shown in the screenshots actually supports.
The new task should exercise a completely different part of the application's functionality.""",
    },
    {
        "name": "adjust_complexity_major",
        "instruction": """Create a SIGNIFICANTLY more complex multi-step task that combines 2-3 different operations from different feature areas.
Combine operations from different feature areas visible in the environment.
For example: "change font" → "format the title in bold, insert a page break after the introduction, and add page numbers at the bottom of every page".
For example: "delete a slide" → "reorganize the presentation by moving slides 3 and 4 after slide 6, then add a transition effect to all slides and insert a new title slide at the beginning".
All referenced elements must exist in the current environment as shown in the screenshots.
Write as a natural, coherent request — not as a numbered list of steps. It should sound like one user request that happens to require multiple actions.""",
    },
    {
        "name": "change_params_major",
        "instruction": """Keep the general category of operation but DRASTICALLY reframe it from a completely different user perspective or use case.
Instead of just changing parameter values (e.g., 14pt → 18pt), reimagine WHY the user would need this operation and describe it from that new perspective.
For example: "change font size to 14" → "I'm preparing a presentation handout for visually impaired attendees, please make the body text significantly larger and ensure the headings are clearly distinguishable from the body."
For example: "enable Do Not Track" → "I'm setting up a shared family computer for my children. Lock down the browser privacy settings as much as possible to protect them from tracking."
The underlying operation should still be achievable in the environment shown in the screenshots, but the framing should be dramatically different.""",
    },
    {
        "name": "batch_operation",
        "instruction": """Expand the single-item operation into a BATCH or conditional operation over multiple objects visible in the environment.
For example: "delete the first slide" → "remove all slides that only have a title with no content beneath it"
"change font of paragraph 1" → "make all headings throughout the document use the same font and size"
"copy this cell" → "copy all cells in column B that contain values greater than 100 to a new sheet"
"bold this text" → "find every instance of a date in the document and make them all bold"
The scope changes from one specific target to multiple targets sharing a property or condition.
IMPORTANT: The objects you reference for the batch operation must be visible in or inferable from the screenshots.
The new query must still be feasible within the same application domain and environment.
Write naturally as a single coherent request.""",
    },
]

# ==================== Dimension mode: prompt templates ====================
PROMPT_TEMPLATE_NO_PRECOND = """You are a GUI task designer creating training data for a computer agent.

Original task query: {query}
Application domain: {domain}

Action steps the agent took to complete the original task:
{steps_summary}

I've provided {num_screenshots} screenshots showing the environment state at various stages of the task execution.
These screenshots show you what elements, data, menus, and objects exist in this environment.

YOUR TASK: Generate ONE new query following this specific dimension:
{dimension_instruction}

CRITICAL CONSTRAINTS:
- The new query MUST be executable in the EXACT same environment shown in the screenshots
- Only reference UI elements, files, data, or objects that are VISIBLE in the screenshots or mentioned in the action steps
- Do NOT invent elements that don't exist in this environment (e.g., don't reference a column that isn't in the spreadsheet, a menu that doesn't exist, or files not shown)
- {modification_level}
- Write in English
- Output ONLY the new query text, nothing else (no quotes, no explanation)"""

PROMPT_TEMPLATE_WITH_PRECOND = """You are a GUI task designer creating training data for a computer agent.

Original task query: {query}
Application domain: {domain}
Environment setup: {precondition_type} — {what_to_prepare}

Action steps the agent took to complete the original task:
{steps_summary}

I've provided {num_screenshots} screenshots showing the environment state at various stages of the task execution.
These screenshots show you what elements, data, menus, and objects exist in this environment.

YOUR TASK: Generate ONE new query following this specific dimension:
{dimension_instruction}

CRITICAL CONSTRAINTS:
- The new query MUST be executable in the EXACT same environment shown in the screenshots
- Only reference UI elements, files, data, or objects that are VISIBLE in the screenshots or mentioned in the action steps
- Do NOT invent elements that don't exist in this environment (e.g., don't reference a column that isn't in the spreadsheet, a menu that doesn't exist, or files not shown)
- {modification_level}
- Write in English
- Output ONLY the new query text, nothing else (no quotes, no explanation)"""

MODIFICATION_LEVEL_MINOR = "The modification should be a small but meaningful change — similar difficulty to the original"
MODIFICATION_LEVEL_MAJOR = "The modification from the original should be SIGNIFICANT — not a minor tweak but a substantially different task"

# ==================== Freeform mode: prompt template ====================
PROMPT_TEMPLATE_FREEFORM = """You are a GUI task designer. Based on the screenshots and trajectory information below,
generate ONE new task query that a user could ask an agent to perform in this EXACT environment.

Application: {domain}
Environment state: {precondition}

Original query: {query}
Original task complexity: approximately {num_steps} steps

The new query should:
- Share at least 1-2 functionality points with the original query (e.g., same feature area like text formatting, same target type like spreadsheet cells, or same interaction pattern like menu navigation)
- But differ in the specific operation, target object, or parameters
- Have SIMILAR difficulty to the original (approximately {num_steps} steps to complete)

Available functionality points observed in this environment:
{functionality_points}

Action history showing what operations this environment supports:
{action_summary}

RULES:
1. The query must be executable in the environment shown in the screenshots
2. Keep at least one functionality point in common with the original (same feature area or interaction type)
3. Only reference UI elements, menus, data, or objects VISIBLE in the screenshots
4. Do NOT directly copy or trivially rephrase the original query — make it genuinely different while staying in a related area
5. Difficulty should be similar to the original (approximately {num_steps} steps)
6. Write in English
7. Output ONLY the new query text, nothing else (no quotes, no explanation)"""

# ==================================================================

_cache_lock = threading.Lock()


def load_cache(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    return {}


def save_cache(cache, cache_file):
    with _cache_lock:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(dict(cache), f, ensure_ascii=False, indent=2)


def load_previous_queries(prev_csvs):
    """Load all previously generated queries from the given CSVs to avoid duplicates."""
    prev_queries = {}
    for csv_path in prev_csvs:
        if not csv_path or not os.path.exists(csv_path):
            print(f"  Previous CSV not found: {csv_path}")
            continue
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                task_id = row["task_id"]
                if task_id not in prev_queries:
                    prev_queries[task_id] = []
                for col_name, val in row.items():
                    if col_name.startswith("new_query_") and val:
                        prev_queries[task_id].append(val)
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


def get_screenshots(task_dir):
    """Get up to MAX_SCREENSHOTS representative screenshots via uniform sampling."""
    all_shots = sorted(
        _glob.glob(os.path.join(task_dir, "screenshot_step*.png")),
        key=lambda p: int(re.search(r'step(\d+)', p).group(1))
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


def get_action_summary(task_data, limit=20):
    """Extract the action history."""
    steps = task_data.get("data", [])
    actions = []
    for step in steps:
        action = step.get("action", "")
        if action:
            actions.append(f"Step {step.get('step', 0)}: {action}")
    return "\n".join(actions[:limit])


def extract_functionality_points(task_data):
    """Extract functionality points from the trajectory's thoughts and actions."""
    points = []
    seen = set()

    for step in task_data.get("data", []):
        # Extract already-used functionality from the action field
        action = step.get("action", "")
        if action and action not in seen:
            points.append(f"[Used] {action}")
            seen.add(action)

        # Extract optional actions and environment descriptions from the thought
        thought = step.get("thought", "")
        if thought:
            lines = thought.split("\n")
            for line in lines:
                line_stripped = line.strip().lstrip("- ")
                if not line_stripped:
                    continue
                # Extract optional actions containing "could" / "can" / "possible"
                if any(kw in line.lower() for kw in ["could", "can ", "possible", "alternative"]):
                    if line_stripped not in seen and len(line_stripped) < 200:
                        points.append(f"[Available] {line_stripped}")
                        seen.add(line_stripped)

    return points[:30]


def get_num_steps(task_data):
    """Get the number of effective steps in the original trajectory."""
    steps = task_data.get("data", [])
    return len([s for s in steps if s.get("is_use", True) and not s.get("is_delete", False)])


def call_gemini(messages, req_id_prefix="newq"):
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
            headers["X-Model-Request-Id"] = f"{req_id_prefix}-{time.time()}"

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


def build_user_content(screenshot_paths, prompt_text):
    """Build the multimodal user message content: screenshots followed by the prompt text."""
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
    return user_content


# ======================== Dimension mode ========================

def generate_one_dimension(task_id, row, cache_key, dimension, is_major, cache,
                           prev_queries, screenshot_paths, steps_summary):
    """Generate one query (minor or major) for dimension mode.

    Returns: (cache_key, result, error)
    """
    if cache_key in cache:
        return cache_key, cache[cache_key], None

    query = row["instruction"]
    domain = row["domain"]
    needs_precondition = row["needs_precondition"]
    precondition_type = row.get("precondition_type", "none")
    what_to_prepare = row.get("what_to_prepare", "nothing")

    modification_level = MODIFICATION_LEVEL_MAJOR if is_major else MODIFICATION_LEVEL_MINOR

    if needs_precondition == "no":
        prompt_text = PROMPT_TEMPLATE_NO_PRECOND.format(
            query=query, domain=domain,
            steps_summary=steps_summary,
            dimension_instruction=dimension["instruction"],
            num_screenshots=len(screenshot_paths),
            modification_level=modification_level,
        )
    else:
        prompt_text = PROMPT_TEMPLATE_WITH_PRECOND.format(
            query=query, domain=domain,
            precondition_type=precondition_type,
            what_to_prepare=what_to_prepare,
            steps_summary=steps_summary,
            dimension_instruction=dimension["instruction"],
            num_screenshots=len(screenshot_paths),
            modification_level=modification_level,
        )

    # Anti-duplication: collect all previously generated queries for this task
    avoid_list = []
    for q in prev_queries.get(task_id, []):
        if q:
            avoid_list.append(q)
    for ck, val in cache.items():
        if ck.startswith(f"{task_id}_") and ck != cache_key and val:
            avoid_list.append(val)

    if avoid_list:
        prompt_text += "\n\nIMPORTANT: The following queries have ALREADY been generated for this task. Your new query MUST be substantially different from ALL of them:\n"
        for aq in avoid_list:
            prompt_text += f"- {aq}\n"
        prompt_text += "\nGenerate something clearly distinct from all the above."

    messages = [{"role": "user", "content": build_user_content(screenshot_paths, prompt_text)}]

    result = call_gemini(messages, req_id_prefix="newq-dim")
    if not result:
        return cache_key, None, "API call failed"

    result = result.strip().strip('"').strip("'")
    with _cache_lock:
        cache[cache_key] = result
    return cache_key, result, None


def generate_task_queries_dimension(task_id, row, cache, prev_queries, dataset_dir):
    """Generate all 15 queries for one task (5 minor + 10 major).

    Minor queries are independent. For major queries, within each dimension the
    'a' and 'b' variants run sequentially so 'b' can see 'a' for de-duplication.
    """
    task_dir = os.path.join(dataset_dir, task_id)
    task_json_path = os.path.join(task_dir, "task.json")

    steps_summary = "Not available"
    screenshot_paths = []

    if os.path.isfile(task_json_path):
        with open(task_json_path, "r", encoding="utf-8") as f:
            task_data = json.load(f)
        steps_summary = get_action_summary(task_data, limit=15)
        screenshot_paths = get_screenshots(task_dir)

    results = []

    # --- 5 minor queries ---
    for dim_idx in range(5):
        cache_key = f"{task_id}_minor_v{dim_idx}"
        ck, result, err = generate_one_dimension(
            task_id, row, cache_key, DIMENSIONS_MINOR[dim_idx],
            is_major=False, cache=cache, prev_queries=prev_queries,
            screenshot_paths=screenshot_paths, steps_summary=steps_summary,
        )
        results.append((ck, result, err))

    # --- 10 major queries (a then b per dimension) ---
    for dim_idx in range(5):
        cache_key_a = f"{task_id}_major_v{dim_idx}_a"
        ck_a, result_a, err_a = generate_one_dimension(
            task_id, row, cache_key_a, DIMENSIONS_MAJOR[dim_idx],
            is_major=True, cache=cache, prev_queries=prev_queries,
            screenshot_paths=screenshot_paths, steps_summary=steps_summary,
        )
        results.append((ck_a, result_a, err_a))

        cache_key_b = f"{task_id}_major_v{dim_idx}_b"
        ck_b, result_b, err_b = generate_one_dimension(
            task_id, row, cache_key_b, DIMENSIONS_MAJOR[dim_idx],
            is_major=True, cache=cache, prev_queries=prev_queries,
            screenshot_paths=screenshot_paths, steps_summary=steps_summary,
        )
        results.append((ck_b, result_b, err_b))

    return results


# ======================== Freeform mode ========================

def generate_one_freeform(task_id, idx, row, cache, prev_queries, screenshot_paths,
                          steps_summary, functionality_points, num_steps):
    """Generate one query for the given task at index idx (freeform mode)."""
    cache_key = f"{task_id}_free_{idx}"
    if cache_key in cache:
        return cache_key, cache[cache_key], None

    query = row["instruction"]
    domain = row["domain"]
    needs_precondition = row["needs_precondition"]
    precondition_type = row.get("precondition_type", "none")
    what_to_prepare = row.get("what_to_prepare", "nothing")

    if needs_precondition == "yes":
        precondition = f"{precondition_type} — {what_to_prepare}"
    else:
        precondition = "Clean/default desktop, no special setup needed"

    prompt_text = PROMPT_TEMPLATE_FREEFORM.format(
        domain=domain,
        precondition=precondition,
        query=query,
        num_steps=num_steps,
        functionality_points="\n".join(functionality_points),
        action_summary=steps_summary,
    )

    # Anti-duplication: previously generated queries plus earlier ones in this batch
    avoid_list = []
    avoid_list.extend(prev_queries.get(task_id, []))
    for i in range(idx):
        ck = f"{task_id}_free_{i}"
        if ck in cache and cache[ck]:
            avoid_list.append(cache[ck])

    if avoid_list:
        prompt_text += "\n\nIMPORTANT: The following queries have ALREADY been generated for this task. Your new query MUST be substantially different from ALL of them:\n"
        for aq in avoid_list:
            prompt_text += f"- {aq}\n"
        prompt_text += "\nGenerate something clearly distinct from all the above."

    messages = [{"role": "user", "content": build_user_content(screenshot_paths, prompt_text)}]

    result = call_gemini(messages, req_id_prefix="newq-free")
    if not result:
        return cache_key, None, "API call failed"

    result = result.strip().strip('"').strip("'")
    with _cache_lock:
        cache[cache_key] = result
    return cache_key, result, None


def generate_task_queries_freeform(task_id, row, cache, prev_queries, dataset_dir,
                                   queries_per_task):
    """Generate all freeform queries for one task (sequential to maintain anti-duplication)."""
    task_dir = os.path.join(dataset_dir, task_id)
    task_json_path = os.path.join(task_dir, "task.json")

    steps_summary = "Not available"
    screenshot_paths = []
    functionality_points = []
    num_steps = 5

    if os.path.isfile(task_json_path):
        with open(task_json_path, "r", encoding="utf-8") as f:
            task_data = json.load(f)
        steps_summary = get_action_summary(task_data, limit=20)
        screenshot_paths = get_screenshots(task_dir)
        functionality_points = extract_functionality_points(task_data)
        num_steps = get_num_steps(task_data)

    results = []
    for idx in range(queries_per_task):
        ck, result, err = generate_one_freeform(
            task_id, idx, row, cache, prev_queries,
            screenshot_paths, steps_summary, functionality_points, num_steps,
        )
        results.append((ck, result, err))

    return results


# ======================== CSV input ========================

def load_precondition_info(precond_csv):
    """Load precondition info from the original collected CSV (dimension mode helper)."""
    precond_map = {}
    if precond_csv and os.path.exists(precond_csv):
        with open(precond_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                precond_map[row["task_id"]] = {
                    "needs_precondition": row.get("needs_precondition", "no"),
                    "precondition_type": row.get("precondition_type", "none"),
                    "what_to_prepare": row.get("what_to_prepare", "nothing"),
                }
    return precond_map


def read_csv_input(csv_input, precond_csv=None):
    """Read the input CSV. If precond_csv is given, enrich rows with precondition info.

    Also normalizes the query column: some CSVs use "original_query" instead of
    "instruction".
    """
    precond_map = load_precondition_info(precond_csv) if precond_csv else {}
    rows = []
    with open(csv_input, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "original_query" in row and "instruction" not in row:
                row["instruction"] = row["original_query"]
            task_id = row["task_id"]
            if task_id in precond_map:
                row["needs_precondition"] = precond_map[task_id]["needs_precondition"]
                row["precondition_type"] = precond_map[task_id]["precondition_type"]
                row["what_to_prepare"] = precond_map[task_id]["what_to_prepare"]
            else:
                row.setdefault("needs_precondition", "no")
                row.setdefault("precondition_type", "none")
                row.setdefault("what_to_prepare", "nothing")
            rows.append(row)
    return rows


# ======================== Output CSV writers ========================

def write_output_dimension(rows, cache, csv_output):
    os.makedirs(os.path.dirname(csv_output), exist_ok=True)
    with open(csv_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_id", "domain", "original_query",
            "new_query_1_change_target", "new_query_2_change_action",
            "new_query_3_adjust_complexity", "new_query_4_change_params",
            "new_query_5_rephrase_tweak",
            "new_query_6_change_target_major_a", "new_query_7_change_target_major_b",
            "new_query_8_change_action_major_a", "new_query_9_change_action_major_b",
            "new_query_10_adjust_complexity_major_a", "new_query_11_adjust_complexity_major_b",
            "new_query_12_change_params_major_a", "new_query_13_change_params_major_b",
            "new_query_14_batch_operation_a", "new_query_15_batch_operation_b",
        ])
        for row in rows:
            task_id = row["task_id"]
            queries = []
            for dim_idx in range(5):
                queries.append(cache.get(f"{task_id}_minor_v{dim_idx}", ""))
            for dim_idx in range(5):
                for variant in ["a", "b"]:
                    queries.append(cache.get(f"{task_id}_major_v{dim_idx}_{variant}", ""))
            writer.writerow([task_id, row["domain"], row["instruction"]] + queries)


def write_output_freeform(rows, cache, csv_output, queries_per_task):
    os.makedirs(os.path.dirname(csv_output), exist_ok=True)
    with open(csv_output, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["task_id", "domain", "original_query"]
        header += [f"new_query_{i+1}" for i in range(queries_per_task)]
        writer.writerow(header)
        for row in rows:
            task_id = row["task_id"]
            queries = [cache.get(f"{task_id}_free_{i}", "") for i in range(queries_per_task)]
            writer.writerow([task_id, row["domain"], row["instruction"]] + queries)


# ======================== Job planning ========================

def count_cached_dimension(row, cache):
    keys = ([f"{row['task_id']}_minor_v{d}" for d in range(5)] +
            [f"{row['task_id']}_major_v{d}_{v}" for d in range(5) for v in ["a", "b"]])
    return sum(1 for k in keys if k in cache)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dimension", "freeform"], required=True,
                        help="dimension: 5 minor + 10 major variants per task; "
                             "freeform: N free-divergence queries per task")
    parser.add_argument("--workers", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="Process only 2 tasks")
    parser.add_argument("--queries-per-task", type=int, default=10,
                        help="Freeform mode only: number of queries per task")

    # Path arguments (placeholders below; override on the command line or via the
    # accompanying bash script).
    parser.add_argument("--csv-input", type=str, default="/path/to/input_tasks.csv",
                        help="Input CSV with at least task_id, domain, instruction/original_query")
    parser.add_argument("--csv-output", type=str, default="/path/to/output/new_queries.csv",
                        help="Output CSV path")
    parser.add_argument("--cache-file", type=str, default="/path/to/output/generate_cache.json",
                        help="JSON cache for incremental/resumable generation")
    parser.add_argument("--dataset-dir", type=str, default="/path/to/dataset/best",
                        help="Directory containing per-task folders with task.json and screenshots")
    parser.add_argument("--prev-csv", type=str, nargs="*", default=[],
                        help="Previously generated query CSV(s) for anti-duplication")
    parser.add_argument("--precond-csv", type=str, default=None,
                        help="Optional CSV providing precondition info (task_id -> needs_precondition/...)")
    args = parser.parse_args()

    is_dimension = args.mode == "dimension"
    queries_per_task = 15 if is_dimension else args.queries_per_task

    rows = read_csv_input(args.csv_input, precond_csv=args.precond_csv)
    print(f"Loaded {len(rows)} tasks from CSV")

    cache = load_cache(args.cache_file)
    print(f"Cache has {len(cache)} entries")

    prev_queries = load_previous_queries(args.prev_csv)

    if args.dry_run:
        rows = rows[:2]
        print("DRY RUN: processing only 2 tasks")

    # Determine which tasks still need work
    jobs = []
    for row in rows:
        task_id = row["task_id"]
        if is_dimension:
            needed = queries_per_task - count_cached_dimension(row, cache)
        else:
            needed = sum(1 for i in range(queries_per_task)
                         if f"{task_id}_free_{i}" not in cache)
        if needed > 0:
            jobs.append((task_id, row))

    total_queries = len(rows) * queries_per_task
    if is_dimension:
        cached_count = sum(count_cached_dimension(row, cache) for row in rows)
    else:
        cached_count = sum(
            1 for row in rows for i in range(queries_per_task)
            if f"{row['task_id']}_free_{i}" in cache
        )
    print(f"Tasks needing work: {len(jobs)}/{len(rows)}")
    print(f"Total query slots: {total_queries}, already cached: {cached_count}")

    if not jobs:
        print("All tasks already cached, writing CSV...")
    else:
        success = 0
        fail = 0
        failures = []

        def submit(executor, tid, row):
            if is_dimension:
                return executor.submit(generate_task_queries_dimension, tid, row,
                                       cache, prev_queries, args.dataset_dir)
            return executor.submit(generate_task_queries_freeform, tid, row,
                                   cache, prev_queries, args.dataset_dir, queries_per_task)

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {submit(executor, tid, row): tid for tid, row in jobs}
            done_count = 0
            for future in as_completed(futures):
                tid = futures[future]
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
                    fail += queries_per_task
                    failures.append((tid, str(e)))
                    print(f"  [FAIL] task {tid[:8]}: {e}")

                if done_count % 10 == 0 or done_count == len(jobs):
                    print(f"  Tasks done: {done_count}/{len(jobs)} (queries ok={success}, fail={fail})")
                save_cache(cache, args.cache_file)

        save_cache(cache, args.cache_file)
        print(f"\nAPI calls done: success={success}, fail={fail}")

        if failures:
            print(f"\nFailed ({len(failures)}):")
            for item in failures[:20]:
                print(f"  {item}")

    # Write output CSV
    if is_dimension:
        write_output_dimension(rows, cache, args.csv_output)
    else:
        write_output_freeform(rows, cache, args.csv_output, queries_per_task)

    print(f"\nOutput written to: {args.csv_output}")
    filled = (sum(count_cached_dimension(row, cache) for row in rows) if is_dimension
              else sum(1 for row in rows for i in range(queries_per_task)
                       if f"{row['task_id']}_free_{i}" in cache))
    print(f"Total cells filled: {filled}/{total_queries}")

    if args.dry_run and rows:
        print(f"\n{'='*60}")
        for row in rows:
            task_id = row["task_id"]
            print(f"\nTask: {task_id}")
            print(f"Domain: {row['domain']}")
            print(f"Original: {row['instruction']}")
            if is_dimension:
                print("  --- Minor ---")
                for i, dim in enumerate(DIMENSIONS_MINOR):
                    q = cache.get(f"{task_id}_minor_v{i}", "N/A")
                    print(f"  [{dim['name']}]: {q}")
                print("  --- Major ---")
                for i, dim in enumerate(DIMENSIONS_MAJOR):
                    qa = cache.get(f"{task_id}_major_v{i}_a", "N/A")
                    qb = cache.get(f"{task_id}_major_v{i}_b", "N/A")
                    print(f"  [{dim['name']}_a]: {qa}")
                    print(f"  [{dim['name']}_b]: {qb}")
            else:
                for i in range(queries_per_task):
                    q = cache.get(f"{task_id}_free_{i}", "N/A")
                    print(f"  [query_{i+1}]: {q}")


if __name__ == "__main__":
    main()
