"""
Evaluate a vision-language model on the AndroidControl dataset.

Sends each step's screenshot + prompt to an OpenAI-compatible VLM API,
parses the predicted action, and computes evaluation metrics:
  - Action type accuracy
  - Grounding accuracy (predicted click inside the target bounding box)
  - Ancestor grounding accuracy (click inside a text-bearing ancestor bbox)

Requirements:
  - requests

Usage:
  python evaluate_androidcontrol.py \
      --steps ./androidcontrol/steps.jsonl \
      --images ./androidcontrol/images \
      --api-base http://localhost:8000/v1 \
      --model Qwen3-VL-8B \
      --output results.jsonl
"""

import argparse
import base64
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

GROUNDING_ACTIONS = ('click', 'long_press')


# ==============================================================================
# Coordinate Utilities
# ==============================================================================

def normalize_coord(x, y, width, height):
    """Map pixel coordinates to the normalized [0, 1000] range."""
    norm_x = round(x / width * 1000)
    norm_y = round(y / height * 1000)
    return min(max(norm_x, 0), 1000), min(max(norm_y, 0), 1000)


def denormalize_coord(norm_x, norm_y, width, height):
    """Map normalized [0, 1000] coordinates back to pixel space."""
    x = norm_x / 1000 * width
    y = norm_y / 1000 * height
    return x, y


def is_coord_in_bbox(norm_x, norm_y, bbox, width, height):
    """Check if a normalized [0,1000] coordinate falls inside a pixel-space bbox."""
    x, y = denormalize_coord(norm_x, norm_y, width, height)
    left, top, right, bottom = bbox
    return left <= x <= right and top <= y <= bottom


# ==============================================================================
# Ground Truth Construction
# ==============================================================================

def build_ground_truth(step_record):
    """Convert a step record into the expected ground truth action."""
    action = step_record['action']
    action_type = action['action_type']
    width = step_record['screenshot_width']
    height = step_record['screenshot_height']

    if action_type == 'click':
        nx, ny = normalize_coord(action['x'], action['y'], width, height)
        return {"action": "click", "coordinate": [nx, ny]}

    elif action_type == 'long_press':
        nx, ny = normalize_coord(action['x'], action['y'], width, height)
        return {"action": "long_press", "coordinate": [nx, ny], "time": 2}

    elif action_type == 'scroll':
        direction = action['direction']
        cx, cy = 500, 500
        d = 300
        if direction == 'down':
            return {"action": "swipe", "coordinate": [cx, cy + d // 2],
                    "coordinate2": [cx, cy - d // 2], "direction": "up"}
        elif direction == 'up':
            return {"action": "swipe", "coordinate": [cx, cy - d // 2],
                    "coordinate2": [cx, cy + d // 2], "direction": "down"}
        elif direction == 'left':
            return {"action": "swipe", "coordinate": [cx + d // 2, cy],
                    "coordinate2": [cx - d // 2, cy], "direction": "left"}
        elif direction == 'right':
            return {"action": "swipe", "coordinate": [cx - d // 2, cy],
                    "coordinate2": [cx + d // 2, cy], "direction": "right"}

    elif action_type == 'input_text':
        return {"action": "type", "text": action['text']}

    elif action_type == 'navigate_back':
        return {"action": "system_button", "button": "Back"}

    elif action_type == 'navigate_home':
        return {"action": "system_button", "button": "Home"}

    elif action_type == 'open_app':
        return {"action": "open_app", "app_name": action['app_name']}

    elif action_type == 'wait':
        return {"action": "wait", "time": 2}

    return {"action": action_type}


# ==============================================================================
# Evaluation Prompt
# ==============================================================================

SYSTEM_PROMPT = """\
You are a mobile device agent. You interact with a touchscreen device.

The screen coordinates range from 0 to 1000 on both x and y axes.

Available actions:
- click: Click at coordinate (x, y)
- long_press: Long press at coordinate (x, y) for a duration
- swipe: Swipe from coordinate (x, y) to coordinate2 (x2, y2)
- type: Input text into the active input field
- system_button: Press a system button (Back or Home)
- open_app: Open an application by name
- wait: Wait for a specified duration
- terminate: End the task

Response format:
1) Thought: one concise sentence explaining what to do next.
2) Action: a short imperative describing the action.
3) A single <tool_call>...</tool_call> block with the JSON action.

Example:
Thought: I need to tap the search icon in the top right corner.
Action: Click on the search icon.
<tool_call>
{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [920, 85]}}
</tool_call>
"""


def _build_user_message(step_record, image_b64):
    """Build the multimodal user message for a single evaluation step."""
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"}
        },
        {
            "type": "text",
            "text": (
                f"Goal: {step_record['goal']}\n\n"
                f"Current step: {step_record['instruction']}\n\n"
                "Based on the screenshot, output the next action to perform."
            )
        }
    ]


# ==============================================================================
# Response Parsing
# ==============================================================================

def parse_tool_call(response_text):
    """Extract the tool_call JSON from the model's response."""
    match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response_text, re.DOTALL)
    if not match:
        return None
    try:
        call = json.loads(match.group(1))
        return call.get('arguments', call)
    except json.JSONDecodeError:
        return None


# ==============================================================================
# Grounding Evaluation
# ==============================================================================

def evaluate_grounding(pred_coord, step_record):
    """Check if predicted coordinate falls inside the ground truth bounding box.

    Returns:
        dict with hit_target and hit_ancestor, or None if unavailable.
    """
    grounding = step_record.get('grounding')
    if grounding is None or grounding['target_bbox'] is None:
        return None

    width = step_record['screenshot_width']
    height = step_record['screenshot_height']
    norm_x, norm_y = pred_coord

    hit_target = is_coord_in_bbox(norm_x, norm_y, grounding['target_bbox'], width, height)
    hit_ancestor = False
    if grounding['ancestor_bbox']:
        hit_ancestor = is_coord_in_bbox(norm_x, norm_y, grounding['ancestor_bbox'], width, height)

    return {
        'hit_target': hit_target,
        'hit_ancestor': hit_ancestor or hit_target,
    }


# ==============================================================================
# API Interaction
# ==============================================================================

def _call_api(base_url, model_name, step_record, image_b64, temperature=0.0, max_tokens=512):
    """Send a single evaluation request to an OpenAI-compatible API."""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(step_record, image_b64)},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    resp = requests.post(f"{base_url}/chat/completions", json=payload, timeout=120)
    resp.raise_for_status()
    result = resp.json()
    return result["choices"][0]["message"]["content"]


def _evaluate_step(step_record, images_dir, base_url, model_name):
    """Evaluate a single step: call API, parse response, compare to ground truth."""
    img_path = images_dir / step_record['screenshot']
    if not img_path.exists():
        return {'error': 'image_not_found', 'step': step_record}

    image_b64 = base64.b64encode(img_path.read_bytes()).decode('utf-8')

    try:
        response_text = _call_api(base_url, model_name, step_record, image_b64)
    except Exception as e:
        return {'error': str(e), 'step': step_record}

    pred_action = parse_tool_call(response_text)
    gt_action = build_ground_truth(step_record)

    result = {
        'episode_id': step_record['episode_id'],
        'step_idx': step_record['step_idx'],
        'goal': step_record['goal'],
        'instruction': step_record['instruction'],
        'gt_action': gt_action,
        'pred_action': pred_action,
        'raw_response': response_text,
        'action_type_match': False,
        'grounding_result': None,
    }

    if pred_action and gt_action:
        pred_type = pred_action.get('action', '')
        gt_type = gt_action.get('action', '')
        result['action_type_match'] = (pred_type == gt_type)

        if pred_type in ('click', 'long_press') and 'coordinate' in pred_action:
            result['grounding_result'] = evaluate_grounding(pred_action['coordinate'], step_record)

    return result


# ==============================================================================
# Metrics Computation
# ==============================================================================

def compute_metrics(results):
    """Compute aggregate evaluation metrics."""
    total = len(results)
    errors = sum(1 for r in results if 'error' in r)
    valid = [r for r in results if 'error' not in r]

    action_type_correct = sum(1 for r in valid if r['action_type_match'])
    grounding_results = [r['grounding_result'] for r in valid if r['grounding_result'] is not None]
    target_hits = sum(1 for g in grounding_results if g['hit_target'])
    ancestor_hits = sum(1 for g in grounding_results if g['hit_ancestor'])

    return {
        'total_steps': total,
        'errors': errors,
        'valid_steps': len(valid),
        'action_type_accuracy': action_type_correct / len(valid) if valid else 0,
        'grounding_steps': len(grounding_results),
        'grounding_target_accuracy': target_hits / len(grounding_results) if grounding_results else 0,
        'grounding_ancestor_accuracy': ancestor_hits / len(grounding_results) if grounding_results else 0,
    }


# ==============================================================================
# Main
# ==============================================================================

def main():
    if requests is None:
        print('Error: requests library is required. Install with: pip install requests')
        return

    parser = argparse.ArgumentParser(
        description='Evaluate a VLM on the AndroidControl grounding task'
    )
    parser.add_argument('--steps', required=True,
                        help='Path to preprocessed steps.jsonl')
    parser.add_argument('--images', required=True,
                        help='Path to images directory')
    parser.add_argument('--api-base', required=True,
                        help='OpenAI-compatible API base URL')
    parser.add_argument('--model', required=True,
                        help='Model name for API requests')
    parser.add_argument('--limit', type=int, default=None,
                        help='Evaluate only first N steps')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel API workers (default: 4)')
    parser.add_argument('--output', default='results.jsonl',
                        help='Output file for per-step results')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature (default: 0.0)')
    args = parser.parse_args()

    images_dir = Path(args.images)
    steps = []
    with open(args.steps, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(json.loads(line))
                if args.limit and len(steps) >= args.limit:
                    break

    print(f'Loaded {len(steps)} steps for evaluation')
    print(f'API: {args.api_base}')
    print(f'Model: {args.model}')
    print(f'Workers: {args.workers}')
    print()

    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for step in steps:
            future = executor.submit(
                _evaluate_step, step, images_dir, args.api_base, args.model
            )
            futures[future] = step

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f'  Progress: {i + 1}/{len(steps)} ({rate:.1f} steps/s)')

    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    metrics = compute_metrics(results)
    elapsed = time.time() - start_time

    print(f'\n{"=" * 60}')
    print(f'Evaluation complete in {elapsed:.1f}s')
    print(f'{"=" * 60}')
    print(f'  Total steps:               {metrics["total_steps"]}')
    print(f'  Errors:                    {metrics["errors"]}')
    print(f'  Action type accuracy:      {metrics["action_type_accuracy"]:.4f}')
    print(f'  Grounding steps:           {metrics["grounding_steps"]}')
    print(f'  Grounding (target bbox):   {metrics["grounding_target_accuracy"]:.4f}')
    print(f'  Grounding (ancestor bbox): {metrics["grounding_ancestor_accuracy"]:.4f}')
    print(f'{"=" * 60}')

    metrics_path = output_path.with_suffix('.metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'\nMetrics: {metrics_path}')
    print(f'Results: {output_path}')


if __name__ == '__main__':
    main()
