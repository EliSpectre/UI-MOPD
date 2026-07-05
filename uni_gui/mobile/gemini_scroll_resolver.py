# -*- coding: utf-8 -*-
"""Call a teacher vision-language model to turn a directional scroll into a
swipe with concrete coordinates (0-1000 normalized).

For each scroll step: send that step's observation screenshot + direction ->
ask for a pair of swipe start/end coordinates.
Retry up to RETRIES times per step; on persistent failure -> fall back to a
geometric default and flag the failure (the caller then sets is_reviewed=True).

Model API config is read from environment variables (see the bash wrappers):
    MODEL_URL, MODEL_NAME, MODEL_PROVIDER_ID, GEMINI_API_KEY
"""
import base64
import json
import os
import re

import requests

# ---- Model API config (read from environment, with placeholder defaults) ----
MODEL_URL = os.environ.get("MODEL_URL", "https://your-model-endpoint/v1/chat/completions")
MODEL_NAME = os.environ.get("MODEL_NAME", "your-model-name")
API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
MODEL_PROVIDER_ID = os.environ.get("MODEL_PROVIDER_ID", "your-provider-id")

RETRIES = 50
TIMEOUT = 120

# direction -> geometric fallback swipe (0-1000). scroll is opposite to the swipe gesture:
# scroll down (to view lower content) = finger swipes up (y decreases).
GEOMETRIC_FALLBACK = {
    "down":  ([500, 750], [500, 250]),
    "up":    ([500, 250], [500, 750]),
    "left":  ([250, 500], [750, 500]),
    "right": ([750, 500], [250, 500]),
}


def _headers():
    return {
        "Authorization": f"Bearer {API_KEY}",
        "X-Model-Provider-Id": MODEL_PROVIDER_ID,
        "Content-Type": "application/json",
        "X-Model-Request-Id": "scroll-resolve",
    }


def _build_prompt(direction):
    d = direction.lower()
    inverse = {
        "down": "reveal lower content, so the finger swipes UP (y decreases)",
        "up": "reveal upper content, so the finger swipes DOWN (y increases)",
        "left": "reveal content to the right, so the finger swipes LEFT (x decreases)",
        "right": "reveal content to the left, so the finger swipes RIGHT (x increases)",
    }.get(d, "perform the swipe corresponding to the scroll direction")
    return (
        "This is an Android phone screenshot. Coordinates are normalized to 0-1000 "
        "with origin at the top-left corner.\n"
        f"The agent decided to SCROLL {direction.upper()} on the scrollable area. "
        f"A scroll {direction} means: {inverse}.\n"
        "Pick a swipe that lands inside the main scrollable content region (avoid the "
        "status bar and navigation bar).\n"
        'Respond with ONLY a JSON object, no prose:\n'
        '{"coordinate":[x,y],"coordinate2":[x2,y2]}\n'
        "where coordinate is the swipe start and coordinate2 the swipe end, both in 0-1000."
    )


def _parse_coords(content):
    """Parse {"coordinate":[..],"coordinate2":[..]} from model output. None on failure."""
    if not content:
        return None
    # Strip a possible ```json wrapper
    m = re.search(r"\{[^{}]*coordinate2[^{}]*\}", content, re.DOTALL)
    blob = m.group(0) if m else content
    try:
        obj = json.loads(blob)
    except (ValueError, TypeError):
        # Fallback: regex-grab two coordinate pairs
        nums = re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", content)
        if len(nums) >= 2:
            c1 = [int(nums[0][0]), int(nums[0][1])]
            c2 = [int(nums[1][0]), int(nums[1][1])]
            return _validate(c1, c2)
        return None
    c1 = obj.get("coordinate")
    c2 = obj.get("coordinate2")
    return _validate(c1, c2)


def _validate(c1, c2):
    def ok(c):
        return (isinstance(c, (list, tuple)) and len(c) == 2
                and all(isinstance(v, (int, float)) and 0 <= v <= 1000 for v in c))
    if ok(c1) and ok(c2):
        return [int(round(c1[0])), int(round(c1[1]))], [int(round(c2[0])), int(round(c2[1]))]
    return None


def resolve_scroll(image_path, direction, retries=RETRIES, session=None):
    """Return (coordinate, coordinate2, used_fallback: bool).

    Success: valid coordinates from the model, used_fallback=False.
    Failure (all retries fail): geometric defaults, used_fallback=True.
    """
    direction = (direction or "down").lower()
    sess = session or requests
    try:
        b64 = base64.b64encode(open(image_path, "rb").read()).decode()
    except OSError:
        c1, c2 = GEOMETRIC_FALLBACK.get(direction, GEOMETRIC_FALLBACK["down"])
        return c1, c2, True

    prompt = _build_prompt(direction)
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
        "stream": False,
    }

    for _ in range(retries):
        try:
            r = sess.post(MODEL_URL, headers=_headers(), json=payload, timeout=TIMEOUT)
            if r.status_code != 200:
                continue
            content = r.json()["choices"][0]["message"]["content"]
            parsed = _parse_coords(content)
            if parsed:
                return parsed[0], parsed[1], False
        except Exception:
            continue

    c1, c2 = GEOMETRIC_FALLBACK.get(direction, GEOMETRIC_FALLBACK["down"])
    return c1, c2, True


if __name__ == "__main__":
    import sys
    img = sys.argv[1]
    d = sys.argv[2] if len(sys.argv) > 2 else "down"
    print(resolve_scroll(img, d, retries=3))
