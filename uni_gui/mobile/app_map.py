# -*- coding: utf-8 -*-
"""Scan the MobileWorld task definition directory at runtime to build a
TaskClassName -> app (category directory) mapping.

No hardcoding: the app name comes directly from the definitions/<app>/ subdirectory name.
"""
import os
import re
import glob

# Path to the MobileWorld task definitions directory. Edit to your environment.
DEFINITIONS_DIR = "/path/to/MobileWorld/src/mobile_world/tasks/definitions"

_CACHE = None


def build_task_app_map(definitions_dir=DEFINITIONS_DIR):
    """Scan `class XxxTask(` in definitions/<app>/**/*.py, return {ClassName: app}."""
    mapping = {}
    if not os.path.isdir(definitions_dir):
        return mapping
    for app in sorted(os.listdir(definitions_dir)):
        app_dir = os.path.join(definitions_dir, app)
        if not os.path.isdir(app_dir) or app.startswith("_") or app == "__pycache__":
            continue
        for py in glob.glob(os.path.join(app_dir, "**", "*.py"), recursive=True):
            try:
                txt = open(py, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            for m in re.finditer(r"class\s+(\w+)\s*\(", txt):
                mapping.setdefault(m.group(1), app)
    return mapping


def task_prefix_from_folder(folder_name):
    """Extract the task class-name prefix from a source folder name,
    e.g. AcceptMeetingTask_v1_rephrase_2 -> AcceptMeetingTask."""
    m = re.match(r"(.+?)_v\d", folder_name)
    if m:
        return m.group(1)
    # No _vN segment; fall back to stripping the _rephrase_N suffix
    return re.sub(r"_rephrase_\d+$", "", folder_name)


def app_for_folder(folder_name):
    """Return the app for this trajectory folder; None if not found."""
    global _CACHE
    if _CACHE is None:
        _CACHE = build_task_app_map()
    prefix = task_prefix_from_folder(folder_name)
    if prefix in _CACHE:
        return _CACHE[prefix]
    # Fall back to a unique startswith match
    cands = [c for c in _CACHE if c.startswith(prefix)]
    if len(cands) == 1:
        return _CACHE[cands[0]]
    return None


if __name__ == "__main__":
    m = build_task_app_map()
    print(f"classes: {len(m)}")
    for k, v in sorted(m.items())[:10]:
        print(f"  {k} -> {v}")
