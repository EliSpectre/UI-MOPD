"""
OpenCUA Data Review Server
Usage:
    python server.py [--data_dir PATH] [--port PORT]
"""

import argparse
import json
import os
import random
from flask import Flask, render_template, jsonify, send_file, abort, request

app = Flask(__name__)

DATA_DIR = ""
MAX_TASKS = 20
TASK_CACHE = None


def build_task_index():
    """Build lightweight index, randomly sample MAX_TASKS from all available."""
    global TASK_CACHE
    if TASK_CACHE is not None:
        return TASK_CACHE

    print(f"Building task index (random {MAX_TASKS}) ...")
    all_dirs = [name for name in os.listdir(DATA_DIR)
                if os.path.isdir(os.path.join(DATA_DIR, name))
                and os.path.isfile(os.path.join(DATA_DIR, name, "task.json"))]
    print(f"  Found {len(all_dirs)} valid task directories")

    if MAX_TASKS > 0:
        sampled = random.sample(all_dirs, min(MAX_TASKS, len(all_dirs)))
    else:
        sampled = all_dirs

    tasks = []
    for name in sampled:
        task_json_path = os.path.join(DATA_DIR, name, "task.json")
        try:
            with open(task_json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            tasks.append({
                "dir_name": name,
                "episode_id": meta.get("episode_id", name),
                "app": meta.get("app", ""),
                "query": meta.get("query", "")[:120],
                "num_steps": len(meta.get("data", [])),
                "verified": meta.get("verified", None),
            })
        except Exception:
            pass

    TASK_CACHE = tasks
    print(f"Index built: {len(tasks)} tasks (sampled from {len(all_dirs)})")
    return tasks


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/tasks")
def api_tasks():
    tasks = build_task_index()
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 100))
    app_filter = request.args.get("app", "")
    keyword = request.args.get("keyword", "").lower()
    verified = request.args.get("verified", "")

    filtered = tasks
    if app_filter:
        filtered = [t for t in filtered if t["app"] == app_filter]
    if keyword:
        filtered = [t for t in filtered if keyword in t["query"].lower() or keyword in t["episode_id"].lower()]
    if verified == "true":
        filtered = [t for t in filtered if t["verified"] is True]
    elif verified == "false":
        filtered = [t for t in filtered if t["verified"] is not True]

    total = len(filtered)
    start = (page - 1) * per_page
    page_tasks = filtered[start:start + per_page]

    apps = sorted(set(t["app"] for t in tasks))
    return jsonify({
        "tasks": page_tasks,
        "apps": apps,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page,
    })


@app.route("/api/tasks/<dir_name>")
def api_task_detail(dir_name):
    task_json = os.path.join(DATA_DIR, dir_name, "task.json")
    if not os.path.isfile(task_json):
        abort(404)
    with open(task_json, "r", encoding="utf-8") as f:
        task = json.load(f)
    return jsonify(task)


@app.route("/api/images/<dir_name>/<filename>")
def api_image(dir_name, filename):
    path = os.path.join(DATA_DIR, dir_name, filename)
    if not os.path.isfile(path):
        abort(404)
    return send_file(path, mimetype="image/png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OpenCUA Data Review Server",
        usage="python server.py [DATA_DIR] [--port PORT] [--max_tasks N]"
    )
    parser.add_argument("data_dir", nargs="?", default=os.path.join(os.path.dirname(__file__), "..", "case"),
                        help="Path to the data directory (e.g. E:\\dataset\\UI-MOPD\\OpenCUA)")
    parser.add_argument("--port", type=int, default=5020)
    parser.add_argument("--max_tasks", type=int, default=20)
    args = parser.parse_args()
    DATA_DIR = os.path.abspath(args.data_dir)
    MAX_TASKS = args.max_tasks
    print(f"Data directory: {DATA_DIR}")
    print(f"Max tasks: {MAX_TASKS}")
    build_task_index()
    app.run(host="0.0.0.0", port=args.port, debug=False)
