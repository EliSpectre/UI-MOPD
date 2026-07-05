# $\color{#FF6700}{\textsf{Uni-GUI Dataset Examples}}$

Example trajectories showing how Uni-GUI data is structured for both desktop and mobile platforms.

Full dataset available at: [https://huggingface.co/UI-MOPD](https://huggingface.co/UI-MOPD)

---

## Data Structure

Each trajectory folder follows this layout:

```
<episode_id>/
├── task.json               # Metadata + per-step records
├── screenshot_step0.png    # Screenshot before step 1
├── screenshot_step1.png    # Screenshot before step 2
└── ...
```

## task.json Schema

| Field | Type | Description |
|-------|------|-------------|
| `task` | string | Data source name (e.g. "OpenCUA", "MobileWorld") |
| `app` | string | Application context (e.g. "OS", "settings", "chrome") |
| `device` | string | Platform: `"computer"` or `"mobile"` |
| `screen_resolution` | [int, int] | Screen size in pixels [width, height] |
| `query` | string | Natural language task instruction |
| `episode_id` | string | Unique trajectory identifier |
| `verified` | bool | Whether the trajectory has been human-verified |
| `task_completed` | bool | Whether the task was successfully completed |
| `data` | list | List of interaction steps (see below) |

## Step Schema (each entry in `data`)

| Field | Type | Description |
|-------|------|-------------|
| `step` | int | Step index (1-based) |
| `thought` | string | Agent's reasoning about the current state |
| `action` | string | Natural language action description |
| `plan` | object | Structured action call (`name` + `arguments`) |
| `screenshot` | string | Filename of the screenshot for this step |
| `bbox` | [[x1,y1],[x2,y2]] | Bounding box of the target UI element |
| `code` | string | Executable code representation of the action |
| `is_use` | bool | Whether this step is used for training |
| `train_test` | string | Split assignment: `"train"` or `"test"` |

---

## Desktop Case

**Task**: Open the terminal, navigate to the "Desktop/pp" directory, create a new blank "goast.out" file, and verify the creation.

| | |
|---|---|
| **Episode** | `00af4cc2-25da-46cc-bbc9-c48f1d7dd242` |
| **Device** | Desktop (1920×1080) |
| **App** | OS |
| **Steps** | 10 |
| **Status** | Completed |

<details>
<summary><b>Step-by-step action sequence</b> (click to expand)</summary>

| Step | Action | Tool Call |
|------|--------|-----------|
| 1 | Click on the terminal icon in the left sidebar | `left_click` at (18, 491) |
| 2 | Navigate to Desktop/pp directory | `type` command |
| 3 | Create goast.out file | `type` command |
| 4 | Verify file creation | `type` command |
| ... | ... | ... |
| 10 | Terminate task | `terminate` (success) |

</details>

**Screenshot (Step 0)**:

<img src="desktop/00af4cc2-25da-46cc-bbc9-c48f1d7dd242/screenshot_step0.jpg" width="600">

---

## Mobile Case

**Task**: Adjust the device's display brightness to the lowest possible setting.

| | |
|---|---|
| **Episode** | `AdjustBrightnessMinimumTask_v1` |
| **Device** | Mobile (1080×2400) |
| **App** | Settings |
| **Steps** | 10 |
| **Status** | Completed |

<details>
<summary><b>Step-by-step action sequence</b> (click to expand)</summary>

| Step | Action | Tool Call |
|------|--------|-----------|
| 1 | Tap the Settings app icon on the home screen | `click` at (540, 891) |
| 2 | Scroll down to find Display settings | `swipe` |
| 3 | Tap on Display option | `click` |
| 4 | Adjust brightness slider to minimum | `swipe` |
| ... | ... | ... |
| 10 | Terminate task | `terminate` (success) |

</details>

**Screenshot (Step 0)**:

<img src="mobile/AdjustBrightnessMinimumTask_v1/screenshot_step0.jpg" width="250">

---

## How This Data is Used

```
Raw Trajectory (task.json)
        │
        ├──→ SFT Parquet (see data/sft/)
        │       Each step → (system_prompt, user_prompt + image, ground_truth_response)
        │
        └──→ MOPD Parquet (see data/mopd/)
                Each step → (prompt, image, ground_truth, bbox) for RL training
```

The SFT format is used in **Stage 1** (training platform-specific teachers), while the MOPD format is used in **Stage 2** (multi-teacher on-policy distillation with reinforcement learning).
