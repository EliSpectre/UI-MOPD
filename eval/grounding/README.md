# GUI Grounding Evaluation

Evaluation scripts for GUI grounding benchmarks: **ScreenSpot-Pro**, **ScreenSpot-V2**, and **OSWorld-G**.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers
- qwen_vl_utils
- flash-attn
- tqdm, Pillow

```bash
pip install torch transformers qwen_vl_utils flash-attn tqdm Pillow
```

## Data Preparation

### ScreenSpot-Pro

```
ScreenSpot-Pro/
├── images/                     # Screenshot images (3840x2160)
├── screenspot_pro_cad.json
├── screenspot_pro_dev.json
├── screenspot_pro_creative.json
├── screenspot_pro_scientific.json
├── screenspot_pro_office.json
└── screenspot_pro_os.json
```

Each JSON item: `{"img_filename", "instruction", "bbox": [x1, y1, x2, y2], "ui_type": "text"|"icon"}`

### ScreenSpot-V2

```
ScreenSpot-v2/
├── screenspotv2_image/          # Screenshot images
├── screenspot_mobile_v2.json
├── screenspot_desktop_v2.json
└── screenspot_web_v2.json
```

Each JSON item: `{"img_filename", "instruction", "bbox": [x, y, w, h], "data_type": "text"|"icon"}`

### OSWorld-G

```
OSWorld-G/
├── images/                      # Desktop screenshots (1920x1080)
├── OSWorld-G.json
└── classification_result.json   # Category mapping for per-category breakdown
```

Each JSON item: `{"id", "image_path", "instruction", "box_coordinates": [x, y, w, h], "box_type": "bbox"|"polygon"|"refusal"}`

`classification_result.json` maps sample IDs to categories: `text_matching`, `element_recognition`, `layout_understanding`, `fine_grained_manipulation`, `refusal`.

## Usage

### Quick Start

1. Edit `run_all.sh` to set model paths and data paths.
2. Run:

```bash
bash run_all.sh
```

Logs will be saved to `logs/`.

### Individual Benchmarks

**ScreenSpot-Pro:**
```bash
torchrun --nproc_per_node=8 screenspot_pro_official.py \
    --model-path /path/to/model \
    --screenspot-imgs /path/to/ScreenSpot-Pro/images \
    --screenspot-test /path/to/ScreenSpot-Pro/ \
    --batch-size 4 --num-workers 4 --max-new-tokens 8192 \
    --enable-thinking --temperature 0.7 --top-p 0.8 --top-k 20
```

**ScreenSpot-V2:**
```bash
torchrun --nproc_per_node=8 screenspot_v2_official.py \
    --model-path /path/to/model \
    --screenspot-imgs /path/to/ScreenSpot-v2/screenspotv2_image/ \
    --screenspot-test /path/to/ScreenSpot-v2/ \
    --batch-size 4 --num-workers 4 --max-new-tokens 8192 \
    --enable-thinking --temperature 0.7 --top-p 0.8 --top-k 20
```

**OSWorld-G:**
```bash
torchrun --nproc_per_node=8 osworld_g_official.py \
    --model-path /path/to/model \
    --data-dir /path/to/OSWorld-G/ \
    --classification-path /path/to/classification_result.json \
    --batch-size 4 --num-workers 4 --max-new-tokens 8192 \
    --enable-thinking --temperature 0.7 --top-p 0.8 --top-k 20
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model-path` | (required) | Path to HuggingFace model |
| `--batch-size` | 4 | Inference batch size per GPU |
| `--num-workers` | 4 | DataLoader workers |
| `--max-new-tokens` | 1024 | Max generation tokens (use 8192 for thinking models) |
| `--enable-thinking` | False | Enable thinking mode (for Thinking models) |
| `--temperature` | 0.7 | Sampling temperature |
| `--top-p` | 0.8 | Top-p sampling |
| `--top-k` | 20 | Top-k sampling |

## Evaluation Details

- **System prompt**: Custom tool-call format with `computer_use` tool definition. Resolution is dynamically set per image. Includes a "notes" field instructing the model to click with cursor tip centered on targets.
- **Coordinate format**: Model outputs coordinates in [0, 1000] normalized space. Converted to pixel coords via `coord / 1000 * img_size`.
- **Correctness**: A prediction is correct if the predicted point falls within the ground-truth bounding box.
- **Refusal (OSWorld-G)**: For refusal samples, the model should output `"action": "terminate"` instead of coordinates.
- **IMAGE_MAX_TOKEN_NUM**: Set to 10000, corresponding to max_pixels = 10,240,000.
- **Distributed**: Uses NCCL backend with 7200s timeout. Each GPU processes a non-overlapping subset of data.

## Output Format

Results are printed to stdout at the end of each run:

```
ScreenSpot-Pro Results:
  CAD         : XX.XX%  text=XX.XX%  icon=XX.XX%  (N/M)
  Dev         : XX.XX%  ...
  ...
  Overall: XX.XX%  (N/M)

ScreenSpot-V2 Results:
  mobile      : XX.XX%  text=XX.XX%  icon=XX.XX%  (N/M)
  desktop     : XX.XX%  ...
  web         : XX.XX%  ...
  Overall: XX.XX%  (N/M)

OSWorld-G Results:
  Text Matching                 : XX.XX%  (N/M)
  Element Recognition           : XX.XX%  (N/M)
  Layout Understanding          : XX.XX%  (N/M)
  Fine-grained Manipulation     : XX.XX%  (N/M)
  Refusal                       : XX.XX%  (N/M)
  Overall                       : XX.XX%  (N/M)
```
