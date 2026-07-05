# $\color{#FF6700}{\textsf{Evaluation}}$

> Five benchmarks. Two platforms. From interactive task completion to fine-grained GUI grounding.

We evaluate UI-MOPD across a comprehensive suite of GUI agent benchmarks covering both **interactive execution** (live environments) and **static understanding** (screenshot-level prediction).

---

## Benchmark Overview

| Benchmark | Platform | Type | Metric | Eval Approach |
|-----------|----------|------|--------|---------------|
| **OSWorld** | Desktop | Interactive | Task Success Rate | Official repo |
| **MobileWorld** | Mobile | Interactive | Task Success Rate | Official repo |
| **ScreenSpot-Pro** | Desktop | Grounding | Point-in-Box Accuracy | Our code (`grounding/`) |
| **ScreenSpot-V2** | Mobile+Desktop+Web | Grounding | Point-in-Box Accuracy | Our code (`grounding/`) |
| **OSWorld-G** | Desktop | Grounding | Point-in-Box + Refusal | Our code (`grounding/`) |
| **AndroidControl** | Mobile | Static | Action Type + Grounding | Our code (`androidcontrol/`) |

---

## Interactive Benchmarks (Official Repos)

For **OSWorld** and **MobileWorld**, evaluation requires live VM/emulator environments. We use the official evaluation harnesses directly — no custom code needed.

### OSWorld

> Desktop task execution in a live Linux VM. The agent interacts with real applications (file manager, terminal, browser, etc.) to complete multi-step tasks.

:point_right: **Use the official OSWorld evaluation framework:**

```
https://github.com/xlang-ai/OSWorld
```

Follow their setup guide to launch the VM, deploy your model as an agent, and run the evaluation suite. Our model interfaces are compatible with their agent API.

### MobileWorld

> Mobile task execution in a live Android emulator. The agent navigates settings, installs apps, adjusts configurations, and more.

:point_right: **Use the official MobileWorld evaluation framework:**

```
https://github.com/aspect-ux/MobileWorld
```

Same approach — deploy UI-MOPD as the agent backend and run their evaluation pipeline. The `mobile_use` action space in our model matches their expected format.

---

## Grounding Benchmarks (Our Code)

For **ScreenSpot-Pro**, **ScreenSpot-V2**, and **OSWorld-G**, we provide complete evaluation scripts with multi-GPU support.

```
eval/grounding/
├── screenspot_pro_official.py     ScreenSpot-Pro evaluation
├── screenspot_v2_official.py      ScreenSpot-V2 evaluation
├── osworld_g_official.py          OSWorld-G evaluation
├── run_all.sh                     One-click run all three benchmarks
└── logs/                          Evaluation logs (our results included)
    ├── UI-MOPD_screenspot_pro.log
    ├── UI-MOPD_screenspot_v2.log
    ├── UI-MOPD_osworld_g.log
    ├── base_thinking_*.log        Qwen3-VL-8B-Thinking baseline
    └── model_merge_*.log          TIES-Merging baseline
```

### Quick Start

```bash
# Edit paths in run_all.sh, then:
bash grounding/run_all.sh
```

Or run individually:

```bash
# ScreenSpot-Pro (8 GPUs)
torchrun --nproc_per_node=8 grounding/screenspot_pro_official.py \
    --model-path /path/to/UI-MOPD \
    --screenspot-imgs /path/to/ScreenSpot-Pro/images \
    --screenspot-test /path/to/ScreenSpot-Pro/ \
    --batch-size 4 --max-new-tokens 8192 \
    --enable-thinking --temperature 0.7

# ScreenSpot-V2 (8 GPUs)
torchrun --nproc_per_node=8 grounding/screenspot_v2_official.py \
    --model-path /path/to/UI-MOPD \
    --screenspot-imgs /path/to/ScreenSpot-v2/screenspotv2_image/ \
    --screenspot-test /path/to/ScreenSpot-v2/ \
    --batch-size 4 --max-new-tokens 8192 \
    --enable-thinking --temperature 0.7

# OSWorld-G (8 GPUs)
torchrun --nproc_per_node=8 grounding/osworld_g_official.py \
    --model-path /path/to/UI-MOPD \
    --data-dir /path/to/OSWorld-G/ \
    --classification-path /path/to/classification_result.json \
    --batch-size 4 --max-new-tokens 8192 \
    --enable-thinking --temperature 0.7
```

### How Grounding Eval Works

1. Model receives a screenshot + natural language instruction (e.g., "Click the Save button")
2. Model outputs a click coordinate in normalized [0, 1000] space
3. Coordinate is converted to pixel space: `pixel = coord / 1000 × image_size`
4. **Correct** if the predicted point falls inside the ground-truth bounding box
5. **Refusal** (OSWorld-G only): model should output `action=terminate` for unanswerable queries

---

## AndroidControl (Our Code)

> Static mobile GUI understanding: predict action type and grounding from a single screenshot.

```
eval/androidcontrol/
└── evaluate_androidcontrol.py     Full evaluation script
```

### Quick Start

```bash
python androidcontrol/evaluate_androidcontrol.py \
    --steps ./androidcontrol/steps.jsonl \
    --images ./androidcontrol/images \
    --api-base http://localhost:8000/v1 \
    --model UI-MOPD \
    --output results.jsonl
```

### Metrics

| Metric | Description |
|--------|-------------|
| **Action Type Accuracy** | Does the predicted action type match ground truth? |
| **Grounding (target)** | Is the predicted click inside the target element bbox? |
| **Grounding (ancestor)** | Is the predicted click inside a text-bearing ancestor bbox? |
| **Overall Accuracy** | Action type correct AND grounding correct |

The script calls an OpenAI-compatible VLM API (e.g., served via SGLang or vLLM), parses the model's `<tool_call>` output, and computes all metrics with concurrent requests for speed.

---

## Our Results

| Benchmark | Base (8B) | Model Merge | **UI-MOPD** |
|-----------|:---------:|:-----------:|:-----------:|
| OSWorld | 24.21% | — | **28.95%** |
| MobileWorld | 28.00% | — | **36.00%** |
| ScreenSpot-Pro | 43.71% | 37.13% | 43.14% |
| ScreenSpot-V2 | 91.27% | 88.60% | 90.88% |
| OSWorld-G | 52.13% | 47.16% | **52.84%** |
| AndroidControl | 78.73% | 74.01% | **80.05%** |

UI-MOPD achieves the best interactive task completion rates while preserving (or improving) static grounding performance — demonstrating that multi-teacher on-policy distillation does not sacrifice GUI understanding.

---

## Dependencies

```bash
# Grounding benchmarks
pip install torch transformers qwen_vl_utils flash-attn tqdm Pillow

# AndroidControl
pip install requests
```

Full model available at: [https://huggingface.co/UI-MOPD](https://huggingface.co/UI-MOPD)
