"""
OSWorld-G evaluation script.
Settings:
  - IMAGE_MAX_TOKEN_NUM=10000 (max_pixels = 10000 * 32*32 = 10,240,000)
  - Custom system prompt with tool-call format
  - Per-category breakdown (Text Matching, Element Recognition, Layout Understanding,
    Fine-grained Manipulation, Refusal)
  - Supports multi-GPU distributed inference via torchrun
"""
import re
import os
import json
import math
import torch
import logging
import argparse
import datetime
from tqdm import tqdm
from PIL import Image
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from transformers import AutoProcessor, AutoModelForImageTextToText
import qwen_vl_utils.vision_process as vision_process
from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.INFO)

vision_process.IMAGE_MAX_TOKEN_NUM = 10000
MAX_PIXELS = 10000 * 32 * 32  # 10,240,000


SYSTEM_PROMPT_TEMPLATE = """You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools> . . . </tools> XML tags:
<tools> {{ "name":"computer_use", "description": "Use a mouse to interact with a computer.
The screen's resolution is {img_width}x{img_height}." "notes": "Click
with the cursor tip centered on targets; avoid edges unless asked. Do not use
other tools (type, key, scroll, left_click_drag). Only left_click and mouse_move
are allowed. If you can't find the element, terminate and report failure.",
"parameters":{{ "type":"object", "required":["action"], "properties":{{ "action":{{
"type":"string", "enum":["mouse_move","left_click"], "description":"The
action to perform." }}, "coordinate":{{ "type":"array", "description":"(x,
y): pixels from left/top. Required for action=mouse_move and action=left_click." }} }} }}
}}
</tools>
For each function call, return a JSON object with function name and arguments within <tool_call>
. . . </tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
Additionally, if you think the task is infeasible (e.g., the task is not related to the image), return:
<tool_call>
{{"name": "computer_use", "arguments": {{"action": "terminate", "status": "failure"}}}}
</tool_call>"""


def make_system_prompt(img_width, img_height):
    return SYSTEM_PROMPT_TEMPLATE.format(img_width=img_width, img_height=img_height)


class NoPaddingDistributedSampler(Sampler):
    def __init__(self, dataset, shuffle=False, seed=0):
        self.dataset = dataset
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.total_size = len(dataset)
        self.per_rank_size = math.ceil(self.total_size / self.world_size)
        self.rank_size = min(self.per_rank_size, self.total_size - self.rank * self.per_rank_size)
        self.start_idx = self.rank * self.per_rank_size
        self.end_idx = self.start_idx + self.rank_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.total_size, generator=g).tolist()
        else:
            indices = list(range(self.total_size))
        indices = indices[self.start_idx:self.end_idx]
        return iter(indices)

    def __len__(self):
        return self.rank_size


CATEGORIES = ["text_matching", "element_recognition", "layout_understanding",
              "fine_grained_manipulation", "refusal"]

CATEGORY_DISPLAY = {
    "text_matching": "Text Matching",
    "element_recognition": "Element Recognition",
    "layout_understanding": "Layout Understanding",
    "fine_grained_manipulation": "Fine-grained Manipulation",
    "refusal": "Refusal",
}


def load_id_to_category(classification_path):
    with open(classification_path, 'r') as f:
        cls_data = json.load(f)
    id_to_cat = {}
    for cat, items in cls_data['classified'].items():
        for item in items:
            id_to_cat[item['id']] = cat
    return id_to_cat


class OSWorldGDataset(Dataset):
    def __init__(self, data_items, processor, imgs_dir, id_to_cat, enable_thinking=False):
        self.data_items = data_items
        self.processor = processor
        self.imgs_dir = imgs_dir
        self.id_to_cat = id_to_cat
        self.enable_thinking = enable_thinking

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        item = self.data_items[idx]
        filename = item["image_path"]
        img_path = os.path.join(self.imgs_dir, filename)

        try:
            image = Image.open(img_path).convert("RGB")
            image_w, image_h = image.size
            instruction = item["instruction"]

            system_prompt = make_system_prompt(image_w, image_h)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ]}
            ]

            # box_coordinates: [x, y, w, h] -> [x1, y1, x2, y2]
            bc = item["box_coordinates"]
            gt_bbox = [bc[0], bc[1], bc[0] + bc[2], bc[1] + bc[3]]

            sample_id = item.get("id", "")
            metadata = {
                "gt_bbox": gt_bbox,
                "img_width": image_w,
                "img_height": image_h,
                "item_id": idx,
                "sample_id": sample_id,
                "img_filename": filename,
                "instruction": instruction,
                "category": self.id_to_cat.get(sample_id, "unknown"),
                "box_type": item.get("box_type", "bbox"),
            }

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.enable_thinking
            )
            image_inputs, video_inputs = process_vision_info(messages)

            return {
                "text": text,
                "images": image_inputs[0] if image_inputs else None,
                "videos": video_inputs[0] if video_inputs else None,
                "metadata": metadata
            }
        except Exception as e:
            logging.error(f"Error processing {img_path}: {e}")
            return {
                "text": "",
                "images": None,
                "videos": None,
                "metadata": {"error": True, "item_id": idx}
            }


def collate_fn(batch):
    valid_batch = [item for item in batch if item["images"] is not None]
    if not valid_batch:
        return None
    return {
        "texts": [item["text"] for item in valid_batch],
        "images": [item["images"] for item in valid_batch],
        "videos": [item["videos"] for item in valid_batch if item["videos"] is not None] or None,
        "metadata": [item["metadata"] for item in valid_batch]
    }


def parse_tool_call(output_text):
    match = re.search(r'<tool_call>(.*?)</tool_call>', output_text, re.DOTALL)
    if match:
        try:
            tc = json.loads(match.group(1).strip())
            args = tc.get("arguments", {})
            if "coordinate" in args:
                return args["coordinate"]
        except:
            pass
    return [0, 0]


def _check_refusal(output_text):
    """Check if model correctly refuses (outputs terminate action)."""
    match = re.search(r'<tool_call>(.*?)</tool_call>', output_text, re.DOTALL)
    if match:
        try:
            tc = json.loads(match.group(1).strip())
            args = tc.get("arguments", {})
            if args.get("action") == "terminate":
                return True
        except:
            pass
    return False


def init_distributed():
    if not dist.is_available():
        return False, 0, 0
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size <= 1:
        return False, rank, world_size
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(seconds=7200))
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return True, rank, world_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to OSWorld-G data directory (contains OSWorld-G.json and images/)")
    parser.add_argument("--classification-path", type=str, required=True,
                        help="Path to classification_result.json for per-category breakdown")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    is_distributed, rank, world_size = init_distributed()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print(f"Distributed: {is_distributed}, world_size: {world_size}")
        print(f"Device: {device}")
        print(f"max_new_tokens: {args.max_new_tokens}")
        print(f"enable_thinking: {args.enable_thinking}")
        print(f"temperature: {args.temperature}, top_p: {args.top_p}, top_k: {args.top_k}")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, padding_side='left')

    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "size"):
        if isinstance(processor.image_processor.size, dict):
            processor.image_processor.size["longest_edge"] = MAX_PIXELS
            if rank == 0:
                print(f"Set longest_edge to {MAX_PIXELS}")

    model.eval()

    # Load data and classification
    data_path = os.path.join(args.data_dir, "OSWorld-G.json")
    imgs_dir = os.path.join(args.data_dir, "images")
    with open(data_path, 'r') as f:
        all_data = json.load(f)
    id_to_cat = load_id_to_category(args.classification_path)
    if rank == 0:
        print(f"Loaded OSWorld-G: {len(all_data)} samples")

    dataset = OSWorldGDataset(all_data, processor, imgs_dir, id_to_cat, enable_thinking=args.enable_thinking)
    sampler = NoPaddingDistributedSampler(dataset, shuffle=False) if is_distributed else None
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, sampler=sampler,
        pin_memory=True, drop_last=False
    )

    cat_stats = {cat: {"correct": 0, "total": 0} for cat in CATEGORIES + ["unknown"]}

    pbar = tqdm(dataloader, desc="Processing OSWorld-G") if rank == 0 else dataloader

    for batch_data in pbar:
        if batch_data is None:
            continue

        try:
            inputs = processor(
                text=batch_data["texts"],
                images=batch_data["images"],
                videos=batch_data["videos"],
                padding=True,
                return_tensors="pt"
            )
            inputs = inputs.to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            for idx, output_text in enumerate(output_texts):
                metadata = batch_data["metadata"][idx]
                gt_bbox = metadata["gt_bbox"]
                category = metadata["category"]

                # For refusal category, check if model outputs terminate action
                if category == "refusal":
                    is_correct = _check_refusal(output_text)
                else:
                    coord = parse_tool_call(output_text)
                    x = round(coord[0] / 1000 * metadata["img_width"])
                    y = round(coord[1] / 1000 * metadata["img_height"])
                    is_correct = (gt_bbox[0] <= x <= gt_bbox[2]) and (gt_bbox[1] <= y <= gt_bbox[3])

                cat_stats[category]["total"] += 1
                if is_correct:
                    cat_stats[category]["correct"] += 1

        except Exception as e:
            if rank == 0:
                logging.error(f"Batch error: {e}")
            continue

    # Gather results
    if is_distributed:
        all_cat_stats = [None] * world_size
        dist.all_gather_object(all_cat_stats, cat_stats)
        if rank == 0:
            merged = {cat: {"correct": 0, "total": 0} for cat in CATEGORIES + ["unknown"]}
            for proc_stats in all_cat_stats:
                for cat in proc_stats:
                    merged[cat]["correct"] += proc_stats[cat]["correct"]
                    merged[cat]["total"] += proc_stats[cat]["total"]
            cat_stats = merged

    if rank == 0:
        print("=" * 100)
        print("OSWorld-G Results:")
        print(f"{'-'*100}")
        total_correct = sum(cat_stats[c]["correct"] for c in CATEGORIES)
        total_count = sum(cat_stats[c]["total"] for c in CATEGORIES)
        for cat in CATEGORIES:
            s = cat_stats[cat]
            acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
            print(f"  {CATEGORY_DISPLAY.get(cat, cat):30s}: {acc:6.2f}%  ({s['correct']}/{s['total']})")
        print(f"{'-'*100}")
        overall_acc = total_correct / total_count * 100 if total_count > 0 else 0
        print(f"  {'Overall':30s}: {overall_acc:6.2f}%  ({total_correct}/{total_count})")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
