"""
ScreenSpot-Pro evaluation script.
Settings:
  - IMAGE_MAX_TOKEN_NUM=10000 (max_pixels = 10000 * 32*32 = 10,240,000)
  - Custom system prompt with tool-call format
  - Supports multi-GPU distributed inference via torchrun
"""
import re
import os
import json
import math
import torch
import logging
import argparse
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
    """Create the system prompt with screen resolution."""
    return SYSTEM_PROMPT_TEMPLATE.format(img_width=img_width, img_height=img_height)


class ScreenSpotProDataset(Dataset):
    def __init__(self, data_items, processor, imgs_dir, enable_thinking=False):
        self.data_items = data_items
        self.processor = processor
        self.imgs_dir = imgs_dir
        self.enable_thinking = enable_thinking

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        item = self.data_items[idx]
        filename = item["img_filename"]
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

            gt_bbox = item["bbox"]  # [x1, y1, x2, y2] format in ScreenSpot-Pro

            metadata = {
                "gt_bbox": gt_bbox,
                "img_width": image_w,
                "img_height": image_h,
                "group": item["group"],
                "ui_type": item["ui_type"],
                "item_id": idx,
                "img_filename": filename,
                "instruction": instruction,
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
    """Parse coordinate from <tool_call> output. Returns (x, y) or (0, 0) on failure."""
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


def init_distributed():
    if not dist.is_available():
        return False, 0, 0
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size <= 1:
        return False, rank, world_size
    import datetime
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size,
                            timeout=datetime.timedelta(seconds=7200))
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return True, rank, world_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument("--screenspot-imgs", type=str, required=True,
                        help="Path to ScreenSpot-Pro images directory")
    parser.add_argument("--screenspot-test", type=str, required=True,
                        help="Path to ScreenSpot-Pro test data directory")
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

    # Load model
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

    # Load data
    data_path = os.path.join(args.screenspot_test, "annotations.json")
    with open(data_path, 'r') as f:
        screenspot_data = json.load(f)
    if rank == 0:
        print(f"Loaded {len(screenspot_data)} samples")

    dataset = ScreenSpotProDataset(screenspot_data, processor, args.screenspot_imgs,
                                   enable_thinking=args.enable_thinking)
    sampler = NoPaddingDistributedSampler(dataset, shuffle=False) if is_distributed else None
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=(sampler is None),
        num_workers=args.num_workers, collate_fn=collate_fn, sampler=sampler,
        pin_memory=True, drop_last=False
    )

    # Stats
    outcome = {
        g: {"text": {"total": 0, "correct": 0}, "icon": {"total": 0, "correct": 0}}
        for g in ["CAD", "Dev", "Creative", "Scientific", "Office", "OS"]
    }

    pbar = tqdm(dataloader, desc="Processing") if rank == 0 else dataloader

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
                group = metadata["group"]
                ui_type = metadata["ui_type"]

                # Parse coordinate
                coord = parse_tool_call(output_text)
                # Qwen3-VL tool calling outputs coordinates in 0-1000 normalized space
                x = round(coord[0] / 1000 * metadata["img_width"])
                y = round(coord[1] / 1000 * metadata["img_height"])

                # Check if click is in GT bbox [x1, y1, x2, y2]
                is_correct = (gt_bbox[0] <= x <= gt_bbox[2]) and (gt_bbox[1] <= y <= gt_bbox[3])

                outcome[group][ui_type]["total"] += 1
                if is_correct:
                    outcome[group][ui_type]["correct"] += 1

        except Exception as e:
            if rank == 0:
                logging.error(f"Batch error: {e}")
            continue

    # Gather results
    if is_distributed:
        all_outcomes = [None] * world_size
        dist.all_gather_object(all_outcomes, outcome)
        if rank == 0:
            final_outcome = {
                g: {"text": {"total": 0, "correct": 0}, "icon": {"total": 0, "correct": 0}}
                for g in outcome
            }
            for proc_outcome in all_outcomes:
                for group in proc_outcome:
                    for ui_type in ["text", "icon"]:
                        final_outcome[group][ui_type]["total"] += proc_outcome[group][ui_type]["total"]
                        final_outcome[group][ui_type]["correct"] += proc_outcome[group][ui_type]["correct"]
            outcome = final_outcome
    else:
        pass

    if rank == 0:
        print("=" * 100)
        print("ScreenSpot-Pro Results:")
        overall_correct = 0
        overall_total = 0
        for group in ["CAD", "Dev", "Creative", "Scientific", "Office", "OS"]:
            res = outcome[group]
            total = res["text"]["total"] + res["icon"]["total"]
            correct = res["text"]["correct"] + res["icon"]["correct"]
            overall_correct += correct
            overall_total += total
            acc = correct / total if total > 0 else 0
            text_acc = res["text"]["correct"] / res["text"]["total"] if res["text"]["total"] > 0 else 0
            icon_acc = res["icon"]["correct"] / res["icon"]["total"] if res["icon"]["total"] > 0 else 0
            print(f"  {group:12s}: {acc*100:.2f}%  text={text_acc*100:.2f}%  icon={icon_acc*100:.2f}%  ({correct}/{total})")

        final_acc = overall_correct / overall_total if overall_total > 0 else 0
        print(f"{'-'*100}")
        print(f"  Overall: {final_acc*100:.2f}%  ({overall_correct}/{overall_total})")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
