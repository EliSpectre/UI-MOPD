"""
拒绝采样脚本 (Rejection Sampling)
=================================
流程:
  1. 读取 train.parquet 数据
  2. 对全部样本用 base model 推理（支持图片加载）
  3. 用 reward 函数评估，把模型回答不出来的样本保存下来（拒绝样本集）

用法:
  python rejection_sampling.py \
      --input_file data/stage1/train.parquet \
      --output_dir data/stage1/rejection_sampling \
      --model_url http://10.29.223.19:8000/v1 \
      --model_name base_model \
      --reward_threshold 0.0 \
      --image_root /path/to/images \
      --seed 42
"""

import argparse
import base64
import json
import os
import random
import re
from sqlite3 import Row
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from ui_mopd.reward.gui_agent import REWARD_FUNCS_REGISTRY, compare_text
from copy import deepcopy

# 自动将项目根目录加入 sys.path，确保能 import ui_mopd 等模块
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
from tqdm import tqdm


# ─────────────────────────── 1. 参数解析 ───────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="GUI Agent 拒绝采样")
    parser.add_argument("--input_file", type=str, default="/main/guiagent/training-out/duanchengzhen/samples_exp/samples_4.parquet",
                        help="输入 parquet 文件路径")
    # parser.add_argument("--output_dir", type=str, default="/main/guiagent/training-out/lianniu/reject_sample",
    #                     help="输出目录")
    parser.add_argument("--output_dir", type=str, default=".local/rejected_exp",
                        help="输出目录")
    parser.add_argument("--model_url", type=str, default="http://s-20260126194509-vlbjd-w8xp7.ak-cloudml.xiaomi.srv/v1",
                        help="Base model 的 OpenAI-compatible API 地址")
    parser.add_argument("--model_name", type=str, default="Qwen3-VL-30B-A3B-Instruct",
                        help="模型名称")
    parser.add_argument("--inference_sample_count", type=int, default=-1,
                        help="推理样本数，-1 表示全量推理")
    parser.add_argument("--reward_threshold", type=float, default=0.0,
                        help="reward 低于此阈值视为'回答不出来'（拒绝样本）")
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="模型生成的最大 token 数")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="模型采样温度")
    parser.add_argument("--num_workers", type=int, default=60,
                        help="并发推理的线程数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅做采样分析，不调用模型推理")
    parser.add_argument("--image_root", type=str, default="",
                        help="图片根目录，用于拼接 images 字段中的相对路径")
    return parser.parse_args()


# ─────────────────────────── 2. 工具函数 ───────────────────────────

class NumpyEncoder(json.JSONEncoder):
    """
    自定义 JSON 编码器，支持 NumPy 数据类型：
    - ndarray → list
    - integer → int
    - floating → float
    - bool_ → bool
    - 其他 NumPy 类型可继续扩展
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        # 如果遇到其他无法序列化的类型，调用基类方法抛出异常
        return super().default(obj)

def get_task_category(extra_info: dict) -> str:
    """从 extra_info 中提取任务类别"""
    task = extra_info.get("task", "")
    if task:
        return task
    # fallback: 用 foreground_app 作为类别
    return extra_info.get("foreground_app", "unknown")


def _smart_resize(height: int, width: int, min_pixels: int, max_pixels: int,
                   factor: int = 28) -> tuple[int, int]:
    """
    根据最小/最大总像素数约束，等比缩放图片尺寸，并对齐到 factor 的整数倍。
    返回 (resized_height, resized_width)。
    """
    import math
    total = height * width
    if total < min_pixels:
        scale = math.sqrt(min_pixels / total)
    elif total > max_pixels:
        scale = math.sqrt(max_pixels / total)
    else:
        scale = 1.0
    new_h = max(factor, round(height * scale / factor) * factor)
    new_w = max(factor, round(width * scale / factor) * factor)
    return new_h, new_w


def encode_image_to_base64(image_path: str,
                           min_pixels: int = 40768,
                           max_pixels: int = 548800,
                           quality: int = 50) -> str | None:
    """
    读取图片并编码为 base64 字符串。
    - PNG 自动转为 JPG 格式以减小体积
    - 根据 min_pixels / max_pixels 做等比智能缩放，而非强行裁剪到固定尺寸
    """
    from PIL import Image
    import io as _io

    # 清理路径：去掉 JSON 转义的反斜杠 \/  →  /
    image_path = image_path.replace("\\/", "/")

    if not os.path.exists(image_path):
        print(f"[ERROR] 图片不存在: {image_path}")
        return None

    try:
        with Image.open(image_path) as img:
            width, height = img.size

            # 等比智能缩放
            resized_h, resized_w = _smart_resize(height, width,
                                                  min_pixels=min_pixels,
                                                  max_pixels=max_pixels)
            if (resized_w, resized_h) != (width, height):
                img = img.resize((resized_w, resized_h))
                print(f"smart_resize {width}x{height} -> {resized_w}x{resized_h}: {image_path}")

            # 统一转为 RGB（去除 alpha 通道）并以 JPEG 格式编码
            img = img.convert("RGB")
            buf = _io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Encode image error: {e}")
        return None


def build_inference_messages(row: dict, image_root: str = "") -> list:
    """
    从数据行构建送入模型的 messages。
    prompt 是多轮对话列表，最后一个 user 消息是当前需要回答的问题。
    我们需要把最后一个 assistant 消息移除(那是 ground truth)
    只保留到最后一个 user 消息为止。
    同时处理 <image> 标签，将对应图片编码为 base64 嵌入消息中。
    """
    prompt = row.get("prompt", [])
    # 兼容 parquet 读出来的 numpy array 或 JSON 字符串
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    elif isinstance(prompt, str):
        try:
            prompt = json.loads(prompt)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(prompt, list):
        return []

    # 加载图片列表
    images_info = row.get("images", []) or []
    if isinstance(images_info, np.ndarray):
        images_info = images_info.tolist()
    elif isinstance(images_info, str):
        try:
            images_info = json.loads(images_info)
        except (json.JSONDecodeError, TypeError):
            images_info = []
    loaded_images = []
    for img_info in images_info:
        if isinstance(img_info, dict):
            img_path = img_info.get("image", "")
        else:
            img_path = str(img_info)
        # 拼接图片根目录
        # 清理路径：去掉 JSON 转义的反斜杠 \/  →  /
        img_path = img_path.replace("\\/", "/")
        if image_root and img_path:
            full_path = os.path.join(image_root, img_path.lstrip("/"))
        else:
            full_path = img_path
        loaded_images.append(full_path)

    # 构建 messages，处理 <image> 标签
    messages = []
    image_offset = 0
    for msg in prompt:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role not in ("system", "user", "assistant"):
            continue

        # 检查 content 中是否含有 <image> 标签
        segments = re.split("(<image>|<video>)", content)
        segments = [item for item in segments if item != ""]

        has_media = any(seg in ("<image>", "<video>") for seg in segments)
        if has_media:
            content_list = []
            for segment in segments:
                if segment == "<image>":
                    if image_offset < len(loaded_images):
                        img_path = loaded_images[image_offset]
                        b64 = encode_image_to_base64(img_path)
                        if b64:
                            content_list.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                            })
                        else:
                            content_list.append({"type": "text", "text": "[图片加载失败]"})
                        image_offset += 1
                    else:
                        content_list.append({"type": "text", "text": "[图片缺失]"})
                elif segment == "<video>":
                    content_list.append({"type": "text", "text": "[视频不支持]"})
                else:
                    content_list.append({"type": "text", "text": segment})
            messages.append({"role": role, "content": content_list})
        else:
            messages.append({"role": role, "content": content})

    # 如果最后一条是 assistant，去掉（那是标注答案）
    # 我们只保留到最后一个 user
    while messages and messages[-1]["role"] == "assistant":
        messages.pop()

    return messages


def call_model(client, model_name: str, messages: list,
               max_tokens: int = 8192, temperature: float = 0.7) -> str:
    """调用 OpenAI-compatible API 获取模型回复"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.8,
        )
        return response.choices[0].message.reasoning_content
    except Exception as e:
        print(f"[ERROR] 模型推理失败: {e}")
        return ""


def evaluate_response(predict_str: str, ground_truth: str, extra_info: dict,
                      data_source: str = "") -> dict:
    """
    用 reward 函数评估模型回答。
    简化版本：不依赖 tokenizer（跳过 think 长度检查），
    直接复用 gui_agent.py 中的 reward 逻辑。
    """
  

    try:
        label = json.loads(ground_truth.split("<answer>")[-1].split("</answer>")[0])
        predict = json.loads(predict_str.split("<answer>")[-1].split("</answer>")[0])

        reward = 0.0
        if label["func"] in REWARD_FUNCS_REGISTRY:
            reward = REWARD_FUNCS_REGISTRY[label["func"]](predict, label, extra_info)
        else:
            reward = -1.0

        reward = 1.0 if reward == 1.0 else -0.5
        if "\uFFFD" in predict_str:
            reward = -0.5
    except Exception:
        reward = -1.0

    # Plan B: 尝试 other_label
    if reward != 1.0 and "other_label" in extra_info and extra_info["other_label"]:
        try:
            other_label = json.loads(
                extra_info["other_label"].split("<answer>")[-1].split("</answer>")[0]
            )
            other_extra_info = deepcopy(extra_info)
            other_extra_info["coordinate"] = other_extra_info.get(
                "other_coordinate", other_extra_info["coordinate"]
            )
            if other_label["func"] in REWARD_FUNCS_REGISTRY:
                r = REWARD_FUNCS_REGISTRY[other_label["func"]](
                    predict, other_label, other_extra_info
                )
                r = 1.0 if r == 1.0 else -0.5
                if "\uFFFD" in predict_str:
                    r = -0.5
                if r == 1.0:
                    reward = r
        except Exception:
            pass

    return {
        "score": reward,
        "is_correct": reward == 1.0,
        "response": predict_str,
    }


def _parse_json_field(val):
    """兼容 parquet 读出来可能是 JSON 字符串或 numpy array 的字段"""
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return val
    return val


# ─────────────────────────── 3. 主流程 ───────────────────────────
# 用当前 base model 过滤数据，找出“难样本”
def main():
    # 读取命令行参数
    args = parse_args()
    random.seed(args.seed)

    # ── 创建输出目录 ──
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 读取数据 ──
    print(f"[1/3] 读取数据: {args.input_file}")
    # 每一行是一个 GUI Agent 训练样本
    df = pd.read_parquet(args.input_file)
    print(f"  总样本数: {len(df)}")

    # 调试：打印第一条数据的字段类型，便于排查 JSON 字符串问题
    if len(df) > 0:
        first_row = df.iloc[0]
        print(f"  [DEBUG] 字段类型: prompt={type(first_row.get('prompt', None)).__name__}, "
              f"images={type(first_row.get('images', None)).__name__}, "
              f"extra_info={type(first_row.get('extra_info', None)).__name__}")

    # ── 确定推理样本 ──
    if args.inference_sample_count > 0 and args.inference_sample_count < len(df):
        inference_indices = random.sample(df.index.tolist(), args.inference_sample_count)
        inference_df = df.loc[inference_indices]
        print(f"  随机抽取 {len(inference_df)} 条进行推理")
    else:
        inference_df = df
        print(f"  全量 {len(inference_df)} 条进行推理")

    # 统计推理集的类别分布
    inference_df_with_task = inference_df.copy()
    inference_df_with_task["_task"] = inference_df["extra_info"].apply(
        lambda x: get_task_category(_parse_json_field(x)) if x else "unknown"
    )
    print(f"  推理集类别分布 (top 10):")
    for task, count in inference_df_with_task["_task"].value_counts().head(10).items():
        print(f"    {task}: {count}")

    # 开启 dry_run 模式时，不做模型推理，只做分布分析，然后直接退出程序
    if args.dry_run:
        print("\n[DRY RUN] 跳过模型推理，仅分析数据分布。")
        inference_path = os.path.join(args.output_dir, "inference_candidates.parquet")
        inference_df.to_parquet(inference_path, index=False)
        print(f"  已保存推理候选集 → {inference_path}")
        return

    # ── 模型推理 ──
    print(f"\n[2/3] 调用 base model 推理 ({args.model_url})")
    from openai import OpenAI
    client = OpenAI(base_url=args.model_url, api_key="EMPTY")

    results = []
    failed_count = 0


    def process_row(idx, row):
        """处理单行数据：构建消息 → 调用模型 → 评估"""
        messages = build_inference_messages(row, image_root=args.image_root)
        if not messages:
            return None

        # 调用模型
        response_str = call_model(
            client, args.model_name, messages,
            max_tokens=args.max_tokens, temperature=args.temperature,
        )

        if not response_str:
            return {
                "index": idx,
                "score": -1.0,
                "is_correct": False,
                "response": "",
                "error": "empty_response",
            }

        # 评估
        extra_info = _parse_json_field(row.get("extra_info", {}))
        if not isinstance(extra_info, dict):
            extra_info = {}
        
        # ground_truth 在 reward_model 字段中，不在 extra_info 中
        reward_model = _parse_json_field(row.get("reward_model", {}))
        if not isinstance(reward_model, dict):
            reward_model = {}
        ground_truth = reward_model.get("ground_truth", "")
        data_source = row.get("data_source", "")

        eval_result = evaluate_response(response_str, ground_truth, extra_info, data_source)
        eval_result["index"] = idx
        return eval_result, row

    # 并发推理
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {}
        for idx, row in inference_df.iterrows():
            future = executor.submit(process_row, idx, row.to_dict())
            futures[future] = idx

        pbar = tqdm(total=len(futures), desc="模型推理")
        for future in as_completed(futures):
            try:
                result, Row = future.result()
                if result is not None:
                    results.append([result, Row])
                else:
                    failed_count += 1
            except Exception as e:
                print(f"[ERROR] 处理异常: {e}")
                failed_count += 1
            pbar.update(1)
        pbar.close()

    print(f"  推理完成: {len(results)} 成功, {failed_count} 失败")

    # ── 按 reward 阈值筛选拒绝样本 ──
    print(f"\n[3/3] 筛选拒绝样本 (reward < {args.reward_threshold})")

    rejected_indices = []
    correct_indices = []
    score_distribution = defaultdict(int)

    for r, row in results:
        score = r["score"]
        # 统计分数分布
        if score == 1.0:
            score_distribution["1.0 (correct)"] += 1
            correct_indices.append(r["index"])
        elif score == -0.5:
            score_distribution["-0.5 (wrong)"] += 1
        elif score == -1.0:
            score_distribution["-1.0 (parse_error)"] += 1
        else:
            score_distribution[f"{score:.1f}"] += 1

        if score < args.reward_threshold:
            rejected_indices.append(r["index"])

    print(f"  分数分布:")
    for k, v in sorted(score_distribution.items()):
        print(f"    {k}: {v}")
    print(f"  模型回答正确: {len(correct_indices)} ({len(correct_indices)/max(len(results),1)*100:.1f}%)")
    print(f"  拒绝样本(模型回答不出来): {len(rejected_indices)} ({len(rejected_indices)/max(len(results),1)*100:.1f}%)")

    # ── 保存拒绝样本 ──
    if rejected_indices:
        rejected_df = inference_df.loc[rejected_indices]
        rejected_path = os.path.join(args.output_dir, "rejected_samples.parquet")
        rejected_df.to_parquet(rejected_path, index=False)
        print(f"  已保存拒绝样本 → {rejected_path}")

    # 保存推理结果明细（用于分析）
    results_path = os.path.join(args.output_dir, "inference_results.jsonl")
    with open(results_path, "w", encoding="utf-8") as f:
        for r, row in results:
            row["response"] = r
            f.write(json.dumps(row, cls=NumpyEncoder, ensure_ascii=False) + "\n")
    print(f"  已保存推理明细 → {results_path}")

    # ── 汇总报告 ──
    print(f"\n{'='*60}")
    print(f"拒绝采样完成！输出目录: {args.output_dir}")
    print(f"{'='*60}")
    print(f"  原始数据:          {len(df)} 条")
    print(f"  模型推理样本:      {len(inference_df)} 条")
    print(f"  → 模型答对:        {len(correct_indices)} 条")
    print(f"  → 拒绝样本:        {len(rejected_indices)} 条 → rejected_samples.parquet")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
