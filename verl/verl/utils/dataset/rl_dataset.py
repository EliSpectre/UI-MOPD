# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import os
import re
import traceback
from collections import defaultdict
from typing import Optional, Dict
import random
import json


import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from ui_mopd.dataset.thought_buffer import ThoughtBuffer

logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \\*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


def get_teacher_system_prompt() -> str:
    prompt = (
"""# Role
你是一个精通GUI交互的智能体分析专家（GUI Agent）。你的核心能力是像人类一样理解屏幕内容，并通过分析操作历史和当前界面状态，规划并执行最合理的操作步骤以完成用户指令。

# Input Context
你将获得以下信息：
1. 用户指令
2. 之前的操作交互历史
3. 当前前台app信息
4. 当前手机屏幕的视觉信息

# Action Space (API)
你必须且只能使用以下 JSON 指令（严禁编造参数）：
1. **Tap (点击)**
   - `{"func": "Tap", "position": [x, y], "times": 1, "action": "点击[元素名称]"}`
   - 说明: 点击指定坐标。times 默认为 1。用于按钮、链接、图标交互。
2. **LongPress (长按)**
   - `{"func": "LongPress", "position": [x, y], "action": "长按[元素名称]"}`
   - 说明: 长按触发更多菜单或选中。
3. **Swipe (滑动)**
   - `{"func": "Swipe", "start_position": [x1, y1], "end_position": [x2, y2], "action": "向[方向]滑动"}`
   - 说明: 模拟手指滑动。浏览列表通常从下往上滑（start_y > end_y）。
4. **Type (输入文本)**
   - `{"func": "Type", "position": [x, y], "text": "内容", "action": "在[输入框]输入内容"}`
   - 说明: 点击输入框并输入文本。确保 position 在输入框范围内。
5. **Search (搜索宏指令)**
   - `{"func": "Search", "position": [x, y], "text": "关键词", "action": "搜索[关键词]"}`
   - 说明: 复合操作。点击位置 -> 清空 -> 输入 text -> 发送回车/搜索键。仅用于明确的搜索场景。
6. **Open (打开应用)**
   - `{"func": "Open", "app": "app名称", "action": "启动[app名称]"}`
   - 说明: 通过系统指令启动 App。app 名称统一使用小写，并优先使用简称（如 "qq"、"boss直聘"、"qq音乐"、"qq浏览器"、"b站"、"qq邮箱"）。
7. **Back (返回)**
   - `{"func": "Back", "action": "返回上一页"}`
   - 说明: 尝试系统级返回。若无效，请寻找页面内的 "返回" 图标或 "X" 关闭按钮并使用 Tap。
8. **Wait (等待)**
   - `{"func": "Wait", "action": "等待页面加载"}`
   - 说明: 遇到加载转圈、页面渲染未完成时使用。
9. **Request (人机交互)**
   - `{"func": "Request", "text": "询问内容", "action": "询问用户"}`
   - 说明: 当存在歧义、多选项或需用户确认时使用。
10. **Fail (任务失败)**
    - `{"func": "Fail", "type": "错误类型", "reason": "详细原因", "action": "报告失败"}`
    - 错误类型参考: 
        * LOGIN_REQUIRED: 需要登录才能继续操作，但当前未登录或登录已过期。
        * CAPTCHA_VERIFICATION: 遇到验证码（图形验证码、滑块验证码等）需要人工验证。
        * RESULT_NOT_FOUND: 搜索或查找操作未找到符合条件的结果。
        * BLUETOOTH_CONNECTION_REQUIRED: 任务需要蓝牙连接，但设备未连接或连接失败。
        * NETWORK_ERROR: 网络连接失败、超时或无法访问服务器。
        * PAYMENT_AUTHENTICATION: 需要支付验证（如密码、指纹等）才能完成支付操作。
        * TASK_CANT_FULLFILLED: 任务本身无法完成，例如任务要求不明确、逻辑矛盾或超出能力范围。
        * REPEAT_OPERATION: 重复执行相同操作多次仍无法达到预期效果，可能陷入死循环。
        * PERMISSION_REQUEST: 需要用户授予特定权限（如位置、相机、存储等）才能继续。
        * PASSWORD_REQUIRED: 需要输入密码但无法自动获取或输入。
        * TAKOVER_EXIT: 用户主动接管操作或要求退出任务。
        * TEMPORARY_TAKEOVER: 临时需要人工介入处理，但后续可能可以继续自动化执行。
        * MANUAL_VERIFICATION_REQUIRED: 需要人工验证或确认才能继续，如身份验证、二次确认等。
11. **Complete (任务完成)**
    - `{"func": "Complete", "action": "任务完成"}`
    - 说明: 确认目标已达成，处于最终结果页面。
12. **Speak (输出文本)**
    - `{"func": "Speak", "text": "文本内容", "action": "输出文本"}`
    - 说明: 总结或评论当前页面或已记录的内容。
13. **ToolUse (任务使用)**
    - `{"func": "ToolUse", "type": "工具类型", "action": "使用[工具名称]"}`

# Operational Constraints
1. **坐标系**: 所有坐标 position 均为相对坐标 [x, y]，取值范围 [0, 1]，保留 3 位小数。左上角为 (0, 0)，右下角为 (1, 1)。**严禁使用绝对像素坐标**（如 [503, 74]、[624, 427] 等大于1的数值均为非法），所有坐标分量必须在 0.0~1.0 之间。
2. **弹窗处理**: 遇到无关弹窗（广告、升级提示、评价请求），优先寻找 "关闭/跳过/X/以后再说" 按钮进行 Tap，而不是 Fail。
3. **循环熔断**: 若连续3步操作页面没有变化，或者一套动作在循环重复执行，必须触发自我修正（尝试Back或者操作不同的位置），若修正无效则 Fail。
4. **主Tab页与子页面判定**:
   - **主Tab页**: 如果屏幕底部有导航栏（Tab Bar），则当前页面就是主Tab页，无论当前选中的是哪个Tab。例如QQ底部有"消息/频道/联系人/动态"四个Tab，这四个页面都是主Tab页；B站底部有"首页/动态/我的"等Tab，这些页面都是主Tab页。**在主Tab页之间切换时，必须使用底部导航栏Tap切换，严禁使用Back。**
   - **子页面**: 如果屏幕底部没有导航栏，或者左上角有"返回箭头/×"按钮，则当前页面是子页面（如设置页、详情页、搜索结果页、弹窗等）。
5. **导航优先级**: 当需要从当前页面导航到某个功能时：
   - **若当前处于子页面**（无底部导航栏、有返回按钮），且该页面与目标无关，应优先使用 **Back** 返回上一层。
   - **若当前处于主Tab页**（有底部导航栏），但不是目标功能所属的主Tab页，则通过底部导航栏 **Tap** 切换到正确的主Tab页。例如：在QQ的"频道"/"联系人"/"动态"页要访问搜索/设置/钱包/会员等功能时，应Tap底部"消息"Tab；在B站的"我的"页要访问动画/影视/直播/发布等功能时，应Tap底部"首页"Tab。
   - **若已处于正确的主Tab页**，则点击页面内的功能入口进入目标功能。
   - **禁止捷径**：不要直接点击左上角头像、右上角设置图标等非标准入口来跳转功能，除非任务明确要求进入个人中心或设置页面本身。
6. **底部导航栏识别**: 底部导航栏通常位于屏幕最底部（y坐标 > 0.9），包含3-5个Tab选项（如"首页"、"消息"、"发现"、"我的"等），是App内跨模块导航的唯一标准入口。
7. **任务完成判定（严格标准）**: 仅在以下条件**全部满足**时才使用 Complete：
   - 已经**实际执行了**目标操作（不是"看到了可以操作的按钮/入口"）
   - 当前页面**不需要任何进一步点击**即可视为任务完成
   - **以下情况不算完成**：看到搜索结果但未切换到目标分类（如番剧/影视/直播）；看到功能入口但未点击进入；看到开关但未点击开启；看到列表但未进入详情页
8. **滑动操作**: 当目标功能在当前可见区域内未找到，但页面可能有更多内容（如设置列表、功能菜单底部被截断），应使用 Swipe 向上/向左滑动以显示更多内容，而不是返回或点击其他入口。
9. **开关/功能项点击**: 对于带有开关按钮的设置项（如"桌面歌词"、"锁屏歌词"、"状态栏歌词"、"定时关闭"等），应点击该设置项的**文字标签区域**（通常在行的左侧或中间），而不是点击右侧的开关按钮本身。
10. **搜索策略**: 当需要在搜索框中输入关键词并执行搜索时，应优先使用 **Search** 宏指令（一步完成清空+输入+搜索），而不是分步使用 Type 输入后再 Tap 搜索按钮。

# Reasoning Framework (<think>)
在生成 JSON 之前，请在 <think> 标签中进行简洁的思维链推导，内容包括：
1. **页面层级判断**: 底部是否有导航栏？如果有，当前是主Tab页（用Tap切换Tab）；如果没有，当前是子页面（可用Back返回）。
2. **完成检查**: 目标操作是否已实际执行完毕且无需任何进一步点击？注意：仅看到目标入口/搜索结果/开关按钮不算完成，必须已完成最终操作。
3. **意图规划**: 距离目标还需几步，当前最佳操作是什么。
4. **参数锚定**: 明确目标元素的视觉特征和相对坐标 [x, y]（必须在0~1范围内）。

注意：思考内容应简洁（2-4句话），避免冗长分析。

# Output Format
输出必须严格遵循如下XML格式，不允许任何偏差：
<think>
[简洁的观察和推理]
</think>
<answer>
{JSON Command}
</answer>

**格式强制要求（违反将导致解析失败）：**
- 必须且只能使用 `<think>`、`</think>`、`<answer>`、`</answer>` 这四个标签
- `<think>` 标签只能出现一次，禁止嵌套（禁止 `<think><think>`）
- `</think>` 后必须紧跟 `<answer>`，中间不得插入其他内容
- `<answer>` 标签内只放一个完整的 JSON 对象，不含任何其他文字
- **严禁**使用 `<tool_call>`、`</tool_call>`、`<function_call>` 或任何其他非指定标签
- **严禁**将 JSON 指令直接放在 `<think>` 标签内，JSON 指令只能出现在 `<answer>` 标签内""")
    return prompt


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]
        
        self.thought_buffer = ThoughtBuffer(config.thought_buffer_size) if config.use_dynamic_history else None

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_samples = max_samples
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.image_patch_size = config.get("image_patch_size", 14)
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.add_reasoning_content = config.get("add_reasoning_content", False)
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        self.tool_config_path = config.get("tool_config_path", None)
        self.tool_schemas = None
        if self.tool_config_path:
            try:
                from verl.tools.utils.tool_registry import \
                    initialize_tools_from_config

                tool_list = initialize_tools_from_config(self.tool_config_path)
                # match ToolAgentLoop behaviour: model_dump to plain dicts
                self.tool_schemas = [
                    tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list
                ]
            except Exception as e:
                logger.warning("Failed to initialize tools from %s: %s", self.tool_config_path, e)
                self.tool_schemas = None

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count()) if self.num_workers is not None else None
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")

        self.use_opsd = config.get("use_opsd", False)
        print(f"use_opsd: {self.use_opsd}")
        self.max_teacher_prompt_length = config.get(
            "max_teacher_prompt_length",
            self.max_prompt_length + 512,
        )

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)
    
    def annotate_sample_indices(self, dataframe: datasets.Dataset) -> datasets.Dataset:
        def _attach_index(example, idx):
            extra_info = example.get("extra_info")
            if extra_info is None:
                extra_info = {}
            else:
                extra_info = dict(extra_info)
            extra_info["index"] = int(idx)
            example["extra_info"] = extra_info
            return example

        return dataframe.map(
            _attach_index,
            with_indices=True,
            desc="Annotating sample indices",
        )

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)
        self.dataframe = self.annotate_sample_indices(self.dataframe)
    
    def _add_reasoning_content(self, messages: list):
        for message in messages:
            if message["role"] == "assistant":
                content = message["content"][0]['text']
                reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n')
                message['reasoning_content'] = reasoning_content
                message["content"][0]['text'] = content.split('</think>')[1].lstrip('\\n')
        return messages
    
    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import (process_image,
                                                             process_video)

                def doc2len(doc) -> int:
                    try:
                        messages = self._build_messages(doc)
                        # pass tool schemas if available so the processor can format prompts
                        apply_kwargs = dict(**self.apply_chat_template_kwargs)
                        if self.tool_schemas is not None:
                            apply_kwargs["tools"] = self.tool_schemas

                        raw_prompt = self.processor.apply_chat_template(
                            messages, add_generation_prompt=True, tokenize=False, **apply_kwargs
                        )
                        if image_key in doc and doc[image_key]:
                            images = [
                                process_image(image, image_patch_size=self.image_patch_size) for image in doc[image_key]
                            ]
                        else:
                            images = None

                        if video_key in doc and doc[video_key]:
                            videos, video_metadata = zip(
                                *[
                                    process_video(
                                        video, image_patch_size=self.image_patch_size, return_video_metadata=True
                                    )
                                    for video in doc[video_key]
                                ],
                                strict=True,
                            )
                            videos = list(videos)
                            video_metadata = list(video_metadata)
                            videos_kwargs = {"video_metadata": video_metadata, "do_sample_frames": False}
                        else:
                            videos = None
                            videos_kwargs = {}

                        return len(
                            processor(text=[raw_prompt], images=images, videos=videos, videos_kwargs=videos_kwargs)[
                                "input_ids"
                            ][0]
                        )
                    except Exception:
                        print("Error processing one of the samples, skipping...")
                        traceback.print_exc()
                        return self.max_prompt_length + 1

            else:

                def doc2len(doc) -> int:
                    try:
                        apply_kwargs = dict(**self.apply_chat_template_kwargs)
                        if self.tool_schemas is not None:
                            apply_kwargs["tools"] = self.tool_schemas

                        return len(
                            tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True, **apply_kwargs)
                        )
                    except Exception:
                        print("Error processing one of the samples, skipping...")
                        traceback.print_exc()
                        return self.max_prompt_length + 1

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")


    def dynamic_build_history(self, messages: list, history_ids: list[str]):
        replace_history_count = 0
        for index in range(len(history_ids)):
            if history_ids[index] and history_ids[index] in self.thought_buffer.buffer:
                history_item = self.thought_buffer.get(history_ids[index])
                if history_item:
                    messages[2 * (index + 1)]["content"] = f"<think>{random.choice(history_item.response_thoughts)}</think><answer>{json.dumps({'func': history_item.func, 'action': random.choice(history_item.response_actions)}, ensure_ascii=False)}</answer>"
                    print(f"dynamic_build_history: {messages[2 * (index + 1)]['content']}")
                    replace_history_count += 1
        print(f"totol history count: {len(history_ids)}, dynamic_build_history: {replace_history_count} history replaced")
        return messages


    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.thought_buffer:
            messages = self.dynamic_build_history(messages, example.get("extra_info", {}).get("history_ids", []))


        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list
        
        # add reasoning content if add_reasoning_content is True
        if self.add_reasoning_content:
            messages = self._add_reasoning_content(messages)
        
        return messages

    def _extract_ground_truth(self, row_dict):
        """从 reward_model 字段中提取 ground_truth。"""
        rm = row_dict.get("reward_model")
        if rm is None:
            return ""
        if isinstance(rm, dict):
            gt = rm.get("ground_truth", "")
            if isinstance(gt, list):
                gt = "\n".join(str(g) for g in gt) if gt else ""
            return gt or ""
        return ""

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import (process_image,
                                                         process_video)

            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            multi_modal_data = {}

            images = None
            row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
                images = [process_image(image, image_patch_size=self.image_patch_size) for image in row_dict_images]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            videos_kwargs = {}
            row_dict_videos = row_dict.pop(self.video_key, None)
            if row_dict_videos:
                videos, video_metadata = zip(
                    *[
                        process_video(video, image_patch_size=self.image_patch_size, return_video_metadata=True)
                        for video in row_dict_videos
                    ],
                    strict=True,
                )
                videos = list(videos)
                video_metadata = list(video_metadata)
                videos_kwargs = {"video_metadata": video_metadata, "do_sample_frames": False}

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [
                    (video.numpy(), metadata) for video, metadata in zip(videos, video_metadata, strict=True)
                ]

            model_inputs = self.processor(
                text=[raw_prompt], images=images, videos=videos, videos_kwargs=videos_kwargs, return_tensors="pt"
            )

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            if self.apply_chat_template_kwargs.get("chat_template") is None:
                assert hasattr(self.tokenizer, "chat_template"), (
                    "chat_template should be provided in apply_chat_template_kwargs or tokenizer config, "
                    "models like GLM can copy chat_template.jinja from instruct models"
                )
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        elif self.processor is not None and "Glm4vImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.glm4v import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=model_inputs.get("image_grid_thw"),
                video_grid_thw=model_inputs.get("video_grid_thw"),
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        if self.use_opsd:
            gt = self._extract_ground_truth(row_dict)
            assert gt != "", "ground_truth is empty"
            teacher_messages = copy.deepcopy(messages)
            for message in teacher_messages:
                if message["role"] == "system":
                    message["content"] = get_teacher_system_prompt()
                    break
                
            teacher_messages.append({
                "role": "user",
                "content": [
                        {
                            "type": "text", "text": "如下是一个参考答案，请重点学习其中的操作策略，特别注意：1)何时使用Back返回 vs 底部Tab切换；2)app名称使用小写简称；3)坐标为0~1的相对坐标；4)是否已到达目标页面应直接Complete："
                        },
                        {
                            "type": "text", "text": gt
                        },
                        {
                            "type": "text", "text": "请严格遵循参考答案的操作策略和目标选择，并按照XML格式输出（<think>简洁推理</think><answer>{JSON}</answer>）。严禁使用<tool_call>标签，<think>不可嵌套，坐标必须为0~1相对值。"
                        }
                    ]
                }
            )

            if self.processor is not None:
                teacher_raw = self.processor.apply_chat_template(
                    teacher_messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                )
                t_mi = self.processor(
                    text=[teacher_raw], images=images, videos=videos,
                    videos_kwargs=videos_kwargs, return_tensors="pt",
                )
                t_ids = t_mi.pop("input_ids")
                t_mask = t_mi.pop("attention_mask")
                if "second_per_grid_ts" in t_mi:
                    t_mi.pop("second_per_grid_ts")
            else:
                teacher_raw = self.tokenizer.apply_chat_template(
                    teacher_messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                )
                t_mi = self.tokenizer(teacher_raw, return_tensors="pt", add_special_tokens=False)
                t_ids = t_mi.pop("input_ids")
                t_mask = t_mi.pop("attention_mask")

            t_ids, t_mask = verl_F.postprocess_data(
                input_ids=t_ids, attention_mask=t_mask,
                max_length=self.max_teacher_prompt_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True, truncation=self.truncation,
            )

            if (self.processor is not None
                    and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__):
                if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                    from verl.models.transformers.qwen3_vl import \
                        get_rope_index
                else:
                    from verl.models.transformers.qwen2_vl import \
                        get_rope_index
                t_vpos = get_rope_index(
                    self.processor, input_ids=t_ids[0],
                    image_grid_thw=t_mi.get("image_grid_thw"),
                    video_grid_thw=t_mi.get("video_grid_thw"),
                    second_per_grid_ts=t_mi.get("second_per_grid_ts"),
                    attention_mask=t_mask[0],
                )
                vm = t_mask[0].bool()
                t_tpos = torch.ones((1, len(t_ids[0])), dtype=torch.long)
                t_tpos[0, vm] = torch.arange(vm.sum().item())
                t_position_ids = [torch.cat((t_tpos, t_vpos), dim=0)]
            elif (self.processor is not None
                    and "Glm4vImageProcessor" in self.processor.image_processor.__class__.__name__):
                from verl.models.transformers.glm4v import get_rope_index
                t_vpos = get_rope_index(
                    self.processor, input_ids=t_ids[0],
                    image_grid_thw=t_mi.get("image_grid_thw"),
                    video_grid_thw=t_mi.get("video_grid_thw"),
                    attention_mask=t_mask[0],
                )
                vm = t_mask[0].bool()
                t_tpos = torch.ones((1, len(t_ids[0])), dtype=torch.long)
                t_tpos[0, vm] = torch.arange(vm.sum().item())
                t_position_ids = [torch.cat((t_tpos, t_vpos), dim=0)]
            else:
                t_position_ids = compute_position_id_with_mask(t_mask)

            row_dict["teacher_input_ids"] = t_ids[0]
            row_dict["teacher_attention_mask"] = t_mask[0]
            row_dict["teacher_position_ids"] = t_position_ids[0]


        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
