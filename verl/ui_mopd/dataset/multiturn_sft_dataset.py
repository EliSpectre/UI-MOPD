# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multi-turn SFT dataset that supports training on conversation data with multiple turns
"""

import copy
import logging
import re
from typing import Any, Optional

import datasets
import torch
from omegaconf import DictConfig, ListConfig
from PIL import Image
from torch.utils.data import Dataset
from transformers import ProcessorMixin

from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.fs import copy_local_path_from_hdfs


class MultiTurnSFTDataset(Dataset):
    """
    Dataset for multi-turn conversations where each assistant response should be trained
    """

    def __init__(
        self,
        data_files: str | list[str],
        config: DictConfig = None,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]
        self.data_files = copy.deepcopy(data_files)
        
        # Set defaults and extract parameters from config if provided
        config = config or {}
        self.config = config
        self.pad_mode = config.get("pad_mode", "right")
        assert self.pad_mode in ["right", "no_padding"], (
            f"Expect pad_mode to be 'right' or 'no_padding'. Got {self.pad_mode}"
        )
        self.prompt_key = config.get("prompt_key", "prompt")
        self.response_key = config.get("response_key", "reward_model")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_length = config.get("max_length", 4096)
        self.truncation = config.get("truncation", "error")
        self.add_reasoning_content = config.get("add_reasoning_content", False)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        assert self.truncation in ["error", "left", "right"]
        self.filter_overlong = config.get("filter_overlong", True)
        self.num_workers = config.get("num_workers", 128)
        
        self.processor = processor
        if self.processor is not None and hasattr(self.processor, "image_processor"):
            min_pixels = config.get("min_pixels", None)
            max_pixels = config.get("max_pixels", None)
            if min_pixels is not None:
                self.processor.image_processor.min_pixels = min_pixels
                print(f"Updated processor min_pixels to {min_pixels}")
            if max_pixels is not None:
                self.processor.image_processor.max_pixels = max_pixels
                print(f"Updated processor max_pixels to {max_pixels}")

        self.image_patch_size = config.get(
            "image_patch_size",
            processor.image_processor.patch_size if processor and hasattr(processor, "image_processor") else 14
        )

        self._download()
        self._read_files_and_process()

    def _download(self):
        for i, data_file in enumerate(self.data_files):
            self.data_files[i] = copy_local_path_from_hdfs(data_file, verbose=True)

    def _read_files_and_process(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframes: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframes)}")
        
        #self.dataframes = self._maybe_filter_out_long(self.dataframes)

    def _add_reasoning_content(self, messages: list):
        for message in messages:
            if message["role"] == "assistant":
                content = message["content"][0]['text']
                reasoning_content = reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
                message['reasoning_content'] = reasoning_content
                message["content"][0]['text'] = content.split('</think>')[1].lstrip('\n')
        return messages

    def _maybe_filter_out_long(self, dataframes: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong:
            processor = self.processor
            image_key = self.image_key
            video_key = self.video_key

            from verl.utils.dataset.vision_utils import (process_image,
                                                         process_video)

            def doc2len(doc) -> int:
                messages = self._build_messages(doc)
                messages_text = self.processor.apply_chat_template(
                    messages, add_generation_prompt=False, tokenize=False, **self.apply_chat_template_kwargs
                )
                images = (
                    [process_image(image, image_patch_size=self.image_patch_size) for image in doc[image_key]]
                    if image_key in doc and doc[image_key]
                    else None
                )
                videos = (
                    [process_video(video) for video in doc[video_key]]
                    if video_key in doc and doc[video_key]
                    else None
                )

                messages_len = len(processor(text=[messages_text], images=images, videos=videos)["input_ids"][0])
                return messages_len

            dataframes = dataframes.filter(
                lambda doc: doc2len(doc) <= self.max_length,
                num_proc=self.num_workers,
                desc=f"Filtering sample longer than {self.max_length} tokens",
            )

            print(f"filter dataset len: {len(dataframes)}")
        return dataframes
    
    def _build_messages(self, example: dict):
        messages: list = copy.deepcopy(example[self.prompt_key])
        messages.append(
            {
                "content": example[self.response_key]["ground_truth"],
                "role": "assistant"
            }
        )

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
    
    def __len__(self):
        return len(self.dataframes)

    def _process_message_tokens(
        self,
        messages: list[dict[str, Any]],
        images: list[Image.Image],
        videos: list[torch.Tensor],
        start_idx: int,
        end_idx: int,
        is_assistant: bool = False,
        data_source: str = None,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Process tokens for a single message or a group of messages.

        Args:
            messages: List of message dictionaries
            start_idx: Start index in messages list
            end_idx: End index in messages list
            is_assistant: Whether this is an assistant message

        Returns:
            Tuple of (tokens, loss_mask, attention_mask)
        """
        def count_content_type(messages: list[dict[str, Any]], content_type: str) -> int:
            count = 0
            for message in messages:
                content = message.get("content", [])
                if isinstance(content, list):
                    count += sum(1 for item in content if isinstance(item, dict) and item.get("type") == content_type)
            return count
        
        images_sidx = count_content_type(messages[:start_idx], "image")
        images_eidx = count_content_type(messages[:end_idx], "image")
        videos_sidx = count_content_type(messages[:start_idx], "video")
        videos_eidx = count_content_type(messages[:end_idx], "video")
        images = images[images_sidx:images_eidx] if images else None
        videos = videos[videos_sidx:videos_eidx] if videos else None
        
        if start_idx > 0:
            prev_applied_text = self.processor.apply_chat_template(
                messages[:start_idx],
                add_generation_prompt=False,
                tokenize=False,
                **self.apply_chat_template_kwargs
            )
            if is_assistant:
                prev_applied_text_w_generation_prompt = self.processor.apply_chat_template(
                    messages[:start_idx],
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs
                )
        else:
            prev_applied_text = ""

        cur_applied_text = self.processor.apply_chat_template(
            messages[:end_idx],
            tokenize=False,
            add_generation_prompt=False,
            **self.apply_chat_template_kwargs,
        )
        # Get tokens for the current message only
        if is_assistant:
            generation_prompt_text = prev_applied_text_w_generation_prompt[len(prev_applied_text) :]
            generation_prompt_tokens = self.processor(
                text=[generation_prompt_text],
            )["input_ids"][0]
            _message_tokens = self.processor(
                text=[cur_applied_text[len(prev_applied_text_w_generation_prompt) :]],
                images=images if images else None,
                videos=videos if videos else None,
            )["input_ids"][0]
            message_tokens = generation_prompt_tokens + _message_tokens
            
            # set message_tokens learnable if previous message is user and has image
            # Only for specific data_source (guiagent, badcase) datasets: set message_tokens learnable if previous user message contains images; use default learnable setting for others
            is_restricted_source = "guiagent" in data_source
            has_image_context = (messages[start_idx - 1]["role"] == "user"
                and count_content_type(messages[start_idx-1:start_idx], "image") > 0)

            if not is_restricted_source or has_image_context:
                loss_mask = [0] * (len(generation_prompt_tokens)) + [1] * (
                    len(message_tokens) - len(generation_prompt_tokens)
                )
            else:
                loss_mask = [0] * len(message_tokens)
        else:
            message_tokens = self.processor(
                text=[cur_applied_text[len(prev_applied_text) :]],
                images=images if images else None,
                videos=videos if videos else None,
            )["input_ids"][0]
            loss_mask = [0] * len(message_tokens)

        attention_mask = [1] * len(message_tokens)

        return message_tokens, loss_mask, attention_mask

    def _validate_and_convert_tokens(
        self,
        full_tokens: torch.Tensor,
        concat_tokens: list[int],
        concat_loss_mask: list[int],
        concat_attention_mask: list[int],
        item: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Validate tokenization and convert to tensors.

        Args:
            full_tokens: Full conversation tokens
            concat_tokens: Concatenated tokens
            concat_loss_mask: Concatenated loss mask
            concat_attention_mask: Concatenated attention mask

        Returns:
            Tuple of (input_ids, loss_mask, attention_mask) as tensors
        """
        full_tokens_list = full_tokens.tolist()

        if len(concat_tokens) != len(full_tokens_list) or not all(
            a == b for a, b in zip(concat_tokens, full_tokens_list, strict=True)
        ):
            logging.warning(
                f"Token mismatch detected! Item: {item}, Full tokenization length: {len(full_tokens_list)}, Concatenated tokens "
                f"length: {len(concat_tokens)}. Using concatenated version."
            )
            return (
                torch.tensor(concat_tokens, dtype=torch.long),
                torch.tensor(concat_loss_mask, dtype=torch.long),
                torch.tensor(concat_attention_mask, dtype=torch.long),
            )

        return (
            full_tokens,
            torch.tensor(concat_loss_mask, dtype=torch.long),
            torch.tensor(concat_attention_mask, dtype=torch.long),
        )

    def __getitem__(self, item):
        try:
            example = self.dataframes[item]
            data_source = example.get("data_source", None)
            messages = self._build_messages(example)
            
            from verl.utils.dataset.vision_utils import (process_image,
                                                        process_video)
            messages_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=False, tokenize=False, **self.apply_chat_template_kwargs
            )
            images = (
                [process_image(image, image_patch_size=self.image_patch_size) for image in example[self.image_key]]
                if self.image_key in example and example[self.image_key]
                else None
            )
            videos = (
                [process_video(video) for video in example[self.video_key]]
                if self.video_key in example and example[self.video_key]
                else None
            )
            
            model_inputs = self.processor(
                text=[messages_text],
                images=images,
                videos=videos,
                return_tensors="pt",
            )
            full_tokens = model_inputs.pop("input_ids")[0]
            model_inputs.pop("attention_mask")
            
            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # Track concatenated tokens for validation
            concat_tokens = []
            concat_loss_mask = []
            concat_attention_mask = []

            i = 0
            while i < len(messages):
                cur_messages = messages[i]
                if cur_messages["role"] == "assistant":
                    # Process assistant message
                    tokens, loss_mask, attention_mask = self._process_message_tokens(
                        messages, images, videos, i, i + 1, is_assistant=True, data_source=data_source
                    )
                    i += 1
                elif cur_messages["role"] in ["user", "system"]:
                    # Process user or system message
                    if cur_messages["role"] == "system" and i != 0:
                        raise ValueError("System message should be the first message")
                    tokens, loss_mask, attention_mask = self._process_message_tokens(
                        messages, images, videos, i, i + 1
                    )
                    i += 1
                else:
                    raise ValueError(f"Unknown role: {cur_messages['role']}")

                concat_tokens.extend(tokens)
                concat_loss_mask.extend(loss_mask)
                concat_attention_mask.extend(attention_mask)

            # Validate and convert tokens
            input_ids, loss_mask, attention_mask = self._validate_and_convert_tokens(
                full_tokens, concat_tokens, concat_loss_mask, concat_attention_mask, item
            )

            # encode prompt
            if messages[0]["role"] == "system":
                assert messages[1]["role"] == "user"
                assert messages[2]["role"] == "assistant"
            elif messages[0]["role"] == "user":
                assert messages[1]["role"] == "assistant"
            else:
                raise ValueError(f"Unknown role: {messages[0]['role']}")

            def compute_position_ids(
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                model_inputs: dict[str, torch.Tensor]
            ):
                # qwen-vl mrope
                if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                    from verl.models.transformers.qwen3_vl import get_rope_index
                else:
                    from verl.models.transformers.qwen2_vl import get_rope_index

                vision_position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids,
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask,
                )  # (3, seq_len)
                valid_mask = attention_mask.bool()
                text_position_ids = torch.ones((1, len(input_ids)), dtype=torch.long)
                text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
                position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
                return position_ids # (4, seq_length)
            
            sequence_length = input_ids.shape[0]
            # Handle sequence length
            if self.pad_mode == DatasetPadMode.RIGHT:
                if sequence_length < self.max_length:
                    # Pad sequences
                    pad_token_id = self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else 0
                    padded_input_ids = torch.full((self.max_length - sequence_length,), pad_token_id, dtype=input_ids.dtype)
                    padded_attention_mask = torch.zeros((self.max_length - sequence_length,), dtype=attention_mask.dtype)
                    padded_loss_mask = torch.zeros((self.max_length - sequence_length,), dtype=loss_mask.dtype)

                    input_ids = torch.cat((input_ids, padded_input_ids))
                    attention_mask = torch.cat((attention_mask, padded_attention_mask))
                    loss_mask = torch.cat((loss_mask, padded_loss_mask))
                elif sequence_length > self.max_length:
                    if self.truncation == "left":
                        input_ids = input_ids[-self.max_length :]
                        attention_mask = attention_mask[-self.max_length :]
                        loss_mask = loss_mask[-self.max_length :]
                    elif self.truncation == "right":
                        input_ids = input_ids[: self.max_length]
                        attention_mask = attention_mask[: self.max_length]
                        loss_mask = loss_mask[: self.max_length]
                    elif self.truncation == "error":
                        raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
                    else:
                        raise ValueError(f"Unknown truncation method {self.truncation}")

                position_ids = compute_position_ids(input_ids, attention_mask, model_inputs)

                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": position_ids,
                    "loss_mask": loss_mask,
                    "multi_modal_inputs": dict(model_inputs),
                }
            elif self.pad_mode == DatasetPadMode.NO_PADDING:
                # truncate input_ids if it is longer than max_length
                if len(input_ids) > self.max_length:
                    input_ids = input_ids[: self.max_length]
                    loss_mask = loss_mask[: self.max_length]
                
                position_ids = compute_position_ids(input_ids, attention_mask, model_inputs)
                
                # return nested tensor with out padding
                return {
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "loss_mask": loss_mask,
                    "multi_modal_inputs": dict(model_inputs),
                }
            else:
                raise ValueError(f"Unknown pad mode {self.pad_mode}")
        except Exception as e:
            import os
            import random

            error_file = "/workspace/car_sft_rl_data/error_img_rl.txt"
            os.makedirs(os.path.dirname(error_file), exist_ok=True)

            # 提取出导致报错的图片信息
            example = self.dataframes[item]
            img_info = example.get(self.image_key, "Unknown Image Info")

            with open(error_file, "a", encoding="utf-8") as f:
                f.write(f"Index: {item} | Error: {e} | Image path/info: {img_info}\n")

            # 随机采样另一条健康的数据替代当前数据，保证训练不中断
            new_item = random.randint(0, len(self.dataframes) - 1)
            return self.__getitem__(new_item)

