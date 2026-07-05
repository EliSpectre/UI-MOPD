# Copyright 2025 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from enum import Enum

import torch


class DatasetPadMode(str, Enum):
    """Padding mode for dataset"""

    RIGHT = "right"
    LEFT_RIGHT = "left_right"
    NO_PADDING = "no_padding"


class SFTTensorCollator:
    """
    A custom collate_fn that handles batching of sequences.
    1. for variable-length sequences, convert them into NestedTensors.
    2. for fixed-length sequences, use default_collate.
    """

    def __init__(self, pad_mode: DatasetPadMode = DatasetPadMode.LEFT_RIGHT):
        self.pad_mode = pad_mode

    def __call__(self, batch: list[dict[str, any]]) -> dict[str, any]:
        if self.pad_mode == DatasetPadMode.NO_PADDING:
            return self.collate_variable_batch(batch)
        elif self.pad_mode in [DatasetPadMode.RIGHT, DatasetPadMode.LEFT_RIGHT]:
            from verl.utils.dataset.rl_dataset import collate_fn

            return collate_fn(batch)
        else:
            raise NotImplementedError(f"pad_mode {self.pad_mode} not implemented")

    def collate_variable_batch(self, batch: list[dict[str, any]]) -> dict[str, any]:

        final_batch = {}
        keys = batch[0].keys()

        # check if all samples have the same keys
        for item in batch:
            assert item.keys() == keys, "Samples in batch have inconsistent keys"

        for key in keys:
            values = [item[key] for item in batch]

            # case 1: Tensor → NestedTensor
            if isinstance(values[0], torch.Tensor):
                final_batch[key] = torch.nested.as_nested_tensor(
                    values, layout=torch.jagged
                )
                continue

            # case 2: number → tensor
            if isinstance(values[0], (int, float)):
                final_batch[key] = torch.tensor(values)
                continue

            # case 3: others → list
            final_batch[key] = values

        return final_batch
