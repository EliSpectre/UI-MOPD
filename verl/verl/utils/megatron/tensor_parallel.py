# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
"""
Utilities for using tensor_parallel in megatron
"""

from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import parallel_state as mpu
from megatron.core.fusions.fused_cross_entropy import calculate_logits_max
from megatron.core.tensor_parallel.utils import VocabUtility
from torch.nn import init

if TYPE_CHECKING:
    from megatron.core import ModelParallelConfig


def update_kwargs_with_config(dictionary: dict, config: "ModelParallelConfig"):
    dictionary["config"] = config
    return dictionary


def get_default_kwargs_for_model_parallel_config():
    model_parallel_config_kwargs = {
        "params_dtype": torch.float32,
        "use_cpu_initialization": False,
        "perform_initialization": True,
        "gradient_accumulation_fusion": False,
        "sequence_parallel": False,
    }
    return model_parallel_config_kwargs


def get_default_model_parallel_config():
    from megatron.core import ModelParallelConfig

    return ModelParallelConfig(**get_default_kwargs_for_model_parallel_config())


def get_common_default_kwargs_for_parallel_linear():
    default_model_parallel_config = get_default_model_parallel_config()
    common_default_kwargs = {
        "init_method": init.xavier_normal_,
        "stride": 1,
        "keep_master_weight_for_test": False,
        "config": default_model_parallel_config,
    }
    return common_default_kwargs


def get_default_kwargs_for_column_parallel_linear():
    from megatron.core import ModelParallelConfig

    model_parallel_config_kwargs = get_default_kwargs_for_model_parallel_config()
    column_parallel_config_kwargs = {
        "async_tensor_model_parallel_allreduce": False,
    }
    model_parallel_config_kwargs.update(column_parallel_config_kwargs)
    column_default_kwargs = {
        "config": ModelParallelConfig(**model_parallel_config_kwargs),
    }
    common_default_kwargs = get_common_default_kwargs_for_parallel_linear()
    common_default_kwargs.update(column_default_kwargs)
    return common_default_kwargs


def get_default_kwargs_for_row_parallel_linear():
    common_default_kwargs = get_common_default_kwargs_for_parallel_linear()
    return common_default_kwargs


def get_default_kwargs_for_parallel_embedding():
    from megatron.core import ModelParallelConfig

    model_parallel_config_kwargs = get_default_kwargs_for_model_parallel_config()
    embedding_default_kwargs = {
        "init_method": init.xavier_normal_,
        "config": ModelParallelConfig(**model_parallel_config_kwargs),
    }
    return embedding_default_kwargs


def is_tensor_parallel_param(param):
    return hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel


def get_tensor_parallel_partition_dim(param):
    assert is_tensor_parallel_param(param)
    return param.partition_dim


def get_tensor_parallel_partition_stride(param):
    assert is_tensor_parallel_param(param)
    return param.partition_stride


class _VocabParallelEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
        @torch.compile(dynamic=True)
        def mul_reduce(a, b):
            return (a * b).sum(dim=-1, keepdim=True)

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(normalized_sum_exp_logits, group=mpu.get_tensor_model_parallel_group())
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
        dist.all_reduce(sum_softmax_times_logits, group=mpu.get_tensor_model_parallel_group())
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        # reuse softmax_logits as grad
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        softmax_logits.mul_(vocab_parallel_logits)
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        # recover vocab_parallel_logits
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        softmax_logits.mul_(-1)
        return softmax_logits


def vocab_parallel_entropy(vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy when the logits are sharded in tp ranks

    Args:
        vocab_parallel_logits: (total_nnz, vocab_size // tp_size)

    Returns: (total_nnz,)

    """
    return _VocabParallelEntropy.apply(vocab_parallel_logits)


def vocab_parallel_log_probs_from_logits(logits, labels):
    """TODO(zhangchi.usc1992): We may change the implementation later"""
    from megatron.core import tensor_parallel

    return -tensor_parallel.vocab_parallel_cross_entropy(vocab_parallel_logits=logits, target=labels)


def vocab_parallel_log_probs_from_logits_response_rmpad(input_ids, attention_mask, logits_rmpad, response_length):
    """Similar to log_probs_from_logits_response_rmpad, but the logits_rmpad is now spliited across tensor parallel
    region.
    This will further reduce the peak memory usage during training

    Args:
        input_ids: [batch_size, seqlen]
        attention_mask: [batch_size, seqlen]
        logits_rmpad: [total_nnz, vocab_size // tp_size]
        response_length: int

    """
    from flash_attn.bert_padding import pad_input, unpad_input

    batch_size, seqlen = input_ids.shape
    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask=attention_mask)
    input_ids_rmpad = input_ids_rmpad.squeeze(-1)
    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=0)
    full_log_probs_rmpad = vocab_parallel_log_probs_from_logits(
        logits=logits_rmpad, labels=input_ids_rmpad_rolled
    )  # (total_nnz,)
    full_output = pad_input(
        hidden_states=full_log_probs_rmpad.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
    )
    output = full_output.squeeze(-1)[:, -response_length - 1 : -1]  # [batch_size, response_length]
    return output


class _VocabParallelTopkLogProbs(torch.autograd.Function):
    """Compute log-probs at specified global vocab indices from vocab-parallel logits.

    Handles the tensor-parallel case where the vocabulary dimension is sharded
    across TP ranks. Each rank holds logits for its partition; this function
    gathers the logit values at arbitrary *global* indices, computes the global
    log-softmax normalization, and returns proper log-probabilities.
    """

    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor, global_indices: torch.Tensor) -> torch.Tensor:
        tp_rank = mpu.get_tensor_model_parallel_rank()
        local_vocab_size = vocab_parallel_logits.size(-1)
        vocab_start_index = tp_rank * local_vocab_size
        vocab_end_index = vocab_start_index + local_vocab_size

        mask = (global_indices >= vocab_start_index) & (global_indices < vocab_end_index)
        local_indices = (global_indices - vocab_start_index).clamp(0, local_vocab_size - 1)

        gathered_logits = torch.gather(vocab_parallel_logits, dim=-1, index=local_indices)
        gathered_logits = gathered_logits.masked_fill(~mask, 0.0)
        dist.all_reduce(gathered_logits, op=dist.ReduceOp.SUM, group=mpu.get_tensor_model_parallel_group())

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())

        # Use fused torch.logsumexp to avoid materializing a full [tokens, local_vocab]
        # exp() tensor, which can easily OOM for large micro-batch × seq_len × vocab combos.
        local_logsumexp = torch.logsumexp(vocab_parallel_logits, dim=-1, keepdim=True)
        sum_exp_sum = (local_logsumexp - logits_max).exp()
        dist.all_reduce(sum_exp_sum, group=mpu.get_tensor_model_parallel_group())

        logsumexp = logits_max + sum_exp_sum.log()
        topk_logps = gathered_logits - logsumexp

        ctx.save_for_backward(vocab_parallel_logits, logits_max, sum_exp_sum, mask, local_indices)
        return topk_logps

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        vocab_parallel_logits, logits_max, sum_exp_sum, mask, local_indices = ctx.saved_tensors

        softmax_local = (vocab_parallel_logits - logits_max).exp() / sum_exp_sum

        grad_sum = grad_output.sum(dim=-1, keepdim=True)
        grad_input = -softmax_local * grad_sum
        grad_input.scatter_add_(-1, local_indices, grad_output * mask.float())

        return grad_input, None


def vocab_parallel_topk_log_probs(
    vocab_parallel_logits: torch.Tensor,
    topk: int,
    topk_indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute top-k log-probabilities from vocab-parallel logits.

    When *topk_indices* is ``None``, selects the global top-k tokens across all
    TP ranks.  When provided, gathers log-probs at those global positions.

    Args:
        vocab_parallel_logits: ``(..., vocab_size // tp_size)``
        topk: number of top tokens to select
        topk_indices: ``(..., k)`` optional pre-computed global indices

    Returns:
        ``(topk_logps, topk_indices)`` each of shape ``(..., k)``
    """
    if topk_indices is None:
        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        local_vocab_size = vocab_parallel_logits.size(-1)

        with torch.no_grad():
            local_k = min(topk, local_vocab_size)
            local_topk_values, local_topk_local_idx = torch.topk(vocab_parallel_logits, local_k, dim=-1)
            global_local_idx = local_topk_local_idx + tp_rank * local_vocab_size

            if tp_size > 1:
                gathered_values = [torch.empty_like(local_topk_values) for _ in range(tp_size)]
                gathered_indices = [torch.empty_like(global_local_idx) for _ in range(tp_size)]
                dist.all_gather(gathered_values, local_topk_values, group=mpu.get_tensor_model_parallel_group())
                dist.all_gather(gathered_indices, global_local_idx, group=mpu.get_tensor_model_parallel_group())

                all_values = torch.cat(gathered_values, dim=-1)
                all_indices = torch.cat(gathered_indices, dim=-1)
                _, select_idx = torch.topk(all_values, topk, dim=-1)
                topk_indices = torch.gather(all_indices, dim=-1, index=select_idx)
            else:
                topk_indices = global_local_idx[..., :topk]

    topk_logps = _VocabParallelTopkLogProbs.apply(vocab_parallel_logits, topk_indices)
    return topk_logps, topk_indices


def _add_tail_log_prob(log_probs: torch.Tensor) -> torch.Tensor:
    """Append a tail log-probability representing prob mass outside top-k.

    log(1 - sum(p_i)) = log(1 - exp(logsumexp(log_probs)))
                      = log(-expm1(logsumexp(log_probs)))
    """
    log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
    log_s = torch.clamp(log_s, max=-1e-7)
    tail_log = torch.log(-torch.expm1(log_s))
    return torch.cat([log_probs, tail_log], dim=-1)


def vocab_parallel_kl_divergence(vocab_parallel_logits, target_topk_logps, target_topk_indices, alpha=0.0):
    """Compute divergence between target and source when logits are TP-sharded.

    Supports forward KL, reverse KL, and Generalized Jensen-Shannon Divergence
    controlled by ``alpha``:

    - ``alpha = 0``: forward KL(target || source)
    - ``alpha = 1``: reverse KL(source || target)
    - ``0 < alpha < 1``: Generalized JSD_α(target, source)

    Args:
        vocab_parallel_logits: logits split across tensor parallel ranks
                               dimension is [sequence_length, batch_size, vocab_size_per_partition]
        target_topk_logps: teacher's top-k log-probabilities
                           dimension is [sequence_length, batch_size, top_k]
        target_topk_indices: teacher's top-k token indices (global vocab ids)
                             dimension is [sequence_length, batch_size, top_k]
        alpha: interpolation coefficient. 0 = forward KL, 1 = reverse KL,
               in between = Generalized JSD.

    Returns:
        per_token_loss: [sequence_length, batch_size]
    """
    source_topk_logps = _VocabParallelTopkLogProbs.apply(vocab_parallel_logits, target_topk_indices)

    source_logps = _add_tail_log_prob(source_topk_logps)
    target_logps = _add_tail_log_prob(target_topk_logps)

    if alpha == 0.0:
        per_token_loss = F.kl_div(source_logps, target_logps, reduction="none", log_target=True).sum(-1)
    elif alpha == 1.0:
        per_token_loss = F.kl_div(target_logps, source_logps, reduction="none", log_target=True).sum(-1)
    else:
        alpha_t = torch.tensor(alpha, dtype=source_logps.dtype, device=source_logps.device)
        mixture_logps = torch.logsumexp(
            torch.stack([source_logps + torch.log(1 - alpha_t), target_logps + torch.log(alpha_t)]),
            dim=0,
        )
        kl_target = F.kl_div(mixture_logps, target_logps, reduction="none", log_target=True)
        kl_source = F.kl_div(mixture_logps, source_logps, reduction="none", log_target=True)
        per_token_loss = torch.lerp(kl_source, kl_target, alpha_t).sum(-1)

    return per_token_loss
