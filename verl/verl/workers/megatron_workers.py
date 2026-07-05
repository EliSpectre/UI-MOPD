# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
The main entry point to run the PPO algorithm
"""

import datetime
import logging
import os
import time
from typing import Any, Optional

import psutil
import ray
import torch
import torch.distributed
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf

try:
    from mindspeed.megatron_adaptor import repatch
except ImportError:
    repatch = None

from megatron.core import parallel_state as mpu

from verl import DataProto
from verl.models.mcore import get_mcore_weight_converter
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import (
    Dispatch, make_nd_compute_dataproto_dispatch_fn, register)
from verl.utils import hf_tokenizer
from verl.utils.checkpoint.megatron_checkpoint_manager import \
    MegatronCheckpointManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import (get_device_id, get_device_name,
                               get_nccl_backend, get_torch_device,
                               set_expandable_segments)
from verl.utils.distributed import set_numa_affinity
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.megatron.router_replay_patch import (RouterReplay,
                                                     RouterReplayAction,
                                                     apply_router_replay_patch)
from verl.utils.megatron_utils import (load_megatron_model_to_gpu,
                                       load_megatron_optimizer,
                                       offload_megatron_model_to_cpu,
                                       offload_megatron_optimizer,
                                       per_tensor_generator,
                                       register_megatron_training_hooks,
                                       unwrap_model)
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.model import (get_hf_model_path, get_hf_model_ref_path,
                              load_mcore_dist_weights,
                              load_megatron_gptmodel_weights)
from verl.utils.profiler import (DistProfiler, DistProfilerExtension,
                                 GPUMemoryLogger, ProfilerConfig,
                                 log_gpu_memory_usage, simple_timer)
from verl.utils.profiler.performance import (reduce_timing,
                                             topk_reduce_ratio_min_max)
from verl.utils.ray_utils import get_event_loop
from verl.utils.router_replay_cache import (ROUTED_EXPERTS_BATCH_KEY,
                                            ROUTED_EXPERTS_CACHE_ID_KEY,
                                            ROUTED_EXPERTS_CACHE_SOURCE_KEY)
from verl.utils.torch_functional import use_original_torch_compile
from verl.workers.actor.megatron_actor import MegatronPPOActor
from verl.workers.config import HFModelConfig, McoreCriticConfig, RolloutConfig
from verl.workers.critic.megatron_critic import MegatronPPOCritic
from verl.workers.reward_model.megatron.reward_model import MegatronRewardModel
from verl.workers.rollout import get_rollout_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def set_random_seed(seed, only_rollout=False):
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if not only_rollout and get_torch_device().device_count() > 0:
        from megatron.core import tensor_parallel

        tensor_parallel.model_parallel_cuda_manual_seed(seed)
    # FIXME: torch cumsum not support deterministic (used in vllm sampler),
    # https://github.com/pytorch/pytorch/issues/89492
    # torch.use_deterministic_algorithms(True, warn_only=True)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


class MegatronWorker(Worker):
    def _init_hf_config_and_tf_config(
        self,
        model_path,
        tokenizer_or_path,
        dtype,
        override_model_config,
        override_transformer_config,
        trust_remote_code=False,
        megatron_config=None,
    ):
        from transformers import AutoConfig

        from verl.models.mcore import hf_to_mcore_config
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.fs import copy_to_local
        from verl.utils.model import update_model_config

        # Step 1: initialize the tokenizer
        self.local_path = copy_to_local(model_path)
        if tokenizer_or_path is None:
            self.tokenizer = hf_tokenizer(self.local_path, trust_remote_code=trust_remote_code)
            self.processor = hf_processor(self.local_path, trust_remote_code=trust_remote_code)
        elif isinstance(tokenizer_or_path, str):
            self.tokenizer = hf_tokenizer(copy_to_local(tokenizer_or_path), trust_remote_code=trust_remote_code)
            self.processor = hf_processor(copy_to_local(tokenizer_or_path), trust_remote_code=trust_remote_code)
        else:
            self.tokenizer = tokenizer_or_path
            self.processor = tokenizer_or_path

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        # Step 2: get the hf
        hf_config = AutoConfig.from_pretrained(self.local_path, trust_remote_code=trust_remote_code)

        # Step 3: override the hf config
        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config.get("model_config", {}))
        self.share_embeddings_and_output_weights = getattr(hf_config, "tie_word_embeddings", False)
        update_model_config(hf_config, override_config_kwargs=override_config_kwargs)
        self.architectures = getattr(hf_config, "architectures", None)
        if self.rank == 0:
            print(f"Model config after override: {hf_config}")

        from verl.models.mcore.config_converter import \
            mapping_string_to_attn_backend

        # todo: remove this line after mcore adopt mbridge 0.15, now for compatibility
        override_transformer_config = mapping_string_to_attn_backend(override_transformer_config)
        fp16 = dtype == torch.float16
        bf16 = dtype == torch.bfloat16
        if fp16:
            assert megatron_config.use_mbridge, "fp16 mode requires use_mbridge to be True"

        self.provider = None
        self.vanilla_bridge = megatron_config.get("vanilla_mbridge", True)
        if megatron_config.use_mbridge:
            if self.vanilla_bridge:
                from verl.models.mcore.mbridge import AutoBridge

                bridge = AutoBridge.from_config(hf_config, dtype=dtype)
                bridge.set_extra_args(**override_transformer_config)
                tf_config = bridge.config
                tf_config.fp16 = fp16
                tf_config.bf16 = bf16
            else:
                from verl.models.mcore.bridge import AutoBridge

                # Use Megatron-Bridge to convert HF config to Megatron config
                bridge = AutoBridge.from_hf_pretrained(self.local_path, trust_remote_code=trust_remote_code)
                # Get Megatron provider and configure it
                provider = bridge.to_megatron_provider(load_weights=False)

                # In case of invalid overrides, we need to make sure some critical params are set correctly
                provider.params_dtype = dtype

                # Pass distributed info
                provider.tensor_model_parallel_size = megatron_config.tensor_model_parallel_size
                provider.pipeline_model_parallel_size = megatron_config.pipeline_model_parallel_size
                provider.expert_model_parallel_size = megatron_config.expert_model_parallel_size
                provider.expert_tensor_parallel_size = megatron_config.expert_tensor_parallel_size
                provider.virtual_pipeline_model_parallel_size = megatron_config.virtual_pipeline_model_parallel_size
                provider.context_parallel_size = megatron_config.context_parallel_size
                provider.sequence_parallel = megatron_config.sequence_parallel

                # Match verl implementation (need variable_seq_lengths)
                from megatron.core.transformer.enums import AttnBackend

                provider.attention_backend = AttnBackend.flash
                provider.variable_seq_lengths = True
                provider.moe_token_dispatcher_type = "alltoall"
                provider.moe_router_load_balancing_type = "none"

                # Apply transformer config overrides
                for key, value in override_transformer_config.items():
                    setattr(provider, key, value)

                provider.finalize()
                self.provider = provider
                tf_config = None  # Will be set after model creation
            self.bridge = bridge
        else:
            tf_config = hf_to_mcore_config(hf_config, dtype, **override_transformer_config)
            self.bridge = None

        if torch.distributed.get_rank() == 0:
            if tf_config is not None:
                print(f"TF config: {tf_config}")
        self.hf_config = hf_config
        self.tf_config = tf_config

        # Get PEFT config from model.lora if specified
        from verl.workers.config.megatron_peft import get_peft_cls

        self.peft_cls = get_peft_cls(
            model_config=self.config.model, bridge=self.bridge, provider=self.provider, dtype=dtype
        )


class ActorRolloutRefWorker(MegatronWorker, DistProfilerExtension):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)
        self.config = config
        if repatch is not None:
            # NPU MindSpeed patch, will be refactored with MindSpeedEngine.
            repatch(self.config.actor.megatron.get("override_transformer_config", {}))

        self.role = role
        assert self.role in ["actor", "rollout", "ref", "actor_rollout", "actor_rollout_ref"]

        self._is_actor = self.role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = self.role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = self.role in ["ref", "actor_rollout_ref"]

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel strategy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        if not torch.distributed.is_initialized():
            set_numa_affinity()
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
            get_torch_device().set_device(rank)

            if self._is_actor or self._is_ref:
                mpu.initialize_model_parallel(
                    tensor_model_parallel_size=self.config.actor.megatron.tensor_model_parallel_size,
                    pipeline_model_parallel_size=self.config.actor.megatron.pipeline_model_parallel_size,
                    virtual_pipeline_model_parallel_size=self.config.actor.megatron.virtual_pipeline_model_parallel_size,
                    use_sharp=False,
                    context_parallel_size=self.config.actor.megatron.context_parallel_size,
                    expert_model_parallel_size=self.config.actor.megatron.expert_model_parallel_size,
                    expert_tensor_parallel_size=self.config.actor.megatron.expert_tensor_parallel_size,
                    nccl_communicator_config_path=None,
                )
        
        if not getattr(torch.distributed, "_broadcast_no_grad_patched", False):
            _orig_broadcast = torch.distributed.broadcast

            def _broadcast_no_grad(*args, **kwargs):
                with torch.no_grad():
                    return _orig_broadcast(*args, **kwargs)

            torch.distributed.broadcast = _broadcast_no_grad
            torch.distributed._broadcast_no_grad_patched = True

        if self._is_actor or self._is_ref:
            is_collect = (
                mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
                and mpu.get_context_parallel_rank() == 0
            )
            self._register_dispatch_collect_info(
                mesh_name="actor", dp_rank=mpu.get_data_parallel_rank(), is_collect=is_collect
            )
        only_rollout = self._is_rollout and not self._is_actor

        self.enable_routing_replay = False
        if self._is_actor:
            self.router_replay = self.config.actor.router_replay
            self.enable_routing_replay = self.router_replay.mode != "disabled"
        self._routed_experts_cache: dict[str, torch.Tensor] = {}
        self._routed_experts_source_handles: dict[str, Any] = {}

        if self.enable_routing_replay:
            apply_router_replay_patch()

        set_random_seed(seed=self.config.actor.megatron.seed, only_rollout=only_rollout)

        if self._is_actor:
            omega_profiler_config = config.actor.get("profiler", {})
        elif self._is_rollout:
            # NOTE: In colocation mode, rollout config may not take effect (follow the actor config)
            # This is for extendability in AsyncRL cases
            omega_profiler_config = config.rollout.get("profiler", {})
        elif self._is_ref:
            omega_profiler_config = config.ref.get("profiler", {})
        else:
            raise ValueError(
                f"Invalid role {self.role}, should be one of "
                "['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']"
            )
        # omega_profiler_config is DictConfig
        # profiler_config is a ProfilerConfig dataclass
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )

        # TODO(sgm): Currently, we only support reference model param offload
        # will support other offload later
        self._is_offload_param = False
        self._is_offload_grad = False
        self._is_offload_optimizer = False

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
            if self.config.actor.get("ppo_micro_batch_size", None):
                self.config.actor.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()
                self.config.rollout.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size
                self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size

            self._is_offload_param = self.config.actor.megatron.get("param_offload", False)
            self._is_offload_grad = self.config.actor.megatron.get("grad_offload", False)
            self._is_offload_optimizer = self.config.actor.megatron.get("optimizer_offload", False)
        elif self._is_ref:
            if self.config.ref.get("log_prob_micro_batch_size", None):
                self.config.ref.log_prob_micro_batch_size //= mpu.get_data_parallel_world_size()
                self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size
            else:
                assert self.config.ref.get("log_prob_micro_batch_size_per_gpu", None) is not None, (
                    "Please note that in the ref policy configuration, `log_prob_micro_batch_size_per_gpu` and "
                    "`log_prob_micro_batch_size` should not be None at the same time."
                )
            self._ref_is_offload_param = self.config.ref.megatron.get("param_offload", False)

    def _build_model_optimizer(
        self, model_path, optim_config, override_model_config, override_transformer_config, override_ddp_config=None
    ):
        from verl.utils.megatron.optimizer import (
            get_megatron_optimizer, get_megatron_optimizer_param_scheduler,
            init_megatron_optim_config)
        from verl.utils.megatron_utils import (McoreModuleWrapperConfig,
                                               make_megatron_module)
        from verl.utils.model import get_generation_config, print_model_size

        self._init_hf_config_and_tf_config(
            model_path,
            self.config.model.get("tokenizer_path") or model_path,
            self.dtype,
            override_model_config,
            override_transformer_config,
            self.config.model.get("trust_remote_code", False),
            self.config.actor.megatron if not self._is_ref else self.config.ref.megatron,
        )
        self.generation_config = get_generation_config(
            self.local_path,
            self.config.model.get("trust_remote_code", False),
        )

        if self._is_actor or self._is_rollout:
            wrap_config = McoreModuleWrapperConfig(
                is_value_model=False,  # actor is not value model
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                wrap_with_ddp=True,
                use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
            )
            actor_module, updated_tf_config = make_megatron_module(
                wrap_config=wrap_config,
                tf_config=self.tf_config,
                hf_config=self.hf_config,
                bridge=self.bridge,
                provider=self.provider,
                override_model_config=override_model_config,
                override_ddp_config=override_ddp_config,
                peft_cls=self.peft_cls,
                peft_config=self.config.model.get("lora", None),
            )
            self.tf_config = updated_tf_config
            print(f"actor_module: {len(actor_module)}")
            if self.config.actor.load_weight:
                if self.config.actor.megatron.use_dist_checkpointing:
                    load_mcore_dist_weights(
                        actor_module,
                        self.config.actor.megatron.dist_checkpointing_path,
                        is_value_model=False,
                        prefix=self.config.actor.megatron.dist_checkpointing_prefix,
                    )
                else:
                    if self.bridge is not None:
                        local_model_path = get_hf_model_path(self.config)
                        if self.vanilla_bridge:
                            self.bridge.load_weights(actor_module, local_model_path)
                        else:
                            self.bridge.load_hf_weights(actor_module, local_model_path)
                    else:
                        load_megatron_gptmodel_weights(
                            self.config, self.hf_config, actor_module, params_dtype=self.dtype, is_value_model=False
                        )

            if self.rank == 0:
                print_model_size(actor_module[0])
            log_gpu_memory_usage("After MegatronPPOActor init", logger=logger)
        elif self._is_ref:
            wrap_config = McoreModuleWrapperConfig(
                is_value_model=False,  # ref is not value model
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                wrap_with_ddp=False,
                use_distributed_optimizer=self.config.ref.megatron.use_distributed_optimizer,
            )
            ref_module, updated_tf_config = make_megatron_module(
                wrap_config=wrap_config,
                tf_config=self.tf_config,
                hf_config=self.hf_config,
                bridge=self.bridge,
                provider=self.provider,
                override_model_config=override_model_config,
            )
            self.tf_config = updated_tf_config
            if self.config.ref.load_weight:  # should align with the actor:
                assert self.config.actor.load_weight == self.config.ref.load_weight
                print("load ref weight start")
                if self.config.ref.megatron.use_dist_checkpointing:
                    load_mcore_dist_weights(
                        ref_module,
                        self.config.ref.megatron.dist_checkpointing_path,
                        is_value_model=False,
                        prefix=self.config.ref.megatron.dist_checkpointing_prefix,
                    )
                else:
                    if self.bridge is not None:
                        local_model_path = get_hf_model_ref_path(self.config)
                        if self.vanilla_bridge:
                            self.bridge.load_weights(ref_module, local_model_path)
                        else:
                            self.bridge.load_hf_weights(ref_module, local_model_path)
                    else:
                        load_megatron_gptmodel_weights(
                            self.config, self.hf_config, ref_module, params_dtype=self.dtype, is_value_model=False
                        )
            log_gpu_memory_usage("After ref module init", logger=logger)
            return ref_module, self.hf_config

        # TODO: add more optimizer args into config
        if self._is_actor:
            optim_config_megatron = init_megatron_optim_config(
                optim_config,
                use_distributed_optimizer=wrap_config.use_distributed_optimizer,
                fp16=self.dtype == torch.float16,
            )
            actor_optimizer = get_megatron_optimizer(model=actor_module, config=optim_config_megatron)
            actor_optimizer_scheduler = get_megatron_optimizer_param_scheduler(
                optimizer=actor_optimizer, config=optim_config
            )
        else:
            optim_config = None
            actor_optimizer = None
            actor_optimizer_scheduler = None

        log_gpu_memory_usage("After actor optimizer init", logger=logger)

        register_megatron_training_hooks(actor_module, actor_optimizer)

        return actor_module, actor_optimizer, actor_optimizer_scheduler, self.hf_config, optim_config

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # 1. parse rollout and huggingface model config
        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)

        # Convert megatron lora config to HFModelConfig
        model_config_dict = OmegaConf.to_container(self.config.model)
        model_config_dict.pop("lora", None)

        model_config: HFModelConfig = omega_conf_to_dataclass(
            OmegaConf.create(model_config_dict), dataclass_type=HFModelConfig
        )

        # 2. build rollout device mesh
        infer_tp = self.config.rollout.tensor_model_parallel_size * self.config.rollout.data_parallel_size
        infer_pp = self.config.rollout.pipeline_model_parallel_size
        infer_world_size = infer_tp * infer_pp
        dp = self.world_size // infer_world_size
        assert self.world_size % infer_world_size == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_world_size: {infer_world_size}"
        )
        rollout_device_mesh = init_device_mesh(
            get_device_name(), mesh_shape=(dp, infer_tp, infer_pp), mesh_dim_names=["dp", "infer_tp", "infer_pp"]
        )

        is_collect = (
            rollout_device_mesh["infer_tp"].get_local_rank() == 0
            and rollout_device_mesh["infer_pp"].get_local_rank() == 0
        )
        self._register_dispatch_collect_info(
            "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
        )

        # 3. init trainer and rollout random states
        self.torch_random_states = get_torch_device().get_rng_state()
        gen_dp_rank = rollout_device_mesh["dp"].get_local_rank()
        get_torch_device().manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

        # 4. build rollout model
        log_gpu_memory_usage(f"Before building {self.config.rollout.name} rollout", logger=logger)
        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
        )
        log_gpu_memory_usage(f"After building {self.config.rollout.name} rollout", logger=logger)

        # 5. switch to trainer mode
        # NOTE: It's critical that hybrid engine in trainer mode initially to load checkpoint.
        # For async mode, we can't call run_until_complete here, so we will switch to trainer mode in AgentLoopManager.
        # Note: sync mode is deprecated and rejected in RolloutConfig.__post_init__

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        from verl.utils.torch_dtypes import PrecisionType

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        if self._is_actor:
            override_transformer_config = OmegaConf.to_container(
                OmegaConf.create(self.config.actor.megatron.get("override_transformer_config", {}))
            )
            if self.enable_routing_replay:
                override_transformer_config["enable_routing_replay"] = True
            override_ddp_config = OmegaConf.to_container(
                OmegaConf.create(self.config.actor.megatron.get("override_ddp_config", {}))
            )
        elif self._is_ref:
            override_transformer_config = OmegaConf.to_container(
                OmegaConf.create(self.config.ref.megatron.get("override_transformer_config", {}))
            )
        else:
            override_transformer_config = {}
        self.param_dtype = PrecisionType.to_dtype(self.config.actor.megatron.dtype)
        log_gpu_memory_usage("Before init actor model and optimizer", logger=logger)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        if self._is_actor:
            # we need the model for actor and rollout
            optim_config = self.config.actor.optim if self._is_actor else None
            (
                self.actor_module,
                self.actor_optimizer,
                self.actor_optimizer_scheduler,
                self.actor_model_config,
                self.actor_optim_config,
            ) = self._build_model_optimizer(
                model_path=self.config.model.path,
                optim_config=optim_config,
                override_model_config=override_model_config,
                override_transformer_config=override_transformer_config,
                override_ddp_config=override_ddp_config,
            )
            if self._is_offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
                log_gpu_memory_usage("After offload actor params and grad during init", logger=logger)
            if self._is_offload_optimizer:
                offload_megatron_optimizer(self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = MegatronPPOActor(
                config=actor_cfg,
                model_config=self.actor_model_config,
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                actor_module=self.actor_module,
                actor_optimizer=self.actor_optimizer,
            )
            print(f"routing replay layers: {len(RouterReplay.router_instances)}")
            log_gpu_memory_usage("After MegatronPPOActor init", logger=logger)

        if self._is_rollout:
            with use_original_torch_compile():
                self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))
            log_gpu_memory_usage("After rollout init", logger=logger)

        if self._is_ref:
            print(f"Building ref model from {self.config.model.ref_path}")
            self.ref_module, self.ref_model_config = self._build_model_optimizer(
                model_path=self.config.model.ref_path,
                optim_config=None,
                override_model_config=override_model_config,
                override_transformer_config=override_transformer_config,
            )
            log_gpu_memory_usage("After ref model init", logger=logger)
            self.ref_policy = MegatronPPOActor(
                config=self.config.ref,
                model_config=self.ref_model_config,
                hf_config=self.hf_config,
                tf_config=self.tf_config,
                actor_module=self.ref_module,
                actor_optimizer=None,
            )
            if self._ref_is_offload_param:
                offload_megatron_model_to_cpu(self.ref_module)
                log_gpu_memory_usage("After offload ref params during init", logger=logger)

            # Dual-teacher OPD: build mobile ref model if ref_mobile_path is set
            self._has_mobile_ref = False
            ref_mobile_path = getattr(self.config.model, "ref_mobile_path", None)
            if ref_mobile_path:
                print(f"Building mobile ref model from {ref_mobile_path}")
                self.ref_mobile_module, self.ref_mobile_model_config = self._build_model_optimizer(
                    model_path=ref_mobile_path,
                    optim_config=None,
                    override_model_config=override_model_config,
                    override_transformer_config=override_transformer_config,
                )
                log_gpu_memory_usage("After mobile ref model init", logger=logger)
                self.ref_mobile_policy = MegatronPPOActor(
                    config=self.config.ref,
                    model_config=self.ref_mobile_model_config,
                    hf_config=self.hf_config,
                    tf_config=self.tf_config,
                    actor_module=self.ref_mobile_module,
                    actor_optimizer=None,
                )
                if self._ref_is_offload_param:
                    offload_megatron_model_to_cpu(self.ref_mobile_module)
                    log_gpu_memory_usage("After offload mobile ref params during init", logger=logger)
                self._has_mobile_ref = True

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_mananager = MegatronCheckpointManager(
                config=self.config,
                checkpoint_config=self.config.actor.checkpoint,
                model_config=self.actor_model_config,
                transformer_config=self.tf_config,
                role="actor",
                model=self.actor_module,
                arch=self.architectures[0],
                hf_config=self.hf_config,
                param_dtype=self.param_dtype,
                share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                optimizer=self.actor_optimizer,
                optimizer_scheduler=self.actor_optimizer_scheduler,
                use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
                use_checkpoint_opt_param_scheduler=self.config.actor.optim.use_checkpoint_opt_param_scheduler,
                bridge=self.bridge,
                provider=self.provider,
                use_dist_checkpointing=self.config.actor.megatron.use_dist_checkpointing,
                peft_cls=self.peft_cls,
            )

            self.layer_name_mapping = {
                "qkv_layer_name": "self_attention.linear_qkv.",
                "gate_proj_layer_name": "linear_fc1.",
            }
            self.weight_converter = None
            if not self.config.actor.megatron.use_mbridge:
                self.weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)

        get_torch_device().empty_cache()
        log_gpu_memory_usage("After init_model finish", logger=logger)

    def _hydrate_routed_experts_from_cache(self, data: DataProto, *, clear_cache=False):
        if not (self.enable_routing_replay and self.config.actor.router_replay.mode == "R3"):
            return
        if data.batch is None or ROUTED_EXPERTS_BATCH_KEY in data.batch.keys():
            return

        cache_ids = data.non_tensor_batch.get(ROUTED_EXPERTS_CACHE_ID_KEY, None)
        if cache_ids is None:
            return

        cache_id_list = [str(cache_id) for cache_id in cache_ids.tolist()]
        self._fetch_missing_routed_experts_from_source(data, cache_id_list)
        missing_ids = [cache_id for cache_id in cache_id_list if cache_id not in self._routed_experts_cache]
        if missing_ids:
            missing_preview = missing_ids[:4]
            raise KeyError(
                f"Missing routed_experts cache entries on rank={self.rank}: "
                f"{missing_preview} (missing={len(missing_ids)})"
            )

        data.batch[ROUTED_EXPERTS_BATCH_KEY] = torch.stack(
            [self._routed_experts_cache[cache_id] for cache_id in cache_id_list], dim=0
        ).contiguous()
        if clear_cache:
            self._routed_experts_cache.clear()

    def _pad_routed_experts_for_sample(self, data: DataProto, sample_idx: int, routed_experts: torch.Tensor):
        if data.batch is None:
            return routed_experts.cpu().contiguous()

        total_length = data.batch["input_ids"].shape[1]
        if routed_experts.shape[0] == total_length:
            return routed_experts.cpu().contiguous()

        if "prompts" in data.batch.keys():
            prompt_length = data.batch["prompts"].shape[1]
        elif "responses" in data.batch.keys():
            prompt_length = total_length - data.batch["responses"].shape[1]
        else:
            raise KeyError("Unable to infer prompt length when hydrating routed_experts from rollout cache")

        prompt_token_num = int(data.batch["attention_mask"][sample_idx, :prompt_length].sum().item())
        start_pos = prompt_length - prompt_token_num
        if start_pos < 0 or start_pos >= total_length:
            raise ValueError(
                f"Invalid routed_experts padding window on rank={self.rank}: "
                f"start_pos={start_pos}, prompt_length={prompt_length}, total_length={total_length}"
            )

        padded = torch.zeros(
            total_length,
            routed_experts.shape[1],
            routed_experts.shape[2],
            dtype=routed_experts.dtype,
        )
        copy_length = min(routed_experts.shape[0], total_length - start_pos)
        padded[start_pos : start_pos + copy_length] = routed_experts[:copy_length]
        return padded.contiguous()

    def _get_routed_experts_source_handle(self, cache_source: str):
        handle = self._routed_experts_source_handles.get(cache_source, None)
        if handle is None:
            handle = ray.get_actor(cache_source)
            self._routed_experts_source_handles[cache_source] = handle
        return handle

    def _fetch_missing_routed_experts_from_source(self, data: DataProto, cache_id_list: list[str]):
        missing_indices = [idx for idx, cache_id in enumerate(cache_id_list) if cache_id not in self._routed_experts_cache]
        if not missing_indices:
            return

        cache_sources = data.non_tensor_batch.get(ROUTED_EXPERTS_CACHE_SOURCE_KEY, None)
        if cache_sources is None:
            return

        source_to_indices: dict[str, list[int]] = {}
        for idx in missing_indices:
            cache_source = str(cache_sources[idx])
            source_to_indices.setdefault(cache_source, []).append(idx)

        for cache_source, indices in source_to_indices.items():
            source_cache_ids = [cache_id_list[idx] for idx in indices]
            source_handle = self._get_routed_experts_source_handle(cache_source)
            routed_experts_list = ray.get(source_handle.fetch_routed_experts.remote(source_cache_ids))
            if self.rank == 0:
                logger.info(
                    "fetch_routed_experts_from_source: source=%s hydrated %s entries on actor worker",
                    cache_source,
                    len(source_cache_ids),
                )
            for idx, cache_id, routed_experts in zip(indices, source_cache_ids, routed_experts_list, strict=True):
                self._routed_experts_cache[cache_id] = self._pad_routed_experts_for_sample(
                    data, idx, routed_experts
                )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def clear_all_routed_experts_cache(self):
        """Clear all routed experts cache entries unconditionally."""
        count = len(self._routed_experts_cache)
        self._routed_experts_cache.clear()
        if self.rank == 0 and count:
            logger.info("clear_all_routed_experts_cache: cleared %s entries on actor workgroup", count)
        return count
    
    async def rollout_mode(self):
        """Context switch hybridengine to rollout mode."""
        aggressive_empty_cache(force_sync=True)
        set_expandable_segments(False)

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor.actor_module, load_grad=False)
            log_gpu_memory_usage("After load actor params during rollout_mode", logger=logger)

        if self.bridge is not None:
            if self.vanilla_bridge:
                per_tensor_param = self.bridge.export_weights(self.actor.actor_module)
            else:
                per_tensor_param = self.bridge.export_hf_weights(self.actor.actor_module)
        else:
            per_tensor_param = per_tensor_generator(
                self.actor.actor_module,
                self.actor_model_config,
                self.weight_converter,
                self.tf_config,
                self.layer_name_mapping,
            )

        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        await self.rollout.update_weights(per_tensor_param)
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor.actor_module)
        aggressive_empty_cache(force_sync=True)
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])

        # important: need to manually set the random states of each tp to be identical.
        self.torch_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.gen_random_states)

    async def trainer_mode(self):
        """Context switch hybridengine to trainer mode."""
        if self.config.rollout.free_cache_engine:
            log_gpu_memory_usage("Before rollout offload", logger=logger)
            await self.rollout.release()
            log_gpu_memory_usage("After rollout offload", logger=logger)

        for model in self.actor.actor_module:
            model.train()
        # add empty cache after each compute
        aggressive_empty_cache(force_sync=True)

        # FIXME(@wuxibin): megatron+sglang failed with `expandable_segments:True` in ci,
        # can't reproduce it in dev environment, temporary disable it.
        # https://github.com/volcengine/verl/actions/runs/17382936845/job/49344264323?pr=3285
        if os.environ.get("MEGATRON_CI_DISABLE_EXPANDABLE_SEGMENTS", "0") == "0":
            set_expandable_segments(True)

        # restore random states
        self.gen_random_states = get_torch_device().get_rng_state()
        get_torch_device().set_rng_state(self.torch_random_states)

    @torch.no_grad()
    def _ema_update_ref_model(self, decay: float = 0.999):
        """将 actor_module 的参数通过 EMA 方式更新到 ref_module 中。
        ref_param = decay * ref_param + (1 - decay) * actor_param
        """
        if not self._is_ref:
            return

        if self._ref_is_offload_param:
            load_megatron_model_to_gpu(self.ref_module, load_grad=False)
            log_gpu_memory_usage("After load ref params for EMA update", logger=logger)

        for actor_chunk, ref_chunk in zip(self.actor_module, self.ref_module):
            actor_unwrapped = unwrap_model(actor_chunk)
            ref_unwrapped = unwrap_model(ref_chunk)
            for (name_a, param_a), (name_r, param_r) in zip(
                actor_unwrapped.named_parameters(), ref_unwrapped.named_parameters()
            ):
                assert name_a == name_r, f"Parameter name mismatch: {name_a} vs {name_r}"
                param_r.data.mul_(decay).add_(param_a.data, alpha=1.0 - decay)
        
        print(f"EMA update ref model, decay: {decay}")

        if self._ref_is_offload_param:
            offload_megatron_model_to_cpu(self.ref_module)
            log_gpu_memory_usage("After offload ref params for EMA update", logger=logger)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @GPUMemoryLogger(role="update_actor", logger=logger)
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
            log_gpu_memory_usage("After load actor params and grad during update_actor", logger=logger)
        if self._is_offload_optimizer:
            load_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After load actor optimizer during update_actor", logger=logger)
        if (
            self.enable_routing_replay
            and self.config.actor.router_replay.mode == "R3"
            and bool(self.config.rollout.get("routed_experts_server_cache", False))
        ):
            self._hydrate_routed_experts_from_cache(data, clear_cache=True)

        micro_batch_size = self.config.actor.ppo_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        dataloader = self.actor.make_minibatch_iterator(data=data)
        with Timer(name="update_policy", logger=None) as timer:
            metrics = self.actor.update_policy(dataloader=dataloader)
        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics["perf/mfu/actor"] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
        metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
        metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
        metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)
        from verl.utils.megatron.optimizer import get_megatron_last_lr

        metrics["actor/lr"] = get_megatron_last_lr(self.actor_optimizer)
        self.actor_optimizer_scheduler.step(1)

        # ema_decay = self.config.actor.get("ema_decay", None)
        # if ema_decay is not None and self._is_ref:
        #     self._ema_update_ref_model(decay=ema_decay)
        #     log_gpu_memory_usage("After EMA update ref model", logger=logger)

        # TODO: here, we should return all metrics
        output = DataProto(meta_info={"metrics": metrics})
        output = output.to("cpu")

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
            log_gpu_memory_usage("After offload actor params and grad during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        aggressive_empty_cache(force_sync=True)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @GPUMemoryLogger(role="generate_sequences", logger=logger)
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        assert self._is_rollout
        prompts = prompts.to(get_device_name())
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)

        timing_generate = {}
        if self._is_actor:  # For rollout only, we do not switch context.
            loop = get_event_loop()
            loop.run_until_complete(self.rollout_mode())
            log_gpu_memory_usage("After switch to rollout mode", logger=logger)

        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)

        if self._is_actor:
            loop.run_until_complete(self.trainer_mode())
            log_gpu_memory_usage("After switch to trainer mode", logger=logger)

        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")
        # clear kv cache
        aggressive_empty_cache(force_sync=True)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @GPUMemoryLogger(role="compute_ref_log_prob", logger=logger)
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref
        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature

        if not self._has_mobile_ref:
            # Single teacher: original logic
            if self._ref_is_offload_param:
                load_megatron_model_to_gpu(self.ref_module, load_grad=False)
                log_gpu_memory_usage("After load ref params during compute_ref_log_prob", logger=logger)
            output, _, _, extra_out = self.ref_policy.compute_log_prob(data=data, calculate_entropy=False)
            tensors = {"ref_log_prob": output}
            tensors.update(extra_out)
            output = DataProto.from_dict(tensors=tensors)
            output = output.to("cpu")
            if self._ref_is_offload_param:
                offload_megatron_model_to_cpu(self.ref_module)
                log_gpu_memory_usage("After offload ref params during compute_ref_log_prob", logger=logger)
        else:
            # Dual-teacher OPD: route by data_source
            import numpy as np
            data_sources = data.non_tensor_batch.get("data_source", None)
            mobile_indices = []
            desktop_indices = []
            if data_sources is not None:
                for i, ds in enumerate(data_sources):
                    if "mobile" in str(ds):
                        mobile_indices.append(i)
                    else:
                        desktop_indices.append(i)
            else:
                desktop_indices = list(range(len(data)))

            batch_size = len(data)
            ref_log_prob_combined = None

            # Desktop teacher forward
            if desktop_indices:
                if self._ref_is_offload_param:
                    load_megatron_model_to_gpu(self.ref_module, load_grad=False)
                desktop_data = data[desktop_indices]
                desktop_data.meta_info = data.meta_info.copy()
                desktop_out, _, _, _ = self.ref_policy.compute_log_prob(data=desktop_data, calculate_entropy=False)
                if self._ref_is_offload_param:
                    offload_megatron_model_to_cpu(self.ref_module)
                aggressive_empty_cache(force_sync=True)
            else:
                desktop_out = None

            # Mobile teacher forward
            if mobile_indices:
                if self._ref_is_offload_param:
                    load_megatron_model_to_gpu(self.ref_mobile_module, load_grad=False)
                mobile_data = data[mobile_indices]
                mobile_data.meta_info = data.meta_info.copy()
                mobile_out, _, _, _ = self.ref_mobile_policy.compute_log_prob(data=mobile_data, calculate_entropy=False)
                if self._ref_is_offload_param:
                    offload_megatron_model_to_cpu(self.ref_mobile_module)
                aggressive_empty_cache(force_sync=True)
            else:
                mobile_out = None

            # Merge results back in original order
            response_length = data.batch["responses"].shape[-1]
            ref_log_prob_combined = torch.zeros(batch_size, response_length, dtype=torch.float32)
            if desktop_out is not None:
                for local_i, global_i in enumerate(desktop_indices):
                    ref_log_prob_combined[global_i] = desktop_out[local_i]
            if mobile_out is not None:
                for local_i, global_i in enumerate(mobile_indices):
                    ref_log_prob_combined[global_i] = mobile_out[local_i]

            output = DataProto.from_dict(tensors={"ref_log_prob": ref_log_prob_combined})

        aggressive_empty_cache(force_sync=True)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @GPUMemoryLogger(role="compute_log_prob", logger=logger)
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module, load_grad=False)
            log_gpu_memory_usage("After load actor params and grad during compute_log_prob", logger=logger)
        if (
            self.enable_routing_replay
            and self.config.actor.router_replay.mode == "R3"
            and bool(self.config.rollout.get("routed_experts_server_cache", False))
        ):
            self._hydrate_routed_experts_from_cache(data)
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature

        if self.enable_routing_replay and self.config.actor.router_replay.mode == "R2":
            RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)

        if self.enable_routing_replay and self.config.actor.router_replay.mode == "R3":
            RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

        output, entropys, layers_topk_idx, extra_out = self.actor.compute_log_prob(data=data, calculate_entropy=True)
        tensors = {"old_log_probs": output, "entropys": entropys}
        tensors.update(extra_out)
        output = DataProto.from_dict(
            tensors=tensors,
            meta_info={"temperature": self.config.rollout.temperature},
        )
        if self.config.actor.router_replay.mode == "R2":
            output.batch["routed_experts"] = layers_topk_idx

        if self.config.actor.router_replay.mode in ["R2", "R3"]:
            RouterReplay.clear_global_indices()
            RouterReplay.clear_global_router_replay_action()

        output = output.to("cpu")
        # clear kv cache
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
            log_gpu_memory_usage("After offload actor params and grad during compute_log_prob", logger=logger)
        aggressive_empty_cache(force_sync=True)
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, checkpoint_path, hdfs_path=None, del_local_after_load=True):
        # No checkpoint to load, just offload the model and optimizer to CPU
        if checkpoint_path is None:
            if self._is_offload_param:
                offload_megatron_model_to_cpu(self.actor_module)
            if self._is_offload_optimizer:
                offload_megatron_optimizer(self.actor_optimizer)
            log_gpu_memory_usage("After offload actor params and optimizer during load_checkpoint", logger=logger)
            return

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        self.checkpoint_mananager.load_checkpoint(
            local_path=checkpoint_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_pretrained_model(self, checkpoint_path, del_local_after_load=True):
        pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, checkpoint_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        if self.checkpoint_mananager.checkpoint_config.async_save and self._is_offload_optimizer:
            load_megatron_optimizer(self.actor_optimizer)
        self.checkpoint_mananager.save_checkpoint(
            local_path=checkpoint_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.actor_module)
        if self.checkpoint_mananager.checkpoint_config.async_save and self._is_offload_optimizer:
            offload_megatron_optimizer(self.actor_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def async_calls_finalize_fn_exec(self, blocking=False):
        from megatron.core.dist_checkpointing.strategies.base import \
            async_calls

        async_calls.maybe_finalize_async_calls(blocking=blocking)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_profile(self, **kwargs) -> None:
        """Start profiling for the current rank in the current training step."""
        self.profiler.start(**kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_profile(self) -> None:
        """Stop profiling for the current rank in the current training step."""
        self.profiler.stop()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def dump_memory_snapshot(self, tag: str = "manual", sub_dir: str = None) -> None:
        """Manually trigger a CUDA memory snapshot dump on all ranks."""
        # Memory snapshot is now handled by the profiler system
        # This method is kept for backward compatibility but delegates to profiler
        if hasattr(self, "profiler") and hasattr(self.profiler, "_impl"):
            try:
                # Try to use the profiler's memory snapshot functionality
                if hasattr(self.profiler._impl, "sampler"):
                    out_dir = OmegaConf.select(self.config, "actor.profiler.save_path") or "."
                    self.profiler._impl.sampler.dump_memory_snapshot(out_dir=out_dir, tag=tag, sub_dir=sub_dir)
            except Exception as e:
                # Log a warning if memory snapshot fails. This might be expected if the profiler doesn't support it.
                logger.warning(f"Failed to dump memory snapshot: {e}")


class AsyncActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def wake_up(self):
        await self.rollout_mode()
        return True

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    async def sleep(self):
        await self.trainer_mode()
        return True

    # ============================ vLLM related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD)
    def get_zeromq_address(self):
        return self.rollout.get_zeromq_address()

    # ============================ SGLang related ============================

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def chat_completion(self, json_request):
        ret = await self.rollout.chat_completion(json_request)
        return ret

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> list[int]:
        ret = await self.rollout.generate(prompt_ids, sampling_params, request_id, image_data=image_data)
        return ret


class CriticWorker(MegatronWorker, DistProfilerExtension):
    def __init__(self, config: McoreCriticConfig):
        Worker.__init__(self)

        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config)
        )
        self.config: McoreCriticConfig = config

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel strategy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        if not torch.distributed.is_initialized():
            set_numa_affinity()
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
            get_torch_device().set_device(rank)

            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.megatron.virtual_pipeline_model_parallel_size,
                use_sharp=False,
                context_parallel_size=self.config.megatron.context_parallel_size,
                expert_model_parallel_size=self.config.megatron.expert_model_parallel_size,
                expert_tensor_parallel_size=self.config.megatron.expert_tensor_parallel_size,
                nccl_communicator_config_path=None,
            )

        is_collect = (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
            and mpu.get_context_parallel_rank() == 0
        )
        self._register_dispatch_collect_info(
            mesh_name="critic", dp_rank=mpu.get_data_parallel_rank(), is_collect=is_collect
        )

        set_random_seed(seed=self.config.megatron.seed)

        # set FSDP offload params
        self._is_offload_param = self.config.megatron.param_offload
        self._is_offload_optimizer = self.config.megatron.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size *= self.config.rollout_n
        self.config.ppo_mini_batch_size //= mpu.get_data_parallel_world_size()
        if self.config.get("ppo_micro_batch_size", None):
            self.config.ppo_micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size

        # TODO(sgm): support critic model offload

    def _build_critic_model_optimizer(
        self, model_path, optim_config, override_model_config, override_transformer_config, override_ddp_config
    ):
        from verl.utils.megatron.optimizer import (
            get_megatron_optimizer, get_megatron_optimizer_param_scheduler,
            init_megatron_optim_config)
        from verl.utils.megatron_utils import (McoreModuleWrapperConfig,
                                               make_megatron_module)
        from verl.utils.model import print_model_size

        self._init_hf_config_and_tf_config(
            model_path,
            self.config.model.get("tokenizer_path") or model_path,
            self.dtype,
            override_model_config,
            override_transformer_config,
            self.config.model.get("trust_remote_code", False),
            self.config.megatron,
        )

        wrap_config = McoreModuleWrapperConfig(
            is_value_model=True,  # critic is value model
            share_embeddings_and_output_weights=False,
            wrap_with_ddp=True,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
        )
        critic_module, updated_tf_config = make_megatron_module(
            wrap_config=wrap_config,
            tf_config=self.tf_config,
            hf_config=self.hf_config,
            bridge=self.bridge,
            provider=self.provider,
            override_model_config=override_model_config,
            override_ddp_config=override_ddp_config,
            peft_cls=self.peft_cls,
            peft_config=self.config.model.get("lora", None),
        )
        self.tf_config = updated_tf_config
        # note that here critic_module will be a list to be compatible with the construction of interleaved pp (vpp).
        # but here, we do not use pp (vpp) yet. For simplicity, we remove the list
        # critic_module = nn.ModuleList(critic_module)

        if self.config.load_weight:
            t0 = time.time()
            if self.config.megatron.use_dist_checkpointing:
                load_mcore_dist_weights(
                    critic_module,
                    self.config.megatron.dist_checkpointing_path,
                    is_value_model=True,
                    prefix=self.config.megatron.dist_checkpointing_prefix,
                )
            else:
                if self.bridge is not None:
                    local_model_path = get_hf_model_path(self.config)
                    if self.vanilla_bridge:
                        self.bridge.load_weights(critic_module, local_model_path)
                    else:
                        self.bridge.load_hf_weights(
                            critic_module, local_model_path, allowed_mismatched_params=["output_layer.weight"]
                        )
                else:
                    load_megatron_gptmodel_weights(
                        self.config, self.hf_config, critic_module, params_dtype=self.dtype, is_value_model=True
                    )
            t1 = time.time()
            if torch.distributed.get_rank() == 0:
                print(f"critic load_weight time: {t1 - t0}")
        if self.rank == 0:
            print_model_size(critic_module[0])

        # TODO: add more optimizer args into config
        optim_config_megatron = init_megatron_optim_config(
            optim_config,
            use_distributed_optimizer=wrap_config.use_distributed_optimizer,
            fp16=self.dtype == torch.float16,
        )
        critic_optimizer = get_megatron_optimizer(model=critic_module, config=optim_config_megatron)
        critic_optimizer_scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=critic_optimizer, config=optim_config
        )
        get_torch_device().empty_cache()

        register_megatron_training_hooks(critic_module, critic_optimizer)

        return critic_module, critic_optimizer, critic_optimizer_scheduler, self.hf_config, optim_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # create critic

        from verl.utils.torch_dtypes import PrecisionType

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)
        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        override_transformer_config = OmegaConf.to_container(
            OmegaConf.create(self.config.megatron.get("override_transformer_config", {}))
        )
        override_ddp_config = OmegaConf.to_container(
            OmegaConf.create(self.config.megatron.get("override_ddp_config", {}))
        )
        self.param_dtype = PrecisionType.to_dtype(self.config.megatron.dtype)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        (
            self.critic_module,
            self.critic_optimizer,
            self.critic_optimizer_scheduler,
            self.critic_model_config,
            critic_optimizer_config,
        ) = self._build_critic_model_optimizer(
            model_path=self.config.model.path,
            optim_config=self.config.optim,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
            override_ddp_config=override_ddp_config,
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)

        self.critic = MegatronPPOCritic(
            config=self.config,
            model_config=self.critic_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            critic_module=self.critic_module,
            critic_optimizer=self.critic_optimizer,
            critic_optimizer_config=critic_optimizer_config,
        )
        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_mananager = MegatronCheckpointManager(
            config=self.config,
            checkpoint_config=self.config.checkpoint,
            model_config=self.critic_model_config,
            transformer_config=self.tf_config,
            role="critic",
            model=self.critic_module,
            arch=self.architectures[0],
            hf_config=self.hf_config,
            param_dtype=self.param_dtype,
            share_embeddings_and_output_weights=False,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            optimizer=self.critic_optimizer,
            optimizer_scheduler=self.critic_optimizer_scheduler,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
            use_checkpoint_opt_param_scheduler=self.config.optim.use_checkpoint_opt_param_scheduler,
            bridge=self.bridge,
            provider=self.provider,
            use_dist_checkpointing=self.config.megatron.use_dist_checkpointing,
            peft_cls=self.peft_cls,
        )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="cyan", role="compute_values")
    def compute_values(self, data: DataProto):
        micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        data = data.to(get_device_id())
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        values = self.critic.compute_values(data=data)
        output = DataProto.from_dict(tensors={"values": values})
        output = output.to("cpu")
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="critic"))
    @DistProfiler.annotate(color="pink", role="critic_update")
    def update_critic(self, data: DataProto):
        data = data.to(get_device_id())

        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_megatron_optimizer(self.critic_optimizer)

        dataloader = self.critic.make_minibatch_iterator(data)
        with Timer(name="update_critic", logger=None) as timer:
            metrics = self.critic.update_critic(dataloader=dataloader)
        delta_time = timer.last
        global_num_tokens = data.meta_info["global_token_num"]
        estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
        metrics["perf/mfu/critic"] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size
        from verl.utils.megatron.optimizer import get_megatron_last_lr

        metrics["critic/lr"] = get_megatron_last_lr(self.critic_optimizer)
        self.critic_optimizer_scheduler.step(1)

        output = DataProto(batch=None, meta_info={"metrics": metrics})

        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)
        output = output.to("cpu")
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, checkpoint_path, hdfs_path=None, del_local_after_load=True):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        self.checkpoint_mananager.load_checkpoint(
            local_path=checkpoint_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.critic_optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, checkpoint_path, hdfs_path=None, global_steps=0, max_ckpt_to_keep=None):
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.critic_module)
        self.checkpoint_mananager.save_checkpoint(
            local_path=checkpoint_path, hdfs_path=hdfs_path, global_step=global_steps, max_ckpt_to_keep=max_ckpt_to_keep
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.critic_module)


class RewardModelWorker(MegatronWorker, DistProfilerExtension):
    """
    Note that we only implement the reward model that is subclass of AutoModelForSequenceClassification.
    """

    def __init__(self, config):
        Worker.__init__(self)

        profiler_config = omega_conf_to_dataclass(config.get("profiler", {}), dataclass_type=ProfilerConfig)
        omega_profiler_config = config.get("profiler", {})
        profiler_config = omega_conf_to_dataclass(omega_profiler_config, dataclass_type=ProfilerConfig)
        if omega_profiler_config.get("tool", None) in ["npu", "nsys", "torch", "torch_memory"]:
            tool_config = omega_conf_to_dataclass(
                omega_profiler_config.get("tool_config", {}).get(omega_profiler_config.get("tool"))
            )
        else:
            tool_config = None
        DistProfilerExtension.__init__(
            self,
            DistProfiler(rank=self.rank, config=profiler_config, tool_config=tool_config),
        )
        self.config = config

        # NOTE(sgm): We utilize colocate WorkerGroup by default.
        # As a result, Workers for different model share the same process.
        # Therefore, we only require one distribute initialization.
        # To utilize different parallel strategy in different models:
        # 1, users should disable WorkerDict; 2.assign different ResourcePool to different models,
        # 3. and apply the following patch in ray==2.10, https://github.com/ray-project/ray/pull/44385
        if not torch.distributed.is_initialized():
            set_numa_affinity()
            rank = int(os.environ["LOCAL_RANK"])
            torch.distributed.init_process_group(
                backend=get_nccl_backend(),
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
            get_torch_device().set_device(rank)

            mpu.initialize_model_parallel(
                tensor_model_parallel_size=self.config.megatron.tensor_model_parallel_size,
                pipeline_model_parallel_size=self.config.megatron.pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size=self.config.megatron.virtual_pipeline_model_parallel_size,
                use_sharp=False,
                context_parallel_size=self.config.megatron.context_parallel_size,
                expert_model_parallel_size=self.config.megatron.expert_model_parallel_size,
                expert_tensor_parallel_size=self.config.megatron.expert_tensor_parallel_size,
                nccl_communicator_config_path=None,
            )

        is_collect = (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
            and mpu.get_context_parallel_rank() == 0
        )
        self._register_dispatch_collect_info(
            mesh_name="reward", dp_rank=mpu.get_data_parallel_rank(), is_collect=is_collect
        )

        set_random_seed(seed=self.config.megatron.seed)

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= mpu.get_data_parallel_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    def _build_rm_model(self, model_path, tokenizer, override_model_config, override_transformer_config):
        from verl.utils.megatron_utils import (McoreModuleWrapperConfig,
                                               make_megatron_module)

        self._init_hf_config_and_tf_config(
            model_path,
            tokenizer,
            self.dtype,
            override_model_config,
            override_transformer_config,
            self.config.model.get("trust_remote_code", False),
            self.config.megatron,
        )

        wrap_config = McoreModuleWrapperConfig(
            is_value_model=True,  # reward model is value model
            share_embeddings_and_output_weights=False,
            wrap_with_ddp=False,
            use_distributed_optimizer=self.config.megatron.use_distributed_optimizer,
        )
        reward_model, updated_tf_config = make_megatron_module(
            wrap_config=wrap_config,
            tf_config=self.tf_config,
            hf_config=self.hf_config,
            bridge=self.bridge,
            provider=self.provider,
            override_model_config=override_model_config,
        )
        self.tf_config = updated_tf_config

        if self.config.load_weight:
            if self.config.megatron.use_dist_checkpointing:
                load_mcore_dist_weights(
                    reward_model,
                    self.config.megatron.dist_checkpointing_path,
                    is_value_model=True,
                    prefix=self.config.megatron.dist_checkpointing_prefix,
                )
            else:
                if self.bridge is not None:
                    local_model_path = get_hf_model_path(self.config)
                    if self.vanilla_bridge:
                        self.bridge.load_weights(reward_model, local_model_path)
                    else:
                        self.bridge.load_hf_weights(
                            reward_model, local_model_path, allowed_mismatched_params=["output_layer.weight"]
                        )
                else:
                    load_megatron_gptmodel_weights(
                        self.config, self.hf_config, reward_model, params_dtype=self.dtype, is_value_model=True
                    )

        get_torch_device().empty_cache()
        return reward_model, self.hf_config

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # create critic

        from verl.utils.torch_dtypes import PrecisionType

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)
        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        override_transformer_config = OmegaConf.to_container(
            OmegaConf.create(self.config.megatron.get("override_transformer_config", {}))
        )

        use_shm = self.config.model.get("use_shm", False)
        sft_tokenizer_local_path = copy_to_local(self.config.model.input_tokenizer, use_shm=use_shm)
        sft_tokenizer = hf_tokenizer(sft_tokenizer_local_path)
        rm_tokenizer_path = self.config.model.get("rm_tokenizer", None)
        rm_tokenizer = None
        if rm_tokenizer_path is not None:
            rm_tokenizer_local_path = copy_to_local(rm_tokenizer_path, use_shm=use_shm)
            rm_tokenizer = hf_tokenizer(
                rm_tokenizer_local_path, trust_remote_code=self.config.model.get("trust_remote_code", False)
            )

        self.param_dtype = PrecisionType.to_dtype(self.config.megatron.dtype)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        reward_model_module, reward_model_config = self._build_rm_model(
            model_path=self.config.model.path,
            tokenizer=rm_tokenizer,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
        )
        # FIXME(sgm): reward model param offload is implemented in MegatronRewardModel
        # should be implemented in workers
        self.rm = MegatronRewardModel(
            config=self.config,
            reward_model_module=reward_model_module,
            model_config=reward_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            sft_tokenizer=sft_tokenizer,
            rm_tokenizer=rm_tokenizer,
        )

    # TODO: reward model use itself tokenizer instead of sft tokenizer
    # the input_ids, responses, attention_mask and position_ids may be different!
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward"))
    @DistProfiler.annotate(color="brown", role="compute_rm_score")
    def compute_rm_score(self, data: DataProto):
        data.meta_info["micro_batch_size"] = self.config.micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.forward_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.use_dynamic_bsz
        data = data.to(get_device_id())
        output = self.rm.compute_reward(data)
        output = output.to("cpu")
        return output
