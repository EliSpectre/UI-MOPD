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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import shutil
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (_compute_response_info,
                                           compute_data_metrics,
                                           compute_throughout_metrics,
                                           compute_timing_metrics,
                                           reduce_metrics)
from verl.trainer.ppo.ray_trainer import (AdvantageEstimator, RayPPOTrainer,
                                          apply_kl_penalty_advantage,
                                          compute_advantage,
                                          compute_response_mask)
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip
from ui_mopd.metric.utils import process_validation_metrics_global
from ui_mopd.reward.gui_agent_thought_verify import \
    check_thought_action_consistency


def _retry_file_open(filepath, mode, max_retries=5, retry_delay=1.0, backoff_factor=2.0, **kwargs):
    """
    带重试机制的文件打开，防止分布式文件系统一次打开失败导致程序崩溃。
    支持 with 语句使用: with _retry_file_open(path, "w") as f: ...
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            return open(filepath, mode, **kwargs)
        except (IOError, OSError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = retry_delay * (backoff_factor ** attempt)
                print(f"[retry] 文件打开失败 (尝试 {attempt + 1}/{max_retries}): {filepath}, 错误: {e}, {wait_time:.1f}s 后重试")
                time.sleep(wait_time)
            else:
                raise last_exception


class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理numpy类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def mask_invalid_response_tokens(processor, data):
    response_ids = data.batch["responses"]
    # mask of exactly the invalid-token positions
    invalid_token_mask = (response_ids == processor.image_token_id)

    # build a mask that's True at and after the first invalid token, per sequence
    # invalid_token_mask.cumsum(dim=1) will, for each position, count how many invalids have appeared up to there.
    # >0 means "this position is invalid or comes after an invalid"
    invalid_or_after = invalid_token_mask.cumsum(dim=1).clamp(max=1).bool()

    # log how many tokens will get replaced
    #num_to_replace = invalid_or_after.sum().item()
    #print(f"{num_to_replace} tokens (invalid or after) will be set to EOS.")

    # set all those tokens to eos_token_id in one go
    response_ids[invalid_or_after] = processor.tokenizer.eos_token_id
    

class RayDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """
    @property
    def _checkpoint_dir(self):
        return os.path.join(self.config.trainer.default_local_dir, "checkpoint")

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe
        from verl.trainer.ppo.ray_trainer import Role

        local_global_step_folder = os.path.join(self._checkpoint_dir, f"global_step_{self.global_steps}")
        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(self._checkpoint_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        from verl.trainer.ppo.ray_trainer import Role
        from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path

        if self.config.trainer.resume_mode == "disable":
            return 0

        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")

        checkpoint_folder = self._checkpoint_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str)
                assert "global_step_" in self.config.trainer.resume_from_path
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    global_step_folder = os.path.join(os.getcwd(), global_step_folder)

        print(f"Load from checkpoint folder: {global_step_folder}")
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))

        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
            print(f"Loaded dataloader state from {dataloader_local_path}")

    def record_gen_batch(self, gen_batch, cur_step):
        batch_data_dir = os.path.join(self.config.trainer.default_local_dir, "batch_data")
        os.makedirs(batch_data_dir, exist_ok=True)
        gen_batch_file = os.path.join(batch_data_dir, f"gen_batch_{cur_step}.jsonl")
        with _retry_file_open(gen_batch_file, "a") as f:
            for i in range(len(gen_batch)):
                sample_non_tensor = {
                    key: val[i] for key, val in gen_batch.non_tensor_batch.items()
                    if key not in ["multi_modal_inputs"]
                }
                f.write(json.dumps(sample_non_tensor, ensure_ascii=False, cls=NumpyEncoder) + "\n")

    def record_batch(self, batch, cur_step):
        batch_data_dir = os.path.join(self.config.trainer.default_local_dir, "batch_data")
        os.makedirs(batch_data_dir, exist_ok=True)
        batch_file = os.path.join(batch_data_dir, f"batch_{cur_step}.jsonl")
        with _retry_file_open(batch_file, "a") as f:
            for i in range(len(batch)):
                sample_non_tensor = {
                    key: val[i] for key, val in batch.non_tensor_batch.items()
                    if key not in ["multi_modal_inputs"]
                }
                f.write(json.dumps(sample_non_tensor, ensure_ascii=False, cls=NumpyEncoder) + "\n")

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "uid", "index"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch
    
    def _data_generator(self):
        while not self.train_dataloader.is_end():
            self.train_dataloader.add_step()
            yield None
            
    def fit(self):
        # 训练主循环
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking
        # tensorboard/wandb 记录
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0
        self.best_acc = 0.0
        best_acc_file = os.path.join(self.config.trainer.default_local_dir, f"best_acc.json")
        if os.path.exists(best_acc_file):  # 存储之前运行最好的acc
            with _retry_file_open(best_acc_file, "r", encoding="utf-8") as f:
                try:
                    self.best_acc = json.load(f)["best_acc"]
                except Exception as e:
                    self.best_acc = 0.0
        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

            best_acc_key = None
            for key in ["val-core/guiagent/os4评测集/acc/mean", "val-core/ui_mopd/guiagent/acc/mean", "val-core/ui_mopd/android_world/acc/mean"]:
                if key in val_metrics:
                    best_acc_key = key
                    break
            if best_acc_key and val_metrics[best_acc_key] > self.best_acc:
                self.best_acc = val_metrics[best_acc_key]
                print(f"val-core best_acc_key: {best_acc_key}, new best acc: {self.best_acc}")
                with _retry_file_open(best_acc_file, "w") as f:
                    tmp = {k: v.item() if hasattr(v, 'item') else v for k, v in val_metrics.items()}
                    f.write(json.dumps(tmp, ensure_ascii=False, indent=4))



        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        for _ in self._data_generator():
            # ① 采样数据
            batch_dict = self.train_dataloader.sample(self.config.data.gen_batch_size)
            metrics = {}

            with marked_timer("start_profile", timing_raw):
                self._start_profiling(
                    not prev_step_profile and curr_step_profile
                    if self.config.global_profiler.profile_continuous_steps
                    else curr_step_profile
                )

            new_batch: DataProto = DataProto.from_single_dict(batch_dict)
            new_batch.non_tensor_batch["uid"] = np.array(
                    [new_batch[i].non_tensor_batch["extra_info"]["id"] for i in range(len(new_batch))], dtype=object
                )

            num_gen_batches += 1
            # ② 准备 rollout 输入
            # 每个 prompt 复制 N 份（如 16），准备采样多条 response
            gen_batch = self._get_gen_batch(new_batch)
            gen_batch = gen_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
            )

            is_last_step = self.global_steps >= self.total_training_steps

            with marked_timer("step", timing_raw):
                # generate a batch
                with marked_timer("gen", timing_raw, "red"):
                    if not self.async_rollout_mode:
                        # 同步框架
                        # ③ 生成 response
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    else:
                        # 异步框架
                        # ③ 生成 response
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        print(f"gen_batch_output: {gen_batch_output.batch.keys()}")
                        if "responses" in gen_batch_output.batch:
                            responses_tensor = gen_batch_output.batch["responses"]
                            if (responses_tensor == 248056).any():
                                print("\n" + "!" * 30)
                                print("[FATAL EVIDENCE] rollout 阶段吐出了 248056！")
                                bad_batch_indices = (responses_tensor == 248056).nonzero(as_tuple=True)[0].unique()
                                for b_idx in bad_batch_indices:
                                    bad_seq = responses_tensor[b_idx].tolist()
                                    print(f"--> Batch Index {b_idx} 生成的原始 ID 序列: {bad_seq}")
                                print("\n" + "!" * 30)
                                
                    timing_raw.update(gen_batch_output.meta_info["timing"])
                    gen_batch_output.meta_info.pop("timing", None)

                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    with marked_timer("gen_max", timing_raw, "red"):
                        gen_baseline_batch = deepcopy(gen_batch)
                        gen_baseline_batch.meta_info["do_sample"] = False
                        gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                        new_batch = new_batch.union(gen_baseline_output)
                        reward_baseline_tensor = self.reward_fn(new_batch)
                        reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                        new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                        new_batch.batch["reward_baselines"] = reward_baseline_tensor

                        del gen_baseline_batch, gen_baseline_output

                # repeat to align with repeated responses in rollout
                # ④ 合并 prompt + response 数据，准备计算奖励
                new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                new_batch = new_batch.union(gen_batch_output)
                # 清理非法 token（如 image token）及其之后的 token，避免它们对奖励计算和模型更新造成干扰
                mask_invalid_response_tokens(self.processor, new_batch)

                with marked_timer("reward", timing_raw, "yellow"):
                    # compute scores. Support both model and function-based.
                    # We first compute the scores using reward model. Then, we call reward_fn to combine
                    # the results from reward model and rule-based results.
                    if self.use_rm:
                        # we first compute reward model score
                        reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                        new_batch = new_batch.union(reward_tensor)

                    # we combine with rule-based rm
                    reward_extra_infos_dict: dict[str, list]
                    try:
                        reward_result = self.reward_fn(new_batch, return_dict=True)
                        reward_tensor = reward_result["reward_tensor"]
                        reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                    except Exception as e:
                        print(f"Error in reward_fn: {e}")
                        reward_tensor = self.reward_fn(new_batch)
                        reward_extra_infos_dict = {}
                        
                    new_batch.batch["token_level_scores"] = reward_tensor

                    if reward_extra_infos_dict:
                        new_batch.non_tensor_batch.update(
                            {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                        )

                if self.config.data.use_dynamic_history:
                    self.train_dataloader.dataset.thought_buffer.add_batch(new_batch)

                if self.config.trainer.record_data:
                    self.record_gen_batch(new_batch, self.global_steps)

                if not self.config.algorithm.filter_groups.enable:
                    batch = new_batch
                else:  # NOTE: When prompts after filtering is less than train batch size,
                    # we skip to the next generation batch
                    metric_name = self.config.algorithm.filter_groups.metric
                    if metric_name == "seq_final_reward":
                        # Turn to numpy for easier filtering
                        new_batch.non_tensor_batch["seq_final_reward"] = (
                            new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                        )
                    elif metric_name == "seq_reward":
                        new_batch.non_tensor_batch["seq_reward"] = (
                            new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                        )
                        
                    # ⑥ DAPO Filter Groups (第 327-406 行) — DAPO 的核心创新
                    # Collect the sequence reward for each trajectory
                    prompt_uid2metric_vals = defaultdict(list)
                    # 按 uid 分组，计算每个 prompt 的平均 acc
                    for uid, metric_val in zip(
                        new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                    ):
                        prompt_uid2metric_vals[uid].append(metric_val)

                    prompt_uid2metric_mean = {}
                    for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                        prompt_uid2metric_mean[prompt_uid] = np.mean(metric_vals)
                    prompt_uid2index = {}
                    for prompt_uid, index in zip(new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch["index"]):
                        prompt_uid2index[prompt_uid] = index
                    
                    passrate_list = []
                    for prompt_uid, index in prompt_uid2index.items():
                        passrate_list.append((index, prompt_uid2metric_mean[prompt_uid]))
                    self.train_dataloader.feedback(passrate_list)

                    # 打印每个 prompt 的 rollout 平均 reward
                    sorted_uids = sorted(prompt_uid2metric_mean.items(), key=lambda x: x[1], reverse=True)
                    print(f"\n[Step {self.global_steps}] Rollout reward per prompt (n={len(sorted_uids)}):")
                    for uid, mean_score in sorted_uids:
                        print(f"  {mean_score:>6.3f} | {uid}")
                    low_score_threshold = self.config.algorithm.filter_groups.low_score_threshold
                    high_score_threshold = self.config.algorithm.filter_groups.high_score_threshold
                    # 过滤全对/全错的 prompt
                    kept_prompt_uids = [
                        uid
                        for uid, score in prompt_uid2metric_mean.items()
                        if low_score_threshold < score < high_score_threshold or len(prompt_uid2metric_vals[uid]) == 1
                    ]
                    
                    num_prompt_in_batch += len(kept_prompt_uids)

                    kept_traj_idxs = []
                    kl_threshold = self.config.algorithm.kl_ctrl.kl_threshold
                    kl_loss_mask = torch.ones_like(new_batch.batch["response_mask"], dtype=torch.int, device=new_batch.batch["response_mask"].device)
                    passrate_0_mask = torch.ones_like(new_batch.batch["response_mask"], dtype=torch.int, device=new_batch.batch["response_mask"].device)
                    for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                        if traj_from_prompt_uid in kept_prompt_uids:
                            kept_traj_idxs.append(idx)
                            kl_loss_mask[idx] = 0 if prompt_uid2metric_mean[traj_from_prompt_uid] > kl_threshold else 1
                            passrate_0_mask[idx] = 0 if prompt_uid2metric_mean[traj_from_prompt_uid] < 0.01 else 1
                    
                    new_batch.batch["kl_loss_mask"] = kl_loss_mask
                    new_batch.batch["passrate_0_mask"] = passrate_0_mask
                    
                    new_batch = new_batch[kept_traj_idxs]
                    batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                    prompt_bsz = self.config.data.train_batch_size
                    # 累积直到凑够 train_batch_size 个有效 prompt
                    if num_prompt_in_batch < prompt_bsz:
                        print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                        max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                        if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                            print(f"{num_gen_batches=}. Keep generating...")
                            self.gen_steps += 1
                            is_last_step = self.global_steps >= self.total_training_steps
                            continue
                        else:
                            raise ValueError(
                                f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                + " Generated too many. Please check if your data are too difficult."
                                + " You could also try set max_num_gen_batches=0 to enable endless trials."
                            )
                    else:
                        # Align the batch
                        traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                        batch = batch[:traj_bsz]
                    
                    del new_batch, gen_batch, gen_batch_output
                
                metrics.update({"need_teach_sample_ratio": batch.batch["kl_loss_mask"][:, 0].to(torch.float).mean().item()})
                # === Updating ===
                if self.config.trainer.record_data:
                    self.record_batch(batch, self.global_steps)

                if self.config.reward_model.thought_knowledge_match.enable:
                    query_stats = defaultdict(lambda: {"total_ids": set(), "match_ids": set()})
                    rollout_stats = defaultdict(lambda: {"total_rollout": 0, "match_rollout": 0})
                    for batch_item in batch:
                        extra_info = batch_item.non_tensor_batch.get("extra_info", {})
                        extra_info = json.loads(extra_info)
                        example_id = extra_info.get("id")
                        cot_category = extra_info.get("cot_category")
                        match_result = extra_info.get("cot_knowledge_match_result")
                        if cot_category and match_result is not None:
                            query_stats[cot_category]["total_ids"].add(example_id)
                            rollout_stats[cot_category]["total_rollout"] += 1
                            if match_result:
                                query_stats[cot_category]["match_ids"].add(example_id)
                                rollout_stats[cot_category]["match_rollout"] += 1
                    
                    for cot_category, stats in query_stats.items():
                        query_num = len(stats["total_ids"])
                        match_query_num = len(stats["match_ids"])
                        match_query_ratio = match_query_num / query_num if query_num > 0 else 0

                        rollout_num =  rollout_stats[cot_category]["total_rollout"]
                        match_rollout_num =  rollout_stats[cot_category]["match_rollout"]
                        match_rollout_ratio = match_rollout_num / rollout_num if rollout_num > 0 else 0

                        print((
                            f"cot_category: {cot_category}, query_num: {query_num}, match: {match_query_num}, ratio:{match_query_ratio:.2f}, "
                            f"rollout_num: {rollout_num}, match: {match_rollout_num}, ratio:{match_rollout_ratio:.2f}"
                        ))

                
                # 判断thought-action-func一致性
                if (self.config.reward_model.thought_match.enable 
                    and self.config.reward_model.thought_match.base_url 
                    and self.global_steps >= self.config.reward_model.thought_match.start_step):
                    print(f"判断thought-action-func一致性, base_url: {self.config.reward_model.thought_match.base_url}")
                    batch = check_thought_action_consistency(batch, self.config.reward_model.thought_match.base_url)
                    
                batch.batch["response_mask"] = compute_response_mask(batch)

                # Balance the number of valid tokens across DP ranks.
                # NOTE: This usually changes the order of data in the `batch`,
                # which won't affect the advantage calculation (since it's based on uid),
                # but might affect the loss calculation (due to the change of mini-batching).
                # TODO: Decouple the DP balancing and mini-batching.
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)

                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                batch.meta_info["use_topk_distillation"] = self.config.algorithm.get("use_topk_distillation", False)
                batch.meta_info["topk_size"] = self.config.algorithm.get("topk_size", 200)
                batch.meta_info["alpha"] = self.config.algorithm.get("alpha", 0.0)
                batch.meta_info["topk_kl_loss_weight"] = self.config.algorithm.get("topk_kl_loss_weight", 1.0)

                try:
                    # ⑦ 重新计算 log_prob 和 entropy，支持基于 log_prob 的奖励（如 KL 惩罚）和蒸馏损失
                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        batch.meta_info["topk_distillation_phase"] = "student"
                        old_log_prob, _ = self._compute_old_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)
                        self._drop_routed_experts_from_batch(batch)

                    # ⑧ 计算 Reference log_prob
                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            batch.meta_info["topk_distillation_phase"] = "teacher"

                            use_opsd = self.config.data.get("use_opsd", False)
                            opsd_active = use_opsd and "teacher_input_ids" in batch.batch

                            if opsd_active:
                                responses = batch.batch["responses"]
                                resp_len = responses.shape[1]
                                t_prompt_ids = batch.batch["teacher_input_ids"]
                                t_prompt_mask = batch.batch["teacher_attention_mask"]
                                t_prompt_pos = batch.batch["teacher_position_ids"]

                                teacher_ids = torch.cat([t_prompt_ids, responses], dim=-1)
                                teacher_mask = torch.cat([t_prompt_mask, batch.batch["response_mask"]], dim=-1)

                                last_pos = t_prompt_pos[..., -1:]
                                resp_offsets = torch.arange(1, resp_len + 1, device=last_pos.device, dtype=last_pos.dtype)
                                teacher_pos = torch.cat([t_prompt_pos, last_pos + resp_offsets], dim=-1)

                                orig_ids = batch.batch["input_ids"]
                                orig_mask = batch.batch["attention_mask"]
                                orig_pos = batch.batch["position_ids"]
                                batch.batch["input_ids"] = teacher_ids
                                batch.batch["attention_mask"] = teacher_mask
                                batch.batch["position_ids"] = teacher_pos

                            ref_log_prob = self._compute_ref_log_prob(batch)

                            if opsd_active:
                                batch.batch["input_ids"] = orig_ids
                                batch.batch["attention_mask"] = orig_mask
                                batch.batch["position_ids"] = orig_pos

                            batch = batch.union(ref_log_prob)

                    # === OPD KL Debug: 打印每条 rollout 的 reward 和 KL 散度 ===
                    if self.use_reference_policy and self.config.actor_rollout_ref.actor.use_kl_loss:
                        from verl.trainer.ppo.core_algos import kl_penalty as _kl_penalty_fn
                        with torch.no_grad():
                            _kld = _kl_penalty_fn(
                                batch.batch["old_log_probs"],
                                batch.batch["ref_log_prob"],
                                kl_penalty=self.config.actor_rollout_ref.actor.kl_loss_type,
                            )
                            _kld = _kld * batch.batch["response_mask"]
                            _mean_kl_per_sample = (_kld.sum(dim=-1) / batch.batch["response_mask"].sum(dim=-1).clamp(min=1))

                            _scores = batch.non_tensor_batch.get("score", [None] * len(_mean_kl_per_sample))
                            _uids = batch.non_tensor_batch.get("uid", ["?"] * len(_mean_kl_per_sample))
                            _kl_coef = self.config.actor_rollout_ref.actor.kl_loss_coef

                            print(f"\n[Step {self.global_steps}] OPD KL Debug (type={self.config.actor_rollout_ref.actor.kl_loss_type}, coef={_kl_coef}):")
                            print(f"  {'uid':<40} | {'reward':>8} | {'mean_kl':>10} | {'kl_loss_contrib':>15}")
                            print(f"  {'-'*40}-+-{'-'*8}-+-{'-'*10}-+-{'-'*15}")
                            for _i in range(min(len(_mean_kl_per_sample), 20)):
                                _uid = str(_uids[_i])[:40]
                                _score = _scores[_i] if _scores[_i] is not None else "N/A"
                                _kl_val = _mean_kl_per_sample[_i].item()
                                _kl_contrib = _kl_val * _kl_coef
                                print(f"  {_uid:<40} | {str(_score):>8} | {_kl_val:>10.4f} | {_kl_contrib:>15.4f}")

                            _global_mean_kl = _mean_kl_per_sample.mean().item()
                            print(f"  Global mean KL: {_global_mean_kl:.4f}, KL loss contribution: {_global_mean_kl * _kl_coef:.4f}")
                            metrics["opd/mean_kl"] = _global_mean_kl
                            metrics["opd/kl_loss_coef"] = _kl_coef

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # Compute rollout correction weights and off-policy metrics (inherited from RayPPOTrainer)
                    from verl.trainer.ppo.rollout_corr_helper import \
                        compute_rollout_correction_and_add_to_batch

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                        batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                        # IS and off-policy metrics already have rollout_corr/ prefix
                        metrics.update(is_metrics)

                    with marked_timer("adv", timing_raw, "brown"):
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                        # compute advantages, executed on the driver process
                        # ⑨ 计算优势估计（如 GAE），支持基于 token 的奖励和优势归一化（如 GRPO 中按 std 归一化）
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )
                    
                    # compute rewards. apply_kl_penalty if available
                    if self.config.algorithm.use_kl_in_advantage:
                        batch, kl_metrics = apply_kl_penalty_advantage(
                            batch,
                            kl_ctrl=self.kl_ctrl_in_reward,
                            kl_penalty=self.config.algorithm.kl_penalty,
                            kl_skip_thought=self.config.algorithm.get("kl_skip_thought", False),
                        )
                        metrics.update(kl_metrics)
                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    # 更新 Actor 之前先判断是否满足 critic_warmup 的条件，如果不满足则跳过 Actor 更新，直接进入下一轮训练循环
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # === Pre-update per-trajectory breakdown ===
                        if self.use_reference_policy and (self.config.actor_rollout_ref.actor.use_kl_loss or self.config.algorithm.use_kl_in_advantage):
                            with torch.no_grad():
                                from verl.trainer.ppo.core_algos import kl_penalty as _kl_fn
                                _old_lp = batch.batch["old_log_probs"]
                                _ref_lp = batch.batch["ref_log_prob"]
                                _resp_mask = batch.batch["response_mask"]
                                _adv = batch.batch["advantages"]
                                _kl_coef = self.config.actor_rollout_ref.actor.kl_loss_coef if self.config.actor_rollout_ref.actor.use_kl_loss else self.config.algorithm.kl_ctrl.kl_coef

                                _kld = _kl_fn(_old_lp, _ref_lp, kl_penalty=self.config.actor_rollout_ref.actor.kl_loss_type)
                                _kld = _kld * _resp_mask
                                _mean_kl = (_kld.sum(dim=-1) / _resp_mask.sum(dim=-1).clamp(min=1))
                                _mean_adv = (_adv * _resp_mask).sum(dim=-1) / _resp_mask.sum(dim=-1).clamp(min=1)

                                _uids = batch.non_tensor_batch.get("uid", ["?"] * len(_mean_kl))
                                _scores = batch.non_tensor_batch.get("score", [None] * len(_mean_kl))
                                _resp_lens = _resp_mask.sum(dim=-1).int()
                                _max_resp_len = int(self.config.data.max_response_length)
                                _num_truncated = int((_resp_lens >= _max_resp_len).sum().item())

                                print(f"\n[Step {self.global_steps}] Pre-update Trajectory Info ({len(_mean_kl)} trajectories, {_num_truncated} truncated at max_len={_max_resp_len}):")
                                print(f"  {'uid':<40} | {'reward':>8} | {'resp_len':>8} | {'advantage':>10} | {'mean_kl':>8} | {'kl×coef':>8}")
                                print(f"  {'-'*40}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
                                for _i in range(len(_mean_kl)):
                                    _uid = str(_uids[_i])[:40]
                                    _sc = f"{_scores[_i]:.1f}" if _scores[_i] is not None else "N/A"
                                    _rlen = _resp_lens[_i].item()
                                    _trunc_mark = "*" if _rlen >= _max_resp_len else " "
                                    print(f"  {_uid:<40} | {_sc:>8} | {_rlen:>7}{_trunc_mark} | {_mean_adv[_i].item():>10.4f} | {_mean_kl[_i].item():>8.4f} | {_mean_kl[_i].item() * _kl_coef:>8.4f}")
                                print(f"  --- Summary: mean_reward={np.mean([s for s in _scores if s is not None]):.4f}, mean_resp_len={_resp_lens.float().mean().item():.1f}, truncated={_num_truncated}/{len(_mean_kl)}, mean_adv={_mean_adv.mean().item():.4f}, mean_kl={_mean_kl.mean().item():.4f}, mean_kl×coef={_mean_kl.mean().item() * _kl_coef:.4f}")

                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            batch.meta_info["topk_distillation_phase"] = "update"
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                        # === Post-update global loss summary ===
                        if self.use_reference_policy and self.config.actor_rollout_ref.actor.use_kl_loss:
                            _pg = actor_output_metrics.get("actor/pg_loss", None)
                            _kl = actor_output_metrics.get("actor/kl_loss", None)
                            _kl_coef = self.config.actor_rollout_ref.actor.kl_loss_coef
                            if _pg is not None and _kl is not None:
                                print(f"[Step {self.global_steps}] Actor Loss: pg_loss={_pg:.6f}, kl_loss={_kl:.6f}, kl_coef={_kl_coef}, total_loss={_pg + _kl * _kl_coef:.6f}")
                            else:
                                print(f"[Step {self.global_steps}] Actor Loss: (metrics not available)")

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)
                finally:
                    self._clear_routed_experts_caches(batch)

            val_metrics = {}
            # validate
            # ⑪ 验证 & 存 checkpoint 
            if (
                self.val_reward_fn is not None
                and self.config.trainer.test_freq > 0
                and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
            ):
                with marked_timer("testing", timing_raw, "green"):
                    val_metrics: dict = self._validate()
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)

            if self.config.trainer.save_freq > 0 and (
                is_last_step or self.global_steps % self.config.trainer.save_freq == 0
            ):
                with marked_timer("save_checkpoint", timing_raw, "green"):
                    self._save_checkpoint()
                    best_acc_key = None
                    for key in ["val-core/guiagent/os4评测集/acc/mean", "val-core/ui_mopd/guiagent/acc/mean"]:
                        if key in val_metrics:
                            best_acc_key = key
                            break
                    if best_acc_key and val_metrics[best_acc_key] > self.best_acc:
                        self.best_acc = val_metrics[best_acc_key]
                        print(f"val-core best_acc_key: {best_acc_key}, new best acc: {self.best_acc}")
                        with _retry_file_open(best_acc_file, "w") as f:
                            tmp = {k: v.item() if hasattr(v, 'item') else v for k, v in val_metrics.items()}
                            f.write(json.dumps(tmp, ensure_ascii=False, indent=4))
                    # 将hugging_face checkpoint存储到final
                    checkpoint_base = os.path.join(self.config.trainer.default_local_dir, "checkpoint")
                    hf_checkpoint_path = os.path.join(checkpoint_base, f"global_step_{self.global_steps}/actor/huggingface")
                    target_path = os.path.join(self.config.trainer.default_local_dir, "final")
                    if os.path.exists(target_path):
                        shutil.rmtree(target_path)
                    shutil.copytree(hf_checkpoint_path, target_path)

            with marked_timer("stop_profile", timing_raw):
                next_step_profile = (
                    self.global_steps + 1 in self.config.global_profiler.steps
                    if self.config.global_profiler.steps is not None
                    else False
                )
                self._stop_profiling(
                    curr_step_profile and not next_step_profile
                    if self.config.global_profiler.profile_continuous_steps
                    else curr_step_profile
                )
                prev_step_profile = curr_step_profile
                curr_step_profile = next_step_profile

            # collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            # TODO: implement actual tflpo and theoretical tflpo
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
            timing_raw = defaultdict(float)  # clear timing
            
            del batch
            metrics["train/num_gen_batches"] = num_gen_batches
            batch = None
            num_prompt_in_batch = 0
            num_gen_batches = 0

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            if is_last_step:
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return

            progress_bar.update(1)
            self.global_steps += 1
            self.gen_steps += 1
            
        
        # check if last step checkpint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, "checkpoint", f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        all_prompt_length = []
        all_response_length = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            results = _compute_response_info(test_output_gen_batch)
            all_prompt_length.extend(results['prompt_length'].tolist())
            all_response_length.extend(results['response_length'].tolist())

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            
            # dump generations
            val_data_dir = self.config.trainer.get("validation_data_dir", None)
            if val_data_dir and self.config.trainer.record_data:
                self._dump_generations(
                    inputs=input_texts,
                    outputs=output_texts,
                    gts=ground_truths,
                    scores=scores,
                    reward_extra_infos_dict={},
                    dump_path=val_data_dir,
                )

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    if key not in ["thought_length", "acc", "score"]:
                        continue
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
            del test_batch, test_gen_batch, test_output_gen_batch, test_gen_batch_padded, test_output_gen_batch_padded, result, reward_tensor, scores

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics_global(data_sources, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            for var_name, metric2val in var2metric2val.items():
                for metric_name, metric_val in metric2val.items():
                    metric_sec = "val-core"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        metric_dict["val-aux/thought_length_max"] = np.max(reward_extra_infos_dict["thought_length"])
        metric_dict["val-aux/thought_length_mean"] = np.mean(reward_extra_infos_dict["thought_length"])
        metric_dict["val-aux/thought_length_min"] = np.min(reward_extra_infos_dict["thought_length"])
        metric_dict["val-aux/thought_length_min"] = np.min(reward_extra_infos_dict["thought_length"])
        metric_dict["val-aux/prompt_length_max"] = np.max(all_prompt_length)
        metric_dict["val-aux/prompt_length_mean"] = np.mean(all_prompt_length)
        metric_dict["val-aux/prompt_length_min"] = np.min(all_prompt_length)
        metric_dict["val-aux/response_length_max"] = np.max(all_response_length)
        metric_dict["val-aux/response_length_mean"] = np.mean(all_response_length)
        metric_dict["val-aux/response_length_min"] = np.min(all_response_length)
        return metric_dict
    
