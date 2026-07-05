#!/usr/bin/env bash
# Stage 2: MOPD — Multi-teacher On-Policy Distillation (Qwen3-VL-8B-Thinking student)
# Usage: bash ui_mopd/runs/run_mopd.sh [extra hydra overrides...]
set -euo pipefail

# ============================================================
# Environment
# ============================================================
export VLLM_USE_V1=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_BACKEND_LOG_LEVEL=error
export RAY_SCHEDULER_EVENTS=0

# ============================================================
# Experiment Config
# ============================================================
PROJECT=${PROJECT:-"MOPD"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"stage2-qwen3-8b-mopd-dual-teacher"}

# --- Paths ---
DEFAULT_TRAIN_FILE="/path/to/mix_mopd_train.parquet"
DEFAULT_VAL_FILE="/path/to/mix_mopd_test.parquet"
DEFAULT_MODEL_PATH="/path/to/Qwen3-VL-8B-Thinking"

TRAIN_FILES=${TRAIN_FILES:-${DEFAULT_TRAIN_FILE}}
VAL_FILES=${VAL_FILES:-${DEFAULT_VAL_FILE}}
MODEL_PATH=${MODEL_PATH:-${DEFAULT_MODEL_PATH}}

# --- Teacher checkpoints (OPD reference models) ---
REF_MODEL_PATH=${REF_MODEL_PATH:-"/path/to/stage1-desktop-teacher/huggingface"}
REF_MODEL_PATH_MOBILE=${REF_MODEL_PATH_MOBILE:-"/path/to/stage1-mobile-teacher/huggingface"}

# --- Output ---
OUTPUT_DIR="${LOCAL_DIR:-/path/to/checkpoints/${PROJECT}/${EXPERIMENT_NAME}}"
mkdir -p "${OUTPUT_DIR}"
echo ">> Checkpoint Directory: ${OUTPUT_DIR}"
export TENSORBOARD_DIR="${OUTPUT_DIR}/tensorboard"

# ============================================================
# Cluster & Parallelism
# ============================================================
NODES=${NODES:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# Student 8B: TP=2, DP=64/2=32
TP_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-2}
PP_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-1}
EP_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
ETP_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}

# Teacher 32B reference: TP=8, DP=64/8=8
REF_TP_SIZE=${REF_TENSOR_MODEL_PARALLEL_SIZE:-8}
REF_PP_SIZE=${REF_PIPELINE_MODEL_PARALLEL_SIZE:-1}
REF_EP_SIZE=${REF_EXPERT_MODEL_PARALLEL_SIZE:-1}
REF_ETP_SIZE=${REF_EXPERT_TENSOR_PARALLEL_SIZE:-1}

# Rollout
ROLLOUT_TP_SIZE=${ROLLOUT_TENSOR_PARALLEL_SIZE:-2}

# ============================================================
# Batch & Training
# ============================================================
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-384}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-128}
PPO_MICRO_BS=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
LOG_PROB_MICRO_BS=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}
TOTAL_STEPS=${TOTAL_TRAINING_STEPS:-3200}
USE_DYNAMIC_BSZ=${DYNAMIC_BSZ:-False}

# Checkpoint
SAVE_FREQ=${SAVE_FREQ:-16}
EVAL_FREQ=${EVAL_FREQ:-16}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-6}

# Eval
val_only=${VAL_ONLY:-False}
val_do_sample=${VAL_DO_SAMPLE:-False}
val_rollout_n=${VAL_ROLLOUT_N:-1}
val_temperature=${VAL_TEMPERATURE:-0.0}

# ============================================================
# OPD: On-Policy Distillation KL Auxiliary Loss
# ============================================================
USE_KL_LOSS=${USE_KL_LOSS:-True}
KL_LOSS_TYPE=${KL_LOSS_TYPE:-k3}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.01}

# ============================================================
# Rollout Engine
# ============================================================
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.60}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1024}
ROLLOUT_N=${ROLLOUT_N:-8}

# ============================================================
# Hydra Parameter Arrays
# ============================================================

# --- [Data] ---
data_params=(
    data.train_files="${TRAIN_FILES}"
    data.val_files="${VAL_FILES}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.gen_batch_size=${GEN_BATCH_SIZE}
    data.val_batch_size=${VAL_BATCH_SIZE:-128}
    data.train_max_samples=${TRAIN_MAX_SAMPLES:--1}
    data.val_max_samples=${VAL_MAX_SAMPLES:--1}
    data.max_prompt_length=${MAX_PROMPT_LENGTH:-8192}
    data.max_response_length=${MAX_RESPONSE_LENGTH:-512}
    data.add_reasoning_content=${ADD_REASONING_CONTENT:-False}
    data.return_raw_chat=True
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=128
    data.truncation=error
    data.image_key=images
    data.use_dynamic_history=${USE_DYNAMIC_HISTORY:-False}
    data.thought_buffer_size=${THOUGHT_BUFFER_SIZE:-5}
)

# --- [Model] ---
model_params=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.ref_path="${REF_MODEL_PATH}"
    actor_rollout_ref.model.ref_mobile_path="${REF_MODEL_PATH_MOBILE}"
    actor_rollout_ref.model.use_fused_kernels=${USE_FUSED_KERNELS:-True}
)

# --- [Trainer] ---
trainer_params=(
    trainer.logger="['console', 'tensorboard']"
    trainer.project_name="${PROJECT}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.default_local_dir="${OUTPUT_DIR}"
    +trainer.validation_data_dir="${OUTPUT_DIR}/Validation"
    trainer.n_gpus_per_node=${GPUS_PER_NODE}
    trainer.nnodes=${NODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${EVAL_FREQ}
    trainer.total_training_steps=${TOTAL_STEPS}
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}
    trainer.resume_mode=auto
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-False}
    trainer.record_data=${RECORD_DATA:-True}
    +trainer.val_only=${val_only}
)

# --- [PPO Actor] ---
ppo_actor_params=(
    actor_rollout_ref.actor.ema_decay=${EMA_DECAY:-null}
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BS}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
    actor_rollout_ref.actor.optim.lr=${OPTIMIZER_LR:-1e-6}
    actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS:--1}
    actor_rollout_ref.actor.optim.lr_decay_style=${LR_DECAY_STYLE:-constant}
    actor_rollout_ref.actor.optim.total_training_steps=${TOTAL_STEPS}
    actor_rollout_ref.actor.optim.weight_decay=0
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${OPTIMIZER_OFFLOAD_FRACTION:-1.0}
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.log_num_zeros_in_grad=True
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}
    actor_rollout_ref.actor.kl_loss_type=${KL_LOSS_TYPE}
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.policy_loss.loss_mode=${LOSS_MODE:-"vanilla"}
    actor_rollout_ref.actor.router_replay.mode=${ROUTER_REPLAY_MODE:-disabled}
    actor_rollout_ref.actor.megatron.dtype=${DTYPE:-"bfloat16"}
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.param_offload=${PARAM_OFFLOAD:-True}
    actor_rollout_ref.actor.megatron.optimizer_offload=${OPTIMIZER_OFFLOAD:-True}
    actor_rollout_ref.actor.megatron.grad_offload=${GRAD_OFFLOAD:-True}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP_SIZE}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP_SIZE}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP_SIZE}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP_SIZE}
    actor_rollout_ref.actor.use_fused_kernels=${USE_FUSED_KERNELS:-True}
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=${GRADIENT_ACCUMULATION_FUSION:-False}
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    actor_rollout_ref.actor.checkpoint.save_contents="['model']"
    actor_rollout_ref.actor.checkpoint.load_contents="['model']"
)

# --- [Reference Model] ---
ref_params=(
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${REF_PP_SIZE}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${REF_TP_SIZE}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${REF_EP_SIZE}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${REF_ETP_SIZE}
    actor_rollout_ref.ref.megatron.param_offload=${REF_PARAM_OFFLOAD:-True}
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BS}
)

# --- [Rollout Engine] ---
rollout_params=(
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME:-sglang}
    actor_rollout_ref.rollout.dtype=${DTYPE:-"bfloat16"}
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.enable_rollout_routing_replay=${ENABLE_ROLLOUT_ROUTING_REPLAY:-False}
    actor_rollout_ref.rollout.calculate_log_probs=${CALCULATE_LOG_PROBS:-True}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}
    actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.multi_stage_wake_up=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BS}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS:-64}
    actor_rollout_ref.rollout.val_kwargs.do_sample=${val_do_sample}
    actor_rollout_ref.rollout.val_kwargs.n=${val_rollout_n}
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature}
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_torch_compile=${SGLANG_ENABLE_TORCH_COMPILE:-False}
)

# --- [Algorithm] ---
algorithm_params=(
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=False
    algorithm.use_kl_in_advantage=${USE_KL_IN_ADVANTAGE:-False}
    algorithm.kl_penalty=k1
    algorithm.kl_ctrl.kl_coef=${KL_COEF:-0.01}
    algorithm.kl_ctrl.kl_threshold=${KL_THRESHOLD:-0.1}
    +algorithm.kl_skip_thought=${KL_SKIP_THOUGHT:-False}
    algorithm.filter_groups.enable=${ENABLE_FILTER_GROUPS:-True}
    algorithm.filter_groups.max_num_gen_batches=0
    algorithm.filter_groups.metric=acc
    algorithm.filter_groups.low_score_threshold=${LOW_SCORE_THRESHOLD:-0.0}
    algorithm.filter_groups.high_score_threshold=${HIGH_SCORE_THRESHOLD:-1.0}
    algorithm.rollout_correction.rollout_is=${ROLLOUT_IS:-token}
    algorithm.rollout_correction.rollout_is_threshold=${ROLLOUT_IS_THRESHOLD:-2.0}
    algorithm.rollout_correction.rollout_is_batch_normalize=${ROLLOUT_IS_BATCH_NORMALIZE:-False}
    algorithm.rollout_correction.rollout_rs=${ROLLOUT_RS:-null}
    algorithm.rollout_correction.rollout_rs_threshold=${ROLLOUT_RS_THRESHOLD:-1.1}
    algorithm.rollout_correction.rollout_rs_threshold_lower=${ROLLOUT_RS_THRESHOLD_LOWER:-0.9}
    algorithm.rollout_correction.rollout_token_veto_threshold=${ROLLOUT_TOKEN_VETO_THRESHOLD:-1e-4}
)

# --- [Reward Model] ---
reward_model_params=(
    reward_model.reward_manager=dapo
    reward_model.overlong_buffer.enable=${OVERLONG_BUFFER_ENABLE:-False}
    reward_model.overlong_buffer.len=${OVERLONG_BUFFER_LEN:-256}
    reward_model.overlong_buffer.penalty_factor=${OVERLONG_BUFFER_PENALTY_FACTOR:-0.1}
    reward_model.thought_match.enable=${THOUGHT_MATCH_ENABLE:-False}
    reward_model.thought_match.base_url=${THOUGHT_MATCH_BASE_URL:-""}
    reward_model.thought_match.start_step=${THOUGHT_MATCH_START_STEP:-0}
)

# ============================================================
# Launch
# ============================================================
python3 -m ui_mopd.main_dapo --config-path=config \
    --config-name='dapo_megatron_trainer.yaml' \
    "${data_params[@]}" \
    "${model_params[@]}" \
    "${trainer_params[@]}" \
    "${ppo_actor_params[@]}" \
    "${ref_params[@]}" \
    "${rollout_params[@]}" \
    "${algorithm_params[@]}" \
    "${reward_model_params[@]}" \
    "$@"
