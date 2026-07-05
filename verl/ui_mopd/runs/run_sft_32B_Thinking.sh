#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1  # For megatron communication/computation overlapping

nodes=${NODES:-1}
gpus_per_node=${GPUS_PER_NODE:-8}
master_addr="${MASTER_ADDR:-localhost}"
master_port="${MASTER_PORT:-6000}"

training_dataset=${TRAINING_DATASET:-"/path/to/sft_train.parquet"}
val_dataset=${VAL_DATASET:-"/path/to/sft_test.parquet"}

project=${PROJECT:-"stage1-sft"}
model_name=${MODEL_NAME:-"Qwen3-VL-32B-Thinking"}
model_path=${MODEL_PATH:-"/path/to/Qwen3-VL-32B-Thinking"}

dataset=${DATASET:-"gui-agent"}
experiment_set=${EXPERIMENT_SET:-"sft-${model_name}_${dataset}_${nodes}node-${total_training_steps}steps"}
local_dir=${LOCAL_DIR:-"/path/to/checkpoints/${model_name}/${project}/${experiment_set}"}
lr=${LR:-1e-6}
train_batch_size=${TRAIN_BATCH_SIZE:-32}
micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU:-1}
total_epochs=${TOTAL_EPOCHS:-1}
max_length=${MAX_LENGTH:-16384}
save_freq=${SAVE_FREQ:-10}
test_freq=${TEST_FREQ:-10}
tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE:-4}
param_offload=${PARAM_OFFLOAD:-False}
grad_offload=${GRAD_OFFLOAD:-True}
optimizer_offload=${OPTIMIZER_OFFLOAD:-True}
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}
max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU:-16384}
lr_decay_style=${LR_DECAY_STYLE:-"cosine"}
add_reasoning_content=${ADD_REASONING_CONTENT:-True}
min_pixels=${MIN_PIXELS:-3136}
max_pixels=${MAX_PIXELS:-6553600}


DISTRIBUTED_ARGS=(
    --nproc_per_node ${gpus_per_node}
    --nnodes ${nodes}
    --master_addr ${master_addr}
    --master_port ${master_port}
)


python -m torch.distributed.run \
  "${DISTRIBUTED_ARGS[@]}" -m ui_mopd.sft \
  --config-path=config \
  --config-name='sft_trainer_engine.yaml' \
  \
  data.train_files=${training_dataset} \
  data.val_files=${val_dataset} \
  data.truncation=error \
  data.max_length=${max_length} \
  data.train_batch_size=${train_batch_size} \
  data.micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
  data.use_dynamic_bsz=${use_dynamic_bsz} \
  data.max_token_len_per_gpu=${max_token_len_per_gpu} \
  data.add_reasoning_content=${add_reasoning_content} \
  data.min_pixels=${min_pixels} \
  data.max_pixels=${max_pixels} \
  \
  model.path=${model_path} \
  \
  engine.param_offload=${param_offload} \
  engine.grad_offload=${grad_offload} \
  engine.optimizer_offload=${optimizer_offload} \
  engine.tensor_model_parallel_size=${tensor_model_parallel_size} \
  +engine.override_transformer_config.gradient_accumulation_fusion=${GRADIENT_ACCUMULATION_FUSION:-False} \
  +engine.override_transformer_config.recompute_method=uniform \
  +engine.override_transformer_config.recompute_granularity=full \
  +engine.override_transformer_config.recompute_num_layers=1 \
  \
  optim.lr=${lr} \
  optim.min_lr=0 \
  optim.betas='[0.9,0.999]' \
  optim.weight_decay=0.01 \
  optim.lr_warmup_steps_ratio=0.1 \
  optim.lr_decay_style=${lr_decay_style} \
  +optim.override_optimizer_config.optimizer_offload_fraction=1.0 \
  +optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
  +optim.override_optimizer_config.use_distributed_optimizer=True \
  +optim.override_optimizer_config.use_precision_aware_optimizer=True \
  +optim.override_optimizer_config.optimizer_cpu_offload=True \
  +optim.override_optimizer_config.log_num_zeros_in_grad=True \
  \
  trainer.project_name=${project} \
  trainer.experiment_name=test \
  trainer.default_local_dir=${local_dir} \
  trainer.total_epochs=${total_epochs} \
  trainer.logger="['console', 'tensorboard']" \
  trainer.save_freq=${save_freq} \
  trainer.test_freq=${test_freq} \
  trainer.max_ckpt_to_keep=null \
  "$@"
