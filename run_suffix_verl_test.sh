#!/bin/bash
# verl SUFFIX 推测解码集成测试
# 使用 SGLang rollout + SUFFIX 推测解码

set -x

# CUDA 兼容性设置
export TORCH_CUDA_ARCH_LIST="9.0"  # H20 GPU 架构
export CUDA_LAUNCH_BLOCKING=1
export CUDA_MODULE_LOADING=LAZY

# 添加自定义 SGLang 到 PYTHONPATH
export PYTHONPATH="/wxyworkspace/python:$PYTHONPATH"

# 配置
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
TRAIN_FILE="/wxyworkspace/verl_test_data/train.parquet"
TEST_FILE="/wxyworkspace/verl_test_data/test.parquet"

# 运行 verl PPO 训练，启用 SUFFIX 推测解码
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=32 \
    data.max_prompt_length=128 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=False \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path="${MODEL_PATH}" \
    +critic.model.override_config.attn_implementation=sdpa \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=8 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_suffix_test' \
    trainer.experiment_name='qwen2.5-7b-suffix' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=5 \
    \
    actor_rollout_ref.model.mtp.enable=True \
    actor_rollout_ref.model.mtp.enable_rollout=True \
    actor_rollout_ref.model.mtp.speculative_algorithm=SUFFIX \
    +actor_rollout_ref.model.mtp.suffix_use_tree_spec=True \
    +actor_rollout_ref.model.mtp.suffix_max_branch_factor=2 \
    +actor_rollout_ref.model.mtp.suffix_max_tree_depth=12 \
    $@
