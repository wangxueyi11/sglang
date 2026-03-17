#!/usr/bin/env python3
"""
TopK Suffix Decoding 对比实验

对比配置：
1. single_branch: 单分支模式 (use_tree_spec=False)
2. multi_branch: 多分支模式 (use_tree_spec=True, max_branch_factor=3)

评价指标：
- Rollout 吞吐量 (tokens/sec)
- 平均生成时间
- 推测解码接受率
"""

import subprocess
import os
import sys
import time
import json
import re
from datetime import datetime

# 配置
MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
TRAIN_FILE = "/wxyworkspace/verl_test_data/train.parquet"
TEST_FILE = "/wxyworkspace/verl_test_data/test.parquet"
LOG_DIR = "/wxyworkspace/experiment_logs"

# 确保日志目录存在
os.makedirs(LOG_DIR, exist_ok=True)

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def run_experiment(config_name: str, use_tree_spec: bool, max_branch_factor: int, 
                   num_steps: int = 10):
    """
    运行单个实验配置
    """
    timestamp = get_timestamp()
    log_file = f"{LOG_DIR}/{config_name}_{timestamp}.log"
    
    print(f"\n{'='*60}")
    print(f"运行实验: {config_name}")
    print(f"use_tree_spec={use_tree_spec}, max_branch_factor={max_branch_factor}")
    print(f"日志文件: {log_file}")
    print(f"{'='*60}\n")
    
    # 构建命令
    cmd = [
        "python3", "-m", "verl.trainer.main_ppo",
        "algorithm.adv_estimator=gae",
        f"data.train_files={TRAIN_FILE}",
        f"data.val_files={TEST_FILE}",
        "data.train_batch_size=16",
        "data.max_prompt_length=128",
        "data.max_response_length=128",
        "data.filter_overlong_prompts=False",
        f"actor_rollout_ref.model.path={MODEL_PATH}",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.ppo_mini_batch_size=16",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=2",
        "actor_rollout_ref.rollout.name=sglang",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
        "critic.optim.lr=1e-5",
        "critic.model.use_remove_padding=True",
        f"critic.model.path={MODEL_PATH}",
        "+critic.model.override_config.attn_implementation=sdpa",
        "critic.model.enable_gradient_checkpointing=True",
        "critic.ppo_micro_batch_size_per_gpu=8",
        "critic.model.fsdp_config.param_offload=False",
        "critic.model.fsdp_config.optimizer_offload=False",
        "algorithm.use_kl_in_reward=False",
        "trainer.critic_warmup=0",
        "trainer.logger=[\"console\"]",
        f"trainer.project_name=verl_topk_suffix_comparison",
        f"trainer.experiment_name={config_name}",
        "trainer.n_gpus_per_node=2",
        "trainer.nnodes=1",
        "trainer.val_before_train=False",
        "trainer.save_freq=-1",
        "trainer.test_freq=1",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={num_steps}",
        # SUFFIX 配置
        "actor_rollout_ref.model.mtp.enable=True",
        "actor_rollout_ref.model.mtp.enable_rollout=True",
        "actor_rollout_ref.model.mtp.speculative_algorithm=SUFFIX",
        f"+actor_rollout_ref.model.mtp.suffix_use_tree_spec={str(use_tree_spec).lower()}",
        f"+actor_rollout_ref.model.mtp.suffix_max_branch_factor={max_branch_factor}",
        "+actor_rollout_ref.model.mtp.suffix_max_tree_depth=12",
    ]
    
    # 设置环境变量
    env = os.environ.copy()
    env["PYTHONPATH"] = "/wxyworkspace/python:/wxyworkspace/verl:" + env.get("PYTHONPATH", "")
    env["CUDA_LAUNCH_BLOCKING"] = "1"
    env["TORCH_CUDA_ARCH_LIST"] = "9.0"
    
    # 运行并记录日志
    start_time = time.time()
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            cwd="/wxyworkspace"
        )
        
        # 实时输出并记录
        for line in process.stdout:
            print(line, end="")
            f.write(line)
            f.flush()
        
        process.wait()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\n实验 {config_name} 完成，耗时: {elapsed:.2f} 秒")
    
    return {
        "config_name": config_name,
        "use_tree_spec": use_tree_spec,
        "max_branch_factor": max_branch_factor,
        "elapsed_time": elapsed,
        "log_file": log_file,
        "return_code": process.returncode
    }

def parse_metrics(log_file: str) -> dict:
    """
    从日志文件中解析指标
    """
    metrics = {
        "throughput_tokens_per_sec": None,
        "avg_generation_time": None,
        "total_tokens": None,
        "total_time": None,
    }
    
    try:
        with open(log_file, "r") as f:
            content = f.read()
        
        # 尝试解析吞吐量指标
        # 查找类似 "throughput: xxx tokens/sec" 的模式
        throughput_match = re.search(r"throughput[:\s]+([\d.]+)\s*tokens", content, re.IGNORECASE)
        if throughput_match:
            metrics["throughput_tokens_per_sec"] = float(throughput_match.group(1))
        
        # 查找生成时间
        gen_time_match = re.search(r"generation.*time[:\s]+([\d.]+)", content, re.IGNORECASE)
        if gen_time_match:
            metrics["avg_generation_time"] = float(gen_time_match.group(1))
        
        # 查找 rollout 相关指标
        rollout_match = re.search(r"rollout.*tokens[:\s]+(\d+)", content, re.IGNORECASE)
        if rollout_match:
            metrics["total_tokens"] = int(rollout_match.group(1))
            
    except Exception as e:
        print(f"解析指标失败: {e}")
    
    return metrics

def main():
    print("=" * 70)
    print("TopK Suffix Decoding 对比实验")
    print("=" * 70)
    
    # 实验配置
    experiments = [
        {
            "name": "single_branch",
            "use_tree_spec": False,
            "max_branch_factor": 1,
            "description": "单分支模式 (baseline)"
        },
        {
            "name": "multi_branch_3",
            "use_tree_spec": True,
            "max_branch_factor": 3,
            "description": "多分支模式 (top-k=3)"
        },
    ]
    
    results = []
    
    # 运行所有实验
    for exp in experiments:
        print(f"\n{'#'*70}")
        print(f"# 实验: {exp['description']}")
        print(f"{'#'*70}")
        
        result = run_experiment(
            config_name=exp["name"],
            use_tree_spec=exp["use_tree_spec"],
            max_branch_factor=exp["max_branch_factor"],
            num_steps=10  # 短测试
        )
        
        # 解析指标
        metrics = parse_metrics(result["log_file"])
        result.update(metrics)
        results.append(result)
        
        # 打印当前结果
        print(f"\n--- {exp['name']} 结果 ---")
        print(f"  总耗时: {result['elapsed_time']:.2f} 秒")
        if metrics["throughput_tokens_per_sec"]:
            print(f"  吞吐量: {metrics['throughput_tokens_per_sec']:.2f} tokens/sec")
    
    # 汇总对比
    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)
    
    print(f"\n{'配置':<20} {'耗时(秒)':<15} {'吞吐量':<20}")
    print("-" * 55)
    for r in results:
        throughput = r.get("throughput_tokens_per_sec")
        throughput_str = f"{throughput:.2f}" if throughput else "N/A"
        print(f"{r['config_name']:<20} {r['elapsed_time']:<15.2f} {throughput_str:<20}")
    
    # 计算加速比
    if len(results) >= 2:
        baseline_time = results[0]["elapsed_time"]
        for i, r in enumerate(results[1:], 1):
            speedup = baseline_time / r["elapsed_time"] if r["elapsed_time"] > 0 else 0
            print(f"\n{r['config_name']} 相对于 {results[0]['config_name']} 的加速比: {speedup:.2f}x")
    
    # 保存结果
    result_file = f"{LOG_DIR}/comparison_results_{get_timestamp()}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n结果已保存到: {result_file}")
    
    return results

if __name__ == "__main__":
    main()
