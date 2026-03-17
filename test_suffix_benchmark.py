#!/usr/bin/env python3
"""
测试 SUFFIX 推测解码性能
"""
import os
import subprocess
import time
import json

# 测试配置
MODEL_PATH = "/root/.cache/huggingface/hub/Qwen2.5-VL-32B-Instruct"
# 备用模型
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"

print("=" * 70)
print("SUFFIX 推测解码基准测试")
print("=" * 70)
print(f"模型: {MODEL_PATH}")

# 测试 prompt
TEST_PROMPT = "The meaning of life is"

# 测试配置列表
configs = [
    {"name": "Baseline (无推测)", "speculative_algorithm": None},
    {"name": "SUFFIX (单路径)", "speculative_algorithm": "SUFFIX", 
     "suffix_use_tree_spec": False},
    {"name": "SUFFIX (多分支, branch=2)", "speculative_algorithm": "SUFFIX", 
     "suffix_use_tree_spec": True, "suffix_max_branch_factor": 2},
    {"name": "SUFFIX (多分支, branch=3)", "speculative_algorithm": "SUFFIX", 
     "suffix_use_tree_spec": True, "suffix_max_branch_factor": 3},
]

def build_server_cmd(config):
    """构建 SGLang 服务器启动命令"""
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", MODEL_PATH,
        "--port", "30000",
        "--tp", "2",  # 根据可用 GPU 调整
        "--dtype", "bfloat16",
        "--mem-fraction-static", "0.8",
    ]
    
    if config.get("speculative_algorithm"):
        cmd.extend(["--speculative-algorithm", config["speculative_algorithm"]])
        
        if config["speculative_algorithm"] == "SUFFIX":
            if "suffix_use_tree_spec" in config:
                cmd.extend(["--speculative-suffix-use-tree-spec", str(config["suffix_use_tree_spec"]).lower()])
            if "suffix_max_branch_factor" in config:
                cmd.extend(["--speculative-suffix-max-branch-factor", str(config["suffix_max_branch_factor"])])
    
    return cmd

def test_with_sglang_bench(config):
    """使用 sglang 的 benchmark 脚本测试"""
    print(f"\n{'='*70}")
    print(f"测试配置: {config['name']}")
    print(f"{'='*70}")
    
    # 构建命令
    cmd = build_server_cmd(config)
    print(f"启动命令: {' '.join(cmd)}")
    
    # 这里我们只打印命令，实际测试需要在服务器运行后进行
    return cmd

# 打印测试命令
print("\n测试配置:")
for config in configs:
    cmd = test_with_sglang_bench(config)
    print(f"\n命令: {' '.join(cmd[:8])}...")  # 只打印前几个参数

print("\n" + "=" * 70)
print("提示: 要运行实际测试，请执行以下步骤:")
print("1. 启动 SGLang 服务器 (使用上述命令)")
print("2. 使用 curl 或 Python 客户端发送请求")
print("=" * 70)
