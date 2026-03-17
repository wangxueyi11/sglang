#!/usr/bin/env python3
"""
SUFFIX 推测解码实验 - 使用 SGLang benchmark
"""
import subprocess
import time
import os
import signal
import sys

# 配置
MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
PORT = 30000
TP = 2  # 使用 2 张 GPU

# 实验配置
EXPERIMENTS = [
    {
        "name": "Baseline (无推测)",
        "args": {}
    },
    {
        "name": "SUFFIX (单路径)", 
        "args": {
            "speculative_algorithm": "SUFFIX",
        }
    },
    {
        "name": "SUFFIX (多分支, branch=2)", 
        "args": {
            "speculative_algorithm": "SUFFIX",
            "speculative_suffix_use_tree_spec": True,
            "speculative_suffix_max_branch_factor": 2,
        }
    },
]

def build_server_args(exp):
    """构建服务器启动参数"""
    args = [
        "--model-path", MODEL_PATH,
        "--port", str(PORT),
        "--tp", str(TP),
        "--dtype", "bfloat16",
        "--mem-fraction-static", "0.7",
        "--disable-cuda-graph",  # 避免 cuda graph 问题
    ]
    
    for k, v in exp["args"].items():
        if isinstance(v, bool):
            args.extend([f"--{k.replace('_', '-')}", str(v).lower()])
        else:
            args.extend([f"--{k.replace('_', '-')}", str(v)])
    
    return args

def run_benchmark():
    """运行 benchmark"""
    # 使用 SGLang 的 benchmark 脚本
    bench_cmd = [
        "python3", "-m", "sglang.bench_serving",
        "--backend", "sglang",
        "--port", str(PORT),
        "--dataset-name", "random",
        "--num-prompts", "20",
        "--random-input-len", "128",
        "--random-output-len", "128",
    ]
    
    print(f"\n运行 benchmark: {' '.join(bench_cmd)}")
    result = subprocess.run(bench_cmd, capture_output=True, text=True, timeout=180)
    
    # 提取关键指标
    output = result.stdout + result.stderr
    print(output[-2000:] if len(output) > 2000 else output)
    
    return output

def run_single_experiment(exp):
    """运行单个实验"""
    print(f"\n{'='*70}")
    print(f"实验: {exp['name']}")
    print(f"{'='*70}")
    
    # 构建服务器参数
    server_args = build_server_args(exp)
    print(f"服务器参数: {' '.join(server_args)}")
    
    # 启动服务器
    server_cmd = ["python3", "-m", "sglang.launch_server"] + server_args
    print(f"\n启动服务器...")
    
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid
    )
    
    # 等待服务器启动
    print("等待服务器启动 (60秒)...")
    time.sleep(60)
    
    try:
        # 检查服务器是否还在运行
        if server_proc.poll() is not None:
            print(f"服务器启动失败!")
            print(server_proc.stdout.read().decode())
            return None
        
        # 运行 benchmark
        result = run_benchmark()
        
    finally:
        # 关闭服务器
        print("\n关闭服务器...")
        os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
        server_proc.wait(timeout=30)
        print("服务器已关闭")
    
    return result

def main():
    print("="*70)
    print("SUFFIX 推测解码实验")
    print("="*70)
    print(f"模型: {MODEL_PATH}")
    print(f"端口: {PORT}")
    print(f"TP: {TP}")
    
    results = []
    for exp in EXPERIMENTS:
        result = run_single_experiment(exp)
        results.append({"name": exp["name"], "result": result})
    
    # 打印总结
    print("\n" + "="*70)
    print("实验总结")
    print("="*70)
    for r in results:
        print(f"\n{r['name']}:")
        if r['result']:
            # 提取 throughput
            for line in r['result'].split('\n'):
                if 'throughput' in line.lower() or 'output' in line.lower():
                    print(f"  {line}")

if __name__ == "__main__":
    main()
