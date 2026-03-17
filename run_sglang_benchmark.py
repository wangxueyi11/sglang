#!/usr/bin/env python3
"""
SGLang Suffix Decoding 直接测试

对比配置：
1. single_branch: 单分支模式 (use_tree_spec=False)
2. multi_branch: 多分支模式 (use_tree_spec=True, max_branch_factor=3)
"""

import subprocess
import os
import time
import re
import json
from datetime import datetime

MODEL_PATH = "/wxyworkspace/Qwen2.5-VL-32B-Instruct"
LOG_DIR = "/wxyworkspace/experiment_logs"
PORT_BASE = 31000

os.makedirs(LOG_DIR, exist_ok=True)

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def run_sglang_server(config_name: str, use_tree_spec: bool, max_branch_factor: int, port: int):
    """启动 SGLang 服务器"""
    
    server_cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", MODEL_PATH,
        "--port", str(port),
        "--tp", "4",
        "--host", "0.0.0.0",
        "--speculative-algorithm", "SUFFIX",
        "--speculative-num-steps", "3",
        "--speculative-num-draft-tokens", "4",
        "--disable-cuda-graph",
        "--speculative-suffix-max-tree-depth", "12",
        "--speculative-suffix-max-cached-requests", "10000",
    ]
    
    if use_tree_spec:
        server_cmd.extend([
            "--speculative-suffix-use-tree-spec",
            "--speculative-suffix-max-branch-factor", str(max_branch_factor),
        ])
    
    env = os.environ.copy()
    env["PYTHONPATH"] = "/wxyworkspace/python:" + env.get("PYTHONPATH", "")
    env["SGLANG_DISABLE_CUDNN_CHECK"] = "1"
    
    log_file = f"{LOG_DIR}/server_{config_name}_{get_timestamp()}.log"
    print(f"启动服务器: {' '.join(server_cmd)}")
    print(f"日志文件: {log_file}")
    
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            server_cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd="/wxyworkspace"
        )
    
    return process, log_file

def wait_for_server(port: int, timeout: int = 300):
    """等待服务器启动"""
    import requests
    
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=5)
            if resp.status_code == 200:
                print(f"服务器在端口 {port} 已就绪")
                return True
        except:
            pass
        time.sleep(5)
    return False

def run_benchmark(port: int, num_requests: int = 20, output_len: int = 100):
    """运行 benchmark"""
    
    benchmark_cmd = [
        "python3", "-m", "sglang.bench_serving",
        "--port", str(port),
        "--backend", "sglang",
        "--num-prompts", str(num_requests),
        "--dataset-name", "random",
        "--random-input-len", "128",
        "--random-output-len", str(output_len),
        "--tokenizer", MODEL_PATH,
    ]
    
    print(f"运行 benchmark: {' '.join(benchmark_cmd)}")
    
    result = subprocess.run(
        benchmark_cmd,
        capture_output=True,
        text=True,
        timeout=300
    )
    
    return result.stdout + result.stderr

def parse_benchmark_output(output: str) -> dict:
    """解析 benchmark 输出"""
    metrics = {
        "throughput": None,
        "latency_mean": None,
        "latency_median": None,
    }
    
    # 解析吞吐量 (tokens/s)
    throughput_match = re.search(r"Throughput[:\s]+([\d.]+)\s*tokens/s", output, re.IGNORECASE)
    if throughput_match:
        metrics["throughput"] = float(throughput_match.group(1))
    
    # 另一种格式
    if not metrics["throughput"]:
        throughput_match = re.search(r"Output token throughput [(]tokens/s[)][: ]+([\d.]+)", output)
        if throughput_match:
            metrics["throughput"] = float(throughput_match.group(1))
    
    # 解析延迟
    latency_match = re.search(r"Mean TTFT [(]ms[)][: ]+([\d.]+)", output)
    if latency_match:
        metrics["latency_mean"] = float(latency_match.group(1))
    
    return metrics

def main():
    print("=" * 70)
    print("SGLang Suffix Decoding 性能对比测试")
    print("=" * 70)
    
    configs = [
        {"name": "single_branch", "use_tree_spec": False, "max_branch_factor": 1, "port": 31001},
        {"name": "multi_branch_3", "use_tree_spec": True, "max_branch_factor": 3, "port": 31002},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{'#'*70}")
        print(f"# 测试配置: {config['name']}")
        print(f"# use_tree_spec={config['use_tree_spec']}, max_branch_factor={config['max_branch_factor']}")
        print(f"{'#'*70}")
        
        # 启动服务器
        server_process, server_log = run_sglang_server(
            config["name"],
            config["use_tree_spec"],
            config["max_branch_factor"],
            config["port"]
        )
        
        try:
            # 等待服务器启动
            if not wait_for_server(config["port"]):
                print(f"服务器启动超时")
                continue
            
            # 运行 benchmark
            time.sleep(10)  # 额外等待稳定
            benchmark_output = run_benchmark(config["port"], num_requests=30, output_len=100)
            print(benchmark_output)
            
            # 解析结果
            metrics = parse_benchmark_output(benchmark_output)
            results.append({
                "config_name": config["name"],
                "use_tree_spec": config["use_tree_spec"],
                "max_branch_factor": config["max_branch_factor"],
                **metrics
            })
            
        finally:
            # 停止服务器
            print(f"停止服务器...")
            server_process.terminate()
            try:
                server_process.wait(timeout=30)
            except:
                server_process.kill()
            
            # 清理残留进程
            os.system(f"pkill -f 'sglang.launch_server.*--port {config['port']}'")
            time.sleep(5)
    
    # 打印汇总
    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)
    
    print(f"\n{'配置':<20} {'吞吐量(tokens/s)':<20} {'平均延迟(ms)':<15}")
    print("-" * 55)
    for r in results:
        throughput = f"{r.get('throughput', 'N/A'):.2f}" if r.get('throughput') else "N/A"
        latency = f"{r.get('latency_mean', 'N/A'):.2f}" if r.get('latency_mean') else "N/A"
        print(f"{r['config_name']:<20} {throughput:<20} {latency:<15}")
    
    # 计算加速比
    if len(results) >= 2 and results[0].get("throughput") and results[1].get("throughput"):
        speedup = results[1]["throughput"] / results[0]["throughput"]
        print(f"\n多分支模式相对于单分支模式的加速比: {speedup:.2f}x")
    
    # 保存结果
    result_file = f"{LOG_DIR}/benchmark_results_{get_timestamp()}.json"
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n结果已保存到: {result_file}")

if __name__ == "__main__":
    main()
