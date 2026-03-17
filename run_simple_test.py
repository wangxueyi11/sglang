#!/usr/bin/env python3
"""
简单直接的 SGLang Suffix Decoding 测试
"""
import subprocess
import os
import time
import requests
import json
from datetime import datetime

MODEL_PATH = "/wxyworkspace/Qwen2.5-VL-32B-Instruct"
LOG_DIR = "/wxyworkspace/experiment_logs"
os.makedirs(LOG_DIR, exist_ok=True)

def get_ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def start_server(port, use_tree_spec, max_branch_factor):
    """启动服务器"""
    cmd = [
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
        cmd.extend([
            "--speculative-suffix-use-tree-spec",
            "--speculative-suffix-max-branch-factor", str(max_branch_factor),
        ])
    
    env = os.environ.copy()
    env["PYTHONPATH"] = "/wxyworkspace/python:" + env.get("PYTHONPATH", "")
    env["SGLANG_DISABLE_CUDNN_CHECK"] = "1"
    
    log_file = f"{LOG_DIR}/server_{get_ts()}.log"
    print(f"启动服务器: port={port}, use_tree_spec={use_tree_spec}, branch={max_branch_factor}")
    print(f"日志: {log_file}")
    
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    
    return proc, log_file

def wait_server_ready(port, timeout=180):
    """等待服务器就绪"""
    print(f"等待服务器就绪 (port={port})...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            # 尝试健康检查
            resp = requests.get(f"http://localhost:{port}/health", timeout=5)
            if resp.status_code == 200:
                print(f"服务器就绪!")
                return True
        except:
            pass
        time.sleep(3)
    print(f"服务器启动超时!")
    return False

def run_benchmark(port, num_prompts=20):
    """运行 benchmark"""
    cmd = [
        "python3", "-m", "sglang.bench_serving",
        "--port", str(port),
        "--backend", "sglang",
        "--num-prompts", str(num_prompts),
        "--dataset-name", "random",
        "--random-input-len", "128",
        "--random-output-len", "64",
        "--tokenizer", MODEL_PATH,
    ]
    
    print(f"运行 benchmark...")
    env = os.environ.copy()
    env["PYTHONPATH"] = "/wxyworkspace/python:" + env.get("PYTHONPATH", "")
    env["SGLANG_DISABLE_CUDNN_CHECK"] = "1"
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
    return result.stdout + result.stderr

def parse_result(output):
    """解析结果"""
    import re
    throughput = None
    
    # 尝试多种格式
    match = re.search(r"Output token throughput [(]tokens/s[)][: ]+([\d.]+)", output)
    if match:
        throughput = float(match.group(1))
    
    if not throughput:
        match = re.search(r"Throughput[:\s]+([\d.]+)\s*tokens/s", output, re.IGNORECASE)
        if match:
            throughput = float(match.group(1))
    
    return {"throughput": throughput}

def stop_server(proc):
    """停止服务器"""
    print("停止服务器...")
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except:
        proc.kill()
    time.sleep(5)

def main():
    print("=" * 60)
    print("SGLang Suffix Decoding 性能对比测试")
    print("=" * 60)
    
    results = []
    
    # 测试配置
    configs = [
        {"name": "single_branch", "port": 31001, "use_tree_spec": False, "branch": 1},
        {"name": "multi_branch_3", "port": 31002, "use_tree_spec": True, "branch": 3},
    ]
    
    for cfg in configs:
        print(f"\n{'#'*60}")
        print(f"# 配置: {cfg['name']}")
        print(f"# use_tree_spec={cfg['use_tree_spec']}, branch={cfg['branch']}")
        print(f"{'#'*60}")
        
        proc, log = start_server(cfg["port"], cfg["use_tree_spec"], cfg["branch"])
        
        try:
            if not wait_server_ready(cfg["port"], timeout=180):
                print("服务器启动失败，跳过")
                continue
            
            # 额外等待稳定
            time.sleep(15)
            
            # 运行 benchmark
            output = run_benchmark(cfg["port"], num_prompts=30)
            print(output[-2000:])  # 打印最后部分
            
            metrics = parse_result(output)
            results.append({
                "name": cfg["name"],
                "use_tree_spec": cfg["use_tree_spec"],
                "branch": cfg["branch"],
                **metrics
            })
            
        except Exception as e:
            print(f"错误: {e}")
        finally:
            stop_server(proc)
            os.system(f"pkill -9 -f 'sglang.*{cfg['port']}' 2>/dev/null")
    
    # 汇总
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    print(f"\n{'配置':<20} {'吞吐量(tokens/s)':<20}")
    print("-" * 40)
    for r in results:
        t = f"{r.get('throughput', 'N/A'):.2f}" if r.get('throughput') else "N/A"
        print(f"{r['name']:<20} {t:<20}")
    
    if len(results) >= 2 and results[0].get("throughput") and results[1].get("throughput"):
        speedup = results[1]["throughput"] / results[0]["throughput"]
        print(f"\n加速比: {speedup:.2f}x")
    
    # 保存
    with open(f"{LOG_DIR}/results_{get_ts()}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到 {LOG_DIR}")

if __name__ == "__main__":
    main()
