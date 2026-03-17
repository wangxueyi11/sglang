#!/usr/bin/env python3
"""快速测试：验证多分支修复效果"""
import subprocess
import os
import time
import requests
import json
import re
from datetime import datetime

MODEL_PATH = "/wxyworkspace/Qwen2.5-VL-32B-Instruct"
LOG_DIR = "/wxyworkspace/experiment_logs"
os.makedirs(LOG_DIR, exist_ok=True)

def get_ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def start_server(port, config):
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", MODEL_PATH,
        "--port", str(port),
        "--tp", "4",
        "--host", "0.0.0.0",
        "--speculative-algorithm", "SUFFIX",
        "--disable-cuda-graph",
        "--speculative-num-steps", "3",
        "--speculative-num-draft-tokens", "4",
        "--speculative-suffix-max-tree-depth", "16",
        "--speculative-suffix-max-cached-requests", "10000",
    ]
    if config.get("use_tree_spec"):
        cmd.extend([
            "--speculative-suffix-use-tree-spec",
            "--speculative-suffix-max-branch-factor", str(config.get("branch_factor", 3)),
        ])
    env = os.environ.copy()
    env["PYTHONPATH"] = "/wxyworkspace/python:" + env.get("PYTHONPATH", "")
    env["SGLANG_DISABLE_CUDNN_CHECK"] = "1"
    log_file = f"{LOG_DIR}/server_{get_ts()}.log"
    print(f"启动服务器: port={port}, use_tree_spec={config.get('use_tree_spec')}")
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return proc, log_file

def wait_health(port, timeout=180):
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=5)
            if resp.status_code == 200:
                return True
        except:
            pass
        time.sleep(3)
    return False

def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except:
        proc.kill()
    time.sleep(5)

def send_request(port, prompt, max_tokens=128):
    url = f"http://localhost:{port}/v1/chat/completions"
    data = {
        "model": MODEL_PATH,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    start = time.time()
    try:
        resp = requests.post(url, json=data, timeout=60)
        latency = time.time() - start
        result = resp.json()
        output_tokens = result.get("usage", {}).get("completion_tokens", 0)
        return {"latency": latency, "tokens": output_tokens, "success": True}
    except Exception as e:
        return {"latency": time.time() - start, "tokens": 0, "success": False, "error": str(e)}

PROMPTS = [
    "Write a Python function to calculate the factorial of a number:\ndef factorial(n):",
    "Write a Python function to reverse a string:\ndef reverse_string(s):",
    "Write a Python function to check if a number is prime:\ndef is_prime(n):",
]

def run_test(port, warmup=2):
    for i in range(warmup):
        send_request(port, "Hello", max_tokens=32)
    time.sleep(2)
    
    results = []
    for prompt in PROMPTS:
        r = send_request(port, prompt)
        results.append(r)
    
    total_tokens = sum(r["tokens"] for r in results if r["success"])
    total_time = sum(r["latency"] for r in results)
    success = sum(1 for r in results if r["success"])
    
    return {
        "throughput": total_tokens / total_time if total_time > 0 else 0,
        "total_tokens": total_tokens,
        "success": success,
    }

def get_stats(log_file):
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        accept_rates = re.findall(r'accept rate: ([\d.]+)', content)
        avg_accept_rate = sum(float(x) for x in accept_rates) / len(accept_rates) if accept_rates else 0
        max_branches_matches = re.findall(r'max_branches=(\d+)', content)
        avg_max_branches = sum(int(x) for x in max_branches_matches) / len(max_branches_matches) if max_branches_matches else 0
        multi_branch = len(re.findall(r'max_branches=[2-9]', content))
        return {"avg_accept_rate": avg_accept_rate, "avg_max_branches": avg_max_branches, "multi_branch_count": multi_branch}
    except:
        return {}

def main():
    print("=" * 60)
    print("快速验证测试：多分支修复效果")
    print("=" * 60)
    
    configs = [
        {"name": "single_branch", "use_tree_spec": False},
        {"name": "multi_branch_3", "use_tree_spec": True, "branch_factor": 3},
    ]
    
    results = []
    port = 32001
    
    for cfg in configs:
        print(f"\n>>> 测试: {cfg['name']}")
        proc, log = start_server(port, cfg)
        
        try:
            if not wait_health(port):
                print("服务器启动失败")
                continue
            time.sleep(10)
            
            metrics = run_test(port)
            stats = get_stats(log)
            
            results.append({**cfg, **metrics, **stats})
            
            print(f"  吞吐量: {metrics['throughput']:.2f} tokens/s")
            print(f"  总tokens: {metrics['total_tokens']}")
            if stats:
                print(f"  平均接受率: {stats.get('avg_accept_rate', 0):.1%}")
                print(f"  平均分支数: {stats.get('avg_max_branches', 0):.2f}")
        finally:
            stop_server(proc)
            os.system(f"pkill -9 -f 'sglang.*--port {port}' 2>/dev/null")
            time.sleep(5)
        port += 1
    
    print("\n" + "=" * 60)
    print("结果对比")
    print("=" * 60)
    for r in results:
        print(f"{r['name']}: {r['throughput']:.2f} tokens/s, 分支数={r.get('avg_max_branches', 0):.2f}")

if __name__ == "__main__":
    main()