#!/usr/bin/env python3
"""
直接使用 HTTP 请求测试 SGLang Suffix Decoding
"""
import subprocess
import os
import time
import requests
import json
import threading
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
    
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    
    return proc

def wait_health(port, timeout=180):
    """等待 health 端点就绪"""
    print(f"等待服务器就绪...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=5)
            if resp.status_code == 200:
                print("Health 端点就绪!")
                return True
        except:
            pass
        time.sleep(3)
    return False

def send_request(port, prompt, max_tokens=64):
    """发送单个请求"""
    url = f"http://localhost:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": MODEL_PATH,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    
    start = time.time()
    resp = requests.post(url, headers=headers, json=data, timeout=60)
    latency = time.time() - start
    
    result = resp.json()
    output_tokens = result.get("usage", {}).get("completion_tokens", 0)
    
    return {
        "latency": latency,
        "output_tokens": output_tokens,
        "throughput": output_tokens / latency if latency > 0 else 0,
    }

def run_test(port, num_requests=30):
    """运行测试"""
    print(f"运行 {num_requests} 个请求...")
    
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about nature.",
        "What are the benefits of exercise?",
        "Describe the process of photosynthesis.",
    ] * (num_requests // 5 + 1)
    prompts = prompts[:num_requests]
    
    results = []
    total_tokens = 0
    total_time = 0
    
    # warmup
    print("Warmup...")
    try:
        send_request(port, "Hello", max_tokens=10)
    except:
        pass
    time.sleep(2)
    
    # 正式测试
    print("开始测试...")
    test_start = time.time()
    
    for i, prompt in enumerate(prompts):
        try:
            r = send_request(port, prompt, max_tokens=64)
            results.append(r)
            total_tokens += r["output_tokens"]
            if (i + 1) % 10 == 0:
                print(f"  完成 {i+1}/{num_requests} 请求")
        except Exception as e:
            print(f"  请求 {i+1} 失败: {e}")
    
    test_end = time.time()
    total_time = test_end - test_start
    
    return {
        "total_tokens": total_tokens,
        "total_time": total_time,
        "num_requests": len(results),
        "throughput": total_tokens / total_time if total_time > 0 else 0,
        "avg_latency": sum(r["latency"] for r in results) / len(results) if results else 0,
    }

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
    print("SGLang Suffix Decoding 直接性能测试")
    print("=" * 60)
    
    results = []
    
    configs = [
        {"name": "single_branch", "port": 31001, "use_tree_spec": False, "branch": 1},
        {"name": "multi_branch_3", "port": 31002, "use_tree_spec": True, "branch": 3},
    ]
    
    for cfg in configs:
        print(f"\n{'#'*60}")
        print(f"# 配置: {cfg['name']}")
        print(f"# use_tree_spec={cfg['use_tree_spec']}, branch={cfg['branch']}")
        print(f"{'#'*60}")
        
        proc = start_server(cfg["port"], cfg["use_tree_spec"], cfg["branch"])
        
        try:
            if not wait_health(cfg["port"]):
                print("服务器启动失败!")
                continue
            
            # 额外等待
            time.sleep(10)
            
            # 运行测试
            metrics = run_test(cfg["port"], num_requests=30)
            
            results.append({
                "name": cfg["name"],
                "use_tree_spec": cfg["use_tree_spec"],
                "branch": cfg["branch"],
                **metrics
            })
            
            print(f"\n结果:")
            print(f"  总tokens: {metrics['total_tokens']}")
            print(f"  总时间: {metrics['total_time']:.2f}s")
            print(f"  吞吐量: {metrics['throughput']:.2f} tokens/s")
            print(f"  平均延迟: {metrics['avg_latency']:.2f}s")
            
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            stop_server(proc)
            os.system(f"pkill -9 -f 'sglang.*--port {cfg['port']}' 2>/dev/null")
            time.sleep(5)
    
    # 汇总
    print("\n" + "=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    
    print(f"\n{'配置':<20} {'吞吐量(tokens/s)':<20} {'平均延迟(s)':<15}")
    print("-" * 55)
    for r in results:
        t = f"{r.get('throughput', 0):.2f}"
        l = f"{r.get('avg_latency', 0):.2f}"
        print(f"{r['name']:<20} {t:<20} {l:<15}")
    
    if len(results) >= 2:
        if results[0].get("throughput") and results[1].get("throughput"):
            speedup = results[1]["throughput"] / results[0]["throughput"]
            print(f"\n多分支模式加速比: {speedup:.2f}x")
    
    # 保存
    with open(f"{LOG_DIR}/direct_results_{get_ts()}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存")

if __name__ == "__main__":
    main()
