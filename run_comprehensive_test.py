#!/usr/bin/env python3
"""
全面测试 SGLang Suffix Decoding 不同参数配置
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

def start_server(port, config):
    """启动服务器"""
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", MODEL_PATH,
        "--port", str(port),
        "--tp", "4",
        "--host", "0.0.0.0",
        "--speculative-algorithm", "SUFFIX",
        "--disable-cuda-graph",
        "--speculative-num-steps", str(config.get("num_steps", 3)),
        "--speculative-num-draft-tokens", str(config.get("num_draft_tokens", 4)),
        "--speculative-suffix-max-tree-depth", str(config.get("max_tree_depth", 12)),
        "--speculative-suffix-max-cached-requests", "10000",
    ]
    if config.get("use_tree_spec", False):
        cmd.extend([
            "--speculative-suffix-use-tree-spec",
            "--speculative-suffix-max-branch-factor", str(config.get("branch_factor", 3)),
        ])
    
    env = os.environ.copy()
    env["PYTHONPATH"] = "/wxyworkspace/python:" + env.get("PYTHONPATH", "")
    env["SGLANG_DISABLE_CUDNN_CHECK"] = "1"
    
    log_file = f"{LOG_DIR}/server_{get_ts()}.log"
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    
    return proc, log_file

def wait_health(port, timeout=180):
    """等待服务器就绪"""
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
    
    return {"latency": latency, "output_tokens": output_tokens}

def run_test(port, num_requests=20):
    """运行测试"""
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about nature.",
        "What are the benefits of exercise?",
        "Describe the process of photosynthesis.",
    ] * (num_requests // 5 + 1)
    prompts = prompts[:num_requests]
    
    # warmup
    try:
        send_request(port, "Hello", max_tokens=10)
    except:
        pass
    time.sleep(2)
    
    results = []
    total_tokens = 0
    test_start = time.time()
    
    for prompt in prompts:
        try:
            r = send_request(port, prompt, max_tokens=64)
            results.append(r)
            total_tokens += r["output_tokens"]
        except Exception as e:
            pass
    
    total_time = time.time() - test_start
    
    return {
        "total_tokens": total_tokens,
        "total_time": total_time,
        "num_requests": len(results),
        "throughput": total_tokens / total_time if total_time > 0 else 0,
        "avg_latency": sum(r["latency"] for r in results) / len(results) if results else 0,
    }

def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except:
        proc.kill()
    time.sleep(5)

def main():
    print("=" * 70)
    print("SGLang Suffix Decoding 全面参数对比实验")
    print("=" * 70)
    
    all_results = []
    port = 31001
    
    # 实验1: 不同 branch_factor (启用 tree_spec)
    print("\n" + "=" * 70)
    print("实验1: 不同 branch_factor 对比 (use_tree_spec=True)")
    print("=" * 70)
    
    branch_configs = [
        {"name": "no_tree", "use_tree_spec": False, "branch_factor": 1},
        {"name": "branch_2", "use_tree_spec": True, "branch_factor": 2},
        {"name": "branch_3", "use_tree_spec": True, "branch_factor": 3},
        {"name": "branch_4", "use_tree_spec": True, "branch_factor": 4},
        {"name": "branch_5", "use_tree_spec": True, "branch_factor": 5},
    ]
    
    for cfg in branch_configs:
        print(f"\n>>> 测试: {cfg['name']}")
        proc, log = start_server(port, cfg)
        
        try:
            if not wait_health(port):
                print("服务器启动失败!")
                continue
            time.sleep(8)
            
            metrics = run_test(port, num_requests=20)
            all_results.append({**cfg, **metrics})
            
            print(f"    吞吐量: {metrics['throughput']:.2f} tokens/s, 延迟: {metrics['avg_latency']:.2f}s")
        except Exception as e:
            print(f"错误: {e}")
        finally:
            stop_server(proc)
            os.system(f"pkill -9 -f 'sglang.*--port {port}' 2>/dev/null")
            time.sleep(5)
        
        port += 1
    
    # 实验2: 不同 max_tree_depth
    print("\n" + "=" * 70)
    print("实验2: 不同 max_tree_depth 对比 (branch_factor=3)")
    print("=" * 70)
    
    depth_configs = [
        {"name": "depth_6", "use_tree_spec": True, "branch_factor": 3, "max_tree_depth": 6},
        {"name": "depth_12", "use_tree_spec": True, "branch_factor": 3, "max_tree_depth": 12},
        {"name": "depth_18", "use_tree_spec": True, "branch_factor": 3, "max_tree_depth": 18},
        {"name": "depth_24", "use_tree_spec": True, "branch_factor": 3, "max_tree_depth": 24},
    ]
    
    for cfg in depth_configs:
        print(f"\n>>> 测试: {cfg['name']}")
        proc, log = start_server(port, cfg)
        
        try:
            if not wait_health(port):
                print("服务器启动失败!")
                continue
            time.sleep(8)
            
            metrics = run_test(port, num_requests=20)
            all_results.append({**cfg, **metrics})
            
            print(f"    吞吐量: {metrics['throughput']:.2f} tokens/s, 延迟: {metrics['avg_latency']:.2f}s")
        except Exception as e:
            print(f"错误: {e}")
        finally:
            stop_server(proc)
            os.system(f"pkill -9 -f 'sglang.*--port {port}' 2>/dev/null")
            time.sleep(5)
        
        port += 1
    
    # 实验3: 不同 num_draft_tokens
    print("\n" + "=" * 70)
    print("实验3: 不同 num_draft_tokens 对比 (branch_factor=3)")
    print("=" * 70)
    
    draft_configs = [
        {"name": "draft_2", "use_tree_spec": True, "branch_factor": 3, "num_draft_tokens": 2},
        {"name": "draft_4", "use_tree_spec": True, "branch_factor": 3, "num_draft_tokens": 4},
        {"name": "draft_6", "use_tree_spec": True, "branch_factor": 3, "num_draft_tokens": 6},
        {"name": "draft_8", "use_tree_spec": True, "branch_factor": 3, "num_draft_tokens": 8},
    ]
    
    for cfg in draft_configs:
        print(f"\n>>> 测试: {cfg['name']}")
        proc, log = start_server(port, cfg)
        
        try:
            if not wait_health(port):
                print("服务器启动失败!")
                continue
            time.sleep(8)
            
            metrics = run_test(port, num_requests=20)
            all_results.append({**cfg, **metrics})
            
            print(f"    吞吐量: {metrics['throughput']:.2f} tokens/s, 延迟: {metrics['avg_latency']:.2f}s")
        except Exception as e:
            print(f"错误: {e}")
        finally:
            stop_server(proc)
            os.system(f"pkill -9 -f 'sglang.*--port {port}' 2>/dev/null")
            time.sleep(5)
        
        port += 1
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)
    
    print("\n实验1: 不同 branch_factor")
    print(f"{'配置':<15} {'吞吐量':<20} {'延迟':<15} {'相对加速':<10}")
    print("-" * 60)
    baseline = None
    for r in all_results:
        if r['name'].startswith('no_tree') or r['name'].startswith('branch'):
            if baseline is None and r['name'] == 'no_tree':
                baseline = r['throughput']
            speedup = r['throughput'] / baseline if baseline else 1.0
            print(f"{r['name']:<15} {r['throughput']:<20.2f} {r['avg_latency']:<15.2f} {speedup:<10.2f}x")
    
    print("\n实验2: 不同 max_tree_depth")
    print(f"{'配置':<15} {'吞吐量':<20} {'延迟':<15}")
    print("-" * 50)
    for r in all_results:
        if r['name'].startswith('depth'):
            print(f"{r['name']:<15} {r['throughput']:<20.2f} {r['avg_latency']:<15.2f}")
    
    print("\n实验3: 不同 num_draft_tokens")
    print(f"{'配置':<15} {'吞吐量':<20} {'延迟':<15}")
    print("-" * 50)
    for r in all_results:
        if r['name'].startswith('draft'):
            print(f"{r['name']:<15} {r['throughput']:<20.2f} {r['avg_latency']:<15.2f}")
    
    # 保存完整结果
    result_file = f"{LOG_DIR}/comprehensive_results_{get_ts()}.json"
    with open(result_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n完整结果已保存: {result_file}")

if __name__ == "__main__":
    main()
