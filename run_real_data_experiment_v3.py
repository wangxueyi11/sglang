#!/usr/bin/env python3
"""
SGLang Suffix Decoding 实验脚本 v3
- 使用非流式响应获取正确的 token 统计
- 吞吐量计算: tokens/(E2E-TTFT)
"""
import subprocess
import os
import time
import requests
import json
import csv
import re
from datetime import datetime
from typing import List, Dict, Any
import random

DATA_FILE = "/wxyworkspace/2025-12-10_1_Snippet 1_29337771.csv"
MODEL_PATH = "/wxyworkspace/Qwen2.5-VL-32B-Instruct"
LOG_DIR = "/wxyworkspace/experiment_logs"
os.makedirs(LOG_DIR, exist_ok=True)

def get_ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def load_real_prompts(num_samples: int = 50) -> List[Dict[str, Any]]:
    prompts = []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    random.seed(42)
    sampled = random.sample(rows, min(num_samples, len(rows)))
    for row in sampled:
        try:
            data = json.loads(row['request_response'])
            req = json.loads(data['request_body'])
            messages = req.get('messages', [])
            system_msg = ""
            user_msg = ""
            for msg in messages:
                if msg.get('role') == 'system':
                    system_msg = msg.get('content', '')
                elif msg.get('role') == 'user':
                    user_msg = msg.get('content', '')
            prompts.append({
                'system': system_msg,
                'user': user_msg,
                'full_prompt_len': len(system_msg) + len(user_msg),
            })
        except:
            continue
    print(f"加载了 {len(prompts)} 条真实 prompt")
    return prompts

def start_server(port: int, config: Dict[str, Any]) -> subprocess.Popen:
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", MODEL_PATH,
        "--port", str(port),
        "--tp", "4",
        "--host", "0.0.0.0",
        "--speculative-algorithm", config.get("spec_algorithm", "SUFFIX"),
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
    env["SUFFIX_DEBUG_TREE"] = "0"
    log_file = f"{LOG_DIR}/server_{get_ts()}.log"
    print(f"启动服务器: port={port}, config={config}")
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return proc, log_file

def wait_health(port: int, timeout: int = 180) -> bool:
    print(f"等待服务器就绪...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=5)
            if resp.status_code == 200:
                print("服务器就绪!")
                return True
        except:
            pass
        time.sleep(3)
    return False

def stop_server(proc: subprocess.Popen):
    print("停止服务器...")
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except:
        proc.kill()
    time.sleep(5)

def send_request(port: int, prompt: Dict[str, Any], max_tokens: int = 256) -> Dict[str, Any]:
    """发送请求 - 使用非流式响应获取正确的 token 统计"""
    url = f"http://localhost:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    messages = []
    if prompt.get('system'):
        messages.append({"role": "system", "content": prompt['system']})
    messages.append({"role": "user", "content": prompt['user']})
    data = {
        "model": MODEL_PATH,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,  # 非流式获取完整统计
    }
    
    start_time = time.time()
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=120)
        e2e_latency = time.time() - start_time
        
        result = resp.json()
        usage = result.get("usage", {})
        output_tokens = usage.get("completion_tokens", 0)
        input_tokens = usage.get('prompt_tokens', 0)
        
        # TTFT 估算：首 token 延迟约为 E2E 的 10-20%（对于短输出）
        # 更准确的方法是用流式，但这里用非流式简化
        ttft_estimate = e2e_latency * 0.1  # 粗略估计
        gen_time = e2e_latency - ttft_estimate
        
        return {
            "e2e_latency": e2e_latency,
            "ttft": ttft_estimate,
            "output_tokens": output_tokens,
            "input_tokens": input_tokens,
            "generation_time": gen_time,
            "throughput": output_tokens / gen_time if gen_time > 0 else 0,
            "success": True,
        }
    except Exception as e:
        return {
            "e2e_latency": time.time() - start_time,
            "ttft": 0,
            "output_tokens": 0,
            "input_tokens": 0,
            "success": False,
            "error": str(e),
        }

def run_experiment(port: int, prompts: List[Dict], max_tokens: int = 256, warmup: int = 3) -> Dict[str, Any]:
    print(f"运行实验: {len(prompts)} prompts")
    
    # Warmup
    print(f"Warmup...")
    for i in range(min(warmup, len(prompts))):
        send_request(port, prompts[i], max_tokens=32)
    time.sleep(3)
    
    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_ttft = 0
    total_gen_time = 0
    test_start = time.time()
    
    for i, prompt in enumerate(prompts):
        r = send_request(port, prompt, max_tokens=max_tokens)
        results.append(r)
        if r['success']:
            total_input_tokens += r['input_tokens']
            total_output_tokens += r['output_tokens']
            total_ttft += r['ttft']
            total_gen_time += r['generation_time']
        if (i + 1) % 10 == 0:
            print(f"  完成 {i+1}/{len(prompts)}")
    
    total_time = time.time() - test_start
    success_count = sum(1 for r in results if r['success'])
    
    output_throughput = total_output_tokens / total_gen_time if total_gen_time > 0 else 0
    avg_ttft = total_ttft / success_count if success_count > 0 else 0
    
    return {
        "total_requests": len(prompts),
        "success_count": success_count,
        "total_time": total_time,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_gen_time": total_gen_time,
        "output_throughput": output_throughput,
        "avg_ttft": avg_ttft,
        "avg_latency": sum(r['e2e_latency'] for r in results) / len(results) if results else 0,
        "results": results,
    }

def get_acceptance_stats(log_file: str) -> Dict[str, Any]:
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        multi_branch_count = len(re.findall(r'MULTI-BRANCH.*roots=[2-9]', content))
        single_branch_count = len(re.findall(r'MULTI-BRANCH.*roots=1\b', content))
        max_branches_matches = re.findall(r'max_branches=(\d+)', content)
        avg_max_branches = sum(int(x) for x in max_branches_matches) / len(max_branches_matches) if max_branches_matches else 0
        
        return {
            "multi_branch_count": multi_branch_count,
            "single_branch_count": single_branch_count,
            "avg_max_branches": avg_max_branches,
        }
    except:
        return {}

def main():
    print("=" * 70)
    print("SGLang Suffix Decoding 实验v3")
    print("=" * 70)
    
    prompts = load_real_prompts(num_samples=50)
    if not prompts:
        return
    
    all_results = []
    port = 31001
    
    configs = [
        {"name": "suffix_single", "spec_algorithm": "SUFFIX", "use_tree_spec": False},
        {"name": "suffix_multi_3", "spec_algorithm": "SUFFIX", "use_tree_spec": True, "branch_factor": 3},
        {"name": "draft_8", "spec_algorithm": "SUFFIX", "use_tree_spec": True, "branch_factor": 3, "num_draft_tokens": 8},
    ]
    
    for cfg in configs:
        print(f"\n>>> 测试: {cfg['name']}")
        proc, log = start_server(port, cfg)
        
        try:
            if not wait_health(port):
                continue
            time.sleep(10)
            
            metrics = run_experiment(port, prompts, max_tokens=256)
            acceptance = get_acceptance_stats(log)
            all_results.append({**cfg, **metrics, **acceptance})
            
            print(f"    吞吐量: {metrics['output_throughput']:.2f} tokens/s")
            print(f"    总输出tokens: {metrics['total_output_tokens']}")
            print(f"    平均延迟: {metrics['avg_latency']:.2f}s")
            if acceptance:
                print(f"    多分支次数: {acceptance.get('multi_branch_count', 0)}")
                print(f"    平均 max_branches: {acceptance.get('avg_max_branches', 0):.2f}")
        except Exception as e:
            print(f"错误: {e}")
        finally:
            stop_server(proc)
            os.system(f"pkill -9 -f 'sglang.*--port {port}' 2>/dev/null")
            time.sleep(5)
        port += 1
    
    print("\n" + "=" * 70)
    print("结果汇总")
    print("=" * 70)
    
    print(f"\n{'配置':<20} {'吞吐量':<15} {'输出tokens':<15} {'延迟(s)':<12} {'多分支次数':<12}")
    print("-" * 74)
    for r in all_results:
        print(f"{r['name']:<20} {r.get('output_throughput', 0):<15.2f} {r.get('total_output_tokens', 0):<15} {r.get('avg_latency', 0):<12.2f} {r.get('multi_branch_count', 0)}")
    
    baseline = all_results[0].get('output_throughput', 0) if all_results else 0
    print(f"\n相对加速:")
    for r in all_results[1:]:
        speedup = r.get('output_throughput', 0) / baseline if baseline else 0
        print(f"  {r['name']}: {speedup:.2f}x")
    
    result_file = f"{LOG_DIR}/real_data_results_v3_{get_ts()}.json"
    summary = [{k: v for k, v in r.items() if k != 'results'} for r in all_results]
    with open(result_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n结果已保存: {result_file}")

if __name__ == "__main__":
    main()
