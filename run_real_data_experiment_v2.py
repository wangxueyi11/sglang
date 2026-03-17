#!/usr/bin/env python3
"""
使用真实电商数据测试 SGLang Suffix Decoding (修复版)
- 添加接受率统计
- 吞吐量计算: token/(E2E-TTFT)
"""
import subprocess
import os
import time
import requests
import json
import csv
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
    """从CSV加载真实请求数据"""
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
    """启动 SGLang 服务器"""
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
    print(f"等待服务器就绪 (port={port})...")
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
    """发送请求并测量 TTFT 和 E2E 延迟"""
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
        "stream": True,  # 使用流式来测量 TTFT
    }
    
    start_time = time.time()
    ttft = None
    output_tokens = 0
    input_tokens = 0
    
    try:
        with requests.post(url, headers=headers, json=data, timeout=120, stream=True) as resp:
            for line in resp.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        chunk = line[6:]
                        if chunk == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(chunk)
                            # TTFT: 第一个 token 的时间
                            if ttft is None and chunk_data.get('choices', [{}])[0].get('delta', {}).get('content'):
                                ttft = time.time() - start_time
                            # 统计 token
                            usage = chunk_data.get('usage', {})
                            if usage:
                                output_tokens = usage.get('completion_tokens', output_tokens)
                                input_tokens = usage.get('prompt_tokens', input_tokens)
                        except:
                            pass
        
        e2e_latency = time.time() - start_time
        ttft = ttft if ttft else e2e_latency  # 如果没有获取到 TTFT，使用 E2E
        
        return {
            "e2e_latency": e2e_latency,
            "ttft": ttft,
            "output_tokens": output_tokens,
            "input_tokens": input_tokens,
            "generation_time": e2e_latency - ttft,  # 纯生成时间
            "throughput": output_tokens / (e2e_latency - ttft) if (e2e_latency - ttft) > 0 else 0,
            "success": True,
        }
    except Exception as e:
        return {
            "e2e_latency": time.time() - start_time,
            "ttft": time.time() - start_time,
            "output_tokens": 0,
            "input_tokens": 0,
            "success": False,
            "error": str(e),
        }

def run_experiment(port: int, prompts: List[Dict], max_tokens: int = 256, warmup: int = 3) -> Dict[str, Any]:
    """运行实验，收集详细指标"""
    print(f"运行实验: {len(prompts)} prompts, max_tokens={max_tokens}")
    
    # Warmup
    print(f"Warmup ({warmup} requests)...")
    for i in range(min(warmup, len(prompts))):
        send_request(port, prompts[i], max_tokens=32)
    time.sleep(3)
    
    # 正式测试
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
            print(f"  完成 {i+1}/{len(prompts)} 请求")
    
    total_time = time.time() - test_start
    success_count = sum(1 for r in results if r['success'])
    
    # 计算正确的吞吐量: tokens / (E2E - TTFT)
    # 即 tokens / 纯生成时间
    output_throughput = total_output_tokens / total_gen_time if total_gen_time > 0 else 0
    input_throughput = total_input_tokens / total_time if total_time > 0 else 0
    avg_ttft = total_ttft / success_count if success_count > 0 else 0
    
    return {
        "total_requests": len(prompts),
        "success_count": success_count,
        "total_time": total_time,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_gen_time": total_gen_time,  # 纯生成时间总和
        "input_throughput": input_throughput,
        "output_throughput": output_throughput,  # tokens/(E2E-TTFT)
        "avg_ttft": avg_ttft,
        "avg_latency": sum(r['e2e_latency'] for r in results) / len(results) if results else 0,
        "results": results,
    }

def get_acceptance_stats(log_file: str) -> Dict[str, Any]:
    """从服务器日志中提取接受率统计"""
    import re
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # 查找多分支树结构日志
        multi_branch_count = len(re.findall(r'MULTI-BRANCH.*roots=[2-9]', content))
        single_branch_count = len(re.findall(r'MULTI-BRANCH.*roots=1\b', content))
        
        # 查找 max_branches 统计
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
    print("SGLang Suffix Decoding 真实数据对比实验 (修复版)")
    print(f"数据来源: {DATA_FILE}")
    print("=" * 70)
    
    prompts = load_real_prompts(num_samples=50)
    if not prompts:
        print("无法加载数据，退出")
        return
    
    all_results = []
    port = 31001
    
    # 实验: 基线对比
    print("\n" + "=" * 70)
    print("实验: 基线对比")
    print("=" * 70)
    
    baseline_configs = [
        {"name": "suffix_single", "spec_algorithm": "SUFFIX", "use_tree_spec": False},
        {"name": "suffix_multi_3", "spec_algorithm": "SUFFIX", "use_tree_spec": True, "branch_factor": 3},
        {"name": "suffix_multi_5", "spec_algorithm": "SUFFIX", "use_tree_spec": True, "branch_factor": 5},
        {"name": "draft_8", "spec_algorithm": "SUFFIX", "use_tree_spec": True, "branch_factor": 3, "num_draft_tokens": 8},
    ]
    
    for cfg in baseline_configs:
        print(f"\n>>> 测试: {cfg['name']}")
        proc, log = start_server(port, cfg)
        
        try:
            if not wait_health(port):
                print("服务器启动失败!")
                continue
            time.sleep(10)
            
            metrics = run_experiment(port, prompts, max_tokens=256)
            acceptance = get_acceptance_stats(log)
            
            all_results.append({**cfg, **metrics, **acceptance})
            
            print(f"    输出吞吐量 (tokens/(E2E-TTFT)): {metrics['output_throughput']:.2f}")
            print(f"    平均 TTFT: {metrics['avg_ttft']:.2f}s")
            print(f"    平均 E2E 延迟: {metrics['avg_latency']:.2f}s")
            if acceptance:
                print(f"    多分支次数: {acceptance.get('multi_branch_count', 0)}")
                print(f"    平均 max_branches: {acceptance.get('avg_max_branches', 0):.1f}")
            
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            stop_server(proc)
            os.system(f"pkill -9 -f 'sglang.*--port {port}' 2>/dev/null")
            time.sleep(5)
        
        port += 1
    
    # 汇总
    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)
    
    print(f"\n{'配置':<20} {'吞吐量(tokens/(E2E-TTFT))':<28} {'TTFT(s)':<12} {'多分支次数':<12}")
    print("-" * 72)
    for r in all_results:
        print(f"{r['name']:<20} {r.get('output_throughput', 0):<28.2f} {r.get('avg_ttft', 0):<12.2f} {r.get('multi_branch_count', 'N/A')}")
    
    # 计算加速比
    baseline_throughput = all_results[0].get('output_throughput', 0) if all_results else 0
    print(f"\n基线吞吐量: {baseline_throughput:.2f} tokens/(E2E-TTFT)")
    for r in all_results[1:]:
        speedup = r.get('output_throughput', 0) / baseline_throughput if baseline_throughput else 0
        print(f"{r['name']}: 加速比 {speedup:.2f}x")
    
    # 保存结果
    result_file = f"{LOG_DIR}/real_data_results_v2_{get_ts()}.json"
    summary = [{k: v for k, v in r.items() if k != 'results'} for r in all_results]
    with open(result_file, "w", encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n完整结果已保存: {result_file}")

if __name__ == "__main__":
    main()
