#!/usr/bin/env python3
"""
使用 MBPP 代码数据集测试 SGLang Suffix Decoding
- 代码补全任务，有大量重复语法结构
- 测量吞吐量、延迟、接受率
"""
import subprocess
import os
import time
import requests
import json
import re
from datetime import datetime
from typing import List, Dict, Any

MODEL_PATH = "/wxyworkspace/Qwen2.5-VL-32B-Instruct"
DATA_FILE = "/wxyworkspace/mbpp_data.json"
LOG_DIR = "/wxyworkspace/experiment_logs"
os.makedirs(LOG_DIR, exist_ok=True)

def get_ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def load_mbpp_data(num_samples: int = 100) -> List[Dict[str, Any]]:
    """加载 MBPP 数据"""
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)
    
    prompts = []
    for item in data[:num_samples]:
        # 构建代码补全 prompt
        prompt = f"""Write a Python function for the following task:

Task: {item['prompt']}

Provide a complete Python function with proper implementation.
```python
"""
        prompts.append({
            'prompt': prompt,
            'expected_code': item['code'],
            'task_id': item['task_id'],
        })
    
    print(f"加载了 {len(prompts)} 条代码补全任务")
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
        "--speculative-suffix-max-tree-depth", str(config.get("max_tree_depth", 16)),
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

def send_request(port: int, prompt: str, max_tokens: int = 256) -> Dict[str, Any]:
    """发送代码补全请求"""
    url = f"http://localhost:{port}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": MODEL_PATH,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    
    start_time = time.time()
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=120)
        e2e_latency = time.time() - start_time
        
        result = resp.json()
        usage = result.get("usage", {})
        output_tokens = usage.get("completion_tokens", 0)
        input_tokens = usage.get('prompt_tokens', 0)
        
        # 获取生成的内容
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        return {
            "e2e_latency": e2e_latency,
            "output_tokens": output_tokens,
            "input_tokens": input_tokens,
            "throughput": output_tokens / e2e_latency if e2e_latency > 0 else 0,
            "success": True,
            "content": content[:100] + "..." if len(content) > 100 else content,
        }
    except Exception as e:
        return {
            "e2e_latency": time.time() - start_time,
            "output_tokens": 0,
            "input_tokens": 0,
            "success": False,
            "error": str(e),
        }

def run_experiment(port: int, prompts: List[Dict], max_tokens: int = 256, warmup: int = 3) -> Dict[str, Any]:
    print(f"运行实验: {len(prompts)} 个代码补全任务")
    
    # Warmup
    print(f"Warmup...")
    for i in range(min(warmup, len(prompts))):
        send_request(port, prompts[i]['prompt'], max_tokens=64)
    time.sleep(3)
    
    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_time = 0
    test_start = time.time()
    
    for i, item in enumerate(prompts):
        r = send_request(port, item['prompt'], max_tokens=max_tokens)
        results.append(r)
        if r['success']:
            total_input_tokens += r['input_tokens']
            total_output_tokens += r['output_tokens']
            total_time += r['e2e_latency']
        if (i + 1) % 20 == 0:
            print(f"  完成 {i+1}/{len(prompts)}")
    
    elapsed = time.time() - test_start
    success_count = sum(1 for r in results if r['success'])
    
    return {
        "total_requests": len(prompts),
        "success_count": success_count,
        "total_elapsed": elapsed,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "output_throughput": total_output_tokens / total_time if total_time > 0 else 0,
        "avg_latency": total_time / success_count if success_count > 0 else 0,
        "results": results,
    }

def get_server_stats(log_file: str) -> Dict[str, Any]:
    """从服务器日志提取统计"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # 多分支统计
        multi_branch_count = len(re.findall(r'MULTI-BRANCH.*roots=[2-9]', content))
        single_branch_count = len(re.findall(r'MULTI-BRANCH.*roots=1\b', content))
        
        # 接受率统计
        accept_rate_matches = re.findall(r'accept rate: ([\d.]+)', content)
        avg_accept_rate = sum(float(x) for x in accept_rate_matches) / len(accept_rate_matches) if accept_rate_matches else 0
        
        accept_len_matches = re.findall(r'accept len: ([\d.]+)', content)
        avg_accept_len = sum(float(x) for x in accept_len_matches) / len(accept_len_matches) if accept_len_matches else 0
        
        # max_branches 统计
        max_branches_matches = re.findall(r'max_branches=(\d+)', content)
        avg_max_branches = sum(int(x) for x in max_branches_matches) / len(max_branches_matches) if max_branches_matches else 0
        
        return {
            "multi_branch_count": multi_branch_count,
            "single_branch_count": single_branch_count,
            "avg_accept_rate": avg_accept_rate,
            "avg_accept_len": avg_accept_len,
            "avg_max_branches": avg_max_branches,
        }
    except:
        return {}

def main():
    print("=" * 70)
    print("MBPP 代码数据集 Suffix Decoding 实验验证")
    print("=" * 70)
    
    prompts = load_mbpp_data(num_samples=100)
    all_results = []
    port = 31001
    
    # 实验配置
    configs = [
        {"name": "no_spec", "spec_algorithm": "NONE"},
        {"name": "suffix_single", "spec_algorithm": "SUFFIX", "use_tree_spec": False},
        {"name": "suffix_multi_3", "spec_algorithm": "SUFFIX", "use_tree_spec": True, "branch_factor": 3},
        {"name": "suffix_multi_5", "spec_algorithm": "SUFFIX", "use_tree_spec": True, "branch_factor": 5},
        {"name": "draft_8", "spec_algorithm": "SUFFIX", "use_tree_spec": True, "branch_factor": 3, "num_draft_tokens": 8},
    ]
    
    for cfg in configs:
        print(f"\n{'='*50}")
        print(f">>> 测试: {cfg['name']}")
        print(f"{'='*50}")
        proc, log = start_server(port, cfg)
        
        try:
            if not wait_health(port):
                print("服务器启动失败!")
                continue
            time.sleep(10)
            
            metrics = run_experiment(port, prompts, max_tokens=256)
            stats = get_server_stats(log)
            
            all_results.append({**cfg, **metrics, **stats})
            
            print(f"\n结果:")
            print(f"  输出吞吐量: {metrics['output_throughput']:.2f} tokens/s")
            print(f"  总输出tokens: {metrics['total_output_tokens']}")
            print(f"  平均延迟: {metrics['avg_latency']:.2f}s")
            if stats:
                print(f"  平均接受率: {stats.get('avg_accept_rate', 0):.2%}")
                print(f"  平均接受长度: {stats.get('avg_accept_len', 0):.2f}")
                print(f"  多分支次数: {stats.get('multi_branch_count', 0)}")
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            stop_server(proc)
            os.system(f"pkill -9 -f 'sglang.*--port {port}' 2>/dev/null")
            time.sleep(5)
        port += 1
    
    # 汇总结果
    print("\n" + "=" * 70)
    print("实验结果汇总 (MBPP 代码数据集)")
    print("=" * 70)
    
    print(f"\n{'配置':<20} {'吞吐量':<15} {'延迟(s)':<12} {'接受率':<12} {'多分支次数':<12}")
    print("-" * 71)
    for r in all_results:
        print(f"{r['name']:<20} {r.get('output_throughput', 0):<15.2f} {r.get('avg_latency', 0):<12.2f} {r.get('avg_accept_rate', 0):<12.1%} {r.get('multi_branch_count', 0)}")
    
    # 计算加速比
    if len(all_results) >= 2:
        baseline = all_results[0].get('output_throughput', 0)
        print(f"\n相对 no_spec 加速比:")
        for r in all_results[1:]:
            speedup = r.get('output_throughput', 0) / baseline if baseline else 0
            print(f"  {r['name']}: {speedup:.2f}x")
    
    # 保存
    result_file = f"{LOG_DIR}/mbpp_results_{get_ts()}.json"
    summary = [{k: v for k, v in r.items() if k != 'results'} for r in all_results]
    with open(result_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n结果已保存: {result_file}")

if __name__ == "__main__":
    main()
