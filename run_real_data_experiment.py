#!/usr/bin/env python3
"""
使用真实电商数据测试 SGLang Suffix Decoding
数据来源: /wxyworkspace/2025-12-10_1_Snippet 1_29337771.csv
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

# ============ 配置 ============
DATA_FILE = "/wxyworkspace/2025-12-10_1_Snippet 1_29337771.csv"
MODEL_PATH = "/wxyworkspace/Qwen2.5-VL-32B-Instruct"
LOG_DIR = "/wxyworkspace/experiment_logs"
os.makedirs(LOG_DIR, exist_ok=True)

def get_ts():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# ============ 数据加载 ============
def load_real_prompts(num_samples: int = 50) -> List[Dict[str, Any]]:
    """从CSV加载真实请求数据"""
    prompts = []
    
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # 随机采样
    random.seed(42)
    sampled = random.sample(rows, min(num_samples, len(rows)))
    
    for row in sampled:
        try:
            data = json.loads(row['request_response'])
            req = json.loads(data['request_body'])
            messages = req.get('messages', [])
            
            # 提取 system + user prompt
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
        except Exception as e:
            continue
    
    print(f"加载了 {len(prompts)} 条真实 prompt")
    
    # 打印长度分布
    lengths = [p['full_prompt_len'] for p in prompts]
    print(f"Prompt 长度分布: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}")
    
    return prompts

# ============ 服务器管理 ============
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
    env["SUFFIX_DEBUG_TREE"] = "0"  # 关闭调试输出
    
    log_file = f"{LOG_DIR}/server_{get_ts()}.log"
    print(f"启动服务器: port={port}, config={config}")
    
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    
    return proc, log_file

def wait_health(port: int, timeout: int = 180) -> bool:
    """等待服务器就绪"""
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
    """停止服务器"""
    print("停止服务器...")
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except:
        proc.kill()
    time.sleep(5)

# ============ 测试执行 ============
def send_request(port: int, prompt: Dict[str, Any], max_tokens: int = 256) -> Dict[str, Any]:
    """发送单个请求"""
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
        "temperature": 0.0,  # 使用确定性输出
    }
    
    start = time.time()
    try:
        resp = requests.post(url, headers=headers, json=data, timeout=120)
        latency = time.time() - start
        
        result = resp.json()
        output_tokens = result.get("usage", {}).get("completion_tokens", 0)
        input_tokens = result.get("usage", {}).get("prompt_tokens", 0)
        
        return {
            "latency": latency,
            "output_tokens": output_tokens,
            "input_tokens": input_tokens,
            "throughput": output_tokens / latency if latency > 0 else 0,
            "success": True,
        }
    except Exception as e:
        return {
            "latency": time.time() - start,
            "output_tokens": 0,
            "input_tokens": 0,
            "throughput": 0,
            "success": False,
            "error": str(e),
        }

def run_experiment(port: int, prompts: List[Dict], max_tokens: int = 256, warmup: int = 3) -> Dict[str, Any]:
    """运行实验"""
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
    test_start = time.time()
    
    for i, prompt in enumerate(prompts):
        r = send_request(port, prompt, max_tokens=max_tokens)
        results.append(r)
        
        if r['success']:
            total_input_tokens += r['input_tokens']
            total_output_tokens += r['output_tokens']
        
        if (i + 1) % 10 == 0:
            print(f"  完成 {i+1}/{len(prompts)} 请求")
    
    total_time = time.time() - test_start
    success_count = sum(1 for r in results if r['success'])
    
    return {
        "total_requests": len(prompts),
        "success_count": success_count,
        "total_time": total_time,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_throughput": total_input_tokens / total_time if total_time > 0 else 0,
        "output_throughput": total_output_tokens / total_time if total_time > 0 else 0,
        "avg_latency": sum(r['latency'] for r in results) / len(results) if results else 0,
        "results": results,
    }

# ============ 主实验 ============
def main():
    print("=" * 70)
    print("SGLang Suffix Decoding 真实数据对比实验")
    print(f"数据来源: {DATA_FILE}")
    print("=" * 70)
    
    # 加载真实数据
    prompts = load_real_prompts(num_samples=50)
    
    if not prompts:
        print("无法加载数据，退出")
        return
    
    all_results = []
    port = 31001
    
    # ========== 实验1: 基线对比 ==========
    print("\n" + "=" * 70)
    print("实验1: 基线对比 (无推测 vs SUFFIX单分支 vs SUFFIX多分支)")
    print("=" * 70)
    
    baseline_configs = [
        {"name": "no_spec", "spec_algorithm": "NONE", "use_tree_spec": False},
        {"name": "suffix_single", "spec_algorithm": "SUFFIX", "use_tree_spec": False, "branch_factor": 1},
        {"name": "suffix_multi_3", "spec_algorithm": "SUFFIX", "use_tree_spec": True, "branch_factor": 3},
        {"name": "suffix_multi_5", "spec_algorithm": "SUFFIX", "use_tree_spec": True, "branch_factor": 5},
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
            all_results.append({**cfg, **metrics})
            
            print(f"    输出吞吐量: {metrics['output_throughput']:.2f} tokens/s")
            print(f"    输入吞吐量: {metrics['input_throughput']:.2f} tokens/s")
            print(f"    平均延迟: {metrics['avg_latency']:.2f}s")
            print(f"    成功率: {metrics['success_count']}/{metrics['total_requests']}")
            
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
        finally:
            stop_server(proc)
            os.system(f"pkill -9 -f 'sglang.*--port {port}' 2>/dev/null")
            time.sleep(5)
        
        port += 1
    
    # ========== 实验2: 不同 num_draft_tokens ==========
    print("\n" + "=" * 70)
    print("实验2: 不同 num_draft_tokens 对比 (多分支)")
    print("=" * 70)
    
    draft_configs = [
        {"name": "draft_2", "spec_algorithm": "SUFFIX", "use_tree_spec": True, "branch_factor": 3, "num_draft_tokens": 2},
        {"name": "draft_4", "spec_algorithm": "SUFFIX", "use_tree_spec": True, "branch_factor": 3, "num_draft_tokens": 4},
        {"name": "draft_6", "spec_algorithm": "SUFFIX", "use_tree_spec": True, "branch_factor": 3, "num_draft_tokens": 6},
        {"name": "draft_8", "spec_algorithm": "SUFFIX", "use_tree_spec": True, "branch_factor": 3, "num_draft_tokens": 8},
    ]
    
    for cfg in draft_configs:
        print(f"\n>>> 测试: {cfg['name']}")
        proc, log = start_server(port, cfg)
        
        try:
            if not wait_health(port):
                print("服务器启动失败!")
                continue
            time.sleep(10)
            
            metrics = run_experiment(port, prompts, max_tokens=256)
            all_results.append({**cfg, **metrics})
            
            print(f"    输出吞吐量: {metrics['output_throughput']:.2f} tokens/s")
            print(f"    平均延迟: {metrics['avg_latency']:.2f}s")
            
        except Exception as e:
            print(f"错误: {e}")
        finally:
            stop_server(proc)
            os.system(f"pkill -9 -f 'sglang.*--port {port}' 2>/dev/null")
            time.sleep(5)
        
        port += 1
    
    # ========== 实验3: 不同 max_tree_depth ==========
    print("\n" + "=" * 70)
    print("实验3: 不同 max_tree_depth 对比 (多分支)")
    print("=" * 70)
    
    depth_configs = [
        {"name": "depth_6", "spec_algorithm": "SUFFIX", "use_tree_spec": True, "branch_factor": 3, "max_tree_depth": 6},
        {"name": "depth_12", "spec_algorithm": "SUFFIX", "use_tree_spec": True, "branch_factor": 3, "max_tree_depth": 12},
        {"name": "depth_24", "spec_algorithm": "SUFFIX", "use_tree_spec": True, "branch_factor": 3, "max_tree_depth": 24},
    ]
    
    for cfg in depth_configs:
        print(f"\n>>> 测试: {cfg['name']}")
        proc, log = start_server(port, cfg)
        
        try:
            if not wait_health(port):
                print("服务器启动失败!")
                continue
            time.sleep(10)
            
            metrics = run_experiment(port, prompts, max_tokens=256)
            all_results.append({**cfg, **metrics})
            
            print(f"    输出吞吐量: {metrics['output_throughput']:.2f} tokens/s")
            print(f"    平均延迟: {metrics['avg_latency']:.2f}s")
            
        except Exception as e:
            print(f"错误: {e}")
        finally:
            stop_server(proc)
            os.system(f"pkill -9 -f 'sglang.*--port {port}' 2>/dev/null")
            time.sleep(5)
        
        port += 1
    
    # ========== 实验4: 长短文本对比 ==========
    print("\n" + "=" * 70)
    print("实验4: 长短文本对比 (使用多分支 suffix_multi_3)")
    print("=" * 70)
    
    # 按长度分组
    sorted_prompts = sorted(prompts, key=lambda x: x['full_prompt_len'])
    short_prompts = sorted_prompts[:25]  # 短文本
    long_prompts = sorted_prompts[25:]   # 长文本
    
    print(f"短文本组: {len(short_prompts)} prompts, avg_len={sum(p['full_prompt_len'] for p in short_prompts)/len(short_prompts):.0f}")
    print(f"长文本组: {len(long_prompts)} prompts, avg_len={sum(p['full_prompt_len'] for p in long_prompts)/len(long_prompts):.0f}")
    
    cfg = {"name": "suffix_multi_3", "spec_algorithm": "SUFFIX", "use_tree_spec": True, "branch_factor": 3}
    
    # 短文本测试
    print("\n>>> 短文本测试")
    proc, log = start_server(port, cfg)
    try:
        if wait_health(port):
            time.sleep(10)
            metrics_short = run_experiment(port, short_prompts, max_tokens=256)
            all_results.append({"name": "short_text_multi", **cfg, **metrics_short})
            print(f"    吞吐量: {metrics_short['output_throughput']:.2f} tokens/s")
    except Exception as e:
        print(f"错误: {e}")
    finally:
        stop_server(proc)
        os.system(f"pkill -9 -f 'sglang.*--port {port}' 2>/dev/null")
        time.sleep(5)
    port += 1
    
    # 长文本测试
    print("\n>>> 长文本测试")
    proc, log = start_server(port, cfg)
    try:
        if wait_health(port):
            time.sleep(10)
            metrics_long = run_experiment(port, long_prompts, max_tokens=256)
            all_results.append({"name": "long_text_multi", **cfg, **metrics_long})
            print(f"    吞吐量: {metrics_long['output_throughput']:.2f} tokens/s")
    except Exception as e:
        print(f"错误: {e}")
    finally:
        stop_server(proc)
        os.system(f"pkill -9 -f 'sglang.*--port {port}' 2>/dev/null")
        time.sleep(5)
    
    # ========== 结果汇总 ==========
    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)
    
    print("\n实验1: 基线对比")
    print(f"{'配置':<20} {'输出吞吐量':<18} {'输入吞吐量':<18} {'平均延迟':<12} {'相对加速':<10}")
    print("-" * 76)
    baseline = None
    for r in all_results:
        if r['name'] in ['no_spec', 'suffix_single', 'suffix_multi_3', 'suffix_multi_5']:
            if baseline is None and r['name'] == 'no_spec':
                baseline = r.get('output_throughput', 0)
            speedup = r.get('output_throughput', 0) / baseline if baseline else 1.0
            print(f"{r['name']:<20} {r.get('output_throughput', 0):<18.2f} {r.get('input_throughput', 0):<18.2f} {r.get('avg_latency', 0):<12.2f} {speedup:<10.2f}x")
    
    print("\n实验2: 不同 num_draft_tokens")
    print(f"{'配置':<15} {'输出吞吐量':<18} {'平均延迟':<12}")
    print("-" * 45)
    for r in all_results:
        if r['name'].startswith('draft'):
            print(f"{r['name']:<15} {r.get('output_throughput', 0):<18.2f} {r.get('avg_latency', 0):<12.2f}")
    
    print("\n实验3: 不同 max_tree_depth")
    print(f"{'配置':<15} {'输出吞吐量':<18} {'平均延迟':<12}")
    print("-" * 45)
    for r in all_results:
        if r['name'].startswith('depth'):
            print(f"{r['name']:<15} {r.get('output_throughput', 0):<18.2f} {r.get('avg_latency', 0):<12.2f}")
    
    print("\n实验4: 长短文本对比")
    print(f"{'配置':<20} {'输出吞吐量':<18} {'平均延迟':<12}")
    print("-" * 50)
    for r in all_results:
        if r['name'] in ['short_text_multi', 'long_text_multi']:
            print(f"{r['name']:<20} {r.get('output_throughput', 0):<18.2f} {r.get('avg_latency', 0):<12.2f}")
    
    # 保存结果
    result_file = f"{LOG_DIR}/real_data_results_{get_ts()}.json"
    # 移除 results 列表（太大）
    summary = [{k: v for k, v in r.items() if k != 'results'} for r in all_results]
    with open(result_file, "w", encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n完整结果已保存: {result_file}")

if __name__ == "__main__":
    main()
