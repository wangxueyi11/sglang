#!/usr/bin/env python3
"""
Suffix Decoding 完整实验
比较单分支 vs 多分支 (不同分支因子) 的效果
包含完整指标：吞吐量、延迟、接受率、接受长度

实验结果将保存到 /wxyworkspace/suffix_experiment_results.json
"""

import os
import json
import time
import subprocess
import signal
import requests
import pandas as pd
from datetime import datetime

# 配置
TARGET_MODEL = "/wxyworkspace/Qwen2.5-VL-32B-Instruct"
DATASET_PATH = "/wxyworkspace/2025-12-10_1_Snippet 1_29337771.csv"
SERVER_PORT = 30000
CUDA_DEVICES = "0,1,2,3"

# 实验配置
EXPERIMENT_CONFIGS = [
    {
        "name": "baseline-no-spec",
        "use_suffix": False,
        "use_tree_spec": False,
        "max_branch_factor": 1,
        "description": "基准测试：无 speculation",
    },
    {
        "name": "suffix-single-path",
        "use_suffix": True,
        "use_tree_spec": False,  # 单分支模式
        "max_branch_factor": 1,
        "description": "SUFFIX 单分支模式",
    },
    {
        "name": "suffix-tree-branch-3",
        "use_suffix": True,
        "use_tree_spec": True,   # 多分支模式
        "max_branch_factor": 3,
        "description": "SUFFIX 多分支模式 (branch=3)",
    },
    {
        "name": "suffix-tree-branch-5",
        "use_suffix": True,
        "use_tree_spec": True,
        "max_branch_factor": 5,
        "description": "SUFFIX 多分支模式 (branch=5)",
    },
]

NUM_WARMUP = 5
NUM_EVAL = 20
MAX_PROMPT_LEN = 2000
MAX_TOKENS = 200


def load_dataset(path: str) -> list:
    """加载数据集"""
    df = pd.read_csv(path)
    samples = []
    
    for idx, row in df.iterrows():
        try:
            sample = json.loads(row['request_response'])
            request_body = json.loads(sample['request_body'])
            messages = request_body.get('messages', [])
            user_msg = ""
            for msg in messages:
                if msg.get('role') == 'user':
                    user_msg = msg.get('content', '')
                    break
            if user_msg:
                samples.append(user_msg)
        except:
            continue
    
    return samples


def build_server_cmd(config: dict) -> list:
    """构建服务器启动命令"""
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", TARGET_MODEL,
        "--host", "localhost",
        "--port", str(SERVER_PORT),
        "--tp", "4",
        "--mem-fraction-static", "0.8",
        "--context-length", "4096",
        "--max-running-requests", "8",
        "--disable-cuda-graph",  # SUFFIX 需要禁用 CUDA graph
    ]
    
    if config["use_suffix"]:
        cmd.extend([
            "--speculative-algorithm", "SUFFIX",
            "--speculative-num-draft-tokens", "16",
            "--speculative-suffix-max-tree-depth", "32",
            "--speculative-suffix-min-token-prob", "0.01",
        ])
        
        if config["use_tree_spec"]:
            cmd.append("--speculative-suffix-use-tree-spec")
            cmd.extend(["--speculative-suffix-max-branch-factor", str(config["max_branch_factor"])])
    
    return cmd


def start_server(config: dict) -> subprocess.Popen:
    """启动服务器"""
    cmd = build_server_cmd(config)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICES
    env["SGLANG_DISABLE_CUDNN_CHECK"] = "1"
    
    print(f"启动服务器: {' '.join(cmd[-5:])}")
    
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        cwd="/wxyworkspace",
    )
    
    return proc


def wait_for_server(timeout: int = 180) -> bool:
    """等待服务器启动"""
    print(f"等待服务器启动 (timeout: {timeout}s)...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(f"http://localhost:{SERVER_PORT}/health", timeout=5)
            if resp.status_code == 200:
                print("服务器已就绪!")
                return True
        except:
            pass
        time.sleep(5)
    
    print("服务器启动超时!")
    return False


def stop_server(proc: subprocess.Popen = None):
    """停止服务器"""
    try:
        subprocess.run(["pkill", "-9", "-f", "sglang.launch_server"], 
                       capture_output=True, timeout=10)
    except:
        pass
    time.sleep(5)


def get_speculative_metrics() -> dict:
    """从 /v1/loads 端点获取 speculative decoding 指标"""
    try:
        resp = requests.get(f"http://localhost:{SERVER_PORT}/v1/loads", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            # 获取第一个 DP rank 的指标
            if isinstance(data, list) and len(data) > 0:
                metrics = data[0]
                speculative = metrics.get("speculative", {})
                return {
                    "accept_length": speculative.get("accept_length", 0),
                    "accept_rate": speculative.get("accept_rate", 0),
                    "draft_length": speculative.get("draft_length", 0),
                }
    except Exception as e:
        print(f"获取 speculative 指标失败: {e}")
    
    return {"accept_length": 0, "accept_rate": 0, "draft_length": 0}


def send_request(prompt: str) -> dict:
    """发送请求"""
    url = f"http://localhost:{SERVER_PORT}/v1/chat/completions"
    
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt[:MAX_PROMPT_LEN]}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
    }
    
    start_time = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=120)
        latency = time.time() - start_time
        
        if resp.status_code == 200:
            data = resp.json()
            output_tokens = data.get("usage", {}).get("completion_tokens", 0)
            return {
                "success": True,
                "output_tokens": output_tokens,
                "latency": latency,
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {resp.status_code}",
                "latency": latency,
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "latency": time.time() - start_time,
        }


def run_experiment(config: dict, samples: list) -> dict:
    """运行单个实验配置"""
    print(f"\n{'='*60}")
    print(f"实验配置: {config['name']}")
    print(f"描述: {config['description']}")
    print(f"{'='*60}")
    
    # 启动服务器
    server_proc = start_server(config)
    
    # 等待服务器启动
    if not wait_for_server():
        stop_server(server_proc)
        return {
            "config": config["name"],
            "success": False,
            "error": "Server failed to start",
        }
    
    # Warmup
    print(f"\nWarmup ({NUM_WARMUP} requests)...")
    for i in range(NUM_WARMUP):
        result = send_request(samples[i])
        status = "OK" if result["success"] else f"FAILED: {result.get('error', '')}"
        print(f"  Warmup {i+1}: {status}")
    
    # 记录开始时的 speculative 指标
    start_metrics = get_speculative_metrics()
    
    # 评估
    print(f"\n评估 ({NUM_EVAL} requests)...")
    total_tokens = 0
    total_time = 0
    success_count = 0
    latencies = []
    
    for i in range(NUM_WARMUP, NUM_WARMUP + NUM_EVAL):
        result = send_request(samples[i])
        latencies.append(result["latency"])
        
        if result["success"]:
            success_count += 1
            total_tokens += result["output_tokens"]
            total_time += result["latency"]
            print(f"  Request {i-NUM_WARMUP+1}: {result['output_tokens']} tokens, {result['latency']:.2f}s")
        else:
            print(f"  Request {i-NUM_WARMUP+1}: FAILED - {result.get('error', '')[:50]}")
    
    # 记录结束时的 speculative 指标
    end_metrics = get_speculative_metrics()
    
    # 计算增量指标
    delta_accept_length = end_metrics["accept_length"] - start_metrics["accept_length"]
    delta_draft_length = end_metrics["draft_length"] - start_metrics["draft_length"]
    accept_rate = end_metrics.get("accept_rate", 0)
    
    # 停止服务器
    print("\n停止服务器...")
    stop_server(server_proc)
    
    # 计算指标
    throughput = total_tokens / total_time if total_time > 0 else 0
    avg_latency = total_time / success_count if success_count > 0 else 0
    
    return {
        "config": config["name"],
        "use_tree_spec": config["use_tree_spec"],
        "max_branch_factor": config["max_branch_factor"],
        "success": True,
        "total_requests": NUM_EVAL,
        "successful_requests": success_count,
        "total_output_tokens": total_tokens,
        "total_time": total_time,
        "throughput": throughput,
        "avg_latency": avg_latency,
        "speculative": {
            "accept_length": delta_accept_length,
            "draft_length": delta_draft_length,
            "accept_rate": accept_rate,
            "avg_accept_length": delta_accept_length / success_count if success_count > 0 else 0,
        },
    }


def main():
    print("="*70)
    print("Suffix Decoding 完整实验")
    print("="*70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型: {TARGET_MODEL}")
    print(f"数据集: {DATASET_PATH}")
    print(f"Warmup 请求: {NUM_WARMUP}")
    print(f"评估请求: {NUM_EVAL}")
    print(f"最大 prompt 长度: {MAX_PROMPT_LEN}")
    print(f"最大输出 tokens: {MAX_TOKENS}")
    
    # 加载数据集
    print("\n加载数据集...")
    samples = load_dataset(DATASET_PATH)
    print(f"加载 {len(samples)} 个样本")
    
    # 运行实验
    all_results = []
    
    for config in EXPERIMENT_CONFIGS:
        result = run_experiment(config, samples)
        all_results.append(result)
        
        # 打印中间结果
        if result["success"]:
            print(f"\n结果 - {config['name']}:")
            print(f"  成功请求: {result['successful_requests']}/{result['total_requests']}")
            print(f"  总输出 tokens: {result['total_output_tokens']}")
            print(f"  总时间: {result['total_time']:.2f}s")
            print(f"  吞吐量: {result['throughput']:.2f} tokens/sec")
            print(f"  平均延迟: {result['avg_latency']:.2f}s")
            if result.get("speculative"):
                spec = result["speculative"]
                print(f"  接受长度: {spec['accept_length']}")
                print(f"  平均接受长度: {spec['avg_accept_length']:.2f}")
                print(f"  接受率: {spec['accept_rate']:.2%}")
        else:
            print(f"\n结果 - {config['name']}: FAILED - {result.get('error', '')}")
        
        time.sleep(10)
    
    # 计算改进率
    baseline_throughput = 0
    for r in all_results:
        if r["success"] and "baseline" in r["config"]:
            baseline_throughput = r["throughput"]
            break
    
    for r in all_results:
        if r["success"] and baseline_throughput > 0:
            r["throughput_improvement"] = (r["throughput"] / baseline_throughput - 1) * 100
        else:
            r["throughput_improvement"] = 0
    
    # 汇总结果
    print("\n" + "="*70)
    print("实验结果汇总")
    print("="*70)
    
    print(f"\n{'配置':<30} {'吞吐量':<15} {'延迟':<10} {'接受率':<10} {'接受长度':<10} {'提升':<10}")
    print("-" * 85)
    
    for r in all_results:
        if r["success"]:
            spec = r.get("speculative", {})
            imp = f"{r.get('throughput_improvement', 0):+.1f}%"
            print(f"{r['config']:<30} {r['throughput']:<15.2f} {r['avg_latency']:<10.2f} "
                  f"{spec.get('accept_rate', 0):<10.2%} {spec.get('avg_accept_length', 0):<10.2f} {imp:<10}")
        else:
            print(f"{r['config']:<30} {'FAILED':<15}")
    
    # 保存结果
    results_data = {
        "experiment_time": datetime.now().isoformat(),
        "config": {
            "model": TARGET_MODEL,
            "dataset": DATASET_PATH,
            "num_warmup": NUM_WARMUP,
            "num_eval": NUM_EVAL,
            "max_prompt_len": MAX_PROMPT_LEN,
            "max_tokens": MAX_TOKENS,
        },
        "results": all_results,
    }
    
    output_path = "/wxyworkspace/suffix_experiment_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_path}")
    
    return all_results


if __name__ == "__main__":
    main()
