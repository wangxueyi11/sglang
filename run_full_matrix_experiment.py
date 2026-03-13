#!/usr/bin/env python3
"""
完整矩阵对比实验：不同 draft_tokens 和 branch_factor 组合

实验配置：
- draft_tokens: [4, 8, 12, 16]
- branch_factor: [1, 2, 3, 5]
- baseline (无推测解码)

评价指标：
- 吞吐量
- 加速比
- TTFT
- 解码时间
- 接受长度
- 接受率
- 平均树节点数
- 平均分支数
"""

import argparse
import json
import os
import subprocess
import time
import csv
import re
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests

# 实验配置
DRAFT_TOKENS_OPTIONS = [4, 8, 12, 16]
BRANCH_FACTOR_OPTIONS = [1, 2, 3, 5]
BASE_PORT = 30001
SAMPLES = 30  # 每个配置的样本数
DATASET_PATH = "/wxyworkspace/2025-12-10_1_Snippet 1_29337771.csv"
MODEL_PATH = "/wxyworkspace/Qwen2.5-VL-32B-Instruct"
LOG_DIR = "/tmp/sglang_matrix_exp"


def load_dataset(max_samples: int = 50) -> List[Dict]:
    """加载真实数据集"""
    samples = []
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_samples:
                break
            try:
                request_response = json.loads(row['request_response'])
                request_body = json.loads(request_response['request_body'])
                messages = request_body.get('messages', [])
                for msg in messages:
                    if msg.get('role') == 'user':
                        samples.append({'prompt': msg.get('content', '')[:2000]})
                        break
            except:
                continue
    return samples


def start_server(
    port: int,
    draft_tokens: int,
    branch_factor: int,
    log_file: str,
    use_tree_spec: bool = True,
) -> Optional[subprocess.Popen]:
    """启动 sglang 服务器"""
    
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", MODEL_PATH,
        "--port", str(port),
        "--tp", "4",
        "--speculative-algorithm", "SUFFIX",
        "--speculative-num-steps", "5",
        "--speculative-num-draft-tokens", str(draft_tokens),
        "--speculative-suffix-max-tree-depth", "8",
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--trust-remote-code",
        "--log-level", "info",
    ]
    
    if use_tree_spec and branch_factor > 1:
        cmd.extend([
            "--speculative-suffix-use-tree-spec",
            "--speculative-suffix-max-branch-factor", str(branch_factor),
        ])
    
    env = os.environ.copy()
    env["SGLANG_DISABLE_CUDNN_CHECK"] = "1"
    
    with open(log_file, 'w') as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )
    
    return proc


def start_baseline_server(port: int, log_file: str) -> Optional[subprocess.Popen]:
    """启动基线服务器（无推测解码）"""
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", MODEL_PATH,
        "--port", str(port),
        "--tp", "4",
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--trust-remote-code",
        "--log-level", "info",
    ]
    
    env = os.environ.copy()
    env["SGLANG_DISABLE_CUDNN_CHECK"] = "1"
    
    with open(log_file, 'w') as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,
        )
    
    return proc


def wait_for_server(port: int, timeout: int = 60) -> bool:
    """等待服务器就绪"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=2)
            if resp.status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    return False


def run_benchmark(port: int, samples: List[Dict]) -> Dict:
    """运行基准测试 - 使用流式请求获取准确的TTFT和解码时间"""
    url = f"http://localhost:{port}/v1/chat/completions"
    
    results = {
        "total_tokens": 0,
        "total_time": 0,
        "total_decode_time": 0,
        "successful_requests": 0,
        "ttfts": [],
        "decode_times": [],
    }
    
    for sample in samples:
        data = {
            "model": MODEL_PATH,
            "messages": [{"role": "user", "content": sample['prompt']}],
            "max_tokens": 100,
            "stream": True,
        }
        
        try:
            start = time.time()
            response = requests.post(url, json=data, timeout=120, stream=True)
            
            ttft = None
            token_count = 0
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data_str)
                            if 'choices' in chunk and chunk['choices']:
                                delta = chunk['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    if ttft is None:
                                        ttft = time.time() - start
                                    # 更准确地计数：按字符数估算（中文约1.5字符/token，英文约4字符/token）
                                    token_count += max(1, len(content) // 2)
                                # 检查 usage 信息（某些实现在最后一个chunk中返回）
                                usage = chunk.get('usage', {})
                                if usage.get('completion_tokens'):
                                    token_count = usage['completion_tokens']
                        except:
                            pass
            
            e2e = time.time() - start
            
            if ttft is not None and token_count > 0:
                results["ttfts"].append(ttft)
                results["decode_times"].append(e2e - ttft)
                results["total_decode_time"] += e2e - ttft
                results["total_tokens"] += token_count
                results["total_time"] += e2e
                results["successful_requests"] += 1
            
        except Exception as e:
            print(f"    请求错误: {e}")
    
    return results


def extract_metrics_from_log(log_file: str) -> Tuple[float, float, float, float]:
    """从日志提取接受率和树结构信息"""
    with open(log_file, 'r') as f:
        content = f.read()
    
    # 接受率和接受长度
    accept_lens = [float(x) for x in re.findall(r'accept len: ([\d.]+)', content)]
    accept_rates = [float(x) for x in re.findall(r'accept rate: ([\d.]+)', content)]
    
    avg_accept_len = sum(accept_lens) / len(accept_lens) if accept_lens else 0
    avg_accept_rate = sum(accept_rates) / len(accept_rates) if accept_rates else 0
    
    # 树结构
    tree_matches = re.findall(r'tokens=(\d+), roots=(\d+)', content)
    if tree_matches:
        avg_tokens = sum(int(t) for t, r in tree_matches) / len(tree_matches)
        avg_roots = sum(int(r) for t, r in tree_matches) / len(tree_matches)
    else:
        avg_tokens = 0
        avg_roots = 0
    
    return avg_accept_len, avg_accept_rate, avg_tokens, avg_roots


def stop_server(proc: subprocess.Popen):
    """停止服务器"""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        time.sleep(2)
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except:
        pass


def run_single_experiment(
    config_name: str,
    port: int,
    samples: List[Dict],
    draft_tokens: Optional[int] = None,
    branch_factor: Optional[int] = None,
    is_baseline: bool = False,
) -> Dict:
    """运行单个实验配置"""
    
    log_file = os.path.join(LOG_DIR, f"{config_name}.log")
    
    print(f"\n{'='*60}")
    print(f"实验配置: {config_name}")
    print(f"{'='*60}")
    
    # 启动服务器
    print(f"启动服务器...")
    if is_baseline:
        proc = start_baseline_server(port, log_file)
    else:
        proc = start_server(port, draft_tokens, branch_factor, log_file)
    
    # 等待就绪
    print(f"等待服务器就绪...")
    if not wait_for_server(port):
        print(f"服务器启动失败!")
        stop_server(proc)
        return {"error": "server timeout"}
    
    print(f"服务器就绪，开始测试...")
    
    # 运行测试
    results = run_benchmark(port, samples)
    
    # 停止服务器
    print(f"停止服务器...")
    stop_server(proc)
    
    # 提取日志指标
    time.sleep(2)
    accept_len, accept_rate, avg_tokens, avg_roots = extract_metrics_from_log(log_file)
    
    # 计算指标
    throughput = results["total_tokens"] / results["total_decode_time"] if results["total_decode_time"] > 0 else 0
    avg_ttft = sum(results["ttfts"]) / len(results["ttfts"]) if results["ttfts"] else 0
    avg_decode = sum(results["decode_times"]) / len(results["decode_times"]) if results["decode_times"] else 0
    avg_latency = results["total_time"] / results["successful_requests"] if results["successful_requests"] > 0 else 0
    
    result = {
        "config_name": config_name,
        "draft_tokens": draft_tokens,
        "branch_factor": branch_factor,
        "total_tokens": results["total_tokens"],
        "successful_requests": results["successful_requests"],
        "throughput": round(throughput, 2),
        "avg_ttft": round(avg_ttft, 4),
        "avg_decode_time": round(avg_decode, 4),
        "avg_latency": round(avg_latency, 4),
        "accept_len": round(accept_len, 2) if accept_len > 0 else None,
        "accept_rate": round(accept_rate, 4) if accept_rate > 0 else None,
        "avg_tree_tokens": round(avg_tokens, 2) if avg_tokens > 0 else None,
        "avg_tree_roots": round(avg_roots, 2) if avg_roots > 0 else None,
    }
    
    print(f"\n{config_name} 结果:")
    print(f"  吞吐量: {result['throughput']} tokens/sec")
    print(f"  解码时间: {result['avg_decode_time']} sec")
    if accept_len > 0:
        print(f"  接受长度: {result['accept_len']}")
        print(f"  接受率: {result['accept_rate']}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="完整矩阵对比实验")
    parser.add_argument("--samples", type=int, default=30, help="每个配置的样本数")
    parser.add_argument("--output", type=str, default="matrix_results.json", help="输出文件")
    parser.add_argument("--draft-tokens", type=int, nargs="+", default=[4, 8, 12, 16])
    parser.add_argument("--branch-factors", type=int, nargs="+", default=[1, 2, 3, 5])
    parser.add_argument("--skip-baseline", action="store_true", help="跳过基线测试")
    args = parser.parse_args()
    
    global SAMPLES
    SAMPLES = args.samples
    
    print("=" * 80)
    print("完整矩阵对比实验")
    print("=" * 80)
    print(f"时间: {datetime.now()}")
    print(f"数据集: {DATASET_PATH}")
    print(f"样本数: {SAMPLES}")
    print(f"模型: {MODEL_PATH}")
    print(f"draft_tokens 选项: {args.draft_tokens}")
    print(f"branch_factor 选项: {args.branch_factors}")
    print("=" * 80)
    
    # 加载数据
    samples = load_dataset(SAMPLES)
    print(f"加载了 {len(samples)} 个样本")
    
    results = []
    port = BASE_PORT
    
    # 运行基线测试
    if not args.skip_baseline:
        result = run_single_experiment(
            "baseline",
            port,
            samples,
            is_baseline=True,
        )
        result["description"] = "无推测解码（基线）"
        results.append(result)
        save_results(results, args.output)
    
    # 运行实验矩阵
    for draft_tokens in args.draft_tokens:
        for branch_factor in args.branch_factors:
            config_name = f"draft{draft_tokens}_branch{branch_factor}"
            port = BASE_PORT + len(results)
            
            result = run_single_experiment(
                config_name,
                port,
                samples,
                draft_tokens=draft_tokens,
                branch_factor=branch_factor,
            )
            result["description"] = f"draft_tokens={draft_tokens}, branch={branch_factor}"
            results.append(result)
            save_results(results, args.output)
    
    # 打印最终结果
    print_final_results(results)


def save_results(results: List[Dict], output_file: str):
    """保存结果"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"结果已保存到: {output_file}")


def print_final_results(results: List[Dict]):
    """打印最终结果表格 - 突出平均接受长度作为核心指标"""
    
    baseline_throughput = None
    for r in results:
        if r.get("config_name") == "baseline":
            baseline_throughput = r.get("throughput", 0)
            break
    
    # 表1: 主要指标（吞吐量、加速比、接受长度）
    print("\n" + "=" * 80)
    print("主要指标对比（按吞吐量排序）")
    print("=" * 80)
    
    header = f"{'配置':<25} {'吞吐量':>10} {'加速比':>10} {'接受长度':>10} {'接受率':>10}"
    print(header)
    print("-" * 80)
    
    # 按吞吐量排序（排除 baseline）
    speculative_results = [r for r in results if r.get("config_name") != "baseline"]
    speculative_results.sort(key=lambda x: x.get("throughput", 0), reverse=True)
    
    for r in speculative_results:
        name = r.get("config_name", "N/A")
        throughput = r.get("throughput", 0)
        accept_len = r.get("accept_len") or 0
        accept_rate = r.get("accept_rate") or 0
        
        if baseline_throughput and baseline_throughput > 0:
            speedup = throughput / baseline_throughput
        else:
            speedup = 0
        
        print(f"{name:<25} {throughput:>10.1f} {speedup:>10.2f}x {accept_len:>10.2f} {accept_rate*100:>9.1f}%")
    
    print("=" * 80)
    
    # 表2: 矩阵视图
    print("\n" + "=" * 80)
    print("吞吐量矩阵 (tokens/sec) - draft_tokens × branch_factor")
    print("=" * 80)
    
    # 提取 draft 和 branch 选项
    draft_options = sorted(set(r.get("draft_tokens", 0) for r in speculative_results if r.get("draft_tokens")))
    branch_options = sorted(set(r.get("branch_factor", 0) for r in speculative_results if r.get("branch_factor")))
    
    # 打印表头
    header = f"{'draft\\branch':<12}"
    for b in branch_options:
        header += f"{'branch=' + str(b):>12}"
    print(header)
    print("-" * (12 + 12 * len(branch_options)))
    
    # 打印数据行
    for d in draft_options:
        row = f"{'draft=' + str(d):<12}"
        for b in branch_options:
            r = next((x for x in speculative_results 
                     if x.get("draft_tokens") == d and x.get("branch_factor") == b), None)
            if r:
                throughput = r.get("throughput", 0)
                speedup = throughput / baseline_throughput if baseline_throughput else 0
                row += f"{throughput:>8.1f} ({speedup:.2f}x)"
            else:
                row += f"{'N/A':>12}"
        print(row)
    
    print()
    
    # 表3: 平均接受长度矩阵
    print("=" * 80)
    print("平均接受长度矩阵 - draft_tokens × branch_factor")
    print("=" * 80)
    
    header = f"{'draft\\branch':<12}"
    for b in branch_options:
        header += f"{'branch=' + str(b):>12}"
    print(header)
    print("-" * (12 + 12 * len(branch_options)))
    
    for d in draft_options:
        row = f"{'draft=' + str(d):<12}"
        for b in branch_options:
            r = next((x for x in speculative_results 
                     if x.get("draft_tokens") == d and x.get("branch_factor") == b), None)
            if r and r.get("accept_len"):
                row += f"{r.get('accept_len'):>12.2f}"
            else:
                row += f"{'N/A':>12}"
        print(row)
    
    print()
    
    # 关键发现
    print("=" * 80)
    print("关键发现")
    print("=" * 80)
    
    if speculative_results:
        best = speculative_results[0]  # 已按吞吐量排序
        best_throughput = best.get("throughput", 0)
        best_speedup = best_throughput / baseline_throughput if baseline_throughput else 0
        best_accept_len = best.get("accept_len", 0)
        
        print(f"最佳配置: {best.get('config_name')}")
        print(f"  - 吞吐量: {best_throughput:.1f} tokens/sec ({best_speedup:.2f}x)")
        print(f"  - 平均接受长度: {best_accept_len:.2f}")
        
        # 分析多分支收益
        print("\n多分支收益分析:")
        for d in draft_options:
            single_branch = next((r for r in speculative_results 
                                 if r.get("draft_tokens") == d and r.get("branch_factor") == 1), None)
            multi_branch = [r for r in speculative_results 
                          if r.get("draft_tokens") == d and r.get("branch_factor", 1) > 1]
            
            if single_branch and multi_branch:
                single_throughput = single_branch.get("throughput", 0)
                best_multi = max(multi_branch, key=lambda x: x.get("throughput", 0))
                multi_throughput = best_multi.get("throughput", 0)
                improvement = (multi_throughput - single_throughput) / single_throughput * 100
                
                single_accept = single_branch.get("accept_len", 0)
                multi_accept = best_multi.get("accept_len", 0)
                accept_improvement = (multi_accept - single_accept) / single_accept * 100 if single_accept > 0 else 0
                
                print(f"  draft={d}: 多分支最佳吞吐量提升 {improvement:+.1f}%, "
                      f"接受长度提升 {accept_improvement:+.1f}% "
                      f"(branch={best_multi.get('branch_factor')})")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
