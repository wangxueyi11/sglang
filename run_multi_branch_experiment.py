#!/usr/bin/env python3
"""
多分支 Suffix Decoding 实验脚本

实验配置：
1. baseline: 无推测解码
2. single_path: 单分支模式
3. multi_branch_3: 多分支模式 (max_branch_factor=3)
4. multi_branch_5: 多分支模式 (max_branch_factor=5)

评价指标：
- 吞吐量 (tokens/sec)
- 加速比 (相对于baseline)
- 平均延迟 (sec)
- 接受长度
- 接受率
"""

import json
import csv
import requests
import time
import re
import subprocess
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# ==================== 配置 ====================
MODEL_PATH = "/wxyworkspace/Qwen2.5-VL-32B-Instruct"
DATASET_PATH = "/wxyworkspace/2025-12-10_1_Snippet 1_29337771.csv"
NUM_SAMPLES = 50  # 每个配置测试的样本数
MAX_TOKENS = 100  # 每个请求的最大token数
TP = 4  # Tensor Parallelism

# 服务器配置
BASE_PORT = 30000
SERVER_START_WAIT = 60  # 服务器启动等待时间(秒)

# 实验配置
EXPERIMENT_CONFIGS = {
    "baseline": {
        "speculative_algorithm": None,
        "use_tree_spec": False,
        "max_branch_factor": 1,
        "port": 30001,
        "description": "无推测解码（基线）"
    },
    "single_path": {
        "speculative_algorithm": "SUFFIX",
        "use_tree_spec": False,
        "max_branch_factor": 1,
        "port": 30002,
        "description": "单分支模式"
    },
    "multi_branch_3": {
        "speculative_algorithm": "SUFFIX",
        "use_tree_spec": True,
        "max_branch_factor": 3,
        "port": 30003,
        "description": "多分支模式 (branch=3)"
    },
    "multi_branch_5": {
        "speculative_algorithm": "SUFFIX",
        "use_tree_spec": True,
        "max_branch_factor": 5,
        "port": 30004,
        "description": "多分支模式 (branch=5)"
    }
}

# 日志目录
LOG_DIR = "/tmp/sglang_experiments"
os.makedirs(LOG_DIR, exist_ok=True)


def load_dataset(num_samples: int) -> List[Dict]:
    """加载数据集样本"""
    samples = []
    with open(DATASET_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= num_samples:
                break
            try:
                request_response = json.loads(row['request_response'])
                request_body = json.loads(request_response['request_body'])
                messages = request_body.get('messages', [])
                for msg in messages:
                    if msg.get('role') == 'user':
                        content = msg.get('content', '')
                        # 截断过长的内容
                        if isinstance(content, str):
                            samples.append({'prompt': content[:2000]})
                        else:
                            samples.append({'prompt': str(content)[:2000]})
                        break
            except Exception as e:
                continue
    print(f"加载了 {len(samples)} 个样本")
    return samples


def build_server_command(config: Dict) -> str:
    """构建服务器启动命令"""
    cmd = f"""SGLANG_DISABLE_CUDNN_CHECK=1 python3 -m sglang.launch_server \\
    --model-path {MODEL_PATH} \\
    --port {config['port']} --tp {TP} \\
    --disable-cuda-graph \\
    --disable-radix-cache \\
    --disable-overlap-schedule \\
    --trust-remote-code \\
    --log-level info"""
    
    if config['speculative_algorithm']:
        cmd += f""" \\
    --speculative-algorithm {config['speculative_algorithm']} \\
    --speculative-num-steps 5 \\
    --speculative-num-draft-tokens 8 \\
    --speculative-suffix-max-tree-depth 6"""
        
        if config['use_tree_spec']:
            max_branch = config['max_branch_factor']
            cmd += f""" \\
    --speculative-suffix-use-tree-spec \\
    --speculative-suffix-max-branch-factor {max_branch}"""
    
    return cmd


def start_server(config_name: str, config: Dict) -> Tuple[bool, str]:
    """启动服务器"""
    log_file = f"{LOG_DIR}/{config_name}.log"
    cmd = build_server_command(config)
    
    print(f"\n{'='*60}")
    print(f"启动服务器: {config_name}")
    print(f"配置: {config['description']}")
    print(f"端口: {config['port']}")
    print(f"日志: {log_file}")
    print(f"{'='*60}")
    
    # 先杀掉可能存在的旧进程
    subprocess.run(f"pkill -f 'sglang.launch_server' || true", shell=True)
    time.sleep(3)
    
    # 启动新服务器
    full_cmd = f"{cmd} > {log_file} 2>&1 &"
    subprocess.run(full_cmd, shell=True)
    
    # 等待服务器启动
    print(f"等待服务器启动 ({SERVER_START_WAIT}秒)...")
    for i in range(SERVER_START_WAIT):
        try:
            response = requests.get(f"http://localhost:{config['port']}/health", timeout=2)
            if response.status_code == 200:
                print(f"服务器就绪! (等待了 {i+1} 秒)")
                return True, log_file
        except:
            pass
        if i % 10 == 9:
            print(f"  等待中... {i+1}/{SERVER_START_WAIT}秒")
        time.sleep(1)
    
    print("服务器启动超时!")
    return False, log_file


def stop_server():
    """停止服务器"""
    print("停止服务器...")
    subprocess.run("pkill -f 'sglang.launch_server' || true", shell=True)
    time.sleep(5)


def run_benchmark(config_name: str, config: Dict, samples: List[Dict]) -> Dict:
    """运行基准测试
    
    吞吐量计算方式：tokens / (E2E - TTFT)
    其中 TTFT 通过流式请求测量，token 数量通过非流式请求获取
    """
    url = f"http://localhost:{config['port']}/v1/chat/completions"
    
    print(f"\n运行基准测试: {config_name}")
    print(f"样本数: {len(samples)}")
    
    total_time = 0
    total_tokens = 0
    successful_requests = 0
    latencies = []
    ttfts = []  # Time to First Token
    decode_times = []  # E2E - TTFT (实际解码时间)
    
    for i, sample in enumerate(samples):
        data = {
            "model": MODEL_PATH,
            "messages": [{"role": "user", "content": sample['prompt']}],
            "max_tokens": MAX_TOKENS,
        }
        
        try:
            # 第一步：流式请求测量 TTFT
            data_stream = {**data, "stream": True}
            start = time.time()
            response = requests.post(url, json=data_stream, timeout=120, stream=True)
            
            if response.status_code == 200:
                first_token_time = None
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            chunk_data = line[6:]
                            if chunk_data == '[DONE]':
                                break
                            try:
                                chunk = json.loads(chunk_data)
                                # 检测第一个 token
                                if first_token_time is None:
                                    delta = chunk.get('choices', [{}])[0].get('delta', {})
                                    if delta.get('content'):
                                        first_token_time = time.time()
                            except:
                                pass
                
                ttft = first_token_time - start if first_token_time else 0.1
                
                # 第二步：非流式请求获取准确的 token 数量
                data_non_stream = {**data, "stream": False}
                response2 = requests.post(url, json=data_non_stream, timeout=120)
                
                if response2.status_code == 200:
                    result = response2.json()
                    tokens = result.get('usage', {}).get('completion_tokens', 0)
                else:
                    tokens = MAX_TOKENS  # 如果失败，使用最大值估算
                
                e2e_time = time.time() - start
                decode_time = e2e_time - ttft
                
                total_time += e2e_time
                total_tokens += tokens
                latencies.append(e2e_time)
                ttfts.append(ttft)
                if decode_time > 0:
                    decode_times.append(decode_time)
                successful_requests += 1
            else:
                print(f"  请求 {i+1} 失败: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"  请求 {i+1} 错误: {e}")
        
        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{len(samples)}")
    
    # 计算吞吐量: tokens / decode_time
    total_decode_time = sum(decode_times) if decode_times else total_time
    throughput = total_tokens / total_decode_time if total_decode_time > 0 else 0
    
    # 计算基本指标
    results = {
        "config_name": config_name,
        "description": config['description'],
        "total_tokens": total_tokens,
        "total_time": total_time,
        "total_decode_time": total_decode_time,
        "successful_requests": successful_requests,
        "throughput": throughput,
        "avg_latency": sum(latencies) / len(latencies) if latencies else 0,
        "avg_ttft": sum(ttfts) / len(ttfts) if ttfts else 0,
        "avg_decode_time": sum(decode_times) / len(decode_times) if decode_times else 0,
        "accept_len": None,
        "accept_rate": None
    }
    
    return results


def extract_accept_metrics(log_file: str) -> Tuple[Optional[float], Optional[float]]:
    """从日志中提取接受率和接受长度"""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        lens = [float(x) for x in re.findall(r'accept len: ([\d.]+)', content)]
        rates = [float(x) for x in re.findall(r'accept rate: ([\d.]+)', content)]
        
        avg_len = sum(lens) / len(lens) if lens else None
        avg_rate = sum(rates) / len(rates) if rates else None
        
        return avg_len, avg_rate
    except Exception as e:
        print(f"提取接受指标失败: {e}")
        return None, None


def run_single_experiment(config_name: str, samples: List[Dict]) -> Dict:
    """运行单个实验配置"""
    config = EXPERIMENT_CONFIGS[config_name]
    
    # 启动服务器
    success, log_file = start_server(config_name, config)
    if not success:
        return {
            "config_name": config_name,
            "description": config['description'],
            "error": "服务器启动失败"
        }
    
    # 运行基准测试
    results = run_benchmark(config_name, config, samples)
    
    # 提取接受指标
    if config['speculative_algorithm']:
        accept_len, accept_rate = extract_accept_metrics(log_file)
        results['accept_len'] = accept_len
        results['accept_rate'] = accept_rate
    
    # 停止服务器
    stop_server()
    
    return results


def print_results_table(results_list: List[Dict]):
    """打印结果表格"""
    # 计算基线吞吐量
    baseline_throughput = None
    for r in results_list:
        if r.get('config_name') == 'baseline':
            baseline_throughput = r.get('throughput', 0)
            break
    
    print("\n" + "="*120)
    print("实验结果汇总")
    print("="*120)
    
    header = f"{'配置':<18} {'吞吐量':<12} {'加速比':<8} {'TTFT':<10} {'解码时间':<10} {'延迟':<10} {'接受长度':<10} {'接受率':<8}"
    print(header)
    print("-"*120)
    
    for r in results_list:
        throughput = r.get('throughput', 0)
        speedup = throughput / baseline_throughput if baseline_throughput and baseline_throughput > 0 else 0
        avg_ttft = r.get('avg_ttft', 0)
        avg_decode = r.get('avg_decode_time', 0)
        avg_latency = r.get('avg_latency', 0)
        accept_len = r.get('accept_len')
        accept_rate = r.get('accept_rate')
        
        accept_len_str = f"{accept_len:.2f}" if accept_len is not None else "N/A"
        accept_rate_str = f"{accept_rate:.2f}" if accept_rate is not None else "N/A"
        
        row = f"{r['config_name']:<18} {throughput:<12.2f} {speedup:<8.2f}x {avg_ttft:<10.2f} {avg_decode:<10.2f} {avg_latency:<10.2f} {accept_len_str:<10} {accept_rate_str:<8}"
        print(row)
    
    print("="*120)


def save_results(results_list: List[Dict], output_file: str):
    """保存结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_file}")


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='多分支 Suffix Decoding 实验')
    parser.add_argument('--config', type=str, default='all',
                       choices=['all', 'baseline', 'single_path', 'multi_branch_3', 'multi_branch_5'],
                       help='要运行的配置 (默认: all)')
    parser.add_argument('--samples', type=int, default=NUM_SAMPLES,
                       help=f'样本数 (默认: {NUM_SAMPLES})')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径')
    args = parser.parse_args()
    
    print("="*100)
    print("多分支 Suffix Decoding 实验")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据集: {DATASET_PATH}")
    print(f"样本数: {args.samples}")
    print(f"模型: {MODEL_PATH}")
    print(f"配置: {args.config}")
    print("="*100)
    
    # 加载数据集
    samples = load_dataset(args.samples)
    if not samples:
        print("错误: 无法加载数据集")
        return
    
    # 确定要运行的配置
    if args.config == 'all':
        config_names = ["baseline", "single_path", "multi_branch_3", "multi_branch_5"]
    else:
        config_names = [args.config]
    
    # 加载已有结果
    results_list = []
    if args.output and os.path.exists(args.output):
        with open(args.output, 'r') as f:
            results_list = json.load(f)
        print(f"加载已有结果: {len(results_list)} 条")
    
    # 运行实验配置
    for config_name in config_names:
        # 检查是否已运行
        if any(r.get('config_name') == config_name for r in results_list):
            print(f"\n跳过已完成的配置: {config_name}")
            continue
            
        print(f"\n{'#'*100}")
        print(f"# 实验配置: {config_name}")
        print(f"# 描述: {EXPERIMENT_CONFIGS[config_name]['description']}")
        print(f"{'#'*100}")
        
        results = run_single_experiment(config_name, samples)
        results_list.append(results)
        
        # 打印当前结果
        print(f"\n{config_name} 结果:")
        print(f"  吞吐量: {results.get('throughput', 0):.2f} tokens/sec")
        print(f"  平均延迟: {results.get('avg_latency', 0):.2f} sec")
        if results.get('accept_len') is not None:
            print(f"  接受长度: {results['accept_len']:.2f}")
            print(f"  接受率: {results['accept_rate']:.2f}")
        
        # 保存中间结果
        if args.output:
            save_results(results_list, args.output)
    
    # 打印汇总表格
    print_results_table(results_list)
    
    # 保存结果
    if not args.output:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = f"{LOG_DIR}/experiment_results_{timestamp}.json"
    save_results(results_list, args.output)
    
    print("\n实验完成!")


if __name__ == "__main__":
    main()
