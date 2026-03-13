#!/usr/bin/env python3
"""
Draft Tokens 对比实验

对比 draft_tokens=8 和 draft_tokens=16 的效果
使用 branch=3 作为固定参数

评价指标：
- 吞吐量 (tokens/sec)
- 加速比
- TTFT
- 解码时间
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
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# ==================== 配置 ====================
MODEL_PATH = "/wxyworkspace/Qwen2.5-VL-32B-Instruct"
DATASET_PATH = "/wxyworkspace/2025-12-10_1_Snippet 1_29337771.csv"
NUM_SAMPLES = 50
MAX_TOKENS = 100
TP = 4

# 实验配置
EXPERIMENT_CONFIGS = {
    "baseline": {
        "draft_tokens": 0,
        "use_spec": False,
        "port": 30001,
        "description": "无推测解码（基线）"
    },
    "draft_8_branch3": {
        "draft_tokens": 8,
        "use_spec": True,
        "branch_factor": 3,
        "port": 30002,
        "description": "draft_tokens=8, branch=3"
    },
    "draft_16_branch3": {
        "draft_tokens": 16,
        "use_spec": True,
        "branch_factor": 3,
        "port": 30003,
        "description": "draft_tokens=16, branch=3"
    },
}

LOG_DIR = "/tmp/sglang_draft_exp"
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
                        if isinstance(content, str):
                            samples.append({'prompt': content[:2000]})
                        else:
                            samples.append({'prompt': str(content)[:2000]})
                        break
            except:
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
    
    if config['use_spec']:
        cmd += f""" \\
    --speculative-algorithm SUFFIX \\
    --speculative-num-steps 5 \\
    --speculative-num-draft-tokens {config['draft_tokens']} \\
    --speculative-suffix-max-tree-depth 8 \\
    --speculative-suffix-use-tree-spec \\
    --speculative-suffix-max-branch-factor {config['branch_factor']}"""
    
    return cmd


def start_server(config_name: str, config: Dict) -> Tuple[bool, str]:
    """启动服务器"""
    log_file = f"{LOG_DIR}/{config_name}.log"
    cmd = build_server_command(config)
    
    print(f"\n{'='*60}")
    print(f"启动服务器: {config_name}")
    print(f"配置: {config['description']}")
    print(f"{'='*60}")
    
    subprocess.run(f"pkill -f 'sglang.launch_server' || true", shell=True)
    time.sleep(5)
    
    full_cmd = f"{cmd} > {log_file} 2>&1 &"
    subprocess.run(full_cmd, shell=True)
    
    print(f"等待服务器启动...")
    for i in range(80):
        try:
            response = requests.get(f"http://localhost:{config['port']}/health", timeout=2)
            if response.status_code == 200:
                print(f"服务器就绪! (等待了 {i+1} 秒)")
                return True, log_file
        except:
            pass
        if i % 10 == 9:
            print(f"  等待中... {i+1}/80秒")
        time.sleep(1)
    
    print("服务器启动超时!")
    return False, log_file


def stop_server():
    """停止服务器"""
    print("停止服务器...")
    subprocess.run("pkill -f 'sglang.launch_server' || true", shell=True)
    time.sleep(5)


def run_benchmark(config_name: str, config: Dict, samples: List[Dict]) -> Dict:
    """运行基准测试"""
    url = f"http://localhost:{config['port']}/v1/chat/completions"
    
    print(f"\n运行基准测试: {config_name}")
    
    total_time = 0
    total_tokens = 0
    successful_requests = 0
    latencies = []
    ttfts = []
    decode_times = []
    
    for i, sample in enumerate(samples):
        data = {
            "model": MODEL_PATH,
            "messages": [{"role": "user", "content": sample['prompt']}],
            "max_tokens": MAX_TOKENS,
        }
        
        try:
            # 流式请求测量 TTFT
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
                                if first_token_time is None:
                                    delta = chunk.get('choices', [{}])[0].get('delta', {})
                                    if delta.get('content'):
                                        first_token_time = time.time()
                            except:
                                pass
                
                ttft = first_token_time - start if first_token_time else 0.1
                
                # 非流式请求获取 token 数量
                data_non_stream = {**data, "stream": False}
                response2 = requests.post(url, json=data_non_stream, timeout=120)
                
                if response2.status_code == 200:
                    result = response2.json()
                    tokens = result.get('usage', {}).get('completion_tokens', 0)
                else:
                    tokens = MAX_TOKENS
                
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
    
    total_decode_time = sum(decode_times) if decode_times else total_time
    throughput = total_tokens / total_decode_time if total_decode_time > 0 else 0
    
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
    except:
        return None, None


def print_results_table(results_list: List[Dict]):
    """打印结果表格"""
    baseline_throughput = None
    for r in results_list:
        if r.get('config_name') == 'baseline':
            baseline_throughput = r.get('throughput', 0)
            break
    
    print("\n" + "="*130)
    print("Draft Tokens 对比实验结果")
    print("="*130)
    
    header = f"{'配置':<22} {'吞吐量':<12} {'加速比':<8} {'TTFT':<10} {'解码时间':<10} {'延迟':<10} {'接受长度':<10} {'接受率':<8}"
    print(header)
    print("-"*130)
    
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
        
        row = f"{r['config_name']:<22} {throughput:<12.2f} {speedup:<8.2f}x {avg_ttft:<10.2f} {avg_decode:<10.2f} {avg_latency:<10.2f} {accept_len_str:<10} {accept_rate_str:<8}"
        print(row)
    
    print("="*130)


def save_results(results_list: List[Dict], output_file: str):
    """保存结果到文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {output_file}")


def main():
    """主函数"""
    print("="*100)
    print("Draft Tokens 对比实验")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"数据集: {DATASET_PATH}")
    print(f"样本数: {NUM_SAMPLES}")
    print(f"模型: {MODEL_PATH}")
    print("="*100)
    
    samples = load_dataset(NUM_SAMPLES)
    if not samples:
        print("错误: 无法加载数据集")
        return
    
    results_list = []
    config_order = ["baseline", "draft_8_branch3", "draft_16_branch3"]
    
    for config_name in config_order:
        config = EXPERIMENT_CONFIGS[config_name]
        
        print(f"\n{'#'*100}")
        print(f"# 实验配置: {config_name}")
        print(f"# 描述: {config['description']}")
        print(f"{'#'*100}")
        
        success, log_file = start_server(config_name, config)
        if not success:
            results_list.append({
                "config_name": config_name,
                "description": config['description'],
                "error": "服务器启动失败"
            })
            continue
        
        results = run_benchmark(config_name, config, samples)
        
        if config['use_spec']:
            accept_len, accept_rate = extract_accept_metrics(log_file)
            results['accept_len'] = accept_len
            results['accept_rate'] = accept_rate
            
            # 提取树结构统计
            with open(log_file, 'r') as f:
                content = f.read()
            matches = re.findall(r'tokens=(\d+), roots=(\d+)', content)
            if matches:
                total_tokens = sum(int(t) for t, r in matches)
                total_roots = sum(int(r) for t, r in matches)
                results['avg_tree_tokens'] = total_tokens / len(matches)
                results['avg_tree_roots'] = total_roots / len(matches)
                results['nodes_per_branch'] = total_tokens / total_roots if total_roots > 0 else 0
        
        results_list.append(results)
        
        print(f"\n{config_name} 结果:")
        print(f"  吞吐量: {results.get('throughput', 0):.2f} tokens/sec")
        print(f"  平均延迟: {results.get('avg_latency', 0):.2f} sec")
        if results.get('accept_len') is not None:
            print(f"  接受长度: {results['accept_len']:.2f}")
            print(f"  接受率: {results['accept_rate']:.2f}")
            if results.get('nodes_per_branch'):
                print(f"  每分支节点数: {results['nodes_per_branch']:.2f}")
        
        stop_server()
        
        # 保存中间结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_results(results_list, f"{LOG_DIR}/draft_exp_{timestamp}.json")
    
    print_results_table(results_list)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_results(results_list, f"{LOG_DIR}/draft_exp_final_{timestamp}.json")
    
    print("\n实验完成!")


if __name__ == "__main__":
    main()
