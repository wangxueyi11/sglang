#!/usr/bin/env python3
"""
调试 SUFFIX 多分支实现
添加详细日志检查：
1. 是否真正生成了多分支树
2. parents 数组是否正确表示树结构
3. 接受率和接受长度是否正确计算
"""

import os
import json
import time
import requests
import subprocess
import signal

# 设置调试环境变量
os.environ["SUFFIX_DEBUG_TREE"] = "10"  # 打印前10个batch的调试信息

TARGET_MODEL = "/wxyworkspace/Qwen2.5-VL-32B-Instruct"
SERVER_PORT = 30000

def start_server(config_name, use_tree_spec, max_branch_factor):
    """启动服务器"""
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", TARGET_MODEL,
        "--host", "localhost",
        "--port", str(SERVER_PORT),
        "--tp", "4",
        "--mem-fraction-static", "0.8",
        "--context-length", "4096",
        "--max-running-requests", "8",
        "--disable-cuda-graph",
        "--speculative-algorithm", "SUFFIX",
        "--speculative-num-draft-tokens", "16",
        "--speculative-suffix-max-tree-depth", "32",
        "--speculative-suffix-min-token-prob", "0.01",
        "--log-level", "info",
    ]
    
    if use_tree_spec:
        cmd.append("--speculative-suffix-use-tree-spec")
        cmd.extend(["--speculative-suffix-max-branch-factor", str(max_branch_factor)])
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    env["SGLANG_DISABLE_CUDNN_CHECK"] = "1"
    env["SUFFIX_DEBUG_TREE"] = "10"
    
    log_file = f"/tmp/server_{config_name}.log"
    
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        cwd="/wxyworkspace",
    )
    
    return proc, log_file

def wait_for_server(timeout=180):
    """等待服务器启动"""
    for i in range(timeout // 5):
        try:
            resp = requests.get(f"http://localhost:{SERVER_PORT}/health", timeout=5)
            if resp.status_code == 200:
                return True
        except:
            pass
        time.sleep(5)
    return False

def stop_server():
    """停止服务器"""
    try:
        subprocess.run(["pkill", "-9", "-f", "sglang.launch_server"], 
                       capture_output=True, timeout=10)
    except:
        pass
    time.sleep(5)

def send_request(prompt):
    """发送请求"""
    resp = requests.post(
        f"http://localhost:{SERVER_PORT}/v1/chat/completions",
        json={
            "model": "default",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "temperature": 0.7,
        },
        timeout=120
    )
    return resp

def analyze_tree_structure(parents, name=""):
    """分析树结构"""
    if not parents:
        return {"depth": 0, "branches": 0, "is_tree": False}
    
    n = len(parents)
    
    # 计算深度
    depths = [0] * n
    for i in range(n):
        depth = 0
        curr = i
        visited = set()
        while curr >= 0 and curr < n and curr not in visited:
            visited.add(curr)
            curr = parents[curr]
            depth += 1
            if depth > 100:  # 防止无限循环
                break
        depths[i] = depth
    
    max_depth = max(depths) if depths else 0
    
    # 计算分支数
    children_count = {}
    for i, p in enumerate(parents):
        if p >= 0 and p < n:
            children_count[p] = children_count.get(p, 0) + 1
    
    max_branches = max(children_count.values()) if children_count else 0
    total_branches = sum(1 for c in children_count.values() if c > 1)
    
    # 检查是否是真正的树（有分支）
    is_tree = max_branches > 1
    
    return {
        "depth": max_depth,
        "max_branches": max_branches,
        "total_branch_nodes": total_branches,
        "is_tree": is_tree,
    }

def check_log_for_tree_info(log_file):
    """从日志中提取树信息"""
    import re
    
    results = []
    with open(log_file, 'r') as f:
        for line in f:
            # 查找 SUFFIX DEBUG 日志
            if "SUFFIX DEBUG" in line:
                results.append(line.strip())
            # 查找 accept rate/len
            if "accept len" in line or "accept rate" in line:
                results.append(line.strip())
    
    return results

def main():
    print("="*70)
    print("SUFFIX 多分支实现调试")
    print("="*70)
    
    # 测试样本
    test_prompts = [
        "请用一句话介绍北京",
        "请用一句话介绍上海",
        "请用一句话介绍深圳",
        "请用一句话介绍杭州",
        "请用一句话介绍成都",
    ]
    
    configs = [
        ("single-path", False, 1),
        ("tree-branch-3", True, 3),
        ("tree-branch-5", True, 5),
    ]
    
    all_results = {}
    
    for config_name, use_tree_spec, max_branch_factor in configs:
        print(f"\n{'='*60}")
        print(f"测试配置: {config_name}")
        print(f"use_tree_spec={use_tree_spec}, max_branch_factor={max_branch_factor}")
        print(f"{'='*60}")
        
        stop_server()
        proc, log_file = start_server(config_name, use_tree_spec, max_branch_factor)
        
        if not wait_for_server():
            print(f"服务器启动失败!")
            continue
        
        print(f"服务器已启动，发送请求...")
        
        # Warmup
        for i, prompt in enumerate(test_prompts[:2]):
            resp = send_request(prompt)
            print(f"Warmup {i+1}: {resp.status_code}")
        
        # 测试请求
        for i, prompt in enumerate(test_prompts[2:]):
            start = time.time()
            resp = send_request(prompt)
            latency = time.time() - start
            
            if resp.status_code == 200:
                data = resp.json()
                tokens = data.get("usage", {}).get("completion_tokens", 0)
                print(f"Request {i+1}: {tokens} tokens, {latency:.2f}s")
            else:
                print(f"Request {i+1}: FAILED {resp.status_code}")
        
        # 分析日志
        print(f"\n--- 日志分析 ---")
        log_info = check_log_for_tree_info(log_file)
        for info in log_info[-20:]:  # 显示最后20条
            print(info)
        
        all_results[config_name] = {
            "log_file": log_file,
            "log_info": log_info,
        }
        
        stop_server()
    
    # 保存结果
    with open("/wxyworkspace/debug_results.json", "w") as f:
        json.dump({k: {"log_file": v["log_file"], "log_info": v["log_info"][:50]} 
                   for k, v in all_results.items()}, f, indent=2)
    
    print("\n调试结果已保存到 /wxyworkspace/debug_results.json")

if __name__ == "__main__":
    main()
