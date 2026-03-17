#!/usr/bin/env python3
"""
Real experiment script for comparing single-branch vs multi-branch suffix decoding.

This script:
1. Starts sglang server with different suffix decoding configurations
2. Sends real requests from the dataset
3. Collects metrics: throughput, acceptance rate, average accepted length
"""

import json
import os
import subprocess
import sys
import time
import signal
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# Configuration
MODEL_PATH = "/wxyworkspace/Qwen2.5-VL-32B-Instruct"
DATASET_PATH = "/wxyworkspace/2025-12-10_1_Snippet 1_29337771.csv"
SERVER_PORT = 30000
SERVER_HOST = "localhost"

# Experiment configurations
EXPERIMENT_CONFIGS = [
    {
        "name": "baseline-no-spec",
        "speculative_algorithm": None,
        "use_tree_spec": False,
        "max_branch_factor": 1,
    },
    {
        "name": "suffix-single-path",
        "speculative_algorithm": "SUFFIX",
        "use_tree_spec": False,
        "max_branch_factor": 1,
    },
    {
        "name": "suffix-tree-branch-3",
        "speculative_algorithm": "SUFFIX",
        "use_tree_spec": True,
        "max_branch_factor": 3,
    },
]


def parse_dataset(csv_path: str, max_samples: int = None) -> pd.DataFrame:
    """Parse the CSV dataset to extract prompts."""
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if max_samples:
        df = df.head(max_samples)
    
    prompts = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing dataset"):
        try:
            req_resp = json.loads(row['request_response'])
            req_body = json.loads(req_resp['request_body'])
            
            # Get the last user message as prompt
            messages = req_body.get('messages', [])
            prompt = ""
            for msg in messages:
                if msg.get('role') == 'user':
                    prompt = msg.get('content', '')
            
            prompts.append(prompt)
        except Exception as e:
            print(f"Error parsing row {idx}: {e}")
            prompts.append("")
    
    return pd.DataFrame({'prompt': prompts})


def start_server(config: Dict) -> subprocess.Popen:
    """Start sglang server with the given configuration."""
    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", MODEL_PATH,
        "--host", SERVER_HOST,
        "--port", str(SERVER_PORT),
        "--tp", "4",  # Use 4 GPUs for 32B model
        "--mem-fraction-static", "0.8",
        "--context-length", "8192",
        "--max-running-requests", "16",
    ]
    
    if config["speculative_algorithm"]:
        cmd.extend([
            "--speculative-algorithm", config["speculative_algorithm"],
            "--speculative-num-draft-tokens", "16",
            "--speculative-suffix-max-tree-depth", "32",
            "--speculative-suffix-use-tree-spec", str(config["use_tree_spec"]).lower(),
            "--speculative-suffix-max-branch-factor", str(config["max_branch_factor"]),
            "--speculative-suffix-min-token-prob", "0.01",
        ])
    
    print(f"Starting server with command: {' '.join(cmd)}")
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    env["SGLANG_DISABLE_CUDNN_CHECK"] = "1"
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        preexec_fn=os.setsid,
    )
    
    return process


def wait_for_server(timeout: int = 600) -> bool:
    """Wait for the server to be ready."""
    print(f"Waiting for server to be ready (timeout: {timeout}s)...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(
                f"http://{SERVER_HOST}:{SERVER_PORT}/health",
                timeout=5
            )
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except:
            pass
        time.sleep(5)
        elapsed = int(time.time() - start_time)
        print(f"  Still waiting... ({elapsed}s)")
    
    print("Server failed to start within timeout!")
    return False


def stop_server(process: subprocess.Popen):
    """Stop the server process."""
    if process:
        print("Stopping server...")
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=30)
        except:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except:
                pass
        print("Server stopped.")


def send_request(prompt: str, max_tokens: int = 512) -> Dict:
    """Send a request to the server and return metrics."""
    url = f"http://{SERVER_HOST}:{SERVER_PORT}/v1/chat/completions"
    
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    
    start_time = time.perf_counter()
    response = requests.post(url, json=payload, timeout=120)
    elapsed = time.perf_counter() - start_time
    
    if response.status_code != 200:
        return {
            "success": False,
            "error": response.text,
            "latency": elapsed,
        }
    
    data = response.json()
    
    # Extract metrics from response headers or metadata
    output_tokens = data.get("usage", {}).get("completion_tokens", 0)
    prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
    
    return {
        "success": True,
        "output_tokens": output_tokens,
        "prompt_tokens": prompt_tokens,
        "latency": elapsed,
        "throughput": output_tokens / elapsed if elapsed > 0 else 0,
    }


def run_experiment(
    config: Dict,
    prompts: List[str],
    num_warmup: int = 5,
    num_eval: int = 50,
) -> Dict:
    """Run a single experiment configuration."""
    print(f"\n{'='*70}")
    print(f"Running experiment: {config['name']}")
    print(f"{'='*70}")
    
    # Start server
    server_process = start_server(config)
    
    results = {
        "config": config["name"],
        "success": False,
        "requests": [],
        "total_output_tokens": 0,
        "total_latency": 0,
        "avg_throughput": 0,
        "error": None,
    }
    
    try:
        # Wait for server
        if not wait_for_server(timeout=600):
            results["error"] = "Server failed to start"
            return results
        
        # Warmup
        print(f"Warming up with {num_warmup} requests...")
        for i in range(num_warmup):
            try:
                send_request(prompts[i % len(prompts)], max_tokens=256)
            except Exception as e:
                print(f"Warmup request {i} failed: {e}")
        
        # Evaluation
        print(f"Running evaluation with {num_eval} requests...")
        successful = 0
        
        for i in tqdm(range(num_eval), desc=f"Evaluating {config['name']}"):
            try:
                result = send_request(prompts[i % len(prompts)], max_tokens=512)
                if result["success"]:
                    successful += 1
                    results["requests"].append(result)
                    results["total_output_tokens"] += result["output_tokens"]
                    results["total_latency"] += result["latency"]
            except Exception as e:
                print(f"Request {i} failed: {e}")
        
        # Compute aggregate metrics
        if successful > 0:
            results["success"] = True
            results["avg_throughput"] = results["total_output_tokens"] / results["total_latency"]
            results["successful_requests"] = successful
        
    except Exception as e:
        results["error"] = str(e)
    finally:
        stop_server(server_process)
        time.sleep(10)  # Wait for GPU memory to be released
    
    return results


def print_results(all_results: List[Dict]):
    """Print comparison table of all results."""
    print("\n" + "="*100)
    print("EXPERIMENT RESULTS")
    print("="*100)
    
    print(f"{'Config':<30} {'Success':<10} {'Requests':<10} {'Output Toks':<15} {'Latency(s)':<12} {'Throughput':<15}")
    print("-"*100)
    
    for r in all_results:
        if r["success"]:
            print(f"{r['config']:<30} {'Yes':<10} {r.get('successful_requests', 0):<10} {r['total_output_tokens']:<15} {r['total_latency']:<12.2f} {r['avg_throughput']:<15.2f}")
        else:
            print(f"{r['config']:<30} {'No':<10} {'-':<10} {'-':<15} {'-':<12} {'-':<15}")
    
    print("="*100)
    
    # Compute improvement relative to baseline
    baseline = all_results[0]
    if baseline["success"]:
        print("\nImprovement relative to baseline (no speculation):")
        print(f"{'Config':<30} {'Throughput Improvement':<25}")
        print("-"*60)
        
        for r in all_results[1:]:
            if r["success"] and baseline["avg_throughput"] > 0:
                improvement = (r["avg_throughput"] / baseline["avg_throughput"] - 1) * 100
                print(f"{r['config']:<30} {improvement:>+.2f}%")
            else:
                print(f"{r['config']:<30} N/A")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-eval", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        sys.exit(1)
    
    # Load dataset
    dataset = parse_dataset(DATASET_PATH, max_samples=args.max_samples)
    prompts = dataset[dataset['prompt'].str.len() > 0]['prompt'].tolist()
    print(f"Loaded {len(prompts)} prompts")
    
    # Run experiments
    all_results = []
    
    for config in EXPERIMENT_CONFIGS:
        result = run_experiment(
            config,
            prompts,
            num_warmup=args.num_warmup,
            num_eval=args.num_eval,
        )
        all_results.append(result)
        print_results(all_results)
    
    # Save results
    results_df = pd.DataFrame([
        {
            "config": r["config"],
            "success": r["success"],
            "successful_requests": r.get("successful_requests", 0),
            "total_output_tokens": r["total_output_tokens"],
            "total_latency": r["total_latency"],
            "avg_throughput": r["avg_throughput"],
            "error": r.get("error", ""),
        }
        for r in all_results
    ])
    results_df.to_csv("/wxyworkspace/real_experiment_results.csv", index=False)
    print("\nResults saved to /wxyworkspace/real_experiment_results.csv")


if __name__ == "__main__":
    main()
