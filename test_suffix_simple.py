#!/usr/bin/env python3
"""
简单测试 SUFFIX 推测解码
"""
import subprocess
import time
import sys

# 测试 1: 启动 baseline 服务器
print("="*60)
print("测试 1: Baseline (无推测解码)")
print("="*60)

# 使用 sglang 的离线模式直接测试
import sglang as sgl
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST

# 简单的生成测试
@sgl.function
def simple_gen(s, prompt):
    s += prompt
    s += s.gen(max_tokens=64)

# 初始化引擎
print("\n初始化 SGLang 引擎...")
try:
    # 直接使用 Runtime API 测试
    from sglang import Runtime
    
    # 测试 Baseline
    print("\n--- Baseline ---")
    engine = Runtime(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        tp_size=2,
        dtype="bfloat16",
        mem_fraction_static=0.7,
    )
    
    # 简单生成
    start = time.time()
    result = engine.generate("The capital of France is", max_new_tokens=32)
    elapsed = time.time() - start
    print(f"结果: {result}")
    print(f"耗时: {elapsed:.2f}s")
    engine.shutdown()
    
    # 测试 SUFFIX
    print("\n--- SUFFIX (多分支) ---")
    engine = Runtime(
        model_path="Qwen/Qwen2.5-7B-Instruct",
        tp_size=2,
        dtype="bfloat16",
        mem_fraction_static=0.7,
        speculative_algorithm="SUFFIX",
        speculative_suffix_use_tree_spec=True,
        speculative_suffix_max_branch_factor=2,
    )
    
    start = time.time()
    result = engine.generate("The capital of France is", max_new_tokens=32)
    elapsed = time.time() - start
    print(f"结果: {result}")
    print(f"耗时: {elapsed:.2f}s")
    engine.shutdown()
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()

print("\n测试完成!")
