#!/usr/bin/env python3
"""
测试 SUFFIX 推测解码集成到 verl
"""
import dataclasses
from sglang.srt.server_args import ServerArgs

print("=" * 60)
print("测试 SUFFIX 集成到 verl")
print("=" * 60)

# 模拟 verl 的 MtpConfig
class MockMtpConfig:
    def __init__(self):
        self.enable = True
        self.enable_rollout = True
        self.speculative_algorithm = "SUFFIX"
        self.speculative_num_steps = 3
        self.speculative_eagle_topk = 1
        self.speculative_num_draft_tokens = 4
        
        # SUFFIX 参数
        self.suffix_max_tree_depth = 12
        self.suffix_max_cached_requests = 5000
        self.suffix_max_spec_factor = 1.0
        self.suffix_min_token_prob = 0.1
        self.suffix_use_tree_spec = True
        self.suffix_max_branch_factor = 2

# 模拟 async_sglang_server.py 中的参数构建
def build_server_args(mtp_config):
    """模拟 verl 中的参数构建逻辑"""
    args = {}
    
    # 这是 verl 中添加的逻辑
    if mtp_config.enable and mtp_config.enable_rollout:
        args["speculative_algorithm"] = mtp_config.speculative_algorithm
        args["speculative_num_steps"] = mtp_config.speculative_num_steps
        args["speculative_eagle_topk"] = mtp_config.speculative_eagle_topk
        args["speculative_num_draft_tokens"] = mtp_config.speculative_num_draft_tokens
        
        # SUFFIX 特有参数
        if mtp_config.speculative_algorithm == "SUFFIX":
            args["speculative_suffix_max_tree_depth"] = mtp_config.suffix_max_tree_depth
            args["speculative_suffix_max_cached_requests"] = mtp_config.suffix_max_cached_requests
            args["speculative_suffix_max_spec_factor"] = mtp_config.suffix_max_spec_factor
            args["speculative_suffix_min_token_prob"] = mtp_config.suffix_min_token_prob
            args["speculative_suffix_use_tree_spec"] = mtp_config.suffix_use_tree_spec
            args["speculative_suffix_max_branch_factor"] = mtp_config.suffix_max_branch_factor
    
    return args

# 测试
mtp = MockMtpConfig()
args = build_server_args(mtp)

print("\n1. MtpConfig 配置:")
print(f"   speculative_algorithm: {mtp.speculative_algorithm}")
print(f"   suffix_use_tree_spec: {mtp.suffix_use_tree_spec}")
print(f"   suffix_max_branch_factor: {mtp.suffix_max_branch_factor}")

print("\n2. 传递给 ServerArgs 的参数:")
for k, v in args.items():
    print(f"   {k}: {v}")

# 验证 ServerArgs 是否有这些字段
print("\n3. 验证 ServerArgs 字段存在:")
server_args_fields = {f.name for f in dataclasses.fields(ServerArgs)}
missing_fields = []
for k in args.keys():
    if k in server_args_fields:
        print(f"   ✅ {k}: 存在")
    else:
        print(f"   ❌ {k}: 不存在")
        missing_fields.append(k)

if missing_fields:
    print(f"\n❌ 缺少字段: {missing_fields}")
    exit(1)
else:
    print("\n" + "=" * 60)
    print("✅ SUFFIX 集成验证通过!")
    print("=" * 60)
