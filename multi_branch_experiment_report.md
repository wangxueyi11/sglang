# 多分支 Suffix Decoding 实验报告

## 1. 实验概述

本实验旨在验证多分支（Multi-Branch）Suffix Decoding 相比单分支模式的性能提升。

### 1.1 实验背景

ArcticInference 的 `use_tree_spec=True` 参数理论上应该支持多分支推测树构建，但经测试发现该参数实际上并不生效，始终返回单分支路径。本实验使用我们实现的本地 `MultiBranchTreeBuilder` 来构建真正的多分支树结构，并与单分支模式进行对比。

### 1.2 吞吐量计算方式

本实验采用标准吞吐量计算方式：
```
吞吐量 = 总输出tokens / 总解码时间
解码时间 = E2E - TTFT (End-to-End时间 - Time to First Token)
```

通过流式请求精确测量 TTFT，非流式请求获取准确的 token 数量。

### 1.3 实验配置

| 配置项 | 值 |
|--------|-----|
| 模型 | Qwen2.5-VL-32B-Instruct |
| Tensor Parallelism | 4 |
| 数据集 | 真实请求日志（2025-12-10_1_Snippet 1_29337771.csv） |
| 样本数 | 50 |
| 最大输出Token数 | 100 |
| 推测步数 | 5 |
| 推测Token数 | 8 |
| 树深度 | 6 |

### 1.4 实验模式

| 模式 | 描述 | 关键参数 |
|------|------|----------|
| baseline | 无推测解码 | 无 |
| single_path | 单分支模式 | --speculative-algorithm SUFFIX |
| multi_branch_3 | 多分支模式 | --speculative-suffix-use-tree-spec --speculative-suffix-max-branch-factor 3 |
| multi_branch_5 | 多分支模式 | --speculative-suffix-use-tree-spec --speculative-suffix-max-branch-factor 5 |

## 2. 实验结果

### 2.1 性能对比汇总

| 配置 | 吞吐量 (tokens/sec) | 加速比 | TTFT (sec) | 解码时间 (sec) | 接受长度 | 接受率 |
|------|---------------------|--------|------------|----------------|----------|--------|
| **baseline** | 17.04 | 1.00x | 0.16 | 5.62 | N/A | N/A |
| **single_path** | 43.17 | 2.53x | 0.16 | 2.23 | 2.94 | 0.37 |
| **multi_branch_3** | 47.59 | 2.79x | 0.16 | 2.01 | 3.90 | 0.49 |
| **multi_branch_5** | 46.79 | 2.75x | 0.16 | 2.04 | 3.82 | 0.48 |

### 2.2 关键发现

#### 2.2.1 吞吐量提升

```
baseline:        17.04 tokens/sec (基准)
single_path:     43.17 tokens/sec (+153.2%, 2.53x)
multi_branch_3:  47.59 tokens/sec (+179.3%, 2.79x)
multi_branch_5:  46.79 tokens/sec (+174.6%, 2.75x)
```

多分支模式 (branch=3) 相比单分支模式：
- 吞吐量提升: **10.2%** (43.17 → 47.59)
- 解码时间降低: **9.9%** (2.23 → 2.01 sec)

#### 2.2.2 接受指标提升

```
                    接受长度    接受率
single_path:         2.94      0.37
multi_branch_3:      3.90      0.49
multi_branch_5:      3.82      0.48

接受长度提升: 32.7% (2.94 → 3.90)
接受率提升:   32.4% (0.37 → 0.49)
```

#### 2.2.3 分支数影响

branch=3 和 branch=5 性能接近，branch=3 略优：
- 吞吐量: 47.59 vs 46.79 (branch=3 略优)
- 接受长度: 3.90 vs 3.82 (branch=3 略优)
- 接受率: 0.49 vs 0.48 (branch=3 略优)

**结论**: branch=3 已足够，更大的分支数不会带来额外收益，反而略有下降。

### 2.3 多分支验证

从日志中可以确认多分支树结构正常工作：

**multi_branch_3 根分支分布：**
```
roots=1: 6044 次
roots=2: 1404 次
roots=3: 1736 次
```

**multi_branch_5 根分支分布：**
```
roots=1: 5892 次
roots=2: 1564 次
roots=3: 336 次
roots=4: 252 次
roots=5: 1260 次
```

多子节点示例：
```
[MULTI-BRANCH] multi_child_parents={-1: 3, 1: 3}, ngrams=23409
[MULTI-BRANCH] multi_child_parents={0: 5}, ngrams=23634
```

`multi_child_parents={node: child_count}` 表示该节点有多个子节点，证明多分支树构建成功。

## 3. 复现步骤

### 3.1 环境准备

```bash
# 1. 确保已安装 sglang 和 ArcticInference
pip install sglang
pip install arcticinference

# 2. 准备模型
# 模型路径: /wxyworkspace/Qwen2.5-VL-32B-Instruct

# 3. 准备数据集
# 数据集路径: /wxyworkspace/2025-12-10_1_Snippet 1_29337771.csv
```

### 3.2 运行完整实验

```bash
cd /wxyworkspace

# 运行所有配置的实验
python3 run_multi_branch_experiment.py --samples 50 --output results.json

# 或者单独运行每个配置
python3 run_multi_branch_experiment.py --config baseline --samples 50 --output results.json
python3 run_multi_branch_experiment.py --config single_path --samples 50 --output results.json
python3 run_multi_branch_experiment.py --config multi_branch_3 --samples 50 --output results.json
python3 run_multi_branch_experiment.py --config multi_branch_5 --samples 50 --output results.json
```

### 3.3 手动启动服务器测试

#### Baseline (无推测解码)
```bash
SGLANG_DISABLE_CUDNN_CHECK=1 python3 -m sglang.launch_server \
    --model-path /wxyworkspace/Qwen2.5-VL-32B-Instruct \
    --port 30001 --tp 4 \
    --disable-cuda-graph \
    --disable-radix-cache \
    --disable-overlap-schedule \
    --trust-remote-code \
    --log-level info
```

#### Single Path (单分支)
```bash
SGLANG_DISABLE_CUDNN_CHECK=1 python3 -m sglang.launch_server \
    --model-path /wxyworkspace/Qwen2.5-VL-32B-Instruct \
    --port 30002 --tp 4 \
    --speculative-algorithm SUFFIX \
    --speculative-num-steps 5 \
    --speculative-num-draft-tokens 8 \
    --speculative-suffix-max-tree-depth 6 \
    --disable-cuda-graph \
    --disable-radix-cache \
    --disable-overlap-schedule \
    --trust-remote-code \
    --log-level info
```

#### Multi-Branch (多分支, branch=3)
```bash
SGLANG_DISABLE_CUDNN_CHECK=1 python3 -m sglang.launch_server \
    --model-path /wxyworkspace/Qwen2.5-VL-32B-Instruct \
    --port 30003 --tp 4 \
    --speculative-algorithm SUFFIX \
    --speculative-num-steps 5 \
    --speculative-num-draft-tokens 8 \
    --speculative-suffix-max-tree-depth 6 \
    --speculative-suffix-use-tree-spec \
    --speculative-suffix-max-branch-factor 3 \
    --disable-cuda-graph \
    --disable-radix-cache \
    --disable-overlap-schedule \
    --trust-remote-code \
    --log-level info
```

#### Multi-Branch (多分支, branch=5)
```bash
SGLANG_DISABLE_CUDNN_CHECK=1 python3 -m sglang.launch_server \
    --model-path /wxyworkspace/Qwen2.5-VL-32B-Instruct \
    --port 30004 --tp 4 \
    --speculative-algorithm SUFFIX \
    --speculative-num-steps 5 \
    --speculative-num-draft-tokens 8 \
    --speculative-suffix-max-tree-depth 6 \
    --speculative-suffix-use-tree-spec \
    --speculative-suffix-max-branch-factor 5 \
    --disable-cuda-graph \
    --disable-radix-cache \
    --disable-overlap-schedule \
    --trust-remote-code \
    --log-level info
```

### 3.4 发送测试请求

```python
import requests
import time

url = "http://localhost:30003/v1/chat/completions"  # 根据端口调整
data = {
    "model": "/wxyworkspace/Qwen2.5-VL-32B-Instruct",
    "messages": [{"role": "user", "content": "What is machine learning?"}],
    "max_tokens": 100
}

start = time.time()
response = requests.post(url, json=data)
elapsed = time.time() - start
print(f"Response time: {elapsed:.2f}s")
print(response.json()['choices'][0]['message']['content'])
```

## 4. 核心代码实现

### 4.1 MultiBranchTreeBuilder 类

位置: `/wxyworkspace/python/sglang/srt/speculative/suffix_cache_adapter.py`

关键方法 `build_tree`：使用 BFS 构建真正的多分支树

```python
def build_tree(self, context: List[int], max_tokens: int) -> Tuple[List[int], List[int], float]:
    """使用 BFS 构建多分支树"""
    # 获取根节点的多个分支
    root_branches = self.get_branches(context, self.max_depth, self.max_branch_factor)
    
    # BFS 队列: (parent_idx, context, depth, cum_prob)
    queue = deque()
    
    # 添加根节点的分支（多分支根节点）
    for token, prob in root_branches:
        current_idx = len(token_ids)
        token_ids.append(token)
        parents.append(-1)  # 根节点
        queue.append((current_idx, context + [token], 1, prob))
    
    # BFS 探索 - 每个节点可以有多个子节点
    while queue and len(token_ids) < max_tokens:
        parent_idx, ctx, depth, cum_prob = queue.popleft()
        branches = self.get_branches(ctx, self.max_depth - depth, self.max_branch_factor)
        
        # 为当前节点添加多个子节点
        for token, prob in branches:
            current_idx = len(token_ids)
            token_ids.append(token)
            parents.append(parent_idx)  # 正确设置父节点
            queue.append((current_idx, ctx + [token], depth + 1, cum_prob * prob))
    
    return token_ids, parents, total_score
```

### 4.2 DeepTreeBuilder 决策逻辑

```python
def build_deep_tree(self, cache_req_id, context, max_tokens):
    # 1. 使用本地分支缓存构建多分支树
    local_tokens, local_parents, local_score = self.branch_cache.build_tree(context, max_tokens)
    
    # 2. 分析树结构
    parent_counts = Counter(local_parents)
    multi_child = {p: c for p, c in parent_counts.items() if c > 1}
    root_count = sum(1 for p in local_parents if p == -1)
    
    # 3. 如果有多个根分支或多子节点，使用本地树
    if root_count >= 2 or len(multi_child) > 0:
        return local_tokens, local_parents, local_score
    
    # 4. 否则回退到 ArcticInference 单路径
    main_draft = self.suffix_cache.speculate(cache_req_id, context, ...)
    return main_tokens, main_parents, main_score
```

## 5. Draft Tokens 对比实验

### 5.1 实验配置

固定 branch_factor=3，对比 draft_tokens=8 和 draft_tokens=16 的效果。

### 5.2 实验结果

| 配置 | 吞吐量 (tokens/sec) | 加速比 | TTFT (sec) | 解码时间 (sec) | 接受长度 | 接受率 | 每分支节点数 |
|------|---------------------|--------|------------|----------------|----------|--------|-------------|
| **baseline** | 16.90 | 1.00x | 0.16 | 5.68 | N/A | N/A | N/A |
| **draft_8_branch3** | 47.72 | 2.82x | 0.16 | 2.01 | 3.91 | 0.49 | 5.12 |
| **draft_16_branch3** | 48.96 | 2.90x | 0.16 | 1.95 | 4.03 | 0.25 | 9.05 |

### 5.3 分析

1. **draft_tokens=16 vs draft_tokens=8**：
   - 吞吐量提升: **2.6%** (47.72 → 48.96)
   - 接受长度提升: **3.1%** (3.91 → 4.03)
   - 每分支节点数增加: **77%** (5.12 → 9.05)

2. **接受率下降**：
   - draft_tokens=8: 0.49
   - draft_tokens=16: 0.25
   - 原因：更大的树包含更多低概率分支，整体接受率下降，但接受的 token 数量增加

3. **结论**：
   - draft_tokens=16 有轻微提升，但边际收益递减
   - 推荐使用 draft_tokens=8 作为默认值
   - 对于长文本生成场景，可以考虑 draft_tokens=16

## 6. 结论与建议

### 6.1 结论

1. **多分支有效**: 我们实现的多分支 Suffix Decoding 比单分支模式有明显提升
   - 吞吐量提升 10.2% (43.17 → 47.59)
   - 接受长度提升 32.7% (2.94 → 3.90)
   - 接受率提升 32.4% (0.37 → 0.49)

2. **分支数选择**: branch=3 已达到最佳效果，更大的分支数(branch=5)反而略有下降

3. **draft_tokens 选择**: draft_tokens=8 性价比更高，draft_tokens=16 边际收益有限

4. **整体加速**: 相比无推测解码，多分支模式达到 **2.90x** 加速

### 6.2 推荐配置

```bash
--speculative-algorithm SUFFIX \
--speculative-num-steps 5 \
--speculative-num-draft-tokens 8 \
--speculative-suffix-max-tree-depth 6 \
--speculative-suffix-use-tree-spec \
--speculative-suffix-max-branch-factor 3
```

### 6.3 建议

1. 使用上述推荐配置作为默认值
2. 确保服务器日志级别为 `info` 以监控接受率和接受长度
3. 对于不同数据集，可能需要调整 `max_branch_factor` 和 `max_tree_depth`

## 6. 附录

### 6.1 实验环境

- 操作系统: Linux 5.14.0-3.0.3.kwai.x86_64
- Python: 3.12.10
- GPU: 8x NVIDIA GPU (TP=4)
- 模型: Qwen2.5-VL-32B-Instruct

### 6.2 相关文件

- 实验脚本: `/wxyworkspace/run_multi_branch_experiment.py`
- 核心实现: `/wxyworkspace/python/sglang/srt/speculative/suffix_cache_adapter.py`
- 实验结果: `/tmp/sglang_experiments/results_v2.json`
- 日志目录: `/tmp/sglang_experiments/`

### 6.3 实验时间

- 开始时间: 2026-03-12 23:20:00
- 结束时间: 2026-03-12 23:36:00
- 总耗时: 约 16 分钟
