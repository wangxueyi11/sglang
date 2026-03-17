# Suffix Decoding 多分支分析报告

## 实验日期
2026-03-13

## 核心发现

### 1. ArcticInference 的 `use_tree_spec=True` 不支持真正的多分支

通过详细的调试日志，我们发现：

```
[SUFFIX TREE MODE] roots=1, tokens=1, parents=[-1], score=0.667
```

**关键问题**：
- `roots=1` 永远为 1，ArcticInference 始终只返回单个根节点
- `parents=[-1]` 表示只有单个 token，没有分支结构
- 即使发送多个具有相同前缀但不同后续的请求，也不会产生多分支

### 2. ArcticInference 的实现限制

ArcticInference 的 `speculate` 方法（C++ 实现）设计为：
- 只返回"最佳匹配"的后续，而不是所有可能的后续
- `use_tree_spec=True` 只是将返回格式变为树兼容格式，但不构建真正的多分支树

### 3. 单路径推测确实有效

实验数据显示：
- 相似前缀的后续请求明显加速 (0.07s vs 0.82s)
- Suffix cache 正在工作，能够找到匹配
- 但每次只返回一个后续路径

## 技术分析

### NGRAM vs SUFFIX 的关键区别

| 特性 | NGRAM | SUFFIX (当前实现) |
|------|-------|-------------------|
| 多分支支持 | ✅ 是 | ❌ 否 |
| Tree mask | ✅ nxn 矩阵 | ❌ 线性链 |
| 分支探索 | BFS 广度优先 | 单路径 |
| 实现位置 | C++ (可定制) | C++ (黑盒) |

### 为什么多分支重要？

在推测解码中，多分支可以：
1. 探索多个可能的后续路径
2. 即使主路径被拒绝，备用路径可能被接受
3. 提高整体接受率和吞吐量

## 实现真正多分支的方案

### 方案 A: 修改 ArcticInference C++ 代码

需要修改 ArcticInference 的 suffix tree 遍历逻辑：
- 在每个节点处探索多个子节点
- 构建真正的树结构而非单路径
- 返回包含所有分支的 parents 数组

### 方案 B: 实现独立的 Suffix Tree

在 sglang 层面实现：
- 跟踪所有缓存的请求/响应对
- 在每个匹配点探索所有可能的后续
- 构建多分支树结构

### 方案 C: 使用 NGRAM 作为参考

参考 `/wxyworkspace/python/sglang/srt/speculative/cpp_ngram/ngram_cache.py` 的实现：
- 使用 tree mask 编码树结构
- BFS 遍历构建多分支

## 实验日志摘要

```
# 单路径模式 (use_tree_spec=False)
[SUFFIX MAIN PATH] tokens=1, score=0.500, first_tokens=[13]

# 树模式 (use_tree_spec=True) - 同样是单路径
[SUFFIX TREE MODE] roots=1, tokens=1, parents=[-1], score=0.667

# 分支探索 - 未找到替代分支
[SUFFIX BRANCH] main_tokens=1, branches_added=0, total_tokens=1, unique_parents=1
```

## 结论

1. **当前 SUFFIX 实现只支持单路径推测**，无法实现真正的多分支
2. **性能提升来自缓存复用**，而非多分支并行探索
3. **要实现真正多分支**，需要修改 ArcticInference C++ 后端或重新实现 suffix tree

## 下一步建议

1. **短期**：继续使用单路径 SUFFIX，但正确测量性能提升
2. **中期**：评估修改 ArcticInference 或实现独立 suffix tree 的成本
3. **长期**：考虑与 ArcticInference 团队合作添加多分支支持

## 相关文件

- `/wxyworkspace/python/sglang/srt/speculative/suffix_cache_adapter.py` - SUFFIX 适配器
- `/wxyworkspace/python/sglang/srt/speculative/suffix_worker.py` - SUFFIX Worker
- `/usr/local/lib/python3.12/dist-packages/arctic_inference/suffix_decoding/cache.py` - ArcticInference 缓存
