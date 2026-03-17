# Suffix Decoding 真实实验报告

## 实验概述
- **实验时间**: 2026-03-12
- **模型**: Qwen2.5-VL-32B-Instruct (32B, 4x GPU)
- **数据集**: 电商视频内容分析任务 (2000 samples)
- **测试样本**: 5 warmup + 20 evaluation requests
- **Max prompt 长度**: 2000 字符
- **Max output tokens**: 200

## 实验配置
| 配置 | use_tree_spec | max_branch_factor | 说明 |
|-----|---------------|-------------------|------|
| baseline-no-spec | False | 1 | 无 speculation 基准 |
| suffix-single-path | False | 1 | SUFFIX 单分支模式 |
| suffix-tree-branch-3 | True | 3 | SUFFIX 多分支 (branch=3) |
| suffix-tree-branch-5 | True | 5 | SUFFIX 多分支 (branch=5) |

## 实验结果

### 吞吐量对比
| 配置 | 吞吐量 (tokens/sec) | 平均延迟 (s) | 相对提升 |
|-----|---------------------|-------------|----------|
| baseline-no-spec | 34.14 | 4.38 | 0% (基准) |
| suffix-single-path | 69.43 | 2.14 | **+103.4%** |
| suffix-tree-branch-3 | 64.88 | 2.30 | +90.1% |
| suffix-tree-branch-5 | 60.06 | 2.83 | +75.9% |

### 接受率和接受长度（从服务器日志提取）
根据服务器日志，典型的 accept rate 约为 0.12-0.22，accept len 约为 2-3.5 tokens

例如:
- `accept len: 2.92, accept rate: 0.18` (branch=3)
- `accept len: 3.58, accept rate: 0.22` (branch=5)

## 关键发现

### 1. SUFFIX 大幅提升吞吐量
- **单分支模式提升 103.4%**，几乎翻倍！
- 所有 SUFFIX 配置都显著优于基准

### 2. 单分支 vs 多分支
**意外发现**：单分支模式表现最好！

可能原因：
1. **验证开销**：多分支需要验证更多 draft tokens，2. **概率分散**：分支越多，单个路径的概率越低
3. **数据集特点**：电商视频分析任务的 prompt 模式相对固定
   - 单分支能更好地利用这种重复模式

### 3. 多分支确实正确实现
从代码分析：
- `DeepTreeBuilder` 类实现了真正的多分支树构建
- `_merge_independent_roots` 方法将多个独立根合并成多分支树
- `max_branch_factor` 参数有效控制分支数量

从服务器日志可见：
- 使用了 `--speculative-suffix-use-tree-spec` 和 `--speculative-suffix-max-branch-factor 3/5`
- 日志显示真实的 accept_rate 和 accept_length

## 结论

1. **SUFFIX speculation 效果显著**：单分支模式可获得 ~100% 的吞吐量提升

2. **对于此数据集**：单分支模式优于多分支模式
   - 如果 prompt/response 模式高度重复，单分支足够
   - 多分支更适合高度多样化的场景

3. **建议**：
   - 首先尝试单分支模式
   - 对于模式重复的任务效果更好
   - 多分支模式在处理复杂、多样化任务时可能有优势

## 下一步优化建议
1. 测试更多样的数据集
2. 调整 `min_token_prob` 阈值
3. 增加 `max_tree_depth` 深度
4. 测试不同的 `draft_token_num`
