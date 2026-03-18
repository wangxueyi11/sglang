"""
Cache adapter that wraps the suffix decoding backend cache to provide
the same interface as NgramCache.

This allows NGRAMWorker to use suffix decoding without modification.
"""

import logging
import os
from collections import deque, defaultdict
from typing import List, Optional, Tuple, Dict, Set

import numpy as np

logger = logging.getLogger(__name__)


class MultiBranchTreeBuilder:
    """
    真正的多分支树构建器
    
    ArcticInference 的 use_tree_spec=True 不返回多分支，
    所以我们维护自己的分支缓存来构建真正的多分支树。
    
    工作原理：
    1. 维护一个 n-gram 到后续 token 的映射
    2. 当添加新的 token 序列时，更新这个映射
    3. 在推测时，使用 BFS 构建多分支树
    """
    
    def __init__(
        self,
        max_depth: int = 8,
        max_branch_factor: int = 3,
        ngram_size: int = 4,
    ):
        self.max_depth = max_depth
        self.max_branch_factor = max_branch_factor
        self.ngram_size = ngram_size
        
        # 核心：n-gram -> {token: count} 映射
        # 记录每个 n-gram 后面可能出现的 token 及其频率
        self._branches: Dict[tuple, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        
        # 记录完整的 token 序列（用于深度匹配）
        self._sequences: List[List[int]] = []
        
        # 统计
        self._total_updates = 0
    
    def add_sequence(self, tokens: List[int]) -> None:
        """添加一个 token 序列，更新分支缓存"""
        if len(tokens) < 2:
            return
        
        self._sequences.append(tokens.copy())
        # 只保留最近的序列以节省内存
        if len(self._sequences) > 10000:
            self._sequences = self._sequences[-5000:]
        
        # 更新 n-gram 分支映射
        for n in range(1, self.ngram_size + 1):
            for i in range(len(tokens) - n):
                ngram = tuple(tokens[i:i+n])
                next_token = tokens[i+n]
                self._branches[ngram][next_token] += 1
        
        self._total_updates += 1
    
    def get_branches(self, context: List[int], max_depth: int, max_branches: int) -> List[Tuple[int, float]]:
        """
        获取给定 context 后的所有可能分支
        
        返回: [(token, probability), ...]
        """
        branches = []
        
        # 尝试不同长度的 n-gram 匹配
        for n in range(min(self.ngram_size, len(context)), 0, -1):
            ngram = tuple(context[-n:])
            if ngram in self._branches:
                token_counts = self._branches[ngram]
                total = sum(token_counts.values())
                if total > 0:
                    # 按频率排序
                    sorted_tokens = sorted(token_counts.items(), key=lambda x: -x[1])
                    for token, count in sorted_tokens[:max_branches]:
                        prob = count / total
                        branches.append((token, prob))
                    break
        
        return branches
    
    def build_tree(
        self,
        context: List[int],
        max_tokens: int,
    ) -> Tuple[List[int], List[int], float]:
        """
        使用 BFS 构建多分支树
        
        返回:
            token_ids: 树中的所有 token
            parents: 每个 token 的父节点索引
            score: 树的总分数
        """
        if max_tokens <= 0:
            return [], [], 0.0
        
        token_ids = []
        parents = []
        total_score = 0.0
        
        # 获取根节点的分支
        root_branches = self.get_branches(context, self.max_depth, self.max_branch_factor)
        
        if not root_branches:
            return token_ids, parents, total_score
        
        # BFS 队列: (token, parent_idx, depth, cumulative_prob)
        queue = deque()
        
        # 添加根节点的分支（多分支根节点）
        used_at_root = set()
        for token, prob in root_branches:
            if token not in used_at_root and len(token_ids) < max_tokens:
                current_idx = len(token_ids)
                token_ids.append(token)
                parents.append(-1)  # 根节点的父节点是 -1
                total_score += prob
                used_at_root.add(token)
                # 加入队列继续探索：(当前token索引, 当前上下文, 深度, 累积概率)
                queue.append((current_idx, context + [token], 1, prob))
        
        # BFS 探索 - 每个节点可以有多个子节点
        visited_states = set()
        
        while queue and len(token_ids) < max_tokens:
            parent_idx, ctx, depth, cum_prob = queue.popleft()
            
            if depth >= self.max_depth:
                continue
            
            # 获取当前上下文的分支
            branches = self.get_branches(ctx, self.max_depth - depth, self.max_branch_factor)
            
            if not branches:
                continue
            
            # 为当前节点添加多个子节点（这是多分支的关键！）
            used_tokens = set()
            for token, prob in branches:
                if len(token_ids) >= max_tokens:
                    break
                if token in used_tokens:
                    continue
                
                # 检查是否已经处理过这个状态
                state_key = (parent_idx, token)
                if state_key in visited_states:
                    continue
                visited_states.add(state_key)
                
                used_tokens.add(token)
                
                # 添加新节点，其父节点是 parent_idx（不是最后一个节点！）
                current_idx = len(token_ids)
                token_ids.append(token)
                parents.append(parent_idx)  # 正确设置父节点
                total_score += prob * (0.9 ** depth)  # 深度衰减
                
                # 继续探索这个分支
                queue.append((current_idx, ctx + [token], depth + 1, cum_prob * prob))
        
        return token_ids, parents, total_score
    
    def stats(self) -> dict:
        """返回统计信息"""
        return {
            "total_updates": self._total_updates,
            "total_ngrams": len(self._branches),
            "total_sequences": len(self._sequences),
        }


class DeepTreeBuilder:
    """
    多分支树构建器包装类
    
    结合 ArcticInference 的推测和本地分支缓存来构建多分支树
    """
    
    def __init__(
        self,
        suffix_cache,
        max_depth: int = 8,
        max_branch_factor: int = 3,
        min_token_prob: float = 0.1,
        max_spec_factor: float = 1.0,
        max_tree_depth: int = 24,
    ):
        self.suffix_cache = suffix_cache
        self.max_depth = max_depth
        self.max_branch_factor = max_branch_factor
        self.min_token_prob = min_token_prob
        self.max_spec_factor = max_spec_factor
        self.max_tree_depth = max_tree_depth
        
        # 本地分支缓存
        self.branch_cache = MultiBranchTreeBuilder(
            max_depth=max_depth,
            max_branch_factor=max_branch_factor,
            ngram_size=min(6, max_tree_depth),
        )
    
    def add_sequence(self, tokens: List[int]) -> None:
        """添加 token 序列到分支缓存"""
        self.branch_cache.add_sequence(tokens)
    
    def build_deep_tree(
        self,
        cache_req_id: str,
        context: List[int],
        max_tokens: int,
    ) -> Tuple[List[int], List[int], float]:
        """
        构建多分支推测树
        
        策略：
        1. 先尝试使用本地分支缓存构建多分支树
        2. 如果本地缓存数据不足，则使用 ArcticInference 的单路径
        3. 如果两者都有数据，选择更好的结果
        """
        if max_tokens <= 0:
            return [], [], 0.0
        
        # 1. 使用本地分支缓存构建多分支树
        local_tokens, local_parents, local_score = self.branch_cache.build_tree(
            context, max_tokens
        )
        
        # 分析本地树结构
        from collections import Counter
        parent_counts = Counter(local_parents)
        multi_child = {p: c for p, c in parent_counts.items() if c > 1}
        root_count = sum(1 for p in local_parents if p == -1)
        
        branch_stats = self.branch_cache.stats()
        
        # 如果本地树有足够的分支（至少2个根分支或多子节点），直接使用
        if root_count >= 2 or len(multi_child) > 0:
            logger.warning(
                "[MULTI-BRANCH] Using local multi-branch tree: "
                "tokens=%d, roots=%d, multi_child_parents=%s, ngrams=%d",
                len(local_tokens), root_count, multi_child, branch_stats["total_ngrams"]
            )
            return local_tokens, local_parents, local_score
        
        # 2. 获取 ArcticInference 的主路径作为补充
        #    注意：如果请求不 active，跳过这一步并使用本地结果
        main_tokens = []
        main_score = 0.0
        try:
            if cache_req_id in self.suffix_cache.active_requests:
                main_draft = self.suffix_cache.speculate(
                    cache_req_id,
                    context,
                    max_spec_tokens=max_tokens,
                    max_spec_factor=self.max_spec_factor,
                    min_token_prob=self.min_token_prob,
                    use_tree_spec=False,
                )
                main_tokens = list(main_draft.token_ids) if main_draft.token_ids else []
                main_score = main_draft.score if hasattr(main_draft, 'score') else 0.0
        except (ValueError, KeyError) as e:
            # 请求不 active 或其他错误，使用本地结果
            logger.warning("[SUFFIX] speculate failed for %s: %s, using local cache", cache_req_id, e)
        
        # 3. 决定使用哪个结果
        # 如果本地树有数据但分支不够多，仍然使用（因为可能有更好的匹配）
        if len(local_tokens) > 0:
            logger.warning(
                "[MULTI-BRANCH] Using local tree (limited branches): "
                "tokens=%d, roots=%d, ngrams=%d",
                len(local_tokens), root_count, branch_stats["total_ngrams"]
            )
            return local_tokens, local_parents, local_score
        
        # 使用 ArcticInference 的单路径
        if main_tokens:
            parents = [-1] + list(range(len(main_tokens) - 1))
            logger.warning(
                "[SUFFIX SINGLE] Using ArcticInference path: tokens=%d, score=%.3f",
                len(main_tokens), main_score
            )
            return main_tokens, parents, main_score
        
        return [], [], 0.0
    
    def _compute_depth(self, parents: List[int]) -> int:
        """计算树的最大深度"""
        if not parents:
            return 0
        depths = {}
        def get_depth(i):
            if i in depths:
                return depths[i]
            if parents[i] == -1:
                depths[i] = 1
            else:
                depths[i] = get_depth(parents[i]) + 1
            return depths[i]
        
        max_depth = 0
        for i in range(len(parents)):
            max_depth = max(max_depth, get_depth(i))
        return max_depth


class SuffixCacheAdapter:
    """
    Adapter that wraps SuffixDecodingCache to match NgramCache interface.

    NGRAMWorker expects:
    - batch_get(batch_tokens: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]
      Returns (draft_tokens, tree_mask) as flat numpy arrays
    - batch_put(batch_tokens: List[List[int]]) -> None
      Updates cache with verified tokens
    - synchronize() -> None
      No-op for suffix cache
    - reset() -> None
      Clears all cached data
    """

    def __init__(
        self,
        draft_token_num: int,
        max_batch_size: int,
        max_tree_depth: int = 24,
        max_cached_requests: int = 10000,
        max_spec_factor: float = 1.0,
        min_token_prob: float = 0.1,
        use_tree_spec: bool = False,
        max_branch_factor: int = 3,  # Max branches per level for tree speculation
    ):
        """
        Args:
            draft_token_num: Fixed number of draft tokens (for padding)
            max_tree_depth: Maximum depth for suffix tree
            max_cached_requests: Maximum number of cached requests
            max_spec_factor: Maximum speculation factor
            min_token_prob: Minimum token probability threshold
            use_tree_spec: If True, use multi-branch tree speculation.
            max_branch_factor: Max number of branches to explore at each level
        """
        # Lazy import to avoid error when Suffix Decoding is not used
        from arctic_inference.suffix_decoding import SuffixDecodingCache

        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=max_tree_depth,
            max_cached_requests=max_cached_requests,
        )
        self.draft_token_num = draft_token_num
        self.max_batch_size = max_batch_size
        self.max_tree_depth = max_tree_depth
        self.max_spec_factor = max_spec_factor
        self.min_token_prob = min_token_prob
        self.use_tree_spec = use_tree_spec
        self.max_branch_factor = max_branch_factor

        # Create deep tree builder for tree speculation mode
        if use_tree_spec:
            self.deep_tree_builder = DeepTreeBuilder(
                self.suffix_cache,
                max_depth=draft_token_num,
                max_branch_factor=max_branch_factor,
                min_token_prob=min_token_prob,
                max_spec_factor=max_spec_factor,
                max_tree_depth=max_tree_depth,
            )
        else:
            self.deep_tree_builder = None

        # Debug toggles (set env e.g. SUFFIX_DEBUG_TREE=1 to dump first batch)
        self.debug_tree_dump_remaining = int(os.environ.get("SUFFIX_DEBUG_TREE", "0"))

        # 统计计数器
        self.stats_total_steps = 0
        self.stats_total_draft_generated = 0  # 实际生成的 draft tokens（不含 padding）
        self.stats_total_draft_padded = 0     # padding 的数量
        self.stats_total_valid_draft = 0      # 非零的 draft tokens
        self.stats_last_print_time = 0

        # Track state by SGlang request ID (stable identifier)
        # Map: sglang_req_id → (arctic_req_id, last_length)
        self.req_state = {}

        # Preallocate buffers to avoid per-step allocations
        self.max_total_drafts = self.max_batch_size * self.draft_token_num
        self.draft_buffer = np.empty((self.max_total_drafts,), dtype=np.int64)
        self.mask_buffer = np.empty(
            (self.max_batch_size, self.draft_token_num, self.draft_token_num),
            dtype=bool,
        )

    def _cleanup_inactive_requests(self, active_req_ids: set):
        """Stop backend requests that are no longer active in SGlang."""
        inactive_req_ids = [
            rid for rid in self.req_state.keys() if rid not in active_req_ids
        ]
        for rid in inactive_req_ids:
            cache_req_id, _ = self.req_state.pop(rid)
            if cache_req_id in getattr(self.suffix_cache, "active_requests", set()):
                self.suffix_cache.stop_request(cache_req_id)

    def _get_or_create_cache_req_id(
        self, sglang_req_id: str, prompt: List[int], tokens: List[int]
    ) -> tuple:
        """Get or create a backend request ID for the given SGlang request.

        Args:
            sglang_req_id: Stable request ID from SGlang
            prompt: Prompt tokens only (no generated tokens)
            tokens: Full token sequence (prompt + outputs)

        Returns: (arctic_req_id, last_length)
        """
        if sglang_req_id not in self.req_state:
            # Use SGlang request ID directly as backend request ID
            cache_req_id = sglang_req_id

            # Initialize the request in suffix cache with ONLY the prompt
            self.suffix_cache.start_request(cache_req_id, prompt)

            # Track: [arctic_req_id, last_length]
            # IMPORTANT: Set last_length to prompt length since the backend already has the prompt
            self.req_state[sglang_req_id] = [cache_req_id, len(prompt)]

            # 添加 prompt 到分支缓存
            if self.deep_tree_builder:
                self.deep_tree_builder.add_sequence(prompt)

        cache_req_id, last_length = self.req_state[sglang_req_id]
        return cache_req_id, last_length

    def batch_get(
        self,
        batch_req_ids: List[str],
        batch_prompts: List[List[int]],
        batch_tokens: List[List[int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get draft tokens for a batch of token sequences.

        This is called BEFORE verification with the current state.
        We speculate based on the current tokens.

        Args:
            batch_req_ids: List of SGlang request IDs (stable)
            batch_prompts: List of prompt tokens (no generated tokens)
            batch_tokens: List of token sequences (prompt + output tokens)

        Returns:
            Tuple of:
            - draft_tokens: np.ndarray of shape (batch_size * draft_token_num,)
            - tree_mask: np.ndarray of shape (batch_size * draft_token_num * draft_token_num,)
        """
        batch_size = len(batch_req_ids)
        if batch_size == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=bool)

        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds configured max_batch_size={self.max_batch_size}"
            )

        total_draft_tokens = batch_size * self.draft_token_num
        draft_view = self.draft_buffer[:total_draft_tokens]
        mask_view = self.mask_buffer[:batch_size]
        mask_view.fill(False)

        active_req_ids = set(batch_req_ids)
        self._cleanup_inactive_requests(active_req_ids)

        for idx, (sglang_req_id, prompt, tokens) in enumerate(
            zip(batch_req_ids, batch_prompts, batch_tokens)
        ):
            cache_req_id, last_length = self._get_or_create_cache_req_id(
                sglang_req_id, prompt, tokens
            )

            # Ensure cache includes the latest verified tokens before speculation.
            current_length = len(tokens)
            if current_length > last_length:
                new_tokens = tokens[last_length:current_length]
                if cache_req_id in self.suffix_cache.active_requests:
                    self.suffix_cache.add_active_response(cache_req_id, new_tokens)
                    self.req_state[sglang_req_id][1] = current_length
                    last_length = current_length
                    
                    # 添加新 tokens 到分支缓存
                    if self.deep_tree_builder:
                        self.deep_tree_builder.add_sequence(tokens)

            # Extract pattern from end of tokens (up to max_tree_depth)
            pattern_start = max(0, len(tokens) - self.max_tree_depth)
            pattern = tokens[pattern_start:]

            # Speculate using suffix cache
            if self.use_tree_spec and self.deep_tree_builder is not None:
                # 使用多分支树构建
                draft_ids, draft_parents, score = self.deep_tree_builder.build_deep_tree(
                    cache_req_id,
                    pattern,
                    max_tokens=self.draft_token_num,
                )
                
                # 分析树结构
                children_count = {}
                for i in range(len(draft_parents)):
                    p = draft_parents[i]
                    if p >= 0 and p < len(draft_parents):
                        children_count[p] = children_count.get(p, 0) + 1
                max_branches = max(children_count.values()) if children_count else 0
                tree_depth = self._compute_max_depth(draft_parents)
                
                logger.warning(
                    "[SUFFIX DEBUG DEEP TREE] use_tree_spec=%s, max_branch_factor=%d, "
                    "tokens=%d, depth=%d, max_branches=%d, parents=%s",
                    self.use_tree_spec, self.max_branch_factor,
                    len(draft_ids), tree_depth, max_branches,
                    draft_parents[:20] if len(draft_parents) > 20 else draft_parents,
                )
            else:
                # Use single-path speculation
                draft = self.suffix_cache.speculate(
                    cache_req_id,
                    pattern,
                    max_spec_tokens=self.draft_token_num,
                    max_spec_factor=self.max_spec_factor,
                    min_token_prob=self.min_token_prob,
                    use_tree_spec=False,
                )
                draft_ids = list(draft.token_ids)
                draft_parents = list(draft.parents)
                
                if self.debug_tree_dump_remaining > 0:
                    logger.warning(
                        "[SUFFIX DEBUG] draft_tokens=%d, "
                        "token_ids=%s, parents=%s, match_len=%d, score=%.3f",
                        len(draft_ids),
                        draft_ids[:20] if len(draft_ids) > 20 else draft_ids,
                        draft_parents[:20] if len(draft_parents) > 20 else draft_parents,
                        draft.match_len,
                        draft.score,
                    )
            draft_ids, draft_parents = self._reorder_tree_bfs(draft_ids, draft_parents)

            context_token = tokens[-1] if tokens else 0
            draft_ids, draft_parents = self._inject_root_node(
                draft_ids, draft_parents, context_token
            )
            
            # Debug: 打印注入后的树结构
            if self.use_tree_spec and self.debug_tree_dump_remaining > 0:
                children_count = {}
                for i in range(len(draft_parents)):
                    p = draft_parents[i]
                    if p >= 0 and p < len(draft_parents):
                        children_count[p] = children_count.get(p, 0) + 1
                max_branches_after = max(children_count.values()) if children_count else 0
                logger.warning(
                    "[SUFFIX DEBUG AFTER INJECT] tokens=%d, max_branches=%d, parents=%s, context_token=%d",
                    len(draft_ids), max_branches_after, draft_parents[:20], context_token
                )

            # Pad or truncate to match draft_token_num (includes root node at index 0)
            original_draft_len = len(draft_ids)
            pad_len = 0
            if original_draft_len < self.draft_token_num:
                pad_len = self.draft_token_num - original_draft_len
                draft_ids.extend([0] * pad_len)
                draft_parents.extend([0] * pad_len)
            elif original_draft_len > self.draft_token_num:
                draft_ids = draft_ids[: self.draft_token_num]
                draft_parents = draft_parents[: self.draft_token_num]
                original_draft_len = self.draft_token_num
            
            # 统计：计算有效 draft tokens（非零）
            valid_draft_count = sum(1 for t in draft_ids if t != 0)
            
            # 更新全局统计
            self.stats_total_steps += 1
            self.stats_total_draft_generated += original_draft_len
            self.stats_total_draft_padded += pad_len
            self.stats_total_valid_draft += valid_draft_count
            
            # 每100步打印一次统计
            if self.stats_total_steps % 100 == 0:
                avg_generated = self.stats_total_draft_generated / self.stats_total_steps
                avg_padded = self.stats_total_draft_padded / self.stats_total_steps
                avg_valid = self.stats_total_valid_draft / self.stats_total_steps
                logger.warning(
                    "[SUFFIX STATS] step=%d: draft_token_num_config=%d, "
                    "avg_generated=%.2f, avg_padded=%.2f, avg_valid=%.2f",
                    self.stats_total_steps, self.draft_token_num,
                    avg_generated, avg_padded, avg_valid
                )

            start = idx * self.draft_token_num
            end = start + self.draft_token_num
            draft_view[start:end] = draft_ids

            # Build tree mask from parent structure
            # Token i can attend to token j if j is an ancestor of i
            mask = mask_view[idx]
            if original_draft_len > 0:
                for i in range(original_draft_len):
                    # Token can attend to itself
                    mask[i, i] = True
                    # And to all ancestors
                    p = draft_parents[i]
                    while p >= 0 and p < self.draft_token_num:
                        mask[i, p] = True
                        p = draft_parents[p] if p < len(draft_parents) else -1

        return draft_view, mask_view.flatten()

    def batch_put(self, batch_req_ids: List[str], batch_tokens: List[List[int]]) -> None:
        """Update cache with verified tokens.
        
        Args:
            batch_req_ids: List of request IDs (not used by suffix cache)
            batch_tokens: List of full token sequences (prompt + output)
        """
        # Tokens are already added in batch_get via add_active_response
        # But we can add them to the branch cache for future speculation
        if self.deep_tree_builder:
            for tokens in batch_tokens:
                self.deep_tree_builder.add_sequence(tokens)

    def synchronize(self) -> None:
        """No-op for suffix cache (no async operations)."""
        pass

    def reset(self) -> None:
        """Clear all cached data."""
        self.suffix_cache = type(self.suffix_cache)(
            max_tree_depth=self.max_tree_depth,
        )
        self.req_state.clear()
        if self.deep_tree_builder:
            self.deep_tree_builder.branch_cache = MultiBranchTreeBuilder(
                max_depth=self.draft_token_num,
                max_branch_factor=self.max_branch_factor,
            )

    def _compute_max_depth(self, parents: List[int]) -> int:
        """Compute the maximum depth of the tree."""
        if not parents:
            return 0
        
        depths = {}
        def get_depth(i: int) -> int:
            if i in depths:
                return depths[i]
            if parents[i] == -1:
                depths[i] = 1
            else:
                depths[i] = get_depth(parents[i]) + 1
            return depths[i]
        
        max_depth = 0
        for i in range(len(parents)):
            max_depth = max(max_depth, get_depth(i))
        return max_depth

    def _reorder_tree_bfs(
        self, token_ids: List[int], parents: List[int]
    ) -> Tuple[List[int], List[int]]:
        """Reorder tree in BFS order for efficient verification."""
        if not token_ids:
            return token_ids, parents

        # Build adjacency list
        children = defaultdict(list)
        for i, p in enumerate(parents):
            children[p].append(i)

        # BFS traversal
        new_ids = []
        new_parents = []
        old_to_new = {}
        queue = deque(children[-1])  # Start with root children

        while queue:
            old_idx = queue.popleft()
            old_to_new[old_idx] = len(new_ids)
            new_ids.append(token_ids[old_idx])

            # Add children to queue
            for child in children.get(old_idx, []):
                queue.append(child)

        # Compute new parent indices
        for old_idx in range(len(token_ids)):
            if old_idx in old_to_new:
                new_idx = old_to_new[old_idx]
                old_parent = parents[old_idx]
                if old_parent == -1:
                    new_parents.append(-1)
                elif old_parent in old_to_new:
                    new_parents.append(old_to_new[old_parent])
                else:
                    new_parents.append(-1)

        return new_ids, new_parents

    def _inject_root_node(
        self, token_ids: List[int], parents: List[int], context_token: int
    ) -> Tuple[List[int], List[int]]:
        """Inject a root node containing the context token.
        
        This method ensures context_token becomes the single root of the tree.
        All original roots become children of the new context_token root.
        This is necessary for proper tree verification where the first token
        must match the last verified token.
        
        Args:
            token_ids: List of draft token IDs
            parents: Parent indices for each token (-1 for root)
            context_token: The last verified token to use as root
            
        Returns:
            Updated (token_ids, parents) with context_token as root
        """
        if not token_ids:
            return [context_token], [-1]

        # Find all root nodes
        roots = [i for i, p in enumerate(parents) if p == -1]
        
        # Case 1: Single root with context_token - no change needed
        if len(roots) == 1 and token_ids[roots[0]] == context_token:
            return token_ids, parents
        
        # Case 2: Check if one of multiple roots is context_token
        if len(roots) > 1:
            for root_idx in roots:
                if token_ids[root_idx] == context_token:
                    # context_token already exists as a root, use it as the single root
                    # and make other roots its children (for consistent tree structure)
                    # This preserves multi-branch: all roots become siblings under context_token
                    pass  # Fall through to inject context_token as parent of all roots
        
        # Inject context_token as the single root
        # All original roots become children of context_token
        new_ids = [context_token] + list(token_ids)
        new_parents = [-1]  # context_token is the new root
        for p in parents:
            if p == -1:
                # Original roots become children of new root (index 0)
                new_parents.append(0)
            else:
                # Non-root nodes: parent index shifts by 1
                new_parents.append(p + 1)
        return new_ids, new_parents