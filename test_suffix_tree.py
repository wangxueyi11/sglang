#!/usr/bin/env python3
"""
Test script to verify suffix decoding tree structure.

This tests:
1. DeepTreeBuilder._apply_topk_limit correctly limits branches
2. _reorder_tree_bfs correctly reorders tree
3. _inject_root_node correctly injects root
4. Tree mask is correctly built from parents
"""

import sys
from collections import deque
from typing import List, Tuple, Dict, Optional


def apply_topk_limit(
    token_ids: List[int],
    parents: List[int],
    probs: List[float],
    max_branch_factor: int,
) -> Tuple[List[int], List[int], float]:
    """
    Apply top-k branch limiting to the tree.
    """
    if not token_ids:
        return token_ids, parents, 0.0
    
    n = len(token_ids)
    
    # Build children mapping
    children: Dict[int, List[Tuple[int, float]]] = {}
    root_indices: List[int] = []
    
    for i in range(n):
        children[i] = []
    
    for i, parent in enumerate(parents):
        if parent < 0 or parent >= n:
            root_indices.append(i)
        else:
            children[parent].append((i, probs[i] if i < len(probs) else 0.0))
    
    # Sort and keep top-k
    nodes_to_keep = set(root_indices)
    
    def add_subtree(node_idx: int):
        if node_idx not in children:
            return
        child_list = children[node_idx]
        child_list.sort(key=lambda x: x[1], reverse=True)
        for child_idx, _ in child_list[:max_branch_factor]:
            nodes_to_keep.add(child_idx)
            add_subtree(child_idx)
    
    for root_idx in root_indices:
        add_subtree(root_idx)
    
    # Build new arrays
    old_to_new = {}
    new_token_ids = []
    new_parents = []
    new_score = 0.0
    
    sorted_nodes = sorted(nodes_to_keep)
    for new_idx, old_idx in enumerate(sorted_nodes):
        old_to_new[old_idx] = new_idx
        new_token_ids.append(token_ids[old_idx])
        if old_idx < len(probs):
            new_score += probs[old_idx]
        
        old_parent = parents[old_idx]
        if old_parent < 0 or old_parent not in nodes_to_keep:
            new_parents.append(-1)
        else:
            new_parents.append(old_to_new[old_parent])
    
    return new_token_ids, new_parents, new_score


def reorder_tree_bfs(
    token_ids: List[int], parents: List[Optional[int]]
) -> Tuple[List[int], List[int]]:
    """Reorder nodes so parents always precede their descendants."""
    n = len(token_ids)
    if n <= 1:
        return token_ids, parents

    children: List[List[int]] = [[] for _ in range(n)]
    roots: List[int] = []
    for idx, parent in enumerate(parents):
        if parent is None or parent < 0 or parent >= n:
            roots.append(idx)
        else:
            children[parent].append(idx)

    if not roots:
        roots = [0]

    order: List[int] = []
    visited = [False] * n
    for root in roots:
        if visited[root]:
            continue
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if visited[node]:
                continue
            visited[node] = True
            order.append(node)
            for child in children[node]:
                if not visited[child]:
                    queue.append(child)

    for idx in range(n):
        if not visited[idx]:
            order.append(idx)

    if order == list(range(n)):
        return token_ids, parents

    remap = {old_idx: new_idx for new_idx, old_idx in enumerate(order)}
    reordered_ids = [token_ids[old_idx] for old_idx in order]
    reordered_parents: List[int] = []
    for old_idx in order:
        parent = parents[old_idx]
        if parent is None or parent < 0:
            reordered_parents.append(-1)
        else:
            reordered_parents.append(remap.get(parent, -1))

    return reordered_ids, reordered_parents


def inject_root_node(
    token_ids: List[int], parents: List[int], context_token: int
) -> Tuple[List[int], List[int]]:
    """Insert the latest verified token as index 0."""
    rooted_ids = [context_token]
    rooted_parents = [-1]
    for parent_idx in parents:
        if parent_idx < 0:
            rooted_parents.append(0)
        else:
            rooted_parents.append(parent_idx + 1)
    rooted_ids.extend(token_ids)
    return rooted_ids, rooted_parents


def build_tree_mask(
    draft_token_num: int, parents: List[int]
) -> List[List[bool]]:
    """Build attention mask from parent structure."""
    mask = [[False] * draft_token_num for _ in range(draft_token_num)]
    for i in range(min(len(parents), draft_token_num)):
        mask[i][i] = True  # Self-attention
        parent_idx = parents[i]
        while parent_idx >= 0 and parent_idx < draft_token_num:
            mask[i][parent_idx] = True
            if parent_idx < len(parents):
                parent_idx = parents[parent_idx]
            else:
                break
    return mask


def print_tree(token_ids: List[int], parents: List[int], probs: List[float] = None):
    """Print tree structure."""
    n = len(token_ids)
    print(f"Tree (n={n}):")
    for i in range(n):
        prob_str = f", prob={probs[i]:.3f}" if probs and i < len(probs) else ""
        parent_str = str(parents[i]) if parents[i] >= 0 else "ROOT"
        print(f"  [{i}] token={token_ids[i]}, parent={parent_str}{prob_str}")
    print()


def test_topk_limit():
    """Test top-k branch limiting."""
    print("=" * 60)
    print("Test 1: Top-k branch limiting")
    print("=" * 60)
    
    # Create a tree with multiple branches
    # Root (0) -> children 1, 2, 3, 4, 5 (5 children)
    # Each child has further children
    token_ids = [100, 1, 2, 3, 4, 5, 11, 12, 21, 22, 31, 32, 41, 42, 51, 52]
    parents = [-1, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    probs = [1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.25, 0.2, 0.15, 0.1, 0.1, 0.08, 0.08, 0.05, 0.05, 0.03]
    
    print("Original tree:")
    print_tree(token_ids, parents, probs)
    
    # Apply top-k with k=2
    max_branch_factor = 2
    new_ids, new_parents, new_score = apply_topk_limit(token_ids, parents, probs, max_branch_factor)
    
    print(f"After top-{max_branch_factor} limiting:")
    print_tree(new_ids, new_parents)
    
    # Verify: each node should have at most max_branch_factor children
    children_count = {}
    for i, p in enumerate(new_parents):
        if p >= 0:
            children_count[p] = children_count.get(p, 0) + 1
    
    max_children = max(children_count.values()) if children_count else 0
    print(f"Max children per node: {max_children} (limit: {max_branch_factor})")
    assert max_children <= max_branch_factor, f"Top-k limit violated: {max_children} > {max_branch_factor}"
    print("PASSED!\n")


def test_reorder_bfs():
    """Test BFS reordering."""
    print("=" * 60)
    print("Test 2: BFS reordering")
    print("=" * 60)
    
    # Create a tree where parents don't precede children
    token_ids = [1, 2, 3, 4, 5]  # tokens
    parents = [2, 0, -1, 1, 1]   # 2 is root, 0 and 1 are children of 2, etc.
    
    print("Original order (parents may not precede children):")
    print_tree(token_ids, parents)
    
    new_ids, new_parents = reorder_tree_bfs(token_ids, parents)
    
    print("After BFS reordering:")
    print_tree(new_ids, new_parents)
    
    # Verify: all parents should precede their children
    for i, p in enumerate(new_parents):
        if p >= 0:
            assert p < i, f"Parent {p} does not precede child {i}"
    
    print("PASSED!\n")


def test_inject_root():
    """Test root injection."""
    print("=" * 60)
    print("Test 3: Root injection")
    print("=" * 60)
    
    # Create a tree with multiple roots (like ArcticInference returns)
    token_ids = [1, 2, 3, 4, 5]
    parents = [-1, -1, 0, 0, 1]  # Two roots: 0 and 1
    
    print("Original tree (two roots):")
    print_tree(token_ids, parents)
    
    context_token = 999
    new_ids, new_parents = inject_root_node(token_ids, parents, context_token)
    
    print(f"After injecting context token {context_token}:")
    print_tree(new_ids, new_parents)
    
    # Verify: new_ids[0] should be context_token, new_parents[0] should be -1
    assert new_ids[0] == context_token, f"Root token mismatch: {new_ids[0]} != {context_token}"
    assert new_parents[0] == -1, f"Root parent should be -1, got {new_parents[0]}"
    
    # Verify: old roots should now point to new root (index 0)
    for i, p in enumerate(parents):
        if p < 0:
            assert new_parents[i + 1] == 0, f"Old root {i} should point to new root 0"
    
    print("PASSED!\n")


def test_full_pipeline():
    """Test the full pipeline."""
    print("=" * 60)
    print("Test 4: Full pipeline simulation")
    print("=" * 60)
    
    # Simulate ArcticInference output (tree mode with multiple branches)
    token_ids = [10, 20, 30, 11, 12, 21, 22, 31, 32]
    parents = [-1, -1, -1, 0, 0, 1, 1, 2, 2]  # Three roots
    probs = [0.4, 0.3, 0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.08]
    draft_token_num = 16
    
    print("Step 1: ArcticInference output (tree mode)")
    print_tree(token_ids, parents, probs)
    
    # Step 2: Apply top-k
    max_branch_factor = 2
    token_ids, parents, score = apply_topk_limit(token_ids, parents, probs, max_branch_factor)
    print(f"Step 2: After top-{max_branch_factor} limiting")
    print_tree(token_ids, parents)
    
    # Step 3: Reorder BFS
    token_ids, parents = reorder_tree_bfs(token_ids, parents)
    print("Step 3: After BFS reordering")
    print_tree(token_ids, parents)
    
    # Step 4: Inject root
    context_token = 99
    token_ids, parents = inject_root_node(token_ids, parents, context_token)
    print(f"Step 4: After injecting context token {context_token}")
    print_tree(token_ids, parents)
    
    # Step 5: Build mask
    mask = build_tree_mask(draft_token_num, parents)
    print("Step 5: Tree mask (first 10x10):")
    for i in range(min(10, len(mask))):
        row = ''.join(['1' if mask[i][j] else '0' for j in range(min(10, draft_token_num))])
        print(f"  [{i}] {row}")
    
    # Verify mask properties
    # Each token should be able to attend to itself
    for i in range(min(len(parents), draft_token_num)):
        assert mask[i][i], f"Self-attention missing for token {i}"
    
    # Each token should be able to attend to all ancestors
    for i in range(min(len(parents), draft_token_num)):
        p = parents[i]
        while p >= 0 and p < draft_token_num:
            assert mask[i][p], f"Ancestor attention missing: {i} -> {p}"
            p = parents[p] if p < len(parents) else -1
    
    print("\nPASSED!\n")


if __name__ == "__main__":
    test_topk_limit()
    test_reorder_bfs()
    test_inject_root()
    test_full_pipeline()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
