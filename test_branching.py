#!/usr/bin/env python3
"""
Direct test to demonstrate use_tree_spec branching.

This test explicitly creates scenarios where the suffix tree has multiple
possible continuations from the same context.
"""

import numpy as np
from arctic_inference.suffix_decoding import SuffixDecodingCache


def test_branching_scenario():
    """
    Create a scenario where the same context has multiple possible continuations.
    
    Example:
    - Context: [1, 2, 3]
    - Continuation 1: [4, 5, 6] (appears 5 times)
    - Continuation 2: [4, 7, 8] (appears 3 times)
    - Continuation 3: [9, 10, 11] (appears 2 times)
    
    With use_tree_spec=True, we should see multiple branches.
    """
    print("=" * 70)
    print("Test: Branching Scenario with use_tree_spec")
    print("=" * 70)
    
    cache = SuffixDecodingCache(max_tree_depth=32, max_cached_requests=100)
    
    # Common context that will be shared
    context = np.array([1, 2, 3], dtype=np.int32)
    
    # Build cache with multiple continuations from the same context
    continuations = [
        ([4, 5, 6, 100, 101], 5),   # Continuation A, appears 5 times
        ([4, 7, 8, 102, 103], 3),   # Continuation B, appears 3 times (shares first token with A)
        ([9, 10, 11, 104, 105], 2), # Continuation C, appears 2 times (different first token)
    ]
    
    req_id = 0
    for cont, count in continuations:
        for i in range(count):
            # Each request has the same context followed by different continuation
            full_seq = np.concatenate([context, np.array(cont, dtype=np.int32)])
            cache.start_request(req_id, full_seq[:3])  # Start with first 3 tokens
            cache.add_active_response(req_id, full_seq[3:])  # Add rest
            cache.stop_request(req_id)
            req_id += 1
    
    print(f"Built cache with {req_id} requests")
    print(f"Context: {context.tolist()}")
    print(f"Continuations:")
    for cont, count in continuations:
        print(f"  {cont} (x{count})")
    
    # Now test speculation with use_tree_spec=False
    print("\n" + "-" * 70)
    print("use_tree_spec=False")
    print("-" * 70)
    
    # Start a test request
    test_req_id = 999
    cache.start_request(test_req_id, context)
    
    draft_path = cache.speculate(
        test_req_id,
        context,
        max_spec_tokens=10,
        max_spec_factor=2.0,
        min_token_prob=0.01,
        use_tree_spec=False,
    )
    
    print(f"Tokens: {draft_path.token_ids}")
    print(f"Parents: {draft_path.parents}")
    print(f"Probs: {[f'{p:.3f}' for p in draft_path.probs] if draft_path.probs else 'None'}")
    print(f"Score: {draft_path.score:.4f}")
    print(f"Match len: {draft_path.match_len}")
    
    cache.stop_request(test_req_id)
    
    # Now test with use_tree_spec=True
    print("\n" + "-" * 70)
    print("use_tree_spec=True")
    print("-" * 70)
    
    cache.start_request(test_req_id, context)
    
    draft_tree = cache.speculate(
        test_req_id,
        context,
        max_spec_tokens=10,
        max_spec_factor=2.0,
        min_token_prob=0.01,
        use_tree_spec=True,
    )
    
    print(f"Tokens: {draft_tree.token_ids}")
    print(f"Parents: {draft_tree.parents}")
    print(f"Probs: {[f'{p:.3f}' for p in draft_tree.probs] if draft_tree.probs else 'None'}")
    print(f"Score: {draft_tree.score:.4f}")
    print(f"Match len: {draft_tree.match_len}")
    
    cache.stop_request(test_req_id)
    
    # Analyze the tree structure
    print("\n" + "-" * 70)
    print("Tree Analysis")
    print("-" * 70)
    
    roots = [i for i, p in enumerate(draft_tree.parents) if p < 0]
    print(f"Number of roots: {len(roots)}")
    
    # Build children map
    n = len(draft_tree.token_ids)
    children = {i: [] for i in range(n)}
    for i, p in enumerate(draft_tree.parents):
        if p >= 0 and p < n:
            children[p].append(i)
    
    print(f"Branching factors:")
    for i in range(min(10, n)):
        if children[i]:
            print(f"  Node {i} (token={draft_tree.token_ids[i]}): {len(children[i])} children -> {[draft_tree.token_ids[c] for c in children[i]]}")
    
    # Key comparison
    print("\n" + "=" * 70)
    print("Key Comparison")
    print("=" * 70)
    
    if len(roots) > 1:
        print(f"SUCCESS: Tree mode produced {len(roots)} root branches!")
        print("This demonstrates the value of use_tree_spec=True")
        print("Multiple independent speculation paths are available for verification.")
    else:
        print("NOTE: Tree mode produced a single root.")
        print("This means only one continuation path had probability above threshold.")
    
    if draft_tree.score > draft_path.score:
        print(f"\nTree mode has HIGHER score: {draft_tree.score:.4f} > {draft_path.score:.4f}")
    elif draft_tree.score < draft_path.score:
        print(f"\nSingle-path mode has HIGHER score: {draft_path.score:.4f} > {draft_tree.score:.4f}")
    else:
        print(f"\nBoth modes have SAME score: {draft_path.score:.4f}")
    
    return draft_path, draft_tree


def test_with_real_text():
    """
    Test with more realistic text-like patterns.
    """
    print("\n" + "=" * 70)
    print("Test: Realistic Text Patterns")
    print("=" * 70)
    
    cache = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=1000)
    
    # Simulate common phrases
    # The pattern "the" can be followed by multiple words
    common_starts = [
        [100, 101, 102],  # "the "
    ]
    
    continuations = [
        [200, 201, 202],  # "quick"
        [200, 203, 204],  # "quiet"
        [205, 206, 207],  # "slow"
        [208, 209, 210],  # "big"
    ]
    
    # Different frequencies for different continuations
    frequencies = [10, 5, 3, 2]
    
    req_id = 0
    for start in common_starts:
        for (cont, freq) in zip(continuations, frequencies):
            for _ in range(freq):
                full_seq = np.array(start + cont, dtype=np.int32)
                cache.start_request(req_id, full_seq[:2])
                cache.add_active_response(req_id, full_seq[2:])
                cache.stop_request(req_id)
                req_id += 1
    
    print(f"Built cache with {req_id} requests")
    print(f"Common start: {common_starts[0]}")
    print(f"Continuations with frequencies:")
    for cont, freq in zip(continuations, frequencies):
        print(f"  {cont} (x{freq})")
    
    # Test context
    test_context = np.array([100, 101], dtype=np.int32)  # Partial match to "the "
    
    print(f"\nTest context: {test_context.tolist()}")
    
    # Test both modes
    test_req_id = 999
    
    # Single-path
    cache.start_request(test_req_id, test_context)
    draft_path = cache.speculate(test_req_id, test_context, max_spec_tokens=8, 
                                  min_token_prob=0.01, use_tree_spec=False)
    cache.stop_request(test_req_id)
    
    print(f"\nSingle-path mode:")
    print(f"  Tokens: {draft_path.token_ids}")
    print(f"  Score: {draft_path.score:.4f}")
    
    # Tree
    cache.start_request(test_req_id, test_context)
    draft_tree = cache.speculate(test_req_id, test_context, max_spec_tokens=8,
                                  min_token_prob=0.01, use_tree_spec=True)
    cache.stop_request(test_req_id)
    
    print(f"\nTree mode:")
    print(f"  Tokens: {draft_tree.token_ids}")
    print(f"  Parents: {draft_tree.parents}")
    print(f"  Probs: {[f'{p:.3f}' for p in draft_tree.probs] if draft_tree.probs else 'None'}")
    print(f"  Score: {draft_tree.score:.4f}")
    
    roots = [i for i, p in enumerate(draft_tree.parents) if p < 0]
    print(f"  Num roots: {len(roots)}")
    
    return draft_path, draft_tree


if __name__ == "__main__":
    test_branching_scenario()
    test_with_real_text()
