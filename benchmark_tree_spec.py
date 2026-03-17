#!/usr/bin/env python3
"""
Benchmark script to compare use_tree_spec=True vs False.

This script:
1. Builds a suffix cache with some sample data
2. Runs speculation with use_tree_spec=True and False
3. Compares the draft tree structure, token count, and score
"""

import time
import numpy as np
from arctic_inference.suffix_decoding import SuffixDecodingCache


def generate_sample_data(num_requests: int, prompt_len: int, response_len: int, vocab_size: int = 32000):
    """Generate sample prompt-response pairs with repeating patterns."""
    np.random.seed(42)
    data = []
    
    # Create some common patterns that will appear multiple times
    common_patterns = [
        np.random.randint(0, vocab_size // 10, size=20).astype(np.int32)
        for _ in range(10)
    ]
    
    for i in range(num_requests):
        # Build prompt with repeating patterns
        prompt_parts = []
        remaining = prompt_len
        while remaining > 0:
            pattern = common_patterns[i % len(common_patterns)]
            take = min(remaining, len(pattern))
            prompt_parts.append(pattern[:take])
            remaining -= take
        prompt = np.concatenate(prompt_parts) if prompt_parts else np.array([], dtype=np.int32)
        
        # Build response with repeating patterns (different from prompt)
        response_parts = []
        remaining = response_len
        while remaining > 0:
            pattern = common_patterns[(i + 5) % len(common_patterns)]
            take = min(remaining, len(pattern))
            response_parts.append(pattern[:take])
            remaining -= take
        response = np.concatenate(response_parts) if response_parts else np.array([], dtype=np.int32)
        
        data.append((i, prompt, response))
    return data


def build_cache(data, max_tree_depth: int = 64, max_cached_requests: int = 1000):
    """Build suffix cache with sample data."""
    cache = SuffixDecodingCache(
        max_tree_depth=max_tree_depth,
        max_cached_requests=max_cached_requests,
    )
    
    for req_id, prompt, response in data:
        cache.start_request(req_id, prompt)
        cache.add_active_response(req_id, response)
        cache.stop_request(req_id)
    
    return cache


def benchmark_speculate(cache, context, max_spec_tokens: int, use_tree_spec: bool, 
                        max_spec_factor: float = 1.0, min_token_prob: float = 0.1, req_id: int = -1):
    """Run speculation and measure results."""
    # Start the request if not already active
    if req_id not in cache.active_requests:
        cache.start_request(req_id, context[:10])  # Use first 10 tokens as prompt
    
    start = time.perf_counter()
    draft = cache.speculate(
        req_id,
        context,
        max_spec_tokens=max_spec_tokens,
        max_spec_factor=max_spec_factor,
        min_token_prob=min_token_prob,
        use_tree_spec=use_tree_spec,
    )
    elapsed = time.perf_counter() - start
    
    # Stop the request after testing
    cache.stop_request(req_id)
    
    return draft, elapsed


def analyze_tree(token_ids, parents, probs=None):
    """Analyze tree structure."""
    if not token_ids:
        return {"num_tokens": 0, "max_depth": 0, "num_roots": 0, "avg_branching": 0}
    
    n = len(token_ids)
    
    # Find roots
    roots = [i for i, p in enumerate(parents) if p < 0]
    
    # Build children map
    children = {i: [] for i in range(n)}
    for i, p in enumerate(parents):
        if p >= 0:
            children[p].append(i)
    
    # Compute max depth
    def get_depth(node):
        if not children[node]:
            return 1
        return 1 + max(get_depth(c) for c in children[node])
    
    max_depth = max(get_depth(r) for r in roots) if roots else 0
    
    # Compute branching stats
    branch_counts = [len(children[i]) for i in range(n) if children[i]]
    avg_branching = sum(branch_counts) / len(branch_counts) if branch_counts else 0
    max_branching = max(branch_counts) if branch_counts else 0
    
    return {
        "num_tokens": n,
        "max_depth": max_depth,
        "num_roots": len(roots),
        "avg_branching": avg_branching,
        "max_branching": max_branching,
    }


def run_experiment():
    print("=" * 70)
    print("Benchmark: use_tree_spec=True vs False")
    print("=" * 70)
    
    # Parameters
    NUM_REQUESTS = 100
    PROMPT_LEN = 256
    RESPONSE_LEN = 512
    MAX_SPEC_TOKENS = 32  # More tokens to allow branching
    MIN_TOKEN_PROB = 0.01  # Lower threshold to allow more branches
    MAX_SPEC_FACTOR = 2.0  # Higher factor
    
    print(f"\nConfig:")
    print(f"  - Num cached requests: {NUM_REQUESTS}")
    print(f"  - Prompt length: {PROMPT_LEN}")
    print(f"  - Response length: {RESPONSE_LEN}")
    print(f"  - Max spec tokens: {MAX_SPEC_TOKENS}")
    print(f"  - Min token prob: {MIN_TOKEN_PROB}")
    print(f"  - Max spec factor: {MAX_SPEC_FACTOR}")
    
    # Generate sample data
    print("\nGenerating sample data...")
    data = generate_sample_data(NUM_REQUESTS, PROMPT_LEN, RESPONSE_LEN)
    
    # Build cache
    print("Building suffix cache...")
    cache = build_cache(data)
    print(f"  Cache built with {len(cache.cached_requests)} cached requests")
    
    # Generate test context (from one of the prompts + partial response)
    test_prompt = data[0][1]
    test_response = data[0][2][:100]  # First 100 tokens of response
    test_context = np.concatenate([test_prompt, test_response]).astype(np.int32)
    
    print(f"\nTest context length: {len(test_context)}")
    
    # Run with use_tree_spec=False
    print("\n" + "-" * 70)
    print("use_tree_spec=False (single-path speculation)")
    print("-" * 70)
    
    draft_path, time_path = benchmark_speculate(
        cache, test_context, MAX_SPEC_TOKENS, 
        use_tree_spec=False, min_token_prob=MIN_TOKEN_PROB
    )
    
    stats_path = analyze_tree(draft_path.token_ids, draft_path.parents, draft_path.probs)
    
    print(f"Time: {time_path*1000:.3f} ms")
    print(f"Num tokens: {stats_path['num_tokens']}")
    print(f"Max depth: {stats_path['max_depth']}")
    print(f"Num roots: {stats_path['num_roots']}")
    print(f"Score: {draft_path.score:.4f}")
    print(f"Match length: {draft_path.match_len}")
    print(f"First 10 tokens: {draft_path.token_ids[:10]}")
    print(f"First 10 parents: {draft_path.parents[:10]}")
    
    # Run with use_tree_spec=True
    print("\n" + "-" * 70)
    print("use_tree_spec=True (tree-based speculation)")
    print("-" * 70)
    
    draft_tree, time_tree = benchmark_speculate(
        cache, test_context, MAX_SPEC_TOKENS,
        use_tree_spec=True, min_token_prob=MIN_TOKEN_PROB
    )
    
    stats_tree = analyze_tree(draft_tree.token_ids, draft_tree.parents, draft_tree.probs)
    
    print(f"Time: {time_tree*1000:.3f} ms")
    print(f"Num tokens: {stats_tree['num_tokens']}")
    print(f"Max depth: {stats_tree['max_depth']}")
    print(f"Num roots: {stats_tree['num_roots']}")
    print(f"Avg branching: {stats_tree['avg_branching']:.2f}")
    print(f"Max branching: {stats_tree['max_branching']}")
    print(f"Score: {draft_tree.score:.4f}")
    print(f"Match length: {draft_tree.match_len}")
    print(f"First 10 tokens: {draft_tree.token_ids[:10]}")
    print(f"First 10 parents: {draft_tree.parents[:10]}")
    if draft_tree.probs:
        print(f"First 10 probs: {[f'{p:.3f}' for p in draft_tree.probs[:10]]}")
    
    # Comparison
    print("\n" + "=" * 70)
    print("Comparison Summary")
    print("=" * 70)
    print(f"{'Metric':<20} {'Single-path':<15} {'Tree-based':<15} {'Improvement'}")
    print("-" * 70)
    
    token_ratio = stats_tree['num_tokens'] / stats_path['num_tokens'] if stats_path['num_tokens'] > 0 else 0
    print(f"{'Num tokens':<20} {stats_path['num_tokens']:<15} {stats_tree['num_tokens']:<15} {token_ratio:.2f}x")
    
    score_ratio = draft_tree.score / draft_path.score if draft_path.score > 0 else 0
    print(f"{'Score':<20} {draft_path.score:<15.4f} {draft_tree.score:<15.4f} {score_ratio:.2f}x")
    
    time_ratio = time_tree / time_path if time_path > 0 else 0
    print(f"{'Time (ms)':<20} {time_path*1000:<15.3f} {time_tree*1000:<15.3f} {time_ratio:.2f}x")
    
    print(f"{'Max depth':<20} {stats_path['max_depth']:<15} {stats_tree['max_depth']:<15}")
    print(f"{'Num roots':<20} {stats_path['num_roots']:<15} {stats_tree['num_roots']:<15}")
    
    # Key insight
    print("\n" + "=" * 70)
    print("Key Insight")
    print("=" * 70)
    if stats_tree['num_roots'] > 1:
        print(f"Tree mode produced {stats_tree['num_roots']} root branches!")
        print("This means multiple independent speculation paths are available.")
        print("Each path can be verified in parallel, improving acceptance rate.")
    else:
        print("Tree mode produced a single root (similar to single-path).")
        print("This may indicate limited branching in the suffix tree.")
    
    if draft_tree.score > draft_path.score:
        print(f"\nTree mode has HIGHER score ({draft_tree.score:.4f} vs {draft_path.score:.4f})")
        print("This indicates better overall speculation quality.")
    
    return draft_path, draft_tree


if __name__ == "__main__":
    run_experiment()
