#!/usr/bin/env python3
"""
Real experiment to compare single-branch vs multi-branch suffix decoding.

This experiment:
1. Parses the dataset from CSV
2. Uses SuffixDecodingCache with real tokenized data
3. Compares single-branch (use_tree_spec=False) vs multi-branch (use_tree_spec=True)
4. Tests different branch factors (1, 2, 3, 4)
5. Measures: throughput (tokens/sec), acceptance rate, average accepted length

Key metrics:
- Acceptance Rate: accepted_tokens / total_speculated_tokens
- Average Accepted Length: average number of tokens accepted per speculation
- Throughput: total_output_tokens / total_time
"""

import argparse
import json
import os
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from arctic_inference.suffix_decoding import SuffixDecodingCache

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class ExperimentResult:
    """Results from a single speculation step."""
    step: int
    num_spec_tokens: int
    num_accepted: int
    num_output: int
    spec_time_ms: float
    match_len: int
    score: float
    tree_roots: int = 1
    tree_depth: int = 0


@dataclass
class ExperimentStats:
    """Aggregated statistics for an experiment."""
    config_name: str
    use_tree_spec: bool
    max_branch_factor: int
    total_spec_tokens: int = 0
    total_accepted_tokens: int = 0
    total_output_tokens: int = 0
    total_spec_time_ms: float = 0.0
    total_requests: int = 0
    total_steps: int = 0
    avg_acceptance_rate: float = 0.0
    avg_accepted_length: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    avg_match_len: float = 0.0
    avg_score: float = 0.0


def parse_dataset(csv_path: str, max_samples: int = None) -> pd.DataFrame:
    """Parse the CSV dataset to extract prompts and responses."""
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if max_samples:
        df = df.head(max_samples)
    
    prompts = []
    responses = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Parsing dataset"):
        try:
            req_resp = json.loads(row['request_response'])
            req_body = json.loads(req_resp['request_body'])
            resp_body = json.loads(req_resp['response_body'])
            
            # Get the last user message as prompt
            messages = req_body.get('messages', [])
            prompt = ""
            for msg in messages:
                if msg.get('role') == 'user':
                    prompt = msg.get('content', '')
            
            # Get the response content
            choices = resp_body.get('choices', [])
            if choices:
                response = choices[0].get('message', {}).get('content', '')
            else:
                response = ""
            
            prompts.append(prompt)
            responses.append(response)
        except Exception as e:
            print(f"Error parsing row {idx}: {e}")
            prompts.append("")
            responses.append("")
    
    return pd.DataFrame({
        'prompt': prompts,
        'response': responses
    })


def tokenize_with_simple_tokenizer(texts: List[str]) -> List[List[int]]:
    """Simple tokenization using whitespace and character-level fallback."""
    tokenized = []
    vocab = defaultdict(lambda: len(vocab))
    
    # Build vocabulary from all texts
    for text in texts:
        for char in text:
            _ = vocab[char]
    
    # Add special tokens
    vocab['<pad>'] = 0
    vocab['<eos>'] = 1
    
    # Tokenize
    for text in texts:
        tokens = [vocab[char] for char in text]
        tokenized.append(np.array(tokens, dtype=np.int32))
    
    return tokenized, dict(vocab)


def run_single_experiment(
    cache: SuffixDecodingCache,
    prompt_tokens: np.ndarray,
    response_tokens: np.ndarray,
    use_tree_spec: bool,
    max_branch_factor: int = 3,
    max_spec_tokens: int = 32,
    max_spec_factor: float = 2.0,
    min_token_prob: float = 0.01,
    apply_branch_limit: bool = False,
) -> List[ExperimentResult]:
    """
    Run a single request experiment with the given cache configuration.
    
    This simulates speculative decoding by:
    1. Speculating tokens based on cache
    2. Verifying against ground truth response
    3. Recording metrics
    """
    results = []
    
    # Start request
    cache.start_request("test_req", prompt_tokens)
    
    # Initialize context with prompt
    current_tokens = list(prompt_tokens)
    response_pos = 0
    
    while response_pos < len(response_tokens):
        # Get context (last N tokens)
        context = np.array(current_tokens[-cache.max_tree_depth:], dtype=np.int32)
        
        # Speculate
        start_time = time.perf_counter()
        draft = cache.speculate(
            "test_req",
            context,
            max_spec_tokens=max_spec_tokens,
            max_spec_factor=max_spec_factor,
            min_token_prob=min_token_prob,
            use_tree_spec=use_tree_spec,
        )
        spec_time = time.perf_counter() - start_time
        
        # Analyze tree structure
        num_roots = sum(1 for p in draft.parents if p < 0)
        max_depth = compute_tree_depth(draft.parents)
        
        # Apply branch limit if needed (for testing different branch factors)
        spec_tokens = list(draft.token_ids)
        spec_parents = list(draft.parents)
        
        if apply_branch_limit and use_tree_spec and max_branch_factor > 0:
            spec_tokens, spec_parents = apply_topk_branch_limit(
                spec_tokens, spec_parents, list(draft.probs) if draft.probs else [], max_branch_factor
            )
        
        # Verify: find accepted tokens by matching with ground truth
        accepted = []
        remaining_response = response_tokens[response_pos:]
        
        # BFS through the tree to find matching path
        node = -1  # Start from root
        for i, token in enumerate(remaining_response):
            # Find child with matching token
            children = [j for j, p in enumerate(spec_parents) if p == node]
            found = False
            for child_idx in children:
                if child_idx < len(spec_tokens) and spec_tokens[child_idx] == token:
                    accepted.append(token)
                    node = child_idx
                    found = True
                    break
            
            # Also check if token matches root-level speculation
            if not found and node == -1:
                for j, parent in enumerate(spec_parents):
                    if parent < 0 and j < len(spec_tokens) and spec_tokens[j] == token:
                        accepted.append(token)
                        node = j
                        found = True
                        break
            
            if not found:
                break
        
        num_accepted = len(accepted)
        
        # Add bonus token if not at end
        output_tokens = list(accepted)
        if response_pos + num_accepted < len(response_tokens):
            bonus_token = response_tokens[response_pos + num_accepted]
            output_tokens.append(bonus_token)
            num_accepted += 1  # Include bonus in accepted for metrics
        
        # Update cache with new tokens
        if output_tokens:
            cache.add_active_response("test_req", np.array(output_tokens, dtype=np.int32))
            current_tokens.extend(output_tokens)
            response_pos += len(output_tokens)
        
        # Record result
        results.append(ExperimentResult(
            step=len(results),
            num_spec_tokens=len(spec_tokens),
            num_accepted=num_accepted,
            num_output=len(output_tokens),
            spec_time_ms=spec_time * 1000,
            match_len=draft.match_len,
            score=draft.score,
            tree_roots=num_roots,
            tree_depth=max_depth,
        ))
    
    # Stop request
    cache.stop_request("test_req")
    
    return results


def compute_tree_depth(parents: List[int]) -> int:
    """Compute maximum depth of a tree from parent array."""
    if not parents:
        return 0
    
    def get_depth(node, memo):
        if node in memo:
            return memo[node]
        if parents[node] < 0:
            memo[node] = 1
            return 1
        depth = get_depth(parents[node], memo) + 1
        memo[node] = depth
        return depth
    
    memo = {}
    return max(get_depth(i, memo) for i in range(len(parents)))


def apply_topk_branch_limit(
    token_ids: List[int],
    parents: List[int],
    probs: List[float],
    max_branch_factor: int,
) -> Tuple[List[int], List[int]]:
    """Apply top-k branch limiting to a speculation tree."""
    if not token_ids or max_branch_factor <= 0:
        return token_ids, parents
    
    n = len(token_ids)
    
    # Build children mapping
    children = defaultdict(list)
    for i, parent in enumerate(parents):
        if parent >= 0 and parent < n:
            children[parent].append(i)
    
    # Sort children by probability and keep top-k
    nodes_to_keep = set()
    
    # Find roots
    roots = [i for i, p in enumerate(parents) if p < 0]
    nodes_to_keep.update(roots)
    
    def add_topk_children(node):
        if node not in children:
            return
        child_list = children[node]
        # Sort by probability (descending)
        if probs:
            child_list.sort(key=lambda x: probs[x] if x < len(probs) else 0, reverse=True)
        # Keep top-k
        for child in child_list[:max_branch_factor]:
            nodes_to_keep.add(child)
            add_topk_children(child)
    
    for root in roots:
        add_topk_children(root)
    
    # Rebuild arrays
    old_to_new = {}
    new_tokens = []
    new_parents = []
    
    for new_idx, old_idx in enumerate(sorted(nodes_to_keep)):
        old_to_new[old_idx] = new_idx
        new_tokens.append(token_ids[old_idx])
        parent = parents[old_idx]
        if parent < 0 or parent not in nodes_to_keep:
            new_parents.append(-1)
        else:
            new_parents.append(old_to_new[parent])
    
    return new_tokens, new_parents


def run_experiments(
    dataset: pd.DataFrame,
    configs: List[Dict],
    max_spec_tokens: int = 32,
    max_spec_factor: float = 2.0,
    min_token_prob: float = 0.01,
    num_warmup: int = 10,
    num_eval: int = 100,
    seed: int = 42,
) -> List[ExperimentStats]:
    """Run experiments for all configurations."""
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    all_texts = []
    for _, row in dataset.iterrows():
        all_texts.append(row['prompt'])
        all_texts.append(row['response'])
    
    prompts, responses = [], []
    prompt_tokens_list, vocab = tokenize_with_simple_tokenizer(
        [row['prompt'] for _, row in dataset.iterrows()]
    )
    response_tokens_list, _ = tokenize_with_simple_tokenizer(
        [row['response'] for _, row in dataset.iterrows()]
    )
    
    all_stats = []
    
    for config in configs:
        config_name = config['name']
        use_tree_spec = config['use_tree_spec']
        max_branch_factor = config.get('max_branch_factor', 3)
        apply_branch_limit = config.get('apply_branch_limit', False)
        
        print(f"\n{'='*70}")
        print(f"Running experiment: {config_name}")
        print(f"  use_tree_spec={use_tree_spec}, max_branch_factor={max_branch_factor}")
        print(f"{'='*70}")
        
        # Build cache with warmup data
        cache = SuffixDecodingCache(max_tree_depth=64, max_cached_requests=10000)
        
        # Warmup: add some requests to the cache
        print(f"Warming up cache with {num_warmup} requests...")
        np.random.seed(seed)
        warmup_indices = np.random.choice(len(dataset), min(num_warmup, len(dataset)), replace=False)
        
        for i, idx in enumerate(warmup_indices):
            prompt_tok = prompt_tokens_list[idx]
            resp_tok = response_tokens_list[idx]
            cache.start_request(f"warmup_{i}", prompt_tok)
            cache.add_active_response(f"warmup_{i}", resp_tok)
            cache.stop_request(f"warmup_{i}")
        
        print(f"Cache built with {len(cache.cached_requests)} cached requests")
        
        # Run evaluation
        print(f"Running evaluation on {num_eval} requests...")
        np.random.seed(seed + 1)
        eval_indices = np.random.choice(len(dataset), min(num_eval, len(dataset)), replace=False)
        
        all_results = []
        
        for idx in tqdm(eval_indices, desc=f"Evaluating {config_name}"):
            prompt_tok = prompt_tokens_list[idx]
            resp_tok = response_tokens_list[idx]
            
            results = run_single_experiment(
                cache,
                prompt_tok,
                resp_tok,
                use_tree_spec=use_tree_spec,
                max_branch_factor=max_branch_factor,
                max_spec_tokens=max_spec_tokens,
                max_spec_factor=max_spec_factor,
                min_token_prob=min_token_prob,
                apply_branch_limit=apply_branch_limit,
            )
            all_results.extend(results)
        
        # Compute statistics
        stats = ExperimentStats(
            config_name=config_name,
            use_tree_spec=use_tree_spec,
            max_branch_factor=max_branch_factor,
        )
        
        for r in all_results:
            stats.total_spec_tokens += r.num_spec_tokens
            stats.total_accepted_tokens += r.num_accepted
            stats.total_output_tokens += r.num_output
            stats.total_spec_time_ms += r.spec_time_ms
            stats.total_steps += 1
            stats.avg_match_len += r.match_len
            stats.avg_score += r.score
        
        stats.total_requests = len(eval_indices)
        
        if stats.total_spec_tokens > 0:
            stats.avg_acceptance_rate = stats.total_accepted_tokens / stats.total_spec_tokens
        
        if stats.total_steps > 0:
            stats.avg_accepted_length = stats.total_accepted_tokens / stats.total_steps
            stats.avg_match_len /= stats.total_steps
            stats.avg_score /= stats.total_steps
        
        if stats.total_spec_time_ms > 0:
            stats.throughput_tokens_per_sec = (stats.total_output_tokens * 1000) / stats.total_spec_time_ms
        
        all_stats.append(stats)
        
        # Print intermediate results
        print(f"\nResults for {config_name}:")
        print(f"  Total speculated tokens: {stats.total_spec_tokens}")
        print(f"  Total accepted tokens: {stats.total_accepted_tokens}")
        print(f"  Acceptance rate: {stats.avg_acceptance_rate:.4f}")
        print(f"  Avg accepted length: {stats.avg_accepted_length:.2f}")
        print(f"  Throughput: {stats.throughput_tokens_per_sec:.2f} tokens/sec")
        print(f"  Avg match length: {stats.avg_match_len:.2f}")
        print(f"  Avg score: {stats.avg_score:.4f}")
    
    return all_stats


def print_comparison_table(stats_list: List[ExperimentStats]):
    """Print a comparison table of all experiment results."""
    print("\n" + "="*100)
    print("EXPERIMENT COMPARISON TABLE")
    print("="*100)
    
    # Header
    header = f"{'Config':<30} {'Accept Rate':<12} {'Avg Accept':<12} {'Throughput':<15} {'Avg Match':<12} {'Avg Score':<12}"
    print(header)
    print("-"*100)
    
    for stats in stats_list:
        row = f"{stats.config_name:<30} {stats.avg_acceptance_rate:<12.4f} {stats.avg_accepted_length:<12.2f} {stats.throughput_tokens_per_sec:<15.2f} {stats.avg_match_len:<12.2f} {stats.avg_score:<12.4f}"
        print(row)
    
    print("="*100)
    
    # Compute improvements relative to baseline (single-path)
    baseline = stats_list[0]
    print("\nImprovement relative to baseline (single-path):")
    print(f"{'Config':<30} {'Accept Rate':<15} {'Throughput':<15}")
    print("-"*60)
    
    for stats in stats_list[1:]:
        accept_improvement = (stats.avg_acceptance_rate / baseline.avg_acceptance_rate - 1) * 100 if baseline.avg_acceptance_rate > 0 else 0
        throughput_improvement = (stats.throughput_tokens_per_sec / baseline.throughput_tokens_per_sec - 1) * 100 if baseline.throughput_tokens_per_sec > 0 else 0
        
        print(f"{stats.config_name:<30} {accept_improvement:>+.2f}%{'':<8} {throughput_improvement:>+.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Experiment: single-branch vs multi-branch suffix decoding")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the CSV dataset")
    parser.add_argument("--num-warmup", type=int, default=50, help="Number of warmup requests")
    parser.add_argument("--num-eval", type=int, default=100, help="Number of evaluation requests")
    parser.add_argument("--max-spec-tokens", type=int, default=32, help="Max speculation tokens")
    parser.add_argument("--max-spec-factor", type=float, default=2.0, help="Max speculation factor")
    parser.add_argument("--min-token-prob", type=float, default=0.01, help="Minimum token probability")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="experiment_results.csv", help="Output CSV file")
    args = parser.parse_args()
    
    # Parse dataset
    dataset = parse_dataset(args.dataset)
    print(f"Dataset size: {len(dataset)} requests")
    
    # Filter out empty entries
    dataset = dataset[(dataset['prompt'].str.len() > 0) & (dataset['response'].str.len() > 0)]
    print(f"After filtering: {len(dataset)} valid requests")
    
    # Define experiment configurations
    configs = [
        # Baseline: single-path (no tree speculation)
        {
            "name": "single-path (baseline)",
            "use_tree_spec": False,
            "max_branch_factor": 1,
            "apply_branch_limit": False,
        },
        # Multi-branch: tree speculation with different branch factors
        {
            "name": "tree-spec (branch=1)",
            "use_tree_spec": True,
            "max_branch_factor": 1,
            "apply_branch_limit": True,
        },
        {
            "name": "tree-spec (branch=2)",
            "use_tree_spec": True,
            "max_branch_factor": 2,
            "apply_branch_limit": True,
        },
        {
            "name": "tree-spec (branch=3)",
            "use_tree_spec": True,
            "max_branch_factor": 3,
            "apply_branch_limit": True,
        },
        {
            "name": "tree-spec (branch=4)",
            "use_tree_spec": True,
            "max_branch_factor": 4,
            "apply_branch_limit": True,
        },
        # Unlimited branches
        {
            "name": "tree-spec (unlimited)",
            "use_tree_spec": True,
            "max_branch_factor": 100,
            "apply_branch_limit": False,
        },
    ]
    
    # Run experiments
    stats_list = run_experiments(
        dataset,
        configs,
        max_spec_tokens=args.max_spec_tokens,
        max_spec_factor=args.max_spec_factor,
        min_token_prob=args.min_token_prob,
        num_warmup=args.num_warmup,
        num_eval=args.num_eval,
        seed=args.seed,
    )
    
    # Print comparison
    print_comparison_table(stats_list)
    
    # Save results
    results_df = pd.DataFrame([
        {
            "config": s.config_name,
            "use_tree_spec": s.use_tree_spec,
            "max_branch_factor": s.max_branch_factor,
            "total_spec_tokens": s.total_spec_tokens,
            "total_accepted_tokens": s.total_accepted_tokens,
            "total_output_tokens": s.total_output_tokens,
            "acceptance_rate": s.avg_acceptance_rate,
            "avg_accepted_length": s.avg_accepted_length,
            "throughput_tokens_per_sec": s.throughput_tokens_per_sec,
            "avg_match_len": s.avg_match_len,
            "avg_score": s.avg_score,
        }
        for s in stats_list
    ])
    results_df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
