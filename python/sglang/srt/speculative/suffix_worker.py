"""
Suffix decoding worker that reuses NGRAMWorker with a cache adapter.

This is a thin wrapper that replaces NgramCache with SuffixCacheAdapter,
allowing all the tree-based verification logic to be reused.
"""

import logging
from typing import Optional

from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.ngram_worker import NGRAMWorker
from sglang.srt.speculative.suffix_cache_adapter import SuffixCacheAdapter
from sglang.srt.utils.common import MultiprocessingSerializer
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

logger = logging.getLogger(__name__)


class SuffixWorker(NGRAMWorker):
    """
    Suffix decoding worker that inherits from NGRAMWorker.

    The only difference is using SuffixCacheAdapter instead of NgramCache.
    All tree-based verification logic is inherited from NGRAMWorker.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Call parent __init__ which sets up all the infrastructure
        super().__init__(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )

        self.ngram_cache = SuffixCacheAdapter(
            draft_token_num=server_args.speculative_num_draft_tokens,
            max_batch_size=self.max_batch_size,
            max_tree_depth=server_args.speculative_suffix_max_tree_depth,
            max_cached_requests=server_args.speculative_suffix_max_cached_requests,
            max_spec_factor=server_args.speculative_suffix_max_spec_factor,
            min_token_prob=server_args.speculative_suffix_min_token_prob,
            use_tree_spec=server_args.speculative_suffix_use_tree_spec,
            max_branch_factor=server_args.speculative_suffix_max_branch_factor,
        )

    def _prepare_draft_tokens(self, batch):
        """
        Override to pass FULL token sequences to the cache adapter.

        NGRAMWorker passes only last N tokens, but the suffix cache needs:
        1. Full prompt for start_request()
        2. Full sequence for suffix tree building
        3. Request identity tracking
        """

        bs = batch.batch_size()

        self.ngram_cache.synchronize()
        batch_req_ids = []
        batch_prompts = []
        batch_tokens = []
        for req in batch.reqs:
            # Pass request ID for stable tracking
            batch_req_ids.append(req.rid)
            # Pass prompt separately (for cache initialization)
            batch_prompts.append(req.origin_input_ids)
            # Pass FULL token sequence (prompt + outputs), not just last N
            full_tokens = req.origin_input_ids + req.output_ids
            batch_tokens.append(full_tokens)

        req_drafts, mask = self.ngram_cache.batch_get(
            batch_req_ids, batch_prompts, batch_tokens
        )
        total_draft_token_num = len(req_drafts)

        assert (
            total_draft_token_num == bs * self.draft_token_num
        ), f"{total_draft_token_num=}, {bs=}, {self.draft_token_num=}"

        return req_drafts, mask

    def _update_ngram_cache(self, batch):
        """
        Override to pass FULL token sequences for cache updates.
        """
        batch_req_ids = []
        batch_tokens = []
        for req in batch.reqs:
            # Pass request ID for stable tracking
            batch_req_ids.append(req.rid)
            # Pass FULL token sequence for delta computation
            full_tokens = req.origin_input_ids + req.output_ids
            batch_tokens.append(full_tokens)

        self.ngram_cache.batch_put(batch_req_ids, batch_tokens)

    def update_weights_from_tensor(self, recv_req):
        """
        Update weights for hybrid engine mode.
        
        For speculative decoding workers, we only need to update the target worker's
        weights since the draft tokens come from the ngram cache, not a separate model.
        """
        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(
            recv_req.serialized_named_tensors[self.tp_rank]
        )
        # Only update target worker since SuffixWorker doesn't have its own model
        success, message = self.target_worker.model_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        return success, message
