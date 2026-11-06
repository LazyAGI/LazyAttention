# SPDX-License-Identifier: Apache-2.0
"""Keep the four Q heads sharing a KV head adjacent, visiting Q tiles backwards.

This changes CTA assignment only. Sparse plans and physical Q/K/V row order
are unchanged. The public operator validates the GQA ratio of four.
"""
import cutlass.cute as cute
from flash_attn.cute.tile_scheduler import SingleTileScheduler


class GroupedReverseTileScheduler(SingleTileScheduler):
    @staticmethod
    def create(params, ctx=None, *, loc=None, ip=None):
        x, y, z = cute.arch.block_idx()
        linear = x + y * params.num_block
        kv_head = linear // (params.num_block * 4)
        within_head = linear % (params.num_block * 4)
        q_tile = params.num_block - 1 - within_head // 4
        q_head = kv_head * 4 + within_head % 4
        return SingleTileScheduler(params, (q_tile, q_head, z), loc=loc, ip=ip)
