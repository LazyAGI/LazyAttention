# SPDX-License-Identifier: Apache-2.0
"""Exact sparse Q-list permutation without device-to-host staging."""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["WIDTH", "GROUPS"])
def _permute(Input, Counts, Output, WIDTH, GROUPS,
             REVERSE: tl.constexpr, INTERLEAVE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    pos = tl.arange(0, BLOCK)
    count = tl.load(Counts + row)
    src = pos
    if INTERLEAVE:
        base = (pos // INTERLEAVE) * INTERLEAVE
        length = tl.minimum(tl.maximum(count - base, 1), INTERLEAVE)
        phase = (row % GROUPS) % length
        src = base + (pos - base + phase) % length
    if REVERSE:
        src = count - 1 - src
    src = tl.where(pos < count, src, pos)
    value = tl.load(Input + row * WIDTH + src, pos < WIDTH, 0)
    tl.store(Output + row * WIDTH + pos, value, pos < WIDTH)


def permute_q_lists_gpu(indices, counts, *, reverse=False, interleave=0):
    """Permute valid lists; preserve every padding slot, shape and dtype."""
    if interleave not in (0, 8):
        raise ValueError('interleave must be0 or8 Q tiles')
    if (indices.shape[:-1] != counts.shape or indices.device != counts.device
            or not indices.is_cuda or indices.dtype not in (torch.int32, torch.int64)):
        raise ValueError('invalid CUDA sparse Q lists/counts')
    if not reverse and not interleave or indices.shape[-1] == 0:
        return indices
    source = indices.contiguous()
    output = torch.empty_like(source)
    width = source.shape[-1]
    _permute[(counts.numel(),)](source, counts.contiguous(), output, width,
        counts.shape[-1], reverse, interleave, triton.next_power_of_2(width))
    return output
