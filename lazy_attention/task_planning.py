# SPDX-License-Identifier: Apache-2.0
"""One-time, activation-independent coarse scheduling of GQA4 KV tasks."""
import torch
import triton
import triton.language as tl


@triton.jit
def _metadata(PC, FC, DOC, GROUP, COST, N: tl.constexpr, S: tl.constexpr,
              QUANTUM: tl.constexpr):
    b = tl.program_id(0)
    x = b * 128 + tl.arange(0, 128)
    d = tl.load(DOC + x, x < S, other=-1)
    lo = tl.min(tl.where(d >= 0, d, 2147483647), 0)
    hi = tl.max(d, 0)
    # Mixed-document KV tiles retain their exact Q lists, in distinct groups.
    group = tl.where(hi < 0, -1, tl.where(lo == hi, lo, -b - 2))
    count = tl.load(PC + b) + tl.load(FC + b)
    tl.store(GROUP + b, group)
    tl.store(COST + b, tl.cdiv(count, QUANTUM))


@triton.jit
def _priorities(GROUP, COST, PRIORITY, FIRST, N: tl.constexpr, BN: tl.constexpr):
    b = tl.program_id(0)
    x = tl.arange(0, BN)
    g = tl.load(GROUP + b)
    gs = tl.load(GROUP + x, x < N, other=-2147483647)
    match = (x < N) & (gs == g)
    costs = tl.load(COST + x, x < N, other=0)
    tl.store(PRIORITY + b, tl.max(tl.where(match, costs, 0), 0))
    tl.store(FIRST + b, tl.min(tl.where(match, x, N), 0))


@triton.jit
def _emit(PRIORITY, FIRST, ORDER, N: tl.constexpr, H: tl.constexpr,
          BN: tl.constexpr, BH: tl.constexpr, PRESERVE_SHORT: tl.constexpr):
    b = tl.program_id(0)
    x = tl.arange(0, BN)
    c, f = tl.load(PRIORITY + b), tl.load(FIRST + b)
    cs = tl.load(PRIORITY + x, x < N, other=-1)
    fs = tl.load(FIRST + x, x < N, other=N)
    before = (x < N) & ((cs > c) | ((cs == c) & (fs < f)))
    same = (x < N) & (fs == f)
    base = tl.sum(before.to(tl.int32), 0)
    local = tl.sum((same & (x < b)).to(tl.int32), 0)
    size = tl.sum(same.to(tl.int32), 0)
    h = tl.arange(0, BH)
    dst = base * H + h * size + local
    if PRESERVE_SHORT:
        long = tl.max(tl.where(x < N, cs, 0), 0) > 1
        dst = tl.where(long, dst, b * H + h)
    # Task IDs use the current scheduler's KV-major encoding, not old h*N+b.
    tl.store(ORDER + dst, b * H + h, h < H)


def _validate_inputs(plan, documents, seqlen, kv_heads, quantum):
    pc, fc = plan.mask_block_cnt, plan.full_block_cnt
    if (plan.block_size != (64, 128) or pc.shape[:2] != (1, 1)
            or not pc.is_cuda or not pc.is_contiguous() or not fc.is_contiguous()
            or fc.device != pc.device or fc.shape != pc.shape
            or documents.ndim != 1 or documents.numel() != seqlen
            or documents.dtype not in (torch.int32, torch.int64)
            or quantum < 1 or kv_heads < 1):
        raise ValueError('invalid sparse plan, logical document IDs or planning options')
    n = pc.numel()
    if not (0 < n <= 1024 and n * 128 == seqlen):
        raise ValueError('requires S divisible by128 and at most1024 KV tiles')
    return pc, fc, n


def build_task_order(plan, documents, seqlen, kv_heads, *, quantum=64,
                     preserve_short=True):
    """Return CUDA int32 task IDs. documents is in logical row order, padding=-1.

    All allocations, CPU-to-GPU document upload and three kernels belong to
    planning cost. The caller retains/reuses the immutable returned order.
    Current small-metadata implementation explicitly supports at most1024 KV tiles.
    """
    pc, fc, n = _validate_inputs(plan, documents, seqlen, kv_heads, quantum)
    docs = documents.to(device=pc.device, dtype=torch.int32).contiguous()
    group = torch.empty((n,), device=pc.device, dtype=torch.int32)
    cost, priority, first = (torch.empty_like(group) for _ in range(3))
    order = torch.empty((n * kv_heads,), device=pc.device, dtype=torch.int32)
    _metadata[(n,)](pc, fc, docs, group, cost, n, seqlen, quantum)
    _priorities[(n,)](group, cost, priority, first, n, triton.next_power_of_2(n))
    _emit[(n,)](priority, first, order, n, kv_heads,
                triton.next_power_of_2(n), triton.next_power_of_2(kv_heads), preserve_short)
    return order


def build_adaptive_task_order(plan, documents, seqlen, kv_heads, *, quantum=64):
    """Planner-owned conservative policy; None means no order argument/read.

    Require a task spanning at least three cost buckets before reordering.
    The scalar D2H synchronization is one-time preparation, charged /42.
    This is an experimental workload rule, not a shape whitelist.
    """
    pc, fc, _ = _validate_inputs(plan, documents, seqlen, kv_heads, quantum)
    counts = pc + fc
    longest = int(counts.max().item())
    if longest <= 2 * quantum:
        return None
    return build_task_order(plan, documents, seqlen, kv_heads, quantum=quantum)
