# SPDX-License-Identifier: Apache-2.0
"""Explicit forward/backward API for the validated Hopper specialization."""

from dataclasses import replace
from functools import lru_cache
import threading
import torch
from ._loader import private_module
from .csrc.fa4.mask import CompactAttentionMask, compact_mask

_lock = threading.RLock()


@lru_cache(maxsize=1)
def _forward_launcher():
    kernel = private_module("flash_attn.cute.flash_fwd_sm90", "forward_kernel")
    kernel.AttentionMask = CompactAttentionMask
    interface = private_module("flash_attn.cute.interface", "forward_interface")
    interface.FlashAttentionForwardSm90 = kernel.FlashAttentionForwardSm90
    return interface._flash_attn_fwd


@lru_cache(maxsize=24)
def _interface(variant, ctas, planned=False, split=False):
    if variant not in ("index", "paired"):
        raise ValueError(f"unknown variant: {variant}")
    kernel_name, interface_name = "backward", "interface"
    suffix = f"{variant}_{ctas}_{planned}_{split}"
    kernel = private_module(
        f"lazy_attention.csrc.fa4.{kernel_name}", suffix + "_kernel"
    )
    kernel.AttentionMask = CompactAttentionMask
    interface = private_module(
        f"lazy_attention.csrc.fa4.{interface_name}", suffix + "_interface"
    )

    def factory(dtype, hd, hdv, ratio, causal, **kw):
        assert hd == hdv == 128 and ratio == 4 and not causal
        obj = kernel.FlashAttentionBackwardSm90(
            dtype, kw["mask_mod"], range_mapping=variant == "paired", physical_ctas=ctas
        )
        obj.planned_order = planned
        obj.split_tasks = split
        return obj

    interface.FlashAttentionBackwardSm90 = factory
    interface.range_mapping = variant == "paired"
    interface.split_tasks = split
    upstream = interface._tile_size_bwd_sm90
    interface._tile_size_bwd_sm90 = lambda *a, **kw: replace(
        upstream(*a, **kw), dQ_single_wg=True
    )
    return interface


def _validate(q, k, v, prepared):
    if not all(
        x.is_cuda
        and x.device == q.device
        and x.dtype == torch.bfloat16
        and x.ndim == 4
        and x.stride(-1) == 1
        for x in (q, k, v)
    ):
        raise ValueError(
            "q/k/v must be BF16 CUDA rank-4 tensors on one device, last stride 1"
        )
    s = prepared.seqlen
    if (
        q.shape != (1, s, k.shape[2] * 4, 128)
        or k.shape != (1, s, k.shape[2], 128)
        or v.shape != k.shape
    ):
        raise ValueError("requires batch 1, self-attention, D128 and GQA4")
    if torch.cuda.get_device_capability(q.device)[0] != 9:
        raise ValueError("requires Hopper (SM90)")
    if torch.are_deterministic_algorithms_enabled():
        raise ValueError("deterministic backward is unsupported")
    if any(x.device != q.device for x in prepared.logical_aux + prepared.physical_aux):
        raise ValueError("prepare metadata on the same device as q/k/v")


def forward(q, k, v, prepared, *, scale=None):
    """Return (out, natural-log LSE) in physical row order; no autograd wrapper."""
    _validate(q, k, v, prepared)
    with _lock:
        launch = _forward_launcher()
    result = launch(
        q,
        k,
        v,
        softmax_scale=scale,
        pack_gqa=False,
        return_lse=True,
        mask_mod=compact_mask,
        aux_tensors=list(prepared.physical_aux),
        block_sparse_tensors=prepared.forward_plan,
    )
    return result[:2]


def backward(q, k, v, out, dout, lse, prepared, *, scale=None, variant="paired", task_order=None, task_plan=None):
    """Return dQ/dK/dV; preparation and forward are outside this chain.

    index uses the frozen full-index baseline. paired is the current exclusive
    range/index implementation.
    """
    _validate(q, k, v, prepared)
    if (
        out.shape != q.shape
        or dout.shape != q.shape
        or any(
            x.device != q.device or x.dtype != q.dtype or x.stride(-1) != 1
            for x in (out, dout)
        )
    ):
        raise ValueError("out/dout must match q shape, device and dtype, last stride 1")
    if (
        lse.shape != (1, q.shape[2], q.shape[1])
        or lse.dtype != torch.float32
        or lse.device != q.device
    ):
        raise ValueError("lse must be FP32 [1, Hq, S] on the q device")
    if variant == "index":
        mapping = prepared.index
        if mapping is None:
            raise ValueError(
                'index baseline requires prepare_attention(strategy="index")'
            )
    else:
        mapping = prepared.descriptor
        if mapping is None:
            raise ValueError(
                'range variants require prepare_attention(strategy="range")'
            )
    if mapping.device != q.device:
        raise ValueError("row mapping is on the wrong device")
    split = task_plan is not None and task_plan.split
    if task_plan is not None:
        if task_order is not None: raise ValueError('pass task_plan or task_order')
        if split and variant != 'paired': raise ValueError('split plan requires paired mapping')
        task_order=task_plan.order
    with _lock:
        ctas = torch.cuda.get_device_properties(q.device).multi_processor_count
        interface = _interface(variant, ctas, task_order is not None, split)
    counter = torch.zeros((1,), device=q.device, dtype=torch.int32)
    if task_order is not None:
        if (task_order.device != q.device or task_order.dtype != torch.int32
                or (task_order.ndim != 2 or task_order.shape[1] != 4 if split else task_order.shape != (prepared.seqlen // 128 * k.shape[2],))
                or not task_order.is_contiguous()):
            raise ValueError('task_order must be contiguous CUDA int32 with one ID per KV/head task')
        task_order.__leading_dim__, task_order.__assumed_align__ = (1 if split else 0), 16
    aux = [*prepared.logical_aux, task_order, counter]
    if variant == "index":
        aux.append(mapping)
    else:
        mapping = mapping.view(-1)
        mapping.__leading_dim__, mapping.__assumed_align__ = 0, 16
        # ABI slot 4 carries only static S, never row data. A scalar allocation
        # backs this stride-0 view; range paths have no full-index allocation.
        aux.extend([counter.expand(q.shape[1]), mapping])
    partials=None
    if split:
        partials=tuple(torch.empty((task_plan.partial_count,128,128),device=q.device,dtype=torch.float32) for _ in range(2))
        for t in partials: t.__leading_dim__,t.__assumed_align__=2,16
        aux.extend(partials)
    result=interface._flash_attn_bwd(
        q,
        k,
        v,
        out,
        dout,
        lse,
        softmax_scale=scale,
        pack_gqa=False,
        V_in_regs=True,
        mask_mod=compact_mask,
        aux_tensors=aux,
        block_sparse_tensors=prepared.backward_plan,
    )[:3]

    if split:
        from .split_reduction import reduce_partials
        reduce_partials(*partials,task_plan.reduction_indices,task_plan.reduction_tasks,mapping,result[1],result[2])
    return result
