# SPDX-License-Identifier: BSD-3-Clause
# Derived from FlashAttention CuTe; see THIRD_PARTY_NOTICES.md.
# Modified: standalone Hopper D128/GQA4 sparse backward launcher.
"""Shared launch chain. The mapping policy is fixed before compilation."""

import inspect
from flash_attn.cute.interface import (
    AuxData,
    _bwd_preprocess,
    _tile_size_bwd_sm90,
    cute,
    get_aux_tensor_metadata,
    get_broadcast_dims,
    maybe_contiguous,
    normalize_block_sparse_config_bwd,
    to_cute_aux_tensor,
    to_cute_block_sparse_tensors,
    to_cute_tensor,
    torch,
    torch2cute_dtype_map,
    utils,
)
from lazy_attention.csrc.fa4.post_launcher import postprocess

range_mapping = False
# Set on the private module by operator._interface; no public FA4 mutation.
FlashAttentionBackwardSm90 = None

if "hdim_multiple_of" not in inspect.signature(_bwd_preprocess).parameters:
    _upstream_preprocess = _bwd_preprocess

    def _bwd_preprocess(*args, hdim_multiple_of=32, **kwargs):
        return _upstream_preprocess(*args, **kwargs)


def _flash_attn_bwd(
    q,
    k,
    v,
    out,
    dout,
    lse,
    softmax_scale=None,
    *,
    pack_gqa=False,
    V_in_regs=True,
    mask_mod=None,
    aux_tensors=None,
    block_sparse_tensors=None,
):
    if pack_gqa or not V_in_regs or block_sparse_tensors is None:
        raise ValueError("requires unpacked GQA4, V in registers and a sparse plan")
    if block_sparse_tensors.block_size != (64, 128):
        raise ValueError("requires a 64x128 backward plan")
    if len(aux_tensors) != (6 if range_mapping else 5):
        raise ValueError("wrong row-map ABI")
    q, k, v, out, dout, lse = [maybe_contiguous(t) for t in (q, k, v, out, dout, lse)]
    batch_size, seqlen, num_head, head_dim = q.shape
    num_head_kv = k.shape[2]
    if (
        batch_size != 1
        or seqlen <= 0
        or seqlen % 128
        or head_dim != 128
        or num_head != 4 * num_head_kv
        or q.dtype != torch.bfloat16
    ):
        raise ValueError("requires BF16 batch1 self-attention, S%128=0, D128 and GQA4")
    cfg = _tile_size_bwd_sm90(128, 128, False, False, sparse_block_size_q=64)
    if tuple(vars(cfg).values()) != (
        64,
        128,
        2,
        2,
        2,
        True,
        False,
        False,
        1,
        2,
        1,
        2,
        True,
    ):
        raise ValueError("incompatible FA4 Hopper tile configuration")
    if softmax_scale is None:
        softmax_scale = 128**-0.5
    dtype = torch2cute_dtype_map[q.dtype]
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    dq_accum = torch.empty(
        1, num_head, seqlen * 128, dtype=torch.float32, device=q.device
    )
    dpsum = torch.empty(1, num_head, seqlen, dtype=torch.float32, device=q.device)
    lse_log2 = torch.empty(1, num_head, seqlen, dtype=torch.float32, device=q.device)
    _bwd_preprocess(
        out,
        dout,
        dpsum,
        lse,
        lse_log2,
        dq_accum,
        None,
        None,
        None,
        dtype,
        128,
        128,
        64,
        cu_total_m_blocks=None,
        fake_mode=False,
        hdim_multiple_of=32,
    )
    sparse, pattern = normalize_block_sparse_config_bwd(
        block_sparse_tensors,
        batch_size=1,
        num_head=num_head,
        seqlen_q=seqlen,
        seqlen_k=seqlen,
        block_size=(64, 128),
        q_subtile_factor=1,
        kv_subtile_factor=1,
    )
    # Keep all shape/stride metadata used by the frozen launcher. S is dynamic
    # in CuTe views; the single-K-block distinction prevents stride poisoning.
    key = (
        dtype,
        num_head,
        num_head_kv,
        tuple(vars(cfg).values()),
        utils.hash_callable(mask_mod),
        tuple(None if t is None else get_aux_tensor_metadata([t]) for t in aux_tensors),
        pattern,
        *(get_broadcast_dims(t) for t in (q, k, v, dout)),
        seqlen // 128 == 1,
    )
    if key not in _flash_attn_bwd.compile_cache:
        tensors = [
            to_cute_tensor(t)
            for t in (q, k, v, dout, lse_log2, dpsum, dq_accum, dk, dv)
        ]
        obj = FlashAttentionBackwardSm90(
            dtype,
            128,
            128,
            4,
            False,
            tile_m=64,
            tile_n=128,
            Q_stage=cfg.num_stages_Q,
            dO_stage=cfg.num_stages_dO,
            PdS_stage=cfg.num_stages_PdS,
            SdP_swapAB=cfg.SdP_swapAB,
            dKV_swapAB=False,
            dQ_swapAB=False,
            AtomLayoutMSdP=cfg.AtomLayoutMSdP,
            AtomLayoutNdKV=cfg.AtomLayoutNdKV,
            AtomLayoutMdQ=cfg.AtomLayoutMdQ,
            num_threads=384,
            V_in_regs=True,
            mask_mod=mask_mod,
            has_aux_tensors=True,
            q_subtile_factor=1,
            dQ_single_wg=True,
        )
        _flash_attn_bwd.compile_cache[key] = cute.compile(
            obj,
            *tensors,
            softmax_scale,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            AuxData([to_cute_aux_tensor(t) if t is not None else None for t in aux_tensors], None),
            to_cute_block_sparse_tensors(sparse),
            None,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    sparse_args = tuple(
        getattr(sparse, name)
        for name in (
            "mask_block_cnt",
            "mask_block_idx",
            "full_block_cnt",
            "full_block_idx",
            "cu_total_m_blocks",
            "cu_block_idx_offsets",
            "dq_write_order",
            "dq_write_order_full",
        )
    )
    _flash_attn_bwd.compile_cache[key](
        q.detach(),
        k.detach(),
        v.detach(),
        dout,
        lse_log2,
        dpsum,
        dq_accum,
        dk,
        dv,
        softmax_scale,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        AuxData(aux_tensors, None),
        sparse_args,
        None,
    )
    postprocess(
        dq_accum, dq, aux_tensors[-1], softmax_scale, range_mapping=range_mapping
    )
    return dq, dk, dv


_flash_attn_bwd.compile_cache = {}
