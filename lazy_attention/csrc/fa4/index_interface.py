# Derived from FlashAttention CuTe; see THIRD_PARTY_NOTICES.md.
"""Private FA4 backward launcher for native four-head, indexed persistent CTAs.

Derived from the verified FA4 launcher. The installed backend supplies shared
validation/preprocessing utilities; this module owns its compilation cache and
the dK/dV ownership and indexed dQ postprocessing changes.
"""
import inspect
from flash_attn.cute.interface import (_LEARNABLE_SINK_DTYPES, BIN_BATCH_SEARCH_THRESH, AuxData,
                                       BlackwellFusedMultiHeadAttentionBackward, BlockSparseTensorsTorch,
                                       FlashAttentionBackwardSm80, FlashAttentionBackwardSm90,
                                       FlashAttentionBackwardSm100, FlashAttentionBackwardSm120, Int32,
                                       LearnableSinkBwdTensors, _bwd_preprocess, _compute_tile_cumsum, _get_device_arch,
                                       _resolve_causal_local_window, _tile_size_bwd_sm90, _validate_head_dims,
                                       _validate_tensor, block_sparse_bwd_supports_2cta, cute, get_aux_tensor_metadata,
                                       get_broadcast_dims, get_kv_subtile_factor, is_fake_mode, math, maybe_contiguous,
                                       normalize_block_sparse_config_bwd, to_cute_aux_tensor,
                                       to_cute_block_sparse_tensors, to_cute_tensor, torch, torch2cute_dtype_map, utils)
from typing import Callable, Optional, Tuple

from lazy_attention.csrc.fa4.index_post_launcher import postprocess

# Training may already have imported the bundled FA4 interface before the
# optional source path is installed. Its older preprocess fixes alignment at
# 32. For this D128 specialization, rounding to either 32 or 64 is identical.
# Resolve the API difference once, without replacing global FA4 modules (which
# would also change Forward and unrelated attention users).
if 'hdim_multiple_of' not in inspect.signature(_bwd_preprocess).parameters:
    _bundled_bwd_preprocess = _bwd_preprocess

    def _bwd_preprocess(*args, hdim_multiple_of=32, **kwargs):
        assert args[10] == args[11] == 128 and hdim_multiple_of in (32, 64)
        return _bundled_bwd_preprocess(*args, **kwargs)


def _bwd_postprocess_convert(accum, output, scale, cu_seqlens, seqused, arch, dtype, hdim, block_size, num_threads,
                            atom_layout, swap_ab, use_2cta_instrs=False, cluster_size=1, cu_total_m_blocks=None,
                            sink_tensors=None, hdim_multiple_of=32, *, fake_mode, row_index=None):
    assert row_index is not None and not fake_mode
    return postprocess(accum, output, row_index, scale, arch, dtype, hdim, block_size, num_threads,
                       atom_layout, swap_ab, hdim_multiple_of)


def _flash_attn_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: float = 0.0,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    m_block_size: int = 64,
    n_block_size: int = 128,
    num_threads: int = 256,
    pack_gqa: bool = False,
    num_stages_Q: int = 2,
    num_stages_dO: int = 2,
    SdP_swapAB: bool = False,
    dKV_swapAB: bool = False,
    dQ_swapAB: bool = False,
    AtomLayoutMSdP: int = 2,
    AtomLayoutNdKV: int = 2,
    AtomLayoutMdQ: int = 2,
    V_in_regs: bool = False,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    deterministic: bool = False,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
    aux_scalars: Optional[tuple] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    dlse: Optional[torch.Tensor] = None,
    learnable_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    aux_scalars = tuple(aux_scalars) if aux_scalars else None
    fake_mode = is_fake_mode()
    arch = _get_device_arch()
    assert arch // 10 in [9, 10, 11, 12], "Unsupported compute capability. Supported: 9.x, 10.x, 11.x, 12.x"
    if block_sparse_tensors is not None:
        assert (
            cu_seqlens_q is None
            and cu_seqlens_k is None
            and seqused_q is None
            and seqused_k is None
        ), "Varlen backward with block sparsity is not yet supported"
    if learnable_sink is not None:
        assert arch // 10 in [9, 10, 11], "Learnable sink backward is supported on SM90 and SM100/SM110"
        assert lse is not None, "learnable_sink backward requires LSE"
    if q.numel() == 0 or k.numel() == 0:
        dq = torch.zeros_like(q) if dq is None else dq.zero_()
        dk = torch.zeros_like(k) if dk is None else dk.zero_()
        dv = torch.zeros_like(v) if dv is None else dv.zero_()
        if learnable_sink is None:
            return dq, dk, dv
        dsink = (
            dlse.sum(dim=(0, 2) if dlse.ndim == 3 else 1).to(learnable_sink.dtype)
            if dlse is not None
            else torch.zeros_like(learnable_sink)
        )
        return dq, dk, dv, dsink
    sparse_q = None
    kv_subtile_factor = 1
    if block_sparse_tensors is not None:
        if block_sparse_tensors.block_size is not None:
            sparse_q = block_sparse_tensors.block_size[0]
        elif arch // 10 == 9:
            sparse_q = 128

    num_head, head_dim = q.shape[-2:]
    head_dim_v = v.shape[-1]

    window_size = [window_size_left, window_size_right]
    causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
        causal, window_size_left, window_size_right
    )

    if arch // 10 == 12:
        # SM120: uses SM80 MMA with 99 KB SMEM, 128 threads (4 warps).
        m_block_size = 64
        n_block_size = 64
        if head_dim <= 64:
            num_stages_Q = 2
            num_stages_dO = 2
        else:
            num_stages_Q = 1
            num_stages_dO = 1
        SdP_swapAB = False
        dKV_swapAB = False
        dQ_swapAB = False
        AtomLayoutMSdP = 4
        AtomLayoutNdKV = 4
        AtomLayoutMdQ = 4
        V_in_regs = False
        dQ_single_wg = False
        cluster_size = 1
        use_2cta_instrs = False
        num_threads = 128
        assert not (block_sparse_tensors is not None), "Block sparsity backward not supported on SM 12.0"
        assert score_mod is None and score_mod_bwd is None, "score_mod backward not supported on SM 12.0"
        assert mask_mod is None, "mask_mod backward not supported on SM 12.0"
        assert deterministic is False, "deterministic backward not supported on SM 12.0"
    elif arch // 10 == 9:
        cfg = _tile_size_bwd_sm90(
            head_dim,
            head_dim_v,
            causal,
            local,
            sparse_block_size_q=sparse_q,
        )
        m_block_size = cfg.m_block_size
        n_block_size = cfg.n_block_size
        num_stages_Q = cfg.num_stages_Q
        num_stages_dO = cfg.num_stages_dO
        num_stages_PdS = cfg.num_stages_PdS
        SdP_swapAB = cfg.SdP_swapAB
        dKV_swapAB = cfg.dKV_swapAB
        dQ_swapAB = cfg.dQ_swapAB
        AtomLayoutMSdP = cfg.AtomLayoutMSdP
        AtomLayoutNdKV = cfg.AtomLayoutNdKV
        AtomLayoutMdQ = cfg.AtomLayoutMdQ
        num_threads = (cfg.num_wg + 1) * 128
        dQ_single_wg = cfg.dQ_single_wg
        cluster_size = 1
        use_2cta_instrs = False
    else:
        m_block_size = 128
        n_block_size = 128
        dQ_swapAB = False
        dKV_swapAB = False
        AtomLayoutMdQ = 1
        AtomLayoutNdKV = 1
        requested_disable_2cta = utils._get_disable_2cta_default()
        kv_subtile_factor = get_kv_subtile_factor(block_sparse_tensors, n_block_size)
        use_2cta_instrs = (
            head_dim >= 128
            and not requested_disable_2cta
            and block_sparse_bwd_supports_2cta(block_sparse_tensors, n_block_size)
        )
        if block_sparse_tensors is not None and head_dim == 192 and not use_2cta_instrs:
            reason = (
                "2CTA was disabled by request"
                if requested_disable_2cta
                else (
                    f"sparse_block_size[1] must cover an even number of tile_n={n_block_size} "
                    f"tiles; got factor {kv_subtile_factor}"
                )
            )
            raise ValueError(
                f"SM100 block-sparse backward with head_dim=192 requires 2CTA; {reason}."
            )
        cluster_size = 2 if use_2cta_instrs else 1

    use_dedicated_hd256_kernel = arch // 10 in [10, 11] and head_dim == 256 and head_dim_v == 256
    if (
        use_dedicated_hd256_kernel
        or (arch // 10 in [10, 11] and cu_seqlens_q is not None)
    ) and torch.is_tensor(max_seqlen_q):
        max_seqlen_q = None
    if (
        use_dedicated_hd256_kernel
        or (arch // 10 in [10, 11] and cu_seqlens_k is not None)
    ) and torch.is_tensor(max_seqlen_k):
        max_seqlen_k = None
    if use_dedicated_hd256_kernel:
        assert learnable_sink is None, (
            "SM100 backward with head_dim=256 does not support learnable_sink"
        )
    use_2cta_instrs = use_2cta_instrs or use_dedicated_hd256_kernel
    is_varlen = (
        cu_seqlens_q is not None
        or cu_seqlens_k is not None
        or seqused_q is not None
        or seqused_k is not None
    )

    q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink = [
        maybe_contiguous(t)
        for t in (
            q,
            k,
            v,
            out,
            dout,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            learnable_sink,
        )
    ]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        total_q = q.shape[0]
        seqlen_q = max_seqlen_q if max_seqlen_q is not None else total_q

    if cu_seqlens_k is None:
        batch_size, seqlen_k = k.shape[:2]
        total_k = batch_size * seqlen_k
    else:
        batch_size = cu_seqlens_k.shape[0] - 1
        total_k = k.shape[0]
        seqlen_k = max_seqlen_k if max_seqlen_k is not None else total_k

    num_head_kv = k.shape[-2]

    use_block_sparsity = block_sparse_tensors is not None
    if sparse_q is not None and (sparse_q <= 0 or sparse_q % m_block_size != 0):
        raise ValueError(
            "Block sparsity requires sparse_block_size[0] to be a multiple of "
            f"tile_m={m_block_size}; got {sparse_q}."
        )
    q_subtile_factor = sparse_q // m_block_size if sparse_q is not None else 2
    seqlen_q_rounded = (seqlen_q + m_block_size - 1) // m_block_size * m_block_size
    seqlen_k_rounded = (seqlen_k + n_block_size - 1) // n_block_size * n_block_size
    num_n_blocks = seqlen_k_rounded // n_block_size
    if cluster_size == 2 and num_n_blocks % cluster_size != 0:
        seqlen_k_rounded = seqlen_k_rounded + n_block_size

    # The single-block specialization below only guards against TVM stride poisoning,
    # which is a host-side branch predicate that selects a kernel variant. When
    # max_seqlen is passed as a tensor (e.g. HF/TE varlen), seqlen_*_rounded are tensors,
    # so `seqlen_*_rounded // block == 1` would leak a tensor into the compile key. Its
    # pickle hash differs every call, forcing a recompile per step. Only specialize when
    # the seqlen is already a host scalar; tensor callers fall back to the multi-block
    # default, keeping the key stable with no device sync.
    single_q_block = (not torch.is_tensor(seqlen_q_rounded)) and (seqlen_q_rounded // m_block_size == 1)
    single_k_block = (not torch.is_tensor(seqlen_k_rounded)) and (seqlen_k_rounded // n_block_size == 1)

    if cu_seqlens_k is None:
        assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
        assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
    else:
        assert k.shape == (total_k, num_head_kv, head_dim)
        assert v.shape == (total_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (batch_size + 1,), (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (batch_size + 1,), (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )

        assert out.shape == (total_q, num_head, head_dim_v)
        assert dout.shape == (total_q, num_head, head_dim_v)
        assert lse.shape == (num_head, total_q), "lse must have shape (num_head, total_q)"
    else:
        assert out.shape == (batch_size, seqlen_q, num_head, head_dim_v)
        assert dout.shape == (batch_size, seqlen_q, num_head, head_dim_v)
        assert lse.shape == (batch_size, num_head, seqlen_q), (
            "lse must have shape (batch_size, num_head, seqlen_q)"
        )

    assert q.dtype in [torch.float16, torch.bfloat16], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype, (
        "inputs must have the same dtype"
    )
    for t in [cu_seqlens_q, cu_seqlens_k]:
        if t is not None:
            assert t.dtype == torch.int32, "cu_seqlens_q, cu_seqlens_k must be int32"
    assert lse.dtype == torch.float32, "lse must be float32"
    if dlse is not None:
        dlse = maybe_contiguous(dlse)
    if learnable_sink is not None:
        assert learnable_sink.shape == (num_head,)
        assert learnable_sink.dtype in _LEARNABLE_SINK_DTYPES, (
            "learnable_sink must be float16, bfloat16, or float32"
        )
    if not fake_mode:
        assert all(
            t is None or t.is_cuda
            for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k, learnable_sink)
        ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    alignment = 16 // q.element_size()
    if arch // 10 != 12:
        _validate_head_dims(head_dim, head_dim_v, arch // 10, alignment)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1
    # pack_gqa backward not yet supported in bwd
    pack_gqa = False

    if softcap != 0.0:
        assert score_mod is None and score_mod_bwd is None, (
            "softcap and score_mod/score_mod_bwd cannot be used together"
        )
        score_mod = utils.create_softcap_scoremod(softcap)
        score_mod_bwd = utils.create_softcap_scoremod_bwd(softcap)
    if score_mod is not None:
        assert score_mod_bwd is not None, "score_mod_bwd is required when score_mod is provided"
        if arch // 10 == 8:
            raise NotImplementedError("Custom user-provided score_mod is not supported on SM8x architectures.")

    device = q.device
    out_torch_dtype = q.dtype

    if dq is None:
        dq = torch.empty_like(q)
    else:
        _validate_tensor(dq, "dq", q.shape, out_torch_dtype, device)

    if dk is None:
        dk = torch.empty_like(k)
    else:
        _validate_tensor(dk, "dk", k.shape, out_torch_dtype, device)

    if dv is None:
        dv = torch.empty_like(v)
    else:
        _validate_tensor(dv, "dv", v.shape, out_torch_dtype, device)

    # Keep accumulator allocation, zeroing, and readback aligned with SM90's swapped MMA.
    hdim_multiple_of = 64 if arch // 10 == 9 and dKV_swapAB else 32
    head_dim_rounded = (head_dim + hdim_multiple_of - 1) // hdim_multiple_of * hdim_multiple_of

    if cu_seqlens_q is None:
        dq_accum = (
            None
            if use_dedicated_hd256_kernel
            else torch.empty(
                batch_size,
                num_head,
                seqlen_q_rounded * head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
        )
        dpsum = torch.empty(
            batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device
        )
        lse_log2 = torch.empty(
            batch_size, num_head, seqlen_q_rounded, dtype=torch.float32, device=device
        )
    else:
        total_q_rounded_padded = (
            (total_q + cu_seqlens_q.shape[0] * m_block_size - 1) // m_block_size * m_block_size
        )
        dq_accum = (
            None
            if use_dedicated_hd256_kernel
            else torch.empty(
                num_head, total_q_rounded_padded * head_dim_rounded, dtype=torch.float32, device=device
            )
        )
        dpsum = torch.empty(num_head, total_q_rounded_padded, dtype=torch.float32, device=device)
        lse_log2 = torch.empty(num_head, total_q_rounded_padded, dtype=torch.float32, device=device)

    # GQA (qhead_per_kvhead > 1) needs dK/dV accum+postprocess since multiple Q heads
    # accumulate into the same dK/dV. SM90 varlen_k with qhead_per_kvhead==1 now uses
    # ragged TMA tensors for direct store, so no longer needs accum+postprocess.
    # hd=256 2CTA backward has its own internal postprocess for dK/dV.
    dKV_postprocess = False  # Native group4 owns all Q heads of a KV head in one CTA.
    if dKV_postprocess:
        # Same rounding as the kernel's tile_hdimv and the postprocess (64 with dKV_swapAB on
        # SM90): the GQA epilogue reduces tile_n * tile_hdimv fp32 values per block.
        head_dim_v_rounded = (
            (head_dim_v + hdim_multiple_of - 1) // hdim_multiple_of * hdim_multiple_of
        )
        if cu_seqlens_k is None:
            dk_accum = torch.zeros(
                batch_size,
                num_head_kv,
                seqlen_k_rounded * head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
            dv_accum = torch.zeros(
                batch_size,
                num_head_kv,
                seqlen_k_rounded * head_dim_v_rounded,
                dtype=torch.float32,
                device=device,
            )
        else:
            cluster_tile_n = cluster_size * n_block_size
            total_k_rounded_padded = (
                (total_k + cu_seqlens_k.shape[0] * cluster_tile_n - 1) // cluster_tile_n * cluster_tile_n
            )
            dk_accum = torch.zeros(
                num_head_kv,
                total_k_rounded_padded * head_dim_rounded,
                dtype=torch.float32,
                device=device,
            )
            dv_accum = torch.zeros(
                num_head_kv,
                total_k_rounded_padded * head_dim_v_rounded,
                dtype=torch.float32,
                device=device,
            )

    dtype = torch2cute_dtype_map[q.dtype]

    if deterministic:
        dQ_semaphore = torch.zeros(batch_size, num_head, seqlen_q_rounded // m_block_size, cluster_size, dtype=torch.int32, device=device)
    else:
        dQ_semaphore = None

    if deterministic and qhead_per_kvhead > 1:
        dK_semaphore = torch.zeros(batch_size, num_head_kv, seqlen_k_rounded // n_block_size, 2, dtype=torch.int32, device=device)
        dV_semaphore = torch.zeros(batch_size, num_head_kv, seqlen_k_rounded // n_block_size, 2, dtype=torch.int32, device=device)
    else:
        dK_semaphore = None
        dV_semaphore = None

    # SingleTileVarlenScheduler batch-lookup aid, above BIN_BATCH_SEARCH_THRESH;
    # shared across preprocess, main bwd, and the three postprocess calls.
    cu_total_m_blocks_q = None
    cu_total_m_blocks_k = None
    if is_varlen and batch_size > BIN_BATCH_SEARCH_THRESH and not use_dedicated_hd256_kernel:
        cu_total_m_blocks_q, _ = _compute_tile_cumsum(
            cu_seqlens=cu_seqlens_q,
            seqused=seqused_q,
            tile_size=m_block_size,
        )
        cu_total_m_blocks_k, _ = _compute_tile_cumsum(
            cu_seqlens=cu_seqlens_k,
            seqused=seqused_k,
            tile_size=n_block_size,
            cluster_shape_m=cluster_size,
        )

    dsink = torch.empty_like(learnable_sink) if learnable_sink is not None else None

    # Preprocess kernel: compute (o * dout).sum(dim=-1) - dLSE, lse * log2_e, and zero out dq_accum.
    # For hd=256 dedicated path, dq_accum is None so preprocess only fills dpsum/lse_log2.
    _bwd_preprocess(
        out, dout, dpsum, lse, lse_log2, dq_accum,
        cu_seqlens_q, seqused_q, dlse,
        dtype, head_dim, head_dim_v, m_block_size,
        cu_total_m_blocks=cu_total_m_blocks_q,
        fake_mode=fake_mode,
        hdim_multiple_of=hdim_multiple_of,
    )
    # num_threads: SM90 derives from BwdConfig.num_wg, SM120 is set to 128 above,
    # SM100/SM110 uses default from function signature (384).
    if arch // 10 not in [9, 12]:
        num_threads = 384

    # Backward kernel: compute dk, dv, dq_accum.
    score_mod_hash = utils.hash_callable(score_mod) if score_mod else False
    score_mod_bwd_hash = utils.hash_callable(score_mod_bwd) if score_mod_bwd else False
    mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod else False
    num_aux_tensors = len(aux_tensors) if aux_tensors else 0
    aux_tensor_metadata = get_aux_tensor_metadata(aux_tensors) if aux_tensors is not None else None
    aux_scalar_metadata = tuple(type(s) for s in aux_scalars) if aux_scalars is not None else None

    block_sparse_broadcast_pattern = None
    normalized_block_sparse_tensors = None
    if use_block_sparsity:
        (
            normalized_block_sparse_tensors,
            block_sparse_broadcast_pattern,
        ) = normalize_block_sparse_config_bwd(
            block_sparse_tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            block_size=(m_block_size, n_block_size),
            q_subtile_factor=q_subtile_factor,
            kv_subtile_factor=kv_subtile_factor,
        )
        if deterministic:
            if normalized_block_sparse_tensors.dq_write_order is None:
                raise ValueError(
                    "deterministic block-sparse backward requires dq_write_order in block_sparse_tensors"
                )
            if (
                normalized_block_sparse_tensors.full_block_cnt is not None
                and normalized_block_sparse_tensors.dq_write_order_full is None
            ):
                raise ValueError(
                    "deterministic block-sparse backward requires dq_write_order_full when full blocks are present"
                )
            if normalized_block_sparse_tensors.spt is None:
                raise ValueError(
                    "deterministic block-sparse backward requires block_sparse_tensors.spt "
                    "to match dq_write_order direction"
                )
    if (
        normalized_block_sparse_tensors is not None
        and normalized_block_sparse_tensors.spt is not None
    ):
        spt = normalized_block_sparse_tensors.spt and deterministic
    else:
        spt = (causal or local) and deterministic

    if arch // 10 in [8, 9, 12]:
        compile_key = (
            arch,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            causal,
            window_size_left is not None,
            window_size_right is not None,
            m_block_size,
            n_block_size,
            num_threads,
            pack_gqa,
            num_stages_Q,
            num_stages_dO,
            SdP_swapAB,
            dKV_swapAB,
            dQ_swapAB,
            AtomLayoutMSdP,
            AtomLayoutNdKV,
            AtomLayoutMdQ,
            V_in_regs,
            dQ_single_wg,
            deterministic,
            cu_seqlens_q is None,
            cu_seqlens_k is None,
            seqused_q is None,
            seqused_k is None,
            score_mod_hash,
            score_mod_bwd_hash,
            mask_mod_hash,
            num_aux_tensors,
            aux_tensor_metadata,
            aux_scalar_metadata,
            use_block_sparsity,
            q_subtile_factor,
            block_sparse_broadcast_pattern,
            get_broadcast_dims(q),
            get_broadcast_dims(k),
            get_broadcast_dims(v),
            get_broadcast_dims(dout),
            # Prevent TVM stride poisoning when only one block is present.
            single_q_block,
            single_k_block,
            cu_total_m_blocks_k is not None,
        )
    else:
        compile_key = (
            arch,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            causal,
            window_size_left is not None,
            window_size_right is not None,
            m_block_size,
            n_block_size,
            num_threads,
            pack_gqa,
            cluster_size,
            use_2cta_instrs,
            q_subtile_factor,
            kv_subtile_factor,
            deterministic,
            spt,
            score_mod_hash,
            score_mod_bwd_hash,
            mask_mod_hash,
            num_aux_tensors,
            aux_tensor_metadata,
            aux_scalar_metadata,
            use_block_sparsity,
            block_sparse_broadcast_pattern,
            cu_seqlens_q is None,
            cu_seqlens_k is None,
            seqused_q is None,
            seqused_k is None,
            get_broadcast_dims(q),
            get_broadcast_dims(k),
            get_broadcast_dims(v),
            get_broadcast_dims(dout),
            # Prevent TVM stride poisoning when only one block is present.
            single_q_block,
            single_k_block,
            cu_total_m_blocks_k is not None,
            use_dedicated_hd256_kernel and cu_seqlens_q is not None and max_seqlen_q is None,
            use_dedicated_hd256_kernel and cu_seqlens_k is not None and max_seqlen_k is None,
        )

    if compile_key not in _flash_attn_bwd.compile_cache:
        current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        cute_aux_tensors = (
            [to_cute_aux_tensor(buf) for buf in aux_tensors]
            if aux_tensors is not None
            else None
        )
        q_tensor, k_tensor, v_tensor, do_tensor, dq_tensor, dk_tensor, dv_tensor = [
            to_cute_tensor(t) for t in (q, k, v, dout, dq, dk, dv)
        ]
        lse_log2_tensor, dpsum_tensor = [to_cute_tensor(t) for t in (lse_log2, dpsum)]
        dq_accum_tensor = to_cute_tensor(dq_accum) if dq_accum is not None else None
        if dKV_postprocess:
            dk_accum_tensor, dv_accum_tensor = [
                to_cute_tensor(t) for t in (dk_accum, dv_accum)
            ]
        cu_seqlens_q_tensor, cu_seqlens_k_tensor, seqused_q_tensor, seqused_k_tensor, cu_total_m_blocks_k_tensor = [
            to_cute_tensor(t, assumed_align=4) if t is not None else None
            for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cu_total_m_blocks_k)
        ]
        dQ_semaphore_tensor, dK_semaphore_tensor, dV_semaphore_tensor = [
            utils.convert_from_dlpack_leading_static(t.detach(), leading_dim=3, alignment=4, stride_order=t.dim_order())
            if t is not None else None
            for t in (dQ_semaphore, dK_semaphore, dV_semaphore)
        ]
        if arch // 10 in [8, 12]:
            flash_bwd_obj_cls = FlashAttentionBackwardSm120 if arch // 10 == 12 else FlashAttentionBackwardSm80
            fa_bwd_obj = flash_bwd_obj_cls(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                m_block_size,
                n_block_size,
                num_stages_Q,
                num_stages_dO,
                num_threads,
                pack_gqa,
                causal,
                local,
                SdP_swapAB,
                dKV_swapAB,
                dQ_swapAB,
                AtomLayoutMSdP,
                AtomLayoutNdKV,
                AtomLayoutMdQ,
                V_in_regs=V_in_regs,
                score_mod=score_mod,
                score_mod_bwd=score_mod_bwd,
            )
        elif arch // 10 == 9:
            fa_bwd_obj = FlashAttentionBackwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                causal,
                is_local=local,
                deterministic=deterministic,
                tile_m=m_block_size,
                tile_n=n_block_size,
                Q_stage=num_stages_Q,
                dO_stage=num_stages_dO,
                PdS_stage=num_stages_PdS,
                SdP_swapAB=SdP_swapAB,
                dKV_swapAB=dKV_swapAB,
                dQ_swapAB=dQ_swapAB,
                AtomLayoutMSdP=AtomLayoutMSdP,
                AtomLayoutNdKV=AtomLayoutNdKV,
                AtomLayoutMdQ=AtomLayoutMdQ,
                num_threads=num_threads,
                V_in_regs=V_in_regs,
                score_mod=score_mod,
                score_mod_bwd=score_mod_bwd,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
                q_subtile_factor=q_subtile_factor,
                dQ_single_wg=dQ_single_wg,
            )
        else:
            if use_dedicated_hd256_kernel:
                assert softcap == 0.0, "SM100 backward with head_dim=256 does not support softcap"
                assert block_sparse_tensors is None, \
                    "SM100 backward with head_dim=256 does not support block sparsity"
                assert dlse is None, \
                    "SM100 backward with head_dim=256 does not support dlse"
                assert seqused_q is None and seqused_k is None, \
                    "SM100 backward with head_dim=256 does not support seqused_q/seqused_k"

                dq_tile_mn = (128, 128)
                dkdv_tile_mn = (128, 64)
                fa_bwd_obj = BlackwellFusedMultiHeadAttentionBackward(
                    head_dim,
                    head_dim_v,
                    is_causal=causal,
                    is_local=local,
                    qhead_per_kvhead=qhead_per_kvhead,
                    is_persistent=False,
                    deterministic=deterministic,
                    cluster_size=cluster_size,
                    use_2cta_instrs=use_2cta_instrs,
                    score_mod=score_mod,
                    score_mod_bwd=score_mod_bwd,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    q_subtile_factor=q_subtile_factor,
                    tile_m_dq=dq_tile_mn[0],
                    tile_n_dq=dq_tile_mn[1],
                    tile_m_dkdv=dkdv_tile_mn[0],
                    tile_n_dkdv=dkdv_tile_mn[1],
                )
            else:
                fa_bwd_obj = FlashAttentionBackwardSm100(
                    head_dim,
                    head_dim_v,
                    is_causal=causal,
                    is_local=local,
                    qhead_per_kvhead=qhead_per_kvhead,
                    tile_m=m_block_size,
                    tile_n=n_block_size,
                    cluster_size=cluster_size,
                    use_2cta_instrs=use_2cta_instrs,
                    deterministic=deterministic,
                    spt=spt,
                    score_mod=score_mod,
                    score_mod_bwd=score_mod_bwd,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    q_subtile_factor=q_subtile_factor,
                    kv_subtile_factor=kv_subtile_factor,
                )

        # Block sparse tensors for backward use Q-direction indexing (transposed from forward).
        sparse_tensors_compile = None
        if normalized_block_sparse_tensors is not None:
            sparse_tensors_compile = to_cute_block_sparse_tensors(normalized_block_sparse_tensors)
        dq_accum_tensor = dq_tensor if use_dedicated_hd256_kernel else dq_accum_tensor

        compile_args = [
            fa_bwd_obj,
            q_tensor,
            k_tensor,
            v_tensor,
            do_tensor,
            lse_log2_tensor,
            dpsum_tensor,
            dq_accum_tensor,
            dk_tensor if not dKV_postprocess else dk_accum_tensor,
            dv_tensor if not dKV_postprocess else dv_accum_tensor,
            softmax_scale,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            window_size_left,
            window_size_right,
            dQ_semaphore_tensor,
            dK_semaphore_tensor,
            dV_semaphore_tensor,
            AuxData(cute_aux_tensors, aux_scalars),
            sparse_tensors_compile,
        ]
        if not use_dedicated_hd256_kernel:
            compile_args.append(cu_total_m_blocks_k_tensor)
        else:
            compile_args.extend(
                (
                    Int32(max_seqlen_q)
                    if cu_seqlens_q is not None and max_seqlen_q is not None
                    else None,
                    Int32(max_seqlen_k)
                    if cu_seqlens_k is not None and max_seqlen_k is not None
                    else None,
                )
            )
        compile_args.append(current_stream)

        # TODO: check @can_implement
        _flash_attn_bwd.compile_cache[compile_key] = cute.compile(
            *compile_args, options="--enable-tvm-ffi"
        )
    if not fake_mode:
        dq_accum = dq if use_dedicated_hd256_kernel else dq_accum
        call_args = [
            q.detach(),
            k.detach(),
            v.detach(),
            dout,
            lse_log2,
            dpsum,
            dq_accum,
            dk if not dKV_postprocess else dk_accum,
            dv if not dKV_postprocess else dv_accum,
            softmax_scale,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            window_size_left,
            window_size_right,
            dQ_semaphore,
            dK_semaphore,
            dV_semaphore,
            AuxData(aux_tensors, aux_scalars),
            (
                normalized_block_sparse_tensors.mask_block_cnt,
                normalized_block_sparse_tensors.mask_block_idx,
                normalized_block_sparse_tensors.full_block_cnt,
                normalized_block_sparse_tensors.full_block_idx,
                normalized_block_sparse_tensors.cu_total_m_blocks,
                normalized_block_sparse_tensors.cu_block_idx_offsets,
                normalized_block_sparse_tensors.dq_write_order,
                normalized_block_sparse_tensors.dq_write_order_full,
            )
            if normalized_block_sparse_tensors is not None
            else None,
        ]
        if not use_dedicated_hd256_kernel:
            call_args.append(cu_total_m_blocks_k)
        else:
            call_args.extend(
                (
                    max_seqlen_q
                    if cu_seqlens_q is not None
                    else None,
                    max_seqlen_k
                    if cu_seqlens_k is not None
                    else None,
                )
            )
        _flash_attn_bwd.compile_cache[compile_key](*call_args)
    # Postprocess: convert dq_accum from float32 to dq in bf16/fp16
    # hd=256 2CTA backward has its own internal postprocess, skip here.
    if not use_dedicated_hd256_kernel:
        if arch // 10 == 9:
            # dQ postprocess: match main kernel's MMA WG count, unless dQ_single_wg
            num_threads_post_dQ = 128 if dQ_single_wg else cfg.num_wg * 128
            num_threads_post_dKV = cfg.num_wg * 128
        else:
            num_threads_post_dQ = 128
            num_threads_post_dKV = 128

        _bwd_postprocess_convert(
            dq_accum, dq, softmax_scale,
            cu_seqlens_q, seqused_q,
            arch, dtype, head_dim, m_block_size, num_threads_post_dQ,
            AtomLayoutMdQ, dQ_swapAB,
            use_2cta_instrs=use_2cta_instrs, cluster_size=1,
            cu_total_m_blocks=cu_total_m_blocks_q,
            sink_tensors=(
                LearnableSinkBwdTensors(dpsum, lse, learnable_sink, dsink)
                if learnable_sink is not None
                else None
            ),
            fake_mode=fake_mode,
            hdim_multiple_of=hdim_multiple_of,
            row_index=aux_tensors[4],
        )

        if dKV_postprocess:
            # Postprocess: convert dk_accum from float32 to dk in bf16/fp16
            _bwd_postprocess_convert(
                dk_accum, dk, softmax_scale,
                cu_seqlens_k, seqused_k,
                arch, dtype, head_dim, n_block_size, num_threads_post_dKV,
                AtomLayoutNdKV, dKV_swapAB,
                cluster_size=cluster_size,
                cu_total_m_blocks=cu_total_m_blocks_k if cluster_size == 1 else None,
                fake_mode=fake_mode,
                hdim_multiple_of=hdim_multiple_of,
            )
            # Postprocess: convert dv_accum from float32 to dv in bf16/fp16
            _bwd_postprocess_convert(
                dv_accum, dv, 1.0,
                cu_seqlens_k, seqused_k,
                arch, dtype, head_dim_v, n_block_size, num_threads_post_dKV,
                AtomLayoutNdKV, dKV_swapAB,
                cluster_size=cluster_size,
                cu_total_m_blocks=cu_total_m_blocks_k if cluster_size == 1 else None,
                fake_mode=fake_mode,
                hdim_multiple_of=hdim_multiple_of,
            )

    return (dq, dk, dv) if learnable_sink is None else (dq, dk, dv, dsink)

_flash_attn_bwd.compile_cache = {}
