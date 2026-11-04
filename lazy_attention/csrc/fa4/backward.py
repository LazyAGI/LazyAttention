# SPDX-License-Identifier: BSD-3-Clause
# Derived from FlashAttention CuTe; see THIRD_PARTY_NOTICES.md.
# Modified: fixed Hopper D128/GQA4 geometry; index/range policy is compile-time.
import math
from lazy_attention.csrc.fa4.paired_copy import pair_segments, pair_tma, pair_gather
from lazy_attention.csrc.fa4.row_map import decode_row_map, mapped_row
from typing import Callable, Optional, Type
from functools import partial
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute import FastDivmodDivisorV2
from cutlass import Float32, Int32, Boolean, const_expr
from cutlass.utils import LayoutEnum
from quack import copy_utils
from quack import layout_utils
from quack import sm90_utils
from quack.sm90_utils import gemm_zero_init, gemm_w_idx
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute import utils
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute import pipeline
from quack.cute_dsl_utils import ParamsBase
from flash_attn.cute.tile_scheduler import TileSchedulerArguments, SingleTileScheduler
from flash_attn.cute.named_barrier import NamedBarrierBwd
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.utils import AuxData
from flash_attn.cute.block_sparse_utils import (
    get_total_q_block_count_bwd,
    get_block_sparse_iteration_info_bwd,
    _store_one_dQaccum_sm90,
)
from lazy_attention.csrc.fa4.scheduler import (
    PersistentScheduler,
    shifted_q_tile,
    SubsetBarrier,
    subset_barrier,
    subset_arrive,
)


class FlashAttentionBackwardSm90:
    arch = 90

    def __init__(self, dtype, mask_mod, range_mapping, physical_ctas):
        self.dtype, self.mask_mod = dtype, mask_mod
        self.range_mapping, self.physical_ctas = range_mapping, physical_ctas
        self.persistent_producer_regs = 24
        self.q_shift = 0
        self.planned_order = False
        self.split_tasks = False
        self.tile_hdim = self.tile_hdimv = 128
        self.tile_m, self.tile_n = 64, 128
        self.num_threads = 384
        self.Q_stage = self.dO_stage = self.PdS_stage = 2
        self.SdP_swapAB, self.dKV_swapAB, self.dQ_swapAB = True, False, False
        self.AtomLayoutMSdP, self.AtomLayoutNdKV, self.AtomLayoutMdQ = 1, 2, 1
        self.num_wg_mma, self.num_wg_dQ = 2, 1
        self.is_causal = self.is_local = self.deterministic = False
        self.shuffle_LSE = self.shuffle_dPsum = False
        self.buffer_align_bytes = 1024
        self.q_subtile_factor = 1

    def _check_type(
        self,
        mQ_type: Type[cutlass.Numeric],
        mK_type: Type[cutlass.Numeric],
        mV_type: Type[cutlass.Numeric],
        mdO_type: Type[cutlass.Numeric],
        mLSE_type: Type[cutlass.Numeric],
        mdPsum_type: Type[cutlass.Numeric],
        mdQaccum_type: Type[cutlass.Numeric],
        mdK_type: Type[cutlass.Numeric],
        mdV_type: Type[cutlass.Numeric],
    ):
        if const_expr(not mQ_type == mK_type == mV_type == mdO_type):
            raise TypeError("All tensors must have the same data type")
        if const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Only Float16 or BFloat16 is supported")
        if const_expr(mLSE_type not in [Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(mdPsum_type not in [Float32]):
            raise TypeError("dPsum tensor must be Float32")
        if const_expr(mdQaccum_type not in [Float32]):
            raise TypeError("dQaccum tensor must be Float32")
        if const_expr(not mdK_type == mdV_type == mQ_type):
            raise TypeError("mdK and mdV tensors must have the same data type as mQ")
        assert mQ_type == self.dtype

    def _setup_attributes(self):
        wg_d_dKV = self.num_wg_mma // self.AtomLayoutNdKV
        self.sQ_layout, self.sdO_layout = [
            sm90_utils.make_smem_layout(
                self.dtype, LayoutEnum.ROW_MAJOR, shape, stage, major_mode_size=mms
            )
            for (shape, stage, mms) in [
                (
                    (self.tile_m, self.tile_hdim),
                    self.Q_stage,
                    self.tile_hdim // wg_d_dKV,
                ),
                (
                    (self.tile_m, self.tile_hdimv),
                    self.dO_stage,
                    self.tile_hdim // wg_d_dKV,
                ),
            ]
        ]
        wg_d_dQ = self.num_wg_dQ // self.AtomLayoutMdQ
        self.sK_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_n, self.tile_hdim),
            stage=None,
            major_mode_size=self.tile_hdim // wg_d_dQ,
        )
        self.sV_layout = sm90_utils.make_smem_layout(
            self.dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_hdimv), None
        )
        wg_n_SdP = self.num_wg_mma // self.AtomLayoutMSdP
        wg_n_dKV = self.AtomLayoutNdKV
        self.sPdS_layout = sm90_utils.make_smem_layout(
            self.dtype,
            LayoutEnum.ROW_MAJOR,
            (self.tile_m, self.tile_n),
            stage=self.PdS_stage,
            major_mode_size=math.gcd(self.tile_n // wg_n_SdP, self.tile_n // wg_n_dKV),
        )
        self.sdQaccum_layout = cute.make_layout(
            (self.tile_m * self.tile_hdim // self.num_wg_dQ, self.num_wg_dQ)
        )
        self.r2s_tiled_copy_dQaccum = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
            ),
            cute.make_layout((self.num_threads_per_warp_group, self.num_wg_dQ)),
            cute.make_layout(128 // Float32.width),
        )

    def _get_tiled_mma(self):
        maybe_swap_mn = lambda shape, swap: (
            (shape[1], shape[0], *shape[2:]) if swap else shape
        )
        atom_layout_SdP = (
            self.AtomLayoutMSdP,
            self.num_wg_mma // self.AtomLayoutMSdP,
            1,
        )
        tiler_mn_SdP = (
            self.tile_m // atom_layout_SdP[0],
            self.tile_n // atom_layout_SdP[1],
        )
        tiled_mma_SdP = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            Float32,
            atom_layout_mnk=maybe_swap_mn(atom_layout_SdP, self.SdP_swapAB),
            tiler_mn=(64, tiler_mn_SdP[0]),
        )
        atom_layout_dKV = (
            self.AtomLayoutNdKV,
            self.num_wg_mma // self.AtomLayoutNdKV,
            1,
        )
        tiler_mn_dK = (
            self.tile_n // atom_layout_dKV[0],
            self.tile_hdim // atom_layout_dKV[1],
        )
        tiler_mn_dV = (
            self.tile_n // atom_layout_dKV[0],
            self.tile_hdimv // atom_layout_dKV[1],
        )
        tiled_mma_dK, tiled_mma_dV = [
            sm90_utils_basic.make_trivial_tiled_mma(
                self.dtype,
                self.dtype,
                warpgroup.OperandMajorMode.K,
                warpgroup.OperandMajorMode.MN,
                Float32,
                atom_layout_mnk=maybe_swap_mn(atom_layout_dKV, self.dKV_swapAB),
                tiler_mn=(64, tiler_mn_d[1]),
                a_source=warpgroup.OperandSource.RMEM,
            )
            for tiler_mn_d in (tiler_mn_dK, tiler_mn_dV)
        ]
        assert self.num_wg_dQ % self.AtomLayoutMdQ == 0
        atom_layout_dQ = (self.AtomLayoutMdQ, self.num_wg_dQ // self.AtomLayoutMdQ, 1)
        tiler_mn_dQ = (
            self.tile_m // atom_layout_dQ[0],
            self.tile_hdim // atom_layout_dQ[1],
        )
        tiled_mma_dQ = sm90_utils_basic.make_trivial_tiled_mma(
            self.dtype,
            self.dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.MN,
            Float32,
            atom_layout_mnk=maybe_swap_mn(atom_layout_dQ, self.dQ_swapAB),
            tiler_mn=(64, tiler_mn_dQ[1]),
        )
        return (tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ)

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct, sV_struct, sdO_struct, sdQaccum_struct = [
            cute.struct.Align[
                cute.struct.MemRange[t, cute.cosize(layout)], self.buffer_align_bytes
            ]
            for (layout, t) in [
                (self.sQ_layout, self.dtype),
                (self.sK_layout, self.dtype),
                (self.sV_layout, self.dtype),
                (self.sdO_layout, self.dtype),
                (self.sdQaccum_layout, Float32),
            ]
        ]
        cosize_sdS = cute.cosize(self.sPdS_layout)
        cosize_sP = 0
        sLSE_struct = cute.struct.Align[
            cute.struct.MemRange[
                Float32, cute.round_up(self.tile_m, 64) * self.Q_stage
            ],
            128,
        ]
        sdPsum_struct = cute.struct.Align[
            cute.struct.MemRange[
                Float32, cute.round_up(self.tile_m, 64) * self.dO_stage
            ],
            128,
        ]

        @cute.struct
        class SharedStorageQKV:
            task_mailbox: cute.struct.MemRange[Int32, 8 if self.split_tasks else 5]
            mbar_ptr_Q: cute.struct.MemRange[cutlass.Int64, self.Q_stage * 2]
            mbar_ptr_dO: cute.struct.MemRange[cutlass.Int64, self.dO_stage * 2]
            if const_expr(self.range_mapping):
                mbar_ptr_KV: cute.struct.MemRange[cutlass.Int64, 1]
            sLSE: sLSE_struct
            sdPsum: sdPsum_struct
            sQ: sQ_struct
            sV: sV_struct
            sK: sK_struct
            sdO: sdO_struct
            sP: cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sP], 1024]
            sdS: cute.struct.Align[cute.struct.MemRange[self.dtype, cosize_sdS], 1024]
            sdQaccum: sdQaccum_struct

        return SharedStorageQKV

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        aux_data: AuxData = AuxData(),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        mCuTotalMBlocks: Optional[cute.Tensor] = None,
        stream: cuda.CUstream = None,
    ):
        self.varlen_k = mCuSeqlensK is not None or mSeqUsedK is not None
        self._check_type(
            *(
                t.element_type if t is not None else None
                for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV)
            )
        )
        self.is_varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None
        mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV = [
            assume_tensor_aligned(t)
            for t in (mQ, mK, mV, mdO, mLSE, mdPsum, mdQaccum, mdK, mdV)
        ]

        def _qkv_transpose(t):
            return layout_utils.select(
                t, [1, 3, 2, 0] if cute.rank(t.shape) == 4 else [0, 2, 1]
            )

        mQ, mK, mV, mdO = [_qkv_transpose(t) for t in (mQ, mK, mV, mdO)]
        mdK, mdV = [_qkv_transpose(t) for t in (mdK, mdV)]
        LSE_dPsum_dQaccum_transpose = (
            [2, 1, 0] if cute.rank(mLSE.shape) == 3 else [1, 0]
        )
        mLSE, mdPsum, mdQaccum = [
            layout_utils.select(t, LSE_dPsum_dQaccum_transpose)
            for t in (mLSE, mdPsum, mdQaccum)
        ]
        tiled_mma_SdP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ = self._get_tiled_mma()
        mdK_semaphore = None
        mdV_semaphore = None
        self.num_mma_threads = tiled_mma_SdP.size
        assert self.num_mma_threads + 128 == self.num_threads
        self.num_threads_per_warp_group = 128
        self.num_producer_threads = 32
        REG_LIMIT = 504
        self.num_mma_regs_wg0 = 256
        self.num_mma_regs_wg1 = 248 - self.persistent_producer_regs
        self.num_mma_regs = self.num_mma_regs_wg0
        self.num_producer_regs = self.persistent_producer_regs
        assert (
            self.num_mma_regs_wg0 + self.num_mma_regs_wg1 + self.num_producer_regs
            <= REG_LIMIT
        )
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for (name, mX, layout) in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
                ("dO", mdO, self.sdO_layout),
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dQ"] = (
            self.tile_m * self.tile_hdim * Float32.width // 8 // self.num_wg_dQ
        )
        self.tma_copy_bytes["dKacc"] = self.tile_n * self.tile_hdim * Float32.width // 8
        self.tma_copy_bytes["dVacc"] = (
            self.tile_n * self.tile_hdimv * Float32.width // 8
        )
        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdim),
        )
        tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mK,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
        )
        tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mV,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv),
        )
        tma_atom_dO, tma_tensor_dO = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1]),
            (self.tile_m, self.tile_hdimv),
        )
        mdK_tma = mdK
        mdV_tma = mdV
        tma_atom_dK, tma_tensor_dK = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mdK_tma,
            cute.select(self.sK_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdim),
        )
        tma_atom_dV, tma_tensor_dV = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            mdV_tma,
            cute.select(self.sV_layout, mode=[0, 1]),
            (self.tile_n, self.tile_hdimv),
        )
        TileScheduler = SingleTileScheduler
        self.spt = (self.is_causal or self.is_local) and self.deterministic
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mK.shape[0]), self.tile_n),
            cute.size(mK.shape[2]),
            cute.size(mK.shape[3]),
            1,
            cute.size(mQ.shape[0]),
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mK.shape[0]) * cute.size(mK.shape[3]),
            tile_shape_mn=(self.tile_n, self.tile_m),
            mCuSeqlensQ=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedK,
            qhead_per_kvhead_packgqa=1,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=self.spt,
            head_swizzle=self.deterministic,
            cu_total_m_blocks_ptr=mCuTotalMBlocks,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = (self.physical_ctas, 1, 1)
        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = softmax_scale * LOG2_E
        fastdiv_mods = None
        if const_expr(aux_data.tensors is not None):
            seqlen_q = cute.size(mQ.shape[0])
            seqlen_k = cute.size(mK.shape[0])
            seqlen_q_divmod = FastDivmodDivisorV2(seqlen_q)
            seqlen_k_divmod = FastDivmodDivisorV2(seqlen_k)
            fastdiv_mods = (seqlen_q_divmod, seqlen_k_divmod)
        qhead_per_kvhead_divmod = None
        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)
        sub_atoms = []
        sub_tensors = []
        for rows in (16, 4):
            for tensor, layout in ((mQ, self.sQ_layout), (mdO, self.sdO_layout)):
                base_layout = cute.select(layout, mode=[0, 1])
                sub_layout = cute.composition(
                    base_layout,
                    cute.make_layout((rows, self.tile_hdim), stride=(1, self.tile_m)),
                )
                atom, tensor_tma = cpasync.make_tiled_tma_atom(
                    cpasync.CopyBulkTensorTileG2SOp(),
                    tensor,
                    sub_layout,
                    (rows, self.tile_hdim),
                )
                sub_atoms.append(atom)
                sub_tensors.append(tensor_tma)
        if const_expr(self.range_mapping):
            for rows in (64, 16, 4):
                for tensor, layout in ((mK, self.sK_layout), (mV, self.sV_layout)):
                    sub_layout = cute.composition(
                        layout,
                        cute.make_layout(
                            (rows, self.tile_hdim), stride=(1, self.tile_n)
                        ),
                    )
                    atom, tensor_tma = cpasync.make_tiled_tma_atom(
                        cpasync.CopyBulkTensorTileG2SOp(),
                        tensor,
                        sub_layout,
                        (rows, self.tile_hdim),
                    )
                    sub_atoms.append(atom)
                    sub_tensors.append(tensor_tma)
        self.kernel(
            tuple(sub_atoms),
            tuple(sub_tensors),
            tma_tensor_Q,
            tma_tensor_dO,
            tma_tensor_K,
            tma_tensor_V,
            tma_tensor_dK,
            tma_tensor_dV,
            mQ,
            mK,
            mV,
            mdO,
            mdK,
            mdV,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dK,
            tma_atom_dV,
            mLSE,
            mdPsum,
            mdQaccum,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sPdS_layout,
            self.sdO_layout,
            self.sdQaccum_layout,
            self.r2s_tiled_copy_dQaccum,
            tiled_mma_SdP,
            tiled_mma_dK,
            tiled_mma_dV,
            tiled_mma_dQ,
            softmax_scale_log2,
            softmax_scale,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            aux_data,
            fastdiv_mods,
            blocksparse_tensors,
            qhead_per_kvhead_divmod,
            mdQ_semaphore,
            mdK_semaphore,
            mdV_semaphore,
            window_size_left,
            window_size_right,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        sub_atoms: tuple,
        sub_tensors: tuple,
        tQ: cute.Tensor,
        tdO: cute.Tensor,
        tK: cute.Tensor,
        tV: cute.Tensor,
        tdK: cute.Tensor,
        tdV: cute.Tensor,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sPdS_layout: cute.ComposedLayout,
        sdO_layout: cute.ComposedLayout,
        sdQaccum_layout: cute.Layout,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        softmax_scale_log2,
        softmax_scale,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisorV2] = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            for atom in [
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_dO,
                tma_atom_dK,
                tma_atom_dV,
            ]:
                if const_expr(atom is not None):
                    cpasync.prefetch_descriptor(atom)
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread
        )
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, self.num_mma_threads // cute.arch.WARP_SIZE
        )
        pipeline_Q = pipeline.PipelineAsync.create(
            barrier_storage=storage.mbar_ptr_Q.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            elect_one_commit=True,
            elect_one_release=True,
            defer_sync=True,
        )
        pipeline_dO = pipeline.PipelineAsync.create(
            barrier_storage=storage.mbar_ptr_dO.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            elect_one_commit=True,
            elect_one_release=True,
            defer_sync=False,
        )
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sP = None
        sdS = storage.sdS.get_tensor(sPdS_layout.outer, swizzle=sPdS_layout.inner)
        sLSE = storage.sLSE.get_tensor(
            cute.make_layout(
                (self.tile_m, self.Q_stage), stride=(1, cute.round_up(self.tile_m, 64))
            )
        )
        sdPsum = storage.sdPsum.get_tensor(
            cute.make_layout(
                (self.tile_m, self.dO_stage), stride=(1, cute.round_up(self.tile_m, 64))
            )
        )
        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)
        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            False,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            tile_m=self.tile_m,
            tile_n=self.tile_n,
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            swap_AB=self.SdP_swapAB,
        )
        TileSchedulerCls = partial(
            PersistentScheduler.create,
            tile_sched_params,
            aux_data.tensors[2],
            aux_data.tensors[3],
            storage.task_mailbox.get_tensor(cute.make_layout(8 if self.split_tasks else 5)),
            participants=384 if self.range_mapping else 320,
            planned=self.planned_order,
            split=self.split_tasks,
        )
        if warp_idx < 4:
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            if warp_idx == 0:
                if const_expr(self.range_mapping):
                    self.load(
                        sub_atoms,
                        sub_tensors,
                        tQ,
                        tdO,
                        tK,
                        tV,
                        mQ,
                        mK,
                        mV,
                        mdO,
                        mLSE,
                        mdPsum,
                        sQ,
                        sK,
                        sV,
                        sdO,
                        sLSE,
                        sdPsum,
                        tma_atom_Q,
                        tma_atom_K,
                        tma_atom_V,
                        tma_atom_dO,
                        pipeline_Q,
                        pipeline_dO,
                        block_info,
                        SeqlenInfoCls,
                        TileSchedulerCls,
                        blocksparse_tensors,
                        qhead_per_kvhead_divmod,
                        aux_data.tensors[4],
                        aux_data.tensors[5],
                        storage.mbar_ptr_KV.data_ptr(),
                    )
                else:
                    self.load(
                        sub_atoms,
                        sub_tensors,
                        tQ,
                        tdO,
                        tK,
                        tV,
                        mQ,
                        mK,
                        mV,
                        mdO,
                        mLSE,
                        mdPsum,
                        sQ,
                        sK,
                        sV,
                        sdO,
                        sLSE,
                        sdPsum,
                        tma_atom_Q,
                        tma_atom_K,
                        tma_atom_V,
                        tma_atom_dO,
                        pipeline_Q,
                        pipeline_dO,
                        block_info,
                        SeqlenInfoCls,
                        TileSchedulerCls,
                        blocksparse_tensors,
                        qhead_per_kvhead_divmod,
                        aux_data.tensors[4],
                    )
            if warp_idx == 1:
                self.dQaccum_store(
                    mdQaccum,
                    sdQaccum,
                    block_info,
                    TileSchedulerCls,
                    SeqlenInfoCls,
                    blocksparse_tensors,
                    mdQ_semaphore,
                )
            if const_expr(self.range_mapping):
                if warp_idx >= 2:
                    self.dKV_scatter(
                        sK,
                        sV,
                        mdK,
                        mdV,
                        tdK,
                        tdV,
                        tma_atom_dK,
                        tma_atom_dV,
                        TileSchedulerCls,
                        aux_data.tensors[5],
                        cute.size(aux_data.tensors[4]) // 64,
                    )
        else:
            tidx, _, _ = cute.arch.thread_idx()
            tidx = tidx - 128
            mma_args = (
                tiled_mma_SdP,
                tiled_mma_dK,
                tiled_mma_dV,
                tiled_mma_dQ,
                mdK,
                mdV,
                mdK_semaphore,
                mdV_semaphore,
                mdQaccum,
                sQ,
                sK,
                sV,
                sdO,
                sP,
                sdS,
                sLSE,
                sdPsum,
                sdQaccum,
                pipeline_Q,
                pipeline_dO,
                tidx,
                tma_atom_dK,
                tma_atom_dV,
                r2s_tiled_copy_dQaccum,
                softmax_scale_log2,
                softmax_scale,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                aux_data,
                fastdiv_mods,
                blocksparse_tensors,
                qhead_per_kvhead_divmod,
            )
            warp_idx_in_mma = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - 4
            if warp_idx_in_mma < 4:
                cute.arch.setmaxregister_increase(self.num_mma_regs_wg0)
                self.mma(*mma_args, is_dQ_wg=True)
            else:
                cute.arch.setmaxregister_increase(self.num_mma_regs_wg1)
                self.mma(*mma_args, is_dQ_wg=False)

    @cute.jit
    def load(
        self,
        sub_atoms: tuple,
        sub_tensors: tuple,
        tQ: cute.Tensor,
        tdO: cute.Tensor,
        tK: cute.Tensor,
        tV: cute.Tensor,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisorV2] = None,
        row_index: Optional[cute.Tensor] = None,
        row_descriptor: Optional[cute.Tensor] = None,
        kv_barrier=None,
    ):
        producer_state_Q = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_dO = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
        )
        tile_scheduler = TileSchedulerCls(is_producer=True)
        work_tile = tile_scheduler.initial_work_tile_info()
        lane = cute.arch.lane_idx()
        if const_expr(self.range_mapping):
            if lane == 0:
                cute.arch.mbarrier_init(kv_barrier, 1)
                cute.arch.mbarrier_init_fence()
            cute.arch.sync_warp()
            kv_phase = Int32(0)
            while work_tile.is_valid_tile:
                n_block, kv_head, batch_idx, _ = work_tile.tile_idx
                pc, pi, fc, fi, count = get_block_sparse_iteration_info_bwd(
                    blocksparse_tensors, batch_idx, kv_head * 4, n_block
                )
                if count > 0:
                    base128 = row_descriptor[n_block * 2]
                    if base128 >= 0 and row_descriptor[n_block * 2 + 1] == base128 + 64:
                        pair_tma(
                            tK,
                            tV,
                            sK,
                            sV,
                            tma_atom_K,
                            tma_atom_V,
                            base128,
                            kv_head,
                            kv_barrier,
                            kv_barrier,
                            128,
                        )
                    else:
                        for half in cutlass.range(2, unroll=1):
                            block = n_block * 2 + half
                            header_kv = row_descriptor[block]
                            dK = cute.local_tile(sK, (64, 128), (half, 0))
                            dV = cute.local_tile(sV, (64, 128), (half, 0))
                            if header_kv >= 0:
                                pair_tma(
                                    sub_tensors[4],
                                    sub_tensors[5],
                                    dK,
                                    dV,
                                    sub_atoms[4],
                                    sub_atoms[5],
                                    header_kv,
                                    kv_head,
                                    kv_barrier,
                                    kv_barrier,
                                    64,
                                )
                            else:
                                ptr = row_descriptor[cute.size(row_index) // 64 + block]
                                if header_kv > -64:
                                    pair_segments(
                                        mK,
                                        mV,
                                        sub_tensors[6],
                                        sub_tensors[7],
                                        sub_tensors[8],
                                        sub_tensors[9],
                                        dK,
                                        dV,
                                        sub_atoms[6],
                                        sub_atoms[7],
                                        sub_atoms[8],
                                        sub_atoms[9],
                                        row_descriptor,
                                        ptr,
                                        row_descriptor[ptr],
                                        row_descriptor[ptr + 1],
                                        -header_kv,
                                        kv_head,
                                        kv_barrier,
                                        kv_barrier,
                                    )
                                else:
                                    pair_gather(
                                        mK,
                                        mV,
                                        dK,
                                        dV,
                                        row_descriptor,
                                        ptr,
                                        Int32(0),
                                        Int32(0),
                                        Int32(0),
                                        Int32(0),
                                        kv_head,
                                        64,
                                    )
                    if lane == 0:
                        cute.arch.mbarrier_arrive(kv_barrier)
                    cute.arch.mbarrier_wait(kv_barrier, kv_phase)
                    kv_phase = kv_phase ^ 1
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_warp()
                    cute.arch.fence_view_async_shared()
                q_start, q_end = tile_scheduler.q_bounds(count)
                for it in cutlass.range(q_start, q_end, unroll=1):
                    m_block, is_full = shifted_q_tile(
                        it, pc, pi, fc, fi, n_block, self.q_shift
                    )
                    header = row_descriptor[m_block]
                    contiguous = header >= 0
                    base_row = header
                    split = Int32(0)
                    payload = Int32(0)
                    base0, base1 = (Int32(0), Int32(0))
                    if header < 0:
                        payload = row_descriptor[cute.size(row_index) // 64 + m_block]
                    if header < 0 and header > -64:
                        split = -header
                        n_q = cute.size(row_index) // 64
                        base0 = row_descriptor[payload]
                        base1 = row_descriptor[payload + 1]
                    bulk_stats = contiguous and base_row % 4 == 0
                    for head_offset in cutlass.range(4, unroll=1):
                        head_idx = kv_head * 4 + head_offset
                        pipeline_Q.producer_acquire(producer_state_Q)
                        pipeline_dO.producer_acquire(producer_state_dO)
                        if contiguous:
                            bar_q = pipeline_Q.producer_get_barrier(producer_state_Q)
                            bar_do = pipeline_dO.producer_get_barrier(producer_state_dO)
                            if lane == 0:
                                cute.arch.mbarrier_expect_tx(bar_q, 64 * 128 * 2)
                                cute.arch.mbarrier_expect_tx(bar_do, 64 * 128 * 2)
                            cute.arch.sync_warp()
                            indexed_tma_load(
                                tQ,
                                sQ[None, None, producer_state_Q.index],
                                tma_atom_Q,
                                base_row,
                                head_idx,
                                bar_q,
                            )
                            indexed_tma_load(
                                tdO,
                                sdO[None, None, producer_state_dO.index],
                                tma_atom_dO,
                                base_row,
                                head_idx,
                                bar_do,
                            )
                        else:
                            pair_segments(
                                mQ,
                                mdO,
                                sub_tensors[0],
                                sub_tensors[1],
                                sub_tensors[2],
                                sub_tensors[3],
                                sQ[None, None, producer_state_Q.index],
                                sdO[None, None, producer_state_dO.index],
                                sub_atoms[0],
                                sub_atoms[1],
                                sub_atoms[2],
                                sub_atoms[3],
                                row_descriptor,
                                payload,
                                base0,
                                base1,
                                split,
                                head_idx,
                                pipeline_Q.producer_get_barrier(producer_state_Q),
                                pipeline_dO.producer_get_barrier(producer_state_dO),
                            )
                            cute.arch.cp_async_commit_group()
                            cute.arch.cp_async_wait_group(0)
                        if bulk_stats:
                            indexed_bulk_stats(
                                mLSE[None, head_idx, batch_idx],
                                sLSE[None, producer_state_Q.index],
                                base_row,
                                pipeline_Q.producer_get_barrier(producer_state_Q),
                            )
                            indexed_bulk_stats(
                                mdPsum[None, head_idx, batch_idx],
                                sdPsum[None, producer_state_dO.index],
                                base_row,
                                pipeline_dO.producer_get_barrier(producer_state_dO),
                            )
                        else:
                            for r in cutlass.range(2, unroll=1):
                                row = lane + r * 32
                                physical = Int32(0)
                                if contiguous:
                                    physical = base_row + row
                                elif split > 0:
                                    physical = affine_row(row, base0, base1, split)
                                else:
                                    physical = row_descriptor[payload + row]
                                sLSE[row, producer_state_Q.index] = mLSE[
                                    physical, head_idx, batch_idx
                                ]
                                sdPsum[row, producer_state_dO.index] = mdPsum[
                                    physical, head_idx, batch_idx
                                ]
                        cute.arch.sync_warp()
                        cute.arch.fence_view_async_shared()
                        if lane == 0:
                            cute.arch.mbarrier_arrive(
                                pipeline_Q.producer_get_barrier(producer_state_Q)
                            )
                            cute.arch.mbarrier_arrive(
                                pipeline_dO.producer_get_barrier(producer_state_dO)
                            )
                        producer_state_Q.advance()
                        producer_state_dO.advance()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()
        else:
            while work_tile.is_valid_tile:
                n_block, kv_head, batch_idx, _ = work_tile.tile_idx
                pc, pi, fc, fi, count = get_block_sparse_iteration_info_bwd(
                    blocksparse_tensors, batch_idx, kv_head * 4, n_block
                )
                if count > 0:
                    indexed_load(
                        mK, sK, row_index, n_block * 128, kv_head, Int32(0), 128, False
                    )
                    indexed_load(
                        mV, sV, row_index, n_block * 128, kv_head, Int32(0), 128, False
                    )
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_warp()
                    cute.arch.fence_view_async_shared()
                q_start, q_end = tile_scheduler.q_bounds(count)
                for it in cutlass.range(q_start, q_end, unroll=1):
                    m_block, is_full = shifted_q_tile(
                        it, pc, pi, fc, fi, n_block, self.q_shift
                    )
                    base_row = row_index[m_block * 64]
                    contiguous = Boolean(True)
                    for r in cutlass.range_constexpr(2):
                        rr = lane + r * 32
                        contiguous = contiguous & (
                            row_index[m_block * 64 + rr] == base_row + rr
                        )
                    contiguous = cute.arch.vote_all_sync(contiguous)
                    for head_offset in cutlass.range(4, unroll=1):
                        head_idx = kv_head * 4 + head_offset
                        pipeline_Q.producer_acquire(producer_state_Q)
                        pipeline_dO.producer_acquire(producer_state_dO)
                        if contiguous:
                            bar_q = pipeline_Q.producer_get_barrier(producer_state_Q)
                            bar_do = pipeline_dO.producer_get_barrier(producer_state_dO)
                            if lane == 0:
                                cute.arch.mbarrier_expect_tx(bar_q, 64 * 128 * 2)
                                cute.arch.mbarrier_expect_tx(bar_do, 64 * 128 * 2)
                            cute.arch.sync_warp()
                            indexed_tma_load(
                                tQ,
                                sQ[None, None, producer_state_Q.index],
                                tma_atom_Q,
                                base_row,
                                head_idx,
                                bar_q,
                            )
                            indexed_tma_load(
                                tdO,
                                sdO[None, None, producer_state_dO.index],
                                tma_atom_dO,
                                base_row,
                                head_idx,
                                bar_do,
                            )
                        else:
                            indexed_segment_load(
                                mQ,
                                sub_tensors[0],
                                sub_tensors[2],
                                sQ[None, None, producer_state_Q.index],
                                sub_atoms[0],
                                sub_atoms[2],
                                row_index,
                                m_block * 64,
                                head_idx,
                                pipeline_Q.producer_get_barrier(producer_state_Q),
                            )
                            indexed_segment_load(
                                mdO,
                                sub_tensors[1],
                                sub_tensors[3],
                                sdO[None, None, producer_state_dO.index],
                                sub_atoms[1],
                                sub_atoms[3],
                                row_index,
                                m_block * 64,
                                head_idx,
                                pipeline_dO.producer_get_barrier(producer_state_dO),
                            )
                            cute.arch.cp_async_commit_group()
                            cute.arch.cp_async_wait_group(0)
                        if contiguous and base_row % 4 == 0:
                            indexed_bulk_stats(
                                mLSE[None, head_idx, batch_idx],
                                sLSE[None, producer_state_Q.index],
                                base_row,
                                pipeline_Q.producer_get_barrier(producer_state_Q),
                            )
                            indexed_bulk_stats(
                                mdPsum[None, head_idx, batch_idx],
                                sdPsum[None, producer_state_dO.index],
                                base_row,
                                pipeline_dO.producer_get_barrier(producer_state_dO),
                            )
                        else:
                            for r in cutlass.range(2, unroll=1):
                                row = lane + r * 32
                                physical = row_index[m_block * 64 + row]
                                sLSE[row, producer_state_Q.index] = mLSE[
                                    physical, head_idx, batch_idx
                                ]
                                sdPsum[row, producer_state_dO.index] = mdPsum[
                                    physical, head_idx, batch_idx
                                ]
                        cute.arch.sync_warp()
                        cute.arch.fence_view_async_shared()
                        if lane == 0:
                            cute.arch.mbarrier_arrive(
                                pipeline_Q.producer_get_barrier(producer_state_Q)
                            )
                            cute.arch.mbarrier_arrive(
                                pipeline_dO.producer_get_barrier(producer_state_dO)
                            )
                        producer_state_Q.advance()
                        producer_state_dO.advance()
                tile_scheduler.advance_to_next_work()
                work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def mma(
        self,
        tiled_mma_SdP: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        mdQaccum: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sP: Optional[cute.Tensor],
        sdS: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        sdQaccum: cute.Tensor,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        tidx: Int32,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        r2s_tiled_copy_dQaccum: cute.TiledCopy,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=(None, None),
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisorV2] = None,
        is_dQ_wg: cutlass.Constexpr[bool] = True,
    ):
        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        warp_group_thread_layout = cute.make_layout(
            self.num_wg_mma, stride=self.num_threads_per_warp_group
        )
        thr_mma_SdP = tiled_mma_SdP.get_slice(tidx)
        wg_mma_SdP = tiled_mma_SdP.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dK = tiled_mma_dK.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dV = tiled_mma_dV.get_slice(warp_group_thread_layout(warp_group_idx))
        wg_mma_dQ = None
        if const_expr(is_dQ_wg):
            wg_idx_dQ = 0
            wg_mma_dQ = tiled_mma_dQ.get_slice(warp_group_thread_layout(wg_idx_dQ))
        shape_mnk_S = (self.tile_m, self.tile_n, self.tile_hdim)
        mask_acc, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
            wg_mma_SdP, shape_mnk_S, sQ, sK, swap_AB=self.SdP_swapAB
        )
        mma_qk_fn = partial(
            gemm_zero_init,
            tiled_mma_SdP,
            shape_mnk_S[:2],
            tSrQ,
            tSrK,
            swap_AB=self.SdP_swapAB,
        )
        shape_mnk_dP = (self.tile_m, self.tile_n, self.tile_hdimv)
        _, tdPrdO, tdPrV = sm90_utils.partition_fragment_ABC(
            wg_mma_SdP, shape_mnk_dP, sdO, sV, swap_AB=self.SdP_swapAB
        )
        mma_dov_fn = partial(
            gemm_zero_init,
            tiled_mma_SdP,
            shape_mnk_dP[:2],
            tdPrdO,
            tdPrV,
            swap_AB=self.SdP_swapAB,
        )
        sPt = layout_utils.transpose_view(sP) if sP is not None else None
        sdOt = layout_utils.transpose_view(sdO)
        shape_mnk_dV = (self.tile_n, self.tile_hdimv, self.tile_m)
        acc_dV, tdVrPt, tdVrdOt = sm90_utils.partition_fragment_ABC(
            wg_mma_dV, shape_mnk_dV, sPt, sdOt, swap_AB=self.dKV_swapAB
        )
        mma_pdo_fn = partial(gemm_w_idx, tiled_mma_dV, acc_dV, tCrB=tdVrdOt)
        sdSt = layout_utils.transpose_view(sdS)
        sQt = layout_utils.transpose_view(sQ)
        shape_mnk_dK = (self.tile_n, self.tile_hdim, self.tile_m)
        acc_dK, tdKrdSt, tdKrQt = sm90_utils.partition_fragment_ABC(
            wg_mma_dK, shape_mnk_dK, sdSt, sQt, swap_AB=self.dKV_swapAB
        )
        mma_dsq_fn = partial(gemm_w_idx, tiled_mma_dK, acc_dK, tCrB=tdKrQt)
        sKt = layout_utils.transpose_view(sK)
        shape_mnk_dQ = (self.tile_m, self.tile_hdim, self.tile_n)
        mma_dsk_fn = None
        if const_expr(is_dQ_wg):
            _, tdQrdS, tdQrKt = sm90_utils.partition_fragment_ABC(
                wg_mma_dQ, shape_mnk_dQ, sdS, sKt, swap_AB=self.dQ_swapAB
            )
            mma_dsk_fn = partial(
                gemm_zero_init,
                tiled_mma_dQ,
                shape_mnk_dQ[:2],
                tdQrdS,
                tdQrKt,
                swap_AB=self.dQ_swapAB,
            )
        copy_P_r2s = None
        mms_PdS = self.tile_n // (self.num_wg_mma // self.AtomLayoutMSdP)
        if const_expr(sP is not None):
            sP_cpy = sPt
            copy_P_r2s, _, _ = copy_utils.get_smem_store_C(
                tiled_mma_SdP,
                sP_cpy,
                tidx,
                transpose=self.SdP_swapAB,
                position_independent=True,
                major_mode_size=mms_PdS,
            )
        sdS_cpy = sdSt
        copy_dS_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma_SdP,
            sdS_cpy,
            tidx,
            transpose=self.SdP_swapAB,
            position_independent=True,
            major_mode_size=mms_PdS,
        )
        tLSEsLSE = layout_utils.mma_partition_C_vec(
            sLSE, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=not self.SdP_swapAB
        )
        tLSEsdPsum = layout_utils.mma_partition_C_vec(
            sdPsum, thr_mma_SdP, expand_shape=self.tile_n, is_colvec=not self.SdP_swapAB
        )
        shfl_copy = copy_utils.tiled_copy_1d(
            sLSE.element_type, num_threads=8, num_copy_elems=2
        )
        tdQsdQaccum = None
        if const_expr(is_dQ_wg):
            smem_thr_copy_dQaccum = r2s_tiled_copy_dQaccum.get_slice(tidx)
            tdQsdQaccum = smem_thr_copy_dQaccum.partition_D(sdQaccum)
        PdS_barrier = SubsetBarrier(
            barrier_id=int(NamedBarrierBwd.PdS), num_threads=self.num_mma_threads
        )
        mma_one_m_block_all = partial(
            self.mma_one_m_block,
            warp_group_idx=warp_group_idx,
            mma_qk_fn=mma_qk_fn,
            mma_dov_fn=mma_dov_fn,
            mma_pdo_fn=mma_pdo_fn,
            mma_dsq_fn=mma_dsq_fn,
            mma_dsk_fn=mma_dsk_fn,
            copy_P_r2s=copy_P_r2s,
            copy_dS_r2s=copy_dS_r2s,
            pipeline_Q=pipeline_Q,
            pipeline_dO=pipeline_dO,
            tLSEsLSE=tLSEsLSE,
            tLSEsdPsum=tLSEsdPsum,
            tdQsdQaccum=tdQsdQaccum,
            softmax_scale_log2=softmax_scale_log2,
            PdS_barrier=PdS_barrier,
            is_dQ_wg=is_dQ_wg,
        )
        consumer_state_Q = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        consumer_state_dO = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            recompute_fastdiv_mods_q = const_expr(
                aux_data.tensors is not None
                and (seqlen.has_cu_seqlens_q or seqlen.has_seqused_q)
            )
            recompute_fastdiv_mods_k = const_expr(
                aux_data.tensors is not None
                and (seqlen.has_cu_seqlens_k or seqlen.has_seqused_k)
            )
            if const_expr(fastdiv_mods is not None and fastdiv_mods[0] is not None):
                seqlen_q_divmod, seqlen_k_divmod = fastdiv_mods
                fastdiv_mods = (
                    (
                        seqlen_q_divmod
                        if not recompute_fastdiv_mods_q
                        else FastDivmodDivisorV2(seqlen.seqlen_q)
                    ),
                    (
                        seqlen_k_divmod
                        if not recompute_fastdiv_mods_k
                        else FastDivmodDivisorV2(seqlen.seqlen_k)
                    ),
                )
            mask = AttentionMaskCls(seqlen)
            m_block_min, m_block_max = block_info.get_m_block_min_max(seqlen, n_block)
            total_m_block_cnt = get_total_q_block_count_bwd(
                blocksparse_tensors,
                batch_idx,
                head_idx,
                n_block,
                q_subtile_factor=self.q_subtile_factor,
                m_block_max=m_block_max,
            )
            process_tile = total_m_block_cnt > Int32(0)
            if process_tile:
                pc, pi, fc, fi, count = get_block_sparse_iteration_info_bwd(
                    blocksparse_tensors, batch_idx, head_idx * 4, n_block
                )
                dKV_accumulate = False
                allowed = cute.make_fragment_like(mask_acc, Boolean)
                q_start, q_end = tile_scheduler.q_bounds(count)
                for it in cutlass.range(q_start, q_end, unroll=1):
                    m_block, is_full = shifted_q_tile(
                        it, pc, pi, fc, fi, n_block, self.q_shift
                    )
                    if not is_full:
                        mask_acc.fill(0.0)
                        mask.apply_mask(
                            mask_acc,
                            batch_idx=batch_idx,
                            head_idx=head_idx * 4,
                            m_block=m_block,
                            n_block=n_block,
                            thr_mma=thr_mma_SdP,
                            mask_seqlen=False,
                            mask_causal=False,
                            mask_mod=self.mask_mod,
                            aux_data=aux_data,
                            fastdiv_mods=fastdiv_mods,
                        )
                        for i in cutlass.range_constexpr(cute.size(allowed)):
                            allowed[i] = mask_acc[i] == 0.0
                    for head_offset in cutlass.range(4, unroll=1):
                        if is_full:
                            consumer_state_Q, consumer_state_dO = mma_one_m_block_all(
                                m_block,
                                consumer_state_Q,
                                consumer_state_dO,
                                mask_fn=None,
                                dKV_accumulate=dKV_accumulate,
                            )
                        else:
                            consumer_state_Q, consumer_state_dO = mma_one_m_block_all(
                                m_block,
                                consumer_state_Q,
                                consumer_state_dO,
                                mask_fn=partial(
                                    self.apply_cached_mask, allowed=allowed
                                ),
                                dKV_accumulate=dKV_accumulate,
                            )
                        dKV_accumulate = True
                acc_dK.store(acc_dK.load() * softmax_scale)
                if const_expr(self.range_mapping):
                    self.epilogue_dKV(
                        acc_dV,
                        mdV,
                        sV,
                        acc_dK,
                        mdK,
                        sK,
                        seqlen,
                        tma_atom_dK,
                        tma_atom_dV,
                        tiled_mma_dK,
                        tiled_mma_dV,
                        tidx,
                        n_block,
                        head_idx,
                        batch_idx,
                        qhead_per_kvhead_divmod,
                        mdK_semaphore,
                        mdV_semaphore,
                        aux_data.tensors[5],
                        cute.size(aux_data.tensors[4]) // 64,
                        partial_k=aux_data.tensors[6] if const_expr(self.split_tasks) else None,
                        partial_v=aux_data.tensors[7] if const_expr(self.split_tasks) else None,
                        partial_slot=tile_scheduler.mailbox[7] if const_expr(self.split_tasks) else Int32(0),
                    )
                else:
                    self.epilogue_dKV(
                        acc_dV,
                        mdV,
                        sV,
                        acc_dK,
                        mdK,
                        sK,
                        seqlen,
                        tma_atom_dK,
                        tma_atom_dV,
                        tiled_mma_dK,
                        tiled_mma_dV,
                        tidx,
                        n_block,
                        head_idx,
                        batch_idx,
                        qhead_per_kvhead_divmod,
                        mdK_semaphore,
                        mdV_semaphore,
                        aux_data.tensors[4],
                    )
            else:
                acc_dK.fill(0.0)
                acc_dV.fill(0.0)
                if const_expr(self.range_mapping):
                    self.epilogue_dKV(
                        acc_dV,
                        mdV,
                        sV,
                        acc_dK,
                        mdK,
                        sK,
                        seqlen,
                        tma_atom_dK,
                        tma_atom_dV,
                        tiled_mma_dK,
                        tiled_mma_dV,
                        tidx,
                        n_block,
                        head_idx,
                        batch_idx,
                        qhead_per_kvhead_divmod,
                        mdK_semaphore,
                        mdV_semaphore,
                        aux_data.tensors[5],
                        cute.size(aux_data.tensors[4]) // 64,
                        partial_k=aux_data.tensors[6] if const_expr(self.split_tasks) else None,
                        partial_v=aux_data.tensors[7] if const_expr(self.split_tasks) else None,
                        partial_slot=tile_scheduler.mailbox[7] if const_expr(self.split_tasks) else Int32(0),
                    )
                else:
                    self.epilogue_dKV(
                        acc_dV,
                        mdV,
                        sV,
                        acc_dK,
                        mdK,
                        sK,
                        seqlen,
                        tma_atom_dK,
                        tma_atom_dV,
                        tiled_mma_dK,
                        tiled_mma_dV,
                        tidx,
                        n_block,
                        head_idx,
                        batch_idx,
                        qhead_per_kvhead_divmod,
                        mdK_semaphore,
                        mdV_semaphore,
                        aux_data.tensors[4],
                    )
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 4:
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @staticmethod
    @cute.jit
    def _get_stat(tSrS: cute.Tensor, row: Int32, lane: Int32, shuffle: bool) -> Float32:
        """Retrieve the statistic for a given accumulator row.

        When shuffle=False, direct register indexing.
        When shuffle=True, warp shuffle from the thread group that holds the value.
        """
        if const_expr(not shuffle):
            return tSrS[row]
        vecsize = cute.size(tSrS, mode=[0, 0])
        idx0, off, idx1 = cute.idx2crd(row, (vecsize, 8, cute.shape(tSrS, mode=[0, 1])))
        return utils.shuffle_sync(
            tSrS[idx0 + idx1 * vecsize], offset=off * 4 + lane % 4
        )

    @cute.jit
    def mma_one_m_block(
        self,
        m_block: Int32,
        consumer_state_Q: cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple,
        consumer_state_dO: (
            cutlass.pipeline.PipelineState | pipeline.PipelineStateSimple
        ),
        warp_group_idx: Int32,
        mma_qk_fn: Callable,
        mma_dov_fn: Callable,
        mma_pdo_fn: Callable,
        mma_dsq_fn: Callable,
        mma_dsk_fn: Callable,
        copy_P_r2s: Optional[Callable],
        copy_dS_r2s: Callable,
        pipeline_Q: cutlass.pipeline.PipelineAsync,
        pipeline_dO: cutlass.pipeline.PipelineAsync,
        tLSEsLSE: cute.Tensor,
        tLSEsdPsum: cute.Tensor,
        tdQsdQaccum: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        PdS_barrier: cutlass.pipeline.NamedBarrier,
        is_dQ_wg: cutlass.Constexpr[bool] = True,
        mask_fn: Optional[Callable] = None,
        dKV_accumulate: Boolean = True,
    ):
        consumer_state_dO_cur = consumer_state_Q
        smem_idx_Q = consumer_state_Q.index
        smem_idx_dO = consumer_state_dO_cur.index
        smem_idx_PdS = smem_idx_Q
        pipeline_Q.consumer_wait(
            consumer_state_Q, pipeline_Q.consumer_try_wait(consumer_state_Q)
        )
        acc_S = mma_qk_fn(A_idx=smem_idx_Q, wg_wait=-1)
        tLSErLSE = copy_utils.load_s2r(tLSEsLSE[None, smem_idx_Q])
        pipeline_dO.consumer_wait(
            consumer_state_dO_cur, pipeline_dO.consumer_try_wait(consumer_state_dO_cur)
        )
        acc_dP = mma_dov_fn(A_idx=smem_idx_Q, wg_wait=1)
        if cutlass.const_expr(mask_fn is not None):
            mask_fn(acc_S, m_block=m_block)
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.SdP_swapAB)
        lane_idx = cute.arch.lane_idx()
        for r in cutlass.range_constexpr(cute.size(acc_S_mn, mode=[0])):
            lse_val = self._get_stat(tLSErLSE, r, lane_idx, shuffle=self.shuffle_LSE)
            for c in cutlass.range(cute.size(acc_S_mn, mode=[1]), unroll_full=True):
                acc_S_mn[r, c] = cute.math.exp2(
                    acc_S_mn[r, c] * softmax_scale_log2 - lse_val, fastmath=True
                )
        tLSErdPsum = copy_utils.load_s2r(tLSEsdPsum[None, smem_idx_dO])
        tdVrP = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_S), self.dtype)
        warpgroup.wait_group(0)
        acc_dP_mn = layout_utils.reshape_acc_to_mn(acc_dP, transpose=self.SdP_swapAB)
        for r in cutlass.range_constexpr(cute.size(acc_dP_mn, mode=[0])):
            dpsum_val = self._get_stat(
                tLSErdPsum, r, lane_idx, shuffle=self.shuffle_dPsum
            )
            for c in cutlass.range(cute.size(acc_dP_mn, mode=[1]), unroll_full=True):
                acc_dP_mn[r, c] = acc_S_mn[r, c] * (acc_dP_mn[r, c] - dpsum_val)
        tdKrdS = utils.cvt_f16(layout_utils.reshape_acc_to_frgA(acc_dP), self.dtype)
        copy_dS_r2s(tdKrdS, dst_idx=smem_idx_PdS)
        mma_pdo_fn(
            tCrA=tdVrP, B_idx=smem_idx_dO, zero_init=not dKV_accumulate, wg_wait=-1
        )
        cute.arch.fence_view_async_shared()
        PdS_barrier.arrive_and_wait()
        if const_expr(is_dQ_wg):
            acc_dQ = mma_dsk_fn(A_idx=smem_idx_PdS, wg_wait=1)
            pipeline_dO.consumer_release(consumer_state_dO_cur)
            mma_dsq_fn(
                tCrA=tdKrdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1
            )
            subset_barrier(
                barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
                number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
            )
            tdQrdQaccum_flat = cute.make_tensor(
                acc_dQ.iterator, cute.make_layout(tdQsdQaccum.shape)
            )
            cute.autovec_copy(tdQrdQaccum_flat, tdQsdQaccum)
            cute.arch.fence_view_async_shared()
            subset_arrive(
                barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
                number_of_threads=self.num_threads_per_warp_group + cute.arch.WARP_SIZE,
            )
            warpgroup.wait_group(0)
            pipeline_Q.consumer_release(consumer_state_Q)
        else:
            mma_dsq_fn(
                tCrA=tdKrdS, B_idx=smem_idx_Q, zero_init=not dKV_accumulate, wg_wait=1
            )
            pipeline_dO.consumer_release(consumer_state_dO_cur)
            warpgroup.wait_group(0)
            pipeline_Q.consumer_release(consumer_state_Q)
        consumer_state_Q.advance()
        consumer_state_dO.advance()
        return (consumer_state_Q, consumer_state_dO)

    @cute.jit
    def dKV_scatter(
        self,
        sK,
        sV,
        mdK,
        mdV,
        tdK,
        tdV,
        atomK,
        atomV,
        TileSchedulerCls: Callable,
        desc,
        n_q: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        tid = tid - 64
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        bar = SubsetBarrier(
            barrier_id=int(NamedBarrierBwd.Epilogue),
            num_threads=self.num_mma_threads + 64,
        )
        while work_tile.is_valid_tile:
            n_block, head, batch, _ = work_tile.tile_idx
            bar.arrive_and_wait()
            bar.arrive_and_wait()
            do_store = Boolean(True)
            if const_expr(self.split_tasks):
                do_store = tile_scheduler.mailbox[7] < 0
            if do_store:
                h0, h1 = (desc[n_block * 2], desc[n_block * 2 + 1])
                if h0 >= 0 and h1 == h0 + 64:
                    if cute.arch.warp_idx() == 2:
                        range_tma_store(sV, tdV, atomV, h0, head)
                        range_tma_store(sK, tdK, atomK, h0, head)
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=False)
                else:
                    range_store128(sV, mdV, desc, n_q, n_block, h0, h1, head, tid, 64)
                    range_store128(sK, mdK, desc, n_q, n_block, h0, h1, head, tid, 64)
            bar.arrive_and_wait()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def epilogue_dKV(
        self,
        acc_dV: cute.Tensor,
        mdV: cute.Tensor,
        sV: cute.Tensor,
        acc_dK: cute.Tensor,
        mdK: cute.Tensor,
        sK: cute.Tensor,
        seqlen: SeqlenInfoQK,
        tma_atom_dK: cute.CopyAtom,
        tma_atom_dV: cute.CopyAtom,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tidx: Int32,
        n_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
        qhead_per_kvhead_divmod: Optional[FastDivmodDivisorV2] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        row_descriptor: Optional[cute.Tensor] = None,
        n_q: Int32 = 0,
        partial_k=None, partial_v=None, partial_slot=0,
    ):
        if const_expr(self.range_mapping):
            epi_barrier = SubsetBarrier(
                barrier_id=int(NamedBarrierBwd.Epilogue),
                num_threads=self.num_mma_threads + 64,
            )
        else:
            epi_barrier = SubsetBarrier(
                barrier_id=int(NamedBarrierBwd.Epilogue),
                num_threads=self.num_mma_threads,
            )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        mdK_cur = seqlen.offset_batch_K(mdK, batch_idx, dim=3, ragged=self.varlen_k)[
            None, None, head_idx
        ]
        mdV_cur = seqlen.offset_batch_K(mdV, batch_idx, dim=3, ragged=self.varlen_k)[
            None, None, head_idx
        ]
        if const_expr(self.split_tasks):
            if partial_slot >= 0:
                store_partial(acc_dK, partial_k, partial_slot, tiled_mma_dK, tidx, self.dKV_swapAB)
                store_partial(acc_dV, partial_v, partial_slot, tiled_mma_dV, tidx, self.dKV_swapAB)
        sdV = sV
        sdK = sK
        copy_dV_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma_dV,
            sdV,
            tidx,
            transpose=self.dKV_swapAB,
            position_independent=True,
        )
        copy_dK_r2s, _, _ = copy_utils.get_smem_store_C(
            tiled_mma_dK,
            sdK,
            tidx,
            transpose=self.dKV_swapAB,
            position_independent=True,
        )
        cute.arch.cp_async_bulk_wait_group(1, read=True)
        epi_barrier.arrive_and_wait()
        copy_dV_r2s(acc_dV, dst_idx=None)
        if const_expr(self.range_mapping):
            pass
        else:
            cute.arch.fence_view_async_shared()
            epi_barrier.arrive_and_wait()
            indexed_store(
                sV,
                mdV,
                row_descriptor,
                n_block * 128,
                head_idx,
                tidx,
                128,
                self.num_mma_threads,
            )
            epi_barrier.arrive_and_wait()
            cute.arch.cp_async_bulk_wait_group(1, read=True)
            epi_barrier.arrive_and_wait()
        copy_dK_r2s(acc_dK, dst_idx=None)
        cute.arch.fence_view_async_shared()
        epi_barrier.arrive_and_wait()
        if const_expr(self.range_mapping):
            pass
        else:
            indexed_store(
                sK,
                mdK,
                row_descriptor,
                n_block * 128,
                head_idx,
                tidx,
                128,
                self.num_mma_threads,
            )
        epi_barrier.arrive_and_wait()

    @cute.jit
    def dQaccum_store(
        self,
        mdQaccum: cute.Tensor,
        sdQaccum: cute.Tensor,
        block_info: BlockInfo,
        TileSchedulerCls: cutlass.Constexpr[Callable],
        SeqlenInfoCls: cutlass.Constexpr[Callable],
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
    ):
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, kv_head, batch_idx, _ = work_tile.tile_idx
            pc, pi, fc, fi, count = get_block_sparse_iteration_info_bwd(
                blocksparse_tensors, batch_idx, kv_head * 4, n_block
            )
            q_start, q_end = tile_scheduler.q_bounds(count)
            for it in cutlass.range(q_start, q_end, unroll=1):
                m_block, is_full = shifted_q_tile(
                    it, pc, pi, fc, fi, n_block, self.q_shift
                )
                for head_offset in cutlass.range(4, unroll=1):
                    head_idx = kv_head * 4 + head_offset
                    mdQaccum_cur = mdQaccum[None, head_idx, batch_idx]
                    gdQaccum = cute.local_tile(
                        mdQaccum_cur,
                        (
                            cute.make_layout(
                                (
                                    self.tile_m * self.tile_hdim // self.num_wg_dQ,
                                    self.num_wg_dQ,
                                )
                            ),
                        ),
                        (None,),
                    )
                    _store_one_dQaccum_sm90(
                        m_block,
                        sdQaccum,
                        gdQaccum,
                        num_dQ_warp_groups=self.num_wg_dQ,
                        num_threads_per_warp_group=self.num_threads_per_warp_group,
                        tma_copy_bytes_dQ=self.tma_copy_bytes["dQ"],
                    )
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
        cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def apply_cached_mask(self, acc_S, m_block, allowed):
        for i in cutlass.range_constexpr(cute.size(acc_S)):
            acc_S[i] = acc_S[i] if allowed[i] else -Float32.inf


@cute.jit
def _store_one_dQaccum_sm90(
    m_block,
    sdQaccum: cute.Tensor,
    gdQaccum: cute.Tensor,
    num_dQ_warp_groups: cutlass.Constexpr,
    num_threads_per_warp_group: cutlass.Constexpr,
    tma_copy_bytes_dQ,
):
    """Store dQaccum for a single m_block."""
    for warp_group_idx in cutlass.range_constexpr(num_dQ_warp_groups):
        cute.arch.cp_async_bulk_wait_group(
            num_dQ_warp_groups - 1 - warp_group_idx, read=True
        )
        subset_arrive(
            barrier_id=int(NamedBarrierBwd.dQEmptyWG0) + warp_group_idx,
            number_of_threads=num_threads_per_warp_group + cute.arch.WARP_SIZE,
        )
    for warp_group_idx in cutlass.range_constexpr(num_dQ_warp_groups):
        subset_barrier(
            barrier_id=int(NamedBarrierBwd.dQFullWG0) + warp_group_idx,
            number_of_threads=num_threads_per_warp_group + cute.arch.WARP_SIZE,
        )
        with cute.arch.elect_one():
            copy_utils.cpasync_reduce_bulk_add_f32(
                sdQaccum[None, warp_group_idx].iterator,
                gdQaccum[(None, warp_group_idx), m_block].iterator,
                tma_copy_bytes_dQ,
            )
        cute.arch.cp_async_bulk_commit_group()


@cute.jit
def indexed_tma_load(src, dst, atom, base, head, bar):
    tile = cute.local_tile(
        cute.domain_offset((base, 0), src[None, None, head, 0]), (64, 128), (0, 0)
    )
    copy, _, _ = copy_utils.tma_get_copy_fn(
        atom, 0, cute.make_layout(1), tile, dst, single_stage=True
    )
    copy(tma_bar_ptr=bar)


@cute.jit
def indexed_bulk_stats(src, dst, base, bar):
    gs = cute.local_tile(cute.domain_offset((base,), src), (64,), (0,))
    if cute.arch.lane_idx() == 0:
        cute.arch.mbarrier_expect_tx(bar, 64 * 4)
    cute.arch.sync_warp()
    copy = copy_utils.cpasync_bulk_get_copy_fn(gs, dst, single_stage=True)
    copy(tma_bar_ptr=bar)


@cute.jit
def affine_row(row, base0, base1, split):
    result = base0 + row
    if row >= split:
        result = base1 + row - split
    return result


@cute.jit
def mapped_store64(src, dst, desc, mapping, head, tid, threads: cutlass.Constexpr):
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), dst.element_type, num_bits_per_copy=128
    )
    for i in cutlass.range(64 * 128 // (threads * 8), unroll=1):
        linear = tid * 8 + i * threads * 8
        row, col = (linear // 128, linear % 128)
        physical = mapped_row(desc, row, mapping)
        ss = cute.local_tile(src[row, None], (8,), (col // 8,))
        gd = cute.local_tile(dst[physical, None, head, 0], (8,), (col // 8,))
        reg = cute.make_rmem_tensor((8,), dst.element_type)
        cute.copy(atom, ss, reg)
        cute.copy(atom, reg, gd)


@cute.jit
def range_store128(
    src, dst, desc, n_q, block, header0, header1, head, tid, threads: cutlass.Constexpr
):
    if header0 >= 0 and header1 == header0 + 64:
        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), dst.element_type, num_bits_per_copy=128
        )
        for i in cutlass.range(128 * 128 // (threads * 8), unroll=1):
            linear = tid * 8 + i * threads * 8
            row, col = (linear // 128, linear % 128)
            ss = cute.local_tile(src[row, None], (8,), (col // 8,))
            gd = cute.local_tile(dst[header0 + row, None, head, 0], (8,), (col // 8,))
            reg = cute.make_rmem_tensor((8,), dst.element_type)
            cute.copy(atom, ss, reg)
            cute.copy(atom, reg, gd)
    else:
        mapping0 = decode_row_map(desc, n_q, block * 2)
        mapping1 = decode_row_map(desc, n_q, block * 2 + 1)
        mapped_store64(
            cute.local_tile(src, (64, 128), (0, 0)),
            dst,
            desc,
            mapping0,
            head,
            tid,
            threads,
        )
        mapped_store64(
            cute.local_tile(src, (64, 128), (1, 0)),
            dst,
            desc,
            mapping1,
            head,
            tid,
            threads,
        )


@cute.jit
def range_tma_store(src, dst, atom, base, head):
    tile = cute.local_tile(
        cute.domain_offset((base, 0), dst[None, None, head, 0]), (128, 128), (0, 0)
    )
    copy, _, _ = copy_utils.tma_get_copy_fn(
        atom, 0, cute.make_layout(1), src, tile, single_stage=True
    )
    copy()


@cute.jit
def indexed_load(
    src,
    dst,
    index,
    start,
    head,
    stage,
    rows: cutlass.Constexpr,
    staged: cutlass.Constexpr,
):
    atom = cute.make_copy_atom(
        cpasync.CopyG2SOp(), src.element_type, num_bits_per_copy=128
    )
    lane = cute.arch.lane_idx()
    for i in cutlass.range(rows * 128 // 256, unroll=1):
        linear = lane * 8 + i * 256
        row, col = (linear // 128, linear % 128)
        physical = index[start + row]
        gs = cute.local_tile(src[physical, None, head, 0], (8,), (col // 8,))
        if const_expr(staged):
            ds = cute.local_tile(dst[row, None, stage], (8,), (col // 8,))
        else:
            ds = cute.local_tile(dst[row, None], (8,), (col // 8,))
        cute.copy(atom, gs, ds)


@cute.jit
def indexed_store(
    src,
    dst,
    index,
    start,
    head,
    tid,
    rows: cutlass.Constexpr,
    threads: cutlass.Constexpr,
):
    atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), dst.element_type, num_bits_per_copy=128
    )
    for i in cutlass.range(rows * 128 // (threads * 8), unroll=1):
        linear = tid * 8 + i * threads * 8
        row, col = (linear // 128, linear % 128)
        physical = index[start + row]
        ss = cute.local_tile(src[row, None], (8,), (col // 8,))
        gd = cute.local_tile(dst[physical, None, head, 0], (8,), (col // 8,))
        reg = cute.make_rmem_tensor((8,), dst.element_type)
        cute.copy(atom, ss, reg)
        cute.copy(atom, reg, gd)


@cute.jit
def indexed_sub_tma(src, dst, atom, base, head, bar, rows: cutlass.Constexpr):
    tile = cute.local_tile(
        cute.domain_offset((base, 0), src[None, None, head, 0]), (rows, 128), (0, 0)
    )
    if cute.arch.lane_idx() == 0:
        cute.arch.mbarrier_expect_tx(bar, rows * 128 * 2)
    cute.arch.sync_warp()
    copy, _, _ = copy_utils.tma_get_copy_fn(
        atom, 0, cute.make_layout(1), tile, dst, single_stage=True
    )
    copy(tma_bar_ptr=bar)


@cute.jit
def indexed_segment_load(raw, t16, t4, dst, a16, a4, index, start, head, bar):
    lane = cute.arch.lane_idx()
    for seg in cutlass.range(4, unroll=1):
        logical = start + seg * 16
        base = index[logical]
        rr = lane % 16
        contig = cute.arch.vote_all_sync(index[logical + rr] == base + rr)
        d16 = cute.local_tile(dst, (16, 128), (seg, 0))
        if contig:
            indexed_sub_tma(t16, d16, a16, base, head, bar, 16)
        else:
            for sub in cutlass.range(4, unroll=1):
                logical4 = logical + sub * 4
                base4 = index[logical4]
                r4 = lane % 4
                contig4 = cute.arch.vote_all_sync(index[logical4 + r4] == base4 + r4)
                d4 = cute.local_tile(d16, (4, 128), (sub, 0))
                if contig4:
                    indexed_sub_tma(t4, d4, a4, base4, head, bar, 4)
                else:
                    indexed_load(raw, d4, index, logical4, head, Int32(0), 4, False)


@cute.jit
def store_partial(acc, output, slot, mma, tid, transpose: cutlass.Constexpr):
    coords = mma.get_slice(tid).partition_C(cute.make_identity_tensor((128,128)))
    values = cute.make_rmem_tensor(acc.shape, Float32)
    values.store(acc.load())
    for i in cutlass.range_constexpr(cute.size(values)):
        a,b=coords[i]
        row,col=(b,a) if const_expr(transpose) else (a,b)
        output[slot,row,col]=values[i]
