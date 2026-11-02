"""Compact128x128 bit-mask specialization for the FA4 kernels."""
import cutlass
import cutlass.cute as cute
from cutlass import const_expr
from flash_attn.cute import utils
from flash_attn.cute.mask import AttentionMask, AuxData
from quack import layout_utils
from typing import Callable, Optional


@cute.jit
def compact_mask(batch, head, m_idx, n_idx, seqlen_info, aux_tensors):
    q, k = m_idx[0], n_idx[0]
    tile = cutlass.Int32(aux_tensors[0][q // 128, k // 128])
    word = aux_tensors[1][tile * 512 + (q % 128) * 4 + (k % 128) // 32]
    return utils.scalar_to_ssa(((word >> (k % 32)) & cutlass.Uint32(1)) != 0, cutlass.Boolean)


class CompactAttentionMask(AttentionMask):
    """Hoist the compact tile lookup out of the accumulator element loop."""

    @cute.jit
    def apply_mask(
            self,
            acc_S: cute.Tensor,
            batch_idx: cutlass.Int32,
            head_idx: cutlass.Int32,
            m_block: cutlass.Int32,
            n_block: cutlass.Int32,
            thr_mma: cute.TiledMma,
            mask_seqlen: cutlass.Constexpr[bool],
            mask_causal: cutlass.Constexpr[bool],
            mask_local: cutlass.Constexpr[bool] = False,
            mask_mod: cutlass.Constexpr[Optional[Callable]] = None,
            aux_data: AuxData = AuxData(),
            fastdiv_mods=(None, None),
    ) -> None:
        if const_expr(mask_mod is None):
            AttentionMask.apply_mask(
                self,
                acc_S,
                batch_idx,
                head_idx,
                m_block,
                n_block,
                thr_mma,
                mask_seqlen,
                mask_causal,
                mask_local,
                mask_mod,
                aux_data,
                fastdiv_mods,
            )
        else:
            assert not mask_causal and not mask_local
            assert self.tile_m in (64, 128) and self.tile_n in (64, 128)
            values = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.swap_AB)
            shape = (self.tile_m, self.tile_n)
            coords = layout_utils.reshape_acc_to_mn(
                thr_mma.partition_C(cute.make_identity_tensor(shape if not self.swap_AB else shape[::-1])),
                transpose=self.swap_AB,
            )
            row = const_expr(1 if self.swap_AB else 0)
            col = const_expr(0 if self.swap_AB else 1)
            lookup, bits = aux_data.tensors[0], aux_data.tensors[1]
            kb = cutlass.Int32(0)
            if n_block >= 0:
                kb = n_block
            block_q = (m_block * self.tile_m // self.qhead_per_kvhead_packgqa) // 128
            base = cutlass.Int32(lookup[block_q, (kb * self.tile_n) // 128]) * 512
            for r in cutlass.range_constexpr(cute.size(coords.shape[0])):
                q = (m_block * self.tile_m + coords[r, 0][row]) // self.qhead_per_kvhead_packgqa
                if const_expr(128 % self.qhead_per_kvhead_packgqa != 0):
                    base = cutlass.Int32(lookup[q // 128, (kb * self.tile_n) // 128]) * 512
                row_base = base + (q % 128) * 4
                for c in cutlass.range_constexpr(cute.size(coords.shape[1])):
                    k = kb * self.tile_n + coords[0, c][col]
                    word = bits[row_base + (k % 128) // 32]
                    allowed = ((word >> (k % 32)) & cutlass.Uint32(1)) != 0
                    if const_expr(mask_seqlen):
                        allowed = allowed & (q < self.seqlen_q) & (k < self.seqlen_k)
                    values[r, c] = values[r, c] if allowed else -cutlass.Float32.inf
