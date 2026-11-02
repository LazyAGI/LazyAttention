# Derived from FlashAttention CuTe; see licenses/flash-attention-BSD-3-Clause.txt.
# Modified: persistent GQA4 scheduling and exclusive range/index row mapping.
"""One 16/4-row traversal for affine ranges and indexed fallback; paired data copies."""
import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.nvgpu import cpasync
from quack import copy_utils

@cute.jit
def pair_row(desc, payload, row, base0, base1, split):
    physical = base0 + row
    if split > 0:
        if row >= split:
            physical = base1 + row - split
    else:
        physical = desc[payload + row]
    return physical

@cute.jit
def pair_tma(t0,t1,d0,d1,a0,a1,base,head,bar0,bar1,rows:cutlass.Constexpr):
    if cute.arch.lane_idx()==0:
        cute.arch.mbarrier_expect_tx(bar0,rows*128*2)
        cute.arch.mbarrier_expect_tx(bar1,rows*128*2)
    cute.arch.sync_warp()
    g0=cute.local_tile(cute.domain_offset((base,0),t0[None,None,head,0]),(rows,128),(0,0))
    g1=cute.local_tile(cute.domain_offset((base,0),t1[None,None,head,0]),(rows,128),(0,0))
    cp0,_,_=copy_utils.tma_get_copy_fn(a0,0,cute.make_layout(1),g0,d0,single_stage=True)
    cp1,_,_=copy_utils.tma_get_copy_fn(a1,0,cute.make_layout(1),g1,d1,single_stage=True)
    cp0(tma_bar_ptr=bar0)
    cp1(tma_bar_ptr=bar1)

@cute.jit
def pair_gather(raw0,raw1,d0,d1,desc,payload,start,base0,base1,split,head,rows:cutlass.Constexpr):
    atom=cute.make_copy_atom(cpasync.CopyG2SOp(),raw0.element_type,num_bits_per_copy=128)
    lane=cute.arch.lane_idx()
    for i in cutlass.range(rows*128//256,unroll=1):
        linear=lane*8+i*256
        row,col=linear//128,linear%128
        physical=pair_row(desc,payload,start+row,base0,base1,split)
        g0=cute.local_tile(raw0[physical,None,head,0],(8,),(col//8,))
        g1=cute.local_tile(raw1[physical,None,head,0],(8,),(col//8,))
        s0=cute.local_tile(d0[row,None],(8,),(col//8,))
        s1=cute.local_tile(d1[row,None],(8,),(col//8,))
        cute.copy(atom,g0,s0)
        cute.copy(atom,g1,s1)

@cute.jit
def pair_segments(raw0,raw1,t160,t161,t40,t41,dst0,dst1,a160,a161,a40,a41,desc,payload,base0,base1,split,head,bar0,bar1):
    lane=cute.arch.lane_idx()
    for seg in cutlass.range(4,unroll=1):
        start=seg*16
        base=Int32(0)
        contiguous=False
        if split>0:
            base=pair_row(desc,payload,start,base0,base1,split)
            contiguous=(start>=split) or (start+16<=split)
        else:
            base=desc[payload+start]
            rr=lane%16
            contiguous=cute.arch.vote_all_sync(desc[payload+start+rr]==base+rr)
        d160=cute.local_tile(dst0,(16,128),(seg,0))
        d161=cute.local_tile(dst1,(16,128),(seg,0))
        if contiguous:
            pair_tma(t160,t161,d160,d161,a160,a161,base,head,bar0,bar1,16)
        else:
            for sub in cutlass.range(4,unroll=1):
                start4=start+sub*4
                base4=Int32(0)
                contiguous4=False
                if split>0:
                    base4=pair_row(desc,payload,start4,base0,base1,split)
                    contiguous4=(start4>=split) or (start4+4<=split)
                else:
                    base4=desc[payload+start4]
                    rr=lane%4
                    contiguous4=cute.arch.vote_all_sync(desc[payload+start4+rr]==base4+rr)
                d40=cute.local_tile(d160,(4,128),(sub,0))
                d41=cute.local_tile(d161,(4,128),(sub,0))
                if contiguous4:
                    pair_tma(t40,t41,d40,d41,a40,a41,base4,head,bar0,bar1,4)
                else:
                    pair_gather(raw0,raw1,d40,d41,desc,payload,start4,base0,base1,split,head,4)
