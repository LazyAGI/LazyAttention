# SPDX-License-Identifier: Apache-2.0
import triton
import triton.language as tl

@triton.jit
def _reduce(K,V,INDEX,TASKS,DESC,DK,DV,H:tl.constexpr,NQ:tl.constexpr,W:tl.constexpr,
            KS:tl.constexpr,KH:tl.constexpr,VS:tl.constexpr,VH:tl.constexpr,
            BLOCK:tl.constexpr):
    task_row=tl.program_id(0); task=tl.load(TASKS+task_row); block=tl.program_id(1)
    x=block*BLOCK+tl.arange(0,BLOCK)
    row=x//128; col=x%128
    logical=(task//H)*128+row; q=logical//64; r=logical%64
    header=tl.load(DESC+q)
    ptr=tl.load(DESC+NQ+q)
    base0=tl.load(DESC+ptr,header<0,other=0)
    base1=tl.load(DESC+ptr+1,header<0,other=0)
    gathered=tl.load(DESC+ptr+r,header<=-64,other=0)
    physical=tl.where(header>=0,header+r,
        tl.where(header>-64,tl.where(r < -header,base0+r,base1+r+header),gathered))
    k=tl.full((BLOCK,),0,tl.float32);v=tl.full((BLOCK,),0,tl.float32)
    for i in tl.static_range(W):
        slot=tl.load(INDEX+task_row*W+i)
        k+=tl.load(K+slot*16384+x,slot>=0,other=0)
        v+=tl.load(V+slot*16384+x,slot>=0,other=0)
    tl.store(DK+physical*KS+(task%H)*KH+col,k)
    tl.store(DV+physical*VS+(task%H)*VH+col,v)

def reduce_partials(k,v,indices,tasks,desc,dk,dv):
    h=dk.shape[2]
    _reduce[(indices.shape[0],16)](k,v,indices,tasks,desc,dk,dv,h,dk.shape[1]//64,
        indices.shape[1],dk.stride(1),dk.stride(2),dv.stride(1),dv.stride(2),1024)
