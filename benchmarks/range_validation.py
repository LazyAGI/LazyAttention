# SPDX-License-Identifier: Apache-2.0
"""Exact range/compact-mask equivalence in packed bits, bounded host memory."""
import numpy as np


def validate_capture_ranges(captured, arrays, size):
    lookup, words=(t.numpy() for t in captured['physical_aux'])
    plan=captured['forward_plan']
    counts=plan['full_block_cnt'].numpy().reshape(-1)
    indices=plan['full_block_idx'].numpy().reshape(len(counts),-1)
    packed=words.astype('<u4',copy=False).view(np.uint8).reshape(-1,128,16)
    qr,kr,kinds=arrays
    for qb in range(size//128):
        start=qb*128;end=start+128
        expected=packed[lookup[qb].astype(np.int64)].transpose(1,0,2).reshape(128,size//8).copy()
        for kb in indices[qb,:counts[qb]]: expected[:,int(kb)*16:(int(kb)+1)*16]=255
        got=np.zeros_like(expected)
        for (q0,q1),(k0,k1),kind in zip(qr,kr,kinds):
            lo,hi=max(start,int(q0)),min(end,int(q1))
            if lo>=hi: continue
            if not (0<=q0<q1<=size and 0<=k0<k1<=size and kind in (0,1,2)): raise ValueError('invalid range')
            for row in range(lo,hi):
                left,right=int(k0),int(k1)
                if kind==1: right=min(right,int(k0)+row-int(q0)-int(q1-q0)+int(k1-k0)+1)
                if kind==2: left=max(left,int(k0)+row-int(q0))
                if left>=right: continue
                first,last=left//8,(right-1)//8
                line=got[row-start]
                if first==last:
                    value=((1<<(right-left))-1)<<(left%8)
                    if int(line[first])&value: raise ValueError('overlap')
                    line[first] |= value
                else:
                    head=(255<<(left%8))&255;tail=(1<<((right-1)%8+1))-1
                    if (int(line[first])&head) or (int(line[last])&tail) or line[first+1:last].any(): raise ValueError('overlap')
                    line[first] |= head;line[last] |= tail;line[first+1:last]=255
        if not np.array_equal(expected,got): raise ValueError(('exact packed mask mismatch',qb))
