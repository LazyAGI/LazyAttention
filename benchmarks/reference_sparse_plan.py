# SPDX-License-Identifier: Apache-2.0
"""Frozen row-wise sparse-plan constructor from 423d0e7 for A/B only."""
def sparse_plan(mask, m, n, backward, device):
    import torch
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    qblocks, kblocks = mask.shape[0] // m, mask.shape[1] // n
    groups = kblocks if backward else qblocks
    full, partial = [[] for _ in range(groups)], [[] for _ in range(groups)]
    for q in range(qblocks):
        for k in range(kblocks):
            tile = mask[q*m:(q+1)*m, k*n:(k+1)*n]
            group, value = (k, q) if backward else (q, k)
            if tile.all():
                full[group].append(value)
            elif tile.any():
                partial[group].append(value)
    def tensors(rows):
        count = torch.tensor([len(x) for x in rows], dtype=torch.int32, device=device).reshape(1,1,groups)
        width = max(map(len, rows), default=0)
        indices = torch.zeros((1,1,groups,width), dtype=torch.int32, device=device)
        for g, row in enumerate(rows):
            if row: indices[0,0,g,:len(row)] = torch.tensor(row, dtype=torch.int32, device=device)
        return count, indices
    pc, pi = tensors(partial); fc, fi = tensors(full)
    return BlockSparseTensorsTorch(mask_block_cnt=pc, mask_block_idx=pi,
        full_block_cnt=fc, full_block_idx=fi, block_size=(m,n))
