"""Run on CUDA: exact sparse-plan ABI against an independent tile reference."""
import itertools
import numpy as np
import torch
from lazy_attention.planning import _sparse_plan


def expected(mask, m, n, backward):
    groups = mask.shape[1] // n if backward else mask.shape[0] // m
    full, partial = [[] for _ in range(groups)], [[] for _ in range(groups)]
    for q in range(mask.shape[0] // m):
        for k in range(mask.shape[1] // n):
            tile = mask[q*m:(q+1)*m, k*n:(k+1)*n]
            group, index = (k, q) if backward else (q, k)
            if tile.all(): full[group].append(index)
            elif tile.any(): partial[group].append(index)
    def pack(rows):
        count = np.array([len(row) for row in rows], dtype=np.int32).reshape(1,1,groups)
        indices = np.zeros((1,1,groups,max(map(len, rows))), dtype=np.int32)
        for g,row in enumerate(rows): indices[0,0,g,:len(row)] = row
        return count, indices
    return (*pack(partial), *pack(full))


def main():
    comparisons = 0
    for size, kind in itertools.product((128,256,384,512,4096), ('full','empty','causal','partial','strided')):
        mask = (np.ones((size,size), bool) if kind == 'full' else
                np.zeros((size,size), bool) if kind == 'empty' else
                np.tri(size, dtype=bool) if kind == 'causal' else
                np.random.default_rng(size).random((size,size)) < .3)
        if kind == 'strided': mask = mask.T
        for backward in (False, True):
            m,n = (64,128) if backward else (128,128)
            got = _sparse_plan(mask,m,n,backward,'cuda')
            for name, ref in zip(('mask_block_cnt','mask_block_idx','full_block_cnt','full_block_idx'), expected(mask,m,n,backward)):
                tensor = getattr(got,name)
                assert tensor.dtype == torch.int32 and tuple(tensor.shape) == ref.shape
                assert tensor.stride(-1) == 1, (size,kind,backward,name,tensor.stride())
                assert np.array_equal(tensor.cpu().numpy(),ref), (size,kind,backward,name)
            comparisons += 1
    print(f'{comparisons} exact forward/backward sparse-plan ABI comparisons passed')


if __name__ == '__main__': main()
