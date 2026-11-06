# SPDX-License-Identifier: Apache-2.0
"""One-time mask/row-map preparation, outside the measured GPU backward chain."""
from dataclasses import dataclass, field
from typing import Any


def encode_rows(index):
    """Encode a logical-to-physical permutation; range blocks own no indices."""
    values = list(index)
    size = len(values)
    if not size or size % 64:
        raise ValueError('row count must be positive and divisible by64')
    if any(type(v) is not int for v in values) or sorted(values) != list(range(size)):
        raise ValueError('index must be an integer permutation of[0, row_count)')
    blocks = size // 64
    headers, pointers, payload, counts = [], [], [], [0, 0, 0]
    for start in range(0, size, 64):
        row = values[start:start + 64]
        cuts = [i for i in range(1, 64) if row[i] != row[i - 1] + 1]
        if not cuts:
            headers.append(row[0]); pointers.append(0); counts[0] += 1
        elif len(cuts) == 1:
            split = cuts[0]
            headers.append(-split); pointers.append(2 * blocks + len(payload))
            payload.extend([row[0], row[split]]); counts[1] += 1
        else:
            headers.append(-64); pointers.append(2 * blocks + len(payload))
            payload.extend(row); counts[2] += 1
    return headers + pointers + payload, dict(mode_counts=counts,
        cache_bytes=4 * (2 * blocks + len(payload)), index_bytes=4 * size,
        retained_range_index_entries=0)


def prepare_rows(index):
    """Return(int32 CUDA/CPU descriptor, info); do not retain the input index."""
    import torch
    if index.ndim != 1 or index.dtype not in (torch.int32, torch.int64):
        raise ValueError('index must be a1D int32/int64 tensor')
    import numpy as np
    values = index.detach().cpu().numpy()
    size = values.size
    if not size or size % 64:
        raise ValueError('row count must be positive and divisible by64')
    if not np.array_equal(np.sort(values), np.arange(size)):
        raise ValueError('index must be an integer permutation of[0, row_count)')
    rows = values.reshape(-1, 64)
    blocks = rows.shape[0]
    cuts = rows[:, 1:] != rows[:, :-1] + 1
    count = cuts.sum(axis=1)
    split = cuts.argmax(axis=1) + 1
    lengths = np.where(count == 0, 0, np.where(count == 1, 2, 64))
    offsets = np.cumsum(lengths) - lengths
    packed = np.empty(2 * blocks + int(lengths.sum()), dtype=np.int32)
    packed[:blocks] = np.where(count == 0, rows[:, 0], np.where(count == 1, -split, -64))
    packed[blocks:2 * blocks] = np.where(count == 0, 0, 2 * blocks + offsets)
    one = np.flatnonzero(count == 1)
    packed[2 * blocks + offsets[one]] = rows[one, 0]
    packed[2 * blocks + offsets[one] + 1] = rows[one, split[one]]
    many = np.flatnonzero(count > 1)
    packed[2 * blocks + offsets[many, None] + np.arange(64)] = rows[many]
    info = dict(mode_counts=[int((count == 0).sum()), int((count == 1).sum()), int((count > 1).sum())],
                cache_bytes=int(packed.nbytes), index_bytes=4 * size, retained_range_index_entries=0)
    return torch.from_numpy(packed).to(device=index.device), info


@dataclass(frozen=True)
class PreparedAttention:
    """Treat tensors as immutable; range strategy retains no full row index/mask."""
    seqlen: int
    descriptor: Any
    logical_aux: tuple
    physical_aux: tuple
    forward_plan: Any
    backward_plan: Any
    info: dict
    index: Any = None
    task_counts_cpu: Any = None
    task_indices_cpu: Any = field(default=None, repr=False, compare=False)
    q_reverse: bool = False
    q_interleave: int = 0
    head_major: bool = False
    base_backward_plan: Any = field(default=None, repr=False, compare=False)
    execution_options: Any = None
    documents_cpu: Any = field(default=None, repr=False, compare=False)
    execution_cache: dict = field(default_factory=dict, repr=False, compare=False)


def _aux(mask, device):
    import numpy as np
    import torch
    size = mask.shape[0] // 128
    lookup = np.zeros((size, size), dtype=np.int32)
    words = [np.zeros(512, dtype=np.uint32)]
    for q in range(size):
        for k in range(size):
            tile = mask[q * 128:(q + 1) * 128, k * 128:(k + 1) * 128]
            if tile.any() and not tile.all():
                lookup[q, k] = len(words)
                words.append(np.packbits(tile.reshape(-1), bitorder='little').view('<u4'))
    a = torch.tensor(lookup, device=device)
    b = torch.tensor(np.concatenate(words), dtype=torch.uint32, device=device)
    a.__leading_dim__, b.__leading_dim__ = 1, 0
    b.__assumed_align__ = 16
    return a, b


def _sparse_plan(mask, m, n, backward, device, counts_out=None, indices_out=None):
    import numpy as np
    import torch
    from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch
    qblocks, kblocks = mask.shape[0] // m, mask.shape[1] // n
    tiles = mask.reshape(qblocks, m, kblocks, n)
    if mask.strides[0] < mask.strides[1]:
        # Separate reductions avoid NumPy's slow multi-axis strided traversal.
        # No full-mask contiguous copy; the intermediate is tile-reduced.
        present = tiles.any(axis=3).any(axis=1)
        full = tiles.all(axis=3).all(axis=1)
    else:
        present = tiles.any(axis=(1, 3))
        full = tiles.all(axis=(1, 3))
    partial = present & ~full
    if backward:
        partial, full = partial.T, full.T
    def tensors(flags):
        groups = flags.shape[0]
        counts = flags.sum(axis=1, dtype=np.int32)
        width = int(counts.max(initial=0))
        if width:
            indices = np.zeros((1, 1, groups, width), dtype=np.int32)
            # CPU-only packing; exactly two H2D transfers per nonempty list.
            for group in range(groups):
                indices[0, 0, group, :counts[group]] = np.flatnonzero(flags[group])
            gpu_indices = torch.as_tensor(indices, device=device)
        else:
            # No CPU list packing for an empty bucket. Keep CuTe's unit stride.
            gpu_indices = torch.empty((1, 1, groups, 0), dtype=torch.int32, device=device)
        if indices_out is not None:
            indices_out.append(indices if width else np.empty((1,1,groups,0),dtype=np.int32))
        return (torch.as_tensor(counts.reshape(1, 1, groups), device=device), gpu_indices, counts)
    pc, pi, pc_cpu = tensors(partial); fc, fi, fc_cpu = tensors(full)
    if counts_out is not None:
        pc_cpu.flags.writeable=False;fc_cpu.flags.writeable=False
        counts_out.extend((pc_cpu,fc_cpu))
    return BlockSparseTensorsTorch(mask_block_cnt=pc, mask_block_idx=pi,
        full_block_cnt=fc, full_block_idx=fi, block_size=(m,n))


def prepare_attention(logical_mask, index=None, *, device='cuda', strategy='range',
                      documents=None, kv_heads=None, reorder=True, split=False,
                      head_major=False, reverse=True, interleave=0):
    """Prepare a boolean[S,S] logical mask and optional logical→physical rows.

    q/k/v/out/dout use physical row order. The forward mask is transformed to
    that order; backward traverses the logical mask and decodes the descriptor.
    Preparation copies metadata to CPU once and is deliberately not graph-safe.
    Defaults reorder tasks, disable splitting, and traverse Q-major in complete
    reverse list order without interleave. documents are logical-row IDs;
    supplying kv_heads also finishes cached task planning before Graph capture.
    """
    import torch
    if logical_mask.ndim != 2 or logical_mask.dtype != torch.bool:
        raise ValueError('logical_mask must be a square boolean matrix')
    size = logical_mask.shape[0]
    if size <= 0 or size % 128 or logical_mask.shape[1] != size:
        raise ValueError('self-attention sequence length must be positive and divisible by128')
    if index is None: index = torch.arange(size, dtype=torch.int32)
    if index.ndim != 1 or index.numel() != size:
        raise ValueError('index length must equal the sequence length')
    if index.dtype not in (torch.int32, torch.int64):
        raise ValueError('index must contain integers')
    packed, info = encode_rows(index.detach().cpu().tolist())
    if strategy not in ('range', 'index'):
        raise ValueError('strategy must be range or index')
    logical = logical_mask.detach().cpu().numpy()
    inverse = torch.argsort(index.detach().cpu()).numpy()
    physical = logical[inverse][:, inverse]
    desc = torch.tensor(packed, device=device, dtype=torch.int32) if strategy == 'range' else None
    row_index = index.to(device=device, dtype=torch.int32).contiguous() if strategy == 'index' else None
    info = {**info, "strategy": strategy, "row_mapping_bytes": 4 * (len(packed) if strategy == "range" else size)}
    counts_cpu=[]; indices_cpu=[]
    prepared = PreparedAttention(size, desc, _aux(logical, device), _aux(physical, device),
        _sparse_plan(physical,128,128,False,device),
        _sparse_plan(logical,64,128,True,device,counts_cpu,indices_cpu), info, row_index, tuple(counts_cpu),tuple(indices_cpu))

    from .execution import prepare_execution
    return prepare_execution(prepared, documents=documents, kv_heads=kv_heads,
        reorder=reorder, split=split, head_major=head_major, reverse=reverse,
        interleave=interleave)
