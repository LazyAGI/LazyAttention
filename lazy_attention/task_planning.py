# SPDX-License-Identifier: Apache-2.0
"""One-time, activation-independent coarse scheduling of GQA4 KV tasks."""
import torch
import triton
import triton.language as tl


@triton.jit
def _metadata(PC, FC, DOC, GROUP, COST, N: tl.constexpr, S: tl.constexpr,
              QUANTUM: tl.constexpr):
    b = tl.program_id(0)
    x = b * 128 + tl.arange(0, 128)
    d = tl.load(DOC + x, x < S, other=-1)
    lo = tl.min(tl.where(d >= 0, d, 2147483647), 0)
    hi = tl.max(d, 0)
    # Mixed-document KV tiles retain their exact Q lists, in distinct groups.
    group = tl.where(hi < 0, -1, tl.where(lo == hi, lo, -b - 2))
    count = tl.load(PC + b) + tl.load(FC + b)
    tl.store(GROUP + b, group)
    tl.store(COST + b, tl.cdiv(count, QUANTUM))


@triton.jit
def _priorities(GROUP, COST, PRIORITY, FIRST, N: tl.constexpr, BN: tl.constexpr):
    b = tl.program_id(0)
    x = tl.arange(0, BN)
    g = tl.load(GROUP + b)
    gs = tl.load(GROUP + x, x < N, other=-2147483647)
    match = (x < N) & (gs == g)
    costs = tl.load(COST + x, x < N, other=0)
    tl.store(PRIORITY + b, tl.max(tl.where(match, costs, 0), 0))
    tl.store(FIRST + b, tl.min(tl.where(match, x, N), 0))


@triton.jit
def _emit(PRIORITY, FIRST, ORDER, N: tl.constexpr, H: tl.constexpr,
          BN: tl.constexpr, BH: tl.constexpr, PRESERVE_SHORT: tl.constexpr):
    b = tl.program_id(0)
    x = tl.arange(0, BN)
    c, f = tl.load(PRIORITY + b), tl.load(FIRST + b)
    cs = tl.load(PRIORITY + x, x < N, other=-1)
    fs = tl.load(FIRST + x, x < N, other=N)
    before = (x < N) & ((cs > c) | ((cs == c) & (fs < f)))
    same = (x < N) & (fs == f)
    base = tl.sum(before.to(tl.int32), 0)
    local = tl.sum((same & (x < b)).to(tl.int32), 0)
    size = tl.sum(same.to(tl.int32), 0)
    h = tl.arange(0, BH)
    dst = base * H + h * size + local
    if PRESERVE_SHORT:
        long = tl.max(tl.where(x < N, cs, 0), 0) > 1
        dst = tl.where(long, dst, b * H + h)
    # Task IDs use the current scheduler's KV-major encoding, not old h*N+b.
    tl.store(ORDER + dst, b * H + h, h < H)


def _validate_inputs(plan, documents, seqlen, kv_heads, quantum):
    pc, fc = plan.mask_block_cnt, plan.full_block_cnt
    if (plan.block_size != (64, 128) or pc.shape[:2] != (1, 1)
            or not pc.is_cuda or not pc.is_contiguous() or not fc.is_contiguous()
            or fc.device != pc.device or fc.shape != pc.shape
            or documents.ndim != 1 or documents.numel() != seqlen
            or documents.dtype not in (torch.int32, torch.int64)
            or quantum < 1 or kv_heads < 1):
        raise ValueError('invalid sparse plan, logical document IDs or planning options')
    n = pc.numel()
    if not (0 < n <= 1024 and n * 128 == seqlen):
        raise ValueError('requires S divisible by128 and at most1024 KV tiles')
    return pc, fc, n


def build_task_order(plan, documents, seqlen, kv_heads, *, quantum=64,
                     preserve_short=True):
    """Return CUDA int32 task IDs. documents is in logical row order, padding=-1.

    All allocations, CPU-to-GPU document upload and three kernels belong to
    planning cost. The caller retains/reuses the immutable returned order.
    Current small-metadata implementation explicitly supports at most1024 KV tiles.
    """
    pc, fc, n = _validate_inputs(plan, documents, seqlen, kv_heads, quantum)
    docs = documents.to(device=pc.device, dtype=torch.int32).contiguous()
    group = torch.empty((n,), device=pc.device, dtype=torch.int32)
    cost, priority, first = (torch.empty_like(group) for _ in range(3))
    order = torch.empty((n * kv_heads,), device=pc.device, dtype=torch.int32)
    _metadata[(n,)](pc, fc, docs, group, cost, n, seqlen, quantum)
    _priorities[(n,)](group, cost, priority, first, n, triton.next_power_of_2(n))
    _emit[(n,)](priority, first, order, n, kv_heads,
                triton.next_power_of_2(n), triton.next_power_of_2(kv_heads), preserve_short)
    return order


def build_adaptive_task_order(plan, documents, seqlen, kv_heads, *, quantum=64):
    """Planner-owned conservative policy; None means no order argument/read.

    Require a task spanning at least three cost buckets before reordering.
    The scalar D2H synchronization is one-time preparation, charged /42.
    This is an experimental workload rule, not a shape whitelist.
    """
    pc, fc, _ = _validate_inputs(plan, documents, seqlen, kv_heads, quantum)
    counts = pc + fc
    longest = int(counts.max().item())
    if longest <= 2 * quantum:
        return None
    return build_task_order(plan, documents, seqlen, kv_heads, quantum=quantum)


from dataclasses import dataclass

@dataclass(frozen=True)
class TaskPlan:
    order: object
    reduction_indices: object = None
    reduction_tasks: object = None
    partial_count: int = 0
    @property
    def split(self):
        return self.reduction_indices is not None


def plan_tasks(plan, documents, seqlen, kv_heads, *, reorder=False,
               split=False, split_size=128):
    """One reusable metadata plan. Both switches off does no device work."""
    if not reorder and not split:
        return None
    if split_size < 1:
        raise ValueError('split_size must be positive')
    if not split:
        order=build_adaptive_task_order(plan,documents,seqlen,kv_heads) if reorder else None
        return TaskPlan(order) if order is not None else None
    import numpy as np
    pc,fc,n=_validate_inputs(plan, documents, seqlen, kv_heads, 64)
    counts=(pc+fc).flatten().cpu().numpy().astype(np.int32)
    if int(counts.max())<=split_size:
        order=build_adaptive_task_order(plan,documents,seqlen,kv_heads) if reorder else None
        return TaskPlan(order) if order is not None else None
    ids=np.arange(n*kv_heads,dtype=np.int32)
    if reorder and counts.max()>128:
        docs=documents.cpu().numpy().reshape(n,128)
        lo=np.where(docs>=0,docs,np.iinfo(np.int32).max).min(axis=1)
        hi=docs.max(axis=1)
        groups=np.where(hi<0,-1,np.where(lo==hi,lo,-np.arange(n)-2))
        buckets=(counts+63)//64
        members=[np.flatnonzero(groups==g) for g in np.unique(groups)]
        members.sort(key=lambda m:(-int(buckets[m].max()),int(m[0])))
        ids=np.concatenate([(m[:,None]*kv_heads+np.arange(kv_heads)).T.ravel() for m in members]).astype(np.int32)
    lengths=np.maximum(1,(counts[ids//kv_heads]+split_size-1)//split_size)
    starts=np.r_[0,np.cumsum(lengths)]
    task=np.repeat(ids,lengths)
    segment=np.arange(len(task))-np.repeat(starts[:-1],lengths)
    begin=segment*split_size
    task_lengths=np.empty(n*kv_heads,dtype=np.int32)
    task_lengths[ids]=lengths
    is_split=task_lengths[task]>1
    slots=np.full(len(task),-1,dtype=np.int32)
    slots[is_split]=np.arange(int(is_split.sum()),dtype=np.int32)
    records=np.column_stack((task,begin,np.minimum(begin+split_size,counts[task//kv_heads]),slots)).astype(np.int32)
    split_tasks=np.flatnonzero(task_lengths>1).astype(np.int32)
    task_rows=np.full(n*kv_heads,-1,dtype=np.int32)
    task_rows[split_tasks]=np.arange(len(split_tasks))
    indices=np.full((len(split_tasks),int(lengths.max())),-1,dtype=np.int32)
    indices[task_rows[task[is_split]],segment[is_split]]=slots[is_split]
    table=torch.from_numpy(records).to(pc.device)
    table.__leading_dim__,table.__assumed_align__=1,16
    return TaskPlan(table,torch.from_numpy(indices).to(pc.device),
                    torch.from_numpy(split_tasks).to(pc.device),int(is_split.sum()))

# Preserve explicit factorial controls for experiments and reproducibility.
_plan_tasks_manual = plan_tasks


def plan_tasks(plan, documents, seqlen, kv_heads, *, reorder=False,
               split=False, split_size=128, policy='auto', sm_count=None,
               diagnostics=None):
    """Choose a2x2 strategy once. None removes all task-plan arguments.

    Flags permit optimizations; policy='manual' retains the previous explicit
    controls. Auto reads only immutable mask/shape metadata, never activations.
    All analysis, D2H/H2D and table construction are preparation cost /42.
    """
    from dataclasses import asdict
    if policy not in ('auto','manual'):
        raise ValueError('policy must be auto or manual')
    if not reorder and not split:
        if diagnostics is not None: diagnostics.update(strategy='none',reason='switches-off')
        return None
    cpu_counts=getattr(plan,'task_counts_cpu',None)
    if hasattr(plan,'backward_plan'):
        if seqlen!=plan.seqlen: raise ValueError('prepared sequence length mismatch')
        plan=plan.backward_plan
    if policy=='manual':
        result=_plan_tasks_manual(plan,documents,seqlen,kv_heads,reorder=reorder,
                                  split=split,split_size=split_size)
        if diagnostics is not None:
            diagnostics.update(strategy='none' if result is None else
                ('both' if reorder else 'split') if result.split else 'reorder',reason='manual')
        return result
    import numpy as np
    from .task_policy import TaskFeatures,select_task_strategy
    pc,fc,n=_validate_inputs(plan,documents,seqlen,kv_heads,64)
    # One transfer of the two count arrays, no activation data or dense mask.
    if cpu_counts is None:
        counters=torch.stack((pc.flatten(),fc.flatten())).cpu().numpy()
        counts_source='gpu-copy'
    else:
        counters=np.stack(cpu_counts).astype(np.int32,copy=False)
        if counters.shape!=(2,n) or (counters<0).any():
            raise ValueError('invalid immutable CPU task counts')
        counts_source='prepared-cpu'
    counts=(counters[0]+counters[1]).astype(np.int32)
    maximum=int(counts.max()); active=counts[counts>0]
    ids=np.arange(n*kv_heads,dtype=np.int32)
    base=ids.copy(); group_count=None
    if reorder:
        docs=documents.cpu().numpy().reshape(n,128)
        lo=np.where(docs>=0,docs,np.iinfo(np.int32).max).min(axis=1);hi=docs.max(axis=1)
        groups=np.where(hi<0,-1,np.where(lo==hi,lo,-np.arange(n)-2))
        buckets=(counts+63)//64
        members=[np.flatnonzero(groups==g) for g in np.unique(groups)]
        members.sort(key=lambda m:(-int(buckets[m].max()),int(m[0])))
        ids=np.concatenate([(m[:,None]*kv_heads+np.arange(kv_heads)).T.ravel() for m in members]).astype(np.int32)
        group_count=len(members)
    sm_count=int(sm_count or torch.cuda.get_device_properties(pc.device).multi_processor_count)
    total=int(counts.sum())
    features=TaskFeatures(int(seqlen),int(kv_heads),sm_count,n*kv_heads,
        len(active)*kv_heads,maximum,float(active.mean()) if len(active) else 0.,
        float(np.percentile(counts,95)),float(counts.std()/counts.mean()) if total else 0.,
        float(counters[1].sum()/total) if total else 0.,
        float((active>split_size).mean()) if len(active) else 0.,total*kv_heads,
        group_count,not np.array_equal(ids,base))
    choice=select_task_strategy(features,reorder=reorder,split=split,split_size=split_size)
    if diagnostics is not None:
        diagnostics.update(strategy=choice.strategy,reason=choice.reason,
                           features=asdict(features),split_size=choice.split_size,counts_source=counts_source)
    if not choice.reorder and not choice.split:
        return None
    if not choice.reorder: ids=base
    if not choice.split:
        table=torch.from_numpy(ids).to(pc.device)
        table.__leading_dim__,table.__assumed_align__=0,16
        return TaskPlan(table)
    chunk=choice.split_size
    lengths=np.maximum(1,(counts[ids//kv_heads]+chunk-1)//chunk)
    starts=np.r_[0,np.cumsum(lengths)]
    task=np.repeat(ids,lengths)
    segment=np.arange(len(task))-np.repeat(starts[:-1],lengths)
    begin=segment*chunk
    task_lengths=np.empty(n*kv_heads,dtype=np.int32);task_lengths[ids]=lengths
    is_split=task_lengths[task]>1
    slots=np.full(len(task),-1,dtype=np.int32);slots[is_split]=np.arange(int(is_split.sum()),dtype=np.int32)
    records=np.column_stack((task,begin,np.minimum(begin+chunk,counts[task//kv_heads]),slots)).astype(np.int32)
    split_tasks=np.flatnonzero(task_lengths>1).astype(np.int32)
    task_rows=np.full(n*kv_heads,-1,dtype=np.int32);task_rows[split_tasks]=np.arange(len(split_tasks))
    indices=np.full((len(split_tasks),int(lengths.max())),-1,dtype=np.int32)
    indices[task_rows[task[is_split]],segment[is_split]]=slots[is_split]
    table=torch.from_numpy(records).to(pc.device)
    table.__leading_dim__,table.__assumed_align__=1,16
    return TaskPlan(table,torch.from_numpy(indices).to(pc.device),
                    torch.from_numpy(split_tasks).to(pc.device),int(is_split.sum()))
