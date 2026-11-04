# SPDX-License-Identifier: Apache-2.0
"""Prepare traversal and a reusable task plan, never in the GPU iteration loop."""
from dataclasses import replace


def prepare_execution(prepared, *, documents=None, kv_heads=None, reorder=True,
                      split=False, split_size=128, policy='auto', head_major=False,
                      reverse=True, interleave=0):
    """Configure once; Q-major/reverse/reorder on, splitting/interleave off.

    documents are logical-row document IDs. Without them each KV128 tile is an
    independent scheduling group; supply IDs to retain whole-document grouping.
    Supply kv_heads here to finish planning before CUDA Graph capture. Otherwise
    the first eager backward caches its plan for that head count. Mask, mapping,
    document IDs and cached plans must remain immutable across reuse.
    """
    import torch
    from .q_traversal import prepare_q_traversal
    if policy not in ('auto','manual') or split_size<1:
        raise ValueError('invalid planning policy or split_size')
    if documents is not None:
        if documents.shape!=(prepared.seqlen,) or documents.dtype not in (torch.int32,torch.int64):
            raise ValueError('documents must be integer logical-row IDs [S]')
        docs=documents.detach().to(device='cpu',dtype=torch.int32).clone()
    else:
        docs=prepared.documents_cpu
    configured=prepare_q_traversal(prepared,reverse=reverse,interleave=interleave)
    configured=replace(configured,head_major=bool(head_major),documents_cpu=docs,
        execution_options=(bool(reorder),bool(split),int(split_size),policy),execution_cache={})
    if kv_heads is not None:
        execution_task_plan(configured,kv_heads)
    return configured


def execution_task_plan(prepared, kv_heads):
    """Return the cached plan, including cached None for a no-op planner."""
    import torch
    if prepared.execution_options is None:
        return None
    if not isinstance(kv_heads,int) or kv_heads<1:
        raise ValueError('kv_heads must be a positive integer')
    if kv_heads in prepared.execution_cache:
        return prepared.execution_cache[kv_heads]
    reorder,split,split_size,policy=prepared.execution_options
    if not reorder and not split:
        prepared.execution_cache[kv_heads]=None
        return None
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError('prepare_execution(..., kv_heads=Hkv) before CUDA Graph capture')
    from .task_planning import plan_tasks
    documents=prepared.documents_cpu
    if documents is None:
        documents=torch.arange(prepared.seqlen,dtype=torch.int32)//128
    plan=plan_tasks(prepared,documents,prepared.seqlen,kv_heads,reorder=reorder,
                    split=split,split_size=split_size,policy=policy)
    prepared.execution_cache[kv_heads]=plan
    return plan
