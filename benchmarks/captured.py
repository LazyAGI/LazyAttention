# SPDX-License-Identifier: Apache-2.0
"""Read existing mask/plan captures, preserving physical layouts; no training import."""
import hashlib
import json
from pathlib import Path
import torch
from lazy_attention.planning import PreparedAttention, prepare_rows, encode_rows
from flash_attn.cute.block_sparsity import BlockSparseTensorsTorch


def load(root, rank, batch):
    directory = Path(root) / f'rank-{rank}'
    files = {name:directory / f'batch-{batch:03d}-{name}.pt' for name in ('mask','index')}
    m, ix = (torch.load(files[name], map_location='cpu', weights_only=True) for name in ('mask','index'))
    events = [json.loads(x) for x in (directory/'events.jsonl').read_text().splitlines()]
    layout = next(e['layouts'] for e in events if e['event']=='backward' and e['batch']==batch)
    torch.manual_seed(2026 + batch*8 + rank)
    def tensor(name):
        spec = layout[name]
        shape, stride, offset = spec['shape'], spec['stride'], spec['storage_offset']
        size = offset + sum((s-1)*d for s,d in zip(shape,stride)) + 1
        out = torch.empty(size,device='cuda',dtype=torch.bfloat16).as_strided(shape,stride,offset)
        out.normal_(std=.5)
        return out
    f = {name:tensor(name) for name in ('q','k','v','dout')}
    f['dout'][:,m['metadata']['mot_dup_boundary'].bool().cuda()] = 0
    f['index'] = ix['index'].to(dtype=torch.int32).cpu()
    f['documents'] = m['metadata']['mot_document_ids'].cpu()[f['index'].long()].to(torch.int32).contiguous()
    def aux(name, source):
        result = tuple(t.cuda() for t in source[name])
        result[0].__leading_dim__ = 1
        result[1].__leading_dim__ = 0
        result[1].__assumed_align__ = 16
        return result
    def sparse(data, size):
        return BlockSparseTensorsTorch(**{name:t.cuda() for name,t in data.items()},block_size=size)
    desc = prepare_rows(f['index'])[0].cuda()
    f['prepared'] = PreparedAttention(f['q'].shape[1],desc,aux('logical_aux',ix),aux('physical_aux',m),
        sparse(m['forward_plan'],(128,128)),sparse(ix['logical_backward_plan'],(64,128)),
        encode_rows(f['index'].tolist())[1],task_counts_cpu=tuple(
            ix['logical_backward_plan'][name].numpy().reshape(-1) for name in ('mask_block_cnt','full_block_cnt')))
    f['capture_sha256'] = {name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
    return f
