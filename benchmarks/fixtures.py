"""Public synthetic fixtures; no training framework or private workload needed."""
def make_fixture(seqlen=512, kv_heads=2, mapping='mixed', mask_kind='causal', seed=0, device='cuda'):
    import torch
    if seqlen < 128 or seqlen % 128:
        raise ValueError('seqlen must be positive and divisible by 128')
    generator = torch.Generator().manual_seed(seed)
    index = torch.arange(seqlen, dtype=torch.int32)
    if mapping == 'two':
        index = (index + 17) % seqlen
    elif mapping == 'index':
        index = torch.randperm(seqlen, generator=generator).to(torch.int32)
    elif mapping == 'mixed':
        # Keep one continuous block, rotate one block (two ranges), shuffle
        # another block if available. All maps remain bijective.
        index[64:128] = index[64:128].roll(17)
        if seqlen >= 256:
            index[128:192] = index[128:192][torch.randperm(64, generator=generator)]
    elif mapping != 'continuous':
        raise ValueError(mapping)
    if mask_kind == 'causal':
        mask = torch.ones(seqlen, seqlen, dtype=torch.bool).tril()
    elif mask_kind == 'full':
        mask = torch.ones(seqlen, seqlen, dtype=torch.bool)
    elif mask_kind == 'sparse':
        mask = torch.rand(seqlen, seqlen, generator=generator) > .7
        mask[128:, :128] = False
        mask[:128, -128:] = True
        mask.fill_diagonal_(True)
    elif mask_kind == 'empty':
        mask = torch.zeros(seqlen, seqlen, dtype=torch.bool)
    else:
        raise ValueError(mask_kind)
    def tensor(heads):
        return torch.randn(1,seqlen,heads,128,generator=generator).to(device=device,dtype=torch.bfloat16)
    return dict(q=tensor(kv_heads*4), k=tensor(kv_heads), v=tensor(kv_heads),
                dout=tensor(kv_heads*4), index=index, logical_mask=mask)


def reference(fixture):
    """FP32 math reference in physical order, including fully masked rows."""
    import torch
    f = fixture
    q,k,v = [f[name].detach().float().requires_grad_() for name in ('q','k','v')]
    inverse = torch.argsort(f['index'].long()).cpu()
    mask = f['logical_mask'].cpu()[inverse][:,inverse].to(q.device)
    qq = q.transpose(1,2)
    kk = k.repeat_interleave(4,dim=2).transpose(1,2)
    vv = v.repeat_interleave(4,dim=2).transpose(1,2)
    scores = (qq @ kk.transpose(-1,-2)) * (128 ** -.5)
    scores = scores.masked_fill(~mask, float('-inf'))
    lse = scores.logsumexp(-1)
    # Avoid softmax(all -inf) NaNs and their backward contamination.
    nonempty = mask.any(-1)
    scores = torch.where(nonempty[None,None,:,None], scores, torch.zeros_like(scores))
    probs = scores.softmax(-1) * nonempty[None,None,:,None]
    out = (probs @ vv).transpose(1,2)
    grads = torch.autograd.grad(out, (q,k,v), f['dout'].float())
    return out.detach(), lse.detach(), grads


def error(actual, expected):
    import torch
    a,b = actual.float(), expected.float()
    return dict(max_abs=(a-b).abs().max().item(),
                rel_l2=((a-b).norm()/b.norm().clamp_min(1e-12)).item())
