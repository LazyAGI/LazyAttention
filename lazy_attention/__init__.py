"""Experimental self-attention operators; GPU dependencies load on first use."""
from .planning import PreparedAttention, prepare_attention, prepare_rows


def forward(*args, **kwargs):
    from .operator import forward as run
    return run(*args, **kwargs)


def backward(*args, **kwargs):
    from .operator import backward as run
    return run(*args, **kwargs)


__all__ = ['PreparedAttention', 'prepare_attention', 'prepare_rows', 'forward', 'backward']
