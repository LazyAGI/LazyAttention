# SPDX-License-Identifier: Apache-2.0
"""Dynamic S and exclusive descriptor lengths must share dQ conversion code."""
import json
import time
import torch
from lazy_attention.planning import prepare_rows
from lazy_attention.csrc.fa4.post_launcher import postprocess, _cache


def main():
    torch.manual_seed(42)
    initial = len(_cache)
    for heads in (8, 32):
        for size in (128, 256, 512, 1024):
            accum = torch.randn(1, heads, size * 128, device='cuda', dtype=torch.float32)
            output = torch.empty(1, size, heads, 128, device='cuda', dtype=torch.bfloat16)
            identity = torch.arange(size, dtype=torch.int32)
            postprocess(accum, output, prepare_rows(identity)[0].cuda(), .125, range_mapping=True)
            reference = output.clone()
            lengths = []
            for index in (identity, identity.roll(17), identity.flip(0), torch.randperm(size).int()):
                mapping = prepare_rows(index)[0].cuda()
                lengths.append(mapping.numel())
                output.fill_(float('nan'))
                start = time.perf_counter()
                before = len(_cache)
                postprocess(accum, output, mapping, .125, range_mapping=True)
                print(json.dumps(dict(heads=heads, size=size, descriptor_length=mapping.numel(),
                    cache_before=before, cache_after=len(_cache), host_ms=(time.perf_counter()-start)*1000)), flush=True)
                torch.testing.assert_close(output[:, index.long().cuda()], reference, rtol=0, atol=0)
                assert len(_cache) == initial + (1 if heads == 8 else 2), ('POSTPROCESS_CACHE_SPECIALIZATION', heads, size, lengths, list(_cache))
            print(json.dumps(dict(heads=heads, size=size, descriptor_lengths=lengths, cache_entries=len(_cache))), flush=True)
    torch.cuda.synchronize()
    print('Dynamic sequence/descriptor cache and exact scatter passed', flush=True)


if __name__ == '__main__':
    main()
