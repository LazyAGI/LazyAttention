# SPDX-License-Identifier: Apache-2.0
"""Independent CPU task permutation oracle and GPU planning correctness."""
from collections import defaultdict


def reference_order(counts, documents, heads, quantum=64, preserve_short=True):
    n=len(counts)
    if preserve_short and max(counts)<=quantum: return list(range(n*heads))
    groups=defaultdict(list)
    for b in range(n):
        ds=set(int(x) for x in documents[b*128:(b+1)*128] if x>=0)
        key=-1 if not ds else next(iter(ds)) if len(ds)==1 else -b-2
        groups[key].append(b)
    buckets=sorted(groups.values(),key=lambda blocks:(-max((counts[b]+quantum-1)//quantum for b in blocks),min(blocks)))
    return [b*heads+h for blocks in buckets for h in range(heads) for b in blocks]


def main():
    import torch
    from types import SimpleNamespace
    from lazy_attention.task_planning import build_task_order
    checks=0
    for n in (2,9,64,256):
        for heads in (1,2,8):
            for kind in ('empty','single','mixed','padding'):
                gen=torch.Generator().manual_seed(n+heads)
                counts=torch.randint(0,2*n+1,(n,),generator=gen,dtype=torch.int32)
                docs=(torch.arange(n*128)//256).int()
                if kind=='empty': counts.zero_();docs.fill_(-1)
                elif kind=='single': docs.zero_()
                elif kind=='mixed': docs[120::128]=-1;docs[64::128]+=1
                else: docs[-128:]=-1;counts[-1]=0
                pc=counts.cuda().reshape(1,1,n);fc=torch.zeros_like(pc)
                plan=SimpleNamespace(mask_block_cnt=pc,full_block_cnt=fc,block_size=(64,128))
                for quantum in (1,64):
                    for preserve in (True,False):
                        got=build_task_order(plan,docs,n*128,heads,quantum=quantum,preserve_short=preserve).cpu().tolist()
                        expected=reference_order(counts.tolist(),docs.tolist(),heads,quantum,preserve)
                        assert got==expected,(n,heads,kind,quantum,preserve)
                        assert sorted(got)==list(range(n*heads))
                        checks+=1
    print(f'{checks} GPU plans match independent CPU oracle, complete bijections',flush=True)
    from lazy_attention import prepare_attention,forward,backward
    from benchmarks.fixtures import make_fixture,reference,error
    for size in (128,512):
        for mask in ('empty','full','causal','sparse'):
            f=make_fixture(size,2,'mixed',mask)
            p=prepare_attention(f['logical_mask'],f['index']);out,lse=forward(f['q'],f['k'],f['v'],p)
            order=build_task_order(p.backward_plan,torch.zeros(size,dtype=torch.int32),size,2,quantum=1,preserve_short=False)
            if size==512:
                assert order.cpu().tolist()!=list(range(order.numel())), 'test must exercise a nonidentity schedule'
            ref=reference(f)[2]
            def call():return backward(f['q'],f['k'],f['v'],out,f['dout'],lse,p,task_order=order)
            got=call()
            assert all(error(a,b)['rel_l2']<.015 and error(a,b)['max_abs']<.125 for a,b in zip(got,ref))
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):call()
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream): captured=call()
            torch.cuda.current_stream().wait_stream(stream)
            for _ in range(3):graph.replay()
            torch.cuda.synchronize()
            assert all(error(a,b)['rel_l2']<.015 and error(a,b)['max_abs']<.125 for a,b in zip(captured,ref))
    print('8 full backward FP32/Graph cases passed; S512 schedules are nonidentity',flush=True)
if __name__=='__main__':main()
