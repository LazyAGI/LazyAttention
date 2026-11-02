"""Run explicitly on an idle Hopper: python -m tests.check_gpu."""
import argparse
import itertools
import json
import torch
from lazy_attention import prepare_attention, forward, backward
from benchmarks.fixtures import make_fixture, reference, error


def check(fixture, variants=('index','paired'), graph=True):
    q,k,v,do = [fixture[x] for x in ('q','k','v','dout')]
    plans = {s:prepare_attention(fixture['logical_mask'],fixture['index'],device=q.device,strategy=s)
             for s in ('index','range')}
    out,lse = forward(q,k,v,plans['range'])
    expected,_,grads = reference(fixture)
    forward_error = error(out,expected)
    assert forward_error['max_abs'] < .0625 and forward_error['rel_l2'] < .006, forward_error
    results = {}; baseline = None
    for variant in variants:
        plan = plans['index' if variant == 'index' else 'range']
        def launch(): return backward(q,k,v,out,do,lse,plan,variant=variant)
        got = launch()
        errors = [error(a,b) for a,b in zip(got,grads)]
        assert all(e['max_abs'] < .125 and e['rel_l2'] < .015 for e in errors), (variant,errors)
        if baseline is None: baseline = [x.clone() for x in got]
        cross = [error(a,b) for a,b in zip(got,baseline)]
        assert all(e['max_abs'] <= .0625 and e['rel_l2'] < .005 for e in cross), (variant,cross)
        if graph:
            stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3): launch()
            torch.cuda.current_stream().wait_stream(stream)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g,stream=stream): captured = launch()
            torch.cuda.current_stream().wait_stream(stream)
            for _ in range(2): g.replay()
            torch.cuda.synchronize()
            replay_error = [error(a,b) for a,b in zip(captured,got)]
            assert all(e['max_abs'] <= .0625 and e['rel_l2'] < .005 for e in replay_error), (variant,replay_error)
            errors_after_replay = [error(a,b) for a,b in zip(captured,grads)]
            assert all(e['max_abs'] < .125 and e['rel_l2'] < .015 for e in errors_after_replay), (variant,errors_after_replay)
            results[variant + '_graph'] = replay_error
        results[variant] = dict(reference=errors,index=cross)
    return dict(forward=forward_error,backward=results)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--quick',action='store_true'); a=p.parse_args()
    shapes = (256,) if a.quick else (128,256,512)
    mappings = ('mixed',) if a.quick else ('continuous','two','index','mixed')
    masks = ('causal',) if a.quick else ('causal','full','sparse','empty')
    for s,m,k in itertools.product(shapes,mappings,masks):
        result = check(make_fixture(s,1,m,k))
        print(json.dumps(dict(seqlen=s,mapping=m,mask=k,result=result)),flush=True)
if __name__ == '__main__': main()
