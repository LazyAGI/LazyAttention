"""Summarize raw paired blocks with circular block-bootstrap sensitivity."""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def estimate(values, seed=0, trim=0):
    x=np.asarray(values,dtype=float)
    if len(x)<12 or len(x)<=2*trim+1: raise ValueError('insufficient paired blocks')
    def statistic(y):
        if trim: y=np.sort(y,axis=-1)[...,trim:-trim]
        return y.mean(axis=-1)
    rng=np.random.default_rng(seed); bounds=[]
    for length in (1,3,5,10):
        starts=rng.integers(0,len(x),(10000,math.ceil(len(x)/length)))
        indices=(starts[:,:,None]+np.arange(length))%len(x)
        samples=x[indices.reshape(10000,-1)[:,:len(x)]]
        bounds.append(100*np.expm1(np.quantile(statistic(samples),[.025,.975])))
    return dict(time_change_pct=float(100*np.expm1(statistic(x))),
                ci95_pct=[float(min(b[0] for b in bounds)),float(max(b[1] for b in bounds))],
                sensitivity_ci95_pct=[b.tolist() for b in bounds],blocks=len(x),trim_each_tail=trim)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('files',nargs='+',type=Path); a=p.parse_args()
    records=[json.loads(path.read_text()) for path in a.files]
    for path,r in zip(a.files,records):
        ds=[b['dlog'] for b in r['blocks']]
        full=estimate(ds)
        row=dict(file=str(path),phase=r['phase'],full=full)
        if r['phase'].endswith('-aa'):
            trimmed=estimate(ds,trim=5)
            def passes(stats,threshold):
                lo,hi=stats['ci95_pct']; return lo>=-threshold and hi<=threshold
            row.update(trimmed=trimmed,full_pass=passes(full,.5),trimmed_pass=passes(trimmed,.3))
            row['aa_pass']=row['full_pass'] or row['trimmed_pass']
        print(json.dumps(row))
    # Independent process repeats are the experimental units for cross-run CI.
    # Only combine the same fixture/phase; do not silently pool different cases.
    if len(records)>=3:
        from scipy.stats import t
        keys=[(r['phase'],r['shape'],r['config']['mapping'],r['config']['mask'],r['fixture_sha256'],r['source_sha256'],r.get('backends'),r.get('protocol'),r.get('environment'),r.get('magi')) for r in records]
        if any(k!=keys[0] for k in keys): raise ValueError('aggregate only identical fixtures and phases')
        ids=[r['repeat'] for r in records]
        if len(set(ids))!=len(ids): raise ValueError('repeat IDs must be distinct')
        values=np.array([np.mean([b['dlog'] for b in r['blocks']]) for r in records])
        mean=values.mean(); half=t.ppf(.975,len(values)-1)*values.std(ddof=1)/np.sqrt(len(values))
        print(json.dumps(dict(independent_process_repeats=len(values),
            time_change_pct=float(100*np.expm1(mean)),ci95_pct=(100*np.expm1([mean-half,mean+half])).tolist())))
if __name__=='__main__': main()
