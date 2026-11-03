# SPDX-License-Identifier: Apache-2.0
"""Summarize unchanged paired blocks and planning/42-adjusted costs."""
import argparse,json,math,statistics
from pathlib import Path
import numpy as np
from scipy.stats import t
from benchmarks.analyze import estimate


def summarize(records,field):
    single=[estimate([b['dlog'] for b in r[field]]) for r in records]
    means=[statistics.mean(b['dlog'] for b in r[field]) for r in records]
    mean=statistics.mean(means)
    half=float(t.ppf(.975,len(means)-1))*statistics.stdev(means)/math.sqrt(len(means))
    return dict(time_change_pct=100*math.expm1(mean),
        ci95_pct=[min(100*math.expm1(mean-half),*[s['ci95_pct'][0] for s in single]),max(100*math.expm1(mean+half),*[s['ci95_pct'][1] for s in single])],
        mean_ms=[statistics.mean(v['ms'] for r in records for b in r[field] for v in b['values'] if v['label']==i) for i in (0,1)],processes=single)


def main():
    p=argparse.ArgumentParser();p.add_argument('files',nargs='+',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    rs=[json.loads(f.read_text()) for f in a.files]
    assert len(rs)>=3 and len({r['repeat'] for r in rs})==len(rs)
    for key in ('fixture_sha256','source_sha256','environment','gpu_uuid','capture_sha256','planning'):
        assert all(r[key]==rs[0][key] for r in rs),key
    result=dict(gpu=summarize(rs,'blocks'),effective=summarize(rs,'effective_blocks'),planning_wall_mean_ms=statistics.mean(v for r in rs for v in r['planning_wall_ms']),planning=rs[0]['planning'])
    if a.output:a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
