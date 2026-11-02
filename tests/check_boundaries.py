# SPDX-License-Identifier: Apache-2.0
"""Exhaustive split boundaries and supported outer-stride layouts on Hopper."""

import json
import torch
from tests.check_gpu import check
from benchmarks.fixtures import make_fixture


def main():
    for cut in range(1, 64):
        f = make_fixture(128, 1, "continuous", "causal")
        f["index"] = torch.arange(128, dtype=torch.int32).roll(cut)
        check(f, graph=True)
        print(json.dumps(dict(split=cut, passed=True)), flush=True)
    for layout in ("gapped", "head_major"):
        f = make_fixture(256, 2, "mixed", "sparse")
        for name in ("q", "k", "v", "dout"):
            x = f[name]
            f[name] = (
                x.repeat_interleave(2, dim=1)[:, ::2]
                if layout == "gapped"
                else x.transpose(1, 2).contiguous().transpose(1, 2)
            )
        check(f, graph=True)
        print(json.dumps(dict(layout=layout, passed=True)), flush=True)


if __name__ == "__main__":
    main()
