# SPDX-License-Identifier: Apache-2.0
"""Check the optional MAGI adapter against FP32 math and CUDA Graph replay."""

import argparse
import itertools
import json
import torch
from benchmarks.fixtures import make_fixture, reference, error
from benchmarks.magi import MagiBackward


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    mappings = ("mixed",) if a.quick else ("continuous", "two", "index", "mixed")
    masks = (
        ("causal",)
        if a.quick
        else ("full", "causal", "sparse", "empty", "external-inverse")
    )
    for mapping, mask in itertools.product(mappings, masks):
        if mask == "external-inverse" and mapping != "continuous":
            continue
        fixture = make_fixture(
            256, 1, mapping, "full" if mask == "external-inverse" else mask
        )
        if mask == "external-inverse":
            fixture["logical_mask"] = fixture["logical_mask"].triu()
            fixture["magi_q_ranges"] = torch.tensor([[0, 256]], dtype=torch.int32)
            fixture["magi_k_ranges"] = torch.tensor([[0, 256]], dtype=torch.int32)
            fixture["magi_attn_type_map"] = torch.tensor([2], dtype=torch.int32)
        adapter = MagiBackward(fixture)
        expected, _, grads = reference(fixture)
        fwd_error = error(adapter.out.unsqueeze(0), expected)
        assert (
            fwd_error["max_abs"] <= 0.0625 and fwd_error["rel_l2"] <= 0.006
        ), fwd_error
        outputs = tuple(torch.empty_like(fixture[name]) for name in ("q", "k", "v"))
        for _ in range(3):
            adapter(0, outputs)
        errors = [error(x, y) for x, y in zip(outputs, grads)]
        assert all(
            e["max_abs"] <= 0.125 and e["rel_l2"] <= 0.015 for e in errors
        ), errors
        eager = [x.clone() for x in outputs]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            adapter(0, outputs)
        torch.cuda.current_stream().wait_stream(stream)
        for _ in range(2):
            graph.replay()
        torch.cuda.synchronize()
        replay = [error(x, y) for x, y in zip(outputs, eager)]
        assert all(
            e["max_abs"] <= 0.0625 and e["rel_l2"] <= 0.005 for e in replay
        ), replay
        print(
            json.dumps(
                dict(
                    mapping=mapping,
                    mask=mask,
                    ranges=adapter.info["ranges"],
                    forward=fwd_error,
                    backward=errors,
                    replay=replay,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
