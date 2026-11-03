# SPDX-License-Identifier: Apache-2.0
import copy
import math
import unittest
from benchmarks.report import summarize, markdown


def record(repeat=0):
    return dict(
        fixture_sha256="same-fixture",
        backends=["magi", "paired"],
        phase="ab",
        protocol="shared-outputs-private-scratch-v1",
        source_sha256={"code": "hash"},
        versions={"torch": "test"},
        environment={},
        repeat=repeat,
        shape=[1, 256, 4, 128],
        config={"mapping": "mixed", "mask": "causal"},
        correctness=[[dict(max_abs=0.001, rel_l2=0.0001)]],
        telemetry=[dict(gpu="GPU-test, H100, telemetry")],
        scope="backward",
        excludes=["forward"],
        blocks=[
            dict(
                dlog=math.log(0.9), values=[dict(label=0, ms=1), dict(label=1, ms=0.9)]
            )
            for _ in range(12)
        ],
    )


class ReportTests(unittest.TestCase):
    def test_paired_effect_and_sign(self):
        rows = summarize([record(i) for i in range(3)])
        self.assertAlmostEqual(rows[0]["time_change_pct"], -10)
        self.assertAlmostEqual(rows[0]["median_us"][0], 1000)
        self.assertIn("magi → paired", markdown(rows))

    def test_duplicate_and_nan_rejected(self):
        with self.assertRaises(ValueError):
            summarize([record(), record()])
        r = record()
        r["correctness"][0][0]["rel_l2"] = float("nan")
        with self.assertRaises(ValueError):
            summarize([r])

    def test_different_comparisons_are_separate(self):
        r = record()
        r["backends"] = ["index", "paired"]
        self.assertEqual(len(summarize([r, record()])), 2)


if __name__ == "__main__":
    unittest.main()
