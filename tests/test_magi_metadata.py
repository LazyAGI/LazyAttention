# SPDX-License-Identifier: Apache-2.0
import unittest
import numpy as np
from benchmarks.magi import encode_mask, validate_ranges


class MagiMetadataTests(unittest.TestCase):
    def test_exact_disjoint_encoding(self):
        rng = np.random.default_rng(0)
        for size in (1, 7, 32):
            for mask in (
                np.zeros((size, size), bool),
                np.ones((size, size), bool),
                np.tri(size, dtype=bool),
                rng.random((size, size)) > 0.4,
            ):
                validate_ranges(mask, encode_mask(mask))

    def test_compact_common_masks(self):
        for mask in (np.ones((128, 128), bool), np.tri(128, dtype=bool)):
            self.assertEqual(len(encode_mask(mask)[0]), 1)

    def test_external_inverse_causal(self):
        mask = np.zeros((5, 5), bool)
        mask[:3] = np.triu(np.ones((3, 5), bool))
        validate_ranges(mask, (np.array([[0, 3]]), np.array([[0, 5]]), np.array([2])))

    def test_overlap_and_missing_rejected(self):
        mask = np.ones((4, 4), bool)
        q, k, t = encode_mask(mask)
        with self.assertRaises(ValueError):
            validate_ranges(mask, (np.r_[q, q], np.r_[k, k], np.r_[t, t]))
        with self.assertRaises(ValueError):
            validate_ranges(~mask, (q, k, t))


if __name__ == "__main__":
    unittest.main()
