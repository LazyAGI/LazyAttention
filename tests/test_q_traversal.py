# SPDX-License-Identifier: Apache-2.0
import unittest
import numpy as np
from lazy_attention.q_traversal import permute_q_lists

class QTraversalTests(unittest.TestCase):
    def test_complete_bijections_and_padding_for_partial_tails(self):
        counts=np.arange(34,dtype=np.int32).reshape(1,1,-1)
        indices=np.arange(34*40,dtype=np.int32).reshape(1,1,34,40)
        for reverse in (False,True):
            for interleave in (0,8):
                out=permute_q_lists(indices,counts,reverse=reverse,interleave=interleave)
                for row,count in enumerate(counts.ravel()):
                    got=out[0,0,row];original=indices[0,0,row]
                    self.assertEqual(sorted(got[:count]),list(original[:count]))
                    self.assertEqual(list(got[count:]),list(original[count:]))
    def test_known_local_order_and_reverse_composition(self):
        a=np.tile(np.arange(12,dtype=np.int32),(1,1,3,1));c=np.full((1,1,3),11,dtype=np.int32)
        self.assertEqual(permute_q_lists(a,c,interleave=8)[0,0,2].tolist(),[2,3,4,5,6,7,0,1,10,8,9,11])
        self.assertEqual(permute_q_lists(a,c,reverse=True,interleave=8)[0,0,2].tolist(),[8,7,6,5,4,3,10,9,0,2,1,11])
    def test_identity_preserves_object(self):
        a=np.arange(8).reshape(1,1,1,8)
        self.assertIs(permute_q_lists(a,np.array([[[8]]])),a)
    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):permute_q_lists(np.zeros((1,1,1,8)),np.array([[[9]]]),reverse=True)
        with self.assertRaises(ValueError):permute_q_lists(np.zeros((1,1,1,8)),np.array([[[8]]]),interleave=3)
if __name__=='__main__':unittest.main()
