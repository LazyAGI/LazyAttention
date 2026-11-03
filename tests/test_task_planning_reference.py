# SPDX-License-Identifier: Apache-2.0
import unittest
from tests.check_task_planning import reference_order

class TaskOrderReferenceTests(unittest.TestCase):
    def test_stable_sample_buckets_and_head_locality(self):
        # Docs0 occupies two KV tiles, docs1 one, final tile spans documents.
        docs=[0]*256+[1]*128+[2]*64+[3]*64
        got=reference_order([1,2,5,9],docs,2,quantum=4)
        self.assertEqual(got,[6,7,4,5,0,2,1,3])
    def test_short_retains_current_kv_major(self):
        self.assertEqual(reference_order([1,2],[0]*256,2),list(range(4)))
    def test_equal_bucket_preserves_first_group_order(self):
        self.assertEqual(reference_order([65,100],[9]*128+[2]*128,2),list(range(4)))
if __name__=='__main__':unittest.main()
