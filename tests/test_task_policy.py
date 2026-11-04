# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace
import unittest
from lazy_attention.task_policy import TaskFeatures, select_task_strategy

class JointPolicyTests(unittest.TestCase):
    def setUp(self):
        self.f=TaskFeatures(32640,8,132,2040,2040,488,159.,450.,.8,.93,.39,320000,8,True)
    def test_four_routes(self):
        self.assertEqual(select_task_strategy(self.f).strategy,'reorder')
        self.assertEqual(select_task_strategy(replace(self.f,max_q_tiles=135,split_fraction=.012)).strategy,'reorder')
        self.assertEqual(select_task_strategy(replace(self.f,max_q_tiles=69,reorder_changes_order=False)).strategy,'none')
        low=replace(self.f,seqlen=512,kv_heads=2,tasks=8,active_tasks=8,max_q_tiles=8,
                    mean_active_q_tiles=5.,total_q_tiles=40,reorder_changes_order=True)
        d=select_task_strategy(low)
        self.assertEqual((d.strategy,d.split_size),('split',2))
    def test_mask_and_occupancy_change_choice(self):
        self.assertEqual(select_task_strategy(replace(self.f,full_fraction=.2)).strategy,'reorder')
        self.assertEqual(select_task_strategy(replace(self.f,active_tasks=0)).strategy,'none')
        self.assertEqual(select_task_strategy(replace(self.f,reorder_changes_order=False)).strategy,'none')
    def test_permissions(self):
        self.assertEqual(select_task_strategy(self.f,split=False).strategy,'reorder')
        self.assertEqual(select_task_strategy(self.f,reorder=False).strategy,'none')
        self.assertEqual(select_task_strategy(self.f,reorder=False,split=False).strategy,'none')
    def test_decision_boundaries(self):
        self.assertEqual(select_task_strategy(replace(self.f,max_q_tiles=256)).strategy,'reorder')
        self.assertEqual(select_task_strategy(replace(self.f,max_q_tiles=257)).strategy,'reorder')
        self.assertEqual(select_task_strategy(replace(self.f,split_fraction=.1249)).strategy,'reorder')
        self.assertEqual(select_task_strategy(replace(self.f,split_fraction=.125)).strategy,'reorder')
        self.assertEqual(select_task_strategy(replace(self.f,full_fraction=.4999)).strategy,'reorder')
        self.assertEqual(select_task_strategy(replace(self.f,full_fraction=.5)).strategy,'reorder')
        low=replace(self.f,max_q_tiles=8,reorder_changes_order=False,tasks=66,active_tasks=66,
                    mean_active_q_tiles=5.,total_q_tiles=330)
        self.assertEqual(select_task_strategy(low).strategy,'none')
        self.assertEqual(select_task_strategy(replace(low,tasks=65,active_tasks=65)).strategy,'split')
        self.assertEqual(select_task_strategy(replace(low,tasks=8,active_tasks=8,mean_active_q_tiles=3.999)).strategy,'none')
    def test_short_nonidentity_order_still_reorders(self):
        self.assertEqual(select_task_strategy(replace(self.f,max_q_tiles=69)).strategy,'reorder')
    def test_invalid_options(self):
        with self.assertRaises(ValueError): select_task_strategy(self.f,split_size=0)
