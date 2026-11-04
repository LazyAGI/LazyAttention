# SPDX-License-Identifier: Apache-2.0
"""One-time policy: full-corpus reorder route and prior underfilled-GPU route."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class TaskFeatures:
    seqlen: int
    kv_heads: int
    sm_count: int
    tasks: int
    active_tasks: int
    max_q_tiles: int
    mean_active_q_tiles: float
    p95_q_tiles: float
    coefficient_of_variation: float
    full_fraction: float
    split_fraction: float
    total_q_tiles: int
    groups: object
    reorder_changes_order: bool

@dataclass(frozen=True)
class TaskDecision:
    reorder: bool
    split: bool
    split_size: int
    reason: str
    @property
    def strategy(self):
        return ('none','split','reorder','both')[int(self.reorder)*2+int(self.split)]

def select_task_strategy(features, *, reorder=True, split=True, split_size=128):
    if split_size<1 or features.sm_count<1:
        raise ValueError('positive split_size and sm_count required')
    if not reorder and not split:
        return TaskDecision(False,False,split_size,'switches-off')
    if not features.active_tasks:
        return TaskDecision(False,False,split_size,'empty-mask')
    r=bool(reorder and features.reorder_changes_order)
    # Keep the previously measured underfilled regime independent of the
    # high-occupancy240-case corpus; short-task reorder was not validated there.
    if features.active_tasks*2 < features.sm_count:
        r=bool(r and features.max_q_tiles>128)
    # Small shape/mask can leave most CTAs idle. Prior small-shape splitting
    # evidence supports this separate regime; minimum2 avoids tiny fragments.
    underfilled=(features.active_tasks*2 < features.sm_count
                 and features.mean_active_q_tiles>=4 and features.max_q_tiles>=4)
    if split and underfilled:
        chunk=min(split_size,max(2,math.ceil(features.total_q_tiles/(2*features.sm_count))))
        if features.max_q_tiles>chunk:
            return TaskDecision(r,True,chunk,'underfilled-gpu')
    # Full240x4 H100 matrix: reorder alone is the best fixed strategy.
    # Max-Q/full-mask thresholds did not justify automatic splitting; leave
    # the split treatment available explicitly through policy='manual'.
    if r:
        return TaskDecision(True,False,split_size,'full-corpus-reorder')
    return TaskDecision(False,False,split_size,'no-supported-profitable-change')
