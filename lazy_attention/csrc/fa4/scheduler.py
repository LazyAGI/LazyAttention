# SPDX-License-Identifier: BSD-3-Clause
# Derived from FlashAttention CuTe; see licenses/flash-attention-BSD-3-Clause.txt.
# Modified: persistent GQA4 scheduling and exclusive range/index row mapping.
"""Producer-prefetched atomic work queue for the three FA4 backward roles."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from flash_attn.cute import utils
from flash_attn.cute.tile_scheduler import WorkTileInfo
from flash_attn.cute.block_sparse_utils import get_m_block_from_iter_bwd


@cute.jit
def shifted_q_tile(it, pc, pi, fc, fi, kv, span: cutlass.Constexpr[int]):
    mapped = it
    if cutlass.const_expr(span > 0):
        # Shift only within the current mask class. Nonempty branch guards modulus.
        if it < pc:
            offset = cutlass.min(kv % span, pc - 1)
            mapped = it + offset
            if mapped >= pc:
                mapped = mapped - pc
        else:
            offset = cutlass.min(kv % span, fc - 1)
            local = it - pc + offset
            if local >= fc:
                local = local - fc
            mapped = pc + local
    return get_m_block_from_iter_bwd(mapped, pc, pi, fc, fi)


@cute.jit
def traversal_q_tile(it, pc, pi, fc, fi, kv, span: cutlass.Constexpr[int],
                     full_first: cutlass.Constexpr[bool] = False):
    if cutlass.const_expr(full_first):
        # Reversed lists are prepared once. Swapping class visitation completes
        # the reversal of[Partial,Full] without an extra per-iteration map load.
        block, is_partial = shifted_q_tile(it, fc, fi, pc, pi, kv, span)
        return block, not is_partial
    else:
        return shifted_q_tile(it, pc, pi, fc, fi, kv, span)


from cutlass.cutlass_dsl import dsl_user_op
from cutlass._mlir.dialects import llvm


@dsl_user_op
def subset_barrier(*, barrier_id, number_of_threads, loc=None, ip=None):
    # Unlike legacy bar.sync (.aligned), barrier.sync permits distinct role paths.
    llvm.inline_asm(
        None,
        [
            Int32(barrier_id).ir_value(loc=loc, ip=ip),
            Int32(number_of_threads).ir_value(loc=loc, ip=ip),
        ],
        "barrier.sync $0, $1;",
        "r,r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def subset_arrive(*, barrier_id, number_of_threads, loc=None, ip=None):
    cute.arch.barrier_arrive(
        barrier_id=barrier_id,
        number_of_threads=number_of_threads,
        aligned=False,
        loc=loc,
        ip=ip,
    )


class SubsetBarrier(cutlass.pipeline.NamedBarrier):
    @dsl_user_op
    def arrive_and_wait(self, *, loc=None, ip=None):
        subset_barrier(
            barrier_id=self.barrier_id,
            number_of_threads=self.num_threads,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def arrive(self, *, loc=None, ip=None):
        subset_arrive(
            barrier_id=self.barrier_id,
            number_of_threads=self.num_threads,
            loc=loc,
            ip=ip,
        )


class PersistentScheduler:
    def __init__(self, params, order, counter, mailbox, is_producer, participants=384, planned=False, split=False):
        self.params, self.order, self.counter = params, order, counter
        self.mailbox, self.is_producer = mailbox, is_producer
        self.participants = participants
        self.planned, self.split = planned, split

    @staticmethod
    @cute.jit
    def create(
        params,
        order,
        counter,
        mailbox,
        is_producer: cutlass.Constexpr[bool] = False,
        participants: cutlass.Constexpr[int] = 384,
        planned: cutlass.Constexpr[bool] = False,
        split: cutlass.Constexpr[bool] = False,
    ):
        return PersistentScheduler(
            params, order, counter, mailbox, is_producer, participants, planned, split
        )

    @cute.jit
    def publish(self, index):
        if cutlass.const_expr(self.is_producer):
            if cute.arch.lane_idx() == 0:
                n, h = self.params.num_block, self.params.num_head
                valid = index < n * h
                if cutlass.const_expr(self.split):
                    valid = index < cute.size(self.order.shape[0])
                if cutlass.const_expr(self.planned):
                    task = Int32(0)
                    if valid:
                        if cutlass.const_expr(self.split):
                            task = self.order[index, 0]
                            self.mailbox[5] = self.order[index, 1]
                            self.mailbox[6] = self.order[index, 2]
                            self.mailbox[7] = self.order[index, 3]
                        else:
                            task = self.order[index]
                    self.mailbox[0] = task // h
                    self.mailbox[1] = task % h
                else:
                    self.mailbox[0] = index // h
                    self.mailbox[1] = index % h
                self.mailbox[2] = Int32(0)
                self.mailbox[3] = Int32(valid)

    @cute.jit
    def prefetch_next_work(self):
        if cutlass.const_expr(self.is_producer):
            if cute.arch.lane_idx() == 0:
                self.mailbox[4] = (
                    utils.atomic_add_i32(1, self.counter.iterator)
                    + cute.arch.grid_dim()[0]
                )

    @cute.jit
    def initial_work_tile_info(self):
        self.publish(Int32(cute.arch.block_idx()[0]))
        SubsetBarrier(barrier_id=13, num_threads=self.participants).arrive_and_wait()
        return self.get_current_work()

    @cute.jit
    def get_current_work(self):
        return WorkTileInfo(
            (
                Int32(self.mailbox[0]),
                Int32(self.mailbox[1]),
                Int32(self.mailbox[2]),
                Int32(0),
            ),
            self.mailbox[3] != 0,
        )

    @cute.jit
    def advance_to_next_work(self):
        cute.arch.cp_async_bulk_wait_group(0, read=True)
        SubsetBarrier(barrier_id=12, num_threads=self.participants).arrive_and_wait()
        if cutlass.const_expr(self.is_producer):
            self.prefetch_next_work()
            self.publish(Int32(self.mailbox[4]))
        SubsetBarrier(barrier_id=13, num_threads=self.participants).arrive_and_wait()

    @cute.jit
    def q_bounds(self, count):
        start, end = Int32(0), count
        if cutlass.const_expr(self.split):
            start, end = Int32(self.mailbox[5]), Int32(self.mailbox[6])
        return start, end

    def _objects(self):
        return (
            (self.params, self.order, self.counter, self.mailbox)
            if self.is_producer
            else (self.mailbox,)
        )

    def __extract_mlir_values__(self):
        values, self._sizes = [], []
        for x in self._objects():
            v = cutlass.extract_mlir_values(x)
            values += v
            self._sizes.append(len(v))
        return values

    def __new_from_mlir_values__(self, values):
        args = []
        for x, n in zip(self._objects(), self._sizes):
            args.append(cutlass.new_from_mlir_values(x, values[:n]))
            values = values[n:]
        if self.is_producer:
            return PersistentScheduler(*args, self.is_producer, self.participants, self.planned, self.split)
        return PersistentScheduler(
            self.params, self.order, self.counter, args[0], False, self.participants, self.planned, self.split
        )
