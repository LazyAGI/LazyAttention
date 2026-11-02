# SPDX-License-Identifier: Apache-2.0
# Project-specific row mapping/copy policy; external CuTe/QuACK APIs are dependencies.
"""Q64 exclusive row descriptor, shared by reads and final gradient stores."""
import cutlass
import cutlass.cute as cute
from cutlass import Int32

@cute.jit
def decode_row_map(desc, n_q, block):
    header = desc[block]
    base0, base1, split, payload = Int32(0), Int32(0), Int32(0), Int32(-1)
    if header >= 0:
        base0, base1, split = header, header, Int32(64)
    else:
        pointer = desc[n_q + block]
        if header > -64:
            base0, base1, split = desc[pointer], desc[pointer + 1], -header
        else:
            payload = pointer
    return header, base0, base1, split, payload

@cute.jit
def mapped_row(desc, row, mapping):
    header, base0, base1, split, payload = mapping
    physical = base0 + row
    if header < 0:
        if split > 0:
            if row >= split:
                physical = base1 + row - split
        else:
            physical = desc[payload + row]
    return physical
