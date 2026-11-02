# SPDX-License-Identifier: Apache-2.0
"""Own private launcher caches without patching public FlashAttention modules."""
import importlib.util
import sys


def private_module(public_name, suffix):
    spec = importlib.util.find_spec(public_name)
    if spec is None or spec.origin is None:
        raise ImportError(f"Cannot locate {public_name}; install the documented FA4 dependencies")
    parent = public_name.rpartition('.')[0]
    name = f"{parent}._lazy_{suffix}"
    private = importlib.util.spec_from_file_location(name, spec.origin)
    module = importlib.util.module_from_spec(private)
    sys.modules[name] = module
    try:
        private.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module
