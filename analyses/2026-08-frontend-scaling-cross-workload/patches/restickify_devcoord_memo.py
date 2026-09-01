# Copyright 2025-2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Prototype: memoize `device_coordinates` for the duration of one beam search.

`device_coordinates(stl, dep, indirect_sizes)` is a pure function of its
arguments and is the second-largest self-time entry in a cProfile of
`beam_global_min_cost` (2.01 s of a profiled beam, ~0.78 ms per call). Measured
call-versus-distinct-argument counts on tiled flash attention show a stable
4.55-4.58x reuse factor, i.e. 78% of the calls recompute a result the search
already produced.

`EdgeCostMap` already memoizes `compute_restickify_needed` per
(in_stl, target_stl), but only within ONE op-input edge -- and that function's
own reuse factor is exactly 1.00, so the per-edge cache is already tight. The
redundancy is one level down, where the same (layout, dep) pair recurs across
edges, which a per-edge cache cannot see.

Scoped to a single `beam_global_min_cost` call rather than process-global. That
is deliberate: the coarse-tile study found a naive global memo on a
dependency-derived helper broke correctness because the cache outlived the
mutations that invalidated it. Clearing on entry and exit keeps the cache inside
a region where candidate layouts are only read.

Returns a copy of the cached list so no caller can mutate another's result
through the cache. The lists hold one expression per device dimension, so the
copy is a few elements.

Enable with SPYRE_RESTICKIFY_DEVCOORD_MEMO=1.
"""

from __future__ import annotations

import functools
import os
import sys
from typing import Any

_MISS = object()

_INSTALLED = False
stats = {"calls": 0, "hits": 0, "misses": 0, "uncacheable": 0, "max_entries": 0}


def install() -> bool:
    """Install the beam-scoped memo. Returns whether it was applied."""
    global _INSTALLED
    if _INSTALLED:
        return True
    if os.environ.get("SPYRE_RESTICKIFY_DEVCOORD_MEMO") != "1":
        return False
    _INSTALLED = True

    from torch_spyre._inductor import optimize_restickify as om
    from torch_spyre._inductor import pass_utils as pu

    orig_dc = pu.device_coordinates
    cache: dict[Any, Any] = {}

    @functools.wraps(orig_dc)
    def _memo_dc(stl, dep, indirect_sizes, *a, **k):
        stats["calls"] += 1
        try:
            isz = (
                None
                if indirect_sizes is None
                else frozenset(indirect_sizes.items())
            )
            key = (stl, dep, isz)
        except TypeError:
            stats["uncacheable"] += 1
            return orig_dc(stl, dep, indirect_sizes, *a, **k)
        got = cache.get(key, _MISS)
        if got is _MISS:
            stats["misses"] += 1
            # An unrepresentable stick expression raises; leave it uncached so
            # the raising path stays identical rather than caching an exception.
            got = orig_dc(stl, dep, indirect_sizes, *a, **k)
            cache[key] = got
            if len(cache) > stats["max_entries"]:
                stats["max_entries"] = len(cache)
        else:
            stats["hits"] += 1
        return list(got)

    pu.device_coordinates = _memo_dc

    orig_beam = om.beam_global_min_cost

    @functools.wraps(orig_beam)
    def _scoped_beam(*a: Any, **k: Any) -> Any:
        cache.clear()
        try:
            return orig_beam(*a, **k)
        finally:
            cache.clear()

    om.beam_global_min_cost = _scoped_beam
    print(
        "restickify_devcoord_memo: installed (beam-scoped)",
        file=sys.stderr,
    )
    return True


__all__ = ["install", "stats"]
