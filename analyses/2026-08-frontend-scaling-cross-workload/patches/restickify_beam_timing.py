# Copyright 2025-2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Counters and timers for `optimize_restickify_locations`'s beam search.

Tests one hypothesis: that the pass's superlinear scaling is driven by
*bookkeeping volume* rather than by the number of hypotheses explored.

`BeamState.assignments` is a tuple parallel to the frontier's buf_names, so
its length is the index of the op being processed. Three places pay O(index)
per state, per op, in `beam_global_min_cost`:

  1. expansion   `state.assignments + (candidate_stl,)` -- a full copy per
                 state x candidate
  2. merge key   a second full-length tuple per expanded state, then hashed
  3. live_indices a scan of all buf_names, once per op

So the predicted cost is proportional to `sum over expansions of
len(assignments)` -- call it the assignment-copy volume -- and NOT to the
expansion count. The two grow differently: volume carries an extra factor of
the op index. Fitting measured time against both separates them.

Instrumentation is deliberately indirect to avoid patching the one large
function: wrapping `BeamState.__init__` yields both the expansion count and
the volume, because every expansion constructs a state from a freshly
concatenated tuple whose length is what we need. It cannot separate the
expansion copy from the merge key, which have the same predicted volume;
that split needs in-function timers and is only worth adding if this
confirms volume is the driver.

Per-expansion work is accumulated and reported once per pass, never emitted
as events: expansions number in the hundreds of thousands.

Gated on TORCH_SPYRE_TIMING=1 via timing_recorder.
"""

from __future__ import annotations

import functools
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from . import timing_recorder as _tr
except ImportError:  # standalone, from patches/
    from torch_spyre._inductor import timing_recorder as _tr


@dataclass
class _BeamCounters:
    """Per-pass accumulators. `assign_len_sum` is the load-bearing one."""

    n_states_built: int = 0
    assign_len_sum: int = 0
    assign_len_max: int = 0
    cost_calls: int = 0
    cost_ns: int = 0
    trim_calls: int = 0
    trim_ns: int = 0
    trim_states_in_max: int = 0
    add_buf_calls: int = 0

    def reset(self) -> None:
        self.__init__()  # type: ignore[misc]

    def as_meta(self) -> dict[str, Any]:
        return {
            "n_states_built": self.n_states_built,
            "assign_len_sum": self.assign_len_sum,
            "assign_len_max": self.assign_len_max,
            "cost_calls": self.cost_calls,
            "cost_ms": self.cost_ns / 1e6,
            "trim_calls": self.trim_calls,
            "trim_ms": self.trim_ns / 1e6,
            "trim_states_in_max": self.trim_states_in_max,
            "n_ops_with_layouts": self.add_buf_calls,
        }


@dataclass
class _Report:
    wrapped: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def as_meta(self) -> dict[str, Any]:
        return {"wrapped": sorted(self.wrapped), "missing": sorted(self.missing)}


_C = _BeamCounters()
_REPORT = _Report()
_INSTALLED = False


def _meta(ev: Any, **kv: Any) -> None:
    if ev is None:
        return
    ev.meta.update({k: v for k, v in kv.items() if v is not None})


def install() -> _Report:
    """Wrap the beam search. Safe to call more than once."""
    global _INSTALLED
    if _INSTALLED:
        return _REPORT
    _INSTALLED = True
    if not _tr.is_enabled():
        return _REPORT

    from torch_spyre._inductor import optimize_restickify as orl

    # Whole-pass bracket, and the once-per-pass phases around the op loop.
    for fn_name, event in (
        ("beam_global_min_cost", "restickify:beam"),
        ("compute_future_min_cost", "restickify:future_min_cost"),
        ("_compute_last_use", "restickify:last_use"),
        ("_reorder_any_in_nodes", "restickify:reorder_any_in_nodes"),
    ):
        orig = getattr(orl, fn_name, None)
        if orig is None:
            _REPORT.missing.append(f"optimize_restickify.{fn_name}")
            continue

        if fn_name == "beam_global_min_cost":

            def _make_beam(_orig: Any) -> Any:
                @functools.wraps(_orig)
                def _timed_beam(*a: Any, **k: Any) -> Any:
                    _C.reset()
                    with _tr.stage("restickify:beam") as ev:
                        try:
                            return _orig(*a, **k)
                        finally:
                            _meta(ev, **_C.as_meta())

                return _timed_beam

            orl.beam_global_min_cost = _make_beam(orig)
        else:

            @functools.wraps(orig)
            def _timed(*a: Any, _orig: Any = orig, _event: str = event, **k: Any) -> Any:
                with _tr.stage(_event):
                    return _orig(*a, **k)

            setattr(orl, fn_name, _timed)
        _REPORT.wrapped.append(f"optimize_restickify.{fn_name}")

    # BeamState construction: one per expansion, and the tuple it receives was
    # just built by the concatenation whose cost we are trying to attribute, so
    # its length is the per-expansion volume.
    state_cls = getattr(orl, "BeamState", None)
    if state_cls is None:
        _REPORT.missing.append("optimize_restickify.BeamState")
    else:
        orig_init = state_cls.__init__

        @functools.wraps(orig_init)
        def _counting_init(self: Any, assignments: Any, *a: Any, **k: Any) -> Any:
            n = len(assignments)
            _C.n_states_built += 1
            _C.assign_len_sum += n
            if n > _C.assign_len_max:
                _C.assign_len_max = n
            return orig_init(self, assignments, *a, **k)

        state_cls.__init__ = _counting_init
        _REPORT.wrapped.append("optimize_restickify.BeamState.__init__")

    frontier_cls = getattr(orl, "Frontier", None)
    if frontier_cls is None:
        _REPORT.missing.append("optimize_restickify.Frontier")
    else:
        orig_trim = frontier_cls.trim

        @functools.wraps(orig_trim)
        def _timed_trim(self: Any) -> Any:
            n = len(self.states)
            if n > _C.trim_states_in_max:
                _C.trim_states_in_max = n
            t0 = time.perf_counter_ns()
            try:
                return orig_trim(self)
            finally:
                _C.trim_ns += time.perf_counter_ns() - t0
                _C.trim_calls += 1

        frontier_cls.trim = _timed_trim
        _REPORT.wrapped.append("optimize_restickify.Frontier.trim")

        orig_add_buf = frontier_cls.add_buf

        @functools.wraps(orig_add_buf)
        def _counting_add_buf(self: Any, name: str) -> Any:
            _C.add_buf_calls += 1
            return orig_add_buf(self, name)

        frontier_cls.add_buf = _counting_add_buf
        _REPORT.wrapped.append("optimize_restickify.Frontier.add_buf")

    # Cost evaluation, to separate real cost-model work from bookkeeping.
    for cls_name in ("AllSameNode", "FixedInOutNode", "AnyInNode"):
        cls = getattr(orl, cls_name, None)
        if cls is None or "cost" not in cls.__dict__:
            _REPORT.missing.append(f"optimize_restickify.{cls_name}.cost")
            continue
        orig_cost = cls.__dict__["cost"]

        @functools.wraps(orig_cost)
        def _timed_cost(self: Any, *a: Any, _orig: Any = orig_cost, **k: Any) -> Any:
            t0 = time.perf_counter_ns()
            try:
                return _orig(self, *a, **k)
            finally:
                _C.cost_ns += time.perf_counter_ns() - t0
                _C.cost_calls += 1

        cls.cost = _timed_cost
        _REPORT.wrapped.append(f"optimize_restickify.{cls_name}.cost")

    _tr.set_run_meta(restickify_beam_timing=_REPORT.as_meta())
    if _REPORT.missing:
        print(
            f"restickify_beam_timing: {len(_REPORT.wrapped)} wrapped, MISSING: "
            f"{'; '.join(sorted(_REPORT.missing))}",
            file=sys.stderr,
        )
    return _REPORT


__all__ = ["install"]
