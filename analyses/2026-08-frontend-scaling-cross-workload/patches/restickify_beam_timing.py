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
import os
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

    # Inputs for reconstructing `live_indices` without patching the function:
    # buf_names in add_buf order, the last_use map, the graph input names, and
    # per-entry expansion counts attributed to the most recent add_buf.
    buf_order: list = field(default_factory=list)
    states_per_entry: list = field(default_factory=list)
    last_use: dict = field(default_factory=dict)
    input_names: set = field(default_factory=set)
    live_len_sum: int = 0
    live_max: int = 0
    live_at_last_op: int = 0

    # Cross-op repetition probe. EdgeCostMap already memoizes
    # compute_restickify_needed per (in_stl, target_stl) for ONE op-input edge,
    # so these counts ask a different question: how much of the work recurs
    # across edges, where that cache cannot see it.
    # Beam-width experiment: does BEAM_WIDTH=200 buy any solution quality?
    beam_width: int = 0
    best_cost: float = -1.0
    max_states_after_trim: int = 0
    states_into_trim_sum: int = 0
    states_after_trim_sum: int = 0

    crn_calls: int = 0
    crn_keys: set = field(default_factory=set)
    crn_unhashable: int = 0
    dc_calls: int = 0
    dc_keys: set = field(default_factory=set)
    dc_unhashable: int = 0

    # Per-edge invariants that compute_restickify_needed recomputes per
    # candidate pair: indirect_info_from_op (a function of `op` alone, and the
    # unconditional first statement) and host_coordinates (whose inputs are the
    # edge's construction-time snapshots). Both are candidates for hoisting
    # into EdgeCostMap.__init__ instead of memoizing.
    iifo_calls: int = 0
    iifo_ns: int = 0
    iifo_ops: set = field(default_factory=set)
    hc_calls: int = 0
    hc_ns: int = 0
    hc_keys: set = field(default_factory=set)

    def reset(self) -> None:
        self.__init__()  # type: ignore[misc]

    def note_state(self, n: int) -> None:
        self.n_states_built += 1
        self.assign_len_sum += n
        if n > self.assign_len_max:
            self.assign_len_max = n
        if self.states_per_entry:
            self.states_per_entry[-1] += 1

    def compute_liveness(self) -> None:
        """Reconstruct what `live_indices` would have counted, per op.

        `live_indices` keeps slot i when `last_use[buf_names[i]] > current_step`.
        current_step advances once per op with layouts, which are exactly the
        buf_order entries that are not graph inputs, in order. So the whole
        curve is recoverable from what the wrappers already saw.
        """
        import bisect

        seen: list[int] = []  # last_use values of entries so far, sorted
        step = -1
        for idx, name in enumerate(self.buf_order):
            bisect.insort(seen, self.last_use.get(name, -1))
            if name in self.input_names:
                continue
            step += 1
            # entries whose last use is strictly after this step are live
            live = len(seen) - bisect.bisect_right(seen, step)
            if live > self.live_max:
                self.live_max = live
            self.live_at_last_op = live
            if idx < len(self.states_per_entry):
                self.live_len_sum += self.states_per_entry[idx] * live

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
            "beam_width": self.beam_width,
            "best_cost": self.best_cost,
            "max_states_after_trim": self.max_states_after_trim,
            "states_into_trim_sum": self.states_into_trim_sum,
            "states_after_trim_sum": self.states_after_trim_sum,
            "crn_calls": self.crn_calls,
            "crn_distinct": len(self.crn_keys),
            "crn_unhashable": self.crn_unhashable,
            "crn_reuse_factor": (
                self.crn_calls / len(self.crn_keys) if self.crn_keys else None
            ),
            "dc_calls": self.dc_calls,
            "dc_distinct": len(self.dc_keys),
            "dc_unhashable": self.dc_unhashable,
            "dc_reuse_factor": (
                self.dc_calls / len(self.dc_keys) if self.dc_keys else None
            ),
            "iifo_calls": self.iifo_calls,
            "iifo_ms": self.iifo_ns / 1e6,
            "iifo_distinct_ops": len(self.iifo_ops),
            "iifo_reuse_factor": (
                self.iifo_calls / len(self.iifo_ops) if self.iifo_ops else None
            ),
            "hc_calls": self.hc_calls,
            "hc_ms": self.hc_ns / 1e6,
            "hc_distinct": len(self.hc_keys),
            "hc_reuse_factor": (
                self.hc_calls / len(self.hc_keys) if self.hc_keys else None
            ),
            "live_len_sum": self.live_len_sum,
            "live_max": self.live_max,
            "live_at_last_op": self.live_at_last_op,
            # What switching to live-slot-only state would save on volume.
            "volume_ratio": (
                self.assign_len_sum / self.live_len_sum
                if self.live_len_sum
                else None
            ),
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
                    # After reset, not at install: reset() restores defaults.
                    try:
                        from torch_spyre._inductor import optimize_restickify as _om

                        _C.beam_width = int(_om.BEAM_WIDTH)
                    except Exception:
                        pass
                    # Diagnostic only: attribute the expansion loop, which the
                    # counters show is ~75% of the beam and is not the cost
                    # evaluation, the trim, or the assignment-tuple copy.
                    if os.environ.get("SPYRE_RESTICKIFY_PROFILE") == "1":
                        import cProfile
                        import pstats

                        prof = cProfile.Profile()
                        prof.enable()
                        try:
                            return _orig(*a, **k)
                        finally:
                            prof.disable()
                            out = os.environ.get(
                                "SPYRE_RESTICKIFY_PROFILE_OUT", "/tmp/beam_profile.txt"
                            )
                            with open(out, "w") as fh:
                                st = pstats.Stats(prof, stream=fh)
                                st.sort_stats("tottime").print_stats(30)
                            print(f"beam profile -> {out}", flush=True)
                    try:
                        from torch._inductor.virtualized import V

                        _C.input_names = set(V.graph.graph_input_names)
                    except Exception:
                        pass
                    with _tr.stage("restickify:beam") as ev:
                        try:
                            return _orig(*a, **k)
                        finally:
                            # Reconstruction is O(N log N) and runs once, after
                            # the search; it inflates this event, which is why
                            # reported times come from the shim-off arm.
                            try:
                                _C.compute_liveness()
                            except Exception:
                                pass
                            _meta(ev, **_C.as_meta())

                return _timed_beam

            orl.beam_global_min_cost = _make_beam(orig)
        else:

            @functools.wraps(orig)
            def _timed(*a: Any, _orig: Any = orig, _event: str = event, **k: Any) -> Any:
                with _tr.stage(_event):
                    out = _orig(*a, **k)
                    if _event == "restickify:last_use" and isinstance(out, dict):
                        _C.last_use = out
                    return out

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
            _C.note_state(len(assignments))
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
        def _counting_add_buf(self: Any, name: str, *a: Any, **k: Any) -> Any:
            _C.add_buf_calls += 1
            _C.buf_order.append(name)
            _C.states_per_entry.append(0)
            return orig_add_buf(self, name, *a, **k)

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

    _install_beam_width_experiment()
    _install_repetition_probes()
    _tr.set_run_meta(restickify_beam_timing=_REPORT.as_meta())
    if _REPORT.missing:
        print(
            f"restickify_beam_timing: {len(_REPORT.wrapped)} wrapped, MISSING: "
            f"{'; '.join(sorted(_REPORT.missing))}",
            file=sys.stderr,
        )
    return _REPORT



def _key(obj: Any) -> Any:
    """Hashable stand-in for a call argument.

    Layouts and MemoryDeps are hashable (EdgeCostMap uses layouts as dict
    keys); FixedLayout and dicts may not be, and repr is a sound fallback for
    a distinctness count since equal reprs here imply equal structure.
    """
    try:
        hash(obj)
        return obj
    except TypeError:
        return repr(obj)


def _install_repetition_probes() -> None:
    """Count calls versus distinct arguments for the two profile hotspots.

    EdgeCostMap already memoizes compute_restickify_needed per
    (in_stl, target_stl) for ONE op-input edge. These counts ask the different
    question of how much work recurs ACROSS edges, where that cache cannot see
    it -- which is what decides whether a wider memo is worth prototyping.
    """
    from torch_spyre._inductor import optimize_restickify as om
    from torch_spyre._inductor import pass_utils as pu

    orig_crn = getattr(om, "compute_restickify_needed", None)
    if orig_crn is not None and not getattr(orig_crn, "_spyre_probed", False):

        @functools.wraps(orig_crn)
        def _probed_crn(in_stl, in_host, in_dep, out_stl, out_dep, op=None, *a, **k):
            _C.crn_calls += 1
            try:
                _C.crn_keys.add(
                    (
                        _key(in_stl),
                        _key(in_host),
                        _key(in_dep),
                        _key(out_stl),
                        _key(out_dep),
                        getattr(op, "name", None),
                    )
                )
            except Exception:
                _C.crn_unhashable += 1
            return orig_crn(in_stl, in_host, in_dep, out_stl, out_dep, op, *a, **k)

        _probed_crn._spyre_probed = True  # type: ignore[attr-defined]
        om.compute_restickify_needed = _probed_crn
        _REPORT.wrapped.append("optimize_restickify.compute_restickify_needed (probe)")

    orig_dc = getattr(pu, "device_coordinates", None)
    if orig_dc is not None and not getattr(orig_dc, "_spyre_probed", False):

        @functools.wraps(orig_dc)
        def _probed_dc(stl, dep, indirect_sizes, *a, **k):
            _C.dc_calls += 1
            try:
                isz = (
                    None
                    if indirect_sizes is None
                    else frozenset((_key(k), v) for k, v in indirect_sizes.items())
                )
                _C.dc_keys.add((_key(stl), _key(dep), isz))
            except Exception:
                _C.dc_unhashable += 1
            return orig_dc(stl, dep, indirect_sizes, *a, **k)

        _probed_dc._spyre_probed = True  # type: ignore[attr-defined]
        pu.device_coordinates = _probed_dc
        _REPORT.wrapped.append("pass_utils.device_coordinates (probe)")

    orig_iifo = getattr(pu, "indirect_info_from_op", None)
    if orig_iifo is not None and not getattr(orig_iifo, "_spyre_probed", False):

        @functools.wraps(orig_iifo)
        def _probed_iifo(op, *a, **k):
            _C.iifo_calls += 1
            _C.iifo_ops.add(getattr(op, "name", id(op)))
            t0 = time.perf_counter_ns()
            try:
                return orig_iifo(op, *a, **k)
            finally:
                _C.iifo_ns += time.perf_counter_ns() - t0

        _probed_iifo._spyre_probed = True  # type: ignore[attr-defined]
        pu.indirect_info_from_op = _probed_iifo
        _REPORT.wrapped.append("pass_utils.indirect_info_from_op (probe)")

    orig_hc = getattr(pu, "host_coordinates", None)
    if orig_hc is not None and not getattr(orig_hc, "_spyre_probed", False):

        @functools.wraps(orig_hc)
        def _probed_hc(host, dep, sizes, *a, **k):
            _C.hc_calls += 1
            try:
                _C.hc_keys.add((_key(host), _key(dep), _key(sizes)))
            except Exception:
                pass
            t0 = time.perf_counter_ns()
            try:
                return orig_hc(host, dep, sizes, *a, **k)
            finally:
                _C.hc_ns += time.perf_counter_ns() - t0

        _probed_hc._spyre_probed = True  # type: ignore[attr-defined]
        pu.host_coordinates = _probed_hc
        _REPORT.wrapped.append("pass_utils.host_coordinates (probe)")


def _install_beam_width_experiment() -> None:
    """Override BEAM_WIDTH and record the solution quality it produces.

    `Frontier(BEAM_WIDTH)` reads the module global when the search starts, so
    the width is settable from outside. `frontier.best().cost` is the committed
    solution's actual cost (future estimates are zero at the end), which is the
    quality number a narrower beam has to preserve to be free.
    """
    from torch_spyre._inductor import optimize_restickify as om

    width = os.environ.get("SPYRE_RESTICKIFY_BEAM_WIDTH")
    if width:
        try:
            om.BEAM_WIDTH = int(width)
            _REPORT.wrapped.append(f"optimize_restickify.BEAM_WIDTH := {width}")
        except ValueError:
            raise ValueError(f"SPYRE_RESTICKIFY_BEAM_WIDTH={width!r} is not an integer")
    _C.beam_width = getattr(om, "BEAM_WIDTH", -1)

    cls = getattr(om, "Frontier", None)
    if cls is None:
        _REPORT.missing.append("optimize_restickify.Frontier")
        return

    orig_best = cls.__dict__.get("best")
    if orig_best is not None and not getattr(orig_best, "_spyre_probed", False):

        @functools.wraps(orig_best)
        def _probed_best(self):
            out = orig_best(self)
            try:
                _C.best_cost = float(out.cost)
            except Exception:
                pass
            return out

        _probed_best._spyre_probed = True  # type: ignore[attr-defined]
        cls.best = _probed_best
        _REPORT.wrapped.append("optimize_restickify.Frontier.best")

    orig_trim = cls.__dict__.get("trim")
    if orig_trim is not None and not getattr(orig_trim, "_spyre_trimmed", False):

        @functools.wraps(orig_trim)
        def _probed_trim(self):
            n_in = len(self.states)
            _C.states_into_trim_sum += n_in
            out = orig_trim(self)
            n_out = len(self.states)
            _C.states_after_trim_sum += n_out
            if n_out > _C.max_states_after_trim:
                _C.max_states_after_trim = n_out
            return out

        _probed_trim._spyre_trimmed = True  # type: ignore[attr-defined]
        cls.trim = _probed_trim
        _REPORT.wrapped.append("optimize_restickify.Frontier.trim (width probe)")

__all__ = ["install"]
