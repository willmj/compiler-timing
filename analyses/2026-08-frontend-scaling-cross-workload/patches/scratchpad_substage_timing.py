# Copyright 2025-2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Substage timers for the `_maybe_scratchpad_planning` pass.

Attributes `ScratchpadAllocator.plan_allocation` to 100% of its named
steps *without editing torch-spyre source*: every boundary we need is
either a method call on `self` or a module-global call inside
`allocator.py`, so the wraps go on at runtime. That matters because the
committed scratchpad dataset was taken on a pod snapshot (`a9316b381`)
which is not current main, and `scratchpad/` has churned since -- a
source-anchored patch would be unverifiable from outside the pod.

Three levels, so instrumentation overhead can be attributed per level:

  1  plan_allocation's ten template steps only (default).
  2  + buffer-preparation helpers, plus the solver's plan_layout/partition.
  3  + per-buffer accumulators inside plan_layout, plus the live-set
     counters the cost model needs.

Level 3's accumulators are deliberately NOT recorder events: they fire
once per buffer, and one event each would both distort the measurement
and bloat the sample JSON. They are summed and stamped onto the
enclosing `scratchpad:plan_layout` event instead.

`plan_layout`'s residual is the hypothesis test. The time-stepping loop
in `greedy_solver.py` rescans every placeable buffer at every transition
tick -- Theta(n_placeable * n_times) -- and that loop is inline, so it
cannot be wrapped. It therefore surfaces as

    plan_layout.inclusive - (try_allocate_ns + try_deallocate_ns)

A large residual confirms the scan is the driver; a small one refutes it
and sends the investigation to the prepare_buffers side instead.

Gated on TORCH_SPYRE_TIMING=1 via `timing_recorder`. Level comes from
SPYRE_SCRATCHPAD_TIMING_LEVEL (default 1).

Usage on the pod -- no tree modification:

    export TORCH_SPYRE_TIMING=1
    export SPYRE_SCRATCHPAD_TIMING_LEVEL=2
    export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_wa_r1
    rm -rf $TORCHINDUCTOR_CACHE_DIR
    python scratchpad_substage_timing.py run \
        $HOME/pr3806/workload_harness.py --Lq 512 --Lk 1024 --out $OUT

Verify the wraps resolve against this tree before burning device time:

    python scratchpad_substage_timing.py check

Alternatively, drop this file into `torch_spyre/_inductor/` and call
`install()` alongside `extra_timers.install_extra_timers()`.
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
except ImportError:  # running as a standalone script from patches/
    from torch_spyre._inductor import timing_recorder as _tr


# The default is 1 because level 1 alone answers the prepare-vs-solve
# question, and that answer decides whether level 2/3 are worth running.
_LEVEL_ENV = "SPYRE_SCRATCHPAD_TIMING_LEVEL"


def _resolve_level() -> int:
    raw = os.environ.get(_LEVEL_ENV, "1")
    try:
        level = int(raw)
    except ValueError:
        raise ValueError(f"{_LEVEL_ENV}={raw!r} is not an integer (want 1, 2, or 3)")
    if level not in (1, 2, 3):
        raise ValueError(f"{_LEVEL_ENV}={level} out of range (want 1, 2, or 3)")
    return level


# ---------------------------------------------------------------------------
# Explicit instrumentation state


@dataclass
class _PlanState:
    """Per-`plan_allocation()` state.

    `_run_passes` is called twice from the same template with the same
    name (pre-passes then post-passes), so a call ordinal is the only
    thing that can distinguish the two events.
    """

    run_passes_calls: int = 0

    def reset(self) -> None:
        self.run_passes_calls = 0


@dataclass
class _SolverCounters:
    """Aggregated per-call costs inside one `plan_layout()`.

    Summed rather than emitted as events (see module docstring).
    `live_sum` is the candidate cost-model normalizer: `_try_allocate`
    both copies `self.usage` and gap-scans it, so the total work is
    proportional to the sum of the live-set size over all allocate
    attempts, not to the buffer count alone.
    """

    try_allocate_ns: int = 0
    try_allocate_calls: int = 0
    try_deallocate_ns: int = 0
    try_deallocate_calls: int = 0
    live_sum: int = 0
    live_max: int = 0
    # Stamped by the partition wrap (level 2+); None when that wrap is off.
    n_placeable: Optional[int] = None
    n_times: Optional[int] = None

    def reset(self) -> None:
        self.__init__()  # type: ignore[misc]

    def observe_live(self, live: int) -> None:
        self.live_sum += live
        if live > self.live_max:
            self.live_max = live

    def as_meta(self) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "n_placeable": self.n_placeable,
            "n_times": self.n_times,
        }
        if self.try_allocate_calls or self.try_deallocate_calls:
            meta.update(
                try_allocate_ms=self.try_allocate_ns / 1e6,
                try_allocate_calls=self.try_allocate_calls,
                try_deallocate_ms=self.try_deallocate_ns / 1e6,
                try_deallocate_calls=self.try_deallocate_calls,
                live_sum=self.live_sum,
                live_max=self.live_max,
            )
        return meta


@dataclass
class _InstallReport:
    """What actually got wrapped, so a run on a drifted tree is
    diagnosable from the sample JSON rather than silently partial."""

    level: int = 0
    wrapped: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    # Skips of boundaries the reconciliation depends on. A subclass that
    # does not override an optional hook, or an abstract declaration, is
    # expected and lands in `skipped` alone; anything here means a
    # boundary vanished and its time would be absorbed into a parent's
    # self_ns without any visible sign.
    missing_required: list[str] = field(default_factory=list)

    def as_meta(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "wrapped": sorted(self.wrapped),
            "skipped": sorted(self.skipped),
            "missing_required": sorted(self.missing_required),
        }


_PLAN = _PlanState()
_SOLVER = _SolverCounters()
_REPORT = _InstallReport()
_INSTALLED = False

# Monotonic across the process. `scratchpad_planning` retries the whole
# allocation with a greedy allocator after a SolveError, so one pass can
# emit two `scratchpad:plan_allocation` events; this makes the retry
# visible instead of looking like a duplicate.
_ATTEMPT = 0


# ---------------------------------------------------------------------------
# Wrap helpers


def _meta(ev: Any, **kv: Any) -> None:
    """Stamp counters onto an open event, dropping unavailable ones.

    A counter we could not read is absent, never zero -- the reducer
    treats absent and zero differently on purpose.
    """
    if ev is None:
        return
    ev.meta.update({k: v for k, v in kv.items() if v is not None})


def _count(obj: Any, attr: str = "") -> Optional[int]:
    try:
        return len(getattr(obj, attr) if attr else obj)
    except Exception:
        return None


def _reason_bucket(reason: Optional[str]) -> Optional[str]:
    """Collapse a residency reason to a stable histogram key.

    Some reasons carry per-buffer detail that would make every barred
    buffer its own key: ``no room on scratchpad (t=..., size=...)``, and
    ``core div mismatch: <why> on '<op>': <numbers>``. The core-division
    sub-reason is worth keeping -- a K-split writer and a broadcast read
    are different problems with different remedies -- so only the op name
    and the numbers are dropped.
    """
    if reason is None:
        return None
    head, _, tail = reason.partition(":")
    head = head.split("(")[0].strip()
    if not tail.strip():
        return head
    sub = tail.strip().split(" on ")[0].split("'")[0].split("(")[0].strip()
    return f"{head}: {sub}" if sub else head


def _claim(
    owner: Any, name: str, *, own_dict: bool = True, required: bool = False
) -> Optional[Any]:
    """Return `owner`'s own attribute `name`, or None (reporting the miss).

    `own_dict` restricts the lookup to the class's own `__dict__` so a
    subclass that does not override a hook is not wrapped twice: the
    inherited base wrap already covers it.

    `required` marks a boundary the reconciliation depends on, so its
    absence is reported as tree drift rather than as an expected gap.
    Whether a miss is benign cannot be inferred from why it missed --
    `_prepare_buffers` being absent from `CoOptimizingAllocator` is
    normal, the same absence on `ScratchpadAllocator` is not.

    Abstract methods are always declined: replacing one with a wrapper
    would strip `__isabstractmethod__` and silently disable ABC
    enforcement, and the body never runs anyway.
    """
    label = f"{getattr(owner, '__name__', owner)}.{name}"

    def _skip(reason: str) -> None:
        _REPORT.skipped.append(f"{label} ({reason})")
        if required and reason != "abstract":
            _REPORT.missing_required.append(f"{label} ({reason})")

    if own_dict and isinstance(owner, type):
        if name not in owner.__dict__:
            _skip("not defined on this class")
            return None
        attr = owner.__dict__[name]
    else:
        attr = getattr(owner, name, None)
        if attr is None:
            _skip("missing")
            return None
    target = attr.__func__ if isinstance(attr, staticmethod) else attr
    if getattr(target, "__isabstractmethod__", False):
        _skip("abstract")
        return None
    _REPORT.wrapped.append(label)
    return attr


def _simple_stage(owner: Any, name: str, event: str, *, required: bool = False) -> None:
    """Wrap a callable whose timing needs no counters of its own."""
    orig = _claim(
        owner, name, own_dict=isinstance(owner, type), required=required
    )
    if orig is None:
        return
    is_static = isinstance(orig, staticmethod)
    func = orig.__func__ if is_static else orig

    @functools.wraps(func)
    def _timed(*args: Any, **kwargs: Any) -> Any:
        with _tr.stage(event):
            return func(*args, **kwargs)

    setattr(owner, name, staticmethod(_timed) if is_static else _timed)


# ---------------------------------------------------------------------------
# Level 1 -- plan_allocation's template steps


def _install_level1(alloc_mod: Any) -> None:
    classes = [alloc_mod.ScratchpadAllocator]
    co_opt = getattr(alloc_mod, "CoOptimizingAllocator", None)
    if co_opt is not None:
        classes.append(co_opt)
    base = classes[0]

    orig_plan = _claim(base, "plan_allocation", required=True)
    if orig_plan is not None:

        @functools.wraps(orig_plan)
        def _timed_plan_allocation(self: Any, graph: Any, *a: Any, **k: Any) -> Any:
            global _ATTEMPT
            _ATTEMPT += 1
            _PLAN.reset()
            with _tr.stage("scratchpad:plan_allocation") as ev:
                _meta(
                    ev,
                    n_ops=_count(graph, "operations"),
                    allocator_class=type(self).__name__,
                    attempt=_ATTEMPT,
                    timing_level=_REPORT.level,
                )
                return orig_plan(self, graph, *a, **k)

        base.plan_allocation = _timed_plan_allocation

    orig_run_passes = _claim(base, "_run_passes", required=True)
    if orig_run_passes is not None:
        run_passes = (
            orig_run_passes.__func__
            if isinstance(orig_run_passes, staticmethod)
            else orig_run_passes
        )

        @functools.wraps(run_passes)
        def _timed_run_passes(passes: Any, graph: Any) -> Any:
            _PLAN.run_passes_calls += 1
            first = _PLAN.run_passes_calls == 1
            event = "scratchpad:run_pre_passes" if first else "scratchpad:run_post_passes"
            with _tr.stage(event) as ev:
                _meta(ev, n_passes=_count(passes))
                return run_passes(passes, graph)

        base._run_passes = staticmethod(_timed_run_passes)

    # Overridden by CoOptimizingAllocator, so wrap per class that defines
    # it. Only the base's copy is required: a subclass that inherits the
    # hook is already covered by the base wrap.
    for cls in classes:
        orig_prepare = _claim(cls, "_prepare_buffers", required=cls is base)
        if orig_prepare is None:
            continue

        def _make_prepare(_orig: Any) -> Any:
            @functools.wraps(_orig)
            def _timed_prepare(self: Any, *a: Any, **k: Any) -> Any:
                graph = a[0] if a else k.get("graph")
                with _tr.stage("scratchpad:prepare_buffers") as ev:
                    _meta(ev, n_ops=_count(graph, "operations"))
                    buffers = _orig(self, *a, **k)
                    _meta(ev, n_buffers=_count(buffers))
                    return buffers

            return _timed_prepare

        cls._prepare_buffers = _make_prepare(orig_prepare)

    orig_build = _claim(base, "_build_solver", required=True)
    if orig_build is not None:

        @functools.wraps(orig_build)
        def _timed_build_solver(self: Any, buffers: Any) -> Any:
            with _tr.stage("scratchpad:build_solver") as ev:
                _meta(ev, n_buffers=_count(buffers))
                solver = orig_build(self, buffers)
                _meta(ev, solver_class=type(solver).__name__)
                return solver

        base._build_solver = _timed_build_solver

    for cls in classes:
        orig_solve = _claim(cls, "_solve", required=cls is base)
        if orig_solve is None:
            continue

        def _make_solve(_orig: Any) -> Any:
            @functools.wraps(_orig)
            def _timed_solve(self: Any, *a: Any, **k: Any) -> Any:
                solver = a[0] if a else k.get("solver")
                with _tr.stage("scratchpad:solve") as ev:
                    _meta(
                        ev,
                        n_buffers=_count(solver, "buffers"),
                        solver_class=type(solver).__name__,
                    )
                    allocation = _orig(self, *a, **k)
                    # The placement rate is the pass's efficacy metric:
                    # everything else here measures how long it took to
                    # decide, not what it achieved.
                    try:
                        placed = sum(
                            1 for b in allocation if b.address is not None
                        )
                        _meta(
                            ev, n_placed=placed, n_spilled=len(allocation) - placed
                        )
                    except Exception:
                        pass
                    return allocation

            return _timed_solve

        cls._solve = _make_solve(orig_solve)

    _simple_stage(
        base,
        "_finalize_lx_relayout_allocation",
        "scratchpad:finalize_lx_relayout",
        required=True,
    )
    _simple_stage(base, "_log_lx_pinning", "scratchpad:log_lx_pinning", required=True)
    _simple_stage(base, "_push_allocation", "scratchpad:push_allocation", required=True)
    for cls in classes:
        req = cls is base
        _simple_stage(cls, "_post_solve", "scratchpad:post_solve", required=req)
        _simple_stage(
            cls, "_get_spill_reasons", "scratchpad:get_spill_reasons", required=req
        )


# ---------------------------------------------------------------------------
# Level 2 -- prepare-side helpers and the solver's own boundaries


def _install_level2(alloc_mod: Any) -> None:
    base = alloc_mod.ScratchpadAllocator

    orig_generate = _claim(base, "_generate_buffers", required=True)
    if orig_generate is not None:

        @functools.wraps(orig_generate)
        def _timed_generate(self: Any, graph: Any, *a: Any, **k: Any) -> Any:
            with _tr.stage("scratchpad:generate_buffers") as ev:
                buffers = orig_generate(self, graph, *a, **k)
                _meta(ev, n_buffers=_count(buffers))
                return buffers

        base._generate_buffers = _timed_generate

    orig_reasons = _claim(base, "_residency_reasons", required=True)
    if orig_reasons is not None:

        @functools.wraps(orig_reasons)
        def _timed_reasons(self: Any, graph: Any, names: Any, **k: Any) -> Any:
            with _tr.stage("scratchpad:residency_reasons") as ev:
                _meta(ev, n_names=_count(names))
                reasons = orig_reasons(self, graph, names, **k)
                try:
                    hist: dict[str, int] = {}
                    eligible = 0
                    for reason in reasons.values():
                        bucket = _reason_bucket(reason)
                        if bucket is None:
                            eligible += 1
                        else:
                            hist[bucket] = hist.get(bucket, 0) + 1
                    _meta(ev, n_eligible=eligible, n_barred=sum(hist.values()),
                          barred_by_reason=dict(sorted(hist.items(), key=lambda kv: -kv[1])))
                except Exception:
                    pass
                return reasons

        base._residency_reasons = _timed_reasons

    _simple_stage(
        base, "_determine_in_place", "scratchpad:determine_in_place", required=True
    )
    _simple_stage(
        base, "_build_bound_buffers", "scratchpad:build_bound_buffers", required=True
    )
    _simple_stage(
        base,
        "_append_lx_relayout_destinations",
        "scratchpad:append_lx_relayout_destinations",
        required=True,
    )

    # These are module-level functions that allocator.py imported by name, so
    # the binding to rebind is allocator.py's own global -- patching the
    # defining module would not affect the already-bound reference, and would
    # also catch callers outside scratchpad planning.
    for func_name, event in (
        ("calculate_liveness", "scratchpad:calculate_liveness"),
        ("get_ncores_for_buffers", "scratchpad:get_ncores_for_buffers"),
        ("mem_usage_by_buf", "scratchpad:mem_usage_by_buf"),
        ("_get_buffer_user_deps", "scratchpad:get_buffer_user_deps"),
        ("collect_lx_relayout_plans", "scratchpad:collect_lx_relayout_plans"),
    ):
        _simple_stage(alloc_mod, func_name, event, required=True)

    _install_solver_boundaries()


def _solver_classes() -> list[tuple[type, bool]]:
    """Solver classes to instrument as `(cls, required)`, base first.

    `partition` lives on `MemoryPlanSolver`, so wrapping the base covers
    every solver; `plan_layout` is overridden per solver.

    Only `GreedyLayoutSolver` is required: `config.layout_solver` resolves
    to `"greedy"` in every committed sample, so that is the class whose
    absence would silently gut the solve-side attribution. The others are
    instrumented opportunistically in case a sweep changes the knob.
    """
    from torch_spyre._inductor.scratchpad import plan_solver as ps

    # The base carries `partition` (required, claimed separately) but not
    # the per-buffer allocate/deallocate hooks, so it is not required for
    # the wraps driven off this list.
    classes: list[tuple[type, bool]] = [(ps.MemoryPlanSolver, False)]
    for mod_name, cls_name, required in (
        ("greedy_solver", "GreedyLayoutSolver", True),
        ("firstfit_bestfit_solver", "BestFitLayoutSolver", False),
        ("firstfit_bestfit_solver", "FirstFitLayoutSolver", False),
        ("simulated_annealing", "SimulatedAnnealingLayoutSolver", False),
    ):
        try:
            mod = __import__(
                f"torch_spyre._inductor.scratchpad.{mod_name}", fromlist=[cls_name]
            )
        except ImportError:
            continue
        cls = getattr(mod, cls_name, None)
        if cls is not None:
            classes.append((cls, required))
    return classes


def _install_solver_boundaries() -> None:
    classes = _solver_classes()
    solver_base = classes[0][0]

    # partition both gets its own event and supplies the two structural
    # counters plan_layout needs (n_placeable, n_times), so it is one wrap.
    orig_partition = _claim(solver_base, "partition", required=True)
    if orig_partition is not None:

        @functools.wraps(orig_partition)
        def _timed_partition(self: Any, *a: Any, **k: Any) -> Any:
            with _tr.stage("scratchpad:solver_partition") as ev:
                result = orig_partition(self, *a, **k)
                placeable = result[0] if isinstance(result, tuple) else result
                _SOLVER.n_placeable = _count(placeable)
                try:
                    times = set()
                    for buf in placeable:
                        times.add(buf.start_time)
                        times.add(buf.end_time)
                    _SOLVER.n_times = len(times)
                except Exception:
                    _SOLVER.n_times = None
                _meta(ev, n_placeable=_SOLVER.n_placeable, n_times=_SOLVER.n_times)
                return result

        solver_base.partition = _timed_partition

    for cls, required in classes:
        orig_plan_layout = _claim(cls, "plan_layout", required=required)
        if orig_plan_layout is None:
            continue

        @functools.wraps(orig_plan_layout)
        def _timed_plan_layout(
            self: Any, *a: Any, _orig: Any = orig_plan_layout, **k: Any
        ) -> Any:
            _SOLVER.reset()
            with _tr.stage("scratchpad:plan_layout") as ev:
                _meta(ev, n_buffers=_count(self, "buffers"))
                try:
                    return _orig(self, *a, **k)
                finally:
                    _meta(ev, **_SOLVER.as_meta())

        cls.plan_layout = _timed_plan_layout


# ---------------------------------------------------------------------------
# Level 3 -- per-buffer accumulators inside plan_layout


def _install_level3() -> None:
    for cls, required in _solver_classes():
        orig_allocate = _claim(cls, "_try_allocate", required=required)
        if orig_allocate is not None:

            def _make_allocate(_orig: Any) -> Any:
              @functools.wraps(_orig)
              def _counting_allocate(self: Any, *a: Any, **k: Any) -> Any:
                live = _count(self, "usage")
                if live is not None:
                    _SOLVER.observe_live(live)
                t0 = time.perf_counter_ns()
                try:
                    return _orig(self, *a, **k)
                finally:
                    _SOLVER.try_allocate_ns += time.perf_counter_ns() - t0
                    _SOLVER.try_allocate_calls += 1
              return _counting_allocate

            cls._try_allocate = _make_allocate(orig_allocate)

        orig_deallocate = _claim(cls, "_try_deallocate", required=required)
        if orig_deallocate is not None:

            def _make_deallocate(_orig: Any) -> Any:
              @functools.wraps(_orig)
              def _counting_deallocate(self: Any, *a: Any, **k: Any) -> Any:
                t0 = time.perf_counter_ns()
                try:
                    return _orig(self, *a, **k)
                finally:
                    _SOLVER.try_deallocate_ns += time.perf_counter_ns() - t0
                    _SOLVER.try_deallocate_calls += 1
              return _counting_deallocate

            cls._try_deallocate = _make_deallocate(orig_deallocate)


# ---------------------------------------------------------------------------
# Install


def install() -> _InstallReport:
    """Wrap the scratchpad boundaries. Safe to call more than once.

    Returns the install report, which is also stamped into the sample
    JSON's run meta so a partially-wrapped run cannot be mistaken for a
    complete one.
    """
    global _INSTALLED
    if _INSTALLED:
        return _REPORT
    _INSTALLED = True
    if not _tr.is_enabled():
        return _REPORT

    _REPORT.level = _resolve_level()

    from torch_spyre._inductor.scratchpad import allocator as alloc_mod

    _install_level1(alloc_mod)
    if _REPORT.level >= 2:
        _install_level2(alloc_mod)
    if _REPORT.level >= 3:
        _install_level3()

    _tr.set_run_meta(scratchpad_timing=_REPORT.as_meta())
    unexpected = _REPORT.missing_required
    if unexpected:
        print(
            f"scratchpad_substage_timing: level {_REPORT.level}, "
            f"{len(_REPORT.wrapped)} wrapped, {len(unexpected)} MISSING "
            f"(tree drift): {'; '.join(sorted(unexpected))}",
            file=sys.stderr,
        )
    return _REPORT


# ---------------------------------------------------------------------------
# CLI


def _cmd_check() -> int:
    """Resolve every wrap against this tree without compiling anything."""
    if not _tr.is_enabled():
        print("TORCH_SPYRE_TIMING is not 1; set it or `check` reports nothing.")
        return 2
    report = install()
    print(f"level {report.level}")
    for label in sorted(report.wrapped):
        print(f"  wrapped  {label}")
    for label in sorted(report.skipped):
        print(f"  skipped  {label}")
    # An abstract declaration or an unoverridden hook is expected. A name
    # that is simply gone means that boundary vanished from the sum, and
    # the reconciliation would silently absorb it.
    unexpected = report.missing_required
    if unexpected:
        print(f"\nFATAL: boundary missing from this tree: {unexpected}", file=sys.stderr)
        return 1
    print("\nall boundaries resolved against this tree")
    return 0


def _cmd_run(argv: list[str]) -> int:
    """Arm the wraps, then run a harness in this process."""
    if not argv:
        print("usage: scratchpad_substage_timing.py run <harness.py> [args...]", file=sys.stderr)
        return 2
    import runpy

    install()
    harness, *harness_args = argv
    sys.argv = [harness, *harness_args]
    runpy.run_path(harness, run_name="__main__")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in ("check", "run"):
        print(
            "usage: scratchpad_substage_timing.py {check | run <harness.py> [args...]}",
            file=sys.stderr,
        )
        return 2
    return _cmd_check() if args[0] == "check" else _cmd_run(args[1:])


__all__ = ["install", "main"]


if __name__ == "__main__":
    sys.exit(main())
