# Copyright 2025-2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Per-pass timers for all six Spyre pass pipelines, installed at runtime.

Emits the same event names the in-tree instrumentation patch produces --
`pipeline:<Class>` and `pass:<Class>:<pass_name>` -- so samples taken with
this shim reduce alongside the committed dataset. Nothing in the
torch-spyre tree is modified.

Each pipeline iterates `self.passes` and calls each entry with the graph,
the FX graph, or the scheduler-node list, so wrapping the list entries at
first `__call__` covers every pass uniformly. Two details make that safe:

* `_uuid()` keys the Inductor cache on `inspect.getfile` of each pass, or
  of its `_pass_sources` when present. Each wrapper therefore carries the
  original's `_pass_sources`, so the cache key is unchanged and a timed
  run is still cache-comparable to an untimed one.
* `_get_pass_name` reads `__name__`, which `functools.wraps` preserves, so
  the pipelines' own provenance observers keep their correct pass names.

The timer brackets `pass_fn` only. It deliberately excludes the
`SpyreGraphTransformObserver` that the pipeline wraps around each pass, so
a pass's time is its own work and not provenance bookkeeping.

Gated on TORCH_SPYRE_TIMING=1 via timing_recorder.
"""

from __future__ import annotations

import functools
import sys
from typing import Any, Optional

try:
    from . import timing_recorder as _tr
except ImportError:  # standalone, from patches/
    from torch_spyre._inductor import timing_recorder as _tr


# The five FX/node pipelines inherit `__call__` from one of two bases and
# only differ in their pass list, so wrapping the bases covers all of them;
# CustomPreSchedulingPasses defines its own. Each wrapper names its events
# from `type(self).__name__`, so the concrete pipeline still labels itself.
_PIPELINE_OWNERS = (
    "_SpyreGraphPassPipeline",
    "_SpyreNodePassPipeline",
    "CustomPreSchedulingPasses",
)

_MARK = "_spyre_timing_wrapped"

_INSTALLED = False
wrapped: list[str] = []
missing: list[str] = []


def _meta(ev: Any, **kv: Any) -> None:
    if ev is None:
        return
    ev.meta.update({k: v for k, v in kv.items() if v is not None})


def _target_counters(target: Any) -> dict[str, int]:
    """Size counters for whichever target shape this pipeline passes along.

    GraphLowering carries `operations` (the pre-scheduling axis, and the
    variable every published per-pass scaling table is plotted against);
    an FX graph carries `nodes`; a node pipeline gets a plain list.
    """
    ops = getattr(target, "operations", None)
    if ops is not None:
        try:
            return {"input_operations": len(ops)}
        except Exception:
            return {}
    nodes = getattr(target, "nodes", None)
    if nodes is not None:
        try:
            return {"n_fx_nodes": len(list(nodes))}
        except Exception:
            return {}
    try:
        return {"n_nodes": len(target)}
    except Exception:
        return {}


def _wrap_pass(orig: Any, pipeline_name: str, index: int) -> Any:
    from torch_spyre._inductor.passes import _get_pass_name

    name = _get_pass_name(orig)
    event = f"pass:{pipeline_name}:{name}"

    @functools.wraps(orig)
    def _timed(target: Any, *a: Any, **k: Any) -> Any:
        with _tr.stage(event) as ev:
            # Two distinct passes can share a name -- CustomPostPasses runs
            # both mm_to_bmm_pass.apply and bmm_unflatten_pass.apply. The
            # event name stays as the in-tree patch emits it so samples
            # reduce together; pass_index is what tells them apart.
            _meta(ev, pass_index=index, **_target_counters(target))
            return orig(target, *a, **k)

    # Keep the Inductor cache key identical to the unwrapped pipeline.
    _timed._pass_sources = getattr(orig, "_pass_sources", (orig,))  # type: ignore[attr-defined]
    setattr(_timed, _MARK, True)
    return _timed


def _wrap_pipeline(cls: type, owner_name: str) -> bool:
    if "__call__" not in cls.__dict__:
        missing.append(f"{owner_name}.__call__ (not defined on this class)")
        return False
    orig_call = cls.__dict__["__call__"]

    @functools.wraps(orig_call)
    def _timed_call(self: Any, target: Any, *a: Any, **k: Any) -> Any:
        name = type(self).__name__
        # Wrapped lazily: `self.passes` is built in __init__, and Inductor
        # reuses one pipeline instance across compiles, so this runs once
        # per instance rather than once per compile.
        passes = getattr(self, "passes", None)
        if passes is not None and not getattr(self, _MARK, False):
            self.passes = [
                p if getattr(p, _MARK, False) else _wrap_pass(p, name, i)
                for i, p in enumerate(passes)
            ]
            setattr(self, _MARK, True)
        with _tr.stage(f"pipeline:{name}") as ev:
            _meta(ev, **_target_counters(target))
            _meta(ev, n_passes=_count(getattr(self, "passes", None)))
            return orig_call(self, target, *a, **k)

    cls.__call__ = _timed_call
    wrapped.append(f"{owner_name}.__call__")
    return True


def _count(obj: Any) -> Optional[int]:
    try:
        return len(obj)
    except Exception:
        return None


def install() -> dict[str, Any]:
    """Wrap every Spyre pass pipeline. Safe to call more than once."""
    global _INSTALLED
    if _INSTALLED:
        return report()
    _INSTALLED = True
    if not _tr.is_enabled():
        return report()

    from torch_spyre._inductor import passes as passes_mod

    for name in _PIPELINE_OWNERS:
        cls = getattr(passes_mod, name, None)
        if cls is None:
            missing.append(f"{name} (absent from passes module)")
            continue
        _wrap_pipeline(cls, name)

    _tr.set_run_meta(pass_pipeline_timing=report())
    if missing:
        print(f"pass_pipeline_timing: {len(wrapped)} wrapped, MISSING: "
              f"{'; '.join(missing)}", file=sys.stderr)
    return report()


def report() -> dict[str, Any]:
    return {"wrapped": sorted(wrapped), "missing": sorted(missing)}


__all__ = ["install", "report"]
