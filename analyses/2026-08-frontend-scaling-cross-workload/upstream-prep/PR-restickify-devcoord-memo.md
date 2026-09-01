# Draft PR — share one `device_coordinates` memo across layout selection

Branch: `restickify/devcoord-memo` (local only, off `upstream/main` @ `0db21d2`).
Two files, +85/-5. **Not pushed.**

Title: `perf(restickify): share one device_coordinates memo across layout selection`

---

## What type of PR is this?

- [x] cleanup

## What this PR does

`optimize_restickify_locations` decides, for every operation, which tensor
layout each input should have. To score a candidate it has to know whether the
producer's layout and the consumer's layout are stick-compatible, and that
question is answered by `device_coordinates(layout, access, indirect_sizes)`,
which works out where each element physically lands on the device.

That function is pure symbolic algebra and costs roughly **0.8 ms per call** —
expensive because it is sympy all the way down.

There is already a cache for the layer above it. `EdgeCostMap` memoizes
`compute_restickify_needed` per `(in_stl, target_stl)`. But an `EdgeCostMap`
covers **one op-input edge**, so the moment the search moves to the next edge
the cache is gone. On a tiled workload the graph is largely the same computation
repeated per tile, so tile 3 and tile 7 ask about structurally identical layouts
and accesses — different edge, different cache, full recompute.

This adds an optional `cache` argument to `device_coordinates` and
`try_device_coordinates`, following the convention already used by
`_per_core_view_on_buf(..., cache)` and `get_ncores_for_buffers(graph, cache)`.
`compute_restickify_needed` forwards it. `optimize_restickify_locations` owns one
dict, installs it on every edge cost map, and detaches it in a `finally`.

```python
key = (stl, dep, sizes_key)
hit = cache.get(key)
if hit is not None:
    return list(hit)
...
cache[key] = coords
return list(coords)
```

Measured on tiled flash attention, Lq=512, cold compiles, matched arms:

| point   | before (ms) | after (ms) | delta |
|---------|------------:|-----------:|------:|
| Lk=2048 |        4946 |       3544 | **-28.3%** |
| Lk=8192 |       44628 |      38459 | **-13.8%** |

## How this was found

Working backwards from measurement each time, and discarding two hypotheses
before arriving here.

**1. The standing hypothesis was wrong.** A prior static audit flagged
`state.assignments + (candidate_stl,)` — a tuple concatenation inside the beam
loop — as the likely cost. Instrumenting it says otherwise. Across a doubling of
the graph, time **per state** is flat (50.3 -> 47.2 us) while time **per
assignment slot copied** halves (310.7 -> 161.5 ns). If copying drove the cost,
per-slot cost would hold constant and total time would track slot volume, which
grew 5.43x against the pass's 2.82x. It tracks `n_states_built` (3.01x) instead.
Reconstructing what a live-slot-only state would carry showed a 10.8-12.4x
volume reduction available, i.e. a large memory win and almost no time win. Not
pursued.

**2. The superlinearity is a saturation transient, not a growth law.**
`trim_states_in_max` is pinned at 600 at both sizes, so the beam frontier is
capped and is not growing. What grows is the *fraction of ops running at the
cap*: 96.8 -> 147.6 states per op, 16% -> 25% of the ceiling. The measured
exponent of ~1.5 is the approach to a linear asymptote of ~600 states per op.
So the lever is the constant, not the exponent — which is what redirected the
search toward per-call cost.

**3. A cProfile of one beam relocated the cost entirely.**

| function | calls | self time | per call |
|---|---:|---:|---:|
| `compute_restickify_needed` | 1,289 | 2.57 s | 2.0 ms |
| `device_coordinates` | 2,578 | 2.01 s | 0.78 ms |
| every expansion-loop entry | millions | <= 0.28 s | — |

Two functions, called a few thousand times, dominate; the loop bookkeeping that
the audit suspected is negligible. The sympy machinery beneath them
(`sympify`, `free_symbols`, `is_ge`) is where the time actually goes.

**4. Counting distinct arguments decided where the cache belongs.** For each of
the two, total calls versus distinct argument sets:

| function | calls | distinct | reuse |
|---|---:|---:|---:|
| `compute_restickify_needed` | 2,577 | 2,577 | **1.00x** |
| `device_coordinates` | 5,154 | 1,126 | **4.58x** |

`compute_restickify_needed` has **zero** cross-edge repetition — `EdgeCostMap`
is already tight, and widening it would gain nothing. One level down, 78% of
calls recompute a known result, and the ratio is stable across graph sizes, so
it is a constant-factor win that holds as graphs grow. That is the whole basis
for this change.

## Special notes for your reviewer

**Lifetime is the load-bearing design choice.** `finalize_layouts` runs in a
later pass and calls `EdgeCostMap.cost()` again, by which point committed
layouts have changed and cached coordinates would be stale. The memo is
therefore detached when layout selection ends rather than left reachable from
the edge maps. Within the search, candidate layouts are only read. This follows
a specific prior lesson: a naive global memo on a dependency-derived helper in
the coarse-tile work broke correctness by outliving the mutations that
invalidated it.

**Aliasing.** A fresh list is returned on every call, so no caller can mutate
another's result through the cache. The lists hold one expression per device
dimension, so the copy is a few elements against 0.8 ms of avoided sympy.

**The raising path is unchanged.** Unsupported stick expressions raise in
`_check_stick_expr_supported` before the store, so they are never cached.

**With no cache supplied, behaviour is byte-identical.** Every other caller of
these functions passes nothing and takes the same path plus one `is not None`
check.

**What this does not do.** The search makes exactly the same decisions — in a
runtime-shim prototype of the same change, `n_states_built`, `cost_calls` and
`device_coordinates` call counts were identical between arms. It skips
recomputation, it does not prune. So it does not change how the pass scales;
that would mean reducing states per op, which is a search-quality question.

**Why no pipeline or whole-compile numbers are quoted.** On the substrate used
for these runs, `_maybe_scratchpad_planning` varies between 55 s and 72 s run to
run at Lk=8192, which swamps both totals. Every other pre-scheduling pass is
flat between arms to within a few ms, so the pass-level delta is the honest
claim and the totals are not.

## Does this PR introduce a user-facing change?

No. Layout selection produces the same result; only the time to compute it
changes.

## Additional note

<!-- AI disclosure, per the repository AI policy. -->
This change was prepared with the assistance of Claude Code (Opus 5), which did
the instrumentation, profiling, distinct-argument counting and A/B measurement,
and wrote the patch. A reviewer should independently satisfy themselves on the
two claims that carry the correctness argument: that the cache cannot be read
after layout selection, and that returning a copy removes the aliasing risk.
