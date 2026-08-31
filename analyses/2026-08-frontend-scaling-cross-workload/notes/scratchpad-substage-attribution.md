# `_maybe_scratchpad_planning` — substage attribution

Closes the pass to 100% of its named steps, characterizes the one superlinear
term, and confirms every other pre-scheduling pass is linear. Answers
opportunity #4 in [`engineering-opportunities.md`](engineering-opportunities.md)
as far as this workload family can, and narrows where workload A's reported
n^1.45 can live.

**Substrate.** torch-spyre `faff191`, torch 2.13.0+cpu, pod `mwj-spyre-dev-pf`.
`layout_solver=greedy`, `co_optimizing_lx_planning=False`, `lx_planning=True`,
`lx_planner_relayout=True`, `sencores=32` — identical to the pr3806 study's
resolved config. Data in
[`../data/scratchpad-substage-validation/`](../data/scratchpad-substage-validation/).

**n=1 per point, no warmup discard.** This is a characterization exercise, not a
committed dataset. Every sample passes `check_timing_json.py`.

## Instrumentation

Three runtime shims, no torch-spyre source modification. Every boundary is a
method call on `self` or a module-global in `allocator.py`, so class-level wraps
suffice; that survives the tree drift a source-anchored patch could not.

| patch | what it adds |
|---|---|
| [`scratchpad_substage_timing.py`](../patches/scratchpad_substage_timing.py) | `plan_allocation`'s 10 template steps (level 1), prepare-side helpers and solver boundaries (level 2), per-buffer accumulators (level 3). Plus the residency histogram and placement rate. |
| [`pass_pipeline_timing.py`](../patches/pass_pipeline_timing.py) | `pipeline:*` and `pass:*` events for all six pipelines, matching the in-tree patch's event names. Preserves `_pass_sources` so the Inductor cache key is unchanged. |
| [`scratchpad_solver_microbench.py`](../patches/scratchpad_solver_microbench.py) | Drives `GreedyLayoutSolver.plan_layout` on synthetic buffer sets. No compile, no device. |

Level 2 -> 3 costs 0.3%, and event count stays flat rather than growing with
buffer count. Level 1 versus uninstrumented is not cleanly separable at n=1
(wall-clock noise is +/-18%).

## 1. The pass is 99% eligibility, 1% packing

Chain workload, dim=128, at 768 input operations. Residual 0.02%.

| step | ms | % of pass |
|---|---:|---:|
| `plan_allocation` | 1343.62 | 100 |
| `prepare_buffers` | ~1330 | 98.7 |
| — `collect_lx_relayout_plans` | ~39% of pass | |
| — `generate_buffers` -> `residency_reasons` | ~35% of pass | |
| `solve` | ~1% | 1.0 |

Deciding *who may reside* costs two orders of magnitude more than deciding
*where*. A static reading of the code predicts the opposite, and that prediction
(committed before measurement) was wrong.

## 2. `solve` is exactly quadratic, but with a small constant

Driven directly, live set fixed at 3, sizes at 20% pressure:

| N buffers | ticks | N x ticks | ms | ratio |
|---:|---:|---:|---:|---:|
| 64 | 68 | 4,352 | 0.429 | — |
| 128 | 132 | 16,896 | 1.312 | 3.06x |
| 256 | 260 | 66,560 | 4.622 | 3.52x |
| 512 | 516 | 264,192 | 18.053 | 3.91x |
| 1024 | 1,028 | 1,052,672 | 71.491 | 3.96x |
| 2048 | 2,052 | 4,202,496 | 283.672 | **3.97x** |

Converges on 4.00x per doubling: `Theta(N^2)`, tracking `N x ticks` exactly. The
driver is the doubly-nested rescan of every placeable buffer at every transition
tick in `plan_layout` — that loop is inline and cannot be wrapped, so it appears
as the residual under the per-buffer accumulators.

**Two hypotheses refuted by the controls.**

- *Live-set size barely matters.* 64x more overlap (live 2 -> 128 at N=512) costs
  1.85x. So `_try_allocate`'s `list(self.usage)` copy and gap search are not the
  driver, and the "long-lived carry buffers make the solver expensive" theory is
  wrong.
- *Ceiling pressure barely matters*, and is non-monotonic: 25 -> 31 ms peaking at
  exactly p=1.0, then falling as heavier pressure spills more and leaves less to
  place.

**Not currently actionable.** Extrapolating, N=4096 is ~1.1 s. Real graphs in
this family reach 128–770 buffers, where solve is single-digit milliseconds.
Quadratic with a small constant, well below the regime that matters.

## 3. Every other pre-scheduling pass is linear

48 -> 768 input operations (16x), each pass against its own entry size:

| pass | ms @ 48 | ms @ 768 | exponent |
|---|---:|---:|---:|
| `optimize_restickify_locations` | 177.67 | 2908.23 | 1.01 |
| `span_reduction` | 157.58 | 2515.33 | 1.00 |
| `_distribute_work` | 109.89 | 1792.67 | 1.01 |
| `propagate_spyre_tensor_layouts` | 111.73 | 1632.93 | 0.97 |
| `_maybe_scratchpad_planning` | 74.37 | 1343.62 | 1.04 |
| `insert_restickify_padding` | 13.28 | 207.04 | 0.99 |
| `enforce_indirect_access_layout` | 10.79 | 177.99 | 1.01 |
| `deadcode_elimination` | 13.81 | 156.75 | 0.88 |
| `validate_ops` | 6.57 | 99.04 | 0.98 |
| `split_multi_ops` / `insert_bmm_padding` | | | 0.93 / 0.92 |

All 0.88–1.04. Note `optimize_restickify_locations` measures 1.01 here against
1.46 in the pr3806 study, and `_maybe_scratchpad_planning` 1.04 against 1.45 —
those studies' superlinear exponents are a property of flash-attention graph
structure, not of the passes in general.

**One false alarm, recorded.** `deadcode_elimination` computed as 1.32 from the
48 -> 384 endpoints and reproduced across three runs at 384 ops (213, 184,
201 ms) — then *dropped* to 157 ms at 768 ops. Its cost tracks how much dead
code exists, which is not a function of graph size. Single-sample endpoint
exponents are fragile exactly as the pr3806 limitations section warns.

## 4. Relayout planning: 39% of the pass, no plans accepted *on these workloads*

> Scoped result. On workload A relayout does accept plans -- see §7.

`collect_lx_relayout_plans` searches producer/consumer edges for per-core
ownership disagreements it can fix with an inserted copy. On these workloads it
finds none, and `finalize_lx_relayout` measures ~0 ms.

A/B at 194 buffers with `SPYRE_LX_PLANNER_RELAYOUT=0`: **336.85 -> 214.70 ms
(-36.3%)**, with `residency_reasons` (118.02 -> 120.26) and `solve` (3.48 ->
3.46) unchanged and output identical. A cleanly isolated effect.

The cost is structural: `_prepare_buffers` reaches the search through the
`supports_paired_buffers` branch, which `greedy` always takes, and the config
check lives inside `collect_lx_relayout_plans` rather than gating the branch. The
expensive inner test, `_compatible_partitions`, compares every source core slice
against every destination core slice — quadratic in `sencores` (32 by default),
per candidate.

**Candidate change:** a cheap necessary-condition pre-filter before the
per-candidate gauntlet (does *any* buffer have two users that disagree about core
division at all?). Not yet prototyped. Note this trade is only unfavorable on
workloads where no plan survives; a workload where relayout fires pays the search
once and saves a reshuffle on every read. No such workload has been measured.

## 5. Efficacy on the synthetic workloads: two thirds of buffers reside

> Scoped result. Workload A's exclusion profile is different -- see §7.

The pass had no aggregate efficacy metric — only per-buffer debug logs. Added
one. At 768 operations:

```
eligible 512   barred 256           placed 512   spilled 258
   128  core div mismatch: K-split writer
   127  core div mismatch: broadcast read
     1  no consumer reads it from LX
```

One gate accounts for 99.6% of exclusions, and it splits almost exactly in half.
Neither half is an allocator-level opportunity:

- **K-split writer (128).** A split-K reduction leaves partial sums on most
  cores; only the k-last cores hold the final value. The data genuinely is not
  there to read. Any remedy is a work-division decision in `_distribute_work`,
  not a scratchpad one.
- **Broadcast read (127).** A consumer splits an iteration axis the buffer does
  not have, so its view covers fewer cores than the op runs, and cores without a
  local copy would read stale scratchpad. Blocked on a missing capability, which
  the source states plainly: there is no single-base LX broadcast.

So the gate is correctly refusing to place data that either does not exist or
cannot be addressed. The only direction here is a *feature* — replicated
residency, placing a shared buffer whole into every core's private LX at `size`
rather than `size/ncores` per core. That suits a broadcast-read buffer's shape
(shared, therefore usually small), and needs both allocator footprint modelling
and codegen support for a full-buffer copy-in. Out of scope for a timing study;
recorded because the histogram is what surfaced it.

## 6. Workload A: the n^1.45 is gone

The earlier sections use synthetic workloads. To close the actual open question
this study left -- opportunity #4, workload A's unattributed n^1.45 scratchpad
scaling -- `patches/workload_harness_flash.py` parameterizes
`tests/inductor/test_opspec_tiling.py::test_flash` (whose hardcoded
Lq=512/Lk=1024 *is* workload A's baseline) and sweeps Lk over the study's own
range.

**The reproduction is structurally exact.** At the baseline point,
`dedup_and_promote_constants` sees 276 input operations and
`_maybe_scratchpad_planning` sees 260 -- identical to the study's published
values -- and dedup measures 866.8 ms against the study's 870 ms (0.4%). Same
graph, same config.

`_maybe_scratchpad_planning`, this tree versus the pr3806 study, 140 -> 2180
input operations:

| input ops | study (ms) | this tree (ms) | ratio |
|---:|---:|---:|---:|
| 140 | 434 | 245.9 | 1.8x |
| 276 | 959 | 463.5 | 2.1x |
| 548 | 2,403 | 934.3 | 2.6x |
| 1,092 | 6,722 | 2,051.7 | 3.3x |
| 2,180 | **21,037** | **5,088.2** | **4.1x** |

| | study | this tree |
|---|---:|---:|
| endpoint exponent | **1.41** | **1.10** |
| per-operation drift | 3.4x | 1.33x |

The improvement *grows* with graph size, which is a scaling-law change rather
than a constant-factor speedup. Repeatable: endpoints re-run at 242.7 ms
(vs 245.9) and 5165.6 ms (vs 5088.2), 1.3-1.5%.

**Opportunity #4 is closed, by tree drift rather than by a deliberate fix** as
far as the notes record. Which commit did it is not determined here -- that
needs the study's tree (`a9316b381`), which is not in this clone. The candidates
are the scratchpad commits since: #3793 (stateless allocator), #3363 and #3849
(native C++ packer), #3926 (LX relayout guards), #3375 (allocator unification).

### The other two superlinear passes did not improve

Same sweep, same tree:

| pass | study exp | this tree exp | ms @ 2180 ops | study ms | status |
|---|---:|---:|---:|---:|---|
| `dedup_and_promote_constants` | 1.96 | **2.02** | 57,343 | 54,646 | PR #4113 in flight |
| `optimize_restickify_locations` | 1.46 | **1.57** | 39,062 | 39,475 | **nothing in flight** |
| `_maybe_scratchpad_planning` | 1.45 | 1.10 | 5,088 | 21,037 | effectively linear |

Every other pass measures 0.99-1.01 with per-operation cost flat to within 3%.

**This reorders the priority list.** `optimize_restickify_locations` is
unchanged from the study (39.1 s versus 39.5 s at the top point) and its
exponent has not moved. Once #4113 lands, it becomes the dominant frontend
scaling problem on this workload family -- and it is the one entry in
`engineering-opportunities.md` with no prototype and an unattributed mechanism.

### The per-buffer hypothesis was right as a mechanism, wrong as the driver

The prediction going in was that per-buffer gate cost rises with expression
complexity. It does: `residency_reasons` measures exponent 1.28 with per-operation
cost drifting **2.15x** across the sweep, while every whole-pass helper around it
stays flat. So the mechanism is real and is what remains of the pass's residual
1.10. It is simply no longer large enough to produce a 1.45.

## 7. Two claims from the synthetic workloads that do not generalize

Recorded because both were stated more broadly than the evidence supported.

**Relayout does find plans on workload A.** `finalize_lx_relayout` grows 0.19 ->
2.50 ms and `append_lx_relayout_destinations` 0.18 -> 20.57 ms across the sweep;
`_finalize_lx_relayout_allocation` early-returns on an empty plan list, so
non-zero time means plans exist and are being accepted. The "39% of the pass and
finds nothing" result in §4 is specific to the synthetic workloads. The
pre-filter idea survives -- it would skip the search only where there is nothing
to find -- but the framing that the search is wasted does not. Relayout is still
32% of the pass at the largest workload-A point (1,638 ms of 5,088 ms).

**The exclusion profile is workload-specific.** §5 found one gate causing 99.6%
of exclusions, split between K-split writer and broadcast read. On workload A the
dominant gate is instead `op not allowed` -- the 19-entry allowlist -- at 960 of
961 barred buffers at the largest point, with 960 of 2,184 buffers placed (44%).
The histogram is doing its job; the conclusion drawn from one workload was not
transferable.

## Incidental

`CustomPreGradPasses` now emits events. `spec/spyre-timeit/SPEC.md` §11 lists it
as an open question — in the stage map but absent from all 93 committed samples.
It exists and runs; the earlier instrumentation did not reach it. Also,
`pass:CustomPostPasses:apply` still collides across two distinct passes
(`mm_to_bmm_pass.apply`, `bmm_unflatten_pass.apply`); event names are kept
as-is for reduce compatibility and a `pass_index` meta field disambiguates.

The residency histogram covers op-produced buffers only. Graph inputs go through
`_input_residency_reason`, which is not wrapped — that is the small
`n_barred` versus `n_spilled` discrepancy in the smaller samples.

## 8. `optimize_restickify_locations` — the bookkeeping is the pass

Follow-on dive, same harness and sweep. Instrument:
[`restickify_beam_timing.py`](../patches/restickify_beam_timing.py), which wraps
`beam_global_min_cost`, the two once-per-pass precomputations, `BeamState.__init__`
and the three `cost` implementations. Wrapping `BeamState.__init__` is what yields
the load-bearing counter: every expansion builds a state from a freshly
concatenated tuple, so the length passed in *is* the per-expansion copy volume.

Its wrappers fire once per expansion, so overhead is material and was measured
rather than assumed: 15-24%, by running the same points with the shim gated off
(`SPYRE_RESTICKIFY_TIMING=0`). Every time below is from the **shim-off** run, with
only the negligible once-per-pass components and the bracketed `cost` time taken
from the shim-on run. Bookkeeping is the derived residual.

| ms (shim off) | 512 | 1024 | 2048 | 4096 | 8192 | exp | exp(1024+) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **pass total** | 518.9 | 1365.5 | 3884.9 | 11918.7 | **38852.8** | 1.57 | 1.62 |
| `future_min_cost` (DP) | 466.4 | 933.2 | 1865.2 | 3914.7 | 7532.0 | **1.01** | 1.01 |
| `cost_fn.cost` | 58.3 | 351.2 | 1021.1 | 2643.8 | 6047.0 | 1.69 | 1.38 |
| `_reorder_any_in_nodes` | 4.3 | 12.4 | 39.8 | 176.4 | 682.0 | 1.85 | 1.94 |
| `_compute_last_use` | 0.1 | 0.1 | 0.2 | 0.6 | 1.9 | — | — |
| **bookkeeping** (derived) | ~0 | 68.7 | 958.5 | 5183.2 | **24589.9** | — | **2.85** |
| bookkeeping share | — | 5.0% | 24.7% | 43.5% | **63.3%** | | |

**The pass is bookkeeping.** At the smallest point it is ~90% backward-DP and
bookkeeping is nil; by the largest it is 63% bookkeeping growing at exponent
2.85, and that crossover *is* the pass's 1.57.

### Mechanism, confirmed

`BeamState.assignments` is a tuple parallel to the frontier's buf_names, so its
length is the index of the op being processed. Three places pay O(index) per
state per op in `beam_global_min_cost`:

1. **expansion** -- `state.assignments + (candidate_stl,)`, a full copy per
   state x candidate;
2. **merge key** -- a second full-length tuple per expanded state, then hashed
   into the `canon` dict (so it also pays `SpyreTensorLayout.__hash__` per slot);
3. **`live_indices`** -- a scan of all buf_names, once per op.

Predicted before measuring (in the patch docstring): cost tracks
`sum over expansions of len(assignments)`, not the expansion count. It does:

| | 512 | 1024 | 2048 | 4096 | 8192 | exp |
|---|---:|---:|---:|---:|---:|---:|
| expansions | 4,700 | 27,090 | 81,462 | 190,262 | 407,862 | 1.63 |
| copy volume (elements) | 439,022 | 4,385,625 | 23,825,377 | 103,838,113 | 428,369,185 | **2.51** |
| ns per expansion | — | 2,536 | 11,766 | 27,242 | 60,290 | rises 24x |
| ns per copied element | — | 15.7 | 40.2 | 49.9 | 57.4 | rises 3.7x |

Per-expansion cost rises 24x; per-element cost rises 3.7x. So volume is the
primary driver and the expansion-count model is dead -- but volume alone does not
fully explain it either. The residual 3.7x is most likely allocation and GC
pressure (428M tuple slots churned at the largest point) plus the layout hashing
in the merge key. Attributing that split needs in-function timers, which is the
one thing this indirect instrumentation cannot do.

### Two hypotheses refuted

- **The DP is linear (1.01), not the problem.** `compute_future_min_cost` has four
  nested loops and its `min_input_cost` call count is proportional to total
  consumer edges, so fan-out growth in a tiled graph looked like a candidate. It
  is not: 1.01 across the whole range. It does dominate *small* graphs (90% at
  Lk=512), which is worth knowing but is not a scaling problem.
- **Trimming and cost evaluation are not the problem.** `Frontier.trim` is 1.5 ms
  total at the baseline; `cost_fn.cost` is 15.6% at the largest point with
  exponent 1.38. The beam is saturated from Lk=1024 on (600 states in, 200 out),
  so the state count is bounded and only the per-state work grows.

### The fix, and the one number that sizes it

**During the search only live slots are ever read.** `Frontier.input_stl` is
called only for deps of the op being processed, which are live by definition, and
the merge key explicitly nulls dead slots. Dead slots matter only for the final
commit. So a state needs a small live-slot map, copied at O(live), plus a parent
pointer for history walked once at the end -- turning both the expansion copy and
the merge key from O(index) to O(live).

Projected, if the live-slot count is bounded: bookkeeping collapses and the pass
at Lk=8192 goes from 38.9 s to roughly DP 7.5 + cost 6.0 + reorder 0.7 + small,
around 15 s, with the exponent falling from 1.62 toward ~1.1.

**That projection is not measured.** `len(live_indices)` is computed inline and
this instrumentation cannot see it, so the win could be much smaller if liveness
in a tiled flash-attention graph is broad. Counting it is a small anchored source
patch plus two runs, and it is the next thing to measure -- before any prototype.

Secondary: `_reorder_any_in_nodes` is exponent 1.94 at 682 ms (1.8%). Small now,
quadratic, worth a look once the main term is gone.
