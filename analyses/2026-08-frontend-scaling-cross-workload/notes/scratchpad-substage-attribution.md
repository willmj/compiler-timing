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

## 4. Relayout planning: 39% of the pass, zero plans accepted

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

## 5. Efficacy: two thirds of buffers reside, and the exclusions are not wins

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

## What this does not resolve

**Workload A's n^1.45 is still unexplained**, and this narrows it rather than
answering it. The solver is quadratic but ~1.1 s at N=4096, roughly 1.5% of the
74 s the pr3806 study measured at that scale; and prepare is linear here. So
neither half of the pass, as measured on this family, produces that slope.

The remaining hypothesis: workload A's *per-buffer* gate cost is not constant.
Several gates call sympy-backed helpers (`op_read_writes`,
`_per_core_view_on_buf`, `try_device_coordinates`,
`_would_produce_lx_back_gap`) whose per-call cost grows with expression
complexity, which itself grows with graph size in a tiled flash-attention graph.
This is the same mechanism the dedup study found inflating its per-pair constant
4.6x between workloads. Both the per-pass timers and the histogram work on any
workload, so a workload-A run would show it directly as per-buffer gate cost
rising with graph size. That is the next measurement.

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
