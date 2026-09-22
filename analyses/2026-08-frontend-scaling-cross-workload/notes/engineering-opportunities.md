# Frontend engineering opportunities — ranked

Guiding principle: **measured impact outranks scary-looking source
complexity.** Every candidate is labeled with evidence level and
whether a prototype has actually demonstrated the fix.

## Legend

- **Evidence**: measured (data on this tree) / source-level (from
  static analysis) / reported (external observation only).
- **Prototype**: measured (working prototype with numbers) / estimated
  (paper analysis with no working prototype).
- **Generality**: cross-workload / workload A only / workload B only.
- **Engineering risk**: low / medium / high — informed by whether prior
  naïve attempts failed for a specific reason.

## Table

| # | Opportunity | Affected path | Workloads | Absolute cost | Observed scaling | Source-level mechanism | Evidence | Proposed change | Expected leverage | Prototype | Risk | Next validation |
|:-:|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 1 | Per-substage reverse-adjacency in `_maybe_coarse_tile_hints` | `wsr/coarse_tile.py:_plan_tiling_propagation` (22%), `wsr/coarse_tile.py:_patch_retiled_load_indexes` (74%) | B (dominant); A (present but far smaller) | 14 s at n=8; 53 s at n=16 (workload B) | ~4× per 2× chunks (near-quadratic) | Per-op `_reads_buffer(op, buf)` scan calls raw `op.get_read_writes()`, run O(N × K) times per substage. Same driver as opportunity #3. | measured | Build one O(N) `readers_by_buffer` + `reads_by_op` per substage from a snapshot of `operations`; scope to substage lifetime so mutations across substage boundaries cannot leave stale entries. | **measured**: `_maybe_coarse_tile_hints` 4.11 s → 1.40 s at n=4 (2.93× speedup); 14.46 s → 3.93 s at n=8 (3.68× speedup). Total Spyre pipes at n=8: 23.3 s → 12.8 s. Growth ratio 4→8 shifts from 3.52× (near-quadratic) to 2.80× (approaching linear). | see `notes/coarse-tile-prototype.md` | medium — naïve global `op_read_writes` swap broke correctness (prototypes.md); per-substage scoping avoids that failure mode | see `coarse-tile-prototype.md` |
| 2 | ~~`_extern_kernel_in_live_range` prefix-sum~~ **REFUTED** | `scratchpad/allocator.py:122` | — | — | — | Static audit predicted this was the driver of workload A's n^1.45 scratchpad scaling; measurement disagrees. | **measured null** — patched scratchpad time was within 1–2% of baseline at both 512×4096 and 512×8192 (see `notes/scratchpad-prototype.md`) | no change to this function | none — measurement refutes the hypothesis | patch preserved as no-op micro-optimization only; not a workload-A fix | scratchpad's real n^1.45 driver is still unattributed | n/a (moves to "needs more investigation") | Add substage instrumentation *inside* `plan_allocation` before proposing another scratchpad prototype |
| 3 | Dedup reverse-adjacency / avoid full graph rescans per duplicate | `_inductor/dedup_constants.py::_redirect_consumers` + `_drop_constant` | cross-workload | 10 s at workload B n=16; up to 225 s at workload A 1024×8192 (b=128 preliminary) | `ops × dups` — near-quadratic when dups ∝ ops | For each duplicate: walks `operations` in `_redirect_consumers` and calls `operations.remove(dup)` (O(N) list scan). Same uncached `get_read_writes` mechanism as #1. | measured | Build a single `consumers_by_buf` and `op_to_position` at pass entry; skip full-graph scan per duplicate; batch-remove instead of `list.remove` per element. | estimated: `ops × dups` shape stays, per-pair constant should drop by the same 4.6× factor that separates workload A's 202 µs/pair from workload B's 931 µs/pair | not built | low | prototype: measure workload A 512×4096 and workload B n=8 |
| 4 | `operations.index(op)` → local `op_to_position` dict in mutating paths | `pass_utils.py:1342` (`replace_computed_buffer_body`); also insert_restickify, split_multi_ops | cross-workload | Bundled with #1 in coarse-tile-hints; visible in restickify insert cost too | O(N) linear scan per splice, called O(K) times per mutating pass | Pattern is already used at `coarse_tile.py:1480` for `op_to_position`; extend to `replace_computed_buffer_body`. | source-level | Add a local `op_to_position` dict maintained across mutations in the same pass; `list.index` → `dict.get`. | estimated: 5–15% off `_patch_retiled_load_indexes` on top of #1; possibly larger benefit on insert_restickify | not built | low | prototype together with #1 or #3 |
| 5 | Restickify: share one `device_coordinates` memo across layout selection | `_inductor/optimize_restickify.py` (`device_coordinates`, `EdgeCostMap`) | B primarily | 2.3 s at n=8; 5.6 s at n=16 (workload B) | ~2.2–2.4× per doubling in the post-fix range | `device_coordinates(layout, access, indirect_sizes)` is pure sympy at ~0.8 ms a call. `EdgeCostMap` memoizes the layer above it, but one map covers **one op-input edge**, so the moment the search moves to the next edge the cache is gone. A tiled workload is largely the same computation per tile, so structurally identical layouts recompute on every edge. **Supersedes the earlier tuple-concatenation hypothesis** (`state.assignments + (candidate_stl,)`), which measurement refuted: the superlinearity is a beam-saturation transient, not tuple copies, and the beam's 200 states buy nothing on workload A. | measured (three-arm attribution; 63% of the pass is bookkeeping at the largest point) | Optional `cache` argument on `device_coordinates` / `try_device_coordinates`, forwarded by `compute_restickify_needed`; one dict owned by `optimize_restickify_locations`, installed on every edge cost map and detached in a `finally`. Follows the existing `_per_core_view_on_buf(..., cache)` convention. | measured on the prototype; see `upstream-prep/PR-restickify-devcoord-memo.md` | built — open as torch-spyre#4245 | low — a pure memo on a referentially transparent function; the beam itself is untouched | none outstanding; awaiting review on #4245 |
| 6 | `dxp_standalone` backend growth | `torch_spyre/execution/async_compile.py:sdsc` — external subprocess | cross-workload | Dominant at scale: 217 s (workload B n=16); >2000 s (workload A largest points) | strongly superlinear | External backend; separate ownership | reported / observed | Backend-team owned | out of frontend scope | separate ownership | separate ownership | separate ownership |
| 7 | `_maybe_reorder_unhinted_interlopers` explicit `O(n²)` docstring | `wsr/coarse_tile_hints.py:reorder_unhinted_interlopers` | either | 0.3–1 ms across all measured points | source says O(n²), measured negligible | Small graphs in practice — the O(n²) admission holds but at these graph sizes the constant is dominant. | measured (as a null: not a real hotspot) | do nothing | none — measured impact refutes the source concern | n/a | none | n/a |
| 8 | Compile-side extent scaling reported in PR #3812 docstring | (unknown — does not reproduce on our base) | B (in the full PR #3812 tree) | reported >2 h at Lq=8192, n=4; on our base <60 s across Lq 64→4096 | reported extreme; not observed here | Unknown — the code path either lives in PR #3812's additional 22 files (including 820 lines of new C++), or requires state our controlled base does not build. | reported only | Build a full pr3812 tree with C-extension, replay the same sweep, isolate. | unknown | not investigated | n/a | build pr3812 and rerun the Lq/Lk sweeps at fixed n_chunks |

## What we would recommend the team do first

1. Take **opportunity #1** (coarse-tile reverse adjacency) into a proper
   engineering PR. Prototype scoping is validated; the measurement
   shows a 2.93×/3.68× reduction on the umbrella pass at n_chunks=4/8
   and shifts the 4→8 growth ratio from 3.52× toward 2.80×. This is
   the clearest measured win in the study.
2. Address **opportunity #3** (dedup reverse adjacency) next: same
   uncached-`get_read_writes` mechanism as #1, different pass, benefits
   both workloads. Not yet prototyped.
3. Investigate the **real n^1.45 scratchpad driver** on workload A. The
   `_extern_kernel_in_live_range` prefix-sum hypothesis was measured to
   be within noise (see `notes/scratchpad-prototype.md`). Substage
   instrumentation inside `plan_allocation` is the right next step —
   the layout solver internals or per-buffer allocation overhead are
   likely candidates, but that has not been measured.
4. Investigate **opportunity #5** (post-fix restickify ~2.2–2.4× per
   doubling) with a diagnostic profile before proposing a change.
5. Consider **opportunity #4** (`operations.index` → `op_to_position`)
   as a bundled improvement alongside #1 or #3; not standalone-worth.
6. Leave **opportunity #8** (extent-scaling repro on full pr3812) as
   "worth investigating if a full pr3812 build becomes convenient."
   Not blocking anything.

## Lesson from the measured null

The scratchpad prototype demonstrates why the "measured impact
outranks scary-looking source complexity" principle matters. The
static-audit hypothesis was source-plausible (documented
`range(min, max+1)` per-buffer scan) and even had a cross-workload
comparison suggesting scratchpad topology-dependency mattered — but
the isinstance check inside the scan is not the hot inner loop.
The n^1.45 slope is real; the cause is elsewhere in
`plan_allocation`.

**Future opportunity entries should be labeled "measured" or
"estimated" prominently, and estimated entries should not be
implemented without the measurement first.**
