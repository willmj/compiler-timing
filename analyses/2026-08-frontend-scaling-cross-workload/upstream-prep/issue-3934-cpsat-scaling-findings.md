# Draft comment for torch-spyre#3934 — CP-SAT compile-time bounds

**Not posted.** Findings for review; needs your commentary before it goes on the
issue, per the repo AI policy.

---

Measurements against task 1 (profile CP-SAT solve time on larger graphs) and
task 2 (solver blowup vs model-construction overhead), on the SDPA-shaped
workload the co-optimizing sweep currently excludes.

**Setup.** Tiled flash attention (the `test_opspec_tiling.py::test_flash` body,
parameterized on Lk), Lq=512, cold compiles with a wiped Inductor cache.
torch-spyre `7c1d5b6`, torch 2.13.0+cpu. Both arms are the *same tree* with only
`LAYOUT_SOLVER` differing, so this isolates the solver. CP-SAT is
nondeterministic across processes (#4196), so it is n=2 per point; greedy is
n=1 and reproduces to ~1.5%.

## Task 2: it is solver blowup, not model construction

`plan_allocation`'s template splits cleanly along the line the task asks about —
`_solve` is the CP-SAT search, `_prepare_buffers` is the buffer/feature
construction that precedes it.

| input ops | 140 | 276 | 548 | 1092 | 2180 | exponent |
|---|---:|---:|---:|---:|---:|---:|
| cpsat `solve` (ms) | 173 | 324 | 1,609 | 8,998 | **60,409** | **2.13** |
| cpsat `prepare_buffers` (ms) | 309 | 465 | 950 | 2,108 | 5,000 | **1.01** |
| solve share of the pass | 36% | 41% | 63% | 81% | **92%** | |

Construction is flat-linear and is not the problem. The solve is near-quadratic
and reaches 92% of the pass at the largest point. For reference, greedy's own
solve measures exponent 1.98 on the same axis but stays at 511 ms, so the
quadratic shape is shared and it is CP-SAT's constant that makes it bite.

Also worth noting: CP-SAT's `prepare_buffers` is consistently *cheaper* than
greedy's (5,000 ms vs 7,567 ms at the largest point), so nothing in the
model-construction path regressed with the default flip.

## Task 1: the pass cost, and the variance

| input ops | 140 | 276 | 548 | 1092 | 2180 | exponent |
|---|---:|---:|---:|---:|---:|---:|
| greedy `_maybe_scratchpad_planning` (ms) | 534 | 792 | 1,640 | 3,639 | 8,300 | 1.00 |
| cpsat (median of 2) | 2,108 | 2,405 | 3,836 | 12,231 | **67,134** | 1.26 |
| ratio | 4.0x | 3.0x | 2.3x | 3.4x | **8.1x** | |

At the largest point CP-SAT's two runs measure 57,030 and 77,238 ms — a **35%
spread** on identical input, which is consistent with the `PYTHONHASHSEED`
nondeterminism #4196 documents. Any compile-time budget for this path has to be
stated against that spread, not a single run.

## The part that argues *for* the default

Looking only at the pass is misleading. The solver picks a different plan, and
the plan changes what the backend compiles:

| | Lk=512 | Lk=1024 | Lk=2048 | Lk=4096 | Lk=8192 |
|---|---:|---:|---:|---:|---:|
| greedy total compile (s) | 31.6 | 53.4 | 96.1 | 190.9 | 395.0 |
| cpsat total compile (s) | 25.6 | 34.7 | 55.5 | 105.9 | **272.4** |
| | −19.0% | −34.9% | −42.3% | −44.5% | **−31.0%** |

**CP-SAT is faster end to end at every size measured**, by 19–44%. At the
largest point it spends 58.8 s more on planning and saves 180.8 s of
backend-dominated remainder. The default flip is well justified on total compile
time for this workload; the concern is the shape, not the current position.

## Where the bound actually bites

The two terms scale differently. CP-SAT's solve measures exponent 2.13 while the
remainder outside the pre-scheduling pipeline measures 0.68. Extrapolating both
one doubling past the largest measured point (~4,360 ops):

- cpsat total ~628 s, of which **solve alone ~264 s**
- greedy total ~739 s

So CP-SAT is still ahead by ~15% at one more doubling, but the margin has fallen
from 31% and is closing, and solve alone would by then exceed the entire current
CP-SAT compile. The frontend share is already 49% of total compile at Lk=8192
(132 s of 272 s), against 19% for greedy.

That is the argument for task 3's bail-out policy being the right next step
rather than a precaution: the crossover is roughly one to two doublings out on
this workload family, and a `CpSolver` time limit with a deterministic greedy
fallback would bound the worst case without giving up the win at today's sizes.
Landing #4139 (certified greedy seed) would also cut the common case where
greedy already attains the objective bound.

## Reproduction

Instrumentation, workload harness, raw samples and the reduction script are in
an external compiler-timing repository; happy to share or to re-run under
different configurations. Every sample passes the timing-JSON invariant checks
(parent/child containment, `self == inclusive - sum(children)`).

The per-pass split above comes from wrapping `plan_allocation`'s ten template
steps at runtime, so it needs no torch-spyre change to reproduce.
