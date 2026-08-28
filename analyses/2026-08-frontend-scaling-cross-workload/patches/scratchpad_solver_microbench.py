"""Characterize GreedyLayoutSolver.plan_layout in isolation.

The solver takes a plain list of LifetimeBoundBuffer, so its scaling can be
measured without a compile and without device time. That matters because
compiled graphs turn out not to reach the regime where the solver is
expensive: Inductor schedules a buffer's consumer close to its producer, so
the live set stays around 3 no matter how wide the graph is. This separates
two questions that a compile-based sweep conflates:

  1. How does plan_layout scale in buffer count and live-set size?
  2. Do real graphs ever produce a live set large enough for (1) to matter?

Buffers are generated as N intervals of length L staggered by one tick, so
the steady-state live set is ~L and the number of transition ticks is ~N+L.
`pressure` sets each buffer's size as a fraction of (ceiling / L), so
pressure near 1.0 fills the scratchpad exactly and above it forces the
gap-search and spill paths.

Run on a Spyre pod (needs torch_spyre importable) but claims no device:

    python scratchpad_solver_microbench.py --out bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: F401  (must precede torch_spyre)


def make_buffers(cls, n: int, live: int, ceiling: int, pressure: float):
    """N intervals of length `live`, staggered by one tick."""
    size = max(128, int((ceiling / max(live, 1)) * pressure))
    size = (size // 128) * 128
    return [
        cls(name=f"b{i}", size=size, uses=list(range(i, i + live + 1)))
        for i in range(n)
    ]


def time_solve(solver_cls, cls, n, live, ceiling, pressure, reps):
    samples = []
    placed = spilled = ticks = 0
    for _ in range(reps):
        buffers = make_buffers(cls, n, live, ceiling, pressure)
        solver = solver_cls(buffers, ceiling)
        t0 = time.perf_counter_ns()
        result = solver.plan_layout()
        samples.append((time.perf_counter_ns() - t0) / 1e6)
        placed = sum(1 for b in result if b.address is not None)
        spilled = len(result) - placed
        ticks = len({b.start_time for b in buffers} | {b.end_time for b in buffers})
    return {
        "n": n, "live": live, "pressure": pressure,
        "ms": statistics.median(samples), "ms_min": min(samples), "ms_max": max(samples),
        "reps": reps, "placed": placed, "spilled": spilled, "ticks": ticks,
        "n_times_product": n * ticks,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    from torch_spyre._inductor.scratchpad.allocator import _lx_planning_size
    from torch_spyre._inductor.scratchpad.greedy_solver import GreedyLayoutSolver
    from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer

    ceiling = _lx_planning_size()
    print(f"ceiling = {ceiling} bytes ({ceiling / 1024:.1f} KiB)", flush=True)

    rows = []
    # Sweep A -- buffer count at a small fixed live set (the compiled-graph regime).
    for n in (64, 128, 256, 512, 1024, 2048):
        rows.append(time_solve(GreedyLayoutSolver, LifetimeBoundBuffer,
                               n, 3, ceiling, 0.20, args.reps) | {"sweep": "A_count_live3"})
        print(f"  A n={n:5d} live=3   {rows[-1]['ms']:9.3f} ms", flush=True)

    # Sweep B -- live-set size at fixed buffer count (the workload-A regime).
    for live in (2, 4, 8, 16, 32, 64, 128):
        rows.append(time_solve(GreedyLayoutSolver, LifetimeBoundBuffer,
                               512, live, ceiling, 0.50, args.reps) | {"sweep": "B_live_n512"})
        print(f"  B n=  512 live={live:3d} {rows[-1]['ms']:9.3f} ms  "
              f"placed={rows[-1]['placed']} spilled={rows[-1]['spilled']}", flush=True)

    # Sweep C -- ceiling pressure at a large live set (gap search plus spills).
    for pressure in (0.25, 0.5, 0.9, 1.0, 1.5, 3.0):
        rows.append(time_solve(GreedyLayoutSolver, LifetimeBoundBuffer,
                               512, 64, ceiling, pressure, args.reps) | {"sweep": "C_pressure"})
        print(f"  C n=  512 live=64 p={pressure:4.2f} {rows[-1]['ms']:9.3f} ms  "
              f"placed={rows[-1]['placed']} spilled={rows[-1]['spilled']}", flush=True)

    json.dump({"ceiling_bytes": ceiling, "rows": rows}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
