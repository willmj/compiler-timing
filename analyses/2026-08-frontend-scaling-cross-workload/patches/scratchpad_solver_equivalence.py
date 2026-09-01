"""Dump GreedyLayoutSolver plans for a fixed scenario set.

Run once against a pristine tree and once against a patched one, then diff
the two JSON files: the tick-bucketing change is meant to be exactly
order-preserving, so every buffer's assigned address must be identical, not
merely as good.

Scenarios deliberately include the cases the time walk treats specially --
in-place parents, paired groups, coincident start/end ticks, buffers barred
by residency_reason, and enough ceiling pressure to force spills -- since a
reordering bug would surface as a different address only when the greedy
choice is contested.
"""

from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: F401  (must precede torch_spyre)


def scenarios(cls, ceiling):
    out = {}

    # 1. Uniform staggered intervals: the shape the microbench sweeps.
    out["uniform"] = [
        cls(name=f"u{i}", size=8192, uses=list(range(i, i + 5)))
        for i in range(120)
    ]

    # 2. Random lifetimes and sizes, seeded.
    rng = random.Random(1234)
    bufs = []
    for i in range(200):
        start = rng.randrange(0, 300)
        length = rng.choice([1, 2, 3, 8, 40, 120])
        size = rng.choice([128, 1024, 16384, 65536, 262144])
        bufs.append(cls(name=f"r{i}", size=size, uses=list(range(start, start + length + 1))))
    out["random"] = bufs

    # 3. Coincident ticks: many buffers sharing exact start and end.
    out["coincident"] = [
        cls(name=f"c{i}", size=4096, uses=[10, 20]) for i in range(60)
    ] + [cls(name=f"d{i}", size=4096, uses=[20, 30]) for i in range(60)]

    # 4. In-place chains: each child claims its parent's slot.
    chain = []
    for i in range(80):
        b = cls(name=f"p{i}", size=32768, uses=[i, i + 1])
        if i:
            b.in_place_parents = [f"p{i - 1}"]
        chain.append(b)
    out["in_place"] = chain

    # 5. Paired groups: relayout source plus destinations, placed atomically.
    paired = []
    for g in range(20):
        root = cls(name=f"g{g}s", size=16384, uses=list(range(g, g + 6)))
        dests = [
            cls(name=f"g{g}d{k}", size=16384, uses=list(range(g + 1, g + 5)))
            for k in range(2)
        ]
        root.paired_with = list(dests)
        paired.append(root)
        paired.extend(dests)
    out["paired"] = paired

    # 6. Heavy pressure: total live footprint far exceeds the ceiling.
    out["pressure"] = [
        cls(name=f"h{i}", size=ceiling // 4, uses=list(range(i, i + 30)))
        for i in range(90)
    ]

    # 7. Barred buffers interleaved with placeable ones.
    mixed = []
    for i in range(100):
        b = cls(name=f"m{i}", size=8192, uses=list(range(i, i + 4)))
        if i % 3 == 0:
            b.residency_reason = "op not allowed"
        mixed.append(b)
    out["mixed_barred"] = mixed

    # 8. Degenerate: nothing placeable.
    allbarred = []
    for i in range(10):
        b = cls(name=f"b{i}", size=8192, uses=[i, i + 1])
        b.residency_reason = "op not allowed"
        allbarred.append(b)
    out["all_barred"] = allbarred

    return out


def main() -> int:
    out_path = sys.argv[1]

    from torch_spyre._inductor.scratchpad.allocator import _lx_planning_size
    from torch_spyre._inductor.scratchpad.greedy_solver import GreedyLayoutSolver
    from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer

    ceiling = _lx_planning_size()
    result = {"ceiling": ceiling, "scenarios": {}}

    for name, buffers in scenarios(LifetimeBoundBuffer, ceiling).items():
        solver = GreedyLayoutSolver(buffers, ceiling)
        planned = solver.plan_layout()
        result["scenarios"][name] = {
            "addresses": {b.name: b.address for b in planned},
            "n_placed": sum(1 for b in planned if b.address is not None),
            "n_total": len(planned),
            "spill_reasons": dict(sorted(solver.spill_reasons.items())),
        }
        placed = result["scenarios"][name]["n_placed"]
        print(f"  {name:14s} placed {placed:4d} / {len(planned):4d}", flush=True)

    json.dump(result, open(out_path, "w"), indent=1, sort_keys=True)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
