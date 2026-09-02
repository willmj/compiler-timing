"""Greedy vs CP-SAT scaling for `_maybe_scratchpad_planning` on workload A.

Both arms come from one tree with only LAYOUT_SOLVER differing, so the
comparison isolates the solver rather than a week of tree drift. CP-SAT is
nondeterministic across processes (torch-spyre#4196), so its points carry a
spread and the exponent is reported from the median.
"""

import collections
import json
import math
import statistics
import sys

S = "pass:CustomPreSchedulingPasses:_maybe_scratchpad_planning"
PIPE = "pipeline:CustomPreSchedulingPasses"
WALL = "first_call_wall"
DEDUP = "pass:CustomPreSchedulingPasses:dedup_and_promote_constants"


def load(path):
    d = json.load(open(path))
    t = collections.defaultdict(float)
    solver = None
    ops = None
    for e in d["events"]:
        t[e["name"]] += e["inclusive_ns"] / 1e6
        m = e.get("meta", {})
        if e["name"] == "scratchpad:build_solver":
            solver = m.get("solver_class")
        if e["name"] == DEDUP:
            ops = m.get("input_operations")
    return d["meta"]["Lk"], solver, ops, t


def main(paths):
    rows = collections.defaultdict(list)
    for p in paths:
        lk, solver, ops, t = load(p)
        arm = "cpsat" if solver == "CpSatLayoutSolver" else "greedy"
        rows[(arm, lk)].append((ops, t))

    lks = sorted({lk for _, lk in rows})
    print(f"{'':30s}" + "".join(f"{'Lk=' + str(k):>13s}" for k in lks))
    ops_row = [rows[("greedy", k)][0][0] for k in lks if ("greedy", k) in rows]
    print(f"{'input_operations':30s}" + "".join(f"{o:13d}" for o in ops_row))
    print()

    stats = {}
    for arm in ("greedy", "cpsat"):
        med, spread = [], []
        for k in lks:
            vals = [t[S] for _, t in rows.get((arm, k), [])]
            med.append(statistics.median(vals) if vals else float("nan"))
            spread.append((min(vals), max(vals)) if len(vals) > 1 else None)
        stats[arm] = med
        print(f"{arm + ' scratchpad (ms)':30s}" + "".join(f"{v:13.0f}" for v in med))
        if any(spread):
            print(f"{'  min-max':30s}" + "".join(
                f"{(f'{s[0]:.0f}-{s[1]:.0f}' if s else '-'):>13s}" for s in spread))
    print(f"{'cpsat / greedy':30s}"
          + "".join(f"{stats['cpsat'][i] / stats['greedy'][i]:12.1f}x"
                    for i in range(len(lks))))
    print()

    for arm in ("greedy", "cpsat"):
        v, o = stats[arm], ops_row
        exp = math.log(v[-1] / v[0]) / math.log(o[-1] / o[0])
        print(f"{arm + ' endpoint exponent':30s}{exp:13.2f}")
    print()

    # torch-spyre#3934 task 2 asks whether the cost is solver blowup or
    # model-construction overhead. plan_allocation's template splits exactly
    # along that line: `solve` is the CP-SAT search, `prepare_buffers` is the
    # buffer/feature construction that precedes it.
    print("solver blowup vs model construction (torch-spyre#3934 task 2)")
    for label, key in (("solve", "scratchpad:solve"),
                       ("prepare_buffers", "scratchpad:prepare_buffers")):
        for arm in ("greedy", "cpsat"):
            vals = [
                statistics.median([t[key] for _, t in rows.get((arm, k), [])] or [0])
                for k in lks
            ]
            print(f"{arm + ' ' + label + ' (ms)':30s}" + "".join(f"{x:13.1f}" for x in vals))
    for k in lks:
        sv = statistics.median(
            [t["scratchpad:solve"] for _, t in rows.get(("cpsat", k), [])] or [0])
        pb = statistics.median(
            [t["scratchpad:prepare_buffers"] for _, t in rows.get(("cpsat", k), [])] or [0])
        tot = sv + pb
        if tot:
            print(f"  Lk={k:5d}  cpsat: solve {100 * sv / tot:5.1f}%   "
                  f"prepare {100 * pb / tot:5.1f}%")
    print()

    for label, key in (("pipeline (ms)", PIPE), ("first_call_wall (ms)", WALL)):
        for arm in ("greedy", "cpsat"):
            vals = [
                statistics.median([t[key] for _, t in rows.get((arm, k), [])] or [0])
                for k in lks
            ]
            print(f"{arm + ' ' + label:30s}" + "".join(f"{x:13.0f}" for x in vals))
    print()

    # A slower plan is not automatically a worse deal. The solver changes the
    # plan, and the plan changes what the backend has to compile, so the
    # remainder outside the pre-scheduling pipeline (dominated by
    # dxp_standalone) has to be read alongside the pass cost.
    print("frontend cost vs everything else (backend-dominated remainder)")
    for arm in ("greedy", "cpsat"):
        rem = []
        for k in lks:
            w = statistics.median([t[WALL] for _, t in rows.get((arm, k), [])] or [0])
            pi = statistics.median([t[PIPE] for _, t in rows.get((arm, k), [])] or [0])
            rem.append(w - pi)
        print(f"{arm + ' wall - pipeline (ms)':30s}" + "".join(f"{x:13.0f}" for x in rem))
    for k in lks:
        gw = statistics.median([t[WALL] for _, t in rows.get(("greedy", k), [])] or [0])
        cw = statistics.median([t[WALL] for _, t in rows.get(("cpsat", k), [])] or [0])
        if gw:
            verdict = "cpsat wins" if cw < gw else "greedy wins"
            print(f"  Lk={k:5d}  total compile: greedy {gw / 1000:7.1f}s   "
                  f"cpsat {cw / 1000:7.1f}s   {verdict} "
                  f"({100 * (cw - gw) / gw:+.1f}%)")
    print()
    for k in lks:
        g = statistics.median([t[PIPE] for _, t in rows.get(("greedy", k), [])] or [0])
        c = statistics.median([t[PIPE] for _, t in rows.get(("cpsat", k), [])] or [0])
        sg = stats["greedy"][lks.index(k)]
        sc = stats["cpsat"][lks.index(k)]
        print(f"  Lk={k:5d}  scratchpad share of pipeline: "
              f"greedy {100 * sg / g:5.1f}%   cpsat {100 * sc / c:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
