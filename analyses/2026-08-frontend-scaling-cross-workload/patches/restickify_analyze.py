"""Restickify beam: component scaling and cost-model fit.

Separates the pass into its once-per-pass DP precomputation and its per-op
beam loop, then tests the two competing cost models for the loop:

  expansions   -- proportional to the number of hypotheses built
  copy volume  -- proportional to sum of len(assignments) over expansions,
                  which carries an extra factor of the op index

Flat cost-per-unit identifies which model the pass actually obeys.
"""
import collections
import json
import math
import sys

rows = []
for path in sys.argv[1:]:
    d = json.load(open(path))
    t = collections.defaultdict(float)
    beam_meta = {}
    ops = None
    for e in d["events"]:
        t[e["name"]] += e["inclusive_ns"] / 1e6
        if e["name"] == "restickify:beam":
            beam_meta = e.get("meta", {})
        if e["name"].endswith(":optimize_restickify_locations"):
            ops = e.get("meta", {}).get("input_operations")
    rows.append((d["meta"]["Lk"], t, beam_meta, ops))
rows.sort()

hdr = "".join(f"{'Lk=' + str(r[0]):>12s}" for r in rows)


def line(label, vals, fmt="{:12.1f}"):
    print(f"{label:34s}" + "".join(fmt.format(v) if v is not None else f"{'--':>12s}" for v in vals))


def exp_of(vals, ns):
    if not all(v and v > 0 for v in vals) or not all(ns) or ns[-1] <= ns[0]:
        return None
    return math.log(vals[-1] / vals[0]) / math.log(ns[-1] / ns[0])


ns = [r[3] for r in rows]
print(f"{'':34s}{hdr}")
line("input_operations", [float(n) if n else None for n in ns], "{:12.0f}")
print()
print("--- component time (ms) ---")
comps = [
    ("pass total (instrumented)", "pass:CustomPreSchedulingPasses:optimize_restickify_locations"),
    ("  restickify:beam", "restickify:beam"),
    ("    future_min_cost (DP)", "restickify:future_min_cost"),
    ("    reorder_any_in_nodes", "restickify:reorder_any_in_nodes"),
    ("    last_use", "restickify:last_use"),
]
for label, key in comps:
    v = [r[1].get(key, 0.0) for r in rows]
    e = exp_of(v, ns)
    line(label, v)
    if e is not None:
        print(f"{'':34s}{'exponent = ' + format(e, '.2f'):>50s}")
# op loop = beam - its named children
loop = [
    r[1].get("restickify:beam", 0.0)
    - r[1].get("restickify:future_min_cost", 0.0)
    - r[1].get("restickify:reorder_any_in_nodes", 0.0)
    - r[1].get("restickify:last_use", 0.0)
    for r in rows
]
line("    op loop (derived)", loop)
e = exp_of(loop, ns)
if e is not None:
    print(f"{'':34s}{'exponent = ' + format(e, '.2f'):>50s}")
cost = [r[2].get("cost_ms", 0.0) for r in rows]
line("      of which cost_fn.cost", cost)
line("      of which bookkeeping", [a - b for a, b in zip(loop, cost)])

print()
print("--- beam structural counters ---")
for label, key, fmt in (
    ("n_ops_with_layouts", "n_ops_with_layouts", "{:12.0f}"),
    ("n_states_built (expansions)", "n_states_built", "{:12.0f}"),
    ("assign_len_max", "assign_len_max", "{:12.0f}"),
    ("assign_len_sum (copy volume)", "assign_len_sum", "{:12.0f}"),
    ("cost_calls", "cost_calls", "{:12.0f}"),
    ("trim_states_in_max", "trim_states_in_max", "{:12.0f}"),
):
    v = [r[2].get(key) for r in rows]
    line(label, [float(x) if x is not None else None for x in v], fmt)
    e = exp_of([x for x in v], ns)
    if e is not None:
        print(f"{'':34s}{'exponent = ' + format(e, '.2f'):>50s}")

print()
print("--- cost model: per-unit cost, flat means the model holds ---")
book = [a - b for a, b in zip(loop, cost)]
per_exp = [
    1e6 * b / r[2]["n_states_built"] if r[2].get("n_states_built") else None
    for b, r in zip(book, rows)
]
per_vol = [
    1e6 * b / r[2]["assign_len_sum"] if r[2].get("assign_len_sum") else None
    for b, r in zip(book, rows)
]
line("ns per expansion", per_exp)
if all(per_exp):
    print(f"{'':34s}{'drift = ' + format(per_exp[-1] / per_exp[0], '.2f') + 'x':>50s}")
line("ns per copied element", per_vol)
if all(per_vol):
    print(f"{'':34s}{'drift = ' + format(per_vol[-1] / per_vol[0], '.2f') + 'x':>50s}")
