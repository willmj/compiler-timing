"""Per-pass cost per input operation across a sweep.

A pass whose ms/op is flat is linear. A pass whose ms/op RISES is
superlinear, and rising per-unit cost -- rather than a superlinear count of
operations -- localizes the cause to the cost of each unit of work.
"""
import collections
import json
import math
import sys

rows = []
for path in sys.argv[1:]:
    d = json.load(open(path))
    t = collections.defaultdict(float)
    io = {}
    for e in d["events"]:
        t[e["name"]] += e["inclusive_ns"] / 1e6
        v = e.get("meta", {}).get("input_operations")
        if v is not None:
            io.setdefault(e["name"], v)
        if e["name"] == "scratchpad:residency_reasons":
            io["scratchpad:residency_reasons"] = e.get("meta", {}).get("n_names")
    rows.append((d["meta"]["Lk"], t, io))
rows.sort()

TRACK = [
    ("dedup_and_promote_constants", "pass:CustomPreSchedulingPasses:dedup_and_promote_constants"),
    ("optimize_restickify_locations", "pass:CustomPreSchedulingPasses:optimize_restickify_locations"),
    ("_maybe_scratchpad_planning", "pass:CustomPreSchedulingPasses:_maybe_scratchpad_planning"),
    ("propagate_spyre_tensor_layouts", "pass:CustomPreSchedulingPasses:propagate_spyre_tensor_layouts"),
    ("span_reduction", "pass:CustomPreSchedulingPasses:span_reduction"),
    ("_distribute_work", "pass:CustomPreSchedulingPasses:_distribute_work"),
    ("validate_ops", "pass:CustomPreSchedulingPasses:validate_ops"),
    ("deadcode_elimination", "pass:CustomPreSchedulingPasses:deadcode_elimination"),
    ("  sp: residency_reasons", "scratchpad:residency_reasons"),
    ("  sp: collect_lx_relayout", "scratchpad:collect_lx_relayout_plans"),
    ("  sp: solve", "scratchpad:solve"),
]

hdr = "".join(f"{'Lk=' + str(r[0]):>11s}" for r in rows)
print(f"{'input_operations':38s}" + "".join(
    f"{str(r[2].get('pass:CustomPreSchedulingPasses:dedup_and_promote_constants')):>11s}" for r in rows))
print()
print(f"{'ABSOLUTE ms':38s}{hdr}   {'exp':>6s}")
for label, key in TRACK:
    v = [r[1].get(key, 0.0) for r in rows]
    n = [r[2].get(key) for r in rows]
    exp = ""
    if all(x > 1.0 for x in v) and n[0] and n[-1] and n[-1] > n[0]:
        exp = f"{math.log(v[-1]/v[0])/math.log(n[-1]/n[0]):6.2f}"
    print(f"{label:38s}" + "".join(f"{x:11.1f}" for x in v) + f"   {exp:>6s}")

print()
print(f"{'PER OPERATION (us/op)':38s}{hdr}   {'drift':>7s}")
for label, key in TRACK:
    v = [r[1].get(key, 0.0) for r in rows]
    n = [r[2].get(key) for r in rows]
    if not all(n) or not all(x > 1.0 for x in v):
        continue
    per = [1000.0 * a / b for a, b in zip(v, n)]
    drift = f"{per[-1]/per[0]:6.2f}x"
    print(f"{label:38s}" + "".join(f"{x:11.1f}" for x in per) + f"   {drift:>7s}")
