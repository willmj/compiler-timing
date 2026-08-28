"""Per-pass growth ratios across a workload sweep.

Reports each pass against its own `input_operations` at entry -- the axis the
committed per-pass scaling tables use -- so "the pass got slower" and "the
graph got bigger" stay separable. An exponent near 1.0 is linear.
"""
import collections
import json
import math
import sys

samples = []
for path in sys.argv[1:]:
    d = json.load(open(path))
    tot = collections.defaultdict(float)
    ops = {}
    for e in d["events"]:
        tot[e["name"]] += e["inclusive_ns"] / 1e6
        io = e.get("meta", {}).get("input_operations")
        if io is not None:
            ops.setdefault(e["name"], io)
    samples.append((d["meta"].get("n"), tot, ops))
samples.sort(key=lambda s: s[0])

names = [k for k in samples[-1][1] if k.startswith("pass:")]
names.sort(key=lambda k: -samples[-1][1][k])

hdr = "".join(f"{'n=' + str(s[0]):>10s}" for s in samples)
print(f"{'pass (ms)':52s}{hdr}   {'exp':>6s}  verdict")
print("-" * (52 + 10 * len(samples) + 18))

for name in names:
    vals = [s[1].get(name) for s in samples]
    nops = [s[2].get(name) for s in samples]
    cells = "".join(f"{v:10.2f}" if v is not None else f"{'--':>10s}" for v in vals)
    exp = None
    if all(v and v > 0.5 for v in vals) and nops[0] and nops[-1] and nops[-1] > nops[0]:
        exp = math.log(vals[-1] / vals[0]) / math.log(nops[-1] / nops[0])
    if exp is None:
        verdict, es = "too small to time", f"{'--':>6s}"
    elif exp < 1.25:
        verdict, es = "linear", f"{exp:6.2f}"
    elif exp < 1.7:
        verdict, es = "SUPERLINEAR", f"{exp:6.2f}"
    else:
        verdict, es = "NEAR-QUADRATIC", f"{exp:6.2f}"
    print(f"{name.replace('pass:', ''):52s}{cells}   {es}  {verdict}")

print()
print(f"{'input_operations at pass entry':52s}" + "".join(
    f"{str(s[2].get('pass:CustomPreSchedulingPasses:_maybe_scratchpad_planning')):>10s}"
    for s in samples))
for label, key in (("pipeline:CustomPreSchedulingPasses", "pipeline:CustomPreSchedulingPasses"),
                   ("first_call_wall", "first_call_wall")):
    print(f"{label:52s}" + "".join(f"{s[1].get(key, 0):10.2f}" for s in samples))
