"""Per-mechanism attribution for the restickify changes.

Three arms on one checkout with one _C.so, patches applied in turn:
base, +device_coordinates memo, +memo+per-edge hoists. Keeping the memo's
own delta separately attributable was a review condition, since the two
hoists are entangled with each other (host_coordinates is only reached past
the stick-compatible early-out, and takes its ind_sizes from the memoized
indirect_info_from_op) but neither is entangled with the memo.
"""

import collections
import json
import sys

R = "pass:CustomPreSchedulingPasses:optimize_restickify_locations"
PIPE = "pipeline:CustomPreSchedulingPasses"
WALL = "first_call_wall"
ARMS = ("base", "memo", "full")
LABEL = {"base": "base", "memo": "+memo", "full": "+memo+hoists"}


def totals(path):
    d = json.load(open(path))
    out = collections.defaultdict(float)
    for e in d["events"]:
        out[e["name"]] += e["inclusive_ns"] / 1e6
    return out


def main(directory):
    got = {}
    for arm in ARMS:
        for lk in (2048, 8192):
            try:
                got[(arm, lk)] = totals(f"{directory}/3a_{arm}_{lk}.json")
            except FileNotFoundError:
                pass

    for key, label in ((R, "optimize_restickify_locations"),
                       (PIPE, "CustomPreSchedulingPasses"),
                       (WALL, "first_call_wall")):
        print(f"\n{label}")
        print(f"  {'':16s}" + "".join(f"{'Lk=' + str(k):>22s}" for k in (2048, 8192)))
        base = {k: got[("base", k)][key] for k in (2048, 8192) if ("base", k) in got}
        for arm in ARMS:
            cells = ""
            for lk in (2048, 8192):
                if (arm, lk) not in got:
                    cells += f"{'--':>22s}"
                    continue
                v = got[(arm, lk)][key]
                d = 100 * (v - base[lk]) / base[lk] if arm != "base" else 0.0
                cells += f"{v:14.0f}{'' if arm == 'base' else f' ({d:+5.1f}%)':>8s}"
            print(f"  {LABEL[arm]:16s}{cells}")

    print("\nincremental contribution of the hoists (full vs memo)")
    for lk in (2048, 8192):
        if ("memo", lk) not in got or ("full", lk) not in got:
            continue
        m, f = got[("memo", lk)][R], got[("full", lk)][R]
        print(f"  Lk={lk:5d}  {m:8.0f} -> {f:8.0f} ms  {100 * (f - m) / m:+6.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "."))
