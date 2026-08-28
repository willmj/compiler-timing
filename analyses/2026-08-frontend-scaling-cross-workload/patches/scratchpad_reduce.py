"""Roll scratchpad:* substage events into an attribution table.

Sums same-named events within a sample (the SolveError retry path and
repeated helper calls both produce duplicates), reports the parent-child
tree, and prints the reconciliation residual explicitly.
"""
import collections
import json
import sys

LEVEL1 = [
    "scratchpad:run_pre_passes",
    "scratchpad:prepare_buffers",
    "scratchpad:build_solver",
    "scratchpad:solve",
    "scratchpad:finalize_lx_relayout",
    "scratchpad:post_solve",
    "scratchpad:get_spill_reasons",
    "scratchpad:push_allocation",
    "scratchpad:log_lx_pinning",
    "scratchpad:run_post_passes",
]


def load(path):
    d = json.load(open(path))
    tot = collections.defaultdict(float)
    cnt = collections.Counter()
    for e in d["events"]:
        tot[e["name"]] += e["inclusive_ns"] / 1e6
        cnt[e["name"]] += 1
    return d, tot, cnt


def main(paths):
    for path in paths:
        d, tot, cnt = load(path)
        m = d["meta"]
        st = m.get("scratchpad_timing", {})
        print("=" * 74)
        print(f"{path}   depth={m.get('depth')} dim={m.get('dim')}")
        print(f"level={st.get('level')} wrapped={len(st.get('wrapped', []))} "
              f"missing_required={st.get('missing_required')}")
        plan = tot.get("scratchpad:plan_allocation", 0.0)
        wall = tot.get("first_call_wall", 0.0)
        print(f"\nfirst_call_wall            {wall:10.2f} ms")
        print(f"scratchpad:plan_allocation {plan:10.2f} ms"
              f"   ({100 * plan / wall:.2f}% of wall, x{cnt['scratchpad:plan_allocation']})")
        if not plan:
            print("  (allocator never ran)")
            continue

        print(f"\n  {'level-1 step':44s} {'ms':>9s} {'% pass':>7s} {'n':>3s}")
        s = 0.0
        for name in LEVEL1:
            if name not in tot:
                continue
            v = tot[name]
            s += v
            print(f"  {name:44s} {v:9.2f} {100 * v / plan:7.2f} {cnt[name]:3d}")
        print(f"  {'SUM of level-1 steps':44s} {s:9.2f} {100 * s / plan:7.2f}")
        print(f"  {'residual (plan_allocation self)':44s} "
              f"{plan - s:9.2f} {100 * (plan - s) / plan:7.2f}")

        sub = sorted(
            (v, k) for k, v in tot.items()
            if k.startswith("scratchpad:") and k not in LEVEL1
            and k != "scratchpad:plan_allocation"
        )
        if sub:
            print(f"\n  {'level-2/3 substage':44s} {'ms':>9s} {'% pass':>7s} {'n':>3s}")
            for v, k in reversed(sub):
                print(f"  {k:44s} {v:9.2f} {100 * v / plan:7.2f} {cnt[k]:3d}")

        for e in d["events"]:
            if e["name"] != "scratchpad:plan_layout":
                continue
            meta = e.get("meta", {})
            if "try_allocate_ms" not in meta:
                continue
            incl = e["inclusive_ns"] / 1e6
            acc = meta["try_allocate_ms"] + meta["try_deallocate_ms"]
            print("\n  plan_layout internals (the hypothesis test)")
            print(f"    inclusive                {incl:9.2f} ms")
            print(f"    _try_allocate            {meta['try_allocate_ms']:9.2f} ms "
                  f"({meta['try_allocate_calls']} calls)")
            print(f"    _try_deallocate          {meta['try_deallocate_ms']:9.2f} ms "
                  f"({meta['try_deallocate_calls']} calls)")
            print(f"    time-walk residual       {incl - acc:9.2f} ms "
                  f"({100 * (incl - acc) / incl:.1f}% of plan_layout)")
            print(f"    n_placeable={meta.get('n_placeable')} "
                  f"n_times={meta.get('n_times')} "
                  f"live_sum={meta.get('live_sum')} live_max={meta.get('live_max')}")
            if meta.get("n_placeable") and meta.get("n_times"):
                print(f"    n_placeable*n_times={meta['n_placeable'] * meta['n_times']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
