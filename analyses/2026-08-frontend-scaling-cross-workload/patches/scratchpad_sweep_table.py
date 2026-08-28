import collections, json, sys

ROWS = [
    ("plan_allocation", "scratchpad:plan_allocation"),
    ("prepare_buffers", "scratchpad:prepare_buffers"),
    ("  collect_lx_relayout_plans", "scratchpad:collect_lx_relayout_plans"),
    ("  generate_buffers", "scratchpad:generate_buffers"),
    ("    residency_reasons", "scratchpad:residency_reasons"),
    ("    get_ncores_for_buffers", "scratchpad:get_ncores_for_buffers"),
    ("    mem_usage_by_buf", "scratchpad:mem_usage_by_buf"),
    ("    determine_in_place", "scratchpad:determine_in_place"),
    ("    build_bound_buffers", "scratchpad:build_bound_buffers"),
    ("    calculate_liveness", "scratchpad:calculate_liveness"),
    ("solve", "scratchpad:solve"),
    ("  plan_layout", "scratchpad:plan_layout"),
]

samples = []
for path in sys.argv[1:]:
    d = json.load(open(path))
    tot = collections.defaultdict(float)
    for e in d["events"]:
        tot[e["name"]] += e["inclusive_ns"] / 1e6
    pl = next((e for e in d["events"]
               if e["name"] == "scratchpad:plan_layout"
               and "try_allocate_ms" in e.get("meta", {})), None)
    samples.append((d["meta"].get("depth"), tot, pl, d["meta"]))

samples.sort(key=lambda s: s[0])
hdr = "".join(f"{'d=' + str(s[0]):>12s}" for s in samples)
print(f"{'ms':32s}{hdr}")
for label, name in ROWS:
    cells = "".join(f"{s[1].get(name, float('nan')):12.2f}" for s in samples)
    print(f"{label:32s}{cells}")

print()
print(f"{'structural':32s}{hdr}")
for key, get in (
    ("n_placeable", lambda pl: pl["meta"].get("n_placeable") if pl else None),
    ("n_times", lambda pl: pl["meta"].get("n_times") if pl else None),
    ("live_sum", lambda pl: pl["meta"].get("live_sum") if pl else None),
    ("try_allocate_calls", lambda pl: pl["meta"].get("try_allocate_calls") if pl else None),
):
    cells = "".join(f"{str(get(s[2])):>12s}" for s in samples)
    print(f"{key:32s}{cells}")
cells = "".join(f"{s[1].get('first_call_wall', 0):12.0f}" for s in samples)
print(f"{'first_call_wall (ms)':32s}{cells}")

print()
print(f"{'growth ratio (prev -> next)':32s}")
for label, name in ROWS:
    vals = [s[1].get(name) for s in samples]
    if any(v is None or v <= 0 for v in vals):
        continue
    rs = "".join(f"{vals[i + 1] / vals[i]:11.2f}x" for i in range(len(vals) - 1))
    print(f"{label:32s}{'':12s}{rs}")
