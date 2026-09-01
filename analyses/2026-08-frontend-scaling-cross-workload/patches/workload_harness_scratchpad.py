"""Scratchpad-stressing workloads for substage timing.

Two topologies, because the live set -- not the buffer count -- is what
makes the solver work:

  chain   x -> softmax(x@w) -> softmax(...) -> ...
          Strictly sequential. Buffer k dies as k+1 is born, so the live
          set never exceeds ~1 and the 1.55 MiB ceiling is never
          approached. Scales buffer count without scaling the packing
          problem.

  fanin   W independent branches, each softmax((x@w)*s_i), summed at the
          end. Every branch output is live from its production until the
          accumulate chain reaches it, so the peak live set is ~W and the
          ceiling binds once W * size_per_core exceeds it. This is the
          long-lived-carry shape that workload A has and the chain does
          not.

The per-branch scalar s_i keeps the branches distinct; without it CSE
collapses all W of them into one buffer and the fan-in disappears.

Ops used (mm, mul, exp, sub, max, sum, div, add) are all on the
OP_OUTPUT_GOOD_FOR_LX_REUSE allowlist, so branch outputs are genuine LX
candidates rather than gate-1 rejects.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import torch_spyre._inductor as _si
import timing_recorder as _tr

sys.modules["torch_spyre._inductor.timing_recorder"] = _tr
_si.timing_recorder = _tr

import pass_pipeline_timing as ppt
import scratchpad_substage_timing as sst
import restickify_beam_timing as rbt


def build_chain(depth):
    def fn(x, w):
        for _ in range(depth):
            x = torch.softmax(x @ w, dim=-1)
        return x

    return fn


def build_fanin(width):
    def fn(x, w):
        parts = [
            torch.softmax((x @ w) * (1.0 + 0.01 * i), dim=-1) for i in range(width)
        ]
        acc = parts[0]
        for p in parts[1:]:
            acc = acc + p
        return acc

    return fn


def build_conflict(n):
    """Two consumers of one buffer reducing over DIFFERENT axes.

    sum(-1) and sum(-2) want different stick dimensions on the same producer
    output, so one of them cannot read it as laid out. That is the case the beam
    exists for: a nonzero-cost assignment where a narrower frontier could pick a
    worse layout than a wider one. The zero-cost workloads cannot test that.
    """

    def fn(x, w):
        acc = None
        for i in range(n):
            a = (x @ w) * (1.0 + 0.01 * i)
            t = a.sum(dim=-1) + a.sum(dim=-2)
            acc = t if acc is None else acc + t
        return acc

    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", choices=("chain", "fanin", "conflict"), default="fanin")
    ap.add_argument("--n", type=int, default=8, help="depth for chain, width for fanin")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare-cpu", action="store_true")
    args = ap.parse_args()

    sp = sst.install()
    pp = ppt.install()
    rb = rbt.install()
    print(
        f"armed: scratchpad level {sp.level} ({len(sp.wrapped)} wraps, "
        f"{len(sp.missing_required)} missing) | "
        f"pipelines {len(pp['wrapped'])} wrapped, {len(pp['missing'])} missing",
        flush=True,
    )

    torch.manual_seed(0)
    # Spyre has no fp32 batchmatmul; fp16 is the supported device dtype.
    x_cpu = torch.randn(args.dim, args.dim, dtype=torch.float16)
    w_cpu = (torch.randn(args.dim, args.dim, dtype=torch.float32) * 0.05).half()
    fn = {
        "chain": build_chain,
        "fanin": build_fanin,
        "conflict": build_conflict,
    }[args.topology](args.n)

    with _tr.stage("device_init_and_transfer"):
        x = x_cpu.to("spyre")
        w = w_cpu.to("spyre")

    compiled = torch.compile(fn, backend="inductor")
    with _tr.stage("first_call_wall", topology=args.topology, n=args.n, dim=args.dim):
        out = compiled(x, w)
        out_cpu = out.to("cpu")

    _tr.set_run_meta(
        workload=f"scratchpad_{args.topology}",
        topology=args.topology,
        n=args.n,
        dim=args.dim,
        TORCHINDUCTOR_CACHE_DIR=os.environ.get("TORCHINDUCTOR_CACHE_DIR", "<unset>"),
        SENCORES=os.environ.get("SENCORES", "<unset>"),
        SPYRE_LX_PLANNER_RELAYOUT=os.environ.get("SPYRE_LX_PLANNER_RELAYOUT", "<unset>"),
    )
    _tr.dump_and_finalize(args.out)
    print(f"wrote {args.out}", flush=True)

    if args.compare_cpu:
        ref = fn(x_cpu.float(), w_cpu.float())
        torch.testing.assert_close(out_cpu.float(), ref, atol=0.15, rtol=0.15)
        print("cpu comparison PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
