"""Workload A -- tiled flash attention, parameterized for a scaling sweep.

The `flash` body is taken verbatim from
`tests/inductor/test_opspec_tiling.py::TestOpSpecTiling::test_flash`, which
hardcodes Lq=512 / Lk=1024 -- workload A's baseline point in the pr3806
study. Block sizes stay at their baseline values while Lk grows, so the
kv-tile count and therefore the inner-body count grow: that is the
graph-growth axis the study swept.

Exists to test one hypothesis the simple workloads cannot: that this
family's reported superlinear pass scaling comes from *per-buffer* cost
rising with expression complexity, rather than from any pass doing
superlinearly many operations. The discriminator is ms/input_operations
across the sweep -- flat means linear, rising means the per-unit cost
itself grows.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import torch_spyre._inductor as _si
import timing_recorder as _tr

sys.modules["torch_spyre._inductor.timing_recorder"] = _tr
_si.timing_recorder = _tr

import pass_pipeline_timing as ppt
import restickify_beam_timing as rbt
import scratchpad_substage_timing as sst


def build(B, H, D, Lq, Lk, b_block_size, h_block_size, q_block_size, kv_block_size):
    def flash(queries, keys, values, mask):
        scale = 1.0 / math.sqrt(math.sqrt(D))
        output = torch.zeros_like(queries)
        real_max = torch.full(
            (B, H, Lq, 64), float("-inf"), device=queries.device, dtype=torch.float16
        ).amax(-1)
        denominator = torch.zeros(
            (B, H, Lq, 64), device=queries.device, dtype=torch.float16
        ).amax(-1)

        for b_start in range(0, B, b_block_size):
            b_end = b_start + b_block_size
            for h_start in range(0, H, h_block_size):
                h_end = h_start + h_block_size
                for lq_start in range(0, Lq, q_block_size):
                    lq_end = lq_start + q_block_size
                    queries_tile = queries[b_start:b_end, h_start:h_end, lq_start:lq_end]
                    real_max_tile = real_max[b_start:b_end, h_start:h_end, lq_start:lq_end]
                    denominator_tile = denominator[
                        b_start:b_end, h_start:h_end, lq_start:lq_end
                    ]
                    output_tile = output[b_start:b_end, h_start:h_end, lq_start:lq_end]

                    for lk_start in range(0, Lk, kv_block_size):
                        lk_end = lk_start + kv_block_size
                        mask_tile = mask[:, :, lq_start:lq_end, lk_start:lk_end]
                        keys_tile = keys[b_start:b_end, h_start:h_end, lk_start:lk_end]
                        values_tile = values[b_start:b_end, h_start:h_end, lk_start:lk_end]
                        keys_tile_T = keys_tile.transpose(-1, -2).contiguous()

                        scores = torch.matmul(queries_tile * scale, keys_tile_T * scale)
                        scores = scores + mask_tile
                        block_max = torch.amax(scores, dim=-1)
                        running_max = torch.maximum(real_max_tile, block_max)

                        exp_scores = torch.exp(scores - running_max.unsqueeze(-1))
                        correction = torch.exp(real_max_tile - running_max)

                        denominator_tile.copy_(
                            denominator_tile * correction + exp_scores.sum(dim=-1)
                        )
                        output_tile.copy_(
                            output_tile * correction.unsqueeze(-1)
                            + torch.matmul(exp_scores, values_tile)
                        )
                        real_max_tile.copy_(running_max)

        return output / denominator.unsqueeze(-1)

    return flash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--Lq", type=int, default=512)
    ap.add_argument("--Lk", type=int, default=1024)
    ap.add_argument("--b-block", type=int, default=1)
    ap.add_argument("--h-block", type=int, default=4)
    ap.add_argument("--q-block", type=int, default=256)
    ap.add_argument("--kv-block", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare-cpu", action="store_true")
    args = ap.parse_args()

    sp = sst.install()
    pp = ppt.install()
    # Gate so the shim's own overhead can be measured against the same tree:
    # its wrappers fire once per expansion and once per cost evaluation.
    rb = rbt.install() if os.environ.get("SPYRE_RESTICKIFY_TIMING", "1") == "1" else rbt._Report()
    print(
        f"armed: scratchpad level {sp.level} ({len(sp.wrapped)} wraps, "
        f"{len(sp.missing_required)} missing) | "
        f"pipelines {len(pp['wrapped'])} wrapped, {len(pp['missing'])} missing | "
        f"restickify {len(rb.wrapped)} wrapped, {len(rb.missing)} missing",
        flush=True,
    )

    torch.manual_seed(0xAFFE)
    B, H, D, Lq, Lk = args.B, args.H, args.D, args.Lq, args.Lk
    flash = build(B, H, D, Lq, Lk, args.b_block, args.h_block, args.q_block, args.kv_block)

    queries_t = torch.randn(B, H, Lq, D, dtype=torch.float16)
    keys_t = torch.randn(B, H, Lk, D, dtype=torch.float16)
    values_t = torch.randn(B, H, Lk, D, dtype=torch.float16)
    causal = torch.tril(torch.ones(Lq, Lk, dtype=torch.bool))
    mask_t = torch.zeros(1, 1, Lq, Lk, dtype=torch.float16)
    mask_t.masked_fill_(~causal, float("-inf"))

    with _tr.stage("device_init_and_transfer"):
        q_s = queries_t.to("spyre")
        k_s = keys_t.to("spyre")
        v_s = values_t.to("spyre")
        m_s = mask_t.to(device="spyre")

    compiled = torch.compile(flash)
    with _tr.stage("first_call_wall", Lq=Lq, Lk=Lk, H=H):
        out = compiled(q_s, k_s, v_s, m_s)
        out_cpu = out.to("cpu")

    _tr.set_run_meta(
        workload="pr3806_test_flash",
        B=B, H=H, D=D, Lq=Lq, Lk=Lk,
        q_block=args.q_block, kv_block=args.kv_block, h_block=args.h_block,
        TORCHINDUCTOR_CACHE_DIR=os.environ.get("TORCHINDUCTOR_CACHE_DIR", "<unset>"),
        SENCORES=os.environ.get("SENCORES", "<unset>"),
    )
    _tr.dump_and_finalize(args.out)
    print(f"wrote {args.out}", flush=True)

    if args.compare_cpu:
        ref = flash(queries_t, keys_t, values_t, mask_t)
        torch.testing.assert_close(out_cpu, ref, atol=0.1, rtol=0.1)
        print("cpu comparison PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
