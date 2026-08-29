"""Benchmark: incremental NVFP4 dequant mirror vs full-prefix dequant.

Measures the cost this branch removes: quantized-KV pools dequantize the
whole cached prefix into an FP8 workspace on every use -- every chunked-
prefill chunk (per layer) and every speculative-verify cycle -- an
O(context) cost per step. The incremental mirror only re-dequantizes
newly written slots.

Simulates an agent-shaped workload: a long-lived prefix (32K-128K tokens)
that grows by a few committed tokens per speculative cycle, across the
model's full-attention layers (Qwen3.8: 16 layers).

Usage:
    python bench_nvfp4_dequant_mirror.py                     # default sweep
    python bench_nvfp4_dequant_mirror.py --prefix 131072 --new-tokens 8
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import torch


def build_quant_pool(num_tokens, head_num, head_dim, device):
    """Random packed FP4 rows + e4m3 block scales, in pool layout
    [size, head_num, dim/2] uint8 / [size, head_num, dim/16] e4m3."""
    k_fp4 = torch.randint(
        0, 256, (num_tokens, head_num, head_dim // 2), dtype=torch.uint8, device=device
    )
    v_fp4 = torch.randint(
        0, 256, (num_tokens, head_num, head_dim // 2), dtype=torch.uint8, device=device
    )
    scales = torch.ones(
        num_tokens, head_num, head_dim // 16, dtype=torch.uint8, device=device
    )
    k_scales = scales.clone()
    v_scales = scales.clone()
    return k_fp4, v_fp4, k_scales, v_scales


def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=int, nargs="+", default=[32768, 131072])
    parser.add_argument("--new-tokens", type=int, nargs="+", default=[8, 64])
    parser.add_argument("--layers", type=int, default=16, help="full-attn layers")
    parser.add_argument("--head-num", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--cycles", type=int, default=16)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU"
    device = "cuda"
    from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
        NVFP4KVCacheMethod,
    )
    from sglang.srt.layers.quantization.nvfp4_dequant_mirror import (
        NVFP4DequantMirror,
    )

    cap = torch.cuda.get_device_capability()
    print(
        f"GPU: {torch.cuda.get_device_name()} (SM{cap[0]}{cap[1]}), "
        f"layers={args.layers} heads={args.head_num} dim={args.head_dim}"
    )

    print(
        f"\n{'prefix':>8s} {'new/cyc':>8s} {'full ms/cyc':>12s} {'mirror ms/cyc':>14s} {'speedup':>8s}"
    )
    for prefix_len in args.prefix:
        # pool: prefix + slack for the new tokens
        size = prefix_len + 1024
        method = NVFP4KVCacheMethod(num_layers=args.layers, device=device)
        k_fp4, v_fp4, k_scales, v_scales = build_quant_pool(
            size, args.head_num, args.head_dim, device
        )
        prefix_idx = torch.arange(prefix_len, device=device)

        for new_tokens in args.new_tokens:
            new_idx = torch.arange(
                prefix_len, prefix_len + new_tokens, device=device
            )

            def full_dequant_cycle():
                """Baseline: dequantize the whole prefix, per layer, per cycle
                (what chunked prefill / spec verify pay without the mirror)."""
                for layer in range(args.layers):
                    method.dequantize_prev_kv(
                        k_fp4[prefix_idx],
                        k_scales[prefix_idx],
                        v_fp4[prefix_idx],
                        v_scales[prefix_idx],
                        layer,
                    )

            mirror = NVFP4DequantMirror(
                size=size,
                mirror_size=size,
                layer_num=args.layers,
                head_num=args.head_num,
                head_dim=args.head_dim,
                device=device,
            )

            def mirror_cycle():
                # simulate the cycle: new tokens committed (one write stamp),
                # then every layer refreshes the prefix through the mirror.
                mirror.note_kv_write(new_idx)
                for layer in range(args.layers):
                    mirror.refresh(
                        layer,
                        prefix_idx,
                        lambda idx, m=method: m.dequantize_prev_kv(
                            k_fp4[idx],
                            k_scales[idx],
                            v_fp4[idx],
                            v_scales[idx],
                            layer,
                        ),
                    )

            full_ms = bench(full_dequant_cycle, warmup=2, iters=args.iters)
            mirror_ms = bench(mirror_cycle, warmup=2, iters=args.iters)
            stats = mirror.stats()
            print(
                f"{prefix_len:8d} {new_tokens:8d} {full_ms:12.3f} {mirror_ms:14.3f} "
                f"{full_ms / mirror_ms:7.2f}x   (dequant ratio "
                f"{stats['dequant_ratio']:.4f})"
            )


if __name__ == "__main__":
    main()
