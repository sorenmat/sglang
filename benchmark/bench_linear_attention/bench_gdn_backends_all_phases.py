"""Benchmark & Correctness: GDN kernel backends across all phases.

Drives the production kernel wrapper classes (TritonGDNKernel /
FlashInferGDNKernel) -- the same objects GDNKernelDispatcher selects --
for each phase (decode / prefill / verify) so per-backend numbers reflect
the real serving paths, not just raw kernels.

Purpose: pick the fastest backend per phase for a target GPU (e.g. Qwen3.8
on RTX PRO 6000 / SM120, where FlashInfer prefill is auto-selected but
decode/verify defaults are Triton), and project the per-step cost across
the model's GDN layers (Qwen3.8: 48 GDN layers dominate the hybrid stack).

Usage:
    python bench_gdn_backends_all_phases.py                     # full sweep
    python bench_gdn_backends_all_phases.py --phases decode,verify
    python bench_gdn_backends_all_phases.py --preset qwen38-27b --state-dtype bfloat16
    SGLANG_GDN_FLASHINFER_VERIFY_SM120=1 python bench_gdn_backends_all_phases.py --phases verify
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import torch

# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS = {
    # Qwen3.8-27B class: 48 GDN layers, linear key/value head dims 128,
    # 16 key heads / 32 value heads (single-GPU, tp=1).
    "qwen38-27b": dict(H=16, HV=32, K=128, V=128, gdn_layers=48),
    # Qwen3-Next 80B per-TP-shard shape (tp=8).
    "qwen3-next-tp8": dict(H=2, HV=4, K=128, V=128, gdn_layers=48),
    "small": dict(H=4, HV=8, K=128, V=128, gdn_layers=24),
}


def make_state_pool(pool_slots, HV, K, V, dtype, device):
    # Production layout: [pool, HV, V, K] (K-last).
    return torch.randn(pool_slots, HV, V, K, device=device, dtype=dtype) * 0.05


def bench(fn, *, warmup=10, iters=50):
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
    return start.elapsed_time(end) / iters  # ms


def build_decode_inputs(B, p, dtype, device, pool_slots):
    return dict(
        q=torch.randn(1, B, p["H"], p["K"], device=device, dtype=dtype),
        k=torch.randn(1, B, p["H"], p["K"], device=device, dtype=dtype),
        v=torch.randn(1, B, p["HV"], p["V"], device=device, dtype=dtype),
        a=torch.randn(1, B, p["HV"], device=device, dtype=dtype),
        b=torch.randn(1, B, p["HV"], device=device, dtype=dtype),
        A_log=torch.randn(p["HV"], device=device, dtype=torch.float32),
        dt_bias=torch.randn(p["HV"], device=device, dtype=torch.float32),
        ssm_states=make_state_pool(pool_slots, p["HV"], p["K"], p["V"], dtype, device),
        cache_indices=torch.arange(B, device=device, dtype=torch.int32),
        query_start_loc=torch.arange(
            0, B + 1, device=device, dtype=torch.int32
        ),
    )


def build_prefill_inputs(T, p, dtype, device, pool_slots, B=8):
    # log-sigmoid-like gates in [-6, 0), beta in (0, 1).
    return dict(
        q=torch.randn(1, T, p["H"], p["K"], device=device, dtype=dtype),
        k=torch.randn(1, T, p["H"], p["K"], device=device, dtype=dtype),
        v=torch.randn(1, T, p["HV"], p["V"], device=device, dtype=dtype),
        g=-torch.rand(1, T, p["HV"], device=device, dtype=dtype) * 6.0,
        beta=torch.rand(1, T, p["HV"], device=device, dtype=dtype),
        ssm_states=make_state_pool(pool_slots, p["HV"], p["K"], p["V"], dtype, device),
        cache_indices=torch.arange(B, device=device, dtype=torch.int32),
        query_start_loc=torch.arange(
            0, T + 1, T // B, device=device, dtype=torch.int32
        )[: B + 1].contiguous(),
    )


def build_verify_inputs(B, D, p, dtype, device, pool_slots):
    n = B * D
    state = make_state_pool(pool_slots, p["HV"], p["K"], p["V"], dtype, device)
    return dict(
        A_log=torch.randn(p["HV"], device=device, dtype=torch.float32),
        dt_bias=torch.randn(p["HV"], device=device, dtype=torch.float32),
        q=torch.randn(1, n, p["H"], p["K"], device=device, dtype=dtype),
        k=torch.randn(1, n, p["H"], p["K"], device=device, dtype=dtype),
        v=torch.randn(1, n, p["HV"], p["V"], device=device, dtype=dtype),
        a=torch.randn(1, n, p["HV"], device=device, dtype=dtype),
        b=torch.randn(1, n, p["HV"], device=device, dtype=dtype),
        ssm_states=state,
        cache_indices=torch.arange(B, device=device, dtype=torch.int32),
        query_start_loc=torch.arange(
            0, n + 1, D, device=device, dtype=torch.int32
        )[: B + 1].contiguous(),
        intermediate_states_buffer=torch.zeros(
            pool_slots, D, p["HV"], p["K"], p["V"], device=device, dtype=dtype
        ),
        intermediate_state_indices=torch.arange(B, device=device, dtype=torch.int32),
        cache_steps=D,
        retrieve_parent_token=None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phases", default="decode,prefill,verify")
    parser.add_argument("--backends", default="triton,flashinfer")
    parser.add_argument("--preset", default="qwen38-27b", choices=PRESETS.keys())
    parser.add_argument("--state-dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--batch-sizes", default="1,8,16,24")
    parser.add_argument("--prefill-tokens", default="8192")
    parser.add_argument("--draft-tokens", type=int, default=4)
    parser.add_argument("--pool-slots", type=int, default=64)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--check", action="store_true", help="cross-backend correctness")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "This benchmark requires a CUDA GPU"
    device = "cuda"
    torch.manual_seed(42)
    p = PRESETS[args.preset]
    dtype = getattr(torch, args.state_dtype)
    cap = torch.cuda.get_device_capability()
    print(
        f"GPU: {torch.cuda.get_device_name()} (SM{cap[0]}{cap[1]}), "
        f"preset={args.preset} H={p['H']} HV={p['HV']} K={p['K']} V={p['V']} "
        f"state={args.state_dtype} gdn_layers={p['gdn_layers']}"
    )

    kernels = {}
    if "triton" in args.backends:
        from sglang.srt.layers.attention.linear.kernels.gdn_triton import (
            TritonGDNKernel,
        )

        kernels["triton"] = TritonGDNKernel()
    if "flashinfer" in args.backends:
        from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
            FlashInferGDNKernel,
        )

        try:
            kernels["flashinfer"] = FlashInferGDNKernel()
        except RuntimeError as e:
            print(f"flashinfer unavailable on this machine: {e}")
            del kernels["flashinfer"]

    phases = [s.strip() for s in args.phases.split(",")]
    batches = [int(s) for s in args.batch_sizes.split(",")]
    prefill_tokens = [int(s) for s in args.prefill_tokens.split(",")]

    results = []  # (phase, shape, backend, ms)
    for phase in phases:
        for B in batches:
            if phase == "decode":
                inputs = build_decode_inputs(B, p, dtype, device, args.pool_slots)
                runner = lambda kern, inp=inputs: kern.decode(**inp)
                shape = f"B={B}"
            elif phase == "prefill":
                T = prefill_tokens[0] if len(prefill_tokens) == 1 else None
                if T is None:
                    continue
                inputs = build_prefill_inputs(T, p, dtype, device, args.pool_slots)
                runner = lambda kern, inp=inputs: kern.extend(**inp)
                shape = f"T={T}"
            elif phase == "verify":
                inputs = build_verify_inputs(
                    B, args.draft_tokens, p, dtype, device, args.pool_slots
                )
                runner = lambda kern, inp=inputs: kern.target_verify(**inp)
                shape = f"B={B},D={args.draft_tokens}"
            else:
                raise ValueError(phase)

            outputs = {}
            for name, kern in kernels.items():
                if phase == "verify" and name == "flashinfer":
                    if not getattr(kern, "supports_target_verify", False):
                        print(
                            f"verify B={B}: flashinfer supports_target_verify=False "
                            "on this GPU (set SGLANG_GDN_FLASHINFER_VERIFY_SM120=1 "
                            "on SM120); skipped"
                        )
                        continue
                try:
                    ms = bench(
                        lambda: runner(kern), warmup=args.warmup, iters=args.iters
                    )
                    results.append((phase, shape, name, ms))
                    outputs[name] = runner(kern)
                except Exception as e:  # noqa: BLE001
                    print(f"{phase} {shape} {name}: FAILED {type(e).__name__}: {e}")

            if args.check and "triton" in outputs and "flashinfer" in outputs:
                t, f = outputs["triton"], outputs["flashinfer"]
                if isinstance(t, tuple):
                    t, f = t[0], f[0]
                same = torch.allclose(
                    t.float(), f.float(), atol=2e-2, rtol=2e-2
                )
                maxdiff = (t.float() - f.float()).abs().max().item()
                print(f"  correctness {phase} {shape}: allclose={same} maxdiff={maxdiff:.4g}")

    if not results:
        print("no results")
        return

    print(f"\n{'phase':8s} {'shape':14s} {'backend':12s} {'ms/layer':>10s} {'ms x48 layers':>14s}")
    for phase, shape, name, ms in results:
        print(
            f"{phase:8s} {shape:14s} {name:12s} {ms:10.4f} {ms * p['gdn_layers']:14.2f}"
        )

    # Per-phase winner summary for backend selection.
    print("\nfastest per shape:")
    by_key = {}
    for phase, shape, name, ms in results:
        by_key.setdefault((phase, shape), []).append((ms, name))
    for (phase, shape), cands in sorted(by_key.items()):
        cands.sort()
        if len(cands) > 1:
            speedup = cands[1][0] / cands[0][0]
            print(f"  {phase} {shape}: {cands[0][1]} ({speedup:.2f}x vs {cands[1][1]})")


if __name__ == "__main__":
    main()
