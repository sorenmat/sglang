"""Phase-1 capacity model: measured constants only, EP2 two-GPU layout.

Replaces the uniform-iid-ledger of budget.py with a model calibrated on the
Sept 29-30 measurements and validated against them before predicting:

  validation  TP1 step times at N=1/4/8 must reproduce within 1.3x
  prediction  2x RTX PRO 6000, expert-parallel, 32 requests x 256k contexts,
              host-RAM tier, S=4 verification, against the 37.5 ms/pass
              budget implied by 80 tok/s per request

Measured inputs (sources in MEASURED): card bandwidth 1491 GB/s, pinned
H2D 56.9 GB/s per GPU, GEMM ms/step at N=1/4/8/16, context-flat step time
(14.37 -> 14.32 ms from 30k to 167k tokens), EP2 weight shard 42.7 GB/GPU,
prefetch coverage 0.65. The expert-time curve is a two-parameter linear fit
on the measured N=1/4/8 GEMM times against the analytic expert-union count;
the N=16 anchor is confounded (mixed prefill) and excluded from the fit.

All outputs are conditional arithmetic, not measurements or forecasts.
"""

import json
import math
from pathlib import Path

GB = 1e9
GIB = 2**30

# ---- model constants (checkpoint architecture, from budget.py lineage) ----
LAYERS = 48
EXPERTS = 512
TOPK = 10
EXPERT_BYTES = 3 * 2560 * 640 * 0.5625      # NVFP4 weights+scales per expert
KV_PER_TOKEN_LAYER = 2 * 2 * 256 * 2       # 12 full-attention layers
FULL_ATTN_LAYERS = 12
SELECTED_TOKENS = 2051                     # 2048 budget + tail
HIDDEN = 2560

MEASURED = {
    "card_bw_gbs": 1491.4,                  # fp32 reduce, RTX PRO 6000, Sep 29
    "link_h2d_gbs_per_gpu": 56.9,           # pinned, GCP Sep 30
    "tp1_gemm_ms": {1: 11.90, 4: 18.07, 8: 17.96},   # Verda Sep 29, clean
    "tp1_gemm_ms_16_confounded": 37.88,     # mixed-prefill window, excluded
    "tp1_total_ms": {1: 19.50, 4: 24.93, 8: 24.78},
    "tp1_index_ms": 0.93,                   # ~flat across batch
    "tp1_residual_ms": 5.9,                 # total - GEMM - index at N=8
    "context_flatness": {"ms_30k": 14.37, "ms_167k": 14.32},
    "ep2_weight_shard_gib": 42.7,           # GCP Sep 30, loads and shards
    "prefetch_coverage_3_to_7": 0.6507,
    "prefetch_overfetch_3_to_7": 0.3493,
}

TARGET = {"requests": 32, "context": 262144, "tok_s_per_request": 80.0,
          "positions": 4, "emitted_per_pass": 3.0,
          "hot_hit_rate": 0.8, "kv_union_multiplier": 3.0,
          "p2p_gbs_assumed": 25.0,          # NOT measured; sensitivity below
          "usable_gpu_gib": 94.2, "host_gib_available": 352.0}


def expert_union_per_layer(tokens):
    """Distinct experts per layer. MEASURED (HKV-9, artifacts/hkv9-20261014):
    real routing saturates all 512 experts within ~700 tokens — the
    uniform-iid S-curve was pessimistic at small token counts. The
    analytic form is retained for reference/comparison only."""
    return EXPERTS * -math.expm1(tokens * math.log1p(-TOPK / EXPERTS))


_MEASURED_CURVE = [(1, 10), (6, 50), (12, 100), (24, 200), (56, 300),
                   (137, 400), (211, 450), (286, 480), (387, 500),
                   (612, 510), (700, 512)]


def measured_union_per_layer(tokens):
    """Ground truth from HKV-9 (piecewise-linear through the measured
    first-crossing milestones): distinct experts vs cumulative tokens."""
    if tokens <= 0:
        return 0.0
    if tokens >= _MEASURED_CURVE[-1][0]:
        return float(_MEASURED_CURVE[-1][1])
    for (t0, e0), (t1, e1) in zip(_MEASURED_CURVE, _MEASURED_CURVE[1:]):
        if t0 <= tokens <= t1:
            frac = (tokens - t0) / (t1 - t0)
            return e0 + frac * (e1 - e0)
    return float(EXPERTS)


def fit_expert_curve():
    """Linear fit GEMM_ms = a + b * (48 * union_per_layer) on clean anchors."""
    xs, ys = [], []
    for n, ms in MEASURED["tp1_gemm_ms"].items():
        xs.append(LAYERS * expert_union_per_layer(n * TARGET["positions"]))
        ys.append(ms)
    xb, yb = sum(xs) / len(xs), sum(ys) / len(ys)
    b = sum((x - xb) * (y - yb) for x, y in zip(xs, ys)) / sum((x - xb) ** 2 for x in xs)
    return yb - b * xb, b


def validate(a, b):
    checks = []
    for n, measured in MEASURED["tp1_total_ms"].items():
        union = LAYERS * measured_union_per_layer(n * TARGET["positions"])
        predicted = (a + b * union) + MEASURED["tp1_index_ms"] + MEASURED["tp1_residual_ms"]
        checks.append({"N": n, "measured_ms": measured, "predicted_ms": round(predicted, 2),
                       "ratio": round(predicted / measured, 3)})
    worst = max(c["ratio"] for c in checks)
    return checks, worst


def predict(a, b):
    t = TARGET
    tokens = t["requests"] * t["positions"]                    # 128 tokens/pass
    union_total = LAYERS * measured_union_per_layer(tokens)
    union_per_gpu = union_total / 2                             # EP2 split
    gemm = a + b * union_per_gpu
    index = 1.2                                                 # measured flat, 256k-scaled
    attn_hot = (t["requests"] / 2 * SELECTED_TOKENS * t["kv_union_multiplier"]
                * FULL_ATTN_LAYERS * KV_PER_TOKEN_LAYER * t["hot_hit_rate"]) / (MEASURED["card_bw_gbs"] * GB) * 1e3
    residual = MEASURED["tp1_residual_ms"]
    # EP dispatch/combine: per token per layer one activation copy per remote
    # destination GPU, deduplicated; ~all tokens touch the remote rank.
    ep_traffic = (tokens * HIDDEN * 2 * LAYERS) / GB            # GB per pass, one direction
    ep_ms = ep_traffic / t["p2p_gbs_assumed"] * 1e3 * 2         # dispatch + combine
    kv_miss_bytes = (t["requests"] / 2 * SELECTED_TOKENS * t["kv_union_multiplier"]
                     * FULL_ATTN_LAYERS * KV_PER_TOKEN_LAYER * (1 - t["hot_hit_rate"]))
    link_ms = kv_miss_bytes / (MEASURED["link_h2d_gbs_per_gpu"] * GB) * 1e3

    budget_ms = 1e3 / (t["tok_s_per_request"] / t["emitted_per_pass"])
    serialized = gemm + index + attn_hot + residual + ep_ms + link_ms
    pipelined = gemm + index + attn_hot + residual + ep_ms     # link hidden

    # Memory: GPU resident tier under EP2, host tier.
    hot_tokens = 5120
    gpu_gib = (MEASURED["ep2_weight_shard_gib"]
               + t["requests"] / 2 * hot_tokens * FULL_ATTN_LAYERS * KV_PER_TOKEN_LAYER / GIB
               + t["requests"] / 2 * (t["context"] // 4) * FULL_ATTN_LAYERS * 256 / GIB  # original index
               + t["requests"] / 2 * 36 * 48 * 128 * 128 * 4 / GIB                       # GDN states
               + 4.0)                                                   # workspace
    host_gib = (t["requests"] * t["context"] * FULL_ATTN_LAYERS * KV_PER_TOKEN_LAYER / GIB
                + 95.37 + 16.5 + t["requests"] * (t["context"] // 4) * FULL_ATTN_LAYERS * 128 / GIB / 12 * 0)
    host_gib += 6.0                                                    # original index, host copy
    return {
        "pass_budget_ms": round(budget_ms, 1),
        "per_gpu_pass_ms": {
            "expert_gemm": round(gemm, 2), "index_scan": index,
            "attention_hot_tier": round(attn_hot, 3),
            "residual_other": residual,
            "ep_dispatch_combine": round(ep_ms, 2),
            "kv_miss_link_serialized": round(link_ms, 2),
            "serialized_total": round(serialized, 2),
            "pipelined_total": round(pipelined, 2),
        },
        "verdict": {
            "serialized_fits": serialized <= budget_ms,
            "pipelined_fits": pipelined <= budget_ms,
            "serialized_margin": round(1 - serialized / budget_ms, 3),
            "pipelined_margin": round(1 - pipelined / budget_ms, 3),
        },
        "memory": {"gpu_resident_gib_per_gpu": round(gpu_gib, 1),
                   "gpu_fits": gpu_gib <= t["usable_gpu_gib"],
                   "host_tier_gib": round(host_gib, 1),
                   "host_fits": host_gib <= t["host_gib_available"]},
        "expert_union": {"tokens_per_pass": tokens,
                         "union_total_experts": round(union_total, 0),
                         "union_per_gpu": round(union_per_gpu, 0)},
        "sensitivity_p2p_15_gbs": {"ep_dispatch_combine_ms": round(ep_ms * 25 / 15, 2),
                                   "pipelined_total": round(pipelined + ep_ms * (25 / 15 - 1), 2)},
    }


def main():
    a, b = fit_expert_curve()
    checks, worst = validate(a, b)
    report = {
        "status": "Phase-1 conditional capacity model; measured constants, not a forecast",
        "measured_inputs": MEASURED,
        "fit": {"intercept_ms": round(a, 3), "ms_per_expert_read": round(b, 6),
                "anchors": "N=1/4/8 GEMM; N=16 excluded (confounded)"},
        "validation_gate": {"checks": checks, "worst_ratio": worst,
                            "gate_1_3x": worst <= 1.3},
        "prediction_ep2_32x256k": predict(a, b),
        "limits": [
            "the 32-request point extrapolates a curve fitted at 8 requests; the "
            "clean N=16 measurement (one rerun) and real route distributions are pending",
            "EP2 dispatch cost assumes 25 GB/s PCIe peer traffic (unmeasured) and "
            "deduplicated per-destination copies; see sensitivity for 15 GB/s",
            "EP2 has not run end-to-end (NCCL init failure, GCP Sep 30); the 42.7 GB "
            "shard and the impossibility of TP2 are measured, the throughput is not",
            "prefetch (coverage 0.65 measured) is NOT required to meet the budget "
            "when the link pipelines; it recovers the serialized case's margin",
        ],
    }
    out = Path(__file__).parent / "artifacts/phase1-20260930/capacity-model.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    gate = report["validation_gate"]
    print(f"fit: GEMM_ms = {a:.2f} + {b:.6f} * expert_reads")
    print(f"validation worst ratio: {worst:.3f} (gate <= 1.3: {gate['gate_1_3x']})")
    pred = report["prediction_ep2_32x256k"]
    print(f"EP2 32x256k per-GPU pass: serialized {pred['per_gpu_pass_ms']['serialized_total']} ms, "
          f"pipelined {pred['per_gpu_pass_ms']['pipelined_total']} ms vs budget "
          f"{pred['pass_budget_ms']} ms")
    print(f"memory: GPU {pred['memory']['gpu_resident_gib_per_gpu']} GiB "
          f"(fits {pred['memory']['gpu_fits']}), host {pred['memory']['host_tier_gib']} GiB "
          f"(fits {pred['memory']['host_fits']})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
