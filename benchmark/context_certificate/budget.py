#!/usr/bin/env python3
"""Conditional FlashNext capacity/traffic ledger; no model execution or forecast.

Example: python budget.py --positions 12 --conditional-acceptance .9 \
    --refinement-fraction .1 --kv-union-multiplier 3 --kv-hit-rate .8
All capacities are binary GiB; bandwidth and explicitly named GB use decimal GB.
"""

import argparse
from dataclasses import asdict, dataclass
import json
import math

GIB = 2**30
GB = 10**9
EXPERT_PARAMS = 48 * 512 * 3 * 2560 * 640
KV_BYTES_PER_TOKEN_PER_LAYER = 2 * 2 * 256 * 2
INSPECTED_RUNTIME_COMMIT = "4ccff141dbe992794f9da6c3aa23535b4f72000d"


@dataclass
class Scenario:
    concurrency: int = 32
    context: int = 262144
    gpu_count: int = 1
    positions: int = 12
    emitted_per_pass: float | None = None
    conditional_acceptance: float | None = None
    weights_total_gib: float = 78.0
    usable_gpu_gib: float = 94.2
    host_gib: float = 512.0
    workspace_per_gpu_gib: float = 4.0
    hot_tokens: int = 5120
    index_placement: str = "compact-with-host-refinement"
    refinement_fraction: float = 1.0
    kv_union_multiplier: float | None = None
    kv_hit_rate: float = 0.0
    reuse_kv_on_gpu: bool = False
    other_gpu_gb_per_pass: float = 0.0
    gpu_bandwidth_gbs: float = 1000.0
    host_link_gbs: float = 20.0
    target_tps_per_request: float = 80.0


def calculate(s: Scenario) -> dict:
    for name in ("concurrency", "context", "gpu_count", "positions"):
        value = getattr(s, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if s.gpu_count not in (1, 2):
        raise ValueError("this fixed two-KV-head model supports only ideal TP1 or TP2")
    if s.index_placement not in ("compact-with-host-refinement", "original-on-gpu"):
        raise ValueError("invalid index_placement")
    original_resident = s.index_placement == "original-on-gpu"
    if type(s.hot_tokens) is not int or s.hot_tokens < 0:
        raise ValueError("hot_tokens must be a nonnegative integer")
    for name in ("weights_total_gib", "usable_gpu_gib", "host_gib",
                 "workspace_per_gpu_gib", "other_gpu_gb_per_pass"):
        value = getattr(s, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for name in ("gpu_bandwidth_gbs", "host_link_gbs", "target_tps_per_request"):
        value = getattr(s, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("refinement_fraction", "kv_hit_rate", "conditional_acceptance"):
        value = getattr(s, name)
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{name} must be in [0, 1]")
    if (s.emitted_per_pass is None) == (s.conditional_acceptance is None):
        raise ValueError("supply exactly one of emitted_per_pass or conditional_acceptance")
    emitted = s.emitted_per_pass
    if emitted is None:
        p = s.conditional_acceptance
        emitted = s.positions if p == 1 else (1 if p == 0 else
                  -math.expm1(s.positions * math.log(p)) / (1 - p))
    if not math.isfinite(emitted) or not 1 <= emitted <= s.positions:
        raise ValueError("emitted_per_pass must be in [1, positions], including bonus")
    union_multiplier = s.positions if s.kv_union_multiplier is None else s.kv_union_multiplier
    if not math.isfinite(union_multiplier) or not 1 <= union_multiplier <= s.positions:
        raise ValueError("kv_union_multiplier must be in [1, positions]")

    batch, context = s.concurrency, s.context
    full_kv = batch * context * 12 * KV_BYTES_PER_TOKEN_PER_LAYER
    index_keys = batch * (context // 4) * 12
    original_index = index_keys * 128 * 2
    compact_index = index_keys * (128 + 8)  # int8 + FP32 scale + FP32 radius
    gpu_index = original_index if original_resident else compact_index
    hot_kv = batch * min(context, s.hot_tokens) * 12 * KV_BYTES_PER_TOKEN_PER_LAYER
    gdn = batch * 36 * 48 * 128 * 128 * 4
    host_items = {
        "target_bf16_kv": full_kv / GIB,
        "target_bf16_index": original_index / GIB,
        "original_bf16_ple": 95.37,
        "draft_one_qsa_layer_bf16_kv_and_index": (full_kv + original_index) / 12 / GIB,
    }
    sharded_items = {
        "main_and_draft_weights_total": s.weights_total_gib,
        ("target_bf16_index" if original_resident else
         "target_int8_index_with_fp32_scale_and_radius"): gpu_index / GIB,
        "target_hot_bf16_kv": hot_kv / GIB,
        ("draft_one_qsa_layer_bf16_index_and_hot_kv" if original_resident else
         "draft_one_qsa_layer_int8_index_and_hot_kv"): (gpu_index + hot_kv) / 12 / GIB,
        "committed_fp32_gdn": gdn / GIB,
        "one_layer_gdn_scratch": gdn / 36 / GIB,
        "recurrence_and_metadata_reserve": .7 * batch / 32 * max(1, s.positions / 12),
    }
    gpu_items = {name: value / s.gpu_count for name, value in sharded_items.items()}
    gpu_items["workspace_per_gpu"] = s.workspace_per_gpu_gib
    gpu_total, host_total = sum(gpu_items.values()), sum(host_items.values())

    # Same hypothetical uniform IID expert union as probe.hardware_budget().
    expert_fraction = -math.expm1(batch * s.positions * math.log1p(-10 / 512))
    selected_per_position = min(context, 2051)  # 2048 selected + up to 3 unfinished
    selected_union_tokens = min(context, selected_per_position * union_multiplier)
    selected_once = batch * selected_per_position * 12 * KV_BYTES_PER_TOKEN_PER_LAYER
    selected_all = selected_once * s.positions
    selected_union = batch * selected_union_tokens * 12 * KV_BYTES_PER_TOKEN_PER_LAYER
    entry_hit_limit = min(1.0, min(context, s.hot_tokens) / selected_union_tokens)
    refined_index = 0 if original_resident else original_index * s.refinement_fraction
    host_kv_miss = selected_union * (1 - s.kv_hit_rate)
    target_ple = batch * s.positions * 2560 * 2
    h2d_items = {
        "original_index_union": refined_index,
        "selected_original_kv_union_misses": host_kv_miss,
        "target_ple_gather_payload": target_ple,
    }
    h2d_total = sum(h2d_items.values())
    gpu_traffic = {
        "target_routed_experts_uniform_iid_union": EXPERT_PARAMS * .5625 * expert_fraction,
        "target_compact_index_one_shared_scan": 0 if original_resident else compact_index,
        "target_resident_bf16_index_one_shared_scan": original_index if original_resident else 0,
        "target_original_index_one_shared_refinement_read": refined_index,
        "target_selected_kv_reads": selected_union if s.reuse_kv_on_gpu else selected_all,
        "h2d_destination_writes": h2d_total,
        "target_ple_gather_reads": target_ple,
        "explicit_other_gpu_traffic": s.other_gpu_gb_per_pass * GB,
    }
    gpu_bytes = sum(gpu_traffic.values())
    gpu_floor = gpu_bytes / s.gpu_count / (s.gpu_bandwidth_gbs * GB)
    link_floor = h2d_total / (s.host_link_gbs * GB)
    floor = max(gpu_floor, link_floor)  # Optimistic complete GPU/link overlap.
    passes_per_second = s.target_tps_per_request / emitted
    emitted_batch = emitted * batch
    minimum_refinement_fraction = min(512, context // 4) / (context // 4) if context >= 4 else 0
    refinement_covers_selection = s.refinement_fraction >= minimum_refinement_fraction
    if original_resident:
        minimum_refinement_fraction = refinement_covers_selection = None
    window_guard_passes = s.positions <= 4
    warnings = []
    if floor > emitted / s.target_tps_per_request:
        warnings.append("Even the optimistic traffic latency floor exceeds the target pass time under these assumptions.")
    if not window_guard_passes:
        warnings.append("This verification window is unsupported by the inspected QSA runtime: "
                        "only chain topk=1 and draft_token_num <= compress_ratio=4 are allowed. "
                        "Positions above four require pending-index state/write-plan redesign "
                        "to avoid ring aliasing; the calculation remains hypothetical.")
    if gpu_total > s.usable_gpu_gib:
        warnings.append("Modeled GPU allocation exceeds usable capacity.")
    if host_total > s.host_gib:
        warnings.append("Static original-precision archive exceeds host RAM.")
    if s.kv_hit_rate > entry_hit_limit:
        warnings.append("Assumed KV union hit rate exceeds pass-entry hot-cache capacity; "
                        "this transfer scenario is infeasible without another residency mechanism.")
    if refinement_covers_selection is False:
        warnings.append("Refinement union is smaller than one position's selected index records; "
                        "this scenario needs an unmodeled original-index cache or is infeasible.")
    return {
        "status": "conditional analytical scenario, not measured throughput or a forecast",
        "units": {"GiB": GIB, "GB": GB, "bandwidth_GB_per_second": "decimal"},
        "inputs": asdict(s),
        "speculation": {
            "emitted_per_request_per_pass_including_bonus": emitted,
            "basis": "explicit assumed mean" if s.emitted_per_pass is not None else
                     "hypothetical constant conditional acceptance: sum(p**i, i=0..S-1)",
        },
        "runtime_compatibility": {
            "inspected_commit": INSPECTED_RUNTIME_COMMIT,
            "source": "python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py:263-278",
            "assumed_chain_topk": 1,
            "maximum_positions_without_pending_ring_redesign": 4,
            "passes_inspected_window_guard": window_guard_passes,
            "requires_pending_index_state_and_write_plan_redesign": not window_guard_passes,
            "note": "Passing this guard establishes neither support for the proposed backend nor measured performance.",
        },
        "memory_GiB": {
            "host_items": host_items, "host_static_total": host_total,
            "host_headroom_before_unmodeled_buffers": s.host_gib - host_total,
            "ideal_TP_per_gpu_items": gpu_items, "per_gpu_total": gpu_total,
            "per_gpu_headroom": s.usable_gpu_gib - gpu_total,
            "host_static_fits": host_total <= s.host_gib,
            "gpu_modeled_fits": gpu_total <= s.usable_gpu_gib,
        },
        "traffic_bytes_per_pass": {
            "all_gpus_items": gpu_traffic, "all_gpus_total": gpu_bytes,
            "ideal_TP_per_gpu_total": gpu_bytes / s.gpu_count,
            "host_to_device_items": h2d_items, "host_to_device_total": h2d_total,
            "selected_kv_all_positions_without_read_reuse": selected_all,
            "selected_kv_unique_union": selected_union,
        },
        "traffic_assumptions": {
            "uniform_iid_expected_experts_per_layer": 512 * expert_fraction,
            "kv_union_tokens_per_request_per_layer": selected_union_tokens,
            "maximum_pass_entry_kv_hit_rate_from_hot_capacity": entry_hit_limit,
            "kv_hit_rate_within_capacity": s.kv_hit_rate <= entry_hit_limit,
            "index_refinement_fraction_is_union_across_positions": not original_resident,
            "index_refinement_fraction_ignored": original_resident,
            "index_refinement_constraint_applicable": not original_resident,
            "minimum_refinement_union_fraction_without_original_index_cache": minimum_refinement_fraction,
            "index_refinement_covers_minimum_selected_records": refinement_covers_selection,
        },
        "bytes_per_emitted_token_all_requests": {
            "all_gpus": gpu_bytes / emitted_batch, "host_to_device": h2d_total / emitted_batch,
        },
        "required_GB_per_second_at_target": {
            "per_gpu": gpu_bytes / s.gpu_count * passes_per_second / GB,
            "host_link_aggregate": h2d_total * passes_per_second / GB,
        },
        "conditional_latency_floor_ms_per_pass": {
            "gpu_traffic_only": 1000 * gpu_floor,
            "host_link_payload_only": 1000 * link_floor,
            "perfect_overlap": 1000 * floor,
            "allowed_at_target": 1000 * emitted / s.target_tps_per_request,
        },
        "fits_conditional_traffic_time_budget": floor <= emitted / s.target_tps_per_request,
        "assumptions_and_exclusions": [
            "Full original target/draft KV and index and BF16 PLE remain on host; "
            "TP and original-on-GPU placement do not reduce the retained host archive. "
            "PLE size is rounded to 95.37 GiB.",
            "Weights are TOTAL main+draft resident GPU allocation, ideally divided by TP count. "
            "All GPU components except per-GPU workspace are ideally sharded; no replication or imbalance.",
            "Memory covers current context only; future generation, prefill/loading/process buffers, "
            "allocator padding and graph growth must be measured separately.",
            "GDN uses deferred commit/replay, not S state snapshots. Reserve is an allowance, "
            "scaled linearly with batch and conservatively beyond 12 positions.",
            "Target routes choose 10 of 512 experts uniformly and independently per position; "
            "each used NVFP4 weight and scale is read once. Actual routes and GEMM rereads differ.",
            "Original-on-GPU indexing scans BF16 keys once with no compact scan or host refinement; "
            "otherwise compact index and union-refinement reads are each shared once across verifier positions. "
            "selected KV reads share only when reuse_kv_on_gpu is true. These require suitable kernels.",
            "H2D payload is union refinement plus KV union misses plus target PLE gathers; "
            "link bandwidth is shared by the entire host, not multiplied by GPU count.",
            "Traffic excludes draft execution, non-routed target weights, GDN replay/state traffic, "
            "activations, selection scratch, D2H writes, transfer/page amplification and TP collectives "
            "except any explicit other GPU traffic allowance.",
            "Latency floor assumes perfect transfer/compute overlap and ideal TP. It excludes FLOPs, "
            "host DRAM limits, launch latency, layer dependencies and communication; it is not a speed prediction.",
        ],
        "warnings": warnings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("concurrency", "context", "gpu_count", "positions", "hot_tokens"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(Scenario, name))
    for name in ("weights_total_gib", "usable_gpu_gib", "workspace_per_gpu_gib",
                 "refinement_fraction", "kv_union_multiplier", "kv_hit_rate",
                 "other_gpu_gb_per_pass", "gpu_bandwidth_gbs", "host_link_gbs",
                 "target_tps_per_request"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(Scenario, name))
    host = parser.add_mutually_exclusive_group()
    host.add_argument("--host-gib", type=float)
    host.add_argument("--host-gb", type=float, help="decimal host capacity, converted to GiB")
    acceptance = parser.add_mutually_exclusive_group(required=True)
    acceptance.add_argument("--emitted-per-pass", type=float, help="includes correction/bonus token")
    acceptance.add_argument("--conditional-acceptance", type=float, help="hypothetical constant probability")
    parser.add_argument("--reuse-kv-on-gpu", action="store_true")
    parser.add_argument("--index-placement", choices=("compact-with-host-refinement", "original-on-gpu"),
                        default=Scenario.index_placement)
    args = vars(parser.parse_args())
    host_gb = args.pop("host_gb")
    if host_gb is not None:
        args["host_gib"] = host_gb * GB / GIB
    elif args["host_gib"] is None:
        args["host_gib"] = Scenario.host_gib
    try:
        result = calculate(Scenario(**args))
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
