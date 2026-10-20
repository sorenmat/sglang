"""CPU research probe, not an inference implementation or a GPU benchmark.

Run with an existing NumPy environment:
  OPENBLAS_NUM_THREADS=1 .venv/bin/python benchmark/context_certificate/probe.py

Checks exact top-k for QSA's sum-of-ReLU index score using conservative
low-precision key bounds. The joint bound uses all subsets of QSA's four
query heads to account for their shared key residual. Quantized arrays are
packed by a CPU codec, then decoded into float64 for scoring. Reported reads
remain modeled GPU traffic, not measured memory transactions or GPU timing.
"""

import json
import math

import numpy as np

from codec import decode, encode


def scores(keys, queries):
    return np.maximum(queries @ keys.T, 0).sum(axis=1) / math.sqrt(keys.shape[1])


def topk(values, count):
    # Equal scores use the lower logical ID, independent of physical layout.
    ids = np.arange(len(values))
    return np.lexsort((ids, -values))[:count]


def make_index(keys, bits):
    index = encode(keys, bits)
    return decode(index), index["radius"].astype(np.float64)


def certified_select(keys, queries, count, bits, bound_mode="basic", *,
                     valid_counts=None, return_details=False):
    keys = np.asarray(keys, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    if (keys.ndim != 2 or queries.ndim != 3 or not keys.size
            or not queries.size or queries.shape[2] != keys.shape[1]):
        raise ValueError("expected nonempty keys[N,D] and queries[S,H,D]")
    if not np.isfinite(keys).all() or not np.isfinite(queries).all():
        raise ValueError("keys and queries must be finite")
    if type(bits) is not int or bits not in (4, 6, 8):
        raise ValueError("bits must be 4, 6 or 8")
    if not isinstance(count, int) or not 1 <= count <= len(keys):
        raise ValueError("top-k must be in [1,N]")
    if valid_counts is None:
        valid_counts = np.full(len(queries), len(keys), dtype=np.int64)
    valid_counts = np.asarray(valid_counts)
    if (valid_counts.shape != (len(queries),)
            or not np.issubdtype(valid_counts.dtype, np.integer)
            or np.any(valid_counts < count) or np.any(valid_counts > len(keys))):
        raise ValueError("valid_counts must contain S integers in [top-k,N]")
    valid = np.arange(len(keys))[None, :] < valid_counts[:, None]
    coarse, radius = make_index(keys, bits)
    if not np.isfinite(coarse).all() or not np.isfinite(radius).all():
        raise ValueError("quantization overflow: input outside reference range")
    dot = queries @ coarse.T
    # Numerical guard for this float64 REFERENCE, not a CUDA error proof.
    guard = 1e-11 * (1 + np.linalg.norm(queries, axis=2)[:, :, None]
                     * np.linalg.norm(keys, axis=1))
    if bound_mode == "basic":
        error = np.linalg.norm(queries, axis=2)[:, :, None] * radius + guard
        lower = np.maximum(dot - error, 0).sum(axis=1)
        upper = np.maximum(dot + error, 0).sum(axis=1)
    elif bound_mode == "joint":
        heads = queries.shape[1]
        if heads > 4:
            raise ValueError("joint subset probe supports at most four QSA heads")
        lower = np.zeros((len(queries), len(keys)))
        upper = np.zeros_like(lower)
        # sum_h ReLU(q_h @ k) = max_S ((sum_{h in S} q_h) @ k).
        # One residual vector serves all heads, so its worst-case direction
        # cannot be independently adversarial for each head. Visit the 16
        # subsets without materializing a [positions, 16, keys] tensor.
        for mask in range(1, 1 << heads):
            selected = [h for h in range(heads) if mask & (1 << h)]
            subset_dot = dot[:, selected].sum(axis=1)
            subset_query = queries[:, selected].sum(axis=1)
            error = np.linalg.norm(subset_query, axis=1)[:, None] * radius
            np.maximum(lower, subset_dot - error, out=lower)
            np.maximum(upper, subset_dot + error, out=upper)
        # Covers float64 reference differences in the probe, not a proof
        # for tensor-core accumulation, TF32, or production CUDA kernels.
        lower -= 2 * guard.sum(axis=1)
        upper += 2 * guard.sum(axis=1)
    else:
        raise ValueError(f"unknown bound mode: {bound_mode}")
    lower /= math.sqrt(keys.shape[1])
    upper /= math.sqrt(keys.shape[1])
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("bound overflow: input outside reference range")
    lower = np.where(valid, lower, -np.inf)
    upper = np.where(valid, upper, -np.inf)
    threshold = np.partition(lower, -count, axis=1)[:, -count]
    candidates = valid & (upper >= threshold[:, None])
    union = np.flatnonzero(candidates.any(axis=0))
    refined = scores(keys[union], queries)
    picks = []
    for row in range(len(queries)):
        local = np.flatnonzero(candidates[row, union])
        # union/local are sorted logical IDs, preserving ties.
        picks.append(union[local[topk(refined[row, local], count)]])
    reference = np.where(valid, scores(keys, queries), -np.inf)
    if (not np.all(reference >= lower - 1e-10)
            or not np.all(reference <= upper + 1e-10)):
        raise AssertionError("CPU reference score escaped an interval")
    for row, picked in enumerate(picks):
        if not np.array_equal(picked, topk(reference[row], count)):
            raise AssertionError("certificate disagrees with CPU reference top-k")
    # fp32 scale + upward-rounded fp32 residual norm: 8 B/key, no hidden metadata.
    coarse_bytes = len(keys) * (math.ceil(keys.shape[1] * bits / 8) + 8)
    refined_bytes = len(union) * keys.shape[1] * 2  # original BF16 keys
    full_bytes = keys.size * 2
    result = {
        "bits": bits,
        "bound_mode": bound_mode,
        "positions": len(queries),
        "exact_topk_match": True,
        "refine_fraction_per_position": (candidates.sum(axis=1) / valid_counts).round(6).tolist(),
        "refine_union_fraction": round(len(union) / len(keys), 6),
        "packed_coarse_read_bytes": int(coarse_bytes),
        "original_key_read_bytes": int(refined_bytes),
        "ratio_to_one_shared_bf16_index_scan": round((coarse_bytes + refined_bytes)
                                                    / full_bytes, 6),
        "fallback_if_more_bytes_than_full_scan": coarse_bytes + refined_bytes > full_bytes,
    }
    if return_details:
        result["details"] = {"selected_ids": np.asarray(picks),
                             "candidates": candidates, "lower": lower,
                             "upper": upper, "reference_scores": reference}
    return result


def hardware_budget():
    batch, context, layers, kv_heads, dim = 32, 262144, 12, 2, 256
    gib = 2**30
    kv_bf16 = batch * context * layers * kv_heads * dim * 2 * 2
    index_original = batch * (context // 4) * layers * 128 * 2
    index_compact = batch * (context // 4) * layers * (128 + 8)
    selected = batch * layers * 2051 * kv_heads * dim * 2 * 2
    expert_params = 48 * 512 * 3 * 2560 * 640
    wave = []
    for positions, acceptance in [(1, 1.0), (4, 0.9), (8, 0.9), (12, 0.9), (16, 0.9)]:
        # positions counts target input positions. 1 + accepted proposals
        # produces at most positions tokens, INCLUDING the correction/bonus.
        emitted = sum(acceptance**i for i in range(positions))
        unique_fraction = 1 - (1 - 10 / 512) ** (batch * positions)
        expert_bytes = expert_params * (0.5 + 1 / 16) * unique_fraction
        wave.append({
            "target_positions": positions,
            "iid_conditional_acceptance_assumption": acceptance,
            "expected_emitted_per_request_including_bonus": round(emitted, 6),
            "expected_experts_per_layer_uniform_independent_assumption": round(512 * unique_fraction, 6),
            "routed_expert_GB_per_verify_pass": round(expert_bytes / 1e9, 6),
            "routed_expert_TBps_for_80_per_request": round(expert_bytes * 80 / emitted / 1e12, 6),
        })
    assert kv_bf16 == 192 * gib
    assert index_original == 6 * gib
    return {
        "batch": batch, "context": context,
        "full_bf16_kv_GiB": kv_bf16 / gib,
        "original_bf16_index_GiB": index_original / gib,
        "eight_bit_index_with_fp32_scale_and_radius_GiB": index_compact / gib,
        "bf16_hot_kv_5120_positions_per_request_GiB": kv_bf16 * 5120 / context / gib,
        "bf16_selected_kv_read_GB_per_accepted_wave_no_reuse": selected / 1e9,
        "bf16_selected_kv_read_GBps_at_80_no_reuse": selected * 80 / 1e9,
        "gdn_fp32_single_state_GiB": 32 * 36 * 48 * 128 * 128 * 4 / gib,
        "routed_expert_params": expert_params,
        "routed_expert_NVFP4_with_scales_GiB": expert_params * 0.5625 / gib,
        "speculation_scenarios_NOT_measurements": wave,
    }


def self_checks():
    rng = np.random.default_rng(991)
    checked = 0
    # Exercise signed queries, exact ties, zeros, all keys selected, singletons.
    for n, d, h, k in [(1, 1, 1, 1), (37, 7, 4, 9), (64, 16, 4, 64), (128, 8, 4, 17)]:
        keys = rng.normal(size=(n, d))
        if n > 1:
            keys[: n // 3] = keys[0]
        queries = rng.normal(size=(3, h, d))
        queries[0] = 0
        for bits in (4, 6, 8):
            for bound_mode in ("basic", "joint"):
                certified_select(keys, queries, k, bits, bound_mode)
                checked += 1
    for trial in range(12):
        keys = rng.normal(size=(257, 32)) * 10 ** (trial % 5 - 2)
        queries = rng.normal(size=(4, 4, 32))
        for bound_mode in ("basic", "joint"):
            certified_select(keys, queries, 19, 8, bound_mode)
            checked += 1
    return f"{checked} adversarial/randomized configurations passed bounds and exact top-k"


def main():
    rng = np.random.default_rng(20260925)
    keys = rng.normal(size=(65536, 128))
    keys *= math.sqrt(128) / np.linalg.norm(keys, axis=1, keepdims=True)
    # Truncate mantissa to BF16 so the archive represents genuine BF16 values.
    raw = keys.astype(np.float32).view(np.uint32)
    keys = (raw & np.uint32(0xFFFF0000)).view(np.float32).astype(np.float64)
    anchor = rng.normal(size=(1, 4, 128))
    cases = {
        "single_query": anchor,
        "eight_queries_small_drift": anchor + 0.02 * rng.normal(size=(8, 4, 128)),
        "eight_independent_queries": rng.normal(size=(8, 4, 128)),
        "twelve_independent_queries": rng.normal(size=(12, 4, 128)),
        "all_scores_tied": np.zeros((1, 4, 128)),
    }
    results = []
    for name, queries in cases.items():
        for bits in (4, 6, 8):
            for bound_mode in ("basic", "joint"):
                results.append({
                    "case": name,
                    **certified_select(keys, queries, 512, bits, bound_mode),
                })
    print(json.dumps({
        "status": "synthetic CPU math check; no model quality or GPU speed measured",
        "seed": 20260925,
        "numpy_version": np.__version__,
        "packing_note": "CPU packed codec implemented for 4/6/8 bits; scoring uses decoded float64 arrays. Modeled GPU reads, not measured traffic; no GPU codec/kernel. Scale and radius cost 8 bytes per key.",
        "checks": self_checks(),
        "budget": hardware_budget(),
        "synthetic_index_results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
