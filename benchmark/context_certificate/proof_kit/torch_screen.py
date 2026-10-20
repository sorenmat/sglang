"""Torch port of the certified QSA index screen for proof measurements.

Two modes:

``--check`` runs on CPU in float64 and must reproduce the NumPy probe's
selected IDs bit-for-bit on the same capture (the probe's agreement with the
native selector is already established). This validates the ported math, not
GPU rounding.

``--bench`` runs on CUDA at target working-set scale (16 request banks of
65,536 compressed keys tiled from the real captures) and measures wall time
of a full BF16 scan against the screened path, bracketing the screened cost:

* ``screened_measured`` includes an int8->fp32 decode that pure torch must
  materialize, so it is an UPPER bound on a fused production kernel;
* ``screened_floor`` applies the byte model at the measured streaming
  bandwidth, so it is a LOWER bound (optimistic for screening).

It also compares screened picks against the full-scan picks ON THE GPU.
Any mismatch means the certified interval let a winner escape under CUDA
arithmetic: that is evidence, still not a proof, about GPU rounding.
"""

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:  # reference implementation; only --check needs it
    from probe import certified_select
except ImportError:
    certified_select = None


SUBSETS = [tuple(h for h in range(4) if mask & (1 << h)) for mask in range(1, 16)]


def encode_index(keys, bits):
    """Offset-binary quantization matching codec.encode, kept as codes."""
    maximum = 2 ** (bits - 1) - 1
    scale = (keys.abs().amax(dim=1) / maximum).to(torch.float32)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quantized = torch.round(keys / scale[:, None].to(keys.dtype)).clamp_(-maximum, maximum)
    coarse = quantized * scale[:, None].to(keys.dtype)
    radius = (keys - coarse).norm(dim=1).to(torch.float32)
    radius = torch.nextafter(radius, torch.full_like(radius, float("inf")))
    return {"codes": quantized.to(torch.int8), "scale": scale, "radius": radius}


def certified_picks(keys, queries, count, index, *, guard=True, valid=None):
    """Joint-subset certified screen; returns per-position selected IDs."""
    coarse = index["codes"].to(keys.dtype) * index["scale"][:, None].to(keys.dtype)
    positions, heads, dim = queries.shape
    lower = torch.zeros(positions, len(keys), dtype=keys.dtype, device=keys.device)
    upper = torch.zeros_like(lower)
    for heads_in_subset in SUBSETS[: (1 << heads) - 1]:
        subset_query = queries[:, list(heads_in_subset), :].sum(dim=1)
        dot = subset_query @ coarse.T
        error = subset_query.norm(dim=1)[:, None] * index["radius"][None, :].to(keys.dtype)
        torch.maximum(lower, dot - error, out=lower)
        torch.maximum(upper, dot + error, out=upper)
    if guard:  # float64 reference guard, identical role to probe.py's
        per_head = queries @ coarse.T
        key_norm = keys.norm(dim=1)
        slack = (queries.norm(dim=2)[:, :, None] * key_norm[None, :] + 1.0) * 1e-11
        lower -= 2 * slack.sum(dim=1)
        upper += 2 * slack.sum(dim=1)
    lower /= math.sqrt(dim)
    upper /= math.sqrt(dim)
    if valid is not None:
        lower = torch.where(valid, lower, torch.full_like(lower, -float("inf")))
        upper = torch.where(valid, upper, torch.full_like(upper, -float("inf")))
    threshold = torch.topk(lower, count, dim=1).values[:, -1:]
    candidates = upper >= threshold
    union = torch.nonzero(candidates.any(dim=0)).flatten()
    refined = torch.relu(queries @ keys[union].T).sum(dim=1) / math.sqrt(dim)
    picks = []
    for row in range(positions):
        local = torch.nonzero(candidates[row, union]).flatten()
        order = torch.argsort(-refined[row, local], stable=True)[:count]
        picks.append(union[local[order]])
    return torch.stack(picks), union


def exact_scores(keys, queries):
    return torch.relu(queries @ keys.T).sum(dim=1) / math.sqrt(keys.shape[1])


def check_against_probe(capture_paths):
    failures = 0
    for path in capture_paths:
        with np.load(path, allow_pickle=False) as archive:
            keys = torch.from_numpy(archive["keys"].astype(np.float64))
            queries = torch.from_numpy(archive["queries"].astype(np.float64))
            native = torch.from_numpy(archive["native_selected_ids"])
            valid_counts = archive["valid_counts"]
        positions = len(queries)
        valid = torch.arange(len(keys))[None, :] < torch.from_numpy(valid_counts)[:, None]
        for bits in (8, 6, 4):
            index = encode_index(keys, bits)
            picks, union = certified_picks(keys, queries, 512, index, valid=valid)
            reference = certified_select(
                keys.numpy(), queries.numpy(), 512, bits, "joint",
                valid_counts=valid_counts, return_details=True)
            details = reference.pop("details")
            match = bool(torch.equal(picks, torch.from_numpy(details["selected_ids"])))
            union_match = len(union) == int(reference["refine_union_fraction"] * len(keys)) or \
                abs(len(union) / len(keys) - reference["refine_union_fraction"]) < 5e-4
            # The native selector's tie ORDER differs; set equality is the
            # equivalence the replay already established.
            native_match = bool(torch.equal(
                picks.sort(dim=1).values, native.sort(dim=1).values))
            status = "ok" if (match and union_match and native_match) else "MISMATCH"
            failures += status != "ok"
            print(f"{Path(path).name[:18]} bits={bits} picks==probe:{match} "
                  f"union==probe:{union_match} picks==native:{native_match} {status}")
    return failures


def _time_cuda(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(np.median(times)), float(np.min(times)), float(np.max(times))


def bench(capture_paths, bank_keys=65536, banks=16, bits=8, count=512):
    device = torch.device("cuda")
    with np.load(capture_paths[0], allow_pickle=False) as archive:
        source_keys = archive["keys"]
        queries = torch.from_numpy(archive["queries"]).to(device, torch.bfloat16)
    rng = np.random.default_rng(20260929)
    keys_np = source_keys[rng.integers(0, len(source_keys), size=(banks, bank_keys))]
    keys = [torch.from_numpy(bank).to(device, torch.bfloat16) for bank in keys_np]
    indexes = []
    for bank in keys:
        idx = encode_index(bank.to(torch.float64), bits)
        indexes.append({k: v.to(device) for k, v in idx.items()})

    def full_scan():
        picks = []
        for bank in keys:
            scores = exact_scores(bank, queries)
            picks.append(torch.stack([row.argsort(descending=True, stable=True)[:count]
                                      for row in scores]))
        return picks

    def screened():
        picks = []
        for bank, index in zip(keys, indexes):
            picks.append(certified_picks(bank, queries, count, index, guard=False)[0])
        return picks

    full_ms = _time_cuda(lambda: full_scan())
    screen_ms = _time_cuda(lambda: screened())

    # Baseline decomposition: bandwidth part (score production only) vs
    # selection part, both on precomputed tensors.
    scan_only = _time_cuda(lambda: [exact_scores(bank, queries).sum() for bank in keys])
    precomputed = [exact_scores(bank, queries) for bank in keys]
    selection_only = _time_cuda(
        lambda: [torch.stack([row.argsort(descending=True, stable=True)[:count]
                              for row in scores]) for scores in precomputed])

    # Streaming-bandwidth floor via an fp32 reduction over 512 MB.
    stream = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=device)
    stream_ms = _time_cuda(lambda: stream.sum())
    stream_bps = stream.numel() * 4 / (stream_ms[0] / 1e3)
    # Byte model at measured bandwidth and at GDDR7 spec for this card.
    coarse_bytes = bank_keys * (128 + 8) * banks
    refine_union = 0.15  # measured 8-bit joint range on real captures
    floor_bytes = int(coarse_bytes + refine_union * bank_keys * 256 * banks)
    floor_ms_measured = floor_bytes / stream_bps * 1e3
    floor_ms_spec = floor_bytes / 1.79e12 * 1e3

    full_picks = full_scan()
    screen_picks = screened()
    mismatches = sum(int(not torch.equal(a, b)) for a, b in zip(full_picks, screen_picks))

    report = {
        "status": "GPU measurement; not a quality result",
        "device": torch.cuda.get_device_name(0),
        "bits": bits, "banks": banks, "bank_keys": bank_keys, "top_k": count,
        "bf16_full_scan_ms_median": full_ms[0],
        "bf16_scan_only_ms_median": scan_only[0],
        "bf16_selection_only_ms_median": selection_only[0],
        "screened_measured_ms_median_upper_bound": screen_ms[0],
        "screened_ratio_measured": round(screen_ms[0] / full_ms[0], 4),
        "stream_gbps_fp32_reduce": round(stream_bps / 1e9, 1),
        "screened_floor_bytes": int(floor_bytes),
        "screened_floor_ms_at_measured_bandwidth": round(floor_ms_measured, 4),
        "screened_floor_ms_at_spec_1790_GBps": round(floor_ms_spec, 4),
        "gpu_pick_mismatched_banks_of_16": mismatches,
        "limits": [
            "screened_measured includes an int8->fp32 decode torch materializes; "
            "a fused kernel would not pay it",
            "the floor is a byte model priced at measured fp32-reduce bandwidth "
            "and at GDDR7 spec; neither includes selection compute",
            "pick mismatches flag CUDA rounding escapes of the certified interval; "
            "absence of mismatches is evidence, not a rounding proof",
        ],
    }
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "bench"))
    parser.add_argument("captures", nargs="+", type=Path)
    parser.add_argument("--bank-keys", type=int, default=65536)
    parser.add_argument("--banks", type=int, default=16)
    parser.add_argument("--bits", type=int, default=8, choices=(8, 6, 4))
    args = parser.parse_args()
    if args.mode == "check":
        raise SystemExit(1 if check_against_probe(args.captures) else 0)
    bench(args.captures, args.bank_keys, args.banks, args.bits)


if __name__ == "__main__":
    main()
