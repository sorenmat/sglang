"""Cross-layer overlap of native QSA selections, for prefetch sizing.

Input: two capture NPZ files from the SAME request captured at DIFFERENT QSA
layers (the overlay accepts comma-separated SGLANG_QSA_CAPTURE_LAYERS).
Verifications are aligned by request ID and query position. For each aligned
verification this reports, per position and for the position-union:

- jaccard(A, B): overlap of the two layers' native top-512 selections
- coverage(B in A): the fraction of layer-B blocks already present in
  layer-A's selection — the prefetch hit rate if A's blocks are prefetched
  for B
- overfetch(A not in B): the fraction of prefetched blocks B never uses

These are descriptive statistics of two layers of one request; they size a
prefetch experiment, they do not prove a prefetch design.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(archive["metadata"].item())
        return {
            "request_id": meta["request_id"],
            "layer_id": meta["layer_id"],
            "positions": archive["query_positions"],
            "selected": archive["native_selected_ids"],
        }


def pair_stats(a_ids, b_ids):
    a, b = set(a_ids.tolist()), set(b_ids.tolist())
    union = a | b
    return {
        "jaccard": round(len(a & b) / len(union), 4),
        "coverage_of_b_by_a": round(len(a & b) / len(b), 4),
        "overfetch_fraction_of_a": round(len(a - b) / len(a), 4),
    }


def overlap(early_path, late_path):
    early, late = load(early_path), load(late_path)
    if early["request_id"] != late["request_id"]:
        raise ValueError("captures must come from the same request")
    if early["layer_id"] == late["layer_id"]:
        raise ValueError("captures must come from different layers")
    by_pos = {}
    for capture in (early, late):
        for row, pos in enumerate(capture["positions"]):
            by_pos.setdefault(int(pos), {})[capture["layer_id"]] = capture["selected"][row]
    per_position, per_union = [], []
    steps = sorted(pos for pos, layers in by_pos.items() if len(layers) == 2)
    step_size = len(early["positions"])
    for start in range(0, len(steps) - step_size + 1, step_size):
        window = steps[start:start + step_size]
        union = {layer: np.concatenate([by_pos[p][layer] for p in window])
                 for layer in by_pos[window[0]]}
        per_union.append(pair_stats(union[early["layer_id"]], union[late["layer_id"]]))
        for p in window:
            per_position.append(pair_stats(by_pos[p][early["layer_id"]],
                                           by_pos[p][late["layer_id"]]))
    if not per_position:
        raise ValueError("captures share no aligned verification window")

    def mean(rows, key):
        return round(sum(r[key] for r in rows) / len(rows), 4)

    return {
        "status": "descriptive overlap of two layers of one request",
        "early_layer": early["layer_id"],
        "late_layer": late["layer_id"],
        "aligned_positions": len(per_position),
        "aligned_verifications": len(per_union),
        "per_position_mean": {k: mean(per_position, k) for k in per_position[0]},
        "per_union_mean": {k: mean(per_union, k) for k in per_union[0]},
        "per_position": per_position,
        "per_union": per_union,
        "limits": [
            "two layers of one request; layer set and request diversity change the numbers",
            "coverage sizes a hit rate only if prefetch latency hides behind compute",
            "selections are logical compressed-block IDs; fetched bytes are the "
            "underlying KV blocks they expand to",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("early_capture", type=Path)
    parser.add_argument("late_capture", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = overlap(args.early_capture, args.late_capture)
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
