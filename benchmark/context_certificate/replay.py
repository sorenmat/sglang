"""Offline QSA capture replay. NumPy only; never imports a serving runtime.

The fixture command creates SYNTHETIC data, not model activations. See
CPU_WORKFLOW.md for the NPZ contract and limits of the float64 certificate.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from probe import certified_select, scores, topk


SEMANTICS = "sum_relu_dot_div_sqrt_dim"


def bf16_values(values):
    """Synthetic conversion by mantissa truncation, not a runtime BF16 cast."""
    words = np.asarray(values, dtype=np.float32).view(np.uint32)
    return (words & np.uint32(0xFFFF0000)).view(np.float32)


def load_capture(path):
    with np.load(path, allow_pickle=False) as archive:
        required = {"metadata", "keys", "queries", "valid_counts", "query_positions",
                    "native_selected_ids"}
        if not required <= set(archive.files):
            raise ValueError(f"capture missing {sorted(required - set(archive.files))}")
        data = {name: archive[name] for name in required}
        if "native_scores" in archive:
            data["native_scores"] = archive["native_scores"]
    try:
        meta = json.loads(data.pop("metadata").item())
    except (ValueError, TypeError) as error:
        raise ValueError("metadata must be a scalar JSON string") from error
    if not isinstance(meta, dict) or meta.get("schema_version") != 1:
        raise ValueError("unsupported capture schema")
    if meta.get("source_kind") not in ("synthetic", "model_capture"):
        raise ValueError("source_kind must distinguish synthetic and model_capture")
    if meta.get("forward_mode") not in ("decode", "target_verify", "synthetic"):
        raise ValueError("unsupported or missing forward_mode")
    if meta.get("compress_ratio") != 4:
        raise ValueError("v1 requires compression ratio four")
    if meta.get("score_semantics") != SEMANTICS:
        raise ValueError("unsupported score semantics; do not silently rescale captures")
    for name in ("model_id", "model_revision", "runtime_revision", "request_id",
                 "verification_id"):
        if not isinstance(meta.get(name), str) or not meta[name].strip():
            raise ValueError(f"metadata requires nonempty {name}")
    if type(meta.get("layer_id")) is not int or meta["layer_id"] < 0:
        raise ValueError("metadata requires nonnegative integer layer_id")
    if meta.get("key_dtype") != "bfloat16" or meta.get("query_dtype") != "bfloat16":
        raise ValueError("v1 requires the actual BF16 key and query scorer inputs")
    if meta.get("shared_key_bank_verified") is not True:
        raise ValueError("capture must verify identical visible key prefixes across positions")
    if meta.get("keys_stage") != "finalized_compressed_index_keys":
        raise ValueError("keys must be finalized compressed index keys")
    if meta.get("queries_stage") != "native_index_score_input":
        raise ValueError("queries must be the actual input to the index scorer")
    keys, queries = data["keys"], data["queries"]
    if (keys.ndim != 2 or keys.shape[1] != 128 or len(keys) == 0
            or queries.ndim != 3 or queries.shape[1:] != (4, 128)
            or len(queries) == 0):
        raise ValueError("expected keys[N,128] and queries[S,4,128]")
    for name, declared in (("keys", "key_dtype"), ("queries", "query_dtype")):
        values = data[name]
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError(f"{name} must contain finite float32 exports")
        if meta[declared] == "bfloat16" and np.any(values.view(np.uint32) & 0xFFFF):
            raise ValueError(f"{name} contains values not exactly representable in BF16")
    count = meta.get("top_k")
    if type(count) is not int or not 1 <= count <= len(keys):
        raise ValueError("top_k must be an integer in [1,N]")
    valid = data["valid_counts"]
    if (valid.shape != (len(queries),) or not np.issubdtype(valid.dtype, np.integer)
            or np.any(valid < count) or np.any(valid > len(keys))
            or np.any(valid[1:] < valid[:-1]) or valid[-1] != len(keys)):
        raise ValueError("valid_counts must be nondecreasing counts in [top_k,N], ending at N")
    positions = data["query_positions"]
    if (positions.dtype != np.int64 or positions.shape != valid.shape
            or np.any(positions < 0) or np.any(positions >= np.iinfo(np.int64).max)
            or np.any(positions[1:] != positions[:-1] + 1)
            or np.any((positions + 1) // 4 != valid)):
        raise ValueError("query_positions must be consecutive int64 logical token positions with valid_counts=(position+1)//4")
    if meta["forward_mode"] == "decode" and len(queries) != 1:
        raise ValueError("one decode capture contains one position per request")
    if meta["source_kind"] == "model_capture":
        if count != 512 or meta["forward_mode"] == "synthetic":
            raise ValueError("model captures require native top-512 and a real forward mode")
        if "native_scores" not in data or data["native_scores"].dtype != np.float32:
            raise ValueError("initial model captures require native FP32 logits")
    selected = data["native_selected_ids"]
    if (selected.shape != (len(queries), count)
            or not np.issubdtype(selected.dtype, np.integer)
            or np.any(selected < 0) or np.any(selected >= valid[:, None])
            or any(len(np.unique(row)) != count for row in selected)):
        raise ValueError("native selections must be distinct logical IDs valid for each position")
    if "native_scores" in data:
        native = data["native_scores"]
        mask = np.arange(len(keys))[None, :] < valid[:, None]
        if (native.shape != mask.shape or not np.issubdtype(native.dtype, np.floating)
                or not np.isfinite(native[mask]).all()):
            raise ValueError("native_scores must be floating [S,N], finite at valid keys")
        # A native tie policy may differ, but selected scores cannot be below
        # any unselected score. Reject a capture with mixed calls or ID spaces.
        for row, n in enumerate(valid):
            threshold = np.partition(native[row, :n], -count)[-count]
            if np.any(native[row, selected[row]] < threshold):
                raise ValueError("native selections disagree with captured native scores")
    return meta, data


def replay(path, bits=(4, 6, 8), positions=None):
    meta, data = load_capture(path)
    if not bits or any(type(bit) is not int or bit not in (4, 6, 8) for bit in bits):
        raise ValueError("bits must be a nonempty sequence of supported precisions (4,6,8)")
    sizes = [len(data["queries"])] if positions is None else positions
    if not sizes:
        raise ValueError("positions must not be empty")
    if any(type(size) is not int or not 1 <= size <= len(data["queries"]) for size in sizes):
        raise ValueError("positions must be positive prefixes of the captured verification")
    cases = []
    for size in sizes:
        valid = data["valid_counts"][:size]
        keys = data["keys"][:valid[-1]]
        queries = data["queries"][:size]
        native_ids = data["native_selected_ids"][:size]
        for precision in bits:
            for mode in ("basic", "joint"):
                result = certified_select(keys, queries, meta["top_k"], precision, mode,
                                          valid_counts=valid, return_details=True)
                details = result.pop("details")
                selected = details["selected_ids"]
                row_match = np.all(np.sort(selected, axis=1) == np.sort(native_ids, axis=1), axis=1)
                misses = (~details["candidates"][np.arange(size)[:, None], native_ids]).sum(axis=1)
                result.update({
                    "native_selected_set_match": bool(row_match.all()),
                    "native_selected_order_match": bool(np.array_equal(selected, native_ids)),
                    "native_mismatch_positions": np.flatnonzero(~row_match).tolist(),
                    "native_winners_excluded_by_screen_per_position": misses.tolist(),
                    "native_score_interval_check": "not_captured",
                })
                if "native_scores" in data:
                    native = data["native_scores"][:size, :len(keys)]
                    mask = np.arange(len(keys))[None, :] < valid[:, None]
                    escaped = mask & ((native < details["lower"]) | (native > details["upper"]))
                    result["native_score_interval_check"] = "fail" if escaped.any() else "pass_on_this_capture"
                    result["native_score_interval_violations"] = int(escaped.sum())
                    result["max_abs_native_vs_float64_score_difference"] = float(
                        np.max(np.abs(native[mask] - details["reference_scores"][mask])))
                cases.append(result)
    native_ids = data["native_selected_ids"]
    adjacent = []
    for left, right in zip(native_ids[:-1], native_ids[1:]):
        adjacent.append(len(np.intersect1d(left, right)) / len(np.union1d(left, right)))
    failed = any(not case["native_selected_set_match"]
                 or case["native_score_interval_check"] == "fail" for case in cases)
    return {
        "status": "capture_mismatch" if failed else "cpu_replay_checks_passed",
        "source": str(Path(path).resolve()),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "metadata": meta,
        "native_selected_microblock_union": int(len(np.unique(native_ids))),
        "native_adjacent_selection_jaccard": adjacent,
        "limits": [
            "Synthetic fixtures establish no model behavior or quality.",
            "Float64 bounds have a numerical guard, not a proved GPU rounding envelope.",
            "Native set agreement is distinct from native order/attention accumulation parity.",
            "CPU packed storage is implemented; byte totals still model GPU reads, not measured GPU traffic or timing.",
            "Comparison baseline is ONE shared BF16 index scan, not one scan per position.",
            "Only finalized index blocks are modeled; partial-block KV, state and draft costs are excluded.",
        ],
        "cases": cases,
    }


def write_fixture(path, *, key_count=4096, positions=12, top_k=512, seed=20260925):
    if min(key_count, positions, top_k) < 1 or (4 * key_count - positions + 1) // 4 < top_k:
        raise ValueError("fixture needs at least top_k visible blocks at its first position")
    rng = np.random.default_rng(seed)
    keys = bf16_values(rng.normal(size=(key_count, 128)))
    queries = bf16_values(rng.normal(size=(1, 4, 128)) +
                          0.03 * rng.normal(size=(positions, 4, 128)))
    query_positions = np.arange(4 * key_count - positions, 4 * key_count, dtype=np.int64)
    valid = (query_positions + 1) // 4
    native_scores = scores(keys.astype(np.float64), queries.astype(np.float64))
    for row, n in enumerate(valid):
        native_scores[row, n:] = -np.inf
    selected = np.stack([topk(row, top_k) for row in native_scores])
    meta = {
        "schema_version": 1, "source_kind": "synthetic", "seed": seed,
        "forward_mode": "synthetic", "compress_ratio": 4,
        "model_id": "none-synthetic", "model_revision": "none-synthetic",
        "runtime_revision": "numpy-float64-fixture", "layer_id": 0,
        "request_id": "synthetic-0", "verification_id": "synthetic-0",
        "score_semantics": SEMANTICS, "top_k": top_k,
        "key_dtype": "bfloat16", "query_dtype": "bfloat16",
        "shared_key_bank_verified": True,
        "keys_stage": "finalized_compressed_index_keys",
        "queries_stage": "native_index_score_input",
    }
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("capture path must end in .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, metadata=json.dumps(meta), keys=keys, queries=queries,
                        query_positions=query_positions, valid_counts=valid,
                        native_selected_ids=selected, native_scores=native_scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("fixture", help="write explicitly synthetic BF16-valued test data")
    fixture.add_argument("output", type=Path)
    fixture.add_argument("--keys", type=int, default=4096)
    fixture.add_argument("--positions", type=int, default=12)
    fixture.add_argument("--seed", type=int, default=20260925)
    run = sub.add_parser("run", help="validate and replay a capture without a GPU")
    run.add_argument("capture", type=Path)
    run.add_argument("--bits", nargs="+", type=int, choices=(4, 6, 8), default=[4, 6, 8])
    run.add_argument("--positions", nargs="+", type=int)
    run.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "fixture":
            write_fixture(args.output, key_count=args.keys, positions=args.positions, seed=args.seed)
            print(json.dumps({"source_kind": "synthetic", "path": str(args.output.resolve())}))
            return 0
        report = replay(args.capture, args.bits, args.positions)
        payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload)
        else:
            print(payload, end="")
        return 2 if report["status"] == "capture_mismatch" else 0
    except (ValueError, OSError, AssertionError) as error:
        parser.exit(2, f"replay: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
