"""Aggregate a torch-profiler chrome trace into an index-stage time share.

Reads the JSON trace the pinned runtime writes via ``/start_profile``, sums
``cat=kernel`` (and optionally memcpy) event durations, clusters them by name,
and reports the QSA index stage (scoring, fast top-k, index prep) as a share
of total GPU kernel time in the profiled window.

Cluster patterns are configurable because TileLang JIT kernel names vary by
build; ALWAYS read the emitted top-kernels table and confirm the clusters
before trusting the headline share. A cluster that swallows an unrelated
kernel invalidates the number.
"""

import argparse
import gzip
import json
import re
from collections import defaultdict

CLUSTERS = {
    # TileLang/standalone scorer plus any kernel whose name says qsa.
    "index_score": r"tilelang|qsa.*score|score.*qsa|fused.*qsa|weighted_mqa",
    # Selection: qsa_fast_topk and torch/flashinfer top-k kernels.
    "index_topk": r"qsa_fast_topk|topk|top_k|radix|bitonic",
    # JIT index prep (RoPE/norm/compressed store).
    "index_prep": r"qsa_index_(q|k)_",
    "gemm": r"gemm|cutlass|nvjet|marlin|sm100|sm90|matmul|dense|nvjet",
    "memcpy": r"Memcpy|Memset",
    "attention": r"flash|attn|fwd_kernel|mha|mla|gdn|recurren|conv1d|causal",
    "norm_rope": r"rms|RMS|rope|RoPE|layernorm|LayerNorm",
    "sampling": r"sample|softmax|argmax|multinomial|penal",
}


def classify(name):
    for cluster, pattern in CLUSTERS.items():
        if re.search(pattern, name, re.IGNORECASE):
            return cluster
    return "other"


def parse_trace(path, include_memcpy=False, assume_scorer=None):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        trace = json.load(handle)
    events = trace.get("traceEvents", [])
    totals = defaultdict(float)
    kernels = defaultdict(float)
    counts = defaultdict(int)
    kernel_events = 0
    for event in events:
        if event.get("cat") != "kernel":
            continue
        name = event.get("name", "")
        dur_us = event.get("dur", 0)
        if not name or dur_us <= 0:
            continue
        cluster = "index_score" if assume_scorer and name == assume_scorer else classify(name)
        if cluster == "memcpy" and not include_memcpy:
            continue
        totals[cluster] += dur_us
        kernels[name] += dur_us
        counts[cluster] += 1
        kernel_events += 1
    if not kernel_events:
        raise ValueError(f"no kernel events in {path}; wrong file or empty window")
    total = sum(totals.values())
    index_total = sum(totals.get(c, 0) for c in ("index_score", "index_topk", "index_prep"))
    return {
        "trace": str(path),
        "assumed_scorer_kernel": assume_scorer,
        "total_kernel_ms": total / 1e3,
        "kernel_events": kernel_events,
        "index_stage_ms": index_total / 1e3,
        "index_stage_share": round(index_total / total, 4),
        "cluster_ms": {k: round(v / 1e3, 3) for k, v in sorted(totals.items(), key=lambda kv: -kv[1])},
        "cluster_counts": dict(counts),
        "top_kernels_ms": [
            {"name": name[:120], "cluster": classify(name), "ms": round(us / 1e3, 3)}
            for name, us in sorted(kernels.items(), key=lambda kv: -kv[1])[:30]
        ],
    }


def selftest():
    def kernel(name, dur, ts):
        return {"cat": "kernel", "name": name, "dur": dur, "ts": ts, "pid": 1, "tid": 1}
    trace = {"traceEvents": [
        kernel("tilelang_qsa_score_kernel", 100, 0),
        kernel("qsa_fast_topk", 40, 100),
        kernel("qsa_index_q_norm_rope_store", 10, 200),
        kernel("nvjet_tst_128x256_64x4_bf16", 400, 300),
        kernel("flash_fwd_splitkv_kernel", 250, 800),
        kernel("Memcpy DtoH", 5, 1100),
        {"cat": "cpu_op", "name": "aten::mm", "dur": 999, "ts": 0},
    ]}
    import tempfile, os
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(trace, handle)
        path = handle.name
    try:
        report = parse_trace(path)
        assert report["index_stage_share"] == round(150 / 800, 4), report
        assert report["cluster_ms"].get("memcpy", 0.0) == 0.0
        report_with_memcpy = parse_trace(path, include_memcpy=True)
        assert report_with_memcpy["cluster_ms"]["memcpy"] == 0.005
        assert report_with_memcpy["index_stage_share"] == round(150 / 805, 4)
        print("selftest ok:", json.dumps(report["cluster_ms"]))
    finally:
        os.unlink(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="*", type=str)
    parser.add_argument("--include-memcpy", action="store_true")
    parser.add_argument("--assume-scorer", type=str, default=None,
                        help="reclassify this exact kernel name as index_score "
                             "(e.g. TileLang's default 'kernel_kernel' when its "
                             "launch count matches the QSA layer count per step)")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return 0
    if not args.traces:
        parser.error("provide trace paths or --selftest")
    for path in args.traces:
        print(json.dumps(parse_trace(path, args.include_memcpy, args.assume_scorer), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
