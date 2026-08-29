"""Shared harness for the scheduler microbenchmark suite (plan.md §4.1).

Every bench in this directory reports per-shape p50/p95/p99 wall-clock in
microseconds over many iterations, and can append its results to
``baselines/*.json`` (``--record``) or compare against a committed baseline
(``--compare``). CI nightly re-runs the Python benches against the
committed baselines to catch Python-side drift (plan.md §11.3).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINES = os.path.join(HERE, "baselines")


def time_it_us(fn, n: int, warmup: int = 5) -> dict:
    """Run ``fn`` ``warmup + n`` times; return percentiles in µs."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)
    samples.sort()
    return {
        "iters": n,
        "mean_us": statistics.fmean(samples) / 1e3,
        "p50_us": samples[len(samples) // 2] / 1e3,
        "p95_us": samples[int(len(samples) * 0.95)] / 1e3,
        "p99_us": samples[int(len(samples) * 0.99)] / 1e3,
    }


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--iters", type=int, default=500, help="timed iterations per shape")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument(
        "--record",
        default=None,
        metavar="NAME",
        help="append results to baselines/<NAME>.json",
    )
    p.add_argument(
        "--compare",
        default=None,
        metavar="NAME",
        help="compare p50 against baselines/<NAME>.json (10%% regression -> exit 1)",
    )
    p.add_argument("--json", action="store_true", help="print results as JSON")


def parse_common_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    add_common_args(p)
    return p.parse_args(argv)


def record(name: str, results: dict, args) -> None:
    if not args.record:
        return
    import sys

    path = os.path.join(BASELINES, f"{args.record}.json")
    entry = {
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": os.uname().nodename,
        "python": sys.version.split()[0],
        "results": results,
    }
    os.makedirs(BASELINES, exist_ok=True)
    with open(path, "w") as f:
        json.dump(entry, f, indent=2)
    print(f"recorded -> {path}")


def compare(name: str, results: dict, args, threshold: float = 0.10) -> int:
    if not args.compare:
        return 0
    path = os.path.join(BASELINES, f"{args.compare}.json")
    if not os.path.exists(path):
        print(f"baseline {path} missing; skipping comparison")
        return 0
    with open(path) as f:
        base = json.load(f).get("results", {})
    bad = []
    for key, cur in results.items():
        b = base.get(key)
        if not b or "p50_us" not in b or b["p50_us"] <= 0:
            continue
        ratio = cur["p50_us"] / b["p50_us"]
        marker = "" if ratio <= 1 + threshold else "  REGRESSION"
        print(f"  {key}: {cur['p50_us']:.1f} µs vs {b['p50_us']:.1f} µs ({ratio:.2f}x){marker}")
        if ratio > 1 + threshold:
            bad.append(key)
    return 1 if bad else 0


def emit(args, results: dict, extra: dict | None = None) -> None:
    if args.json:
        print(json.dumps(results, indent=2))
        return
    for key, v in results.items():
        print(
            f"  {key:<40} p50={v['p50_us']:9.1f} µs  "
            f"p95={v['p95_us']:9.1f} µs  p99={v['p99_us']:9.1f} µs"
        )
    if extra:
        for k, v in extra.items():
            print(f"  {k}: {v}")
