#!/usr/bin/env python3
"""Python-side M1–M4 baselines for the unmodified ``RadixCache`` (plan.md §4.1).

These are the *baseline* numbers the Phase-1 gate ("Rust match_prefix /
insert >= 10x the Python baseline p50 at a 100k-token tree") compares
against; the Rust-side counterpart is measured through the ``_scheduler``
extension in ``bench_rust_scheduler.py`` (M1–M4, identical shapes).

The trees are the real ``sglang.srt.mem_cache.radix_cache.RadixCache`` in
simulated mode (no memory pools). Needs CPU-only torch, so run this on the
target machine:

  python3 benchmark/scheduler/bench_py_radix.py --record py_m1_m4

Shapes (kept identical to the Rust bench for 1:1 comparison):
  M1 match   tree 1k/10k/100k tokens, key = one full 8-leaf branch
  M2 insert  fresh 128-token keys into a 10k/100k-token tree
  M3 evict   evict 1k/10k tokens from a freshly built 100k-token tree
             (fresh tree per sample, like the Rust bench, so the tree
             can't drain mid-measurement)
  M4 lock    inc/dec_lock_ref walk, chain depth 1/32/256
"""

from __future__ import annotations

import array
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_common  # noqa: E402


class LCG:
    """Deterministic token-id source (mirrors bench_rust_scheduler.py)."""

    def __init__(self, seed: int = 42):
        self.s = seed

    def next(self) -> int:
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return self.s % 100_000

    def run(self, n: int, salt: int = 0) -> list:
        return [self.next() + salt for _ in range(n)]


def main():
    args = bench_common.parse_common_args()
    it, wu = args.iters, args.warmup
    # M3 builds trees inside the timed loop; keep its sample count bounded
    # independently of --iters.
    m3_samples = min(it, 10)

    try:
        import torch  # noqa: F401

        from sglang.srt.mem_cache.base_prefix_cache import (
            EvictParams,
            InsertParams,
            MatchPrefixParams,
        )
        from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
    except Exception as e:  # torch or sglang unavailable
        sys.exit(
            f"this bench needs the real RadixCache (torch, CPU-only is fine); "
            f"import failed: {e!r}\nRun it on the target machine."
        )

    def build_tree(cache, n_tokens: int, leaves: int = 8) -> list:
        """Insert ``leaves`` non-overlapping runs totalling ~n_tokens; return
        the runs (one per leaf) so M1 can match a full branch."""
        per = n_tokens // leaves
        runs = []
        for j in range(leaves):
            ids = LCG(1000 + j).run(per, salt=j * 1_000_000)
            cache.insert(InsertParams(key=RadixKey(token_ids=array("q", ids))))
            runs.append(ids)
        return runs

    results = {}

    # M1: match a full branch against a tree of n tokens
    for n in (1_000, 10_000, 100_000):
        cache = RadixCache.create_simulated(page_size=1)
        key = build_tree(cache, n)[0]

        def do_match(c=cache, k=key):
            return len(
                c.match_prefix(
                    MatchPrefixParams(key=RadixKey(token_ids=array("q", k)))
                ).device_indices
            )

        results[f"M1_match_tree{n}_key{len(key)}"] = bench_common.time_it_us(
            do_match, it, wu
        )

    # M2: insert fresh 128-token keys into a growing tree
    for n in (10_000, 100_000):
        cache = RadixCache.create_simulated(page_size=1)
        build_tree(cache, n)
        rnd = LCG(999)
        counter = {"i": 0}

        def do_ins(c=cache):
            counter["i"] += 1
            ids = rnd.run(128, salt=50_000 + counter["i"])
            c.insert(InsertParams(key=RadixKey(token_ids=array("q", ids))))

        results[f"M2_insert_tree{n}"] = bench_common.time_it_us(do_ins, it, wu)

    # M3: evict from a freshly built 100k-token tree (fresh per sample)
    for n_ev in (1_000, 10_000):
        samples = []
        for _ in range(m3_samples):
            cache = RadixCache.create_simulated(page_size=1)
            build_tree(cache, 100_000)
            for _ in range(wu // 5):
                cache.evict(EvictParams(num_tokens=1))  # settle LRU clocks
            t0 = time.perf_counter_ns()
            res = cache.evict(EvictParams(num_tokens=n_ev))
            samples.append(time.perf_counter_ns() - t0)
            assert res.num_tokens_evicted > 0
        samples.sort()
        results[f"M3_evict{n_ev}_of100k"] = {
            "iters": len(samples),
            "mean_us": sum(samples) / len(samples) / 1e3,
            "p50_us": samples[len(samples) // 2] / 1e3,
            "p95_us": samples[int(len(samples) * 0.95)] / 1e3,
            "p99_us": samples[min(len(samples) - 1, int(len(samples) * 0.99))] / 1e3,
        }

    # M4: inc/dec_lock_ref walk down a single-token chain of depth d
    for depth in (1, 32, 256):
        cache = RadixCache.create_simulated(page_size=1)
        node = cache.root_node
        for i in range(depth):
            node = cache.insert(
                InsertParams(key=RadixKey(token_ids=array("q", [i])))
            ).last_device_node
        leaf = node

        def do_lock(c=cache, leaf=leaf):
            c.inc_lock_ref(leaf)
            c.dec_lock_ref(leaf)

        results[f"M4_lock_walk_depth{depth}"] = bench_common.time_it_us(
            do_lock, it, wu
        )

    bench_common.emit(args, results)
    bench_common.record("py_radix", results, args)
    sys.exit(bench_common.compare("py_radix", results, args))


if __name__ == "__main__":
    main()
