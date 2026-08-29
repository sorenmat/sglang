#!/usr/bin/env python3
"""Rust-side scheduler microbenchmarks through the ``_scheduler`` extension
(plan.md §4.1 M1–M7 + an M11-style full loop, measured *through the PyO3
boundary* — i.e. the cost the Python driver actually pays per call).

No torch required: the cdylib is loaded directly. Build it first with
``cargo build`` in ``rust/`` (debug is fine; use a release build for
comparable absolute numbers).

Usage:
  python3 benchmark/scheduler/bench_rust_scheduler.py [--so PATH]
      [--iters N] [--record rust_m1_m7] [--compare rust_m1_m7]
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_common  # noqa: E402

DEFAULT_SO = os.path.join(HERE, "..", "..", "rust", "target", "debug", "libsglang_scheduler.so")


def load_so(path: str):
    spec = importlib.util.spec_from_file_location("_scheduler", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------- shape utils


class LCG:
    """Deterministic token-id source."""

    def __init__(self, seed: int = 12345):
        self.x = seed

    def __call__(self) -> int:
        self.x = (self.x * 6364136223846793005 + 1442695040888963407) % (1 << 64)
        return int(self.x % 100_000)


def token_run(rnd, n: int, salt: int = 0) -> list:
    return [int(rnd() + salt) for _ in range(n)]


def build_tree(mod, n_tokens: int, leaves: int = 0):
    """Fill a fresh RadixTree with ~n_tokens tokens across disjoint leaves.
    Returns (tree, leaf key-runs)."""
    tree = mod.RadixTree(1, False, "lru")
    leaves = leaves or max(1, n_tokens // 128)
    per = max(1, n_tokens // leaves)
    runs = []
    for i in range(leaves):
        key = token_run(LCG(1000 + i), per)
        tree.insert(key, list(range(10_000 * (i + 1) + 1, 10_000 * (i + 1) + 1 + per)), 0, False)
        runs.append(key)
    return tree, runs


CFG = {
    "policy": "fcfs",
    "page_size": 1,
    "max_prefill_tokens": 16384,
    "chunked_prefill_size": None,
    "mixed_chunk": True,
    "priority_scheduling": False,
    "low_priority_values_first": False,
    "clip_max_new_tokens": 4096,
    "in_batch_check_threshold": 32,
    "in_batch_deprioritize_threshold": 32,
    "prefill_max_requests": None,
    "truncation_align_size": None,
    "lpm_queue_degrade_at": 128,
    "random_seed": 0,
    "disable_tree": False,
    "ntr_init_raw": 0.7,
    "schedule_conservativeness": 1.0,
    "ntr_min_factor": 0.14,
    "ntr_decay_steps": 600,
    "retract_decode_steps": 20,
}


def plan_req(i: int, origin: int, out: int = 0, prefix: int = 0, max_new: int = 1024) -> dict:
    return {
        "pool_idx": i,
        "origin_len": origin,
        "out_len": out,
        "committed_len": out,
        "prefix_len": prefix,
        "last_node": 0,
        "priority": 0,
        "arrival_seq": i,
        "max_new_tokens": max_new,
        "routing_key": 0,
        "ignore_eos": False,
        "finished": False,
        "retracted_stain": False,
        "host_hit_length": 0,
    }


def env(avail: int, evictable: int = 0, nreqs: int = 4096, full: bool = False) -> dict:
    return {
        "allocator_avail_tokens": avail,
        "tree_evictable_tokens": evictable,
        "num_allocatable_reqs": nreqs,
        "batch_is_full": full,
        "mixed_chunk_allowed": True,
    }


# ------------------------------------------------------------------- bench


def main():
    import argparse

    p = argparse.ArgumentParser()
    bench_common.add_common_args(p)
    p.add_argument("--so", default=DEFAULT_SO)
    args = p.parse_args()

    if not os.path.exists(args.so):
        sys.exit(f"extension not found at {args.so} (cargo build in rust/ first)")
    mod = load_so(args.so)
    results = {}
    it, wu = args.iters, args.warmup

    # M1: match_prefix through pybind — tree of n tokens, key = one full
    # leaf (n/8 tokens for 8-leaf trees), i.e. match depth grows with size.
    # `match` returns the full KV-index list (what a real cache lookup needs
    # to index the KV pool); `match_meta` is the shadow-check fast path that
    # the dual-write facade uses (length + node handle only, no list build).
    for n in (1_000, 10_000, 100_000):
        tree, runs = build_tree(mod, n, leaves=8)
        key = runs[0]
        results[f"M1_match_tree{n}_key{len(key)}"] = bench_common.time_it_us(
            lambda: tree.match_prefix(key), it, wu
        )
        results[f"M1_match_meta_tree{n}_key{len(key)}"] = bench_common.time_it_us(
            lambda: tree.match_prefix_meta(key), it, wu
        )

    # M2: insert through pybind (fresh keys into a growing tree)
    for n in (10_000, 100_000):
        tree, _ = build_tree(mod, n)
        rnd = LCG(999)
        i = 0

        def ins():
            nonlocal i
            i += 1
            key = token_run(rnd, 128, salt=50_000 + i)
            tree.insert(key, [i] * 128, 0, False)

        results[f"M2_insert_tree{n}"] = bench_common.time_it_us(ins, it, wu)

    # M3: evict through pybind — each sample times ONE evict on a freshly
    # built 100k tree (a single evict(10k) call cannot drain the tree, so
    # the sample measures real eviction work; the build itself is untimed).
    for n_ev in (1_000, 10_000):
        samples = []
        for _ in range(min(it, 10)):
            tree, _ = build_tree(mod, 100_000)
            for _ in range(wu // 5):
                tree.evict(1)  # settle LRU clocks without draining
            t0 = time.perf_counter_ns()
            tree.evict(n_ev)
            samples.append(time.perf_counter_ns() - t0)
        samples.sort()
        n = len(samples)
        results[f"M3_evict{n_ev}_of100k"] = {
            "iters": n,
            "mean_us": sum(samples) / n / 1e3,
            "p50_us": samples[n // 2] / 1e3,
            "p95_us": samples[int(n * 0.95)] / 1e3,
            "p99_us": samples[min(n - 1, int(n * 0.99))] / 1e3,
        }

    # M4: lock-ref walks (depth 1/32/256) — a chain of single-token edges
    for depth in (1, 32, 256):
        tree = mod.RadixTree(1, False, "lru")
        key = []
        node = mod.ROOT
        for i in range(depth):
            key.append(i + 1)
            # full-path insert: one KV index per token of the path so far
            _p, node = tree.insert(key, [1000 + j for j in range(len(key))], 0, False)
        results[f"M4_lock_walk_depth{depth}"] = bench_common.time_it_us(
            lambda: tree.inc_lock_ref(node), it, wu
        )

    # M5: planner — waiting N, fcfs + lpm
    for pol in ("fcfs", "lpm"):
        for nwait in (16, 64, 128, 256):
            cfg = dict(CFG, policy=pol)
            waiting = [plan_req(i, 512 + (i % 7) * 16, prefix=(i * 37) % 256) for i in range(nwait)]
            scores = [int((i * 37) % 256) for i in range(nwait)]
            deprio = [False] * nwait
            ee = env(100_000)
            results[f"M5_plan_{pol}_wait{nwait}"] = bench_common.time_it_us(
                lambda: mod.plan_next_batch(cfg, 0.7, waiting, [], None, scores, deprio, ee, 0),
                it,
                wu,
            )

    # M6: admission loop with chunked continuation (waiting + chunked req)
    for nwait in (16, 64, 256):
        waiting = [plan_req(i, 4096) for i in range(nwait)]
        chunked = plan_req(9999, 4096, out=1024, prefix=1024)
        chunked["committed_len"] = 1024
        scores = [0] * nwait
        ee = env(32_768, 0, 4096)
        results[f"M6_admit_wait{nwait}_chunked"] = bench_common.time_it_us(
            lambda: mod.plan_next_batch(CFG, 0.7, waiting, [], chunked, scores, [False] * nwait, ee, 0),
            it,
            wu,
        )

    # M7: decode planning (running 64/256, with and without memory pressure)
    for nrun in (64, 256):
        for avail, evictable in ((100_000, 0), (4096, 10_000)):
            running = [
                plan_req(i, 2048, out=128 + i, prefix=2048, max_new=4096) for i in range(nrun)
            ]
            for i, r in enumerate(running):
                r["committed_len"] = 2048 + 128 + i
            ee = env(avail, evictable, nreqs=max(8, 4096 - nrun))
            press = "press" if avail < 20_000 else "idle"
            results[f"M7_decode_run{nrun}_{press}"] = bench_common.time_it_us(
                lambda: mod.plan_next_batch(CFG, 0.7, [], running, None, [], [], ee, 0),
                it,
                wu,
            )

    # M11-style: full core loop (plan -> apply_result for last_batch)
    core = mod.SchedulerCore(CFG, "lru")
    origins = {i: token_run(LCG(7 + i), 64) for i in range(32)}
    idx = core.ingest(
        [
            {
                "rid": 1000 + i,
                "pool_idx": i,
                "origin": origins[i],
                "max_new_tokens": 10**9,  # never finishes -> stable decode load
                "priority": 0,
                "arrival_seq": i,
                "routing_key": 0,
                "ignore_eos": False,
            }
            for i in range(32)
        ]
    )
    out_len = {c: 0 for c in idx}

    def core_step():
        core.plan(env(100_000, 0, 4096))
        lb = core.last_batch()
        if not lb:
            return
        rows = [
            {"accepted": [5], "finished": False, "finish_reason": 0,
             "spec": None}
            for _ in lb
        ]
        kv = []
        for c in lb:
            pool = int(c)
            out_len[c] += 1
            row = list(range(10_000 * (pool + 1) + 1, 10_000 * (pool + 1) + 65 + out_len[c]))
            kv.append({"core_idx": c, "row": row})
        core.apply_result(rows, kv)

    results["M11_core_loop_32req"] = bench_common.time_it_us(core_step, it, wu)

    bench_common.emit(args, results)
    bench_common.record("rust", results, args)
    sys.exit(bench_common.compare("rust", results, args))


if __name__ == "__main__":
    main()
