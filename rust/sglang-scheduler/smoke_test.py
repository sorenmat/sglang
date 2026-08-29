"""Smoke test for the _scheduler PyO3 extension (run after cargo build)."""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.path.join(HERE, "..", "target", "debug", "libsglang_scheduler.so")
SO = os.path.abspath(SO)


def load():
    spec = importlib.util.spec_from_file_location("_scheduler", SO)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    s = load()
    print("module:", s.__name__)
    assert s.CHUNKED_IDX == 2**32 - 1
    assert s.ROOT == 0

    # ---- RadixTree facade ----
    t = s.RadixTree(1, False, "lru")
    idx, node = t.match_prefix([1, 2, 3])
    assert idx == [] and node == 0, (idx, node)
    p0, n0 = t.insert([1, 2, 3, 4], [10, 11, 12, 13], 0, False)
    assert p0 == 0, p0
    idx, node = t.match_prefix([1, 2, 9])
    assert idx == [10, 11], idx
    mlen, mnode = t.match_prefix_meta([1, 2, 9])
    assert (mlen, mnode) == (2, node), (mlen, mnode)
    assert t.match_prefix_meta([7, 7, 7]) == (0, 0)
    p1, n1 = t.insert([1, 2, 9, 8], [20, 21, 22, 23], 0, False)
    assert p1 == 2, p1  # prefix 1,2 already in tree
    assert t.total_size() == 6, t.total_size()
    assert t.protected_size() == 0
    # lock deltas are evictable-size deltas: -4 = 4 tokens protected.
    assert t.inc_lock_ref(n0) == -4
    assert t.protected_size() == 4, t.protected_size()
    assert t.dec_lock_ref(n0) == 4
    # runs are whole nodes (a request can be overfulfilled, like the
    # Python cache); one 2-token run is evicted here.
    runs, n = t.evict(2)
    assert n == 2 and sum(len(r) for r in runs) == 2, (runs, n)
    assert runs in [[[12, 13]], [[22, 23]]], runs
    # [1,2] + the surviving 2-token branch remain.
    assert t.evictable_size() == 4
    assert t.total_size() == 4
    print("RadixTree: ok")

    # ---- shadow planner ----
    cfg = {
        "policy": "fcfs",
        "page_size": 1,
        "max_prefill_tokens": 1024,
        "chunked_prefill_size": None,
        "mixed_chunk": False,
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
        "ntr_min_factor": 0.1,
        "ntr_decay_steps": 600,
        "retract_decode_steps": 20,
    }
    def wreq(i):
        return {"pool_idx": i, "origin_len": 50, "out_len": 0, "committed_len": 50,
                "prefix_len": 0, "last_node": 0, "priority": 0,
                "arrival_seq": i, "max_new_tokens": 64, "routing_key": 0,
                "ignore_eos": False, "finished": False, "retracted_stain": False,
                "host_hit_length": 0}
    w = [wreq(i) for i in range(4)]
    env = {
        "allocator_avail_tokens": 4096,
        "tree_evictable_tokens": 0,
        "num_allocatable_reqs": 256,
        "batch_is_full": False,
        "mixed_chunk_allowed": False,
    }
    mode, bif, prefill, decode = s.plan_next_batch(
        cfg, 0.7, w, [], None, [0] * 4, [False] * 4, env, 0)
    assert mode == s.MODE_PREFILL, (mode, prefill, decode)
    assert prefill is not None
    admitted, chunked, mixed, ext, pages = prefill
    assert len(admitted) == 4 and ext == 200, (admitted, ext)
    for a in admitted:
        wi, plen, es, ee = a
        assert (plen, es, ee) == (0, 0, 50), a

    # decode path: one running req finishes -> filtered out, batch empties
    # (no batch to run -> MODE_NONE, but finished_removed still reported).
    running = [{"pool_idx": 7, "origin_len": 10, "out_len": 3, "committed_len": 13,
                "prefix_len": 0, "last_node": 0, "priority": 0, "arrival_seq": 0,
                "max_new_tokens": 64, "routing_key": 0, "ignore_eos": False,
                "finished": True, "retracted_stain": False, "host_hit_length": 0}]
    mode, bif, prefill, decode = s.plan_next_batch(
        cfg, 0.7, [], running, None, [], [], env, 1)
    assert mode == s.MODE_NONE, (mode, decode)
    d_decode, d_fin, d_retract, d_abort, d_evict, d_pages, d_ntr = decode
    assert d_decode == [] and d_fin == [0], decode
    print("shadow planner: ok")

    # ---- NTR helpers ----
    assert abs(s.ntr_next_after_decay(0.7, 0.7, 1.0, 0.1, 600) - (0.7 - 0.63 / 600)) < 1e-18
    assert s.ntr_estimate_after_retract([1, 3], [100, 200], 20) == 44.0 / 301.0
    assert s.ntr_estimate_after_retract([1, 3], [10, 20], 20) == 1.0
    print("NTR helpers: ok")

    # ---- SchedulerCore: prefill -> stash -> decode -> finish ----
    core = s.SchedulerCore(cfg, "lru")
    def ireq(rid, pool, tokens, seq):
        return {"rid": rid, "pool_idx": pool, "origin": tokens,
                "max_new_tokens": 64, "priority": 0, "arrival_seq": seq,
                "routing_key": 0, "ignore_eos": False}
    ci = core.ingest([
        ireq(1, 100, list(range(1, 31)), 0),
        ireq(2, 101, list(range(31, 61)), 1),
    ])
    assert ci == [0, 1], ci
    assert core.waiting() == [0, 1]

    plan, events = core.plan(env)
    mode, bif, prefill, decode = plan
    assert mode == s.MODE_PREFILL and not events, (mode, events)
    admitted, chunked, mixed, ext, pages = prefill
    assert len(admitted) == 2 and ext == 60, prefill

    # KV rows: each req got 32 slots (30 prompt + ... use 32).
    rows = [
        {"core_idx": 0, "row": list(range(1000, 1032))},
        {"core_idx": 1, "row": list(range(2000, 2032))},
    ]
    events = core.apply_result(
        [{"accepted": [10000], "finished": False, "finish_reason": 0},
         {"accepted": [10001], "finished": False, "finish_reason": 0}],
        rows)
    stash = [e for e in events if e[0] == "stash_row_write"]
    assert len(stash) == 2, events
    for e in stash:
        _, pool, start, new_idx = e
        assert start == 0, e
        assert len(new_idx) == 30, e  # page-1: full 30-token prompt stashed
    assert core.tree_stats()[0] == 60, core.tree_stats()
    # prefilled reqs wait in pending_merge: last_batch is consumed by the
    # result apply, and they join running at the next plan.
    assert core.running() == [] and core.last_batch() == [], (
        core.running(), core.last_batch())
    assert core.req_pool_idx(0) == 100 and core.req_rid(1) == 2
    assert core.req_out_len(0) == 1

    # decode step 1: nothing special
    plan, events = core.plan(env)
    mode, bif, prefill, decode = plan
    assert mode == s.MODE_DECODE and not prefill, mode
    d_decode, d_fin, d_retract, d_abort, d_evict, d_pages, d_ntr = decode
    assert d_decode == [0, 1] and d_pages == 2, decode
    ntr0 = core.new_token_ratio()
    events = core.apply_result(
        [{"accepted": [10002], "finished": False, "finish_reason": 0},
         {"accepted": [10003], "finished": True, "finish_reason": 1}],
        rows)
    fin = [e for e in events if e[0] == "finished"]
    assert fin and fin[0][1] == 1, events  # req 1 finished
    assert core.req_retracted_stain(1) is False
    assert core.new_token_ratio() <= ntr0

    # next decode: finished req filtered, only 0 decodes
    plan, events = core.plan(env)
    mode, bif, prefill, decode = plan
    d_decode, d_fin, *_ = decode
    assert d_decode == [0] and d_fin == [1], (d_decode, d_fin)
    # user-initiated abort of a running req: KV free + abort notice.
    events = core.drop(0)
    assert events[0][0] == "free_segments" and events[0][2] == [(0, 31)], events
    assert events[1][:2] == ("finished", 0) and events[1][3] == 2, events
    assert core.running() == []
    print("SchedulerCore.drop: ok")

    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
