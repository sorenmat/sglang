#!/usr/bin/env python3
"""Python-side M5–M7 baselines for the unmodified scheduler (plan.md §4.1).

Baselines the Rust planner/core gates (plan.md §5/§7: "planner total <= 50 µs
at waiting 128 / running 256") are compared against. The Rust-side
counterparts live in ``bench_rust_scheduler.py`` (M5–M7, same shapes).

  M5  waiting-queue priority pass, waiting 16/64/128/256:
      - fcfs: the per-req prefix match loop that ``get_new_batch_prefill``
        runs before the adder (a plain ``RadixCache`` reports
        ``supports_fast_match_prefix() == False``, so ``calc_priority``
        itself is a no-op for fcfs — the matching is the real cost).
      - lpm:  ``SchedulePolicy.calc_priority`` (prefix re-scoring +
        in-batch dedup tree + sort; degrades to fcfs above 128 reqs,
        which is what is measured, matching production behavior).
  M6  ``PrefillAdder.add_one_req`` admission loop over 16/64/256 candidates,
      chunked off (rem_chunk_tokens=None) and on (4096): fresh adder per
      timed iteration (mirrors get_new_batch_prefill), real simulated
      RadixCache, synthetic allocator.
  M7  ``ScheduleBatch.filter_batch`` over a running batch of 64/256 with a
      ~25% finished mix (the decode-path filter the Rust planner's decode
      branch replaces). ``retract_decode``'s memory-pressure loop is bound
      to the live allocator pools and is baseline-recorded via the §4.2
      trace replay / §4.3 e2e A/B instead.

Needs CPU-only torch — run this on the target machine:

  python3 benchmark/scheduler/bench_py_planner.py --record py_planner
"""

from __future__ import annotations

import array
import os
import sys
import types

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


GROUP_PREFIXES = 16
PREFIX_LEN = 512
TAIL_LEN = 512
MAX_NEW = 256


def make_reqs(n: int, seed: int = 7) -> list:
    """``n`` fake waiting reqs: each shares one of GROUP_PREFIXES 512-token
    prefixes plus a unique 512-token tail (1024 fill tokens)."""
    prefix_ids = [LCG(300 + g).run(PREFIX_LEN, salt=g * 10_000) for g in range(GROUP_PREFIXES)]
    reqs = []
    for i in range(n):
        g = i % GROUP_PREFIXES
        ids = prefix_ids[g] + LCG(5000 + i).run(TAIL_LEN, salt=i)
        sp = types.SimpleNamespace(ignore_eos=False, max_new_tokens=MAX_NEW)
        ts = types.SimpleNamespace(wait_queue_entry_time=float(i))
        r = types.SimpleNamespace(
            rid=f"r{i}",
            origin_input_ids=ids,
            output_ids=[],
            full_untruncated_fill_ids=ids,
            extra_key=None,
            cache_salt=None,
            prefix_indices=None,
            last_node=None,
            last_host_node=None,
            best_match_node=None,
            host_hit_length=0,
            swa_host_hit_length=0,
            mamba_host_hit_length=0,
            storage_hit_length=0,
            num_matched_prefix_tokens=0,
            priority=0,
            time_stats=ts,
            sampling_params=sp,
            retracted_stain=False,
            mamba_pool_idx=None,
            skip_lock_node_ids={},
            beam_group=None,
            to_finish=None,
            return_logprob=False,
            grammar=None,
        )
        r._compute_max_prefix_len = lambda n_tokens, _ids=ids: len(_ids)
        r.needs_host_load_back = lambda: False
        r.finished = lambda: False

        def set_extend_range(s, e, _r=r):
            _r.extend_range = types.SimpleNamespace(start=s, end=e)

        r.set_extend_range = set_extend_range
        reqs.append(r)
    return reqs


def main():
    args = bench_common.parse_common_args()
    it, wu = args.iters, args.warmup

    try:
        import torch

        from sglang.srt.managers.schedule_batch import (
            CaptureHiddenMode,
            ScheduleBatch,
        )
        from sglang.srt.managers.schedule_policy import (
            PrefillAdder,
            SchedulePolicy,
            match_prefix_for_req,
        )
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams
        from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
    except Exception as e:  # torch or sglang unavailable
        sys.exit(
            f"this bench needs the real scheduler classes (torch, CPU-only is "
            f"fine); import failed: {e!r}\nRun it on the target machine."
        )

    results = {}

    # Shared live tree: 16 group prefixes; each waiting req matches 512.
    tree = RadixCache.create_simulated(page_size=1)
    for g in range(GROUP_PREFIXES):
        ids = LCG(300 + g).run(PREFIX_LEN, salt=g * 10_000)
        tree.insert(InsertParams(key=RadixKey(token_ids=array("q", ids))))

    # pre-match the reqs once (like get_new_batch_prefill does per req)
    for n_wait in (16, 64, 128, 256):
        reqs = make_reqs(n_wait)
        for r in reqs:
            match_prefix_for_req(tree, r)

        # M5 fcfs: the match loop itself is the fcfs cost (calc_priority is
        # a no-op for fcfs + plain RadixCache).
        def fcfs_match(rs=reqs):
            for r in rs:
                match_prefix_for_req(tree, r)

        results[f"M5_fcfs_wait{n_wait}"] = bench_common.time_it_us(fcfs_match, it, wu)

        # M5 lpm: full calc_priority (re-scoring + in-batch tree + sort).
        policy = SchedulePolicy(
            "lpm", tree, False, False, False
        )

        def lpm_calc(p=policy, rs=reqs):
            p.calc_priority(list(rs))

        results[f"M5_lpm_wait{n_wait}"] = bench_common.time_it_us(lpm_calc, it, wu)

    # M6: PrefillAdder.add_one_req admission loop
    class FakeAlloc:
        def __init__(self, avail):
            self.avail = avail

        def available_size(self):
            return self.avail

    alloc = FakeAlloc(10_000_000)
    for n_wait in (16, 64, 256):
        for chunked in (False, True):
            reqs = make_reqs(n_wait, seed=900 + n_wait)
            for r in reqs:
                match_prefix_for_req(tree, r)

            def admit(rs=reqs, chunked=chunked):
                adder = PrefillAdder(
                    page_size=1,
                    tree_cache=tree,
                    token_to_kv_pool_allocator=alloc,
                    running_batch=None,
                    new_token_ratio=0.7,
                    rem_input_tokens=16_384,
                    # 256 < the 512 extend tokens, so the chunked variant
                    # exercises the truncation branch (like a small
                    # chunked_prefill_size against long prompts).
                    rem_chunk_tokens=256 if chunked else None,
                )
                admitted = []
                for r in rs:
                    res = adder.add_one_req(r, has_chunked_req=False, truncation_align_size=None)
                    if res.name == "CONTINUE":
                        admitted.append(r)
                # undo the permanent admission locks so the next iteration
                # starts from identical tree state
                for r in admitted:
                    tree.dec_lock_ref(r.last_node)
                return len(admitted)

            results[f"M6_admit_wait{n_wait}_{'chunked' if chunked else 'plain'}"] = (
                bench_common.time_it_us(admit, it, wu)
            )

    # M7: ScheduleBatch.filter_batch over running 64/256, ~25% finished
    for n_run in (64, 256):
        def make_batch():
            reqs = []
            for i in range(n_run):
                fin = i % 4 == 0

                def finished(_f=fin):
                    return _f

                reqs.append(
                    types.SimpleNamespace(
                        finished=finished,
                        return_logprob=False,
                        grammar=None,
                        return_hidden_states_mode=CaptureHiddenMode.NULL,
                        beam_group=None,
                    )
                )
            b = SB.__new__(SB)
            b.reqs = reqs
            b.beam_tail = None
            b.return_hidden_states = False
            b.model_config = types.SimpleNamespace(is_encoder_decoder=False)
            b.device = "cpu"
            b.req_pool_indices = torch.arange(n_run, dtype=torch.int64)
            b.req_pool_indices_cpu = list(range(n_run))
            b.seq_lens = torch.full((n_run,), 64, dtype=torch.int32)
            b.orig_seq_lens = torch.full((n_run,), 64, dtype=torch.int32)
            b.input_ids = torch.zeros(n_run, dtype=torch.int64)
            b.seq_lens_cpu = None
            b.multimodal_inputs = None
            b.sampling_info = types.SimpleNamespace(filter_batch=lambda *a, **k: None)
            b.spec_info = None
            return b

        def filt():
            make_batch().filter_batch()

        results[f"M7_filter_batch_run{n_run}"] = bench_common.time_it_us(filt, it, wu)

    bench_common.emit(args, results)
    bench_common.record("py_planner", results, args)
    sys.exit(bench_common.compare("py_planner", results, args))


if __name__ == "__main__":
    main()
