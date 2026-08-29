"""M6 differential parity test: Rust spec-v2 accept-run resolution +
per-req spec counters (sglang-scheduler) vs the Python bookkeeping they
mirror (plan.md §9).

Three suites:

1. **Resolve parity** — the same stride-padded ``next_token_ids`` +
   ``accept_lens`` resolved two ways and compared run-for-run: the Python
   reference re-implements the commit contract of
   ``_resolve_spec_v2_tokens`` (slice + grammar-retained substitution +
   retracted/finished gate + the four result fields); the Rust side runs
   ``resolve_spec_runs``. Covers grammar truncation, unsettled rows,
   the MTP block/cap columns, and the Python-slice clamping semantics.

2. **Counter parity** — ``SpecCounters`` driven with a fixed update
   sequence vs the Python reference (the ``Req`` spec counters + the two
   growable histograms from ``schedule_batch.py``); when torch is
   available the reference itself is cross-checked against the real
   ``Req.update_spec_*`` methods, so Rust == pure reference == production
   semantics.

3. **Core path parity** — a spec result row folded through a live
   ``SchedulerCore`` (the ``apply_result`` spec metadata) and compared to
   the Python mirror of the same row sequence.

The extension is required; the suites skip when it cannot be loaded (the
CPU CI job builds it via SGLANG_BUILD_RUST_EXTS=all).
"""

import random
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

try:
    from sglang.srt.rust_extensions import load_rust_extension

    _ext = load_rust_extension("sglang.srt.rust_extensions._scheduler")
except Exception:  # noqa: S110 - parity is best-effort without the extension
    _ext = None


# --------------------------------------------------------------------------
# Python references
# --------------------------------------------------------------------------


def py_resolve(next_token_ids, stride, accept_lens, retracted, finished,
               grammar_retained, block_accept_lens, cap_lens):
    """The commit contract of ``_resolve_spec_v2_tokens``.

    Returns ``(runs, num_correct_drafts, per_req, block_total, cap_total)``
    where ``runs[i]`` is the committed accepted run — empty for unsettled
    rows (retracted / pre-finished reqs commit nothing; Python's raw slice
    there is dead data the overlap-skip path never consumes).
    """
    n = len(accept_lens)
    num_correct_drafts = sum(accept_lens) - n
    per_req = [x - 1 for x in accept_lens]
    block_total = sum(block_accept_lens) if block_accept_lens else 0
    cap_total = sum(cap_lens) if cap_lens else 0
    runs = []
    for i in range(n):
        if retracted[i] or finished[i]:
            runs.append([])
            continue
        raw = next_token_ids[i * stride: i * stride + accept_lens[i]]
        if grammar_retained is not None and grammar_retained[i] is not None:
            runs.append(list(grammar_retained[i]))
        else:
            runs.append(list(raw))
    return runs, num_correct_drafts, per_req, block_total, cap_total


def _bump(hist, idx):
    """`update_spec_*_histogram`: extend with zeros, then += 1."""
    if len(hist) <= idx:
        hist.extend([0] * (idx - len(hist) + 1))
    hist[idx] += 1


def py_counters(updates):
    """The per-req spec counters over one update sequence."""
    state = {
        "spec_verify_ct": 0,
        "spec_num_correct_drafts": 0,
        "spec_num_block_accept_tokens": 0,
        "spec_num_cap_tokens": 0,
        "correct_drafts_histogram": [],
        "cap_lens_histogram": [],
    }
    for correct, block, cap in updates:
        state["spec_verify_ct"] += 1
        state["spec_num_correct_drafts"] += correct
        _bump(state["correct_drafts_histogram"], correct)
        if block is not None:
            state["spec_num_block_accept_tokens"] += block
        if cap is not None:
            state["spec_num_cap_tokens"] += cap
            _bump(state["cap_lens_histogram"], cap)
    return state


def _req_counters(updates):
    """The same counter sequence driven through the real ``Req`` methods
    (needs torch; CI only)."""
    from sglang.srt.managers.schedule_batch import Req

    req = Req.__new__(Req)
    req.spec_verify_ct = 0
    req.spec_num_correct_drafts = 0
    req.spec_num_block_accept_tokens = 0
    req.spec_num_cap_tokens = 0
    req.spec_correct_drafts_histogram = []
    req.spec_cap_lens_histogram = []
    for correct, block, cap in updates:
        req.spec_verify_ct += 1
        req.spec_num_correct_drafts += correct
        req.update_spec_correct_drafts_histogram(correct)
        if block is not None:
            req.spec_num_block_accept_tokens += block
        if cap is not None:
            req.spec_num_cap_tokens += cap
            req.update_spec_cap_lens_histogram(cap)
    return {
        "spec_verify_ct": req.spec_verify_ct,
        "spec_num_correct_drafts": req.spec_num_correct_drafts,
        "spec_num_block_accept_tokens": req.spec_num_block_accept_tokens,
        "spec_num_cap_tokens": req.spec_num_cap_tokens,
        "correct_drafts_histogram": list(req.spec_correct_drafts_histogram),
        "cap_lens_histogram": list(req.spec_cap_lens_histogram),
    }


# --------------------------------------------------------------------------
# Shape generation
# --------------------------------------------------------------------------


def gen_shape(rng, n, stride, clamp_tail=False):
    """One resolve shape: inputs + expected per-row classification."""
    buf = list(range(1_000_000, 1_000_000 + n * stride))
    accept = []
    for i in range(n):
        a = 1 + (i * 7 + stride) % stride
        if clamp_tail and i == n - 1:
            a = stride + 37  # slice clamps like a Python slice
        accept.append(a)
    retracted = [i % 5 == 4 for i in range(n)]
    finished = [i == n - 2 for i in range(n)]
    grammar = None
    if rng.random() < 0.5:
        grammar = []
        for i in range(n):
            if retracted[i] or finished[i]:
                grammar.append(None)
                continue
            raw = buf[i * stride: i * stride + accept[i]]
            k = 1 + rng.randrange(len(raw)) if raw else 0
            grammar.append(list(raw[:k]) if k else None)
    block = [2 if i % 2 == 0 else 0 for i in range(n)] if rng.random() < 0.5 else None
    cap = [3 + i % 4 for i in range(n)] if rng.random() < 0.5 else None
    return buf, stride, accept, retracted, finished, grammar, block, cap


@unittest.skipUnless(_ext is not None, "sglang-scheduler extension unavailable")
class TestRustSpecParity(CustomTestCase):
    def test_resolve_parity(self):
        rng = random.Random(20260829)
        for trial in range(64):
            n = rng.choice([1, 2, 5, 17])
            stride = rng.choice([2, 4, 64])
            clamp = trial % 7 == 0
            (buf, stride_, accept, retracted,
             finished, grammar, block, cap) = gen_shape(rng, n, stride, clamp)
            got = _ext.resolve_spec_runs(
                buf, stride_, accept, retracted, finished, grammar, block, cap
            )
            want = py_resolve(
                buf, stride_, accept, retracted, finished, grammar, block, cap
            )
            self.assertEqual(got, want, f"trial {trial} (n={n}, s={stride_})")

    def test_stride_zero_errors(self):
        with self.assertRaises(ValueError):
            _ext.resolve_spec_runs([1, 2], 0, [2], [False], [False], None, None, None)

    def test_column_length_mismatch_errors(self):
        for bad in (
            dict(retracted=[False]),
            dict(finished=[False]),
            dict(grammar_retained=[None, None]),
            dict(block_accept_lens=[1, 2]),
            dict(cap_lens=[1, 2]),
        ):
            with self.assertRaises(ValueError, msg=str(bad)):
                _ext.resolve_spec_runs(
                    [1, 2], 2, [2],
                    bad.get("retracted", [False, False]),
                    bad.get("finished", [False, False]),
                    bad.get("grammar_retained"),
                    bad.get("block_accept_lens"),
                    bad.get("cap_lens"),
                )

    def test_counters_parity(self):
        rng = random.Random(99)
        updates = [
            (rng.randrange(0, 6),
             rng.choice([None, 0, 1, 2]),
             rng.choice([None, 0, 1, 4, 9]))
            for _ in range(40)
        ]
        c = _ext.SpecCounters()
        for correct, block, cap in updates:
            c.update(correct, block, cap)
        want = py_counters(updates)
        self.assertEqual(c.as_dict(), want)
        # Cross-check the pure reference against the production Req
        # methods where torch is available (CPU CI); the host-gated path
        # (this box, no torch) runs the pure reference only.
        try:
            req_want = _req_counters(updates)
        except Exception:  # noqa: S110 - no torch on this host
            pass
        else:
            self.assertEqual(req_want, want)

    def test_core_path_parity(self):
        """Spec rows through a live SchedulerCore vs the Python mirror."""
        cfg = {
            "policy": "fcfs", "page_size": 1, "max_prefill_tokens": 64,
            "chunked_prefill_size": None, "mixed_chunk": False,
            "priority_scheduling": False, "low_priority_values_first": False,
            "clip_max_new_tokens": 4096, "in_batch_check_threshold": 32,
            "in_batch_deprioritize_threshold": 32, "prefill_max_requests": None,
            "truncation_align_size": None, "lpm_queue_degrade_at": 128,
            "random_seed": 0, "disable_tree": False, "ntr_init_raw": 0.7,
            "schedule_conservativeness": 1.0, "ntr_min_factor": 0.14,
            "ntr_decay_steps": 604, "retract_decode_steps": 20,
        }
        env = {"allocator_avail_tokens": 100_000, "tree_evictable_tokens": 0,
               "num_allocatable_reqs": 1000, "batch_is_full": False,
               "mixed_chunk_allowed": False}
        core = _ext.SchedulerCore(cfg, "lru")
        ci = core.ingest([
            {"rid": 1, "pool_idx": 0, "origin": list(range(32)),
             "max_new_tokens": 16, "priority": 0, "arrival_seq": 0,
             "routing_key": 0, "ignore_eos": False}
        ])[0]

        out_len = 0
        mirror = py_counters([])

        def settle(spec):
            """Mirror one settled spec row into the Python counter state."""
            correct = spec["accept_len"] - 1
            mirror["spec_verify_ct"] += 1
            mirror["spec_num_correct_drafts"] += correct
            _bump(mirror["correct_drafts_histogram"], correct)
            if spec["block_accept_len"] is not None:
                mirror["spec_num_block_accept_tokens"] += spec["block_accept_len"]
            if spec["cap_len"] is not None:
                mirror["spec_num_cap_tokens"] += spec["cap_len"]
                _bump(mirror["cap_lens_histogram"], spec["cap_len"])

        def apply(accepted, spec):
            nonlocal out_len
            core.plan(env)
            core.apply_result(
                [{"accepted": accepted, "finished": False, "finish_reason": 0,
                  "spec": spec}],
                [],
            )
            out_len += len(accepted)
            if spec is not None and spec["settled"]:
                settle(spec)
            self.assertEqual(core.req_out_len(ci), out_len)
            self.assertEqual(core.spec_counters(ci), mirror)

        # Prefill sampled token.
        core.plan(env)
        core.apply_result(
            [{"accepted": [7], "finished": False, "finish_reason": 0,
              "spec": None}],
            [{"core_idx": ci, "row": [0] * 32}],
        )
        out_len = 1

        # Settled spec step: 3 committed tokens, accept_len 4.
        apply([11, 12, 13], {"accept_len": 4, "settled": True,
                             "block_accept_len": 2, "cap_len": 3})
        # Unsettled step: nothing commits, counters untouched.
        apply([], {"accept_len": 2, "settled": False,
                   "block_accept_len": None, "cap_len": 5})
        # Non-spec step: counters untouched.
        apply([99], None)


if __name__ == "__main__":
    unittest.main()
