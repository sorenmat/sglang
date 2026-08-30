"""CPU-only unit tests for --enable-adaptive-prefill scheduling policy.

Pins the decode-pressure policy: running decode requests get a latency
budget; prefill chunks shrink so their projected cost (from a measured
prefill-throughput EWMA) fits in the remaining budget, and once the budget
is exceeded the scheduler yields the iteration to decode. Targets agent
traffic (bursty tool-return prefills alongside steady decoding) where
prefill-first scheduling otherwise starves decode.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _scheduler(
    *,
    enabled=True,
    budget_ms=50.0,
    min_chunk=2048,
    base_chunk=8192,
    tps=None,
    page_size=1,
    waiting_queue=(),
    chunked_req=None,
):
    s = object.__new__(Scheduler)
    s.enable_adaptive_prefill = enabled
    s.decode_latency_budget_s = budget_ms / 1000.0
    s.adaptive_prefill_min_chunk_tokens = min_chunk
    s.chunked_prefill_size = base_chunk
    s.page_size = page_size
    s.waiting_queue = list(waiting_queue)
    s.chunked_req = chunked_req
    s._prefill_dispatch_probe = None
    s._prefill_tps_ewma = tps
    s._last_adaptive_yield_log_t = 0.0
    s._last_decode_dispatch_t = 0.0
    return s


def _running_batch(n=4):
    return SimpleNamespace(
        is_empty=lambda: n == 0,
        is_prefill_only=False,
        batch_size=lambda: n,
    )


def _batch(mode, extend_lens=None):
    return SimpleNamespace(forward_mode=mode, extend_lens=extend_lens or [])


class TestAdaptiveChunkBudget(unittest.TestCase):
    def test_disabled_is_uncapped(self):
        s = _scheduler(enabled=False)
        self.assertEqual(s._adaptive_prefill_chunk_budget(_running_batch()), -1)
        self.assertIsNone(s._adaptive_chunk_size(_running_batch()))

    def test_no_decode_pressure_is_uncapped(self):
        s = _scheduler(tps=1000.0)
        empty = SimpleNamespace(
            is_empty=lambda: True, is_prefill_only=False, batch_size=lambda: 0
        )
        self.assertEqual(s._adaptive_prefill_chunk_budget(empty), -1)
        self.assertIsNone(s._adaptive_chunk_size(empty))

    def test_no_throughput_estimate_keeps_base_chunk(self):
        """Before the EWMA sees its first prefill sample, keep the configured
        chunk (the estimate needs at least one chunk to bootstrap)."""
        import time

        s = _scheduler(tps=None)
        s._last_decode_dispatch_t = time.monotonic()  # wait ~0 < budget
        self.assertEqual(s._adaptive_prefill_chunk_budget(_running_batch()), -1)
        self.assertIsNone(s._adaptive_chunk_size(_running_batch()))

    def test_chunk_fits_remaining_budget(self):
        """Chunk = measured tokens/sec * remaining budget, clamped to base and
        floored. The budget's floor-chunk invariant (>= floor/2000 s) means
        the remaining-budget cap only binds when the operator sets a budget
        above that bound."""
        import time

        # budget_ms 1200 > floor bound 1024 ms: 25 ms remain at a 1175 ms wait
        # -> 100K tok/s * 25 ms = 2500 tokens, floored from below by nothing.
        s2 = _scheduler(tps=100_000.0, budget_ms=1200.0, base_chunk=8192)
        s2._last_decode_dispatch_t = time.monotonic() - 1.175
        self.assertAlmostEqual(
            s2._adaptive_chunk_size(_running_batch()), 2500, delta=4
        )

        # low tps: the cap falls under the 2048 floor
        s = _scheduler(tps=1000.0, budget_ms=1200.0, base_chunk=8192)
        s._last_decode_dispatch_t = time.monotonic() - 1.175
        self.assertEqual(s._adaptive_chunk_size(_running_batch()), 2048)

        # remaining budget allows more than the base chunk: stay at base
        s3 = _scheduler(tps=10_000_000.0, budget_ms=1200.0, base_chunk=8192)
        s3._last_decode_dispatch_t = time.monotonic() - 1.175
        self.assertEqual(s3._adaptive_chunk_size(_running_batch()), 8192)

    def test_backlog_grows_the_chunk_floor(self):
        """A large prefill backlog raises the floor toward the base chunk so
        admission is never starved by tiny chunks + yield rounds."""
        import time
        from types import SimpleNamespace

        queue = [SimpleNamespace(origin_input_ids=[0] * 40000) for _ in range(6)]
        # remaining-budget cap = 20K tok/s * 25 ms = 500 tokens; the backlog
        # floor (240K/48 = 5000) must win over both the cap and the 2048 min.
        s = _scheduler(tps=20_000.0, budget_ms=100.0, waiting_queue=queue)
        s._last_decode_dispatch_t = time.monotonic() - 0.075
        # stretched budget (1.5*5000/20K=375ms) minus the 75ms wait caps the
        # chunk near 6000; the 5000 floor bounds it from below either way.
        size = s._adaptive_chunk_size(_running_batch())
        self.assertGreaterEqual(size, 5000)
        self.assertLessEqual(size, 8192)

        flooded = [SimpleNamespace(origin_input_ids=[0] * 100000) for _ in range(8)]
        s2 = _scheduler(tps=20_000.0, budget_ms=100.0, waiting_queue=flooded)
        s2._last_decode_dispatch_t = time.monotonic() - 0.075
        # 800K backlog -> floor saturates at the 8192 base chunk -> the
        # policy steps aside entirely (prefill-first, no shrink, no yield)
        self.assertEqual(s2._adaptive_prefill_chunk_budget(_running_batch()), -1)
        self.assertIsNone(s2._adaptive_chunk_size(_running_batch()))

        # without a throughput estimate (EWMA cold) nothing shrinks at all
        s3 = _scheduler(tps=None, waiting_queue=queue)
        import time as t
        s3._last_decode_dispatch_t = t.monotonic()
        self.assertIsNone(s3._adaptive_chunk_size(_running_batch()))

    def test_budget_covers_floor_chunk_cost(self):
        """Consistency invariant: the decode budget must cover one floor-sized
        chunk (at a conservative 2000 tok/s), otherwise every chunk overshoots
        and the yield fires every round -- time-slicing prefill/decode 50/50
        (measured: halves admission throughput). The invariant holds even with
        an inflated EWMA (overlap scheduling measures CPU cadence, not GPU)."""
        import time
        from types import SimpleNamespace

        # floor 2048 (no backlog) -> budget >= 1.024 s: a 200 ms wait must
        # NOT yield even though decode_latency_budget_ms is only 100.
        s = _scheduler(tps=1_000_000.0, budget_ms=100.0)  # inflated EWMA
        s._last_decode_dispatch_t = time.monotonic() - 0.2
        self.assertGreater(s._adaptive_prefill_chunk_budget(_running_batch()), 0)
        # a wait beyond floor/2000 still yields
        s2 = _scheduler(tps=1_000_000.0, budget_ms=100.0)
        s2._last_decode_dispatch_t = time.monotonic() - 1.5
        self.assertEqual(s2._adaptive_prefill_chunk_budget(_running_batch()), 0)
        # backlog-grown floor 5000 -> budget >= 2.5 s
        queue = [SimpleNamespace(origin_input_ids=[0] * 40000) for _ in range(6)]
        s3 = _scheduler(tps=1_000_000.0, budget_ms=100.0, waiting_queue=queue)
        s3._last_decode_dispatch_t = time.monotonic() - 1.0
        self.assertGreater(s3._adaptive_prefill_chunk_budget(_running_batch()), 0)
        s3._last_decode_dispatch_t = time.monotonic() - 3.0
        self.assertEqual(s3._adaptive_prefill_chunk_budget(_running_batch()), 0)

    def test_over_budget_yields(self):
        import time

        # the yield needs a wait beyond the invariant bound: floor 2048 ->
        # budget >= 1024 ms; a 1.5 s wait yields, a 0.5 s wait does not.
        s = _scheduler(tps=1000.0, budget_ms=50.0)
        s._last_decode_dispatch_t = time.monotonic() - 1.5
        self.assertEqual(s._adaptive_prefill_chunk_budget(_running_batch()), 0)
        s_nog = _scheduler(tps=1000.0, budget_ms=50.0)
        s_nog._last_decode_dispatch_t = time.monotonic() - 0.5
        self.assertGreater(
            s_nog._adaptive_prefill_chunk_budget(_running_batch()), 0
        )

    def test_floor_respects_page_size(self):
        import time

        s = _scheduler(tps=1.0, budget_ms=100.0, min_chunk=64, page_size=256)
        s._last_decode_dispatch_t = time.monotonic() - 0.075
        self.assertEqual(s._adaptive_chunk_size(_running_batch()), 256)


class TestAdaptiveSignals(unittest.TestCase):
    def test_ewma_first_sample_then_blend(self):
        import time

        s = _scheduler(tps=None)
        now = time.monotonic()
        s._prefill_dispatch_probe = (now - 1.0, 8192)  # ~8192 tok/s
        s._update_adaptive_prefill_signals()
        self.assertAlmostEqual(s._prefill_tps_ewma, 8192, delta=16)
        s._prefill_dispatch_probe = (now - 1.0, 4096)
        s._update_adaptive_prefill_signals()
        self.assertAlmostEqual(s._prefill_tps_ewma, 0.3 * 4096 + 0.7 * 8192, delta=16)

    def test_probe_cleared_and_zero_tokens_ignored(self):
        import time

        s = _scheduler(tps=None)
        now = time.monotonic()
        s._prefill_dispatch_probe = (now, 0)
        s._update_adaptive_prefill_signals()
        self.assertIsNone(s._prefill_tps_ewma)
        self.assertIsNone(s._prefill_dispatch_probe)

    def test_dispatch_stamp_modes(self):
        import time

        s = _scheduler()
        s._last_decode_dispatch_t = 0.0

        s._stamp_adaptive_prefill_dispatch(_batch(ForwardMode.DECODE))
        decode_t = s._last_decode_dispatch_t
        self.assertGreater(decode_t, 0)
        self.assertIsNone(s._prefill_dispatch_probe)

        # TARGET_VERIFY (spec decode's decode step) also stamps decode.
        s._stamp_adaptive_prefill_dispatch(_batch(ForwardMode.TARGET_VERIFY))
        self.assertGreaterEqual(s._last_decode_dispatch_t, decode_t)

        # EXTEND stamps the token probe.
        s._prefill_dispatch_probe = None
        s._stamp_adaptive_prefill_dispatch(
            _batch(ForwardMode.EXTEND, extend_lens=[1000, 233])
        )
        self.assertEqual(s._prefill_dispatch_probe[1], 1233)

        # MIXED does both.
        s._prefill_dispatch_probe = None
        t0 = s._last_decode_dispatch_t
        s._stamp_adaptive_prefill_dispatch(
            _batch(ForwardMode.MIXED, extend_lens=[512, 1, 1])
        )
        self.assertEqual(s._prefill_dispatch_probe[1], 514)
        self.assertGreaterEqual(s._last_decode_dispatch_t, t0)

        # None and disabled are no-ops.
        before = s._prefill_dispatch_probe
        s._stamp_adaptive_prefill_dispatch(None)
        self.assertEqual(s._prefill_dispatch_probe, before)
        s2 = _scheduler(enabled=False)
        s2._stamp_adaptive_prefill_dispatch(_batch(ForwardMode.DECODE))
        self.assertEqual(s2._last_decode_dispatch_t, 0.0)


class TestInitGating(unittest.TestCase):
    def _publish(self, **fields):
        from sglang.srt import runtime_context as rc

        return rc.get_context().override_server_args(**fields)

    def _init(self, *, chunked_prefill_size=8192, dynamic_chunking=False):
        s = object.__new__(Scheduler)
        s.chunked_prefill_size = chunked_prefill_size
        s.enable_dynamic_chunking = dynamic_chunking
        s._init_adaptive_prefill()
        return s

    def test_enabled_by_default_path(self):
        with self._publish(
            enable_adaptive_prefill=True,
            decode_latency_budget_ms=40.0,
            adaptive_prefill_min_chunk_tokens=2048,
        ):
            s = self._init()
            self.assertTrue(s.enable_adaptive_prefill)
            self.assertAlmostEqual(s.decode_latency_budget_s, 0.04)
            self.assertEqual(s.adaptive_prefill_min_chunk_tokens, 2048)

    def test_disabled_without_chunked_prefill(self):
        with self._publish(enable_adaptive_prefill=True):
            s = self._init(chunked_prefill_size=None)
            self.assertFalse(s.enable_adaptive_prefill)

    def test_disabled_under_dp_attention(self):
        with self._publish(enable_adaptive_prefill=True, enable_dp_attention=True):
            s = self._init()
            self.assertFalse(s.enable_adaptive_prefill)

    def test_disabled_under_pp_dynamic_chunking(self):
        with self._publish(enable_adaptive_prefill=True):
            s = self._init(dynamic_chunking=True)
            self.assertFalse(s.enable_adaptive_prefill)

    def test_off_by_default(self):
        with self._publish():
            s = self._init()
            self.assertFalse(s.enable_adaptive_prefill)


if __name__ == "__main__":
    unittest.main()
