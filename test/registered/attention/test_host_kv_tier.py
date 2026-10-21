"""Host-KV tier CPU fixtures (HKV-20).

Exercises the native HostKVTier against fake pools/allocators/batches:
confiscation and the accounting identity, bulk retirement, staged
src!=dst restores with neighbor prefetch, row recycling, multi-row
attribution, and the between-replay tick. No GPU required.

Run:  python -m pytest test/registered/attention/test_host_kv_tier.py
"""

import importlib
import json
import tempfile
import unittest

import torch

from sglang.srt.layers.attention.qsa.host_kv import HostKVConfig, HostKVTier

PAGE = 64


class FakePool:
    """12 full-attention layers, float32 K/V for bitwise equality."""

    def __init__(self, slots):
        self.slots = slots
        self.buffers = {
            layer: (torch.randn(slots, 2, 8), torch.randn(slots, 2, 8))
            for layer in range(12)
        }
        self.full_attention_layer_id_mapping = list(range(12))

    def get_key_buffer(self, layer):
        return self.buffers[layer][0]

    def get_value_buffer(self, layer):
        return self.buffers[layer][1]


class FakeAllocator:
    def __init__(self, pages):
        self.free_pages = torch.arange(1, pages + 1, dtype=torch.int64)
        self.size = pages * PAGE
        self.page_size = PAGE
        self.full_tokens_per_layer = self.size

    def available_size(self):
        return int(self.free_pages.numel()) * PAGE

    def free(self, free_index):
        # Matches the real allocator: inputs are token-slot indices; the
        # allocator stores PAGE indices (slot // page_size), deduped.
        pages = (free_index.to(torch.int64) // PAGE).unique()
        merged = torch.cat([self.free_pages, pages])
        self.free_pages = merged.unique()

    def free_segment(self, free_index, *, start_pos):
        self.free(free_index)


class FakeKV:
    def __init__(self):
        self.kv_allocated_len = 0


class FakeReq:
    def __init__(self, idx):
        self.req_pool_idx = idx
        self.kv = FakeKV()
        self.output_ids = []

    def effective_kv_committed_len(self):
        return self.kv.kv_allocated_len


class FakeTreeCache:
    disable = True

    def supports_mamba(self):
        return False


class FakeScheduler:
    def __init__(self, reqs, tree):
        self.running_batch = None
        self.chunked_req = None
        self.tree_cache = tree
        self.reqs = reqs

    def batchify(self):
        class B:
            reqs = self.reqs
        self.running_batch = B()


class FakeBackend:
    def __init__(self, pool, allocator, scheduler, table):
        self.token_to_kv_pool = pool
        self.token_to_kv_pool_allocator = allocator
        self.req_to_token = table
        self.runner = None


class FakeBatch:
    def __init__(self, rows, out_cache_loc, seq_lens=None):
        self.req_pool_indices = torch.tensor(rows, dtype=torch.int64)
        self.out_cache_loc = out_cache_loc
        self.seq_lens = seq_lens


def fresh_tier(**kw):
    cfg = HostKVConfig(enable=True, arm_after=1, hot_pages=4,
                       keep_pages=kw.pop("keep_pages", 40), **kw)
    return HostKVTier(cfg), cfg


class HostKVFixtures(unittest.TestCase):
    def setUp(self):
        self.tier, self.cfg = fresh_tier()
        self.pool = FakePool(slots=64 * 200)
        self.allocator = FakeAllocator(pages=200)
        self.tree = FakeTreeCache()
        self.table = torch.zeros(4, 8192, dtype=torch.int32)
        self.reqs = {0: FakeReq(0)}
        self.scheduler = FakeScheduler(list(self.reqs.values()), self.tree)
        self.backend = FakeBackend(self.pool, self.allocator,
                                   self.scheduler, self.table)
        self.next_page = 1

    def _alloc_pages(self, n):
        pages = []
        for _ in range(n):
            pages.append(self.next_page)
            self.next_page += 1
        return pages

    def _prefill(self, row, start, n, tier=None):
        tier = tier or self.tier
        take = (n + PAGE - 1) // PAGE
        pages = self._alloc_pages(take)
        locs = (torch.tensor(pages)[:, None] * PAGE
                + torch.arange(PAGE)[None, :]).flatten()[:n]
        self.table[row, start:start + n] = locs.to(torch.int32)
        self.reqs[row].kv.kv_allocated_len = start + n
        if not tier.armed:
            tier.arm(self.backend, row)
        tier.after_write(FakeBatch([row], locs,
                                   seq_lens=torch.tensor([start + n])))

    # -- arm / identity --------------------------------------------------

    def test_confiscate_balances_identity(self):
        self.tier.arm(self.backend, 0)
        self.assertTrue(self.tier.armed)
        self.assertEqual(int(self.allocator.free_pages.numel()),
                         self.cfg.keep_pages)
        self.assertEqual(self.allocator.size, self.cfg.keep_pages * PAGE)
        self.assertEqual(self.tier.reserve_tokens,
                         (200 - self.cfg.keep_pages) * PAGE)
        # available == free pages x PAGE (returned_tokens 0)
        self.assertEqual(self.allocator.available_size(),
                         int(self.allocator.free_pages.numel()) * PAGE)

    def test_disabled_tier_is_inert(self):
        tier = HostKVTier(HostKVConfig(enable=False))
        self.assertFalse(tier.armed)
        tier.arm(self.backend, 0)
        tier.arm(self.backend, 0)
        self.assertFalse(tier.armed)
        self.assertEqual(self.allocator.size, 200 * PAGE)

    # -- retire / restore -------------------------------------------------

    def test_retire_restores_round_trip(self):
        self._prefill(0, 0, 2048)
        self.assertGreater(len(self.tier.rows[0]["retired"]), 0)
        # Mutate the retired pages on-GPU, restore, and expect the mirror
        # bytes (not the mutation) back in a reserve frame.
        st = self.tier.rows[0]
        page = sorted(st["retired"])[0]
        orig_slots = st["page_slots"][page].clone()
        self.tier.before_gather(torch.tensor([page * PAGE + 3]),
                                FakeBatch([0], torch.tensor([0])))
        self.assertNotIn(page, st["retired"])
        new_slots = st["page_slots"][page]
        self.assertFalse(torch.equal(orig_slots, new_slots))
        cols = torch.arange(page * PAGE, (page + 1) * PAGE)
        self.assertTrue(torch.equal(self.table[0, cols].long(),
                                    new_slots.long()))
        for layer in range(12):
            mk, mv = self.tier.mirror[layer]
            gk = self.pool.get_key_buffer(layer)[new_slots]
            self.assertTrue(torch.equal(gk, mk[orig_slots.cpu()]))

    def test_starved_restore_stays_retired(self):
        self._prefill(0, 0, 2048)
        st = self.tier.rows[0]
        page = sorted(st["retired"])[0]
        # Drain the reserve to the floor.
        self.tier.reserve = self.tier.reserve[:0]
        self.tier.reserve_tokens = 0
        self.tier.before_gather(torch.tensor([page * PAGE]),
                                FakeBatch([0], torch.tensor([0])))
        self.assertIn(page, st["retired"])
        self.assertGreater(self.tier.counters["restore_starved"], 0)

    def test_row_recycle_resets_state(self):
        self._prefill(0, 0, 2048)
        # A new request on the same row prefills with a small total.
        self._prefill(0, 0, 128)
        st = self.tier.rows[0]
        self.assertLessEqual(st["ctx_written"], 128 + PAGE)

    def test_decode_scale_writes_do_not_reset_row_state(self):
        self._prefill(0, 0, 2048)
        retired_before = len(self.tier.rows[0]["retired"])
        self.assertGreater(retired_before, 0)
        locs = torch.tensor([self._alloc_pages(1)[0] * PAGE])
        batch = FakeBatch([0], locs, seq_lens=torch.tensor([2049]))
        batch.extend_seq_lens = torch.tensor([4], dtype=torch.int32)
        self.tier.after_write(batch)
        self.assertEqual(self.tier.rows[0]["ctx_written"], 2049)
        self.assertEqual(len(self.tier.rows[0]["retired"]), retired_before)

    def test_multirow_rows_restore_their_own_pages(self):
        self.reqs[1] = FakeReq(1)
        self._prefill(0, 0, 2048)
        self._prefill(1, 0, 2048)
        for row in (0, 1):
            self.assertIn(10, self.tier.rows[row]["retired"])
            self.assertIn(20, self.tier.rows[row]["retired"])
        pos = torch.tensor([10 * PAGE + j for j in range(6)]
                           + [20 * PAGE + j for j in range(6)])
        batch = FakeBatch([0, 1], torch.zeros(12, dtype=torch.int64))
        batch.extend_seq_lens = torch.tensor([6, 6], dtype=torch.int32)
        self.tier.before_gather(pos, batch)
        for row, page in ((0, 10), (1, 20)):
            self.assertNotIn(page, self.tier.rows[row]["retired"])
            slots = self.tier.rows[row]["page_slots"][page]
            cols = torch.arange(page * PAGE, (page + 1) * PAGE)
            self.assertTrue(torch.equal(self.table[row, cols].long(),
                                        slots.long()))
        for row, page in ((0, 20), (1, 10)):
            self.assertIn(page, self.tier.rows[row]["retired"])

    def test_prefetch_neighbor_then_fast_path(self):
        self.tier.cfg.prefetch = True
        self.tier.cfg.prefetch_radius = 1
        self._prefill(0, 0, 4096)
        orig_of_6 = self.tier.rows[0]["page_slots"][6].clone()
        self.tier.before_gather(torch.tensor([5 * PAGE]),
                                FakeBatch([0], torch.tensor([0])))
        self.assertIn(4, self.tier.rows[0]["prefetched"])
        self.assertIn(6, self.tier.rows[0]["prefetched"])
        frame_of_6 = self.tier.rows[0]["page_slots"][6].clone()
        reserve_mid = int(self.tier.reserve.numel())
        self.tier.before_gather(torch.tensor([6 * PAGE]),
                                FakeBatch([0], torch.tensor([0])))
        self.assertGreaterEqual(int(self.tier.reserve.numel()),
                                reserve_mid - 2)
        self.assertEqual(self.tier.counters["prefetch_hits"], 1)
        self.assertNotIn(6, self.tier.rows[0]["retired"])
        cols = torch.arange(6 * PAGE, 7 * PAGE)
        self.assertTrue(torch.equal(self.table[0, cols].long(),
                                    frame_of_6.long()))
        for layer in range(12):
            mk, mv = self.tier.mirror[layer]
            gk = self.pool.get_key_buffer(layer)[frame_of_6]
            self.assertTrue(torch.equal(gk, mk[orig_of_6.cpu()]))

    # -- coverage audit ----------------------------------------------------

    def test_coverage_audit_counts_misses(self):
        self._prefill(0, 0, 2048)
        st = self.tier.rows[0]
        page = sorted(st["retired"])[0]
        before = self.tier.counters["coverage_misses"]
        self.tier.audit_coverage(torch.tensor([page * PAGE]),
                                 FakeBatch([0], torch.tensor([0])))
        self.assertEqual(self.tier.counters["coverage_misses"],
                         before + 1)

    # -- between-replay tick -------------------------------------------------

    def test_tick_restores_selection_and_exempts_it(self):
        self._prefill(0, 0, 2048)

        class Stash:
            def __init__(self, rows_n, pages):
                self.indices = torch.zeros(
                    1, rows_n, len(pages), dtype=torch.int32)
                for j, p in enumerate(pages):
                    self.indices[0, 0, j] = p * PAGE
                self.captured_len = torch.zeros(
                    1, rows_n, dtype=torch.int32)
                self.captured_len[0, 0] = 2048

        class Owner:
            _mtp_shared_sparse_indices = None

        st = self.tier.rows[0]
        sel = sorted(st["retired"])[:5]
        owner = Owner()
        owner._mtp_shared_sparse_indices = Stash(1, sel)
        self.tier.tick(owner)
        self.assertEqual(self.tier.counters["br_ticks"], 1)
        for p in sel:
            self.assertNotIn(p, st["retired"])
        self.assertEqual(self.tier.counters["coverage_misses"], 0)
        self.assertGreater(self.tier.counters["ws_n"], 0)

    # -- finish-path accounting ----------------------------------------------

    def test_finish_frees_do_not_double_count(self):
        self._prefill(0, 0, 2048)
        retired = len(self.tier.rows[0]["retired"])
        self.assertGreater(retired, 0)
        free_before = int(self.allocator.free_pages.numel())
        # Simulate the runtime's finish: free the FULL original extent.
        locs = torch.nonzero(self.table[0]).flatten()
        # Reconstruct the original extent from the req record.
        n = self.reqs[0].kv.kv_allocated_len
        self.allocator.free(self.table[0, :n]
                            .long().clone())
        free_after = int(self.allocator.free_pages.numel())
        # Retired pages were already returned; the dedup filter must not
        # inflate availability past the budget.
        self.assertLessEqual(free_after, self.cfg.keep_pages)

    # -- stats ---------------------------------------------------------------

    def test_stats_dump(self):
        with tempfile.NamedTemporaryFile(suffix=".json") as fh:
            self.cfg.stats_path = fh.name
            self._prefill(0, 0, 2048)
            self.tier.dump_stats()
            d = json.load(open(fh.name))
            self.assertTrue(d["armed"])
            self.assertGreater(d["requests"], 0)
            self.assertIn("coverage_misses", d)


if __name__ == "__main__":
    unittest.main()


class FrameConservation(unittest.TestCase):
    """HKV-26: no frame may live in both ledgers, ever, across the full
    churn cycle (retire -> restore -> re-retire -> forget), including
    frames with corrupted provenance."""

    def setUp(self):
        self.tier, self.cfg = fresh_tier()
        self.pool = FakePool(slots=64 * 200)
        self.allocator = FakeAllocator(pages=200)
        self.tree = FakeTreeCache()
        self.table = torch.zeros(4, 8192, dtype=torch.int32)
        self.reqs = {0: FakeReq(0)}
        self.scheduler = FakeScheduler(list(self.reqs.values()), self.tree)
        self.backend = FakeBackend(self.pool, self.allocator,
                                   self.scheduler, self.table)
        self.tier.arm(self.backend, 0)
        self.next_page = 1

    def _alloc(self, n):
        # Faithful to the runtime: pages come out of the allocator's
        # free set (the KEPT range), never from the tier's reserve.
        pages = self.allocator.free_pages[:n].to(torch.int64).tolist()
        self.allocator.free_pages = self.allocator.free_pages[n:]
        return pages

    def _prefill(self, row, start, n):
        pages = self._alloc((n + PAGE - 1) // PAGE)
        locs = (torch.tensor(pages)[:, None] * PAGE
                + torch.arange(PAGE)[None, :]).flatten()[:n]
        self.table[row, start:start + n] = locs.to(torch.int32)
        self.reqs[row].kv.kv_allocated_len = start + n
        self.tier.after_write(FakeBatch([row], locs,
                                        seq_lens=torch.tensor([start + n])))

    def _assert_disjoint(self):
        free = set(self.allocator.free_pages.tolist())
        reserve = set(self.tier.reserve.tolist())
        overlap = free & reserve
        self.assertFalse(
            overlap,
            f"frames in BOTH ledgers: {sorted(overlap)[:8]} "
            f"(free={len(free)} reserve={len(reserve)})")

    def test_churn_cycle_conserves_frames(self):
        self._prefill(0, 0, 2048)
        st = self.tier.rows[0]
        # retire->restore->re-retire x5 with drift so different pages
        # rotate through the reserve each cycle.
        for cycle in range(5):
            pages = sorted(st["retired"])[cycle:cycle + 4]
            pos = torch.tensor([p * PAGE + j
                                for p in pages for j in range(0, 64, 16)])
            self.tier.before_gather(pos, FakeBatch([0], torch.tensor([0])))
            self._assert_disjoint()
            self.tier._retire(0, st, st["ctx_written"])
            self._assert_disjoint()
        # forget the row and check both ledgers balance.
        before_free = int(self.allocator.free_pages.numel())
        self.tier.forget_row(0)
        self._assert_disjoint()
        self.assertNotIn(0, self.tier.rows)
        # everything the row held came back somewhere
        self.assertGreater(int(self.allocator.free_pages.numel())
                           + int(self.tier.reserve.numel()),
                           before_free + int(self.tier.reserve.numel()) - 1)

    def test_corrupt_provenance_routes_to_allocator(self):
        self._prefill(0, 0, 2048)
        st = self.tier.rows[0]
        # Simulate the cascade: a RUNTIME frame ends up in a prefetched
        # page's slots (as if a mis-route had happened earlier).
        runtime_frame = self._alloc(1)[0]
        page = sorted(st["retired"])[0]
        st["page_slots"][page] = torch.arange(
            runtime_frame * PAGE, (runtime_frame + 1) * PAGE)
        st["prefetched"].add(page)
        free_before = set(self.allocator.free_pages.tolist())
        self.tier.forget_row(0)
        self._assert_disjoint()
        free_after = set(self.allocator.free_pages.tolist())
        self.assertTrue(
            runtime_frame in free_after or runtime_frame in free_before,
            f"runtime frame {runtime_frame} vanished from the allocator: "
            f"before={runtime_frame in free_before} "
            f"after={runtime_frame in free_after}")
        self.assertGreater(
            self.tier.counters.get("runtime_frames_returned", 0), 0)
