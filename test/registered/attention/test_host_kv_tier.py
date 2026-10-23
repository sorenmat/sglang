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


class FaithfulSegmentAllocator(FakeAllocator):
    """Mimics PagedTokenToKVPoolAllocator exactly where the tier's wraps
    touch it: free() pushes unique page ids of the given slots WITHOUT
    deduping against the existing free list, and free_segment() recovers
    page ids from stride slices (valid only on unmasked segments)."""

    def free(self, free_index):
        if free_index.numel() == 0:
            return
        pages = torch.unique(free_index.to(torch.int64) // PAGE)
        self.free_pages = torch.cat((pages, self.free_pages))

    def free_segment(self, free_index, *, start_pos):
        if free_index.numel() == 0:
            return
        ps = PAGE
        offset = start_pos % ps
        if offset == 0:
            pieces = (free_index[::ps],)
        else:
            pieces = (free_index[:1], free_index[ps - offset::ps])
        self.free_pages = torch.cat(
            (*(p // ps for p in pieces), self.free_pages))


class FreeSegmentContamination(unittest.TestCase):
    """HKV-27 hardening lock: a tier-contaminated segment (zeroed retired
    slots, staged tier frames) must flow through the free_segment wrap
    page-exactly. Contamination is always page-granular in practice, which
    preserves the stride-slice alignment - this test pins that the wrap
    frees exactly the runtime pages, keeps tier frames out of the
    allocator ledger, and never duplicates a page id, whatever path it
    takes internally."""

    def setUp(self):
        self.tier, self.cfg = fresh_tier()
        self.pool = FakePool(slots=PAGE * 260)
        self.allocator = FaithfulSegmentAllocator(pages=259)
        self.tree = FakeTreeCache()
        self.table = torch.zeros(4, 8192, dtype=torch.int32)
        self.reqs = {0: FakeReq(0)}
        self.scheduler = FakeScheduler(list(self.reqs.values()), self.tree)
        self.backend = FakeBackend(self.pool, self.allocator,
                                   self.scheduler, self.table)
        self.tier.arm(self.backend, 0)
        self.next_page = 1

    def _alloc(self, n):
        pages = self.allocator.free_pages[:n].to(torch.int64).tolist()
        self.allocator.free_pages = self.allocator.free_pages[n:]
        return pages

    def _prefill(self, row, start, n):
        # Chunked like the runtime: frames recycle through retire as the
        # context streams past the kept budget.
        pos = start
        while pos < start + n:
            chunk = min(512, start + n - pos)
            pages = self._alloc((chunk + PAGE - 1) // PAGE)
            locs = (torch.tensor(pages)[:, None] * PAGE
                    + torch.arange(PAGE)[None, :]).flatten()[:chunk]
            self.table[row, pos:pos + chunk] = locs.to(torch.int32)
            pos += chunk
            self.reqs[row].kv.kv_allocated_len = pos
            self.tier.after_write(FakeBatch([row], locs,
                                            seq_lens=torch.tensor([pos])))
            self.tier._retire(row, self.tier.rows[row], pos)

    def test_contaminated_segment_frees_exact_pages(self):
        self._prefill(0, 0, 4096)
        st = self.tier.rows[0]
        # Restore one retired page so the row holds a staged TIER frame
        # and the table carries live staged slots; other retired pages
        # read as slot 0 after retirement zeroing.
        page = sorted(st["retired"])[0]
        positions = torch.tensor(
            [page * PAGE + j for j in range(0, PAGE, 16)])
        self.tier.before_gather(positions,
                                FakeBatch([0], torch.tensor([0])))
        self.assertIn(page, st["restored"])
        # The segment the runtime frees at finish: full extent,
        # contaminated with zeros and one tier frame.
        extent = self.table[0, :4096].long().clone()
        zeroed = int((extent == 0).sum())
        self.assertGreater(zeroed, PAGE)
        tier_held = len(st["restored"]) + len(st["prefetched"])
        expect_free_pages = set(self.allocator.free_pages.tolist())
        for p in range(0, 4096 // PAGE):
            slots = extent[p * PAGE:(p + 1) * PAGE]
            frame = int(slots[0].item()) // PAGE
            if frame == 0 or p == page:
                continue  # zeroed retired page or the staged tier page
            expect_free_pages.add(frame)
        before_free = int(self.allocator.free_pages.numel())
        self.allocator.free_segment(extent, start_pos=0)
        after = self.allocator.free_pages.tolist()
        # No page may double-enter the free list (silent cross-request
        # frame sharing downstream).
        self.assertEqual(len(after), len(set(after)),
                         f"duplicate page ids in free list: "
                         f"{sorted({x for x in after if after.count(x) > 1})}")
        self.assertEqual(set(after), expect_free_pages)
        # No tier frame leaked into the allocator's ledger.
        overlap = set(after) & set(self.tier.reserve.tolist())
        self.assertFalse(overlap, f"tier frames in free list: {overlap}")
        staged = int(st["page_slots"][page][0].item()) // PAGE
        self.assertNotIn(staged, after)
        # And the tier still owns its staged frame for the row.
        self.assertEqual(len(st["restored"]) + len(st["prefetched"]),
                         tier_held)
        self.assertGreater(int(self.allocator.free_pages.numel()),
                           before_free)

    def test_finish_cycles_conserve_exactly(self):
        """HKV-27: full request lifetimes (chunked prefill with retire,
        restore + prefetch, decode growth, forget, full-extent finish
        free) must conserve the frame ledger EXACTLY, cycle after cycle -
        no drift is acceptable, however small."""
        keep = int(self.tier.cfg.keep_pages)
        baseline = (int(self.allocator.free_pages.numel())
                    + int(self.tier.reserve.numel()))
        for cycle in range(40):
            self._prefill(0, 0, 4096)
            st = self.tier.rows[0]
            retired_sorted = sorted(st["retired"])
            if retired_sorted:
                positions = torch.tensor(
                    [p * PAGE + j for p in retired_sorted[:3]
                     for j in range(0, PAGE, 16)])
                self.tier.before_gather(
                    positions, FakeBatch([0], torch.tensor([0])))
            dec_pages = self._alloc(3)
            dec_locs = (torch.tensor(dec_pages)[:, None] * PAGE
                        + torch.arange(PAGE)[None, :]).flatten()
            self.table[0, 4096:4096 + len(dec_locs)] = \
                dec_locs.to(torch.int32)
            self.tier.after_write(FakeBatch(
                [0], dec_locs,
                seq_lens=torch.tensor([4096 + len(dec_locs)])))
            extent = self.table[0, :4096 + len(dec_locs)].long().clone()
            self.tier.forget_row(0)
            self.allocator.free(extent)
            self.table[0, :4096 + len(dec_locs)] = 0
            held = sum(len(st2["restored"]) + len(st2["prefetched"])
                       for st2 in self.tier.rows.values())
            total = (int(self.allocator.free_pages.numel())
                     + int(self.tier.reserve.numel()) + held)
            self.assertEqual(
                total, baseline,
                f"cycle {cycle}: ledger drifted by {total - baseline} "
                f"frames (free={int(self.allocator.free_pages.numel())} "
                f"reserve={int(self.tier.reserve.numel())} held={held})")
            self.assertLessEqual(
                int(self.allocator.free_pages.numel()), keep)


    def test_orphan_sweeper_reclaims_after_two_forgets(self):
        """HKV-27: a page allocated by the runtime but never table-written
        (final verify/draft tail) is unreachable to every table-read-based
        release path. The sweeper in forget_row must return it to the
        allocator once it stays unreferenced across two forgets."""
        with tempfile.NamedTemporaryFile(suffix=".json") as fh:
            self.tier.cfg.stats_path = fh.name
            self._prefill(0, 0, 2048)
            # Make a genuine orphan: take a page from the allocator (as
            # alloc_extend would) and let go of every reference to it.
            free_before = set(self.allocator.free_pages.tolist())
            orphan = free_before.pop()
            self.allocator.free_pages = torch.tensor(
                sorted(free_before), dtype=torch.int64)
            self.assertNotIn(orphan,
                             set(self.allocator.free_pages.tolist()))

            def _finish_cycle(n_tokens):
                self._prefill(0, 0, n_tokens)
                self.tier.forget_row(0)
                self.allocator.free(
                    self.table[0, :n_tokens].long().clone())
                self.table[0, :4096] = 0

            # Distinct chunk counts per cycle: regression guard for the
            # global (first-slot, size) dedupe, which skipped the first
            # chunk of a recycled request whose pages reproduced the
            # previous forward's key (rows[0] never re-created).
            _finish_cycle(1024)  # first forget: orphan seen, not swept
            self.assertNotIn(orphan,
                             set(self.allocator.free_pages.tolist()))
            _finish_cycle(1152)  # second forget: stable orphan -> swept
            self.assertIn(
                orphan, set(self.allocator.free_pages.tolist()),
                f"orphan page {orphan} not reclaimed; "
                f"counters={self.tier.counters}")
            self.assertGreaterEqual(
                self.tier.counters.get("orphan_reclaimed", 0), 1)

    def test_orphan_sweeper_runs_without_stats_path(self):
        """The sweep is the HKV-27 fix, not instrumentation: it must
        run with stats dumping disabled (stats_path unset)."""
        self.assertIsNone(self.tier.cfg.stats_path)
        self._prefill(0, 0, 2048)
        free_before = set(self.allocator.free_pages.tolist())
        orphan = free_before.pop()
        self.allocator.free_pages = torch.tensor(
            sorted(free_before), dtype=torch.int64)

        def _finish_cycle(n_tokens):
            self._prefill(0, 0, n_tokens)
            self.tier.forget_row(0)
            self.allocator.free(
                self.table[0, :n_tokens].long().clone())
            self.table[0, :4096] = 0

        _finish_cycle(1024)
        _finish_cycle(1152)
        self.assertIn(orphan, set(self.allocator.free_pages.tolist()))
        self.assertGreaterEqual(
            self.tier.counters.get("orphan_reclaimed", 0), 1)


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


class AutoArm(unittest.TestCase):
    def _tier(self, min_tokens):
        return HostKVTier(HostKVConfig(
            enable=True, arm_after=1, keep_pages=40,
            arm_min_input_tokens=min_tokens))

    def test_auto_stays_dormant_below_threshold(self):
        tier = self._tier(200000)
        pool = FakePool(64 * 200)
        allocator = FakeAllocator(200)
        table = torch.zeros(4, 8192, dtype=torch.int32)
        reqs = [FakeReq(0)]
        sched = FakeScheduler(reqs, FakeTreeCache())
        backend = FakeBackend(pool, allocator, sched, table)
        tier.arm(backend, 0)
        tier.arm(backend, 0)
        self.assertFalse(tier.armed)
        self.assertTrue(tier._auto_pending)
        # A small request does not arm.
        batch = FakeBatch([0], torch.tensor([100 * PAGE]),
                          seq_lens=torch.tensor([100 * PAGE]))
        tier._row_state(0)
        tier.after_write(batch)
        self.assertFalse(tier.armed)
        self.assertEqual(allocator.size, 200 * PAGE)  # no confiscation

    def test_auto_arms_on_oversized_request(self):
        tier = self._tier(2000)
        pool = FakePool(64 * 200)
        allocator = FakeAllocator(200)
        table = torch.zeros(4, 8192, dtype=torch.int32)
        reqs = [FakeReq(0)]
        sched = FakeScheduler(reqs, FakeTreeCache())
        backend = FakeBackend(pool, allocator, sched, table)
        tier.arm(backend, 0)
        self.assertFalse(tier.armed)
        batch = FakeBatch([0], torch.tensor([50 * PAGE]),
                          seq_lens=torch.tensor([50 * PAGE]))
        tier._row_state(0)
        tier.after_write(batch)
        self.assertTrue(tier.armed)
        self.assertLess(allocator.size, 200 * PAGE)  # confiscated
