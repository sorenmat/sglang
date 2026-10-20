"""HKV-4 CPU tests: the shrunk mode's accounting moves in hosttier23m.

Fake scheduler/tree-cache/allocator/reqs drive the REAL module functions;
the oracle is the runtime's own identity
(available + protected + uncached == total) after every move.
"""

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "host_kv"))

KV_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]
PAGE = 64


class FakeAllocator:
    """Mirrors the real allocator: page 0 (the padding slot) is never in
    free_pages. After confiscation size is SET to KEEP_PAGES*64, so the
    kept pages balance it exactly — no permanent padding term."""
    def __init__(self, pages):
        self.page_size = PAGE
        self.size = pages * PAGE
        self.free_pages = torch.arange(1, pages, dtype=torch.int32)

    def get_kvcache(self):
        return None

    def free(self, free_index):
        if free_index.numel() == 0:
            return
        pages = torch.unique((free_index // PAGE).to(torch.int32))
        have = set(self.free_pages.tolist())
        add = [p for p in pages.tolist() if p not in have]
        if add:
            self.free_pages = torch.cat(
                [self.free_pages, torch.tensor(add, dtype=torch.int32)])

    def free_segment(self, free_index, *, start_pos):
        self.free(free_index)

    def available_size(self):
        return int(self.free_pages.numel()) * PAGE


class FakeTreeCache:
    def __init__(self):
        self.prot = 0

    def protected_size(self):
        return self.prot

    def full_protected_size(self):
        return self.prot


class FakeReq:
    def __init__(self, row):
        self.req_pool_idx = row
        self.kv = type("KV", (), {"kv_allocated_len": 0})()
        self.cache_protected_len = 0


class FakeScheduler:
    def __init__(self, reqs, tree):
        self.reqs = reqs
        self.running_batch = type("RB", (), {"reqs": reqs})()
        self.chunked_req = None
        self.tree_cache = tree


class FakePool:
    def __init__(self, slots):
        self.full_attention_layer_id_mapping = {l: i for i, l in enumerate(KV_LAYERS)}
        self.k = [torch.randn(slots, 2, 256, dtype=torch.bfloat16) for _ in KV_LAYERS]
        self.v = [torch.randn(slots, 2, 256, dtype=torch.bfloat16) for _ in KV_LAYERS]

    def _t(self, layer):
        return self.full_attention_layer_id_mapping[layer]

    def get_key_buffer(self, layer):
        return self.k[self._t(layer)]

    def get_value_buffer(self, layer):
        return self.v[self._t(layer)]


class FakeBackend:
    def __init__(self, pool, allocator, scheduler, table):
        self.token_to_kv_pool = pool
        self.token_to_kv_pool_allocator = allocator
        self.req_to_token = table
        self.runner = type("R", (), {"scheduler": scheduler})()


class FakeBatch:
    def __init__(self, rows, locs):
        self.req_pool_indices = torch.tensor(rows, dtype=torch.int64)
        self.out_cache_loc = locs


def fresh_shrunk(keep_pages=40):
    log = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    log.close()
    os.environ.update({"SGLANG_HOSTTIER_ENABLE": "1",
                       "SGLANG_HOSTTIER_23M_LOG": log.name,
                       "SGLANG_HOSTTIER_ARM_AFTER": "1",
                       "SGLANG_HOSTTIER_HOT_PAGES": "4",
                       "SGLANG_HOSTTIER_SHRUNK": "1",
                       "SGLANG_HOSTTIER_KEEP_PAGES": str(keep_pages)})
    module = importlib.import_module("hosttier23m")
    importlib.reload(module)
    return module, log.name


class ShrunkMode(unittest.TestCase):
    def setUp(self):
        self.ht, self.log = fresh_shrunk(keep_pages=40)
        self.pool = FakePool(slots=64 * 200)
        self.allocator = FakeAllocator(pages=200)
        self.tree = FakeTreeCache()
        self.table = torch.zeros(4, 8192, dtype=torch.int32)
        self.reqs = {0: FakeReq(0)}
        self.scheduler = FakeScheduler(list(self.reqs.values()), self.tree)
        self.backend = FakeBackend(self.pool, self.allocator, self.scheduler, self.table)
        self.next_slot = 0
        self.ht.arm(self.backend, 0)  # arm BEFORE allocations, as at startup

    def _identity_lhs(self):
        # The runtime checks available + evictable + protected + uncached
        # == total. Simulate the radix's finish-time claim of the tier's
        # returned tokens (the wrap subtracts them from available until
        # exactly this claim reconciles them).
        radix_claim = self.ht.G.get("returned_tokens", 0)
        uncached = sum(r.kv.kv_allocated_len - r.cache_protected_len
                       for r in self.reqs.values())
        return (self.allocator.available_size()
                + self.tree.protected_size() + radix_claim + uncached)

    def _prefill(self, row, start, n):
        # allocate pages like the runtime: consume from free_pages first
        take = (n + PAGE - 1) // PAGE
        pages = self.allocator.free_pages[:take].to(torch.int64)
        self.allocator.free_pages = self.allocator.free_pages[take:]
        locs = (pages[:, None] * PAGE + torch.arange(PAGE)[None, :]).flatten()[:n]
        self.table[row, start:start + n] = locs.to(torch.int32)
        self.reqs[row].kv.kv_allocated_len = start + n  # runtime's record
        self.ht.arm(self.backend, row)
        self.ht.after_write(FakeBatch([row], locs))

    def test_confiscate_balances_identity(self):
        self.ht.arm(self.backend, 0)
        self.assertTrue(self.ht.G["armed"])
        self.assertEqual(int(self.allocator.free_pages.numel()), 40)  # pages 1..40
        self.assertEqual(self.allocator.size, 40 * PAGE)  # shrunk total
        self.assertEqual(self._identity_lhs(), self.allocator.size)
        self.assertEqual(self.ht.G["orig_size"], 200 * PAGE)

    def test_retire_returns_pages_and_decrements_uncached(self):
        self._prefill(0, 0, 2048)   # 32 pages; hot margin 4 -> 28 retire
        st = self.ht.R[0]
        self.assertEqual(len(st["retired"]), 28)
        self.assertEqual(self._identity_lhs(), self.allocator.size)
        # freed pages back in circulation
        self.assertEqual(int(self.allocator.free_pages.numel()), 40 - 32 + 28)
        # runtime record decremented to the live tokens
        self.assertEqual(self.reqs[0].kv.kv_allocated_len, 4 * PAGE)
        # cleared columns read ZERO (retire-to-0): the radix finish-insert
        # reads table content; sink-filled columns poisoned its evictable
        # count (the 33,280-token over-count)
        self.assertTrue((self.table[0, :28 * PAGE] == 0).all())

    def test_stage_takes_from_reserve_and_rebalances(self):
        self._prefill(0, 0, 2048)
        reserve_before = int(self.ht.G["reserve"].numel())
        self.ht.before_gather(torch.tensor([5]), FakeBatch([0], torch.tensor([0])))
        self.assertEqual(int(self.ht.G["reserve"].numel()), reserve_before - 1)
        self.assertEqual(self._identity_lhs(), self.allocator.size)
        self.assertEqual(self.reqs[0].kv.kv_allocated_len, 4 * PAGE)  # stage does not touch it

    def test_finish_free_never_returns_sink(self):
        self._prefill(0, 0, 2048)
        sink = self.ht.G["sink_page"] * PAGE
        sink_slots = torch.full((PAGE,), sink, dtype=torch.int64)
        live = torch.arange(self.next_slot - 4 * PAGE, self.next_slot, dtype=torch.int64)
        before = int(self.allocator.free_pages.numel())
        self.allocator.free(torch.cat([live, sink_slots]))  # finish-path shape
        after = int(self.allocator.free_pages.numel())
        self.assertEqual(after - before, 4)  # live pages only, sink filtered
        have = set(self.allocator.free_pages.tolist())
        self.assertNotIn(self.ht.G["sink_page"], have)

    def test_finish_free_drops_zero_and_tier_pages(self):
        """Zero-filled retired columns freed at finish must not inject
        page 0 (or any tier page) into circulation."""
        self._prefill(0, 0, 2048)
        zero_slots = torch.zeros(PAGE, dtype=torch.int64)
        live = torch.arange(self.next_slot - 4 * PAGE, self.next_slot, dtype=torch.int64)
        before = int(self.allocator.free_pages.numel())
        self.allocator.free(torch.cat([live, zero_slots]))  # finish shape
        after = int(self.allocator.free_pages.numel())
        self.assertEqual(after - before, 4)      # live pages only
        have = set(self.allocator.free_pages.tolist())
        self.assertNotIn(0, have)                # padding page filtered
        self.assertNotIn(self.ht.G["sink_page"], have)

    def test_re_retire_returns_frame_to_reserve_not_allocator(self):
        """The oracle's catch: a staged page re-retired through the
        unfiltered free inflated available beyond total. The frame must
        return to the tier's reserve instead."""
        self._prefill(0, 0, 2048)
        free_before = int(self.allocator.free_pages.numel())
        reserve_before = int(self.ht.G["reserve"].numel())
        # stage page 5 (restores it to a reserve frame)
        self.ht.before_gather(torch.tensor([5 * PAGE]), FakeBatch([0], torch.tensor([0])))
        staged_frame = int(self.ht.R[0]["page_slots"][5][0].item()) // PAGE
        self.assertGreaterEqual(staged_frame, 40)  # a tier frame
        # re-retire: write more context so page 5 falls outside hot margin
        self._prefill(0, 2048, 2048)  # ctx 4096; hot 4 -> pages 0..59 retire
        self.assertIn(5, self.ht.R[0]["retired"])
        free_after = int(self.allocator.free_pages.numel())
        reserve_after = int(self.ht.G["reserve"].numel())
        # the frame came back to the reserve, not to the allocator
        self.assertEqual(reserve_after, reserve_before)  # -1 stage +1 return
        # allocator only gained the ORIGINAL pages retired this pass
        self.assertLessEqual(free_after - free_before, 60)

    def test_full_pool_mode_unchanged_when_shrunk_off(self):
        # fresh world: shrunk flag off, allocator untouched by setUp's arm
        os.environ["SGLANG_HOSTTIER_SHRUNK"] = ""
        import hosttier23m
        importlib.reload(hosttier23m)
        self.assertFalse(hosttier23m.SHRUNK)
        pool = FakePool(slots=64 * 200)
        allocator = FakeAllocator(pages=200)   # not the confiscating one
        tree = FakeTreeCache()
        table = torch.zeros(4, 8192, dtype=torch.int32)
        reqs = [FakeReq(0)]
        backend = FakeBackend(pool, allocator, FakeScheduler(reqs, tree), table)
        hosttier23m.arm(backend, 0)
        self.assertIsNone(hosttier23m.G["reserve"])
        self.assertEqual(int(allocator.free_pages.numel()), 199)  # untouched


if __name__ == "__main__":
    unittest.main()


class ModeFreezeAndPrefetch(unittest.TestCase):
    def setUp(self):
        os.environ["SGLANG_HOSTTIER_PREFETCH"] = "1"
        os.environ["SGLANG_HOSTTIER_PREFETCH_RADIUS"] = "1"
        self.ht, self.log = fresh_shrunk(keep_pages=40)
        self.pool = FakePool(slots=64 * 200)
        self.allocator = FakeAllocator(pages=200)
        self.tree = FakeTreeCache()
        self.table = torch.zeros(4, 8192, dtype=torch.int32)
        self.reqs = {0: FakeReq(0)}
        self.scheduler = FakeScheduler(list(self.reqs.values()), self.tree)
        self.backend = FakeBackend(self.pool, self.allocator, self.scheduler, self.table)
        self.ht.arm(self.backend, 0)
        self._prefill(0, 0, 2048)  # 32 pages; hot 4 -> ~28 retired

    def _prefill(self, row, start, n):
        take = (n + PAGE - 1) // PAGE
        pages = self.allocator.free_pages[:take].to(torch.int64)
        self.allocator.free_pages = self.allocator.free_pages[take:]
        locs = (pages[:, None] * PAGE + torch.arange(PAGE)[None, :]).flatten()[:n]
        self.table[row, start:start + n] = locs.to(torch.int32)
        self.reqs[row].kv.kv_allocated_len = start + n
        self.ht.arm(self.backend, row)
        self.ht.after_write(FakeBatch([row], locs))

    def test_freeze_by_decode_mode_stops_retirement(self):
        class FakeMode:
            def is_decode(self):
                return True

            def is_target_verify(self):
                return False

        batch = FakeBatch([0], torch.tensor([self.next_free_slot()]))
        batch.forward_mode = FakeMode()
        retired_before = len(self.ht.R[0]["retired"])
        # First decode-ish batch arms the freeze; a second at the same
        # ctx confirms prefill is done (prefill-time MTP verify batches
        # interleave with ctx-growing chunks and must not freeze).
        # Isolated decode-ish batches (MTP draft steps during prefill)
        # must not freeze; a sustained streak must.
        for _ in range(2):
            self.ht.after_write(batch)
            self.assertFalse(self.ht.R[0].get("decode_frozen"))
        for _ in range(self.ht.FREEZE_STREAK):
            self.ht.after_write(batch)
        self.assertTrue(self.ht.R[0].get("decode_frozen"))
        # Further prefill-scale writes must not retire anything more.
        self._prefill(0, 2048, 2048)
        self.assertEqual(len(self.ht.R[0]["retired"]), retired_before)

    def test_prefetch_neighbor_then_fast_path(self):
        # Prefill more so page 5 is comfortably below the hot horizon.
        self._prefill(0, 2048, 2048)
        reserve_before = int(self.ht.G["reserve"].numel())
        # The mirror holds page 6's bytes at its pre-staging mapping.
        orig_of_6 = self.ht.R[0]["page_slots"][6].clone()
        # Stage page 5; with radius 1 the tier also prefetches 4 and 6.
        self.ht.before_gather(torch.tensor([5 * PAGE]),
                              FakeBatch([0], torch.tensor([0])))
        self.assertLess(int(self.ht.G["reserve"].numel()), reserve_before)
        prefetched = set(self.ht.R[0].get("prefetched", set()))
        self.assertIn(4, prefetched)
        self.assertIn(6, prefetched)
        frame_of_6 = self.ht.R[0]["page_slots"][6].clone()
        reserve_mid = int(self.ht.G["reserve"].numel())
        # Second gather selects page 6: fast path - table rewrite only,
        # no further reserve consumption, page content matches mirror.
        self.ht.before_gather(torch.tensor([6 * PAGE]),
                              FakeBatch([0], torch.tensor([0])))
        # The fast-path page itself consumes no frame; this gather's own
        # neighbor prefetch may take at most radius frames.
        self.assertGreaterEqual(int(self.ht.G["reserve"].numel()),
                                reserve_mid - 2)
        self.assertEqual(self.ht.G.get("prefetch_hits", 0), 1)
        self.assertNotIn(6, self.ht.R[0]["retired"])
        self.assertTrue(torch.equal(self.ht.R[0]["page_slots"][6], frame_of_6))
        cols = torch.arange(6 * PAGE, 7 * PAGE)
        self.assertTrue(torch.equal(self.table[0, cols].long(),
                                    frame_of_6.long()))
        for layer in self.ht.G["layers"]:
            mk, mv = self.ht.G["host_mirror"][layer]
            gk = self.pool.get_key_buffer(layer)[frame_of_6]
            self.assertTrue(torch.equal(gk, mk[orig_of_6.cpu()]))

    def next_free_slot(self):
        page = int(self.allocator.free_pages[0].item())
        return page * PAGE


class MultiRowGather(unittest.TestCase):
    def setUp(self):
        self.ht, self.log = fresh_shrunk(keep_pages=40)
        self.pool = FakePool(slots=64 * 200)
        self.allocator = FakeAllocator(pages=200)
        self.tree = FakeTreeCache()
        self.table = torch.zeros(4, 8192, dtype=torch.int32)
        self.reqs = {0: FakeReq(0), 1: FakeReq(1)}
        self.scheduler = FakeScheduler(list(self.reqs.values()), self.tree)
        self.backend = FakeBackend(self.pool, self.allocator, self.scheduler, self.table)
        self.ht.arm(self.backend, 0)
        for row in (0, 1):
            self._prefill(row, 0, 2048)

    def _prefill(self, row, start, n):
        take = (n + PAGE - 1) // PAGE
        pages = self.allocator.free_pages[:take].to(torch.int64)
        self.allocator.free_pages = self.allocator.free_pages[take:]
        locs = (pages[:, None] * PAGE + torch.arange(PAGE)[None, :]).flatten()[:n]
        self.table[row, start:start + n] = locs.to(torch.int32)
        self.reqs[row].kv.kv_allocated_len = start + n
        self.ht.arm(self.backend, row)
        self.ht.after_write(FakeBatch([row], locs))

    def test_rows_restore_their_own_pages(self):
        # Both rows retire pages 0..27; each row's selections must stage
        # that row's own page image, never the other row's.
        for row in (0, 1):
            self.assertIn(10, self.ht.R[row]["retired"])
            self.assertIn(20, self.ht.R[row]["retired"])
        pos = torch.tensor([10 * PAGE + j for j in range(6)]
                           + [20 * PAGE + j for j in range(6)])
        batch = FakeBatch([0, 1], torch.zeros(12, dtype=torch.int64))
        batch.extend_seq_lens = torch.tensor([6, 6], dtype=torch.int32)
        # The mirror holds each page at its PRE-restore mapping; a staged
        # restore must land those bytes in the reserve frame.
        orig = {r: {p: self.ht.R[r]["page_slots"][p].clone()
                    for p in (10, 20)}
                for r in (0, 1)}
        self.ht.before_gather(pos, batch)
        for row, page in ((0, 10), (1, 20)):
            self.assertNotIn(page, self.ht.R[row]["retired"])
            slots = self.ht.R[row]["page_slots"][page]
            cols = torch.arange(page * PAGE, (page + 1) * PAGE)
            self.assertTrue(torch.equal(self.table[row, cols].long(),
                                        slots.long()))
            for layer in self.ht.G["layers"]:
                mk, mv = self.ht.G["host_mirror"][layer]
                gk = self.pool.get_key_buffer(layer)[slots]
                self.assertTrue(torch.equal(gk, mk[orig[row][page].cpu()]))
        # The cross pages must remain retired and zeroed for both rows.
        for row, page in ((0, 20), (1, 10)):
            self.assertIn(page, self.ht.R[row]["retired"])
            cols = torch.arange(page * PAGE, (page + 1) * PAGE)
            self.assertTrue(bool((self.table[row, cols] == 0).all()))


class StalenessGuard(unittest.TestCase):
    def setUp(self):
        self.ht, self.log = fresh_shrunk(keep_pages=40)
        self.pool = FakePool(slots=64 * 200)
        self.allocator = FakeAllocator(pages=200)
        self.tree = FakeTreeCache()
        self.table = torch.zeros(4, 8192, dtype=torch.int32)
        self.reqs = {0: FakeReq(0)}
        self.scheduler = FakeScheduler(list(self.reqs.values()), self.tree)
        self.backend = FakeBackend(self.pool, self.allocator, self.scheduler, self.table)
        self.ht.arm(self.backend, 0)
        self._prefill(0, 0, 2048)

    def _prefill(self, row, start, n):
        take = (n + PAGE - 1) // PAGE
        pages = self.allocator.free_pages[:take].to(torch.int64)
        self.allocator.free_pages = self.allocator.free_pages[take:]
        locs = (pages[:, None] * PAGE + torch.arange(PAGE)[None, :]).flatten()[:n]
        self.table[row, start:start + n] = locs.to(torch.int32)
        self.reqs[row].kv.kv_allocated_len = start + n
        self.ht.arm(self.backend, row)
        self.ht.after_write(FakeBatch([row], locs))

    def test_decode_scale_writes_do_not_reset_row_state(self):
        # A decode/verify batch: tiny out_locs AND extend_seq_lens carrying
        # per-row deltas (4), which the stale check must NOT read as the
        # request's total. Regression for the ctx=3987/retired=0 wipe that
        # froze retirement during decode.
        retired_before = len(self.ht.R[0]["retired"])
        self.assertGreater(retired_before, 0)
        locs = torch.tensor([int(self.allocator.free_pages[0].item()) * PAGE])
        batch = FakeBatch([0], locs)
        batch.extend_seq_lens = torch.tensor([4], dtype=torch.int32)
        self.ht.after_write(batch)
        self.assertEqual(self.ht.R[0]["ctx_written"], 2048 + 1)
        self.assertEqual(len(self.ht.R[0]["retired"]), retired_before)
