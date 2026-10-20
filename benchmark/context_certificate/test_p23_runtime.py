"""Tests for the P2.3 runtime glue (torch CPU tensors, no GPU)."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "host_kv"))

from multipager import PageMapper
from p23_runtime import (EvictionRewriter, PassDriver, PrefillDriver,
                          RESERVED_SLOT, logical_page_of, new_slots_for,
                          page_token_span)

REQS, CTX = 4, 4096  # 64 pages per request


class Arithmetic(unittest.TestCase):
    def test_page_span_and_slots(self):
        start, end = page_token_span(7)
        self.assertEqual((start, end), (448, 512))
        slots = new_slots_for(7, 13)
        self.assertEqual(slots.tolist(), list(range(13 * 64, 14 * 64)))
        self.assertEqual(logical_page_of(511), 7)
        self.assertEqual(logical_page_of(512), 8)


class PassFlow(unittest.TestCase):
    def setUp(self):
        self.mapper = PageMapper(gpu_frames=4)
        self.table = torch.zeros(REQS, CTX, dtype=torch.int64)
        # prefill pages 0..2 of request row 1 into frames
        for page in (0, 1, 2):
            frame, _ = self.mapper.alloc("r1", page)
            self.table[1, page * 64:(page + 1) * 64] = new_slots_for(page, frame)
        self.driver = PassDriver(self.mapper, self.table)

    def test_selection_mixes_resident_and_missing(self):
        plan = self.driver.begin({1: ("r1", [10, 300, 130])})  # pages 0,2,4
        staged = [page for page, _ in plan["stage_copies"]]
        self.assertEqual(staged, [4])                      # 0 and 2 resident
        self.assertEqual(len(plan["evictions"]), 0)         # one free frame left
        self.assertEqual(self.mapper.used_frames(), 4)
        self.driver.apply_rewrites(plan)
        page, frame = plan["stage_copies"][0]
        row_slots = self.table[1, 4 * 64:5 * 64]
        self.assertEqual(row_slots.tolist(),
                         new_slots_for(page, frame).tolist())
        self.driver.end_pass()
        self.assertEqual(len(self.mapper.pinned), 0)

    def test_evicted_page_table_cleared(self):
        # page 0 is LRU after touching others; allocate until it evicts
        rewriter = EvictionRewriter(self.table, row_of=lambda request: 1)
        self.mapper.alloc("r1", 0)
        self.mapper.alloc("r1", 1)
        self.mapper.alloc("r1", 2)
        _, evictions = self.mapper.alloc("r1", 4, dirty=False)  # fills 4th + pressure? no: takes last free
        for ev in evictions:
            rewriter.on_eviction(ev)
        # one more page forces the eviction of the globally-LRU page (0)
        _, evictions = self.mapper.alloc("r1", 5, dirty=False)
        for ev in evictions:
            rewriter.on_eviction(ev)
        # request row 1, page 0's tokens now point at the reserved slot
        self.assertTrue((self.table[1, 0:64] == RESERVED_SLOT).all())

    def test_pin_protects_selection_across_alloc(self):
        plan = self.driver.begin({1: ("r1", [10])})         # pin page 0
        self.driver.apply_rewrites(plan)
        # allocation pressure cannot evict the pinned page
        _, evictions = self.mapper.alloc("r1", 5, dirty=False)
        victims = {(e.request, e.logical_page) for e in evictions}
        self.assertNotIn(("r1", 0), victims)
        self.driver.end_pass()


class PrefillStreaming(unittest.TestCase):
    def test_chunk_bookkeeping(self):
        mapper = PageMapper(gpu_frames=2)
        driver = PrefillDriver(mapper)
        pages = driver.chunk_written("r1", 0, 130)           # chunk spans 0,1,2
        self.assertEqual(pages, [0, 1, 2])
        for page in pages:
            mapper.alloc("r1", page, dirty=True)
        done = driver.pending_mirror[:]
        for request, page in done:
            driver.mirror_done(request, [page])
        self.assertEqual(mapper.requests["r1"].dirty, set())


if __name__ == "__main__":
    unittest.main()
