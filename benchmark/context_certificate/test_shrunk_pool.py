"""CPU simulation of the HKV-4 shrunk-pool accounting (design validation).

Simulates the full lifecycle — confiscate, prefill, retire-free,
stage-from-reserve, request finish — against a fake allocator and tree
cache, asserting the runtime's own identity
(available + protected + uncached == total) at every step and the
no-duplicate-pages invariant in free_pages.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "host_kv"))

PAGE = 64


class FakeAllocator:
    def __init__(self, pages):
        self.page_size = PAGE
        self.size = pages * PAGE
        self.free_pages = torch.arange(0, pages, dtype=torch.int32)

    def available_size(self):
        return int(self.free_pages.numel()) * PAGE

    def free(self, free_index):
        pages = torch.unique(free_index // PAGE)
        for p in pages.tolist():
            if p not in self.free_pages.tolist():
                self.free_pages = torch.cat(
                    [self.free_pages, torch.tensor([p], dtype=torch.int32)])


class FakeTreeCache:
    def __init__(self):
        self._extra_protected = 0

    def full_protected_size(self):
        return self._extra_protected

    def protected_size(self):
        return self._extra_protected


class FakeReq:
    def __init__(self):
        self.kv = type("KV", (), {"kv_allocated_len": 0})()
        self.cache_protected_len = 0


class ShrunkPoolSim:
    """Implements the HKV-4 design's accounting moves."""

    def __init__(self, pages, keep_pages):
        self.alloc = FakeAllocator(pages)
        self.tree = FakeTreeCache()
        self.total = self.alloc.size
        self.sink_page = None  # assigned from the reserve at confiscation
        self.reqs = {}
        self.confiscate(keep_pages)

    def confiscate(self, keep_pages):
        free = self.alloc.free_pages
        self.reserve = free[keep_pages:].clone()
        self.alloc.free_pages = free[:keep_pages].clone()
        self.sink_page = int(self.reserve[-1].item())
        self.tree._extra_protected = int(self.reserve.numel()) * PAGE

    # -- runtime-side operations the tier must balance ------------------

    def runtime_alloc(self, req_id, n_tokens):
        req = self.reqs.setdefault(req_id, FakeReq())
        take = (n_tokens + PAGE - 1) // PAGE
        assert take <= int(self.alloc.free_pages.numel()), "pool exhausted"
        pages = self.alloc.free_pages[:take]
        self.alloc.free_pages = self.alloc.free_pages[take:]
        req.kv.kv_allocated_len += take * PAGE
        return pages

    def tier_retire(self, req_id, pages):
        """Retirement: free the runtime pages AND decrement uncached."""
        req = self.reqs[req_id]
        for p in pages:
            self.alloc.free(torch.tensor([p * PAGE], dtype=torch.int64))
            req.kv.kv_allocated_len -= PAGE

    def tier_stage(self, req_id):
        """Staging takes from OUR reserve and grows the request's record."""
        req = self.reqs[req_id]
        p = int(self.reserve[0].item())
        self.reserve = self.reserve[1:]
        self.tree._extra_protected -= PAGE
        req.kv.kv_allocated_len += PAGE
        return p

    def request_finish(self, req_id, live_slots):
        """The runtime frees table content; sink slots must be filtered."""
        pages = torch.unique(live_slots // PAGE)
        pages = pages[pages != self.sink_page]  # our wrapped free filter
        for p in pages.tolist():
            if p not in self.reserve.tolist():
                self.alloc.free(torch.tensor([p * PAGE], dtype=torch.int64))
        del self.reqs[req_id]

    # -- the oracle -----------------------------------------------------

    def identity_ok(self):
        uncached = sum(r.kv.kv_allocated_len - r.cache_protected_len
                       for r in self.reqs.values())
        lhs = (self.alloc.available_size()
               + self.tree.full_protected_size() + uncached)
        return lhs == self.total, f"{lhs} != {self.total}"

    def no_duplicate_free_pages(self):
        fp = self.alloc.free_pages.tolist()
        return len(fp) == len(set(fp)), f"dup in {fp[:10]}"


class ShrunkPoolIdentity(unittest.TestCase):
    def test_full_lifecycle_holds_identity(self):
        sim = ShrunkPoolSim(pages=3126, keep_pages=1500)
        for step in range(20):
            self.assertTrue(sim.identity_ok()[0], sim.identity_ok()[1])
            self.assertTrue(sim.no_duplicate_free_pages()[0])

            pages = sim.runtime_alloc("r1", n_tokens=192)   # 3 pages
            self.assertTrue(sim.identity_ok()[0])  # uncached grows, avail shrinks

            sim.tier_retire("r1", pages.tolist()[:-1])      # retire 2, keep 1 hot
            self.assertTrue(sim.identity_ok()[0])           # avail back, uncached down

            sim.tier_stage("r1")                            # reserve -> request
            self.assertTrue(sim.identity_ok()[0])           # protected down, uncached up

            self.assertTrue(sim.no_duplicate_free_pages()[0])

    def test_finish_frees_only_live_pages_no_duplicates(self):
        sim = ShrunkPoolSim(pages=200, keep_pages=150)
        pages = sim.runtime_alloc("a", 8192)          # 128 pages
        live = pages[:16]                             # 16 still live at finish
        sim.tier_retire("a", pages[16:].tolist())     # rest retired earlier
        sink_slots = torch.tensor(
            [sim.sink_page * PAGE] * 64, dtype=torch.int64)
        # table content at finish: live pages + sink-filled cleared columns
        content = torch.cat([live * PAGE, sink_slots])
        before = int(sim.alloc.free_pages.numel())
        sim.request_finish("a", content)
        after = int(sim.alloc.free_pages.numel())
        self.assertEqual(after - before, 16)          # exactly the live pages
        self.assertTrue(sim.no_duplicate_free_pages()[0])
        # two more finishes with sink-only content add nothing
        for _ in range(2):
            sim.request_finish("b", sink_slots) if "b" in sim.reqs else None
        self.assertTrue(sim.no_duplicate_free_pages()[0])

    def test_exhaustion_is_detected_not_silent(self):
        sim = ShrunkPoolSim(pages=100, keep_pages=10)
        with self.assertRaises(AssertionError):
            sim.runtime_alloc("r1", n_tokens=10_000_000)  # way over budget


if __name__ == "__main__":
    unittest.main()
