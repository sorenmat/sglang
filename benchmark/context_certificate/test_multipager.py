"""Tests for the P2.3 multi-request page mapper (CPU, pure logic)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "host_kv"))

from multipager import Eviction, PageMapper, Residence


class MappingBasics(unittest.TestCase):
    def test_alloc_and_residence(self):
        m = PageMapper(gpu_frames=4)
        frame, ev = m.alloc("r1", 0)
        self.assertEqual(frame, 3)                      # highest frame first
        self.assertEqual(ev, [])
        self.assertIs(m.residence("r1", 0), Residence.GPU)
        self.assertIs(m.residence("r1", 1), Residence.HOST)

    def test_realloc_same_page_is_free(self):
        m = PageMapper(gpu_frames=2)
        f1, _ = m.alloc("r1", 5, dirty=True)
        f2, ev = m.alloc("r1", 5, dirty=True)
        self.assertEqual(f1, f2)
        self.assertEqual(ev, [])
        self.assertEqual(m.stats["allocations"], 1)

    def test_close_request_frees_frames(self):
        m = PageMapper(gpu_frames=2)
        m.alloc("r1", 0)
        m.close_request("r1")
        self.assertEqual(m.used_frames(), 0)
        self.assertEqual(sorted(m.free_frames), [0, 1])


class EvictionBehavior(unittest.TestCase):
    def test_lru_eviction_order(self):
        m = PageMapper(gpu_frames=2)
        m.alloc("r1", 0)
        m.alloc("r1", 1)
        m.alloc("r1", 0)                                # touch 0: 1 is LRU
        _, evictions = m.alloc("r1", 2)
        self.assertEqual(len(evictions), 1)
        self.assertEqual(evictions[0].logical_page, 1)
        self.assertTrue(evictions[0].dirty)

    def test_pinned_pages_survive(self):
        m = PageMapper(gpu_frames=2)
        m.alloc("r1", 0)
        m.alloc("r1", 1)
        m.pin("r1", [0, 1])
        with self.assertRaises(RuntimeError):
            m.alloc("r1", 2)                            # nothing evictable
        m.unpin_all()
        _, evictions = m.alloc("r1", 2)                 # now evictable
        self.assertEqual(len(evictions), 1)

    def test_frames_never_aliased(self):
        m = PageMapper(gpu_frames=3)
        for req in ("a", "b", "c"):
            m.alloc(req, 0)
        owners = set(m.frame_owner.values())
        self.assertEqual(len(owners), 3)
        _, evictions = m.alloc("d", 0)                  # evicts one LRU page
        self.assertEqual(len(m.frame_owner), 3)         # still 3 distinct
        self.assertEqual(len(set(m.frame_owner.values())), 3)
        for ev in evictions:
            self.assertNotIn(ev.frame, m.free_frames)   # handed to the new page

    def test_eviction_ordering_contract(self):
        # Dirty evictions free the frame immediately; the caller must run
        # the eviction copy before the allocating write. The plan carries
        # everything needed.
        m = PageMapper(gpu_frames=1)
        f0, _ = m.alloc("r1", 0, dirty=True)
        f1, evictions = m.alloc("r2", 0, dirty=True)
        self.assertEqual(f0, f1)                        # same frame reused
        self.assertEqual(evictions, [Eviction("r1", 0, f0, True)])
        self.assertIs(m.residence("r1", 0), Residence.HOST)


class DemandStaging(unittest.TestCase):
    def test_ensure_stages_missing_only(self):
        m = PageMapper(gpu_frames=4)
        m.alloc("r1", 0)
        m.alloc("r1", 1)
        m.close_request("r1")
        m.open_request("r1")                            # fresh: all host
        m.alloc("r1", 7)
        plan = m.ensure("r1", [7, 8, 9])
        staged_pages = [page for page, _, _ in plan]
        self.assertEqual(staged_pages, [8, 9])          # 7 already resident
        self.assertEqual(m.stats["stages"], 2)

    def test_ensure_respects_budget_with_evictions(self):
        m = PageMapper(gpu_frames=2)
        for page in (0, 1):
            m.alloc("r1", page, dirty=True)
        plan = m.ensure("r1", [5, 6])
        self.assertEqual(len(plan), 2)
        all_ev = [e for _, _, evs in plan for e in evs]
        self.assertEqual(len(all_ev), 2)                # both old pages evicted
        self.assertEqual(m.used_frames(), 2)

    def test_selection_pin_then_pass(self):
        m = PageMapper(gpu_frames=4)
        for page in range(4):
            m.alloc("r1", page, dirty=False)
        plan = m.ensure("r1", [0, 1])                   # this pass's selection
        self.assertEqual(plan, [])
        m.pin("r1", [0, 1])
        m.unpin_all()
        self.assertEqual(len(m.pinned), 0)


class StreamingPrefill(unittest.TestCase):
    def test_prefill_beyond_capacity(self):
        """A 262k-token prefill against 1,300 frames streams: every page
        written once, mirrored, evicted; the plan describes the copies."""
        pages_total, frames = 40, 3
        m = PageMapper(gpu_frames=frames)
        dirty_evictions = 0
        for page in range(pages_total):
            _, evictions = m.alloc("r1", page, dirty=True)
            for _ in evictions:
                dirty_evictions += 1                    # caller mirrors here
            m.mark_clean("r1", page)                    # write-through done
        self.assertEqual(m.stats["allocations"], pages_total)
        self.assertEqual(dirty_evictions, pages_total - frames)
        self.assertEqual(m.used_frames(), frames)
        # Early pages fell back to host; the tail sits in frames.
        self.assertIs(m.residence("r1", 0), Residence.HOST)
        self.assertIs(m.residence("r1", pages_total - 1), Residence.GPU)

    def test_two_requests_interleave(self):
        m = PageMapper(gpu_frames=2)
        m.alloc("a", 0)
        m.alloc("b", 0)
        # touching a's page makes b's page the global LRU victim
        m.alloc("a", 0)
        _, evictions = m.alloc("a", 1)
        self.assertEqual(evictions[0].request, "b")


if __name__ == "__main__":
    unittest.main()
